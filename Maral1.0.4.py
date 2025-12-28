import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Page configuration
st.set_page_config(
    page_title='Maral 1.0.4',
    page_icon='📊',
    layout='wide'
)

st.title('📊 MARAL', text_alignment='center')
st.markdown('<p style="text-align: center;">Binomial Option Pricing Calculator</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Creator: Mehr</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Version 1.0.4</p>', unsafe_allow_html=True)
st.markdown(
    '''
    <p style="text-align: center;">
    This application implements the <b>Binomial Option Pricing Model</b> for
    various asset classes with support for dividends and different risk-neutral measures.
    </p>
    ''', unsafe_allow_html=True
)

# Sidebar for parameters
st.sidebar.header('Asset Class & Parameters')

# Asset class selection
asset_class = st.sidebar.selectbox(
    'Asset Class',
    ['Stock Option', 'Stock Index Option', 'Futures Option', 'Currency Option'],
    help='Select the type of underlying asset'
)

# Method selection
calculation_method = st.sidebar.selectbox(
    'Calculation Method',
    ['Use Volatility (σ)', 'Use Manual u/d'],
    help='Choose whether to calculate u/d from volatility or input them manually'
)

# Parameters
col1, col2 = st.sidebar.columns(2)

with col1:
    option_type = st.selectbox(
        'Option Type',
        ['American', 'European'],
        help='American option can be executed anytime till maturity.',
        key='option_type'
    )

    option_kind = st.selectbox(
        'Option Kind',
        ['Call', 'Put'],
        help='Call option gives the right to buy. Put option gives the right to sell.',
        key='option_kind'
    )

    current_price = st.number_input(
        f'Current {"Price" if asset_class != "Currency Option" else "Exchange Rate"} (S₀)',
        min_value=0.01,
        value=100.0,
        step=1.0,
        help=f'Current {"price" if asset_class != "Currency Option" else "exchange rate"} of the underlying asset'
    )

    strike_price = st.number_input(
        'Strike Price (K)',
        min_value=0.01,
        value=100.0,
        step=1.0,
        help='Price at which the option can be executed'
    )

    # Dividend input for stocks and stock indices
    if asset_class in ['Stock Option', 'Stock Index Option']:
        dividend_yield = st.number_input(
            'Dividend Yield (%)',
            min_value=0.0,
            value=0.0,
            step=0.1,
            help='Annual dividend yield as percentage of asset price',
            key='dividend_yield'
        ) / 100
    else:
        dividend_yield = 0.0

with col2:
    # Time inputs are always shown
    time_months = st.number_input(
        'Time to Maturity (Months)',
        min_value=1,
        value=9,
        step=1,
        help='Time until option expiration',
        key='time_months'
    )
    time_to_maturity = time_months / 12
    
    steps = st.number_input(
        'Steps in Binomial Tree (n)',
        min_value=1,
        value=3,
        step=1,
        help='Number of time steps in the binomial tree',
        key='steps'
    )
    
    if calculation_method == 'Use Volatility (σ)':
        # Only show volatility input
        volatility = st.number_input(
            'Volatility (σ) - Annual %',
            min_value=0.1,
            value=28.0,
            step=0.5,
            help='Annualized volatility of the underlying asset',
            key='volatility_input'
        ) / 100
        
        # Initialize u and d as None
        u = None
        d = None
        
        # Calculate and display per-step u and d
        dt = time_to_maturity / steps
        u_calculated = np.exp(volatility * np.sqrt(dt))
        d_calculated = 1 / u_calculated
        
        st.info(f"""
        **Calculated per-step parameters:**
        - Time step (Δt): {dt:.4f} years ({dt*12:.1f} months)
        - Up factor (u): {u_calculated:.4f} ({(u_calculated-1)*100:.2f}%)
        - Down factor (d): {d_calculated:.4f} ({(1-d_calculated)*100:.2f}%)
        """)
        
    else:  # Manual u/d mode
        # Initialize volatility as None
        volatility = None
        
        # Show u and d inputs ONLY here
        st.warning("⚠️ Input u and d as PER-STEP percentages")
        
        u = st.number_input(
            'Up Factor (u) % per step',
            min_value=0.1,
            value=15.03,
            step=0.5,
            help='Percentage increase in price per time step',
            key='u_manual_input'
        ) / 100
        
        d = st.number_input(
            'Down Factor (d) % per step',
            min_value=0.1,
            value=13.06,
            step=0.5,
            help='Percentage decrease in price per time step',
            key='d_manual_input'
        ) / 100

# Additional parameters based on asset class
st.sidebar.subheader(f'{asset_class} Parameters')

if asset_class == 'Stock Option' or asset_class == 'Stock Index Option':
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Interest Rate (r) %",
        min_value=0.0,
        value=3.0,
        step=0.5,
        help="Annual risk-free interest rate",
        key='risk_free_rate'
    ) / 100
    
elif asset_class == 'Futures Option':
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Interest Rate (r) %",
        min_value=0.0,
        value=3.0,
        step=0.5,
        help="Annual risk-free interest rate",
        key='risk_free_rate_futures'
    ) / 100
    
elif asset_class == 'Currency Option':
    domestic_rate = st.sidebar.number_input(
        "Domestic Risk-Free Rate (rₕ) %",
        min_value=0.0,
        value=3.0,
        step=0.5,
        help="Annual domestic risk-free interest rate",
        key='domestic_rate'
    ) / 100
    
    foreign_rate = st.sidebar.number_input(
        "Foreign Risk-Free Rate (r_f) %",
        min_value=0.0,
        value=1.0,
        step=0.5,
        help="Annual foreign risk-free interest rate",
        key='foreign_rate'
    ) / 100
    
    # For currency options, the risk-neutral measure uses interest rate differential
    risk_free_rate = domestic_rate  # This will be used for discounting

# Core function - Binomial tree calculation
def calculate_binomial_tree(S, K, T, r, sigma, n, u, d, option_type, option_kind, 
                          asset_class, dividend_yield=0.0, foreign_rate=0.0, use_volatility=False):
    '''
    Calculate binomial tree for option pricing with different asset classes
    
    Parameters:
    S: Current Price/Exchange Rate
    K: Strike Price
    T: Time to Maturity (years)
    r: Risk-free interest Rate (domestic for currency)
    sigma: Volatility (annual)
    n: Number of Steps
    u: Up Factor (percentage if manual, None if using volatility)
    d: Down Factor (percentage if manual, None if using volatility)
    option_type: 'American' or 'European'
    option_kind: 'Call' or 'Put'
    asset_class: Type of underlying asset
    dividend_yield: Annual dividend yield (for stocks/indices)
    foreign_rate: Foreign risk-free rate (for currency options)
    use_volatility: Whether to calculate u/d from volatility
    
    Returns:
    stock_tree: Price tree
    option_tree: Option Price Tree
    exercise_tree: Boolean Tree for Early Exercise Points
    '''
    
    # Calculate dt and discount factor
    dt = T / n
    
    # Different discount factors based on asset class
    if asset_class == 'Currency Option':
        # For currency options, discount using domestic rate
        discount = np.exp(-r * dt)
    else:
        discount = np.exp(-r * dt)
    
    # Calculate u and d factors
    if use_volatility and sigma is not None:
        # Calculate u and d from volatility (CRR model)
        u_factor = np.exp(sigma * np.sqrt(dt))
        d_factor = 1 / u_factor
        
        # Convert to percentage form for consistency
        u_percent = u_factor - 1
        d_percent = 1 - d_factor
        
    else:
        # Use manual u and d (percentage form)
        u_percent = u
        d_percent = d
        
        # Convert percentages to multiplicative factors
        u_factor = 1 + u_percent
        d_factor = 1 - d_percent
    
    # Calculate risk-neutral probability based on asset class
    if asset_class == 'Stock Option':
        # For stocks with dividends
        adjusted_rate = r - dividend_yield
        p = (np.exp(adjusted_rate * dt) - d_factor) / (u_factor - d_factor)
        
    elif asset_class == 'Stock Index Option':
        # Similar to stocks, index has dividend yield
        adjusted_rate = r - dividend_yield
        p = (np.exp(adjusted_rate * dt) - d_factor) / (u_factor - d_factor)
        
    elif asset_class == 'Futures Option':
        # For futures, risk-neutral probability simplifies
        p = (1 - d_factor) / (u_factor - d_factor)
        # For futures, no cost of carry in the probability
        
    elif asset_class == 'Currency Option':
        # For currency options, use interest rate differential
        adjusted_rate = r - foreign_rate  # Domestic - Foreign
        p = (np.exp(adjusted_rate * dt) - d_factor) / (u_factor - d_factor)
    
    # Build price tree
    stock_tree = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S * (u_factor ** (i - j)) * (d_factor ** j)
    
    # Initialize option and exercise trees
    option_tree = np.zeros((n + 1, n + 1))
    exercise_tree = np.zeros((n + 1, n + 1), dtype=bool)
    
    # Calculate option values at expiration
    for j in range(n + 1):
        if option_kind.lower() == 'call':
            option_tree[j, n] = max(stock_tree[j, n] - K, 0)
        else:  # put option
            option_tree[j, n] = max(K - stock_tree[j, n], 0)
    
    # Backward induction for option prices
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            # Expected value in risk-neutral world
            expected_value = discount * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            
            if option_type.lower() == 'european':
                option_tree[j, i] = expected_value
            else:  # American Option
                # Immediate exercise value
                if option_kind.lower() == 'call':
                    exercise_value = max(stock_tree[j, i] - K, 0)
                else:  # put option
                    exercise_value = max(K - stock_tree[j, i], 0)
                
                # Choose maximum of expected value and exercise value
                option_tree[j, i] = max(expected_value, exercise_value)
                
                # Mark if early exercise is optimal
                exercise_tree[j, i] = (exercise_value > expected_value)
    
    # Return calculated u and d in both forms for display
    return stock_tree, option_tree, exercise_tree, p, dt, discount, u_factor, d_factor, u_percent, d_percent

# Calculate option price
use_volatility_flag = (calculation_method == 'Use Volatility (σ)')

# Prepare parameters based on asset class
if asset_class == 'Currency Option':
    result = calculate_binomial_tree(
        current_price, strike_price, time_to_maturity, domestic_rate,
        volatility, steps, u, d, option_type, option_kind, 
        asset_class, dividend_yield, foreign_rate, use_volatility_flag
    )
else:
    result = calculate_binomial_tree(
        current_price, strike_price, time_to_maturity, risk_free_rate,
        volatility, steps, u, d, option_type, option_kind,
        asset_class, dividend_yield, 0.0, use_volatility_flag
    )

# Unpack results
stock_tree, option_tree, exercise_tree, p, dt, discount, u_factor, d_factor, u_percent, d_percent = result

# Display Results
st.header('Results')

# Show calculation method summary
col1, col2 = st.columns(2)
with col1:
    st.info(f"**Asset Class:** {asset_class}")
    st.info(f"**Calculation Method:** {calculation_method}")
    
with col2:
    if calculation_method == 'Use Volatility (σ)':
        st.info(f"**Annual Volatility:** {volatility*100:.2f}%")
    else:
        st.info(f"**Manual u/d:** u={u_percent*100:.2f}%, d={d_percent*100:.2f}%")
    
    if asset_class in ['Stock Option', 'Stock Index Option'] and dividend_yield > 0:
        st.info(f"**Dividend Yield:** {dividend_yield*100:.2f}%")

# Show asset-specific parameters
st.info(f"""
**Per-step parameters:**
- Time step (Δt): {dt:.4f} years ({dt*12:.1f} months)
- Up factor: {u_factor:.4f} ({(u_factor-1)*100:.2f}%)
- Down factor: {d_factor:.4f} ({(1-d_factor)*100:.2f}%)
- Risk-neutral probability (p): {p:.4f}
- Discount factor: {discount:.4f}
""")

# Show interest rate info
if asset_class == 'Currency Option':
    st.info(f"""
    **Currency Option Parameters:**
    - Domestic Rate (rₕ): {domestic_rate*100:.2f}%
    - Foreign Rate (r_f): {foreign_rate*100:.2f}%
    - Rate Differential: {(domestic_rate - foreign_rate)*100:.2f}%
    """)
else:
    st.info(f"**Risk-Free Rate:** {risk_free_rate*100:.2f}%")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(['Tree Visualization', 'Option Price', 'Tabular Summary', 'Calculation Details'])

with tab1:
    st.subheader("Binomial Tree Visualization")
    
    # Create figure for tree visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Create positions for nodes
    G = nx.Graph()
    pos = {}
    node_labels_stock = {}
    node_labels_option = {}
    node_colors = []
    
    # Add nodes and edges
    for i in range(steps + 1):
        for j in range(i + 1):
            # Node ID
            node_id = f"{i},{j}"
            G.add_node(node_id)
            
            # Position (x = step, y = j - i/2 to center)
            pos[node_id] = (i, (i - 2 * j) / 2)
            
            # Stock price label
            stock_price = stock_tree[j, i]
            node_labels_stock[node_id] = f"${stock_price:.2f}"
            
            # Option price label
            option_price = option_tree[j, i]
            
            # For American options, highlight early exercise
            if option_type == "American" and i < steps and exercise_tree[j, i]:
                node_labels_option[node_id] = f"${option_price:.2f}*"
                node_colors.append('lightcoral')  # Red for early exercise
            else:
                node_labels_option[node_id] = f"${option_price:.2f}"
                node_colors.append('lightblue')  # Blue for no early exercise
            
            # Add edges to next nodes
            if i < steps:
                G.add_edge(node_id, f"{i+1},{j}")  # Down
                G.add_edge(node_id, f"{i+1},{j+1}")  # Up
    
    # Draw stock price tree
    nx.draw(G, pos, ax=ax1, with_labels=True, labels=node_labels_stock, 
            node_color='lightgreen', node_size=1500, font_size=8, font_weight='bold')
    ax1.set_title(f"Price Tree (S₀ = ${current_price:.2f})", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Time Steps", fontsize=12)
    ax1.set_ylabel("Price Movements", fontsize=12)
    
    # Draw option price tree
    nx.draw(G, pos, ax=ax2, with_labels=True, labels=node_labels_option, 
            node_color=node_colors, node_size=1500, font_size=8, font_weight='bold')
    
    option_title = f"{option_type} {option_kind} Option Price Tree"
    if option_type == "American":
        option_title += "\n* = Optimal Early Exercise Point"
    ax2.set_title(option_title, fontsize=14, fontweight='bold')
    ax2.set_xlabel("Time Steps", fontsize=12)
    ax2.set_ylabel("Price Movements", fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader('Option Price Summary')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label=f'{option_type} {option_kind} Option Price',
            value=f'${option_tree[0, 0]:.4f}'
        )
    
    with col2:
        if option_kind.lower() == 'call':
            intrinsic_value = max(current_price - strike_price, 0)
        else:
            intrinsic_value = max(strike_price - current_price, 0)
        st.metric(
            label='Intrinsic Value',
            value=f'${intrinsic_value:.4f}'
        )

    with col3:
        time_value = option_tree[0, 0] - intrinsic_value
        st.metric(
            label='Time Value',
            value=f'${time_value:.4f}'
        )
    
    # Display additional metrics
    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            label='Risk-Neutral Up Probability (p)',
            value=f'{p:.4f}'
        )

    with col5:
        st.metric(
            label='Risk-Neutral Down Probability (1-p)',
            value=f'{1-p:.4f}'
        )

    with col6:
        st.metric(
            label='Discount Factor',
            value=f'{discount:.4f}'
        )
    
    # Display stock price at expiration
    st.subheader('Possible Prices at Expiration')
    expiration_prices = []
    for j in range(steps + 1):
        expiration_prices.append(stock_tree[j, steps])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Minimum:**")
        st.info(f'${min(expiration_prices):.2f}')
    
    with col2:
        st.write("**Maximum:**")
        st.info(f'${max(expiration_prices):.2f}')
    
    with col3:
        if asset_class == 'Futures Option':
            # For futures, expected price equals current price
            expected_price = current_price
        else:
            # Adjust for dividend yield if applicable
            if asset_class in ['Stock Option', 'Stock Index Option']:
                growth_rate = risk_free_rate - dividend_yield
            elif asset_class == 'Currency Option':
                growth_rate = domestic_rate - foreign_rate
            else:
                growth_rate = risk_free_rate
            expected_price = current_price * np.exp(growth_rate * time_to_maturity)
        st.write('**Expected (Risk-Neutral):**') 
        st.info(f'${expected_price:.2f}')

with tab3:
    st.subheader('Tabular Summary')

    # Create dataframes for the trees
    stock_df = pd.DataFrame()
    option_df = pd.DataFrame()

    for i in range(steps + 1):
        stock_col = []
        option_col = []
        for j in range(steps + 1):
            if j <= i:
                stock_col.append(f"${stock_tree[j, i]:.2f}")

                # Add asterisk for early exercise in American options
                if option_type == 'American' and i < steps and exercise_tree[j, i]:
                    option_col.append(f'${option_tree[j, i]:.2f}*')
                else:
                    option_col.append(f'${option_tree[j, i]:.2f}')
            else:
                stock_col.append("")
                option_col.append("")

        stock_df[f'Step {i}'] = stock_col
        option_df[f'Step {i}'] = option_col
    
    # Display the tables
    col1, col2 = st.columns(2)

    with col1:
        st.write('**Price Tree**')
        st.dataframe(stock_df, use_container_width=True)

    with col2:
        st.write(f'**{option_type} {option_kind} Option Price Tree**')
        if option_type == 'American':
            st.caption('* Indicates optimal early exercise point')
        st.dataframe(option_df, use_container_width=True)
    
    # Early exercise summary for American options
    if option_type == 'American':
        st.subheader('Early Exercise Analysis')
        early_exercise_points = []

        for i in range(steps):
            for j in range(i + 1):
                if exercise_tree[j, i]:
                    if option_kind.lower() == 'call':
                        exercise_val = max(stock_tree[j, i] - strike_price, 0)
                    else:
                        exercise_val = max(strike_price - stock_tree[j, i], 0)
                    
                    early_exercise_points.append({
                        'Step': i,
                        'Node': j,
                        'Price': f'${stock_tree[j, i]:.2f}',
                        'Option Value': f'${option_tree[j, i]:.2f}',
                        'Exercise Value': f'${exercise_val:.2f}'
                    })
        
        if early_exercise_points:
            early_exercise_df = pd.DataFrame(early_exercise_points)
            st.dataframe(early_exercise_df, use_container_width=True)
        else:
            st.info('No optimal early exercise points found for this option')

with tab4:
    st.subheader('Calculation Details')
    
    # Asset class specific formulas
    if asset_class == 'Stock Option':
        formula_explanation = f"""
        ### Stock Option with Dividend Yield
        
        **Risk-Neutral Probability Formula:**
        ```
        p = [e^((r - q)×Δt) - d] / (u - d)
          = [e^(({risk_free_rate:.4f} - {dividend_yield:.4f})×{dt:.4f}) - {d_factor:.4f}] / ({u_factor:.4f} - {d_factor:.4f})
          = [e^({risk_free_rate-dividend_yield:.4f}×{dt:.4f}) - {d_factor:.4f}] / ({u_factor:.4f} - {d_factor:.4f})
          = {p:.4f}
        ```
        
        where:
        - r = Risk-free rate = {risk_free_rate:.4f}
        - q = Dividend yield = {dividend_yield:.4f}
        - Δt = Time step = {dt:.4f}
        """
        
    elif asset_class == 'Stock Index Option':
        formula_explanation = f"""
        ### Stock Index Option with Dividend Yield
        
        **Risk-Neutral Probability Formula:**
        ```
        p = [e^((r - q)×Δt) - d] / (u - d)
          = [e^(({risk_free_rate:.4f} - {dividend_yield:.4f})×{dt:.4f}) - {d_factor:.4f}] / ({u_factor:.4f} - {d_factor:.4f})
          = {p:.4f}
        ```
        
        where:
        - r = Risk-free rate = {risk_free_rate:.4f}
        - q = Dividend yield = {dividend_yield:.4f}
        - Δt = Time step = {dt:.4f}
        """
        
    elif asset_class == 'Futures Option':
        formula_explanation = f"""
        ### Futures Option
        
        **Risk-Neutral Probability Formula:**
        ```
        p = (1 - d) / (u - d)
          = (1 - {d_factor:.4f}) / ({u_factor:.4f} - {d_factor:.4f})
          = {p:.4f}
        ```
        
        **Note:** For futures options, the cost of carry is zero in the risk-neutral measure.
        """
        
    elif asset_class == 'Currency Option':
        formula_explanation = f"""
        ### Currency Option
        
        **Risk-Neutral Probability Formula:**
        ```
        p = [e^((rₕ - r_f)×Δt) - d] / (u - d)
          = [e^(({domestic_rate:.4f} - {foreign_rate:.4f})×{dt:.4f}) - {d_factor:.4f}] / ({u_factor:.4f} - {d_factor:.4f})
          = [e^({domestic_rate-foreign_rate:.4f}×{dt:.4f}) - {d_factor:.4f}] / ({u_factor:.4f} - {d_factor:.4f})
          = {p:.4f}
        ```
        
        where:
        - rₕ = Domestic risk-free rate = {domestic_rate:.4f}
        - r_f = Foreign risk-free rate = {foreign_rate:.4f}
        - Δt = Time step = {dt:.4f}
        """
    
    st.markdown(formula_explanation)
    
    st.markdown(f"""
    ### General Parameters
    
    **Time Parameters:**
    - Total time (T): {time_to_maturity:.4f} years ({time_to_maturity*12:.1f} months)
    - Number of steps (n): {steps}
    - Time per step (Δt): T/n = {dt:.4f} years
    
    **Tree Parameters:**
    - Up factor (u): {u_factor:.4f} ({(u_factor-1)*100:.2f}% per step)
    - Down factor (d): {d_factor:.4f} ({(1-d_factor)*100:.2f}% per step)
    
    **Discount Factor:**
    ```
    discount = e^(-r×Δt) = e^(-{risk_free_rate if asset_class != 'Currency Option' else domestic_rate:.4f}×{dt:.4f}) = {discount:.4f}
    ```
    
    ### Price Tree Formula
    
    At node (j, i):
    ```
    S[j,i] = S₀ × u^(i-j) × d^j
           = {current_price:.2f} × {u_factor:.4f}^(i-j) × {d_factor:.4f}^j
    ```
    where i = time step (0 to n), j = number of down moves (0 to i)
    """)
    
    # No-arbitrage condition check
    st.subheader("Parameter Validation")
    if asset_class == 'Futures Option':
        # For futures, condition is 0 < d < 1 < u
        condition_met = (0 < d_factor < 1 < u_factor)
        condition_text = "0 < d < 1 < u"
    else:
        # For other assets, standard no-arbitrage condition
        if asset_class in ['Stock Option', 'Stock Index Option']:
            adjusted_rate = risk_free_rate - dividend_yield
        elif asset_class == 'Currency Option':
            adjusted_rate = domestic_rate - foreign_rate
        else:
            adjusted_rate = risk_free_rate
            
        growth_factor = np.exp(adjusted_rate * dt)
        condition_met = (d_factor < growth_factor < u_factor)
        condition_text = f"d < e^(adjusted_rate×Δt) < u"
    
    if condition_met:
        st.success(f"✅ **No-arbitrage condition satisfied:** {condition_text}")
    else:
        st.error(f"⚠️ **Warning: No-arbitrage condition may be violated:** {condition_text}")

# Footer
st.markdown("---")
st.markdown("""
**Note**: This binomial tree model assumes:
1. No transaction costs or taxes
2. Interest rates are constant
3. Volatility is constant (when using volatility method)
4. Markets are efficient
5. For currency options, interest rates are constant in both countries
""")

# Add this anywhere in your app (sidebar or footer)
st.markdown(
    """
    <style>
    .social-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 20px 0;
    }
    .social-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.3s;
    }
    .social-icon:hover {
        transform: scale(1.1);
    }
    .github-icon {
        background: #333;
        color: white;
    }
    .linkedin-icon {
        background: #0077B5;
        color: white;
    }
    .gmail-icon {
        background: #D14836;
        color: white;
    }
    </style>
    
    <div class="social-container">
        <a href="https://github.com/rasoul-ehsghi" target="_blank" class="social-icon github-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
        </a>
        <a href="https://www.linkedin.com/in/rasoul-eshghi/" target="_blank" class="social-icon linkedin-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
            </svg>
        </a>
        <a href="mail to:cfte.mehr@gmail.com" class="social-icon gmail-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M24 4.5v15c0 .85-.65 1.5-1.5 1.5H21V7.387l-9 6.463-9-6.463V21H1.5C.649 21 0 20.35 0 19.5v-15c0-.425.162-.8.431-1.068C.7 3.16 1.076 3 1.5 3H2l10 7.25L22 3h.5c.425 0 .8.162 1.069.432.27.268.431.643.431 1.068z"/>
            </svg>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)