import seaborn as sns
import matplotlib.pyplot as plt

def plot_countplots(df, columns, target=None, n_cols=3, figsize=(15, 5), colors=["#6099ff", "#ffa060", "#60ff99"]):
    """
    Vẽ nhiều countplot cho các cột trong `columns` với màu cố định.
    
    Args:
        df (pd.DataFrame): DataFrame dữ liệu.
        columns (list): Danh sách các cột muốn vẽ.
        target (str, optional): Nếu có, dùng làm hue (phân loại theo cột đích).
        n_cols (int): Số biểu đồ mỗi hàng.
        figsize (tuple): Kích thước tổng thể figure.
        color (str): Mã màu (vd: '#e78ac3', 'skyblue', 'salmon', ...).
    """
    n = len(columns)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.countplot(data=df, x=col, hue=target, ax=axes[i], palette=colors[:len(df[target].unique())] if target else colors[0])
        axes[i].set_title(f"Countplot of {col}", fontsize=11, fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

        for container in axes[i].containers:
            axes[i].bar_label(container, fmt='%d', label_type='edge', fontsize=9)
        # Hiển thị legend nếu có hue
        if target:
            axes[i].legend(title=target, fontsize=8, title_fontsize=9)
        else:
            axes[i].legend().set_visible(False)

    # Ẩn subplot trống
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
    
def plot_piecharts(df, cols, hue=None, colors=None):
    """
    Vẽ nhiều biểu đồ tròn (pie chart) cho các cột phân loại.
    - df: DataFrame
    - cols: danh sách các cột cần vẽ
    - hue: cột nhóm (nếu muốn chia theo lớp, ví dụ 'Survived')
    - colors: danh sách màu tùy chọn
    """
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        if hue:
            # Nếu có hue, ta vẽ nhiều lát tương ứng với mỗi nhóm
            data = (
                df.groupby([col, hue])
                .size()
                .reset_index(name='Count')
            )
            pivot = data.pivot(index=col, columns=hue, values='Count').fillna(0)
            pivot_percent = pivot.div(pivot.sum(axis=1), axis=0) * 100

            # Vẽ tổng hợp cho từng giá trị của hue
            pivot.sum(axis=1).plot.pie(
                ax=ax,
                autopct='%.1f%%',
                startangle=90,
                colors=colors,
                textprops={'fontsize': 9}
            )
            ax.set_ylabel('')
            ax.set_title(f'{col} distribution (overall)', fontweight='bold')

        else:
            counts = df[col].value_counts(normalize=True) * 100
            counts.plot.pie(
                ax=ax,
                autopct='%.1f%%',
                startangle=90,
                explode=(0.05, 0.05),
                colors=colors,
                textprops={'fontsize': 9}
            )
            ax.set_ylabel('')
            ax.set_title(f'Percentage of {col}', fontweight='bold')

    plt.tight_layout()
    plt.show()
    
def plot_percentplots(df, cols, hue=None, colors=None):
    """
    Vẽ nhiều biểu đồ phần trăm (tương tự countplot nhưng hiển thị %).
    - cols: danh sách cột cần vẽ.
    - hue: cột phân nhóm (nếu có).
    - colors: danh sách màu thủ công.
    """
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))

    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        # Tính tỷ lệ phần trăm
        if hue:
            count_df = (
                df.groupby([col, hue])
                .size()
                .reset_index(name='Count')
            )
            count_df['Percent'] = (
                count_df.groupby(col)['Count']
                .transform(lambda x: x / x.sum() * 100)
            )
            sns.barplot(
                data=count_df,
                x=col,
                y='Percent',
                hue=hue,
                ax=ax,
                palette=colors
            )
        else:
            count_df = df[col].value_counts(normalize=True).mul(100).reset_index()
            count_df.columns = [col, 'Percent']
            sns.barplot(
                data=count_df,
                x=col,
                y='Percent',
                ax=ax,
                color=colors[0] if colors else None
            )

        # Cấu hình biểu đồ
        ax.set_title(f'Percentage of {col}', fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Percentage (%)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        # Ghi nhãn phần trăm lên đầu cột
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_distribution(df, cols, group_col, colors=['#FFD700', '#7EC0EE'], bins=30, alpha=0.4):
    """
    Vẽ biểu đồ phân phối (hist + KDE) cho nhiều cột, chia theo nhóm (group_col).
    
    Params:
        df: DataFrame
        cols: danh sách các cột số cần vẽ
        group_col: cột nhóm (ví dụ 'label', 'Survived', 'Outcome'...)
        colors: dict, ví dụ {'healthy': '#8CC6DB', 'diabetic': '#FFD966'}
        bins: số lượng bins của histogram
        alpha: độ trong suốt của histogram
    """
    n = len(cols)
    ncols = 2
    nrows = n//ncols + (n % ncols > 0)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()
    
    unique_groups = df[group_col].unique()
    
    # Tự chọn màu nếu không truyền vào
    if colors is None:
        palette = sns.color_palette("Set2", len(unique_groups))
        colors = dict(zip(unique_groups, palette))
    
    for ax, col in zip(axes, cols):
        for g in unique_groups:
            subset = df[df[group_col] == g][col].dropna()
            sns.histplot(
                subset,
                bins=bins,
                kde=True,
                stat='density',
                color=colors[g],
                label=str(g),
                ax=ax,
                alpha=alpha,
                edgecolor=None,
            )
            
            
        
        ax.set_title(f'Distribution of {col} by {group_col}', fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.legend(title=group_col)
        ax.grid(True, alpha=0.2)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.show()