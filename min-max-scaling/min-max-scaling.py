def min_max_scaling(data: list) -> list:
    """
    Returns each data column scaled to the range from 0 through 1.
    """
    # Write code here
    rows=len(data)
    cols=len(data[0])
    res=[[0.0] * cols for _ in range(rows)]
    for j in range(cols):
        mini=min(data[i][j] for i in range(rows))
        maxi=max(data[i][j] for i in range(rows))
        for i in range(rows):
            if maxi == mini:
                res[i][j] = 0.0
            else:
                res[i][j] = (data[i][j] - mini) / (maxi - mini)

    return res
    #pass