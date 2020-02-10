#APS_System_Failure_PLotting_Functions

print('Defining Plot_Correlation_w_Class(df, _title):')
def Plot_Correlation_w_Class(df, _title):
     fig, ax = plt.subplots(1, 1, figsize = (18, 9))
     plt.plot(df['class'][:])
     plt.title(_title)
     plt.tick_params(labelbottom=[])
     plt.grid('black')
     plt.ylim(-0.2, 1)
     plt.show()
     return


print('Defining Plot_Probs_Histogram(y_probs, _title):')
def Plot_Probs_Histogram(y_probs, _title):
     fig, ax = plt.subplots(1, 1, figsize = (10, 5))
     sns.distplot(y_probs[:,1], bins = 200, color = 'tab:blue', kde = False)
     plt.title(_title)
     plt.yscale('log')
     plt.show()


print('Defining Plot_Cost_v_DT(y_test, y_probs, sample_name, Zoom_Y_lim): ')
def Plot_Cost_v_DT(y_test, y_probs, sample_name, Zoom_Y_lim): 
    costs = []
    costs_balanced = []
    DT_ = []
    for dt in range(0,200):
        y_pred = y_probs[:,1] >= dt/200
        cm = confusion_matrix(y_test, y_pred)
        cost = cost = 10*cm[0][1] + 500*cm[1][0]
        costs.append(cost)
        DT_.append(dt/200)
        cost_balanced = 10*cm[0][1] + 10*cm[1][0]
        costs_balanced.append(cost_balanced)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18, 6))

    ax1.plot( DT_, costs, label = 'Cost model 10*FP + 500*FN')
    ax1.plot( DT_, costs_balanced, label = 'Balanced Cost model 10*FP + 10*FN')
    ax1.set_title(('Cost vs Decision Threshold For\n{}\nCustom Cost & Balanced Cost Model').format(sample_name))    
    ax1.legend()
    ax1.set_ylim(0, 100000)
    ax1.grid('black')

    ax2.plot( DT_, costs, label = 'Cost model 10*FP + 500*FN')
    ax2.plot( DT_, costs_balanced, label = 'Balanced Cost model 10*FP + 10*FN')
    ax2.plot([0.0196,0.0196],[0,Zoom_Y_lim], 'r--', label = 'Decision Threshold')
    ax2.set_title(('Zoom in of Cost vs Decision Threshold For\n{}\nCustom Cost & Balanced Cost Model').format(sample_name))
    ax2.legend()
    ax2.set_ylim(0, Zoom_Y_lim)
    ax2.set_xlim(0, 0.2)
    ax2.grid('black')
    plt.show()

print('Defining Plot_Conf_Matrix(cm, Title):')
def Plot_Conf_Matrix(cm, Title):
    log_norm = LogNorm(vmin =0, vmax = 15000)
    sns.set(font_scale=1.4)
    plt.title(('{}\nConfusion Matrix\nCost is {}, 500*FN + 10* FP').format(Title, 10*cm[0][1] + 500*cm[1][0]))
    c_map = ['red','red','red','blue','blue','orange','orange','orange','blue']
    sns.heatmap(cm, norm = log_norm, cmap = c_map,  annot=True, cbar = False, fmt = 'g', annot_kws={"size": 24, "weight": 'bold'})
    plt.xlabel('Normal                  Defect\nPredicted')
    plt.ylabel('Actual\n Defect            Normal')
    plt.show()

