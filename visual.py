import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

sn.set()
ngn = [[0.22, 0.24], [0.1, 0.44]]

pgn = [[0.5, 0], [0.39, 0.11]]

gnb = [[0.11, 0.45], [0.11, 0.33]]

longformer = [[0.5, 0], [0.5, 0]]

natb = [[0.15, 0.32], [0.22, 0.31]]

patb = [[0.22, 0.28], [0.17, 0.33]]

df_ngn = pd.DataFrame(ngn, index = [i for i in "FA"],
                  columns = [i for i in ["Favour","Against"]])

df_pgn = pd.DataFrame(pgn, index = [i for i in "FA"],
                  columns = [i for i in ["Favour","Against"]])

df_gnb = pd.DataFrame(gnb, index = [i for i in "FA"],
                  columns = [i for i in ["Favour","Against"]])

df_rb = pd.DataFrame(longformer, index = [i for i in "FA"],
                  columns = [i for i in ["Favour","Against"]])

df_natb = pd.DataFrame(natb, index = [i for i in "FA"],
                  columns = [i for i in ["Favour","Against"]])

df_patb = pd.DataFrame(patb, index = [i for i in "FA"],
                  columns = [i for i in ["Favour","Against"]])

fig, ((ax11, ax12), (ax21, ax22), (ax31, ax32)) = plt.subplots(3, 2, sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.01)
fig.subplots_adjust(hspace=0.3)

im = sn.heatmap(df_rb, cmap="crest", ax=ax21, vmin=0, vmax =0.5, cbar=False, annot=True)
ax11.set_title('Naïve-ATB')

sn.heatmap(df_gnb, cmap="crest", ax=ax22, vmin=0, vmax =0.5, cbar=False, annot=True)
ax12.set_title('Preferred-ATB')


sn.heatmap(df_natb, cmap="crest", ax=ax11, vmin=0, vmax =0.5, cbar=False, annot=True)
ax21.set_title('Longformer')

sn.heatmap(df_patb, cmap="crest", ax=ax12, vmin=0, vmax =0.5, cbar=False, annot=True)
ax22.set_title('Graph Network Baseline')


sn.heatmap(df_ngn, cmap="crest", ax=ax31, vmin=0, vmax =0.5, cbar=False, annot=True)
ax31.set_title('Naïve-GN')

sn.heatmap(df_pgn, cmap="crest", ax=ax32, vmin=0, vmax =0.5, cbar=False, annot=True)
ax32.set_title('Preferred-GN')

mappable = im.get_children()[0]
plt.colorbar(mappable, ax = [ax11,ax12,ax21, ax22, ax31, ax32], orientation = 'vertical')

fig.savefig("out.pdf", bbox_inches=None)
plt.show()
exit()
