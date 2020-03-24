'''Main script'''
from utils import PopulationCode, PopulationCodeV2, GSPopulationCode, GSPopulationCodeV2, GSPopulationCodeExact

x = GSPopulationCodeV2()
opt_params = x.optimize_fisher()
x.plot_results(opt_params)

# y = GSPopulationCodeExact()
# print(y.loss)

