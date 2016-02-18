library(pander)
md.table = function(x) pandoc.table(x, split.table=Inf, style='rmarkdown')

mean.sd = function(x, digits=2){
	# Returns string with "Mean (Standard deviation)" of intput.
	# Nice for APA tables!
	x.mean = mean(x, na.rm=T) %>% round(digits)
	x.sd = sd(x, na.rm=T) %>% round(digits)
	paste(x.mean, ' (', x.sd, ')', sep='') %>% return()
}


this.ci = function(model, logistic=T){
	param.names = names(fixef(model))
	param = param.names[2:length(param.names)] # Don't bother with intercept
	ci = stats::confint(model, parm=param)
	b = fixef(model)[param]
	beta = cbind(beta=b, ci)
	if(logistic){
		out = rbind(beta, exp(beta), 1/exp(beta))
		headings = c('Estimate', 'Exponent', '1/Exponent')	
		rownames(out) = do.call(paste, expand.grid(param, ':', headings))
	} else {
		out = rbind(beta, exp(beta), 1/exp(beta))
	}
	return(out)
}

z = function(x){scale(x)}
center  = function(x){scale(x, scale=F)}
median.split = function(x) x > median(x)

pretty.p = function (p, digits=4){
	p = round(p, digits)
	if(p < .0001) { return (paste("<.0001", "***"))}
	if(p < .001) { return (paste("<.001", "***"))}
	if(p < .01)  { return (paste(substr(p, 2, 5), "**"))}
	if(p < .05)  { return (paste(substr(p, 2, 5), "*"))}
	if(p < .1)   { return (paste(substr(p, 2, 5), "."))}
	else{return (as.character(p))}
}

coef.table = function(m){
	co = summary(m) %>% coef
	co[,1:ncol(co)-1] %<>% round(3)
	co[,ncol(co)] %<>% sapply(pretty.p)
	md.table(co)
}
anova.table = function(m1, m2){
	a = anova(m1, m2) %>% data.frame
	a[2, ncol(a)] %<>% pretty.p
	a$Chisq %<>% round(1) %>% as.character
	md.table(a)
}
	
protect = function(x) {
	p = capture.output(x)
	cat(paste('\t## ', p, '\n', sep=''))
}

# nlopt optimizer for faster convergence. See https://cran.r-project.org/web/packages/lme4/vignettes/lmerperf.html
nlopt <- function(par, fn, lower, upper, control) {
	.nloptr <<- res <- nloptr(par, fn, lb = lower, ub = upper, 
							  opts = list(algorithm = "NLOPT_LN_BOBYQA", print_level = 1,
							  			maxeval = 1000, xtol_abs = 1e-6, ftol_abs = 1e-6))
	list(par = res$solution,
		 fval = res$objective,
		 conv = if (res$status > 0) 0 else res$status,
		 message = res$message
	)
}

# Colours
green = "#4daf4a"
red = "#e41a1c"
blue = "#377eb8"
blue2 = "#37B8B2"
black = '#000000'
orange = "#FFAE00"
