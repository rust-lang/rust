// checks case typos with libstd::env structs
fn main(){}

fn test_args(_x: args){}
//~^ ERROR: cannot find type `args` in this scope
fn test_argsos(_x: Argsos){}
//~^ ERROR: cannot find type `Argsos` in this scope
fn test_sp(_x: Splitpaths<'_>){}
//~^ ERROR: cannot find type `Splitpaths` in this scope
fn test_vars(_x: vars){}
//~^ ERROR: cannot find type `vars` in this scope
fn test_varsos(_x: Varsos){}
//~^ ERROR: cannot find type `Varsos` in this scope
