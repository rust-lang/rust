// checks case typos with libstd::fmt structs
fn main(){}

fn test_dbglist(_x: Debuglist){}
//~^ ERROR: cannot find type `Debuglist` in this scope
fn test_dbgmap(_x: Debugmap){}
//~^ ERROR: cannot find type `Debugmap` in this scope
fn test_dbgset(_x: Debugset){}
//~^ ERROR: cannot find type `Debugset` in this scope
fn test_dbgstruct(_x: Debugstruct){}
//~^ ERROR: cannot find type `Debugstruct` in this scope
fn test_dbgtuple(_x: Debugtuple){}
//~^ ERROR: cannot find type `Debugtuple` in this scope
fn test_fmter(mut _x: formatter){}
//~^ ERROR: cannot find type `formatter` in this scope
