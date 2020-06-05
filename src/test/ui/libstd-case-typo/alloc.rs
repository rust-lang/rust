// checks case typos with libstd::alloc structs
fn main(){}

fn test_layout(_x: LayOut){}
//~^ ERROR: cannot find type `LayOut` in this scope
fn test_system(_x: system){}
//~^ ERROR: cannot find type `system` in this scope
