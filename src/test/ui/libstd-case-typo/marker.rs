// checks case typos with libstd::marker structs
fn main(){}

fn test_phandat(_x: Phantomdata){}
//~^ ERROR: cannot find type `Phantomdata` in this scope
fn test_phanpin(_x: Phantompinned){}
//~^ ERROR: cannot find type `Phantompinned` in this scope
