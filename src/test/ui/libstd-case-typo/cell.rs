// checks case typos with libstd::cell structs
fn main(){}

fn test_cell(_x: cell<()>){}
//~^ ERROR: cannot find type `cell` in this scope
