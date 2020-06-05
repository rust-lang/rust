// checks case typos with libstd::pin structs
fn main(){}

fn test_pin(_x: pin<()>){}
//~^ ERROR: cannot find type `pin` in this scope
