// checks case typos with libstd::rc structs
fn main(){}

fn test_rc(_x: rc<()>){}
//~^ ERROR: cannot find type `rc` in this scope
fn test_weak(_x: weak<()>){}
//~^ ERROR: cannot find type `weak` in this scope
