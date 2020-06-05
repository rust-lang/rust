// checks case typos with libstd::ptr structs
fn main(){}

fn test_nonnull(_x: Nonnull<()>){}
//~^ ERROR: cannot find type `Nonnull` in this scope
