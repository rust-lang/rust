// checks case typos with libstd::panic structs
fn main(){}

fn test_assertus(_x: AssertUnwindsafe<()>){}
//~^ ERROR: cannot find type `AssertUnwindsafe` in this scope
fn test_loc(_x: location<()>){}
//~^ ERROR: cannot find type `location` in this scope
fn test_pinfo(_x: Panicinfo<()>){}
//~^ ERROR: cannot find type `Panicinfo` in this scope
