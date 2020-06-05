// checks case typos with libstd::task structs
fn main(){}

fn test_context(_x: context){}
//~^ ERROR: cannot find type `context` in this scope
fn test_rwake(_x: Rawwaker){}
//~^ ERROR: cannot find type `Rawwaker` in this scope
fn test_rwakevt(_x: RawwakerVTable){}
//~^ ERROR: cannot find type `RawwakerVTable` in this scope
fn test_waker(_x: waker){}
//~^ ERROR: cannot find type `waker` in this scope
