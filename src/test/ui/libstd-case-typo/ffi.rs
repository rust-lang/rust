// checks case typos with libstd::ffi structs
fn main(){}

fn test_cstr(_x: cStr){}
//~^ ERROR: cannot find type `cStr` in this scope
fn test_osstr(_x: Osstr){}
//~^ ERROR: cannot find type `Osstr` in this scope
fn test_osstring(_x: Osstring){}
//~^ ERROR: cannot find type `Osstring` in this scope
