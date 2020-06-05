// checks case typos with libstd::path structs
fn main(){}

fn test_ances(_x: ancestors){}
//~^ ERROR: cannot find type `ancestors` in this scope
fn test_comp(_x: components){}
//~^ ERROR: cannot find type `components` in this scope
fn test_pathbuf(_x: Pathbuf){}
//~^ ERROR: cannot find type `Pathbuf` in this scope
fn test_pcomp(_x: Prefixcomponent){}
//~^ ERROR: cannot find type `Prefixcomponent` in this scope
