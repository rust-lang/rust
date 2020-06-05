// checks case typos with libstd::time structs
fn main(){}

fn test_dur(_x: duration){}
//~^ ERROR: cannot find type `duration` in this scope
fn test_ins(_x: instant){}
//~^ ERROR: cannot find type `instant` in this scope
fn test_systime(_x: Systemtime){}
//~^ ERROR: cannot find type `Systemtime` in this scope
