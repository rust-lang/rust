// checks case typos with libstd::thread structs
fn main(){}

fn test_build(_x: builder){}
//~^ ERROR: cannot find type `builder` in this scope
fn test_jhand(_x: Joinhandle<()>){}
//~^ ERROR: cannot find type `Joinhandle` in this scope
fn test_lkey(_x: Localkey<()>){}
//~^ ERROR: cannot find type `Localkey` in this scope
fn test_thread(_x: thread){}
//~^ ERROR: cannot find type `thread` in this scope
fn test_threadid(_x: ThreadID){}
//~^ ERROR: cannot find type `ThreadID` in this scope
