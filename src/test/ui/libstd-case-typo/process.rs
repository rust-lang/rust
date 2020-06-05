// checks case typos with libstd::process structs
fn main(){}

fn test_child(_x: child){}
//~^ ERROR: cannot find type `child` in this scope
fn test_childse(_x: ChildStdErr){}
//~^ ERROR: cannot find type `ChildStdErr` in this scope
fn test_childsi(_x: ChildStdIn){}
//~^ ERROR: cannot find type `ChildStdIn` in this scope
fn test_childso(_x: ChildStdOut){}
//~^ ERROR: cannot find type `ChildStdOut` in this scope
fn test_command(_x: command){}
//~^ ERROR: cannot find type `command` in this scope
fn test_exits(_x: Exitstatus){}
//~^ ERROR: cannot find type `Exitstatus` in this scope
fn test_output(_x: output){}
//~^ ERROR: cannot find type `output` in this scope
fn test_stdio(_x: StdIo){}
//~^ ERROR: cannot find type `StdIo` in this scope
