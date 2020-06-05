// checks case typos with libstd::io structs
fn main(){}

fn test_bufrd(_x: Bufreader<()>){}
//~^ ERROR: cannot find type `Bufreader` in this scope
fn test_bufwr(_x: Bufwriter<()>){}
//~^ ERROR: cannot find type `Bufwriter` in this scope
fn test_bytes(_x: bytes<()>){}
//~^ ERROR: cannot find type `bytes` in this scope
fn test_chain(_x: chain<(), ()>){}
//~^ ERROR: cannot find type `chain` in this scope
fn test_cursor(_x: cursor<()>){}
//~^ ERROR: cannot find type `cursor` in this scope
fn test_empty(_x: empty){}
//~^ ERROR: cannot find type `empty` in this scope
fn test_ios(_x: Ioslice){}
//~^ ERROR: cannot find type `Ioslice` in this scope
fn test_iosm(_x: IosliceMut){}
//~^ ERROR: cannot find type `IosliceMut` in this scope
fn test_linewr(_x: Linewriter<()>){}
//~^ ERROR: cannot find type `Linewriter` in this scope
fn test_lines(_x: lines<()>){}
//~^ ERROR: cannot find type `lines` in this scope
fn test_repeat(_x: repeat){}
//~^ ERROR: cannot find type `repeat` in this scope
fn test_sink(_x: sink){}
//~^ ERROR: cannot find type `sink` in this scope
fn test_split(_x: split<()>){}
//~^ ERROR: cannot find type `split` in this scope
fn test_stderr(_x: StdErr){}
//~^ ERROR: cannot find type `StdErr` in this scope
fn test_stderr_l(_x: StdErrLock){}
//~^ ERROR: cannot find type `StdErrLock` in this scope
fn test_stdind(_x: StdIn){}
//~^ ERROR: cannot find type `StdIn` in this scope
fn test_stdin_l(_x: StdInLock){}
//~^ ERROR: cannot find type `StdInLock` in this scope
fn test_stdout(_x: StdOut){}
//~^ ERROR: cannot find type `StdOut` in this scope
fn test_stdout_l(_x: StdOutLock){}
//~^ ERROR: cannot find type `StdOutLock` in this scope
fn test_take(_x: take){}
//~^ ERROR: cannot find type `take` in this scope
