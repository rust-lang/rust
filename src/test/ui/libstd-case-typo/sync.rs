// checks case typos with libstd::sync structs
fn main(){}

fn test_arc(_x: arc<()>){}
//~^ ERROR: cannot find type `arc` in this scope
fn test_barrier(_x: barrier<()>){}
//~^ ERROR: cannot find type `barrier` in this scope
fn test_bwr(_x: BarrierwaitResult<()>){}
//~^ ERROR: cannot find type `BarrierwaitResult` in this scope
fn test_cvar(_x: CondVar<()>){}
//~^ ERROR: cannot find type `CondVar` in this scope
fn test_mutex(_x: mutex<()>){}
//~^ ERROR: cannot find type `mutex` in this scope
fn test_mutexguard(_x: Mutexguard<()>){}
//~^ ERROR: cannot find type `Mutexguard` in this scope
fn test_once(_x: once<()>){}
//~^ ERROR: cannot find type `once` in this scope
fn test_rwl(_x: RWlock<()>){}
//~^ ERROR: cannot find type `RWlock` in this scope
fn test_rwlrg(_x: RWlockReadGuard<()>){}
//~^ ERROR: cannot find type `RWlockReadGuard` in this scope
fn test_rwlwg(_x: RWlockWriteGuard<()>){}
//~^ ERROR: cannot find type `RWlockWriteGuard` in this scope
fn test_wtr(_x: WaittimeoutResult<()>){}
//~^ ERROR: cannot find type `WaittimeoutResult` in this scope
fn test_weak(_x: weak<()>){}
//~^ ERROR: cannot find type `weak` in this scope
