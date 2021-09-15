// Testing the display of Mutex and MutexGuard in cdb.

// cdb-only
// min-cdb-version: 10.0.21287.1005
// compile-flags:-g
// FIXME: Failed on update to 10.0.22000.1
// ignore-windows

// === CDB TESTS ==================================================================================
//
// cdb-command:g
//
// cdb-command:dx m,d
// cdb-check:m,d              [Type: std::sync::mutex::Mutex<i32>]
// cdb-check:    [...] inner            [Type: std::sys_common::mutex::MovableMutex]
// cdb-check:    [...] poison           [Type: std::sync::poison::Flag]
// cdb-check:    [...] data             [Type: core::cell::UnsafeCell<i32>]

//
// cdb-command:dx m.data,d
// cdb-check:m.data,d         [Type: core::cell::UnsafeCell<i32>]
// cdb-check:    [...] value            : 0 [Type: int]

//
// cdb-command:dx lock,d
// cdb-check:lock,d           : Ok [Type: enum$<core::result::Result<std::sync::mutex::MutexGuard<i32>, enum$<std::sync::poison::TryLockError<std::sync::mutex::MutexGuard<i32> >, 0, 1, Poisoned> > >]
// cdb-check:    [...] variant$         : Ok (0) [Type: core::result::Result]
// cdb-check:    [...] __0              [Type: std::sync::mutex::MutexGuard<i32>]

use std::sync::Mutex;

#[allow(unused_variables)]
fn main()
{
    let m = Mutex::new(0);
    let lock = m.try_lock();
    zzz(); // #break
}

fn zzz() {}
