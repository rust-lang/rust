// Testing the display of Mutex and MutexGuard in cdb.

// cdb-only
// min-cdb-version: 10.0.21287.1005
// compile-flags:-g
// ignore-tidy-linelength

// === CDB TESTS ==================================================================================
//
// cdb-command:g
//
// cdb-command:dx m,d
// cdb-check:m,d              [Type: std::sync::mutex::Mutex<i32>]
// cdb-check:    [+0x000] inner            [Type: std::sys_common::mutex::MovableMutex]
// cdb-check:    [+0x008] poison           [Type: std::sync::poison::Flag]
// cdb-check:    [+0x00c] data             [Type: core::cell::UnsafeCell<i32>]

//
// cdb-command:dx m.data,d
// cdb-check:m.data,d         [Type: core::cell::UnsafeCell<i32>]
// cdb-check:    [+0x000] value            : 0 [Type: int]

//
// cdb-command:dx lock,d
// cdb-check:lock,d           : Ok({...}) [Type: core::result::Result<std::sync::mutex::MutexGuard<i32>, std::sync::poison::TryLockError<std::sync::mutex::MutexGuard<i32>>>]
// cdb-check:    [value]          [Type: std::sync::mutex::MutexGuard<i32>]

use std::sync::Mutex;

#[allow(unused_variables)]
fn main()
{
    let m = Mutex::new(0);
    let lock = m.try_lock();
    zzz(); // #break
}

fn zzz() {}
