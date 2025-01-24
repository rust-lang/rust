// Testing the display of Mutex and MutexGuard in cdb.

// cdb-only
//@ min-cdb-version: 10.0.21287.1005
//@ compile-flags:-g

// === CDB TESTS ==================================================================================
//
// cdb-command:g
//
// cdb-command:dx m,d
// cdb-check:m,d              [Type: std::sync::poison::mutex::Mutex<i32>]
// cdb-check:    [...] inner            [Type: std::sys::sync::mutex::futex::Mutex]
// cdb-check:    [...] poison           [Type: std::sync::poison::Flag]
// cdb-check:    [...] data             : 0 [Type: core::cell::UnsafeCell<i32>]

//
// cdb-command:dx m.data,d
// cdb-check:m.data,d         : 0 [Type: core::cell::UnsafeCell<i32>]
// cdb-check:    [<Raw View>]     [Type: core::cell::UnsafeCell<i32>]

//
// cdb-command:dx _lock,d
// cdb-check:_lock,d          : Ok [Type: enum2$<core::result::Result<std::sync::poison::mutex::MutexGuard<i32>,enum2$<std::sync::poison::TryLockError<std::sync::poison::mutex::MutexGuard<i32> > > > >]
// cdb-check:    [...] __0              [Type: std::sync::poison::mutex::MutexGuard<i32>]

use std::sync::Mutex;

fn main() {
    let m = Mutex::new(0);
    let _lock = m.try_lock();

    println!("this line avoids an `Ambiguous symbol error` while setting the breakpoint");

    zzz(); // #break
}

#[inline(never)]
fn zzz() {}
