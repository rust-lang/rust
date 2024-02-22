// Testing the display of RwLock and RwLockReadGuard in cdb.

// cdb-only
//@ min-cdb-version: 10.0.18317.1001
//@ compile-flags:-g

// === CDB TESTS ==================================================================================
//
// cdb-command:g
//
// cdb-command:dx l
// cdb-check:l                [Type: std::sync::rwlock::RwLock<i32>]
// cdb-check:    [...] poison           [Type: std::sync::poison::Flag]
// cdb-check:    [...] data             : 0 [Type: core::cell::UnsafeCell<i32>]
//
// cdb-command:dx r
// cdb-check:r                [Type: std::sync::rwlock::RwLockReadGuard<i32>]
// cdb-check:    [...] data             : NonNull([...]: 0) [Type: core::ptr::non_null::NonNull<i32>]
// cdb-check:    [...] inner_lock       : [...] [Type: std::sys::locks::rwlock::windows::RwLock *]

#[allow(unused_variables)]

use std::sync::RwLock;

fn main()
{
    let l = RwLock::new(0);
    let r = l.read().unwrap();
    zzz(); // #break
}

fn zzz() {}
