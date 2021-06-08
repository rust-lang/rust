// Testing the display of RwLock and RwLockReadGuard in cdb.

// cdb-only
// min-cdb-version: 10.0.18317.1001
// compile-flags:-g

// === CDB TESTS ==================================================================================
//
// cdb-command:g
//
// cdb-command:dx l
// cdb-check:l                [Type: std::sync::rwlock::RwLock<i32>]
// cdb-check:    [+0x000] inner            : [...] [Type: std::sys_common::rwlock::RWLock *]
// cdb-check:    [+0x008] poison           [Type: std::sync::poison::Flag]
// cdb-check:    [+0x00c] data             [Type: core::cell::UnsafeCell<i32>]
//
// cdb-command:dx r
// cdb-check:r                [Type: std::sync::rwlock::RwLockReadGuard<i32>]
// cdb-check:    [+0x000] lock             : [...] [Type: std::sync::rwlock::RwLock<i32> *]
//
// cdb-command:dx r.lock->data,d
// cdb-check:r.lock->data,d   [Type: core::cell::UnsafeCell<i32>]
// cdb-check:    [+0x000] value            : 0 [Type: int]

#[allow(unused_variables)]

use std::sync::RwLock;

fn main()
{
    let l = RwLock::new(0);
    let r = l.read().unwrap();
    zzz(); // #break
}

fn zzz() {}
