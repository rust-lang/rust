// Testing the display of RwLockWriteGuard.

// cdb-only
//@ min-cdb-version: 10.0.18317.1001
//@ compile-flags:-g

// === CDB TESTS ==================================================================================
//
// cdb-command:g
//
// cdb-command:dx w
// cdb-check:w                [Type: std::sync::poison::rwlock::RwLockWriteGuard<i32>]
// cdb-check:    [...] lock             : [...] [Type: std::sync::poison::rwlock::RwLock<i32> *]
// cdb-check:    [...] poison           [Type: std::sync::poison::Guard]

#[allow(unused_variables)]

use std::sync::RwLock;

fn main()
{
    let l = RwLock::new(0);
    let w = l.write().unwrap();
    zzz(); // #break
}

fn zzz() {}
