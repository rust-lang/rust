// Testing the display of JoinHandle and Thread in cdb.

// cdb-only
//@ min-cdb-version: 10.0.18317.1001
//@ compile-flags:-g

// === CDB TESTS ==================================================================================
//
// cdb-command:g
//
// cdb-command:dx join_handle,d
// cdb-check:join_handle,d    [Type: std::thread::JoinHandle<tuple$<> >]
// cdb-check:    [...] __0              [Type: std::thread::JoinInner<tuple$<> >]
//
// cdb-command:dx t,d
// cdb-check:t,d              : [...] [Type: std::thread::Thread *]
// cdb-check:[...] inner [...][Type: core::pin::Pin<alloc::sync::Arc<std::thread::Inner,alloc::alloc::Global> >]

use std::thread;

#[allow(unused_variables)]
fn main()
{
    let join_handle = thread::spawn(|| {
        println!("Initialize a thread");
    });
    let t = join_handle.thread();
    zzz(); // #break
}

fn zzz() {}
