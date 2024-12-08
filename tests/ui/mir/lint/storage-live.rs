//@ compile-flags: -Zlint-mir -Ztreat-err-as-bug
//@ failure-status: 101
//@ error-pattern: broken MIR in
//@ error-pattern: StorageLive(_1) which already has storage here
//@ normalize-stderr-test: "note: .*\n\n" -> ""
//@ normalize-stderr-test: "thread 'rustc' panicked.*\n" -> ""
//@ normalize-stderr-test: "storage_live\[....\]" -> "storage_live[HASH]"
//@ normalize-stderr-test: "(delayed at [^:]+):\d+:\d+ - " -> "$1:LL:CC - "
//@ rustc-env:RUST_BACKTRACE=0

#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;
use core::ptr::{addr_of, addr_of_mut};

#[custom_mir(dialect = "built")]
fn multiple_storage() {
    mir! {
        let a: usize;
        {
            StorageLive(a);
            StorageLive(a);
            Return()
        }
    }
}

fn main() {
    multiple_storage()
}
