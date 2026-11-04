// Verify that pseudo probes are inserted with `-Zpseudo-probe-for-profiling`:
//  - no probe and no probe descriptor without the flag
//  - probes and probe descriptors with the flag
//
//@ add-minicore
//@ revisions: DEFAULT PROBE
//@ compile-flags: -Copt-level=2 -Cdebuginfo=line-tables-only
//@ [PROBE] compile-flags: -Zpseudo-probe-for-profiling
// DEFAULT-NOT: llvm.pseudoprobe
// DEFAULT-NOT: llvm.pseudo_probe_desc
// PROBE: call void @llvm.pseudoprobe(
// PROBE: !llvm.pseudo_probe_desc =

#![feature(no_core)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

extern "C" {
    fn add(_x: i32, _y: i32) -> i32;
    fn compute(_x: i32) -> i32;
    fn cond() -> bool;
}

#[no_mangle]
pub fn f() -> i32 {
    unsafe {
        let mut sum = 0;
        while cond() {
            sum = add(sum, compute(sum));
        }
        sum
    }
}
