// This test checks that -Ztrack-diagnostics reports the correct source locations for an ICE
// triggered with `span_bug!`.
//
//@ compile-flags: -Zvalidate-mir -Ztrack-diagnostics
//@ rustc-env:RUST_BACKTRACE=0
//@ failure-status: 101
//
// Normalize the emitted location so this doesn't need
// updating everytime someone adds or removes a line.
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:CC"
//@ normalize-stderr: "note: rustc .+ running on .+" -> "note: rustc $$VERSION running on $$TARGET"
//@ normalize-stderr: "/rustc(?:-dev)?/[a-z0-9.]+/" -> ""
//@ normalize-stderr: "track7\[....\]" -> "track7[HASH]"
// The test becomes too flaky if we care about exact args. If `-Z ui-testing`
// from compiletest and `-Z track-diagnostics` from `// compile-flags` at the
// top of this file are present, then assume all args are present.
//@ normalize-stderr: "note: compiler flags: .*-Z ui-testing.*-Z track-diagnostics" -> "note: compiler flags: ... -Z ui-testing ... -Z track-diagnostics"

#![feature(custom_mir, core_intrinsics)]
extern crate core;
use core::intrinsics::mir::*;

fn bar(_x: i32) {}

// Use of `mir!` here is just because it's an easy way to trigger a `span_bug!`.
#[custom_mir(dialect = "built")]
pub fn main() {
    mir! {
        let a: (i32, i32);
        {
            a = (1, 2);
            Call(RET = bar(Move(a.0)), ReturnTo(retblock), UnwindContinue())
            //~^ ERROR broken MIR in
            //~| ERROR encountered `Move` of a non-local, non-box place in `Call` terminator
        }
        retblock = {
            Return()
        }
    }
}
