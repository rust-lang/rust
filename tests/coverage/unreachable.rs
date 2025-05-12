#![feature(core_intrinsics, coverage_attribute)]
//@ edition: 2021

// <https://github.com/rust-lang/rust/issues/116171>
// If we instrument a function for coverage, but all of its counter-increment
// statements are removed by MIR optimizations, LLVM will think it isn't
// instrumented and it will disappear from coverage maps and coverage reports.
// Most MIR opts won't cause this because they tend not to remove statements
// from bb0, but `UnreachablePropagation` can do so if it sees that bb0 ends
// with `TerminatorKind::Unreachable`.

use std::hint::{black_box, unreachable_unchecked};

static UNREACHABLE_CLOSURE: fn() = || unsafe { unreachable_unchecked() };

fn unreachable_function() {
    unsafe { unreachable_unchecked() }
}

// Use an intrinsic to more reliably trigger unreachable-propagation.
fn unreachable_intrinsic() {
    unsafe { std::intrinsics::unreachable() }
}

#[coverage(off)]
fn main() {
    if black_box(false) {
        UNREACHABLE_CLOSURE();
    }
    if black_box(false) {
        unreachable_function();
    }
    if black_box(false) {
        unreachable_intrinsic();
    }
}
