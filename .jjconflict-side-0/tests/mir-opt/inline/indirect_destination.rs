// Test for inlining with an indirect destination place.
//
//@ test-mir-pass: Inline
//@ edition: 2021
//@ needs-unwind
#![crate_type = "lib"]
#![feature(custom_mir, core_intrinsics)]
use core::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "initial")]
// CHECK-LABEL: fn f(
// CHECK:      bb1: {
// CHECK-NEXT:   StorageLive([[A:.*]]);
// CHECK-NEXT:   [[A]] = &raw mut (*_1);
// CHECK-NEXT:   StorageLive([[B:.*]]);
// CHECK-NEXT:   [[B]] = const 42_u8;
// CHECK-NEXT:   (*[[A]]) = move [[B]];
// CHECK-NEXT:   StorageDead([[B]]);
// CHECK-NEXT:   StorageDead([[A]]);
// CHECK-NEXT:   goto -> bb1;
// CHECK-NEXT: }
pub fn f(a: *mut u8) {
    mir! {
        {
            Goto(bb1)
        }
        bb1 = {
            Call(*a = g(), ReturnTo(bb1), UnwindUnreachable())
        }
    }
}

#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(always)]
fn g() -> u8 {
    mir! {
        {
            RET = 42;
            Return()
        }
    }
}

// Reading the argument must not invalidate the saved destination pointer.
#[custom_mir(dialect = "runtime", phase = "initial")]
// CHECK-LABEL: fn indexed(
// CHECK: [[DEST:_[0-9]+]] = &raw mut _2[_1];
// CHECK: [[ARG:_[0-9]+]] = copy _2[_1];
// CHECK-NEXT: [[RET:_[0-9]+]] = copy [[ARG]];
// CHECK-NEXT: (*[[DEST]]) = move [[RET]];
pub fn indexed(index: usize) -> u32 {
    mir! {
        let values: [u32; 2];
        {
            values = [41, 99];
            Call(values[index] = identity(values[index]), ReturnTo(done), UnwindContinue())
        }
        done = {
            RET = values[index];
            Return()
        }
    }
}

#[inline(always)]
fn identity(value: u32) -> u32 {
    value
}

// Saving an uninhabited destination must not construct an invalid reference.
#[custom_mir(dialect = "runtime", phase = "initial")]
// CHECK-LABEL: fn uninhabited(
// CHECK: [[PTR:_[0-9]+]] = &raw mut _1;
// CHECK: [[DEST:_[0-9]+]] = &raw mut (*[[PTR]]);
// CHECK: panic_fmt
pub fn uninhabited() {
    mir! {
        let slot: !;
        let ptr: *mut !;
        {
            ptr = &raw mut slot;
            Call(*ptr = fail(), ReturnTo(done), UnwindContinue())
        }
        done = {
            Return()
        }
    }
}

#[inline(always)]
fn fail() -> ! {
    panic!("expected panic")
}
