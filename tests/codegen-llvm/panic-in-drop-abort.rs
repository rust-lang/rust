//@ compile-flags: -Z panic-in-drop=abort -Copt-level=3
//@ ignore-msvc

// Ensure that unwinding code paths are eliminated from the output after
// optimization.

// This test uses ignore-msvc, because the expected optimization does not happen on targets using
// SEH exceptions with the new LLVM pass manager anymore, see
// https://github.com/llvm/llvm-project/issues/51311.

// CHECK-NOT: {{(call|invoke).*}}should_not_appear_in_output

#![crate_type = "lib"]
use std::any::Any;
use std::mem::forget;

pub struct ExternDrop;
impl Drop for ExternDrop {
    #[inline(always)]
    fn drop(&mut self) {
        // This call may potentially unwind.
        extern "Rust" {
            fn extern_drop();
        }
        unsafe {
            extern_drop();
        }
    }
}

struct AssertNeverDrop;
impl Drop for AssertNeverDrop {
    #[inline(always)]
    fn drop(&mut self) {
        // This call should be optimized away as unreachable.
        extern "C" {
            fn should_not_appear_in_output();
        }
        unsafe {
            should_not_appear_in_output();
        }
    }
}

#[no_mangle]
pub fn normal_drop(x: ExternDrop) {
    let guard = AssertNeverDrop;
    drop(x);
    forget(guard);
}

#[no_mangle]
pub fn indirect_drop(x: Box<dyn Any>) {
    let guard = AssertNeverDrop;
    drop(x);
    forget(guard);
}
