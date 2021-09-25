// compile-flags: -Z panic-in-drop=abort -O -Z new-llvm-pass-manager=no

// Ensure that unwinding code paths are eliminated from the output after
// optimization.

// This test uses -Z new-llvm-pass-manager=no, because the expected optimization does not happen
// on targets using SEH exceptions (i.e. MSVC) anymore. The core issue is that Rust promises that
// the drop_in_place() function can't unwind, but implements it in a way that *can*, because we
// currently go out of our way to allow longjmps, which also use the unwinding mechanism on MSVC
// targets. We should either forbid longjmps, or not assume nounwind, making this optimization
// incompatible with the current behavior of running cleanuppads on longjmp unwinding.

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
