//! Regression test for <https://github.com/rust-lang/rust/issues/60044>.

//@ assembly-output: emit-asm
//@ only-x86_64

// We want to check that the None case is optimized away
//@ compile-flags: -O

// Simplify the generated assembly
//@ compile-flags: -Cforce-unwind-tables=no

#![crate_type = "lib"]

use std::num::NonZeroUsize;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

pub static X: AtomicUsize = AtomicUsize::new(1);

/// This function function shall look like this:
/// ```
/// some_non_zero_from_atomic_get:
///         movq    _RNvCs7C4TuIcXqwO_25some_non_zero_from_atomic1X@GOTPCREL(%rip), %rax
///         movq    (%rax), %rax
///         retq
/// ```
// CHECK-LABEL: some_non_zero_from_atomic_get:
// CHECK-NEXT: movq    {{[_a-zA-Z0-9]+}}@GOTPCREL(%rip), %rax
// CHECK-NEXT: movq    (%rax), %rax
// CHECK-NEXT: retq
#[no_mangle]
pub unsafe fn some_non_zero_from_atomic_get() -> Option<NonZeroUsize> {
    let x = X.load(Relaxed);
    Some(NonZeroUsize::new_unchecked(x))
}

/// This function shall be identical to the above, which means:
// CHECK-DAG: some_non_zero_from_atomic_get2 = some_non_zero_from_atomic_get
#[no_mangle]
pub unsafe fn some_non_zero_from_atomic_get2() -> usize {
    match some_non_zero_from_atomic_get() {
        Some(x) => x.get(),
        None => unreachable!(), // shall be optimized out
    }
}
