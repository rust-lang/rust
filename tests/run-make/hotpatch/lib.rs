// hotpatch has two requirements:
// 1. the first instruction of a functin must be at least two bytes long
// 2. there must not be a jump to the first instruction

// the hotpatch flag should insert nops as needed to fullfil the requirements,
// but only if the the function does not already fulfill them.
// Over 99% of function in regular codebases already fulfill the conditions,
//  so its important to check that those are
// unneccessarily affected

// ------------------------------------------------------------------------------------------------

// regularly this tailcall would jump to the first instruction the function
// CHECK-LABEL: <tailcall_fn>:
// CHECK: jne 0x0 <tailcall_fn>

// hotpatch insert nops so that the tailcall will not jump to the first instruction of the function
// HOTPATCH-LABEL: <tailcall_fn>:
// HOTPATCH-NOT: jne 0x0 <tailcall_fn>

#[no_mangle]
pub fn tailcall_fn() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    if COUNT.fetch_sub(1, Ordering::Relaxed) != 0 {
        tailcall_fn()
    }
}

// ------------------------------------------------------------------------------------------------

// empty_fn just returns. Note that 'ret' is a single byte instruction, but hotpatch requires a two
// or more byte instructions to be at the start of the functions.
// Preferably we would also tests a different single byte instruction,
// but I was not able to make rustc emit anything but 'ret'.

// CHECK-LABEL: <empty_fn>:
// CHECK-NEXT: ret

// HOTPATCH-LABEL: <empty_fn>:
// HOTPATCH-NOT: ret
// HOTPATCH: ret

#[no_mangle]
#[inline(never)]
pub fn empty_fn() {}

// ------------------------------------------------------------------------------------------------

// return_42 should not be affected by hotpatch

// CHECK-LABEL: <return_42>:
// CHECK-NEXT: 0:
// CHECK-NEXT: ret

// HOTPATCH-LABEL: <return_42>:
// HOTPATCH-NEXT: 0:
// HOTPATCH-NEXT: ret

#[no_mangle]
#[inline(never)]
pub fn return_42() -> i32 {
    42
}
