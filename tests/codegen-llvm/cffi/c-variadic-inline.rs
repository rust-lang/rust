//@ compile-flags: -C opt-level=3
#![feature(c_variadic)]

// Test that the inline attributes are accepted on C-variadic functions.
//
// Currently LLVM is unable to inline C-variadic functions, but that is valid because despite
// the name even `#[inline(always)]` is just a hint.

#[inline(always)]
unsafe extern "C" fn inline_always(mut ap: ...) -> u32 {
    ap.arg::<u32>()
}

#[inline]
unsafe extern "C" fn inline(mut ap: ...) -> u32 {
    ap.arg::<u32>()
}

#[inline(never)]
unsafe extern "C" fn inline_never(mut ap: ...) -> u32 {
    ap.arg::<u32>()
}

#[cold]
unsafe extern "C" fn cold(mut ap: ...) -> u32 {
    ap.arg::<u32>()
}

#[unsafe(no_mangle)]
#[inline(never)]
fn helper() {
    // CHECK-LABEL: helper
    // CHECK-LABEL: call c_variadic_inline::inline_always
    // CHECK-LABEL: call c_variadic_inline::inline
    // CHECK-LABEL: call c_variadic_inline::inline_never
    // CHECK-LABEL: call c_variadic_inline::cold
    unsafe {
        inline_always(1);
        inline(2);
        inline_never(3);
        cold(4);
    }
}

fn main() {
    helper()
}
