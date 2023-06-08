// Checks that only functions with compatible attributes are inlined.
//
// only-x86_64
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]
#![feature(no_sanitize)]
#![feature(target_feature_11)]
#![feature(c_variadic)]

// EMIT_MIR inline_compatibility.inlined_target_feature.Inline.diff
#[target_feature(enable = "sse2")]
pub unsafe fn inlined_target_feature() {
    target_feature();
}

// EMIT_MIR inline_compatibility.not_inlined_target_feature.Inline.diff
pub unsafe fn not_inlined_target_feature() {
    target_feature();
}

// EMIT_MIR inline_compatibility.inlined_no_sanitize.Inline.diff
#[no_sanitize(address)]
pub unsafe fn inlined_no_sanitize() {
    no_sanitize();
}

// EMIT_MIR inline_compatibility.not_inlined_no_sanitize.Inline.diff
pub unsafe fn not_inlined_no_sanitize() {
    no_sanitize();
}

#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn target_feature() {}

#[inline]
#[no_sanitize(address)]
pub unsafe fn no_sanitize() {}

// EMIT_MIR inline_compatibility.not_inlined_c_variadic.Inline.diff
pub unsafe fn not_inlined_c_variadic() {
    let s = sum(4u32, 4u32, 30u32, 200u32, 1000u32);
}

#[no_mangle]
#[inline(always)]
unsafe extern "C" fn sum(n: u32, mut vs: ...) -> u32 {
    let mut s = 0;
    let mut i = 0;
    while i != n {
        s += vs.arg::<u32>();
        i += 1;
    }
    s
}
