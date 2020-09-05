// Checks that only functions with compatible attributes are inlined.
//
// only-x86_64
// needs-sanitizer-address
// compile-flags: -Zsanitizer=address

#![crate_type = "lib"]
#![feature(no_sanitize)]
#![feature(target_feature_11)]

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
#[no_sanitize(address, memory)]
pub unsafe fn no_sanitize() {}
