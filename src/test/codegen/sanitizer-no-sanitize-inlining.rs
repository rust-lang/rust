// Verifies that no_sanitize attribute prevents inlining when
// given sanitizer is enabled, but has no effect on inlining otherwise.
//
// needs-sanitizer-support
// only-x86_64
//
// revisions: ASAN LSAN
//
//[ASAN] compile-flags: -Zsanitizer=address -C opt-level=3 -Z mir-opt-level=3
//[LSAN] compile-flags: -Zsanitizer=leak    -C opt-level=3 -Z mir-opt-level=3

#![crate_type="lib"]
#![feature(no_sanitize)]

// ASAN-LABEL: define void @test
// ASAN:         call {{.*}} @random_inline
// ASAN:       }
//
// LSAN-LABEL: define void @test
// LSAN-NO:      call
// LSAN:       }
#[no_mangle]
pub fn test(n: &mut u32) {
    random_inline(n);
}

#[no_sanitize(address)]
#[inline]
#[no_mangle]
pub fn random_inline(n: &mut u32) {
    *n = 42;
}
