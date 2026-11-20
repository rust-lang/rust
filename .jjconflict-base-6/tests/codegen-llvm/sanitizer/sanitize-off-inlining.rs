// Verifies that sanitize(xyz = "off") attribute prevents inlining when
// given sanitizer is enabled, but has no effect on inlining otherwise.
//@ needs-sanitizer-address
//@ needs-sanitizer-leak
//@ revisions: ASAN LSAN
//@       compile-flags: -C unsafe-allow-abi-mismatch=sanitizer
//@       compile-flags: -Copt-level=3 -Zmir-opt-level=4 -Ctarget-feature=-crt-static
//@[ASAN] compile-flags: -Zsanitizer=address
//@[LSAN] compile-flags: -Zsanitizer=leak

#![crate_type = "lib"]
#![feature(sanitize)]

// ASAN-LABEL: define void @test
// ASAN:         call {{.*}} @random_inline
// ASAN:       }
//
// LSAN-LABEL: define void @test
// LSAN-NOT:     call
// LSAN:       }
#[no_mangle]
pub fn test(n: &mut u32) {
    random_inline(n);
}

#[sanitize(address = "off")]
#[inline]
#[no_mangle]
pub fn random_inline(n: &mut u32) {
    *n = 42;
}
