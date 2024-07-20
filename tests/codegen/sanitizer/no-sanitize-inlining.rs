// Verifies that no_sanitize attribute prevents inlining when
// given sanitizer is enabled, but has no effect on inlining otherwise.
//
//@ needs-sanitizer-address
//@ needs-sanitizer-leak
//@ revisions: asan lsan
//@       compile-flags: -Copt-level=3 -Zmir-opt-level=4 -Ctarget-feature=-crt-static
//@[asan] compile-flags: -Zsanitizer=address
//@[lsan] compile-flags: -Zsanitizer=leak

#![crate_type = "lib"]
#![feature(no_sanitize)]

// CHECK-ASAN-LABEL: define void @test
// CHECK-ASAN:         call {{.*}} @random_inline
// CHECK-ASAN:       }
//
// CHECK-LSAN-LABEL: define void @test
// CHECK-LSAN-NOT:     call
// CHECK-LSAN:       }
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
