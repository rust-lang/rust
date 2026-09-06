// Verifies that the addresses of function pointers can be compared (i.e.,
// through the compiler-generated FnPtr implementations for them).
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

fn foo1() {}

fn foo2() {}

fn main() {
    let f: fn() = std::hint::black_box(foo1);
    let g: fn() = std::hint::black_box(foo1);
    let h: fn() = std::hint::black_box(foo2);
    // The <fn() as FnPtr>::addr FnPtrAddrShims are not transformed, as the FnPtr trait is not
    // dyn compatible.
    assert!(std::ptr::fn_addr_eq(f, g));
    assert!(!std::ptr::fn_addr_eq(f, h));
}
