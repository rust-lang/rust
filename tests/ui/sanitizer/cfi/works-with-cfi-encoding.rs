// Verifies that user-defined CFI encodings can be used.
//
//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Ccodegen-units=1 -Clto -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=cfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

#![feature(cfi_encoding, extern_types)]

#[cfi_encoding = "3Foo"]
struct Type1(i32);

unsafe extern "C" {
    #[cfi_encoding = "3Bar"]
    type Type2;
}

// Type3 is not transformed, as it has an user-defined CFI encoding
#[cfi_encoding = "3Baz"]
#[repr(transparent)]
struct Type3(i32);

fn foo1(x: Type1) -> i32 {
    assert_eq!(x.0, 1);
    1
}

fn foo2(_: *const Type2) -> i32 {
    2
}

fn foo3(x: Type3) -> i32 {
    assert_eq!(x.0, 3);
    3
}

fn main() {
    let f: fn(Type1) -> i32 = std::hint::black_box(foo1);
    assert_eq!(f(Type1(1)), 1);
    let f: fn(*const Type2) -> i32 = std::hint::black_box(foo2);
    assert_eq!(f(&() as *const () as *const Type2), 2);
    let f: fn(Type3) -> i32 = std::hint::black_box(foo3);
    assert_eq!(f(Type3(3)), 3);
}
