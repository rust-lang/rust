//@ needs-sanitizer-kcfi
//@ no-prefer-dynamic
//@ compile-flags: -Zsanitizer=kcfi -Cpanic=abort -Cunsafe-allow-abi-mismatch=sanitizer
//@ ignore-backends: gcc
//@ run-pass

#![feature(c_variadic)]

trait Trait {
    unsafe extern "C" fn foo(x: i32, y: i32, mut ap: ...) -> i32 {
        x + y + ap.arg::<i32>() + ap.arg::<i32>()
    }
}

impl Trait for i32 {}

fn main() {
    let f = i32::foo as unsafe extern "C" fn(i32, i32, ...) -> i32;
    assert_eq!(unsafe { f(1, 2, 3, 4) }, 1 + 2 + 3 + 4);
}
