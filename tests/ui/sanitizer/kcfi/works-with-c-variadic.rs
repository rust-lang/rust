// Verifies that C variadic trait methods can be called through function
// pointers.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

trait Trait1 {
    // <Type1 as Trait1>::foo is transformed into <dyn Trait1 as Trait1>::foo
    unsafe extern "C" fn foo(x: i32, y: i32, mut ap: ...) -> i32 {
        x + y + ap.next_arg::<i32>() + ap.next_arg::<i32>()
    }
}

struct Type1;

impl Trait1 for Type1 {}

fn main() {
    let f = std::hint::black_box(Type1::foo as unsafe extern "C" fn(i32, i32, ...) -> i32);
    // The indirect call is not transformed, as the type id is encoded from the fn pointer type
    assert_eq!(unsafe { f(1, 2, 3, 4) }, 1 + 2 + 3 + 4);
}
