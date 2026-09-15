// Verifies that functions with trait types (i.e., trait objects) as argument
// types can be called through function pointers.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

trait Trait1 {
    fn foo(&self) -> i32;
}

struct Type1;

impl Trait1 for Type1 {
    fn foo(&self) -> i32 {
        1
    }
}

fn foo1(x: &dyn Trait1) -> i32 {
    assert_eq!(x.foo(), 1);
    1
}

fn foo2(x: &mut dyn Trait1) -> i32 {
    assert_eq!(x.foo(), 1);
    2
}

fn foo3(x: Box<dyn Trait1>) -> i32 {
    assert_eq!(x.foo(), 1);
    3
}

// A trait object without a principal trait, so its predicates are all auto traits
fn foo4(_: &dyn Send) -> i32 {
    4
}

fn main() {
    let f: fn(&dyn Trait1) -> i32 = std::hint::black_box(foo1);
    assert_eq!(f(&Type1), 1);
    let f: fn(&mut dyn Trait1) -> i32 = std::hint::black_box(foo2);
    assert_eq!(f(&mut Type1), 2);
    let f: fn(Box<dyn Trait1>) -> i32 = std::hint::black_box(foo3);
    assert_eq!(f(Box::new(Type1)), 3);
    let f: fn(&dyn Send) -> i32 = std::hint::black_box(foo4);
    assert_eq!(f(&Type1), 4);
}
