// Verifies that functions with repr(transparent) types (including
// self-referential repr(transparent) types) as argument types can be called
// through function pointers and trait objects.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

use std::marker::PhantomData;

#[repr(transparent)]
struct Type1(i32);

#[repr(transparent)]
struct Type2<'a>(&'a i32);

trait Trait1 {}

impl Trait1 for Type1 {}

// A repr(transparent) type with a represented type that has regions
#[repr(transparent)]
struct Type3(Box<dyn Trait1 + 'static>);

// A repr(transparent) type without a non-ZST field
#[repr(transparent)]
struct Type4(PhantomData<i32>);

struct Struct1<T> {
    _x: u8,
    p: PhantomData<T>,
}

#[repr(transparent)]
struct Type5(Struct1<Type5>);

trait Trait2 {
    fn foo(&self, x: Type5) -> i32;
}

struct Type6;

impl Trait2 for Type6 {
    fn foo(&self, _: Type5) -> i32 {
        6
    }
}

// A repr(transparent) type that is a pointer and references itself, which is generalized to avoid
// a reference cycle.
#[repr(transparent)]
struct Type7(*const Type7);

fn foo1(x: Type1) -> i32 {
    assert_eq!(x.0, 1);
    1
}

fn foo2(x: Type2<'_>) -> i32 {
    assert_eq!(*x.0, 2);
    2
}

fn foo3(_: Type3) -> i32 {
    3
}

fn foo4(_: Type4) -> i32 {
    4
}

fn foo5(_: Type7) -> i32 {
    5
}

fn main() {
    let f: fn(Type1) -> i32 = std::hint::black_box(foo1);
    assert_eq!(f(Type1(1)), 1);
    let f: fn(Type2<'_>) -> i32 = std::hint::black_box(foo2);
    assert_eq!(f(Type2(&2)), 2);
    let f: fn(Type3) -> i32 = std::hint::black_box(foo3);
    assert_eq!(f(Type3(Box::new(Type1(1)))), 3);
    let f: fn(Type4) -> i32 = std::hint::black_box(foo4);
    assert_eq!(f(Type4(PhantomData)), 4);
    let f: fn(Type7) -> i32 = std::hint::black_box(foo5);
    assert_eq!(f(Type7(std::ptr::null())), 5);
    let x = &Type6 as &dyn Trait2;
    assert_eq!(x.foo(Type5(Struct1 { _x: 0, p: PhantomData })), 6);
}
