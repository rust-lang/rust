//@ run-pass
#![allow(unused_braces)]
#![allow(dead_code)]
#![allow(unused_unsafe)]

struct Foo {
    a: usize,
    b: *const ()
}

unsafe impl Sync for Foo {}

fn foo<T>(a: T) -> T {
    a
}

static BLOCK_INTEGRAL: usize = { 1 };
static BLOCK_EXPLICIT_UNIT: () = { () };
static BLOCK_IMPLICIT_UNIT: () = { };
static BLOCK_FLOAT: f64 = { 1.0 };
static BLOCK_ENUM: Option<usize> = { Some(100) };
static BLOCK_STRUCT: Foo = { Foo { a: 12, b: std::ptr::null::<()>() } };
static BLOCK_UNSAFE: usize = unsafe { 1000 };

static BLOCK_FN_INFERRED: fn(usize) -> usize = { foo };

static BLOCK_FN: fn(usize) -> usize = { foo::<usize> };

static BLOCK_ENUM_CONSTRUCTOR: fn(usize) -> Option<usize> = { Some };

pub fn main() {
    assert_eq!(BLOCK_INTEGRAL, 1);
    assert_eq!(BLOCK_EXPLICIT_UNIT, ());
    assert_eq!(BLOCK_IMPLICIT_UNIT, ());
    assert_eq!(BLOCK_FLOAT, 1.0_f64);
    assert_eq!(BLOCK_STRUCT.a, 12);
    assert_eq!(BLOCK_STRUCT.b, std::ptr::null::<()>());
    assert_eq!(BLOCK_ENUM, Some(100));
    assert_eq!(BLOCK_UNSAFE, 1000);
    assert_eq!(BLOCK_FN_INFERRED(300), 300);
    assert_eq!(BLOCK_FN(300), 300);
    assert_eq!(BLOCK_ENUM_CONSTRUCTOR(200), Some(200));
}
