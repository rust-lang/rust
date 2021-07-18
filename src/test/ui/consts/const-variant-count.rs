// run-pass
#![allow(dead_code, enum_intrinsics_non_enums)]
#![feature(variant_count)]
#![feature(never_type)]

use std::mem::variant_count;

enum Void {}

enum Foo {
    A,
    B,
    C,
}

enum Bar {
    A,
    B,
    C,
    D(usize),
    E { field_1: usize, field_2: Foo },
}

struct Baz {
    a: u32,
    b: *const u8,
}

const TEST_VOID: usize = variant_count::<Void>();
const TEST_FOO: usize = variant_count::<Foo>();
const TEST_BAR: usize = variant_count::<Bar>();

const NO_ICE_STRUCT: usize = variant_count::<Baz>();
const NO_ICE_BOOL: usize = variant_count::<bool>();
const NO_ICE_PRIM: usize = variant_count::<*const u8>();

fn main() {
    assert_eq!(TEST_VOID, 0);
    assert_eq!(TEST_FOO, 3);
    assert_eq!(TEST_BAR, 5);
    assert_eq!(variant_count::<Void>(), 0);
    assert_eq!(variant_count::<Foo>(), 3);
    assert_eq!(variant_count::<Bar>(), 5);
    assert_eq!(variant_count::<Option<char>>(), 2);
    assert_eq!(variant_count::<Option<!>>(), 2);
    assert_eq!(variant_count::<Result<!, !>>(), 2);
}
