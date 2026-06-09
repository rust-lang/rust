// This test case makes sure that two identical invocations of the compiler
// (i.e., same code base, same compile-flags, same compiler-versions, etc.)
// produce the same output. In the past, symbol names of monomorphized functions
// were not deterministic (which we want to avoid).
//
// The test tries to exercise as many different paths into symbol name
// generation as possible:
//
// - regular functions
// - generic functions
// - methods
// - statics
// - closures
// - enum variant constructors
// - tuple struct constructors
// - drop glue
// - FnOnce adapters
// - Trait object shims
// - Fn Pointer shims

// ignore-tidy-linelength

#![allow(dead_code, warnings)]

extern crate reproducible_build_aux;

static STATIC: i32 = 1234;

pub struct Struct<T1, T2> {
    x: T1,
    y: T2,
}

fn regular_fn(_: i32) {}

fn generic_fn<T1, T2>() {}

impl<T1, T2> Drop for Struct<T1, T2> {
    fn drop(&mut self) {}
}

pub enum Enum {
    Variant1,
    Variant2(u32),
    Variant3 { x: u32 },
}

struct TupleStruct(i8, i16, i32, i64);

impl TupleStruct {
    pub fn bar(&self) {}
}

trait Trait<T1, T2> {
    fn foo(&self);
}

impl Trait<i32, u64> for u64 {
    fn foo(&self) {}
}

impl reproducible_build_aux::Trait<char, String> for TupleStruct {
    fn foo(&self) {}
}

fn main() {
    regular_fn(STATIC);
    generic_fn::<u32, char>();
    generic_fn::<char, Struct<u32, u64>>();
    generic_fn::<Struct<u64, u32>, reproducible_build_aux::Struct<u32, u64>>();

    let dropped = Struct { x: "", y: 'a' };

    let _ = Enum::Variant1;
    let _ = Enum::Variant2(0);
    let _ = Enum::Variant3 { x: 0 };
    let _ = TupleStruct(1, 2, 3, 4);

    let closure = |x| x + 1i32;

    fn inner<F: Fn(i32) -> i32>(f: F) -> i32 {
        f(STATIC)
    }

    println!("{}", inner(closure));

    let object_shim: &Trait<i32, u64> = &0u64;
    object_shim.foo();

    fn with_fn_once_adapter<F: FnOnce(i32)>(f: F) {
        f(0);
    }

    with_fn_once_adapter(|_: i32| {});

    reproducible_build_aux::regular_fn(STATIC);
    reproducible_build_aux::generic_fn::<u32, char>();
    reproducible_build_aux::generic_fn::<char, Struct<u32, u64>>();
    reproducible_build_aux::generic_fn::<Struct<u64, u32>, reproducible_build_aux::Struct<u32, u64>>(
    );

    let _ = reproducible_build_aux::Enum::Variant1;
    let _ = reproducible_build_aux::Enum::Variant2(0);
    let _ = reproducible_build_aux::Enum::Variant3 { x: 0 };
    let _ = reproducible_build_aux::TupleStruct(1, 2, 3, 4);

    let object_shim: &reproducible_build_aux::Trait<char, String> = &TupleStruct(0, 1, 2, 3);
    object_shim.foo();

    let pointer_shim: &Fn(i32) = &regular_fn;

    TupleStruct(1, 2, 3, 4).bar();
}
