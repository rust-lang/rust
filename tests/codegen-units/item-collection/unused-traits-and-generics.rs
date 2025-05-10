//@ compile-flags:-Clink-dead-code

#![crate_type = "lib"]
#![deny(dead_code)]

// This test asserts that no codegen items are generated for generic items that
// are never instantiated in the local crate.

pub trait Trait {
    fn foo() {}
    fn bar(&self) {}
}

pub fn foo<T: Copy>(x: T) -> (T, T) {
    (x, x)
}

pub struct Struct<T> {
    x: T,
}

impl<T> Struct<T> {
    pub fn foo(self) -> T {
        self.x
    }

    pub fn bar() {}
}

pub enum Enum<T> {
    A(T),
    B { x: T },
}

impl<T> Enum<T> {
    pub fn foo(self) -> T {
        match self {
            Enum::A(x) => x,
            Enum::B { x } => x,
        }
    }

    pub fn bar() {}
}

pub struct TupleStruct<T>(T);

impl<T> TupleStruct<T> {
    pub fn foo(self) -> T {
        self.0
    }

    pub fn bar() {}
}

pub type Pair<T> = (T, T);

pub struct NonGeneric {
    x: i32,
}

impl NonGeneric {
    pub fn foo(self) -> i32 {
        self.x
    }

    pub fn generic_foo<T>(&self, x: T) -> (T, i32) {
        (x, self.x)
    }

    pub fn generic_bar<T: Copy>(x: T) -> (T, T) {
        (x, x)
    }
}

// Only the non-generic methods should be instantiated:
//~ MONO_ITEM fn NonGeneric::foo
