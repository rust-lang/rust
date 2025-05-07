// https://github.com/rust-lang/rust/issues/130591
#![deny(rustdoc::broken_intra_doc_links)]
#![crate_name = "foo"]

/// [value@Foo::X] //~ERROR broken
pub enum Foo {
    X,
}

/// [tst][value@MyStruct] //~ERROR broken
pub struct MyStruct;

pub enum MyEnum {
    Internals,
}

pub use MyEnum::*;

/// In this context, [a][type@Internals] is a struct,
/// while [b][value@Internals] fails. //~ERROR broken
/// Also, [c][struct@Internals] is a struct,
/// while [d][variant@Internals] fails. //~ERROR broken
pub struct Internals {
    foo: (),
}

pub mod inside {
    pub struct Internals2;
}

use inside::*;

/// In this context, [a][type@Internals2] is an enum,
/// while [b][value@Internals2] fails. //~ERROR broken
pub enum Internals2 {}
