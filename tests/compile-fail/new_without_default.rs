#![feature(plugin, const_fn)]
#![plugin(clippy)]

#![allow(dead_code)]
#![deny(new_without_default, new_without_default_derive)]

pub struct Foo;

impl Foo {
    pub fn new() -> Foo { Foo }
    //~^ERROR: you should consider deriving a `Default` implementation for `Foo`
    //~|HELP try this
    //~^^^SUGGESTION #[derive(Default)]
    //~^^^SUGGESTION pub fn new
}

pub struct Bar;

impl Bar {
    pub fn new() -> Self { Bar }
    //~^ERROR: you should consider deriving a `Default` implementation for `Bar`
    //~|HELP try this
    //~^^^SUGGESTION #[derive(Default)]
    //~^^^SUGGESTION pub fn new
}

pub struct Ok;

impl Ok {
    pub fn new() -> Self { Ok }
}

impl Default for Ok {
    fn default() -> Self { Ok }
}

pub struct Params;

impl Params {
    pub fn new(_: u32) -> Self { Params }
}

pub struct GenericsOk<T> {
    bar: T,
}

impl<U> Default for GenericsOk<U> {
    fn default() -> Self { unimplemented!(); }
}

impl<'c, V> GenericsOk<V> {
    pub fn new() -> GenericsOk<V> { unimplemented!() }
}

pub struct LtOk<'a> {
    foo: &'a bool,
}

impl<'b> Default for LtOk<'b> {
    fn default() -> Self { unimplemented!(); }
}

impl<'c> LtOk<'c> {
    pub fn new() -> LtOk<'c> { unimplemented!() }
}

pub struct LtKo<'a> {
    foo: &'a bool,
}

impl<'c> LtKo<'c> {
    pub fn new() -> LtKo<'c> { unimplemented!() }
    //~^ERROR: you should consider adding a `Default` implementation for
    //~^^HELP try
    //~^^^SUGGESTION impl Default for LtKo<'c> {
    //~^^^SUGGESTION     fn default() -> Self {
    //~^^^SUGGESTION         Self::new()
    //~^^^SUGGESTION     }
    //~^^^SUGGESTION }
    // FIXME: that suggestion is missing lifetimes
}

struct Private;

impl Private {
    fn new() -> Private { unimplemented!() } // We don't lint private items
}

struct Const;

impl Const {
    pub const fn new() -> Const { Const } // const fns can't be implemented via Default
}
fn main() {}
