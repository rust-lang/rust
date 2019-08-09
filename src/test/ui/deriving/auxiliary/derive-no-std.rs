// no-prefer-dynamic

#![crate_type = "rlib"]
#![no_std]

// Issue #16803

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord,
         Debug, Default, Copy)]
pub struct Foo {
    pub x: u32,
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord,
         Debug, Copy)]
pub enum Bar {
    Qux,
    Quux(u32),
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord,
         Debug, Copy)]
pub enum Void {}
#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord,
         Debug, Copy)]
pub struct Empty;
#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord,
         Debug, Copy)]
pub struct AlsoEmpty {}
