//! Check that when the feature `import_trait_associated_functions` is enabled,
//! and one trys to import inherent associated items, the error message is
//! updated to reflect that only trait associated items can be imported.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/148009>.

//@ check-fail

#![feature(import_trait_associated_functions, extern_types)]

pub struct TestStruct;

impl TestStruct {
    pub fn m1() {}
    pub const C1: usize = 0;
}

pub use self::TestStruct::{C1, m1};
//~^ ERROR unresolved import `self::TestStruct` [E0432]
//~| NOTE `TestStruct` is a struct, not a module or a trait
//~| NOTE cannot import inherent associated items, only trait associated items

pub union TestUnion {
    pub f: f32,
    pub i: i32,
}

impl TestUnion {
    pub fn m2() {}
    pub const C2: usize = 0;
}

pub use self::TestUnion::{C2, m2};
//~^ ERROR unresolved import `self::TestUnion` [E0432]
//~| NOTE `TestUnion` is a union, not a module or a trait
//~| NOTE cannot import inherent associated items, only trait associated items

pub enum TestEnum {
    V1,
    V2,
}

impl TestEnum {
    pub fn m3() {}
    pub const C3: usize = 0;
}

pub use self::TestEnum::{C3, m3};
//~^ ERROR unresolved imports `self::TestEnum::C3`, `self::TestEnum::m3` [E0432]
//~| NOTE no `m3` in `TestEnum`
//~| NOTE no `C3` in `TestEnum`
//~| NOTE cannot import inherent associated items, only trait associated items

extern "C" {
    pub type TestForeignTy;
}

impl TestForeignTy {
    pub fn m4() {}
    pub const C4: usize = 0;
}

pub use self::TestForeignTy::{C4, m4};
//~^ ERROR unresolved import `self::TestForeignTy` [E0432]
//~| NOTE `TestForeignTy` is a foreign type, not a module or a trait
//~| NOTE cannot import inherent associated items, only trait associated items

fn main() {}
