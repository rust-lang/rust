#![crate_type="lib"]

// These items are for testing that associated consts work cross-crate.
pub trait Foo {
    const BAR: usize;
}

pub struct FooNoDefault;

impl Foo for FooNoDefault {
    const BAR: usize = 0;
}

// These test that defaults and default resolution work cross-crate.
pub trait FooDefault {
    const BAR: usize = 1;
}

pub struct FooOverwriteDefault;

impl FooDefault for FooOverwriteDefault {
    const BAR: usize = 2;
}

pub struct FooUseDefault;

impl FooDefault for FooUseDefault {}

// Test inherent impls.
pub struct InherentBar;

impl InherentBar {
    pub const BAR: usize = 3;
}
