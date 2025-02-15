#![crate_type = "lib"]
#![warn(clippy::return_self_not_must_use)]

#[derive(Clone)]
pub struct Bar;

pub trait Whatever {
    fn what(&self) -> Self;
    //~^ return_self_not_must_use

    // There should be no warning here! (returns a reference)
    fn what2(&self) -> &Self;
}

impl Bar {
    // There should be no warning here! (note taking a self argument)
    pub fn not_new() -> Self {
        Self
    }
    pub fn foo(&self) -> Self {
        //~^ return_self_not_must_use

        Self
    }
    pub fn bar(self) -> Self {
        //~^ return_self_not_must_use

        self
    }
    // There should be no warning here! (private method)
    fn foo2(&self) -> Self {
        Self
    }
    // There should be no warning here! (returns a reference)
    pub fn foo3(&self) -> &Self {
        self
    }
    // There should be no warning here! (already a `must_use` attribute)
    #[must_use]
    pub fn foo4(&self) -> Self {
        Self
    }
}

impl Whatever for Bar {
    // There should be no warning here! (comes from the trait)
    fn what(&self) -> Self {
        self.foo2()
    }
    // There should be no warning here! (comes from the trait)
    fn what2(&self) -> &Self {
        self
    }
}

#[must_use]
pub struct Foo;

impl Foo {
    // There should be no warning here! (`Foo` already implements `#[must_use]`)
    fn foo(&self) -> Self {
        Self
    }
}
