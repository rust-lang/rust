#![crate_type = "lib"]

#[derive(Clone)]
pub struct Bar;

pub trait Whatever {
    fn what(&self) -> Self;
    // There should be no warning here!
    fn what2(&self) -> &Self;
}

impl Bar {
    // There should be no warning here!
    pub fn not_new() -> Self {
        Self
    }
    pub fn foo(&self) -> Self {
        Self
    }
    pub fn bar(self) -> Self {
        self
    }
    // There should be no warning here!
    fn foo2(&self) -> Self {
        Self
    }
    // There should be no warning here!
    pub fn foo3(&self) -> &Self {
        self
    }
}

impl Whatever for Bar {
    // There should be no warning here!
    fn what(&self) -> Self {
        self.foo2()
    }
    // There should be no warning here!
    fn what2(&self) -> &Self {
        self
    }
}
