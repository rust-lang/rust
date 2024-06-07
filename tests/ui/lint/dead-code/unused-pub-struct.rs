#![deny(dead_code)]

pub struct NotLint1(());
pub struct NotLint2(std::marker::PhantomData<i32>);

pub struct NeverConstructed(i32); //~ ERROR struct `NeverConstructed` is never constructed

impl NeverConstructed {
    pub fn not_construct_self(&self) {}
}

impl Clone for NeverConstructed {
    fn clone(&self) -> NeverConstructed {
        NeverConstructed(0)
    }
}

pub trait Trait {
    fn not_construct_self(&self);
}

impl Trait for NeverConstructed {
    fn not_construct_self(&self) {
        self.0;
    }
}

pub struct Constructed(i32);

impl Constructed {
    pub fn construct_self() -> Self {
        Constructed(0)
    }
}

impl Clone for Constructed {
    fn clone(&self) -> Constructed {
        Constructed(0)
    }
}

impl Trait for Constructed {
    fn not_construct_self(&self) {
        self.0;
    }
}

fn main() {}
