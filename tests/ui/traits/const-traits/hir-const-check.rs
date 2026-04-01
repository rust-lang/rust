//@ check-pass
//@ compile-flags: -Znext-solver

// Regression test for #69615.

#![feature(const_trait_impl)]
#![feature(const_try)]

pub const trait MyTrait {
    fn method(&self) -> Option<()>;
}

impl const MyTrait for () {
    fn method(&self) -> Option<()> {
        Some(())?;
        None
    }
}

fn main() {}
