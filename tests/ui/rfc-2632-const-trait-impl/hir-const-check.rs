// Regression test for #69615.

#![feature(const_trait_impl)]

#[const_trait]
pub trait MyTrait {
    fn method(&self) -> Option<()>;
}

impl const MyTrait for () {
    fn method(&self) -> Option<()> {
        Some(())?; //~ ERROR `?` is not allowed in a `const fn`
        None
    }
}

fn main() {}
