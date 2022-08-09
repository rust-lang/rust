// check-pass
#![feature(const_trait_impl)]

#[const_trait]
pub trait Index {
    type Output;
}

#[const_trait]
pub trait IndexMut where Self: Index {
    fn foo(&mut self) -> <Self as Index>::Output;
}

fn main() {}
