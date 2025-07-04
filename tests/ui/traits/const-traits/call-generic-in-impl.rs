//@ check-pass
#![feature(const_trait_impl)]

#[const_trait]
trait MyPartialEq {
    fn eq(&self, other: &Self) -> bool;
}

impl<T: [const] PartialEq> const MyPartialEq for T {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(self, other)
    }
}

fn main() {}
