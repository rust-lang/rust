//@ check-pass
#![feature(const_trait_impl, const_cmp)]

const trait MyPartialEq {
    fn eq(&self, other: &Self) -> bool;
}

const impl<T: [const] PartialEq> MyPartialEq for T {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(self, other)
    }
}

fn main() {}
