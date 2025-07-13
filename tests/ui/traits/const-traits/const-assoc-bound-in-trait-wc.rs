//@ check-pass

#![feature(const_clone)]
#![feature(const_trait_impl)]

#[const_trait]
trait A where Self::Target: [const] Clone {
    type Target;
}

const fn foo<T>() where T: [const] A {}

fn main() {}
