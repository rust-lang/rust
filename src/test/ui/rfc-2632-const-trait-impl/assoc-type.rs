// ignore-test

// FIXME: This test should fail since, within a const impl of `Foo`, the bound on `Foo::Bar` should
// require a const impl of `Add` for the associated type.

#![allow(incomplete_features)]
#![feature(const_trait_impl)]
#![feature(const_fn)]

struct NonConstAdd(i32);

impl std::ops::Add for NonConstAdd {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        NonConstAdd(self.0 + rhs.0)
    }
}

trait Foo {
    type Bar: std::ops::Add;
}

impl const Foo for NonConstAdd {
    type Bar = NonConstAdd;
}

fn main() {}
