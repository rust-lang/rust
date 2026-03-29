//@ compile-flags: -Znext-solver=globally

#![allow(incomplete_features)]
#![feature(non_lifetime_binders)]

fn auto_trait()
where
    for<T> T: PartialEq + PartialOrd,
{}

fn main() {
    auto_trait();
    //~^ ERROR can't compare `T` with `T`
}
