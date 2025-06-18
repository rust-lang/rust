//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

pub trait A {}
pub trait B: A {}

pub trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

pub fn foo<'a>(x: &'a <dyn B + 'static as Mirror>::Assoc) -> &'a (dyn A + 'static) {
    x
}

fn main() {}
