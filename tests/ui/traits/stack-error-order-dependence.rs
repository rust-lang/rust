//@ check-pass
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

// Regression test for <https://github.com/rust-lang/rust/issues/123303>.

pub trait Trait: Supertrait {}

trait Impossible {}
impl<F: Impossible> Trait for F {}

pub trait Supertrait {}

impl<T: Trait + Impossible> Supertrait for T {}

fn needs_supertrait<T: Supertrait>() {}
fn needs_trait<T: Trait>() {}

fn main() {
    needs_supertrait::<dyn Trait>();
    needs_trait::<dyn Trait>();
}
