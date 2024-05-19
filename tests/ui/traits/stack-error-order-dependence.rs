//@ check-pass
// Regression test for <https://github.com/rust-lang/rust/issues/123303>.

pub trait Trait: Supertrait {}

trait Impossible {}
impl<F: ?Sized + Impossible> Trait for F {}

pub trait Supertrait {}

impl<T: ?Sized + Trait + Impossible> Supertrait for T {}

fn needs_supertrait<T: ?Sized + Supertrait>() {}
fn needs_trait<T: ?Sized + Trait>() {}

fn main() {
    needs_supertrait::<dyn Trait>();
    needs_trait::<dyn Trait>();
}
