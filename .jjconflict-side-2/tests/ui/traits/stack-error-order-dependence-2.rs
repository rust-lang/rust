//@ check-pass
// Regression test for <https://github.com/rust-lang/rust/issues/123303>.
// This time EXCEPT without `dyn` builtin bounds :^)

pub trait Trait: Supertrait {}

trait Impossible {}
impl<F: ?Sized + Impossible> Trait for F {}

pub trait Supertrait {}

impl<T: ?Sized + Trait + Impossible> Supertrait for T {}

fn needs_supertrait<T: ?Sized + Supertrait>() {}
fn needs_trait<T: ?Sized + Trait>() {}

struct A;
impl Trait for A where A: Supertrait {}
impl Supertrait for A {}

fn main() {
    needs_supertrait::<A>();
    needs_trait::<A>();
}
