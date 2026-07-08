// Check that trait alias refs correctly induce trait object lifetime defaults.
// issue: <https://github.com/rust-lang/rust/issues/140710>
#![feature(trait_alias)]

//@ check-pass

trait Outer<'a, T: 'a + ?Sized> = Carry<T>;
trait Carry<T: ?Sized> {}

// We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of trait `Outer`.
fn bound<'r, T>() where T: Outer<'r, dyn Inner> {}
fn check_bound<'r, T>() where T: Outer<'r, dyn Inner + 'r> { bound::<'r, T>();}

// We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of trait `Outer`.
fn dyn_trait<'r>(_: Box<dyn Outer<'r, dyn Inner>>) {}
fn check_dyn_trait<'r>(x: Box<dyn Outer<'r, dyn Inner + 'r>>) { dyn_trait(x) }

// We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of trait `Outer`.
fn impl_trait<'r>(_: impl Outer<'r, dyn Inner>) {}
fn check_impl_trait<'r>(x: impl Outer<'r, dyn Inner + 'r>) { impl_trait(x) }

// Reasonably, trait aliases can't be used as the qself in fully qualified paths,
// so we don't need to test them here.

trait Inner {}

fn main() {}
