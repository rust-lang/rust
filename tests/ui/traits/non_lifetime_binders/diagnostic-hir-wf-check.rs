// Make sure not to construct predicates with escaping bound vars in `diagnostic_hir_wf_check`.
// Regression test for <https://github.com/rust-lang/rust/issues/139330>.

#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

trait A<T: ?Sized> {}
impl<T: ?Sized> A<T> for () {}

trait B {}
struct W<T: B>(T);

fn b() -> (W<()>, impl for<C> A<C>) { (W(()), ()) }
//~^ ERROR the trait bound `(): B` is not satisfied
//~| ERROR the trait bound `(): B` is not satisfied
//~| ERROR the trait bound `(): B` is not satisfied

fn main() {}
