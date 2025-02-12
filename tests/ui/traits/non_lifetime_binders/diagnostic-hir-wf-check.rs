// Make sure not to construct predicates with escaping bound vars in `diagnostic_hir_wf_check`.
// Regression test for <https://github.com/rust-lang/rust/issues/139330>.

#![feature(sized_hierarchy)]
#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

use std::marker::PointeeSized;

trait A<T: PointeeSized> {}
impl<T: PointeeSized> A<T> for () {}

trait B {}
struct W<T: B>(T);

fn b() -> (W<()>, impl for<C> A<C>) { (W(()), ()) }
//~^ ERROR the trait bound `(): B` is not satisfied
//~| ERROR the trait bound `(): B` is not satisfied
//~| ERROR the trait bound `(): B` is not satisfied

fn main() {}
