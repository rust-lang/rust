//@ compile-flags: -Zassumptions-on-binders

//! Nested candidate replay during unsizing can fail to instantiate (e.g. when a
//! canonical response would leak a placeholder universe). That failure must
//! reject the coercion: `CoerceVisitor` maps it to `ControlFlow::Break`, because
//! `VisitorResult::output()` is `Continue` and would otherwise treat the failed
//! replay as a successful unsizing walk.

#![feature(sized_hierarchy)]
#![feature(non_lifetime_binders)]
#![allow(incomplete_features)]

use std::fmt::Debug;
use std::marker::PointeeSized;

trait Trait<T: PointeeSized> {}

impl<T: PointeeSized> Trait<T> for i32 {}

fn produce() -> impl for<T> Trait<T> {
    16
}

fn main() {
    let x = produce();
    let _: &dyn Debug = &x;
    //~^ ERROR the trait bound `&impl Trait<T>: CoerceUnsized<&dyn Debug>` is not satisfied
}
