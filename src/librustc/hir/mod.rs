//! HIR datatypes. See the [rustc guide] for more info.
//!
//! [rustc guide]: https://rust-lang.github.io/rustc-guide/hir.html

pub mod check_attr;
pub mod exports;
pub mod map;

use crate::ty::query::Providers;

pub fn provide(providers: &mut Providers<'_>) {
    map::provide(providers);
}
