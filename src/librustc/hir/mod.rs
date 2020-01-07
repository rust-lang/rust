//! HIR datatypes. See the [rustc guide] for more info.
//!
//! [rustc guide]: https://rust-lang.github.io/rustc-guide/hir.html

pub mod check_attr;
pub mod exports;
pub mod map;
pub mod upvars;

use crate::ty::query::Providers;

pub fn provide(providers: &mut Providers<'_>) {
    check_attr::provide(providers);
    map::provide(providers);
    upvars::provide(providers);
}
