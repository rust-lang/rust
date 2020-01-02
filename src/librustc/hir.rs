//! HIR datatypes. See the [rustc guide] for more info.
//!
//! [rustc guide]: https://rust-lang.github.io/rustc-guide/hir.html

pub mod check_attr;
pub use rustc_hir::def;
pub mod exports;
pub use rustc_hir::def_id;
pub use rustc_hir::hir_id::*;
pub mod intravisit;
pub use rustc_hir::itemlikevisit;
pub mod map;
pub use rustc_hir::pat_util;
pub use rustc_hir::print;
pub mod upvars;

pub use rustc_hir::BlockCheckMode::*;
pub use rustc_hir::FunctionRetTy::*;
pub use rustc_hir::PrimTy::*;
pub use rustc_hir::UnOp::*;
pub use rustc_hir::UnsafeSource::*;
pub use rustc_hir::*;

use crate::ty::query::Providers;

pub fn provide(providers: &mut Providers<'_>) {
    check_attr::provide(providers);
    map::provide(providers);
    upvars::provide(providers);
}
