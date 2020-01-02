//! HIR datatypes. See the [rustc guide] for more info.
//!
//! [rustc guide]: https://rust-lang.github.io/rustc-guide/hir.html

pub mod check_attr;
pub mod def;
pub use rustc_hir::def_id;
pub use rustc_hir::hir_id::*;
pub mod intravisit;
pub mod itemlikevisit;
pub mod map;
pub mod pat_util;
pub mod print;
pub mod upvars;

mod hir;
pub use hir::BlockCheckMode::*;
pub use hir::FunctionRetTy::*;
pub use hir::PrimTy::*;
pub use hir::UnOp::*;
pub use hir::UnsafeSource::*;
pub use hir::*;

use crate::ty::query::Providers;

pub fn provide(providers: &mut Providers<'_>) {
    check_attr::provide(providers);
    map::provide(providers);
    upvars::provide(providers);
}
