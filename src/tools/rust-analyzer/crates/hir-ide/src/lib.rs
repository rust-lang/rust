//! The final bits of analysis; Display, extra diagnostics, and tests.

#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]
// It's useful to refer to code that is private in doc comments.
#![allow(rustdoc::private_intra_doc_links)]

extern crate ra_ap_rustc_abi as rustc_abi;
extern crate ra_ap_rustc_ast_ir as rustc_ast_ir;
extern crate ra_ap_rustc_pattern_analysis as rustc_pattern_analysis;
extern crate ra_ap_rustc_type_ir as rustc_type_ir;

pub mod diagnostics;
pub mod display;
mod impl_db_macro;
pub mod mir_pretty;

#[doc(hidden)]
pub mod __private {
    pub use hir_def;
    pub use salsa;
}

#[cfg(test)]
mod test_db;
#[cfg(test)]
mod tests;

use hir_def::ModuleId;
use hir_ty::{db::HirDatabase, next_solver::Const};
use syntax::ast::{ConstArg, make};

pub use hir_ty::*;

use crate::display::HirDisplay;

pub fn known_const_to_ast<'db>(
    konst: Const<'db>,
    db: &'db dyn HirDatabase,
    target_module: ModuleId,
) -> Option<ConstArg> {
    Some(make::expr_const_value(
        &konst.display_source_code(db, target_module, true).unwrap_or_else(|_| "_".to_owned()),
    ))
}
