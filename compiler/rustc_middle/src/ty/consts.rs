use rustc_type_ir::{self as ir};

use crate::ty::TyCtxt;

mod int;
mod kind;
mod lit;
mod valtree;

pub use int::*;
pub use kind::*;
pub use lit::*;
pub use valtree::*;

pub type ConstKind<'tcx> = ir::ConstKind<TyCtxt<'tcx>>;
pub type AliasConst<'tcx> = ir::AliasConst<TyCtxt<'tcx>>;
pub type AliasConstKind<'tcx> = ir::AliasConstKind<TyCtxt<'tcx>>;
pub type Const<'tcx> = ir::Const<TyCtxt<'tcx>>;

#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(ConstKind<'_>, 32);
