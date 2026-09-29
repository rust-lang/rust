use rustc_macros::extension;
use rustc_type_ir::{self as ir};

use crate::ty::{self, Ty, TyCtxt};

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

#[extension(pub trait ConstExt<'tcx>)]
impl<'tcx> Const<'tcx> {
    /// Creates a constant with the given integer value and interns it.
    #[inline]
    fn from_bits(
        tcx: TyCtxt<'tcx>,
        bits: u128,
        typing_env: ty::TypingEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> Self {
        let size = tcx
            .layout_of(typing_env.as_query_input(ty))
            .unwrap_or_else(|e| panic!("could not compute layout for {ty:?}: {e:?}"))
            .size;
        let valtree =
            ty::ValTree::from_scalar_int(tcx, ScalarInt::try_from_uint(bits, size).unwrap());
        Self::new_value(tcx, valtree, ty)
    }
}
