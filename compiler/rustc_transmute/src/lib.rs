//! Checks whether one type's representation can be reinterpreted as another.
//!
//! The analysis converts compiler layouts into trees of bytes, references, and
//! definition markers. It prunes destination paths that may carry safety invariants unless
//! safety is assumed, then converts the trees into deterministic finite automata.
//! Comparing the automata produces an `Answer`; reference transitions can leave
//! `Condition`s for the trait solver to discharge.

// tidy-alphabetical-start
#![cfg_attr(test, feature(test))]
#![feature(option_into_flat_iter)]
// tidy-alphabetical-end

pub(crate) use rustc_data_structures::fx::{FxIndexMap as Map, FxIndexSet as Set};

pub mod layout;
mod maybe_transmutable;

/// Proof obligations supplied by the caller rather than checked by the analysis.
///
/// This mirrors `core::mem::Assume`. A `true` field transfers the corresponding
/// obligation to the caller; the default leaves all four obligations to the compiler.
#[derive(Copy, Clone, Debug, Default)]
pub struct Assume {
    pub alignment: bool,
    pub lifetimes: bool,
    pub safety: bool,
    pub validity: bool,
}

#[cfg(feature = "rustc")]
mod rustc {
    use rustc_attr_ir::lang_items::LangItem;
    use rustc_middle::ty::consts::ConstExt;
    use rustc_middle::ty::transmute::Answer;
    use rustc_middle::ty::{Const, Region, Ty, TyCtxt};

    use super::*;

    pub struct TransmuteTypeEnv<'tcx> {
        tcx: TyCtxt<'tcx>,
    }

    impl<'tcx> TransmuteTypeEnv<'tcx> {
        pub fn new(tcx: TyCtxt<'tcx>) -> Self {
            Self { tcx }
        }

        pub fn is_transmutable(
            &mut self,
            src: Ty<'tcx>,
            dst: Ty<'tcx>,
            assume: crate::Assume,
        ) -> Answer<Region<'tcx>, Ty<'tcx>> {
            crate::maybe_transmutable::MaybeTransmutableQuery::new(src, dst, assume, self.tcx)
                .answer()
        }
    }

    impl Assume {
        /// Constructs an `Assume` from a given const-`Assume`.
        pub fn from_const<'tcx>(tcx: TyCtxt<'tcx>, ct: Const<'tcx>) -> Option<Self> {
            use rustc_middle::ty::ScalarInt;
            use rustc_span::sym;

            let cv = ct.try_to_value()?;
            let adt_def = cv.ty.ty_adt_def()?;

            if !tcx.is_lang_item(adt_def.did(), LangItem::TransmuteOpts) {
                tcx.dcx().delayed_bug(format!(
                    "The given `const` was not marked with the `{}` lang item.",
                    LangItem::TransmuteOpts.name()
                ));
                return Some(Self {
                    alignment: true,
                    lifetimes: true,
                    safety: true,
                    validity: true,
                });
            }

            let variant = adt_def.non_enum_variant();
            let fields = cv.to_branch();

            let get_field = |name| {
                let (field_idx, _) = variant
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, field_def)| name == field_def.name)
                    .unwrap_or_else(|| panic!("There were no fields named `{name}`."));
                fields[field_idx].try_to_leaf().map(|leaf| leaf == ScalarInt::TRUE)
            };

            Some(Self {
                alignment: get_field(sym::alignment)?,
                lifetimes: get_field(sym::lifetimes)?,
                safety: get_field(sym::safety)?,
                validity: get_field(sym::validity)?,
            })
        }
    }
}

#[cfg(feature = "rustc")]
pub use rustc::*;
