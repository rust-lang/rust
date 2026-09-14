//! Checks whether one type's representation can be reinterpreted as another.
//!
//! The analysis converts compiler layouts into trees of bytes, references, and
//! definition markers. It prunes destination paths that may carry safety invariants unless
//! safety is assumed, then converts the trees into deterministic finite automata.
//! Comparing the automata produces an [`Answer`]; reference transitions can leave
//! [`Condition`]s for the trait solver to discharge.

// tidy-alphabetical-start
#![cfg_attr(bootstrap, feature(never_type))]
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

/// The result of a transmutability query under the supplied [`Assume`] options.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub enum Answer<R, T> {
    /// The analysis requires no further conditions.
    Yes,
    /// The analysis could not establish transmutability.
    No(Reason<T>),
    /// Transmutability depends on conditions that the trait solver must discharge.
    If(Condition<R, T>),
}

/// A condition which must hold for safe transmutation to be possible.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub enum Condition<R, T> {
    /// `Src` is transmutable into `Dst`, if `src` is transmutable into `dst`.
    Transmutable { src: T, dst: T },

    /// The region `long` must outlive `short`.
    Outlives { long: R, short: R },

    /// The type `ty` must satisfy `Freeze`.
    Immutable { ty: T },

    /// `Src` is transmutable into `Dst`, if all of the enclosed requirements are met.
    IfAll(Vec<Condition<R, T>>),

    /// `Src` is transmutable into `Dst` if any of the enclosed requirements are met.
    IfAny(Vec<Condition<R, T>>),
}

/// Answers "why wasn't the source type transmutable into the destination type?"
#[derive(Debug, Hash, Eq, PartialEq, PartialOrd, Ord, Clone)]
pub enum Reason<T> {
    /// The layout of the source type is not yet supported.
    SrcIsNotYetSupported,
    /// The layout of the destination type is not yet supported.
    DstIsNotYetSupported,
    /// The layout of the destination type is bit-incompatible with the source type.
    DstIsBitIncompatible,
    /// The destination type is uninhabited.
    DstUninhabited,
    /// The destination type may carry safety invariants.
    DstMayHaveSafetyInvariants,
    /// `Dst` is larger than `Src`, and the excess bytes were not exclusively uninitialized.
    DstIsTooBig,
    /// The destination referent is larger than the source referent.
    DstRefIsTooBig {
        /// The referent of the source type.
        src: T,
        /// The size of the source type's referent.
        src_size: usize,
        /// The too-large referent of the destination type.
        dst: T,
        /// The size of the destination type's referent.
        dst_size: usize,
    },
    /// The destination referent requires stricter alignment than the source referent.
    DstHasStricterAlignment { src_min_align: usize, dst_min_align: usize },
    /// Can't go from shared pointer to unique pointer
    DstIsMoreUnique,
    /// Encountered a type error
    TypeError,
    /// The layout of src is unknown
    SrcLayoutUnknown,
    /// The layout of dst is unknown
    DstLayoutUnknown,
    /// The size of src is overflow
    SrcSizeOverflow,
    /// The size of dst is overflow
    DstSizeOverflow,
}

#[cfg(feature = "rustc")]
mod rustc {
    use rustc_hir::attrs::lang_items::LangItem;
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
        ) -> crate::Answer<Region<'tcx>, Ty<'tcx>> {
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
