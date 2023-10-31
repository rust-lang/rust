#![feature(alloc_layout_extra, decl_macro, iterator_try_reduce, never_type)]
#![allow(dead_code, unused_variables)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate tracing;

pub(crate) use rustc_data_structures::fx::{FxIndexMap as Map, FxIndexSet as Set};

pub mod layout;
pub(crate) mod maybe_transmutable;

#[derive(Default)]
pub struct Assume {
    pub alignment: bool,
    pub lifetimes: bool,
    pub safety: bool,
    pub validity: bool,
}

/// Either we have an error, transmutation is allowed, or we have an optional
/// Condition that must hold.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub enum Answer<R> {
    Yes,
    No(Reason),
    If(Condition<R>),
}

/// A condition which must hold for safe transmutation to be possible.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub enum Condition<R> {
    /// `Src` is transmutable into `Dst`, if `src` is transmutable into `dst`.
    IfTransmutable { src: R, dst: R },

    /// `Src` is transmutable into `Dst`, if all of the enclosed requirements are met.
    IfAll(Vec<Condition<R>>),

    /// `Src` is transmutable into `Dst` if any of the enclosed requirements are met.
    IfAny(Vec<Condition<R>>),
}

/// Answers "why wasn't the source type transmutable into the destination type?"
#[derive(Debug, Hash, Eq, PartialEq, PartialOrd, Ord, Clone)]
pub enum Reason {
    /// The layout of the source type is unspecified.
    SrcIsUnspecified,
    /// The layout of the destination type is unspecified.
    DstIsUnspecified,
    /// The layout of the destination type is bit-incompatible with the source type.
    DstIsBitIncompatible,
    /// There aren't any public constructors for `Dst`.
    DstIsPrivate,
    /// `Dst` is larger than `Src`, and the excess bytes were not exclusively uninitialized.
    DstIsTooBig,
    /// Src should have a stricter alignment than Dst, but it does not.
    DstHasStricterAlignment { src_min_align: usize, dst_min_align: usize },
    /// Can't go from shared pointer to unique pointer
    DstIsMoreUnique,
    /// Encountered a type error
    TypeError,
    /// The layout of src is unknown
    SrcLayoutUnknown,
    /// The layout of dst is unknown
    DstLayoutUnknown,
}

#[cfg(feature = "rustc")]
mod rustc {
    use super::*;

    use rustc_hir::lang_items::LangItem;
    use rustc_infer::infer::InferCtxt;
    use rustc_macros::TypeVisitable;
    use rustc_middle::traits::ObligationCause;
    use rustc_middle::ty::Const;
    use rustc_middle::ty::ParamEnv;
    use rustc_middle::ty::Ty;
    use rustc_middle::ty::TyCtxt;
    use rustc_middle::ty::ValTree;

    /// The source and destination types of a transmutation.
    #[derive(TypeVisitable, Debug, Clone, Copy)]
    pub struct Types<'tcx> {
        /// The source type.
        pub src: Ty<'tcx>,
        /// The destination type.
        pub dst: Ty<'tcx>,
    }

    pub struct TransmuteTypeEnv<'cx, 'tcx> {
        infcx: &'cx InferCtxt<'tcx>,
    }

    impl<'cx, 'tcx> TransmuteTypeEnv<'cx, 'tcx> {
        pub fn new(infcx: &'cx InferCtxt<'tcx>) -> Self {
            Self { infcx }
        }

        #[allow(unused)]
        pub fn is_transmutable(
            &mut self,
            cause: ObligationCause<'tcx>,
            types: Types<'tcx>,
            scope: Ty<'tcx>,
            assume: crate::Assume,
        ) -> crate::Answer<crate::layout::rustc::Ref<'tcx>> {
            crate::maybe_transmutable::MaybeTransmutableQuery::new(
                types.src,
                types.dst,
                scope,
                assume,
                self.infcx.tcx,
            )
            .answer()
        }
    }

    impl Assume {
        /// Constructs an `Assume` from a given const-`Assume`.
        pub fn from_const<'tcx>(
            tcx: TyCtxt<'tcx>,
            param_env: ParamEnv<'tcx>,
            c: Const<'tcx>,
        ) -> Option<Self> {
            use rustc_middle::ty::ScalarInt;
            use rustc_middle::ty::TypeVisitableExt;
            use rustc_span::symbol::sym;

            let c = c.eval(tcx, param_env);

            if let Err(err) = c.error_reported() {
                return Some(Self {
                    alignment: true,
                    lifetimes: true,
                    safety: true,
                    validity: true,
                });
            }

            let adt_def = c.ty().ty_adt_def()?;

            assert_eq!(
                tcx.require_lang_item(LangItem::TransmuteOpts, None),
                adt_def.did(),
                "The given `Const` was not marked with the `{}` lang item.",
                LangItem::TransmuteOpts.name(),
            );

            let variant = adt_def.non_enum_variant();
            let fields = match c.try_to_valtree() {
                Some(ValTree::Branch(branch)) => branch,
                _ => {
                    return Some(Self {
                        alignment: true,
                        lifetimes: true,
                        safety: true,
                        validity: true,
                    });
                }
            };

            let get_field = |name| {
                let (field_idx, _) = variant
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, field_def)| name == field_def.name)
                    .unwrap_or_else(|| panic!("There were no fields named `{name}`."));
                fields[field_idx].unwrap_leaf() == ScalarInt::TRUE
            };

            Some(Self {
                alignment: get_field(sym::alignment),
                lifetimes: get_field(sym::lifetimes),
                safety: get_field(sym::safety),
                validity: get_field(sym::validity),
            })
        }
    }
}

#[cfg(feature = "rustc")]
pub use rustc::*;
