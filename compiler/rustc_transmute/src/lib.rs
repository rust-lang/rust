#![feature(alloc_layout_extra, decl_macro, iterator_try_reduce, never_type)]
#![allow(dead_code, unused_variables)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate tracing;

pub(crate) use rustc_data_structures::fx::{FxIndexMap as Map, FxIndexSet as Set};

pub(crate) mod layout;
pub(crate) mod maybe_transmutable;

#[derive(Default)]
pub struct Assume {
    pub alignment: bool,
    pub lifetimes: bool,
    pub safety: bool,
    pub validity: bool,
}

/// The type encodes answers to the question: "Are these types transmutable?"
#[derive(Debug, Hash, Eq, PartialEq, PartialOrd, Ord, Clone)]
pub enum Answer<R>
where
    R: layout::Ref,
{
    /// `Src` is transmutable into `Dst`.
    Yes,

    /// `Src` is NOT transmutable into `Dst`.
    No(Reason),

    /// `Src` is transmutable into `Dst`, if `src` is transmutable into `dst`.
    IfTransmutable { src: R, dst: R },

    /// `Src` is transmutable into `Dst`, if all of the enclosed requirements are met.
    IfAll(Vec<Answer<R>>),

    /// `Src` is transmutable into `Dst` if any of the enclosed requirements are met.
    IfAny(Vec<Answer<R>>),
}

/// Answers: Why wasn't the source type transmutable into the destination type?
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
}

#[cfg(feature = "rustc")]
mod rustc {
    use super::*;

    use rustc_hir::lang_items::LangItem;
    use rustc_infer::infer::InferCtxt;
    use rustc_macros::{TypeFoldable, TypeVisitable};
    use rustc_middle::traits::ObligationCause;
    use rustc_middle::ty::Binder;
    use rustc_middle::ty::Const;
    use rustc_middle::ty::ParamEnv;
    use rustc_middle::ty::Ty;
    use rustc_middle::ty::TyCtxt;

    /// The source and destination types of a transmutation.
    #[derive(TypeFoldable, TypeVisitable, Debug, Clone, Copy)]
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
            src_and_dst: Binder<'tcx, Types<'tcx>>,
            scope: Ty<'tcx>,
            assume: crate::Assume,
        ) -> crate::Answer<crate::layout::rustc::Ref<'tcx>> {
            let src = src_and_dst.map_bound(|types| types.src).skip_binder();
            let dst = src_and_dst.map_bound(|types| types.dst).skip_binder();
            crate::maybe_transmutable::MaybeTransmutableQuery::new(
                src,
                dst,
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
            let fields = c.to_valtree().unwrap_branch();

            let get_field = |name| {
                let (field_idx, _) = variant
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, field_def)| name == field_def.name)
                    .expect(&format!("There were no fields named `{name}`."));
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
