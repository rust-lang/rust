//! HIR ty lowering: Lowers type-system entities[^1] from the [HIR][hir] to
//! the [`rustc_middle::ty`] representation.
//!
//! Not to be confused with *AST lowering* which lowers AST constructs to HIR ones
//! or with *THIR* / *MIR* *lowering* / *building* which lowers HIR *bodies*
//! (i.e., “executable code”) to THIR / MIR.
//!
//! Most lowering routines are defined on [`dyn HirTyLowerer`](HirTyLowerer) directly,
//! like the main routine of this module, `lower_ty`.
//!
//! This module used to be called `astconv`.
//!
//! [^1]: This includes types, lifetimes / regions, constants in type positions,
//! trait references and bounds.

mod bounds;
mod cmse;
mod dyn_compatibility;
pub mod errors;
pub mod generics;
mod lint;

use std::assert_matches::assert_matches;
use std::slice;

use rustc_ast::TraitObjectSyntax;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagCtxtHandle, ErrorGuaranteed, FatalError, struct_span_code_err,
};
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Namespace, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir, AnonConst, GenericArg, GenericArgs, HirId};
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_infer::traits::ObligationCause;
use rustc_middle::middle::stability::AllowUnstable;
use rustc_middle::mir::interpret::{LitToConstError, LitToConstInput};
use rustc_middle::ty::fold::fold_regions;
use rustc_middle::ty::print::PrintPolyTraitRefExt as _;
use rustc_middle::ty::{
    self, Const, GenericArgKind, GenericArgsRef, GenericParamDefKind, ParamEnv, Ty, TyCtxt,
    TypeVisitableExt, TypingMode,
};
use rustc_middle::{bug, span_bug};
use rustc_session::lint::builtin::AMBIGUOUS_ASSOCIATED_ITEMS;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, kw};
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::wf::object_region_bounds;
use rustc_trait_selection::traits::{self, ObligationCtxt};
use tracing::{debug, instrument};

use crate::bounds::Bounds;
use crate::check::check_abi_fn_ptr;
use crate::errors::{AmbiguousLifetimeBound, BadReturnTypeNotation, WildPatTy};
use crate::hir_ty_lowering::errors::{GenericsArgsErrExtend, prohibit_assoc_item_constraint};
use crate::hir_ty_lowering::generics::{check_generic_arg_count, lower_generic_args};
use crate::middle::resolve_bound_vars as rbv;
use crate::require_c_abi_if_c_variadic;

/// A path segment that is semantically allowed to have generic arguments.
#[derive(Debug)]
pub struct GenericPathSegment(pub DefId, pub usize);

#[derive(Copy, Clone, Debug)]
pub enum PredicateFilter {
    /// All predicates may be implied by the trait.
    All,

    /// Only traits that reference `Self: ..` are implied by the trait.
    SelfOnly,

    /// Only traits that reference `Self: ..` and define an associated type
    /// with the given ident are implied by the trait. This mode exists to
    /// side-step query cycles when lowering associated types.
    SelfTraitThatDefines(Ident),

    /// Only traits that reference `Self: ..` and their associated type bounds.
    /// For example, given `Self: Tr<A: B>`, this would expand to `Self: Tr`
    /// and `<Self as Tr>::A: B`.
    SelfAndAssociatedTypeBounds,

    /// Filter only the `~const` bounds, which are lowered into `HostEffect` clauses.
    ConstIfConst,

    /// Filter only the `~const` bounds which are *also* in the supertrait position.
    SelfConstIfConst,
}

#[derive(Debug)]
pub enum RegionInferReason<'a> {
    /// Lifetime on a trait object that is spelled explicitly, e.g. `+ 'a` or `+ '_`.
    ExplicitObjectLifetime,
    /// A trait object's lifetime when it is elided, e.g. `dyn Any`.
    ObjectLifetimeDefault,
    /// Generic lifetime parameter
    Param(&'a ty::GenericParamDef),
    RegionPredicate,
    Reference,
    OutlivesBound,
}

/// A context which can lower type-system entities from the [HIR][hir] to
/// the [`rustc_middle::ty`] representation.
///
/// This trait used to be called `AstConv`.
pub trait HirTyLowerer<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx>;

    fn dcx(&self) -> DiagCtxtHandle<'_>;

    /// Returns the [`LocalDefId`] of the overarching item whose constituents get lowered.
    fn item_def_id(&self) -> LocalDefId;

    /// Returns the region to use when a lifetime is omitted (and not elided).
    fn re_infer(&self, span: Span, reason: RegionInferReason<'_>) -> ty::Region<'tcx>;

    /// Returns the type to use when a type is omitted.
    fn ty_infer(&self, param: Option<&ty::GenericParamDef>, span: Span) -> Ty<'tcx>;

    /// Returns the const to use when a const is omitted.
    fn ct_infer(&self, param: Option<&ty::GenericParamDef>, span: Span) -> Const<'tcx>;

    fn register_trait_ascription_bounds(
        &self,
        bounds: Vec<(ty::Clause<'tcx>, Span)>,
        hir_id: HirId,
        span: Span,
    );

    /// Probe bounds in scope where the bounded type coincides with the given type parameter.
    ///
    /// Rephrased, this returns bounds of the form `T: Trait`, where `T` is a type parameter
    /// with the given `def_id`. This is a subset of the full set of bounds.
    ///
    /// This method may use the given `assoc_name` to disregard bounds whose trait reference
    /// doesn't define an associated item with the provided name.
    ///
    /// This is used for one specific purpose: Resolving “short-hand” associated type references
    /// like `T::Item` where `T` is a type parameter. In principle, we would do that by first
    /// getting the full set of predicates in scope and then filtering down to find those that
    /// apply to `T`, but this can lead to cycle errors. The problem is that we have to do this
    /// resolution *in order to create the predicates in the first place*.
    /// Hence, we have this “special pass”.
    fn probe_ty_param_bounds(
        &self,
        span: Span,
        def_id: LocalDefId,
        assoc_name: Ident,
    ) -> ty::EarlyBinder<'tcx, &'tcx [(ty::Clause<'tcx>, Span)]>;

    /// Lower an associated type to a projection.
    ///
    /// This method has to be defined by the concrete lowering context because
    /// dealing with higher-ranked trait references depends on its capabilities:
    ///
    /// If the context can make use of type inference, it can simply instantiate
    /// any late-bound vars bound by the trait reference with inference variables.
    /// If it doesn't support type inference, there is nothing reasonable it can
    /// do except reject the associated type.
    ///
    /// The canonical example of this is associated type `T::P` where `T` is a type
    /// param constrained by `T: for<'a> Trait<'a>` and where `Trait` defines `P`.
    fn lower_assoc_ty(
        &self,
        span: Span,
        item_def_id: DefId,
        item_segment: &hir::PathSegment<'tcx>,
        poly_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Ty<'tcx>;

    fn lower_fn_sig(
        &self,
        decl: &hir::FnDecl<'tcx>,
        generics: Option<&hir::Generics<'_>>,
        hir_id: HirId,
        hir_ty: Option<&hir::Ty<'_>>,
    ) -> (Vec<Ty<'tcx>>, Ty<'tcx>);

    /// Returns `AdtDef` if `ty` is an ADT.
    ///
    /// Note that `ty` might be a alias type that needs normalization.
    /// This used to get the enum variants in scope of the type.
    /// For example, `Self::A` could refer to an associated type
    /// or to an enum variant depending on the result of this function.
    fn probe_adt(&self, span: Span, ty: Ty<'tcx>) -> Option<ty::AdtDef<'tcx>>;

    /// Record the lowered type of a HIR node in this context.
    fn record_ty(&self, hir_id: HirId, ty: Ty<'tcx>, span: Span);

    /// The inference context of the lowering context if applicable.
    fn infcx(&self) -> Option<&InferCtxt<'tcx>>;

    /// Convenience method for coercing the lowering context into a trait object type.
    ///
    /// Most lowering routines are defined on the trait object type directly
    /// necessitating a coercion step from the concrete lowering context.
    fn lowerer(&self) -> &dyn HirTyLowerer<'tcx>
    where
        Self: Sized,
    {
        self
    }
}

/// The "qualified self" of an associated item path.
///
/// For diagnostic purposes only.
enum AssocItemQSelf {
    Trait(DefId),
    TyParam(LocalDefId, Span),
    SelfTyAlias,
}

impl AssocItemQSelf {
    fn to_string(&self, tcx: TyCtxt<'_>) -> String {
        match *self {
            Self::Trait(def_id) => tcx.def_path_str(def_id),
            Self::TyParam(def_id, _) => tcx.hir().ty_param_name(def_id).to_string(),
            Self::SelfTyAlias => kw::SelfUpper.to_string(),
        }
    }
}

/// In some cases, [`hir::ConstArg`]s that are being used in the type system
/// through const generics need to have their type "fed" to them
/// using the query system.
///
/// Use this enum with `<dyn HirTyLowerer>::lower_const_arg` to instruct it with the
/// desired behavior.
#[derive(Debug, Clone, Copy)]
pub enum FeedConstTy {
    /// Feed the type.
    ///
    /// The `DefId` belongs to the const param that we are supplying
    /// this (anon) const arg to.
    Param(DefId),
    /// Don't feed the type.
    No,
}

/// New-typed boolean indicating whether explicit late-bound lifetimes
/// are present in a set of generic arguments.
///
/// For example if we have some method `fn f<'a>(&'a self)` implemented
/// for some type `T`, although `f` is generic in the lifetime `'a`, `'a`
/// is late-bound so should not be provided explicitly. Thus, if `f` is
/// instantiated with some generic arguments providing `'a` explicitly,
/// we taint those arguments with `ExplicitLateBound::Yes` so that we
/// can provide an appropriate diagnostic later.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum ExplicitLateBound {
    Yes,
    No,
}

#[derive(Copy, Clone, PartialEq)]
pub enum IsMethodCall {
    Yes,
    No,
}

/// Denotes the "position" of a generic argument, indicating if it is a generic type,
/// generic function or generic method call.
#[derive(Copy, Clone, PartialEq)]
pub(crate) enum GenericArgPosition {
    Type,
    Value, // e.g., functions
    MethodCall,
}

/// A marker denoting that the generic arguments that were
/// provided did not match the respective generic parameters.
#[derive(Clone, Debug)]
pub struct GenericArgCountMismatch {
    pub reported: ErrorGuaranteed,
    /// A list of indices of arguments provided that were not valid.
    pub invalid_args: Vec<usize>,
}

/// Decorates the result of a generic argument count mismatch
/// check with whether explicit late bounds were provided.
#[derive(Clone, Debug)]
pub struct GenericArgCountResult {
    pub explicit_late_bound: ExplicitLateBound,
    pub correct: Result<(), GenericArgCountMismatch>,
}

/// A context which can lower HIR's [`GenericArg`] to `rustc_middle`'s [`ty::GenericArg`].
///
/// Its only consumer is [`generics::lower_generic_args`].
/// Read its documentation to learn more.
pub trait GenericArgsLowerer<'a, 'tcx> {
    fn args_for_def_id(&mut self, def_id: DefId) -> (Option<&'a GenericArgs<'tcx>>, bool);

    fn provided_kind(
        &mut self,
        preceding_args: &[ty::GenericArg<'tcx>],
        param: &ty::GenericParamDef,
        arg: &GenericArg<'tcx>,
    ) -> ty::GenericArg<'tcx>;

    fn inferred_kind(
        &mut self,
        preceding_args: &[ty::GenericArg<'tcx>],
        param: &ty::GenericParamDef,
        infer_args: bool,
    ) -> ty::GenericArg<'tcx>;
}

impl<'tcx> dyn HirTyLowerer<'tcx> + '_ {
    /// Lower a lifetime from the HIR to our internal notion of a lifetime called a *region*.
    #[instrument(level = "debug", skip(self), ret)]
    pub fn lower_lifetime(
        &self,
        lifetime: &hir::Lifetime,
        reason: RegionInferReason<'_>,
    ) -> ty::Region<'tcx> {
        if let Some(resolved) = self.tcx().named_bound_var(lifetime.hir_id) {
            self.lower_resolved_lifetime(resolved)
        } else {
            self.re_infer(lifetime.ident.span, reason)
        }
    }

    /// Lower a lifetime from the HIR to our internal notion of a lifetime called a *region*.
    #[instrument(level = "debug", skip(self), ret)]
    pub fn lower_resolved_lifetime(&self, resolved: rbv::ResolvedArg) -> ty::Region<'tcx> {
        let tcx = self.tcx();
        let lifetime_name = |def_id| tcx.hir().name(tcx.local_def_id_to_hir_id(def_id));

        match resolved {
            rbv::ResolvedArg::StaticLifetime => tcx.lifetimes.re_static,

            rbv::ResolvedArg::LateBound(debruijn, index, def_id) => {
                let name = lifetime_name(def_id);
                let br = ty::BoundRegion {
                    var: ty::BoundVar::from_u32(index),
                    kind: ty::BoundRegionKind::Named(def_id.to_def_id(), name),
                };
                ty::Region::new_bound(tcx, debruijn, br)
            }

            rbv::ResolvedArg::EarlyBound(def_id) => {
                let name = tcx.hir().ty_param_name(def_id);
                let item_def_id = tcx.hir().ty_param_owner(def_id);
                let generics = tcx.generics_of(item_def_id);
                let index = generics.param_def_id_to_index[&def_id.to_def_id()];
                ty::Region::new_early_param(tcx, ty::EarlyParamRegion { index, name })
            }

            rbv::ResolvedArg::Free(scope, id) => {
                let name = lifetime_name(id);
                ty::Region::new_late_param(
                    tcx,
                    scope.to_def_id(),
                    ty::LateParamRegionKind::Named(id.to_def_id(), name),
                )

                // (*) -- not late-bound, won't change
            }

            rbv::ResolvedArg::Error(guar) => ty::Region::new_error(tcx, guar),
        }
    }

    pub fn lower_generic_args_of_path_segment(
        &self,
        span: Span,
        def_id: DefId,
        item_segment: &hir::PathSegment<'tcx>,
    ) -> GenericArgsRef<'tcx> {
        let (args, _) = self.lower_generic_args_of_path(span, def_id, &[], item_segment, None);
        if let Some(c) = item_segment.args().constraints.first() {
            prohibit_assoc_item_constraint(self, c, Some((def_id, item_segment, span)));
        }
        args
    }

    /// Lower the generic arguments provided to some path.
    ///
    /// If this is a trait reference, you also need to pass the self type `self_ty`.
    /// The lowering process may involve applying defaulted type parameters.
    ///
    /// Associated item constraints are not handled here! They are either lowered via
    /// `lower_assoc_item_constraint` or rejected via `prohibit_assoc_item_constraint`.
    ///
    /// ### Example
    ///
    /// ```ignore (illustrative)
    ///    T: std::ops::Index<usize, Output = u32>
    /// // ^1 ^^^^^^^^^^^^^^2 ^^^^3  ^^^^^^^^^^^4
    /// ```
    ///
    /// 1. The `self_ty` here would refer to the type `T`.
    /// 2. The path in question is the path to the trait `std::ops::Index`,
    ///    which will have been resolved to a `def_id`
    /// 3. The `generic_args` contains info on the `<...>` contents. The `usize` type
    ///    parameters are returned in the `GenericArgsRef`
    /// 4. Associated item constraints like `Output = u32` are contained in `generic_args.constraints`.
    ///
    /// Note that the type listing given here is *exactly* what the user provided.
    ///
    /// For (generic) associated types
    ///
    /// ```ignore (illustrative)
    /// <Vec<u8> as Iterable<u8>>::Iter::<'a>
    /// ```
    ///
    /// We have the parent args are the args for the parent trait:
    /// `[Vec<u8>, u8]` and `generic_args` are the arguments for the associated
    /// type itself: `['a]`. The returned `GenericArgsRef` concatenates these two
    /// lists: `[Vec<u8>, u8, 'a]`.
    #[instrument(level = "debug", skip(self, span), ret)]
    fn lower_generic_args_of_path(
        &self,
        span: Span,
        def_id: DefId,
        parent_args: &[ty::GenericArg<'tcx>],
        segment: &hir::PathSegment<'tcx>,
        self_ty: Option<Ty<'tcx>>,
    ) -> (GenericArgsRef<'tcx>, GenericArgCountResult) {
        // If the type is parameterized by this region, then replace this
        // region with the current anon region binding (in other words,
        // whatever & would get replaced with).

        let tcx = self.tcx();
        let generics = tcx.generics_of(def_id);
        debug!(?generics);

        if generics.has_self {
            if generics.parent.is_some() {
                // The parent is a trait so it should have at least one
                // generic parameter for the `Self` type.
                assert!(!parent_args.is_empty())
            } else {
                // This item (presumably a trait) needs a self-type.
                assert!(self_ty.is_some());
            }
        } else {
            assert!(self_ty.is_none());
        }

        let arg_count = check_generic_arg_count(
            self,
            def_id,
            segment,
            generics,
            GenericArgPosition::Type,
            self_ty.is_some(),
        );

        // Skip processing if type has no generic parameters.
        // Traits always have `Self` as a generic parameter, which means they will not return early
        // here and so associated item constraints will be handled regardless of whether there are
        // any non-`Self` generic parameters.
        if generics.is_own_empty() {
            return (tcx.mk_args(parent_args), arg_count);
        }

        struct GenericArgsCtxt<'a, 'tcx> {
            lowerer: &'a dyn HirTyLowerer<'tcx>,
            def_id: DefId,
            generic_args: &'a GenericArgs<'tcx>,
            span: Span,
            infer_args: bool,
            incorrect_args: &'a Result<(), GenericArgCountMismatch>,
        }

        impl<'a, 'tcx> GenericArgsLowerer<'a, 'tcx> for GenericArgsCtxt<'a, 'tcx> {
            fn args_for_def_id(&mut self, did: DefId) -> (Option<&'a GenericArgs<'tcx>>, bool) {
                if did == self.def_id {
                    (Some(self.generic_args), self.infer_args)
                } else {
                    // The last component of this tuple is unimportant.
                    (None, false)
                }
            }

            fn provided_kind(
                &mut self,
                _preceding_args: &[ty::GenericArg<'tcx>],
                param: &ty::GenericParamDef,
                arg: &GenericArg<'tcx>,
            ) -> ty::GenericArg<'tcx> {
                let tcx = self.lowerer.tcx();

                if let Err(incorrect) = self.incorrect_args {
                    if incorrect.invalid_args.contains(&(param.index as usize)) {
                        return param.to_error(tcx);
                    }
                }

                let handle_ty_args = |has_default, ty: &hir::Ty<'tcx>| {
                    if has_default {
                        tcx.check_optional_stability(
                            param.def_id,
                            Some(arg.hir_id()),
                            arg.span(),
                            None,
                            AllowUnstable::No,
                            |_, _| {
                                // Default generic parameters may not be marked
                                // with stability attributes, i.e. when the
                                // default parameter was defined at the same time
                                // as the rest of the type. As such, we ignore missing
                                // stability attributes.
                            },
                        );
                    }
                    self.lowerer.lower_ty(ty).into()
                };

                match (&param.kind, arg) {
                    (GenericParamDefKind::Lifetime, GenericArg::Lifetime(lt)) => {
                        self.lowerer.lower_lifetime(lt, RegionInferReason::Param(param)).into()
                    }
                    (&GenericParamDefKind::Type { has_default, .. }, GenericArg::Type(ty)) => {
                        handle_ty_args(has_default, ty)
                    }
                    (&GenericParamDefKind::Type { has_default, .. }, GenericArg::Infer(inf)) => {
                        handle_ty_args(has_default, &inf.to_ty())
                    }
                    (GenericParamDefKind::Const { .. }, GenericArg::Const(ct)) => {
                        self.lowerer.lower_const_arg(ct, FeedConstTy::Param(param.def_id)).into()
                    }
                    (&GenericParamDefKind::Const { .. }, GenericArg::Infer(inf)) => {
                        self.lowerer.ct_infer(Some(param), inf.span).into()
                    }
                    (kind, arg) => span_bug!(
                        self.span,
                        "mismatched path argument for kind {kind:?}: found arg {arg:?}"
                    ),
                }
            }

            fn inferred_kind(
                &mut self,
                preceding_args: &[ty::GenericArg<'tcx>],
                param: &ty::GenericParamDef,
                infer_args: bool,
            ) -> ty::GenericArg<'tcx> {
                let tcx = self.lowerer.tcx();

                if let Err(incorrect) = self.incorrect_args {
                    if incorrect.invalid_args.contains(&(param.index as usize)) {
                        return param.to_error(tcx);
                    }
                }
                match param.kind {
                    GenericParamDefKind::Lifetime => {
                        self.lowerer.re_infer(self.span, RegionInferReason::Param(param)).into()
                    }
                    GenericParamDefKind::Type { has_default, .. } => {
                        if !infer_args && has_default {
                            // No type parameter provided, but a default exists.
                            if let Some(prev) =
                                preceding_args.iter().find_map(|arg| match arg.unpack() {
                                    GenericArgKind::Type(ty) => ty.error_reported().err(),
                                    _ => None,
                                })
                            {
                                // Avoid ICE #86756 when type error recovery goes awry.
                                return Ty::new_error(tcx, prev).into();
                            }
                            tcx.at(self.span)
                                .type_of(param.def_id)
                                .instantiate(tcx, preceding_args)
                                .into()
                        } else if infer_args {
                            self.lowerer.ty_infer(Some(param), self.span).into()
                        } else {
                            // We've already errored above about the mismatch.
                            Ty::new_misc_error(tcx).into()
                        }
                    }
                    GenericParamDefKind::Const { has_default, .. } => {
                        let ty = tcx
                            .at(self.span)
                            .type_of(param.def_id)
                            .no_bound_vars()
                            .expect("const parameter types cannot be generic");
                        if let Err(guar) = ty.error_reported() {
                            return ty::Const::new_error(tcx, guar).into();
                        }
                        if !infer_args && has_default {
                            tcx.const_param_default(param.def_id)
                                .instantiate(tcx, preceding_args)
                                .into()
                        } else if infer_args {
                            self.lowerer.ct_infer(Some(param), self.span).into()
                        } else {
                            // We've already errored above about the mismatch.
                            ty::Const::new_misc_error(tcx).into()
                        }
                    }
                }
            }
        }

        let mut args_ctx = GenericArgsCtxt {
            lowerer: self,
            def_id,
            span,
            generic_args: segment.args(),
            infer_args: segment.infer_args,
            incorrect_args: &arg_count.correct,
        };
        let args = lower_generic_args(
            self,
            def_id,
            parent_args,
            self_ty.is_some(),
            self_ty,
            &arg_count,
            &mut args_ctx,
        );

        (args, arg_count)
    }

    #[instrument(level = "debug", skip_all)]
    pub fn lower_generic_args_of_assoc_item(
        &self,
        span: Span,
        item_def_id: DefId,
        item_segment: &hir::PathSegment<'tcx>,
        parent_args: GenericArgsRef<'tcx>,
    ) -> GenericArgsRef<'tcx> {
        debug!(?span, ?item_def_id, ?item_segment);
        let (args, _) =
            self.lower_generic_args_of_path(span, item_def_id, parent_args, item_segment, None);
        if let Some(c) = item_segment.args().constraints.first() {
            prohibit_assoc_item_constraint(self, c, Some((item_def_id, item_segment, span)));
        }
        args
    }

    /// Lower a trait reference as found in an impl header as the implementee.
    ///
    /// The self type `self_ty` is the implementer of the trait.
    pub fn lower_impl_trait_ref(
        &self,
        trait_ref: &hir::TraitRef<'tcx>,
        self_ty: Ty<'tcx>,
    ) -> ty::TraitRef<'tcx> {
        let _ = self.prohibit_generic_args(
            trait_ref.path.segments.split_last().unwrap().1.iter(),
            GenericsArgsErrExtend::None,
        );

        self.lower_mono_trait_ref(
            trait_ref.path.span,
            trait_ref.trait_def_id().unwrap_or_else(|| FatalError.raise()),
            self_ty,
            trait_ref.path.segments.last().unwrap(),
            true,
        )
    }

    /// Lower a polymorphic trait reference given a self type into `bounds`.
    ///
    /// *Polymorphic* in the sense that it may bind late-bound vars.
    ///
    /// This may generate auxiliary bounds iff the trait reference contains associated item constraints.
    ///
    /// ### Example
    ///
    /// Given the trait ref `Iterator<Item = u32>` and the self type `Ty`, this will add the
    ///
    /// 1. *trait predicate* `<Ty as Iterator>` (known as `Ty: Iterator` in the surface syntax) and the
    /// 2. *projection predicate* `<Ty as Iterator>::Item = u32`
    ///
    /// to `bounds`.
    ///
    /// ### A Note on Binders
    ///
    /// Against our usual convention, there is an implied binder around the `self_ty` and the
    /// `trait_ref` here. So they may reference late-bound vars.
    ///
    /// If for example you had `for<'a> Foo<'a>: Bar<'a>`, then the `self_ty` would be `Foo<'a>`
    /// where `'a` is a bound region at depth 0. Similarly, the `trait_ref` would be `Bar<'a>`.
    /// The lowered poly-trait-ref will track this binder explicitly, however.
    #[instrument(level = "debug", skip(self, span, constness, bounds))]
    pub(crate) fn lower_poly_trait_ref(
        &self,
        trait_ref: &hir::TraitRef<'tcx>,
        span: Span,
        constness: hir::BoundConstness,
        polarity: hir::BoundPolarity,
        self_ty: Ty<'tcx>,
        bounds: &mut Bounds<'tcx>,
        predicate_filter: PredicateFilter,
    ) -> GenericArgCountResult {
        let trait_def_id = trait_ref.trait_def_id().unwrap_or_else(|| FatalError.raise());
        let trait_segment = trait_ref.path.segments.last().unwrap();

        let _ = self.prohibit_generic_args(
            trait_ref.path.segments.split_last().unwrap().1.iter(),
            GenericsArgsErrExtend::None,
        );
        self.complain_about_internal_fn_trait(span, trait_def_id, trait_segment, false);

        let (generic_args, arg_count) = self.lower_generic_args_of_path(
            trait_ref.path.span,
            trait_def_id,
            &[],
            trait_segment,
            Some(self_ty),
        );

        let tcx = self.tcx();
        let bound_vars = tcx.late_bound_vars(trait_ref.hir_ref_id);
        debug!(?bound_vars);

        let poly_trait_ref = ty::Binder::bind_with_vars(
            ty::TraitRef::new_from_args(tcx, trait_def_id, generic_args),
            bound_vars,
        );

        let polarity = match polarity {
            rustc_ast::BoundPolarity::Positive => ty::PredicatePolarity::Positive,
            rustc_ast::BoundPolarity::Negative(_) => ty::PredicatePolarity::Negative,
            rustc_ast::BoundPolarity::Maybe(_) => {
                // Validate associated type at least. We may want to reject these
                // outright in the future...
                for constraint in trait_segment.args().constraints {
                    let _ = self.lower_assoc_item_constraint(
                        trait_ref.hir_ref_id,
                        poly_trait_ref,
                        constraint,
                        &mut Default::default(),
                        &mut Default::default(),
                        constraint.span,
                        predicate_filter,
                    );
                }
                return arg_count;
            }
        };

        if let hir::BoundConstness::Always(span) | hir::BoundConstness::Maybe(span) = constness
            && !self.tcx().is_const_trait(trait_def_id)
        {
            let (def_span, suggestion, suggestion_pre) =
                match (trait_def_id.is_local(), self.tcx().sess.is_nightly_build()) {
                    (true, true) => (
                        None,
                        Some(tcx.def_span(trait_def_id).shrink_to_lo()),
                        if self.tcx().features().const_trait_impl() {
                            ""
                        } else {
                            "enable `#![feature(const_trait_impl)]` in your crate and "
                        },
                    ),
                    (false, _) | (_, false) => (Some(tcx.def_span(trait_def_id)), None, ""),
                };
            self.dcx().emit_err(crate::errors::ConstBoundForNonConstTrait {
                span,
                modifier: constness.as_str(),
                def_span,
                trait_name: self.tcx().def_path_str(trait_def_id),
                suggestion_pre,
                suggestion,
            });
        }

        match predicate_filter {
            // This is only concerned with trait predicates.
            PredicateFilter::SelfTraitThatDefines(..) => {
                bounds.push_trait_bound(tcx, poly_trait_ref, span, polarity);
            }
            PredicateFilter::All
            | PredicateFilter::SelfOnly
            | PredicateFilter::SelfAndAssociatedTypeBounds => {
                debug!(?poly_trait_ref);
                bounds.push_trait_bound(tcx, poly_trait_ref, span, polarity);

                match constness {
                    hir::BoundConstness::Always(span) => {
                        if polarity == ty::PredicatePolarity::Positive {
                            bounds.push_const_bound(
                                tcx,
                                poly_trait_ref,
                                ty::BoundConstness::Const,
                                span,
                            );
                        }
                    }
                    hir::BoundConstness::Maybe(_) => {
                        // We don't emit a const bound here, since that would mean that we
                        // unconditionally need to prove a `HostEffect` predicate, even when
                        // the predicates are being instantiated in a non-const context. This
                        // is instead handled in the `const_conditions` query.
                    }
                    hir::BoundConstness::Never => {}
                }
            }
            // On the flip side, when filtering `ConstIfConst` bounds, we only need to convert
            // `~const` bounds. All other predicates are handled in their respective queries.
            //
            // Note that like `PredicateFilter::SelfOnly`, we don't need to do any filtering
            // here because we only call this on self bounds, and deal with the recursive case
            // in `lower_assoc_item_constraint`.
            PredicateFilter::ConstIfConst | PredicateFilter::SelfConstIfConst => match constness {
                hir::BoundConstness::Maybe(span) => {
                    if polarity == ty::PredicatePolarity::Positive {
                        bounds.push_const_bound(
                            tcx,
                            poly_trait_ref,
                            ty::BoundConstness::Maybe,
                            span,
                        );
                    }
                }
                hir::BoundConstness::Always(_) | hir::BoundConstness::Never => {}
            },
        }

        let mut dup_constraints = FxIndexMap::default();
        for constraint in trait_segment.args().constraints {
            // Don't register any associated item constraints for negative bounds,
            // since we should have emitted an error for them earlier, and they
            // would not be well-formed!
            if polarity != ty::PredicatePolarity::Positive {
                self.dcx().span_delayed_bug(
                    constraint.span,
                    "negative trait bounds should not have assoc item constraints",
                );
                break;
            }

            // Specify type to assert that error was already reported in `Err` case.
            let _: Result<_, ErrorGuaranteed> = self.lower_assoc_item_constraint(
                trait_ref.hir_ref_id,
                poly_trait_ref,
                constraint,
                bounds,
                &mut dup_constraints,
                constraint.span,
                predicate_filter,
            );
            // Okay to ignore `Err` because of `ErrorGuaranteed` (see above).
        }

        arg_count
    }

    /// Lower a monomorphic trait reference given a self type while prohibiting associated item bindings.
    ///
    /// *Monomorphic* in the sense that it doesn't bind any late-bound vars.
    fn lower_mono_trait_ref(
        &self,
        span: Span,
        trait_def_id: DefId,
        self_ty: Ty<'tcx>,
        trait_segment: &hir::PathSegment<'tcx>,
        is_impl: bool,
    ) -> ty::TraitRef<'tcx> {
        self.complain_about_internal_fn_trait(span, trait_def_id, trait_segment, is_impl);

        let (generic_args, _) =
            self.lower_generic_args_of_path(span, trait_def_id, &[], trait_segment, Some(self_ty));
        if let Some(c) = trait_segment.args().constraints.first() {
            prohibit_assoc_item_constraint(self, c, Some((trait_def_id, trait_segment, span)));
        }
        ty::TraitRef::new_from_args(self.tcx(), trait_def_id, generic_args)
    }

    fn probe_trait_that_defines_assoc_item(
        &self,
        trait_def_id: DefId,
        assoc_kind: ty::AssocKind,
        assoc_name: Ident,
    ) -> bool {
        self.tcx()
            .associated_items(trait_def_id)
            .find_by_name_and_kind(self.tcx(), assoc_name, assoc_kind, trait_def_id)
            .is_some()
    }

    fn lower_path_segment(
        &self,
        span: Span,
        did: DefId,
        item_segment: &hir::PathSegment<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx();
        let args = self.lower_generic_args_of_path_segment(span, did, item_segment);

        if let DefKind::TyAlias = tcx.def_kind(did)
            && tcx.type_alias_is_lazy(did)
        {
            // Type aliases defined in crates that have the
            // feature `lazy_type_alias` enabled get encoded as a type alias that normalization will
            // then actually instantiate the where bounds of.
            let alias_ty = ty::AliasTy::new_from_args(tcx, did, args);
            Ty::new_alias(tcx, ty::Weak, alias_ty)
        } else {
            tcx.at(span).type_of(did).instantiate(tcx, args)
        }
    }

    /// Search for a trait bound on a type parameter whose trait defines the associated item
    /// given by `assoc_name` and `kind`.
    ///
    /// This fails if there is no such bound in the list of candidates or if there are multiple
    /// candidates in which case it reports ambiguity.
    ///
    /// `ty_param_def_id` is the `LocalDefId` of the type parameter.
    #[instrument(level = "debug", skip_all, ret)]
    fn probe_single_ty_param_bound_for_assoc_item(
        &self,
        ty_param_def_id: LocalDefId,
        ty_param_span: Span,
        kind: ty::AssocKind,
        assoc_name: Ident,
        span: Span,
    ) -> Result<ty::PolyTraitRef<'tcx>, ErrorGuaranteed> {
        debug!(?ty_param_def_id, ?assoc_name, ?span);
        let tcx = self.tcx();

        let predicates = &self.probe_ty_param_bounds(span, ty_param_def_id, assoc_name);
        debug!("predicates={:#?}", predicates);

        self.probe_single_bound_for_assoc_item(
            || {
                let trait_refs = predicates
                    .iter_identity_copied()
                    .filter_map(|(p, _)| Some(p.as_trait_clause()?.map_bound(|t| t.trait_ref)));
                traits::transitive_bounds_that_define_assoc_item(tcx, trait_refs, assoc_name)
            },
            AssocItemQSelf::TyParam(ty_param_def_id, ty_param_span),
            kind,
            assoc_name,
            span,
            None,
        )
    }

    /// Search for a single trait bound whose trait defines the associated item given by `assoc_name`.
    ///
    /// This fails if there is no such bound in the list of candidates or if there are multiple
    /// candidates in which case it reports ambiguity.
    #[instrument(level = "debug", skip(self, all_candidates, qself, constraint), ret)]
    fn probe_single_bound_for_assoc_item<I>(
        &self,
        all_candidates: impl Fn() -> I,
        qself: AssocItemQSelf,
        assoc_kind: ty::AssocKind,
        assoc_name: Ident,
        span: Span,
        constraint: Option<&hir::AssocItemConstraint<'tcx>>,
    ) -> Result<ty::PolyTraitRef<'tcx>, ErrorGuaranteed>
    where
        I: Iterator<Item = ty::PolyTraitRef<'tcx>>,
    {
        let tcx = self.tcx();

        let mut matching_candidates = all_candidates().filter(|r| {
            self.probe_trait_that_defines_assoc_item(r.def_id(), assoc_kind, assoc_name)
        });

        let Some(bound) = matching_candidates.next() else {
            let reported = self.complain_about_assoc_item_not_found(
                all_candidates,
                qself,
                assoc_kind,
                assoc_name,
                span,
                constraint,
            );
            return Err(reported);
        };
        debug!(?bound);

        if let Some(bound2) = matching_candidates.next() {
            debug!(?bound2);

            let assoc_kind_str = errors::assoc_kind_str(assoc_kind);
            let qself_str = qself.to_string(tcx);
            let mut err = self.dcx().create_err(crate::errors::AmbiguousAssocItem {
                span,
                assoc_kind: assoc_kind_str,
                assoc_name,
                qself: &qself_str,
            });
            // Provide a more specific error code index entry for equality bindings.
            err.code(
                if let Some(constraint) = constraint
                    && let hir::AssocItemConstraintKind::Equality { .. } = constraint.kind
                {
                    E0222
                } else {
                    E0221
                },
            );

            // FIXME(#97583): Print associated item bindings properly (i.e., not as equality predicates!).
            // FIXME: Turn this into a structured, translateable & more actionable suggestion.
            let mut where_bounds = vec![];
            for bound in [bound, bound2].into_iter().chain(matching_candidates) {
                let bound_id = bound.def_id();
                let bound_span = tcx
                    .associated_items(bound_id)
                    .find_by_name_and_kind(tcx, assoc_name, assoc_kind, bound_id)
                    .and_then(|item| tcx.hir().span_if_local(item.def_id));

                if let Some(bound_span) = bound_span {
                    err.span_label(
                        bound_span,
                        format!("ambiguous `{assoc_name}` from `{}`", bound.print_trait_sugared(),),
                    );
                    if let Some(constraint) = constraint {
                        match constraint.kind {
                            hir::AssocItemConstraintKind::Equality { term } => {
                                let term: ty::Term<'_> = match term {
                                    hir::Term::Ty(ty) => self.lower_ty(ty).into(),
                                    hir::Term::Const(ct) => {
                                        self.lower_const_arg(ct, FeedConstTy::No).into()
                                    }
                                };
                                if term.references_error() {
                                    continue;
                                }
                                // FIXME(#97583): This isn't syntactically well-formed!
                                where_bounds.push(format!(
                                    "        T: {trait}::{assoc_name} = {term}",
                                    trait = bound.print_only_trait_path(),
                                ));
                            }
                            // FIXME: Provide a suggestion.
                            hir::AssocItemConstraintKind::Bound { bounds: _ } => {}
                        }
                    } else {
                        err.span_suggestion_verbose(
                            span.with_hi(assoc_name.span.lo()),
                            "use fully-qualified syntax to disambiguate",
                            format!("<{qself_str} as {}>::", bound.print_only_trait_path()),
                            Applicability::MaybeIncorrect,
                        );
                    }
                } else {
                    err.note(format!(
                        "associated {assoc_kind_str} `{assoc_name}` could derive from `{}`",
                        bound.print_only_trait_path(),
                    ));
                }
            }
            if !where_bounds.is_empty() {
                err.help(format!(
                    "consider introducing a new type parameter `T` and adding `where` constraints:\
                     \n    where\n        T: {qself_str},\n{}",
                    where_bounds.join(",\n"),
                ));
                let reported = err.emit();
                return Err(reported);
            }
            err.emit();
        }

        Ok(bound)
    }

    /// Lower a [type-relative] path referring to an associated type or to an enum variant.
    ///
    /// If the path refers to an enum variant and `permit_variants` holds,
    /// the returned type is simply the provided self type `qself_ty`.
    ///
    /// A path like `A::B::C::D` is understood as `<A::B::C>::D`. I.e.,
    /// `qself_ty` / `qself` is `A::B::C` and `assoc_segment` is `D`.
    /// We return the lowered type and the `DefId` for the whole path.
    ///
    /// We only support associated type paths whose self type is a type parameter or a `Self`
    /// type alias (in a trait impl) like `T::Ty` (where `T` is a ty param) or `Self::Ty`.
    /// We **don't** support paths whose self type is an arbitrary type like `Struct::Ty` where
    /// struct `Struct` impls an in-scope trait that defines an associated type called `Ty`.
    /// For the latter case, we report ambiguity.
    /// While desirable to support, the implementation would be non-trivial. Tracked in [#22519].
    ///
    /// At the time of writing, *inherent associated types* are also resolved here. This however
    /// is [problematic][iat]. A proper implementation would be as non-trivial as the one
    /// described in the previous paragraph and their modeling of projections would likely be
    /// very similar in nature.
    ///
    /// [type-relative]: hir::QPath::TypeRelative
    /// [#22519]: https://github.com/rust-lang/rust/issues/22519
    /// [iat]: https://github.com/rust-lang/rust/issues/8995#issuecomment-1569208403
    //
    // NOTE: When this function starts resolving `Trait::AssocTy` successfully
    // it should also start reporting the `BARE_TRAIT_OBJECTS` lint.
    #[instrument(level = "debug", skip_all, ret)]
    pub fn lower_assoc_path(
        &self,
        hir_ref_id: HirId,
        span: Span,
        qself_ty: Ty<'tcx>,
        qself: &'tcx hir::Ty<'tcx>,
        assoc_segment: &'tcx hir::PathSegment<'tcx>,
        permit_variants: bool,
    ) -> Result<(Ty<'tcx>, DefKind, DefId), ErrorGuaranteed> {
        debug!(%qself_ty, ?assoc_segment.ident);
        let tcx = self.tcx();

        let assoc_ident = assoc_segment.ident;

        // Check if we have an enum variant or an inherent associated type.
        let mut variant_resolution = None;
        if let Some(adt_def) = self.probe_adt(span, qself_ty) {
            if adt_def.is_enum() {
                let variant_def = adt_def
                    .variants()
                    .iter()
                    .find(|vd| tcx.hygienic_eq(assoc_ident, vd.ident(tcx), adt_def.did()));
                if let Some(variant_def) = variant_def {
                    if permit_variants {
                        tcx.check_stability(variant_def.def_id, Some(hir_ref_id), span, None);
                        let _ = self.prohibit_generic_args(
                            slice::from_ref(assoc_segment).iter(),
                            GenericsArgsErrExtend::EnumVariant { qself, assoc_segment, adt_def },
                        );
                        return Ok((qself_ty, DefKind::Variant, variant_def.def_id));
                    } else {
                        variant_resolution = Some(variant_def.def_id);
                    }
                }
            }

            // FIXME(inherent_associated_types, #106719): Support self types other than ADTs.
            if let Some((ty, did)) = self.probe_inherent_assoc_ty(
                assoc_ident,
                assoc_segment,
                adt_def.did(),
                qself_ty,
                hir_ref_id,
                span,
            )? {
                return Ok((ty, DefKind::AssocTy, did));
            }
        }

        let qself_res = if let hir::TyKind::Path(hir::QPath::Resolved(_, path)) = &qself.kind {
            path.res
        } else {
            Res::Err
        };

        // Find the type of the associated item, and the trait where the associated
        // item is declared.
        let bound = match (qself_ty.kind(), qself_res) {
            (_, Res::SelfTyAlias { alias_to: impl_def_id, is_trait_impl: true, .. }) => {
                // `Self` in an impl of a trait -- we have a concrete self type and a
                // trait reference.
                let Some(trait_ref) = tcx.impl_trait_ref(impl_def_id) else {
                    // A cycle error occurred, most likely.
                    self.dcx().span_bug(span, "expected cycle error");
                };

                self.probe_single_bound_for_assoc_item(
                    || {
                        traits::supertraits(
                            tcx,
                            ty::Binder::dummy(trait_ref.instantiate_identity()),
                        )
                    },
                    AssocItemQSelf::SelfTyAlias,
                    ty::AssocKind::Type,
                    assoc_ident,
                    span,
                    None,
                )?
            }
            (
                &ty::Param(_),
                Res::SelfTyParam { trait_: param_did } | Res::Def(DefKind::TyParam, param_did),
            ) => self.probe_single_ty_param_bound_for_assoc_item(
                param_did.expect_local(),
                qself.span,
                ty::AssocKind::Type,
                assoc_ident,
                span,
            )?,
            _ => {
                let reported = if variant_resolution.is_some() {
                    // Variant in type position
                    let msg = format!("expected type, found variant `{assoc_ident}`");
                    self.dcx().span_err(span, msg)
                } else if qself_ty.is_enum() {
                    let mut err = struct_span_code_err!(
                        self.dcx(),
                        assoc_ident.span,
                        E0599,
                        "no variant named `{}` found for enum `{}`",
                        assoc_ident,
                        qself_ty,
                    );

                    let adt_def = qself_ty.ty_adt_def().expect("enum is not an ADT");
                    if let Some(variant_name) = find_best_match_for_name(
                        &adt_def
                            .variants()
                            .iter()
                            .map(|variant| variant.name)
                            .collect::<Vec<Symbol>>(),
                        assoc_ident.name,
                        None,
                    ) && let Some(variant) =
                        adt_def.variants().iter().find(|s| s.name == variant_name)
                    {
                        let mut suggestion = vec![(assoc_ident.span, variant_name.to_string())];
                        if let hir::Node::Stmt(hir::Stmt {
                            kind: hir::StmtKind::Semi(ref expr),
                            ..
                        })
                        | hir::Node::Expr(ref expr) = tcx.parent_hir_node(hir_ref_id)
                            && let hir::ExprKind::Struct(..) = expr.kind
                        {
                            match variant.ctor {
                                None => {
                                    // struct
                                    suggestion = vec![(
                                        assoc_ident.span.with_hi(expr.span.hi()),
                                        if variant.fields.is_empty() {
                                            format!("{variant_name} {{}}")
                                        } else {
                                            format!(
                                                "{variant_name} {{ {} }}",
                                                variant
                                                    .fields
                                                    .iter()
                                                    .map(|f| format!("{}: /* value */", f.name))
                                                    .collect::<Vec<_>>()
                                                    .join(", ")
                                            )
                                        },
                                    )];
                                }
                                Some((hir::def::CtorKind::Fn, def_id)) => {
                                    // tuple
                                    let fn_sig = tcx.fn_sig(def_id).instantiate_identity();
                                    let inputs = fn_sig.inputs().skip_binder();
                                    suggestion = vec![(
                                        assoc_ident.span.with_hi(expr.span.hi()),
                                        format!(
                                            "{variant_name}({})",
                                            inputs
                                                .iter()
                                                .map(|i| format!("/* {i} */"))
                                                .collect::<Vec<_>>()
                                                .join(", ")
                                        ),
                                    )];
                                }
                                Some((hir::def::CtorKind::Const, _)) => {
                                    // unit
                                    suggestion = vec![(
                                        assoc_ident.span.with_hi(expr.span.hi()),
                                        variant_name.to_string(),
                                    )];
                                }
                            }
                        }
                        err.multipart_suggestion_verbose(
                            "there is a variant with a similar name",
                            suggestion,
                            Applicability::HasPlaceholders,
                        );
                    } else {
                        err.span_label(
                            assoc_ident.span,
                            format!("variant not found in `{qself_ty}`"),
                        );
                    }

                    if let Some(sp) = tcx.hir().span_if_local(adt_def.did()) {
                        err.span_label(sp, format!("variant `{assoc_ident}` not found here"));
                    }

                    err.emit()
                } else if let Err(reported) = qself_ty.error_reported() {
                    reported
                } else {
                    self.maybe_report_similar_assoc_fn(span, qself_ty, qself)?;

                    let traits: Vec<_> =
                        self.probe_traits_that_match_assoc_ty(qself_ty, assoc_ident);

                    // Don't print `ty::Error` to the user.
                    self.report_ambiguous_assoc_ty(
                        span,
                        &[qself_ty.to_string()],
                        &traits,
                        assoc_ident.name,
                    )
                };
                return Err(reported);
            }
        };

        let trait_did = bound.def_id();
        let assoc_ty = self
            .probe_assoc_item(assoc_ident, ty::AssocKind::Type, hir_ref_id, span, trait_did)
            .expect("failed to find associated type");
        let ty = self.lower_assoc_ty(span, assoc_ty.def_id, assoc_segment, bound);

        if let Some(variant_def_id) = variant_resolution {
            tcx.node_span_lint(AMBIGUOUS_ASSOCIATED_ITEMS, hir_ref_id, span, |lint| {
                lint.primary_message("ambiguous associated item");
                let mut could_refer_to = |kind: DefKind, def_id, also| {
                    let note_msg = format!(
                        "`{}` could{} refer to the {} defined here",
                        assoc_ident,
                        also,
                        tcx.def_kind_descr(kind, def_id)
                    );
                    lint.span_note(tcx.def_span(def_id), note_msg);
                };

                could_refer_to(DefKind::Variant, variant_def_id, "");
                could_refer_to(DefKind::AssocTy, assoc_ty.def_id, " also");

                lint.span_suggestion(
                    span,
                    "use fully-qualified syntax",
                    format!("<{} as {}>::{}", qself_ty, tcx.item_name(trait_did), assoc_ident),
                    Applicability::MachineApplicable,
                );
            });
        }
        Ok((ty, DefKind::AssocTy, assoc_ty.def_id))
    }

    fn probe_inherent_assoc_ty(
        &self,
        name: Ident,
        segment: &hir::PathSegment<'tcx>,
        adt_did: DefId,
        self_ty: Ty<'tcx>,
        block: HirId,
        span: Span,
    ) -> Result<Option<(Ty<'tcx>, DefId)>, ErrorGuaranteed> {
        let tcx = self.tcx();

        // Don't attempt to look up inherent associated types when the feature is not enabled.
        // Theoretically it'd be fine to do so since we feature-gate their definition site.
        // However, due to current limitations of the implementation (caused by us performing
        // selection during HIR ty lowering instead of in the trait solver), IATs can lead to cycle
        // errors (#108491) which mask the feature-gate error, needlessly confusing users
        // who use IATs by accident (#113265).
        if !tcx.features().inherent_associated_types() {
            return Ok(None);
        }

        let candidates: Vec<_> = tcx
            .inherent_impls(adt_did)
            .iter()
            .filter_map(|&impl_| {
                let (item, scope) =
                    self.probe_assoc_item_unchecked(name, ty::AssocKind::Type, block, impl_)?;
                Some((impl_, (item.def_id, scope)))
            })
            .collect();

        if candidates.is_empty() {
            return Ok(None);
        }

        //
        // Select applicable inherent associated type candidates modulo regions.
        //

        // In contexts that have no inference context, just make a new one.
        // We do need a local variable to store it, though.
        let infcx_;
        let infcx = match self.infcx() {
            Some(infcx) => infcx,
            None => {
                assert!(!self_ty.has_infer());
                infcx_ = tcx.infer_ctxt().ignoring_regions().build(TypingMode::non_body_analysis());
                &infcx_
            }
        };

        // FIXME(inherent_associated_types): Acquiring the ParamEnv this early leads to cycle errors
        // when inside of an ADT (#108491) or where clause.
        let param_env = tcx.param_env(block.owner);

        let mut universes = if self_ty.has_escaping_bound_vars() {
            vec![None; self_ty.outer_exclusive_binder().as_usize()]
        } else {
            vec![]
        };

        let (impl_, (assoc_item, def_scope)) = crate::traits::with_replaced_escaping_bound_vars(
            infcx,
            &mut universes,
            self_ty,
            |self_ty| {
                self.select_inherent_assoc_type_candidates(
                    infcx, name, span, self_ty, param_env, candidates,
                )
            },
        )?;

        self.check_assoc_item(assoc_item, name, def_scope, block, span);

        // FIXME(fmease): Currently creating throwaway `parent_args` to please
        // `lower_generic_args_of_assoc_item`. Modify the latter instead (or sth. similar) to
        // not require the parent args logic.
        let parent_args = ty::GenericArgs::identity_for_item(tcx, impl_);
        let args = self.lower_generic_args_of_assoc_item(span, assoc_item, segment, parent_args);
        let args = tcx.mk_args_from_iter(
            std::iter::once(ty::GenericArg::from(self_ty))
                .chain(args.into_iter().skip(parent_args.len())),
        );

        let ty =
            Ty::new_alias(tcx, ty::Inherent, ty::AliasTy::new_from_args(tcx, assoc_item, args));

        Ok(Some((ty, assoc_item)))
    }

    fn select_inherent_assoc_type_candidates(
        &self,
        infcx: &InferCtxt<'tcx>,
        name: Ident,
        span: Span,
        self_ty: Ty<'tcx>,
        param_env: ParamEnv<'tcx>,
        candidates: Vec<(DefId, (DefId, DefId))>,
    ) -> Result<(DefId, (DefId, DefId)), ErrorGuaranteed> {
        let tcx = self.tcx();
        let mut fulfillment_errors = Vec::new();

        let applicable_candidates: Vec<_> = candidates
            .iter()
            .copied()
            .filter(|&(impl_, _)| {
                infcx.probe(|_| {
                    let ocx = ObligationCtxt::new_with_diagnostics(infcx);
                    let self_ty = ocx.normalize(&ObligationCause::dummy(), param_env, self_ty);

                    let impl_args = infcx.fresh_args_for_item(span, impl_);
                    let impl_ty = tcx.type_of(impl_).instantiate(tcx, impl_args);
                    let impl_ty = ocx.normalize(&ObligationCause::dummy(), param_env, impl_ty);

                    // Check that the self types can be related.
                    if ocx.eq(&ObligationCause::dummy(), param_env, impl_ty, self_ty).is_err() {
                        return false;
                    }

                    // Check whether the impl imposes obligations we have to worry about.
                    let impl_bounds = tcx.predicates_of(impl_).instantiate(tcx, impl_args);
                    let impl_bounds =
                        ocx.normalize(&ObligationCause::dummy(), param_env, impl_bounds);
                    let impl_obligations = traits::predicates_for_generics(
                        |_, _| ObligationCause::dummy(),
                        param_env,
                        impl_bounds,
                    );
                    ocx.register_obligations(impl_obligations);

                    let mut errors = ocx.select_where_possible();
                    if !errors.is_empty() {
                        fulfillment_errors.append(&mut errors);
                        return false;
                    }

                    true
                })
            })
            .collect();

        match &applicable_candidates[..] {
            &[] => Err(self.complain_about_inherent_assoc_ty_not_found(
                name,
                self_ty,
                candidates,
                fulfillment_errors,
                span,
            )),

            &[applicable_candidate] => Ok(applicable_candidate),

            &[_, ..] => Err(self.complain_about_ambiguous_inherent_assoc_ty(
                name,
                applicable_candidates.into_iter().map(|(_, (candidate, _))| candidate).collect(),
                span,
            )),
        }
    }

    /// Given name and kind search for the assoc item in the provided scope and check if it's accessible[^1].
    ///
    /// [^1]: I.e., accessible in the provided scope wrt. visibility and stability.
    fn probe_assoc_item(
        &self,
        ident: Ident,
        kind: ty::AssocKind,
        block: HirId,
        span: Span,
        scope: DefId,
    ) -> Option<ty::AssocItem> {
        let (item, scope) = self.probe_assoc_item_unchecked(ident, kind, block, scope)?;
        self.check_assoc_item(item.def_id, ident, scope, block, span);
        Some(item)
    }

    /// Given name and kind search for the assoc item in the provided scope
    /// *without* checking if it's accessible[^1].
    ///
    /// [^1]: I.e., accessible in the provided scope wrt. visibility and stability.
    fn probe_assoc_item_unchecked(
        &self,
        ident: Ident,
        kind: ty::AssocKind,
        block: HirId,
        scope: DefId,
    ) -> Option<(ty::AssocItem, /*scope*/ DefId)> {
        let tcx = self.tcx();

        let (ident, def_scope) = tcx.adjust_ident_and_get_scope(ident, scope, block);
        // We have already adjusted the item name above, so compare with `.normalize_to_macros_2_0()`
        // instead of calling `filter_by_name_and_kind` which would needlessly normalize the
        // `ident` again and again.
        let item = tcx
            .associated_items(scope)
            .filter_by_name_unhygienic(ident.name)
            .find(|i| i.kind == kind && i.ident(tcx).normalize_to_macros_2_0() == ident)?;

        Some((*item, def_scope))
    }

    /// Check if the given assoc item is accessible in the provided scope wrt. visibility and stability.
    fn check_assoc_item(
        &self,
        item_def_id: DefId,
        ident: Ident,
        scope: DefId,
        block: HirId,
        span: Span,
    ) {
        let tcx = self.tcx();

        if !tcx.visibility(item_def_id).is_accessible_from(scope, tcx) {
            self.dcx().emit_err(crate::errors::AssocItemIsPrivate {
                span,
                kind: tcx.def_descr(item_def_id),
                name: ident,
                defined_here_label: tcx.def_span(item_def_id),
            });
        }

        tcx.check_stability(item_def_id, Some(block), span, None);
    }

    fn probe_traits_that_match_assoc_ty(
        &self,
        qself_ty: Ty<'tcx>,
        assoc_ident: Ident,
    ) -> Vec<String> {
        let tcx = self.tcx();

        // In contexts that have no inference context, just make a new one.
        // We do need a local variable to store it, though.
        let infcx_;
        let infcx = if let Some(infcx) = self.infcx() {
            infcx
        } else {
            assert!(!qself_ty.has_infer());
            infcx_ = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
            &infcx_
        };

        tcx.all_traits()
            .filter(|trait_def_id| {
                // Consider only traits with the associated type
                tcx.associated_items(*trait_def_id)
                        .in_definition_order()
                        .any(|i| {
                            i.kind.namespace() == Namespace::TypeNS
                                && i.ident(tcx).normalize_to_macros_2_0() == assoc_ident
                                && matches!(i.kind, ty::AssocKind::Type)
                        })
                    // Consider only accessible traits
                    && tcx.visibility(*trait_def_id)
                        .is_accessible_from(self.item_def_id(), tcx)
                    && tcx.all_impls(*trait_def_id)
                        .any(|impl_def_id| {
                            let impl_header = tcx.impl_trait_header(impl_def_id);
                            impl_header.is_some_and(|header| {
                                let trait_ref = header.trait_ref.instantiate(
                                    tcx,
                                    infcx.fresh_args_for_item(DUMMY_SP, impl_def_id),
                                );

                                let value = fold_regions(tcx, qself_ty, |_, _| tcx.lifetimes.re_erased);
                                // FIXME: Don't bother dealing with non-lifetime binders here...
                                if value.has_escaping_bound_vars() {
                                    return false;
                                }
                                infcx
                                    .can_eq(
                                        ty::ParamEnv::empty(),
                                        trait_ref.self_ty(),
                                        value,
                                    ) && header.polarity != ty::ImplPolarity::Negative
                            })
                        })
            })
            .map(|trait_def_id| tcx.def_path_str(trait_def_id))
            .collect()
    }

    /// Lower a qualified path to a type.
    #[instrument(level = "debug", skip_all)]
    fn lower_qpath(
        &self,
        span: Span,
        opt_self_ty: Option<Ty<'tcx>>,
        item_def_id: DefId,
        trait_segment: &hir::PathSegment<'tcx>,
        item_segment: &hir::PathSegment<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx();

        let trait_def_id = tcx.parent(item_def_id);
        debug!(?trait_def_id);

        let Some(self_ty) = opt_self_ty else {
            return self.error_missing_qpath_self_ty(trait_def_id, span, item_segment);
        };
        debug!(?self_ty);

        let trait_ref =
            self.lower_mono_trait_ref(span, trait_def_id, self_ty, trait_segment, false);
        debug!(?trait_ref);

        let item_args =
            self.lower_generic_args_of_assoc_item(span, item_def_id, item_segment, trait_ref.args);

        Ty::new_projection_from_args(tcx, item_def_id, item_args)
    }

    fn error_missing_qpath_self_ty(
        &self,
        trait_def_id: DefId,
        span: Span,
        item_segment: &hir::PathSegment<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx();
        let path_str = tcx.def_path_str(trait_def_id);

        let def_id = self.item_def_id();
        debug!(item_def_id = ?def_id);

        // FIXME: document why/how this is different from `tcx.local_parent(def_id)`
        let parent_def_id =
            tcx.hir().get_parent_item(tcx.local_def_id_to_hir_id(def_id)).to_def_id();
        debug!(?parent_def_id);

        // If the trait in segment is the same as the trait defining the item,
        // use the `<Self as ..>` syntax in the error.
        let is_part_of_self_trait_constraints = def_id.to_def_id() == trait_def_id;
        let is_part_of_fn_in_self_trait = parent_def_id == trait_def_id;

        let type_names = if is_part_of_self_trait_constraints || is_part_of_fn_in_self_trait {
            vec!["Self".to_string()]
        } else {
            // Find all the types that have an `impl` for the trait.
            tcx.all_impls(trait_def_id)
                .filter_map(|impl_def_id| tcx.impl_trait_header(impl_def_id))
                .filter(|header| {
                    // Consider only accessible traits
                    tcx.visibility(trait_def_id).is_accessible_from(self.item_def_id(), tcx)
                        && header.polarity != ty::ImplPolarity::Negative
                })
                .map(|header| header.trait_ref.instantiate_identity().self_ty())
                // We don't care about blanket impls.
                .filter(|self_ty| !self_ty.has_non_region_param())
                .map(|self_ty| tcx.erase_regions(self_ty).to_string())
                .collect()
        };
        // FIXME: also look at `tcx.generics_of(self.item_def_id()).params` any that
        // references the trait. Relevant for the first case in
        // `src/test/ui/associated-types/associated-types-in-ambiguous-context.rs`
        let reported =
            self.report_ambiguous_assoc_ty(span, &type_names, &[path_str], item_segment.ident.name);
        Ty::new_error(tcx, reported)
    }

    pub fn prohibit_generic_args<'a>(
        &self,
        segments: impl Iterator<Item = &'a hir::PathSegment<'a>> + Clone,
        err_extend: GenericsArgsErrExtend<'_>,
    ) -> Result<(), ErrorGuaranteed> {
        let args_visitors = segments.clone().flat_map(|segment| segment.args().args);
        let mut result = Ok(());
        if let Some(_) = args_visitors.clone().next() {
            result = Err(self.report_prohibit_generics_error(
                segments.clone(),
                args_visitors,
                err_extend,
            ));
        }

        for segment in segments {
            // Only emit the first error to avoid overloading the user with error messages.
            if let Some(c) = segment.args().constraints.first() {
                return Err(prohibit_assoc_item_constraint(self, c, None));
            }
        }

        result
    }

    /// Probe path segments that are semantically allowed to have generic arguments.
    ///
    /// ### Example
    ///
    /// ```ignore (illustrative)
    ///    Option::None::<()>
    /// //         ^^^^ permitted to have generic args
    ///
    /// // ==> [GenericPathSegment(Option_def_id, 1)]
    ///
    ///    Option::<()>::None
    /// // ^^^^^^        ^^^^ *not* permitted to have generic args
    /// // permitted to have generic args
    ///
    /// // ==> [GenericPathSegment(Option_def_id, 0)]
    /// ```
    // FIXME(eddyb, varkor) handle type paths here too, not just value ones.
    pub fn probe_generic_path_segments(
        &self,
        segments: &[hir::PathSegment<'_>],
        self_ty: Option<Ty<'tcx>>,
        kind: DefKind,
        def_id: DefId,
        span: Span,
    ) -> Vec<GenericPathSegment> {
        // We need to extract the generic arguments supplied by the user in
        // the path `path`. Due to the current setup, this is a bit of a
        // tricky process; the problem is that resolve only tells us the
        // end-point of the path resolution, and not the intermediate steps.
        // Luckily, we can (at least for now) deduce the intermediate steps
        // just from the end-point.
        //
        // There are basically five cases to consider:
        //
        // 1. Reference to a constructor of a struct:
        //
        //        struct Foo<T>(...)
        //
        //    In this case, the generic arguments are declared in the type space.
        //
        // 2. Reference to a constructor of an enum variant:
        //
        //        enum E<T> { Foo(...) }
        //
        //    In this case, the generic arguments are defined in the type space,
        //    but may be specified either on the type or the variant.
        //
        // 3. Reference to a free function or constant:
        //
        //        fn foo<T>() {}
        //
        //    In this case, the path will again always have the form
        //    `a::b::foo::<T>` where only the final segment should have generic
        //    arguments. However, in this case, those arguments are declared on
        //    a value, and hence are in the value space.
        //
        // 4. Reference to an associated function or constant:
        //
        //        impl<A> SomeStruct<A> {
        //            fn foo<B>(...) {}
        //        }
        //
        //    Here we can have a path like `a::b::SomeStruct::<A>::foo::<B>`,
        //    in which case generic arguments may appear in two places. The
        //    penultimate segment, `SomeStruct::<A>`, contains generic arguments
        //    in the type space, and the final segment, `foo::<B>` contains
        //    generic arguments in value space.
        //
        // The first step then is to categorize the segments appropriately.

        let tcx = self.tcx();

        assert!(!segments.is_empty());
        let last = segments.len() - 1;

        let mut generic_segments = vec![];

        match kind {
            // Case 1. Reference to a struct constructor.
            DefKind::Ctor(CtorOf::Struct, ..) => {
                // Everything but the final segment should have no
                // parameters at all.
                let generics = tcx.generics_of(def_id);
                // Variant and struct constructors use the
                // generics of their parent type definition.
                let generics_def_id = generics.parent.unwrap_or(def_id);
                generic_segments.push(GenericPathSegment(generics_def_id, last));
            }

            // Case 2. Reference to a variant constructor.
            DefKind::Ctor(CtorOf::Variant, ..) | DefKind::Variant => {
                let (generics_def_id, index) = if let Some(self_ty) = self_ty {
                    let adt_def = self.probe_adt(span, self_ty).unwrap();
                    debug_assert!(adt_def.is_enum());
                    (adt_def.did(), last)
                } else if last >= 1 && segments[last - 1].args.is_some() {
                    // Everything but the penultimate segment should have no
                    // parameters at all.
                    let mut def_id = def_id;

                    // `DefKind::Ctor` -> `DefKind::Variant`
                    if let DefKind::Ctor(..) = kind {
                        def_id = tcx.parent(def_id);
                    }

                    // `DefKind::Variant` -> `DefKind::Enum`
                    let enum_def_id = tcx.parent(def_id);
                    (enum_def_id, last - 1)
                } else {
                    // FIXME: lint here recommending `Enum::<...>::Variant` form
                    // instead of `Enum::Variant::<...>` form.

                    // Everything but the final segment should have no
                    // parameters at all.
                    let generics = tcx.generics_of(def_id);
                    // Variant and struct constructors use the
                    // generics of their parent type definition.
                    (generics.parent.unwrap_or(def_id), last)
                };
                generic_segments.push(GenericPathSegment(generics_def_id, index));
            }

            // Case 3. Reference to a top-level value.
            DefKind::Fn | DefKind::Const | DefKind::ConstParam | DefKind::Static { .. } => {
                generic_segments.push(GenericPathSegment(def_id, last));
            }

            // Case 4. Reference to a method or associated const.
            DefKind::AssocFn | DefKind::AssocConst => {
                if segments.len() >= 2 {
                    let generics = tcx.generics_of(def_id);
                    generic_segments.push(GenericPathSegment(generics.parent.unwrap(), last - 1));
                }
                generic_segments.push(GenericPathSegment(def_id, last));
            }

            kind => bug!("unexpected definition kind {:?} for {:?}", kind, def_id),
        }

        debug!(?generic_segments);

        generic_segments
    }

    /// Lower a type `Path` to a type.
    #[instrument(level = "debug", skip_all)]
    pub fn lower_path(
        &self,
        opt_self_ty: Option<Ty<'tcx>>,
        path: &hir::Path<'tcx>,
        hir_id: HirId,
        permit_variants: bool,
    ) -> Ty<'tcx> {
        debug!(?path.res, ?opt_self_ty, ?path.segments);
        let tcx = self.tcx();

        let span = path.span;
        match path.res {
            Res::Def(DefKind::OpaqueTy, did) => {
                // Check for desugared `impl Trait`.
                assert_matches!(tcx.opaque_ty_origin(did), hir::OpaqueTyOrigin::TyAlias { .. });
                let item_segment = path.segments.split_last().unwrap();
                let _ = self
                    .prohibit_generic_args(item_segment.1.iter(), GenericsArgsErrExtend::OpaqueTy);
                let args = self.lower_generic_args_of_path_segment(span, did, item_segment.0);
                Ty::new_opaque(tcx, did, args)
            }
            Res::Def(
                DefKind::Enum
                | DefKind::TyAlias
                | DefKind::Struct
                | DefKind::Union
                | DefKind::ForeignTy,
                did,
            ) => {
                assert_eq!(opt_self_ty, None);
                let _ = self.prohibit_generic_args(
                    path.segments.split_last().unwrap().1.iter(),
                    GenericsArgsErrExtend::None,
                );
                self.lower_path_segment(span, did, path.segments.last().unwrap())
            }
            Res::Def(kind @ DefKind::Variant, def_id) if permit_variants => {
                // Lower "variant type" as if it were a real type.
                // The resulting `Ty` is type of the variant's enum for now.
                assert_eq!(opt_self_ty, None);

                let generic_segments =
                    self.probe_generic_path_segments(path.segments, None, kind, def_id, span);
                let indices: FxHashSet<_> =
                    generic_segments.iter().map(|GenericPathSegment(_, index)| index).collect();
                let _ = self.prohibit_generic_args(
                    path.segments.iter().enumerate().filter_map(|(index, seg)| {
                        if !indices.contains(&index) { Some(seg) } else { None }
                    }),
                    GenericsArgsErrExtend::DefVariant,
                );

                let GenericPathSegment(def_id, index) = generic_segments.last().unwrap();
                self.lower_path_segment(span, *def_id, &path.segments[*index])
            }
            Res::Def(DefKind::TyParam, def_id) => {
                assert_eq!(opt_self_ty, None);
                let _ = self.prohibit_generic_args(
                    path.segments.iter(),
                    GenericsArgsErrExtend::Param(def_id),
                );
                self.lower_ty_param(hir_id)
            }
            Res::SelfTyParam { .. } => {
                // `Self` in trait or type alias.
                assert_eq!(opt_self_ty, None);
                let _ = self.prohibit_generic_args(
                    path.segments.iter(),
                    if let [hir::PathSegment { args: Some(args), ident, .. }] = &path.segments {
                        GenericsArgsErrExtend::SelfTyParam(
                            ident.span.shrink_to_hi().to(args.span_ext),
                        )
                    } else {
                        GenericsArgsErrExtend::None
                    },
                );
                tcx.types.self_param
            }
            Res::SelfTyAlias { alias_to: def_id, forbid_generic, .. } => {
                // `Self` in impl (we know the concrete type).
                assert_eq!(opt_self_ty, None);
                // Try to evaluate any array length constants.
                let ty = tcx.at(span).type_of(def_id).instantiate_identity();
                let _ = self.prohibit_generic_args(
                    path.segments.iter(),
                    GenericsArgsErrExtend::SelfTyAlias { def_id, span },
                );
                // HACK(min_const_generics): Forbid generic `Self` types
                // here as we can't easily do that during nameres.
                //
                // We do this before normalization as we otherwise allow
                // ```rust
                // trait AlwaysApplicable { type Assoc; }
                // impl<T: ?Sized> AlwaysApplicable for T { type Assoc = usize; }
                //
                // trait BindsParam<T> {
                //     type ArrayTy;
                // }
                // impl<T> BindsParam<T> for <T as AlwaysApplicable>::Assoc {
                //    type ArrayTy = [u8; Self::MAX];
                // }
                // ```
                // Note that the normalization happens in the param env of
                // the anon const, which is empty. This is why the
                // `AlwaysApplicable` impl needs a `T: ?Sized` bound for
                // this to compile if we were to normalize here.
                if forbid_generic && ty.has_param() {
                    let mut err = self.dcx().struct_span_err(
                        path.span,
                        "generic `Self` types are currently not permitted in anonymous constants",
                    );
                    if let Some(hir::Node::Item(&hir::Item {
                        kind: hir::ItemKind::Impl(impl_),
                        ..
                    })) = tcx.hir().get_if_local(def_id)
                    {
                        err.span_note(impl_.self_ty.span, "not a concrete type");
                    }
                    let reported = err.emit();
                    Ty::new_error(tcx, reported)
                } else {
                    ty
                }
            }
            Res::Def(DefKind::AssocTy, def_id) => {
                debug_assert!(path.segments.len() >= 2);
                let _ = self.prohibit_generic_args(
                    path.segments[..path.segments.len() - 2].iter(),
                    GenericsArgsErrExtend::None,
                );
                self.lower_qpath(
                    span,
                    opt_self_ty,
                    def_id,
                    &path.segments[path.segments.len() - 2],
                    path.segments.last().unwrap(),
                )
            }
            Res::PrimTy(prim_ty) => {
                assert_eq!(opt_self_ty, None);
                let _ = self.prohibit_generic_args(
                    path.segments.iter(),
                    GenericsArgsErrExtend::PrimTy(prim_ty),
                );
                match prim_ty {
                    hir::PrimTy::Bool => tcx.types.bool,
                    hir::PrimTy::Char => tcx.types.char,
                    hir::PrimTy::Int(it) => Ty::new_int(tcx, ty::int_ty(it)),
                    hir::PrimTy::Uint(uit) => Ty::new_uint(tcx, ty::uint_ty(uit)),
                    hir::PrimTy::Float(ft) => Ty::new_float(tcx, ty::float_ty(ft)),
                    hir::PrimTy::Str => tcx.types.str_,
                }
            }
            Res::Err => {
                let e = self
                    .tcx()
                    .dcx()
                    .span_delayed_bug(path.span, "path with `Res::Err` but no error emitted");
                Ty::new_error(tcx, e)
            }
            Res::Def(..) => {
                assert_eq!(
                    path.segments.get(0).map(|seg| seg.ident.name),
                    Some(kw::SelfUpper),
                    "only expected incorrect resolution for `Self`"
                );
                Ty::new_error(
                    self.tcx(),
                    self.dcx().span_delayed_bug(span, "incorrect resolution for `Self`"),
                )
            }
            _ => span_bug!(span, "unexpected resolution: {:?}", path.res),
        }
    }

    /// Lower a type parameter from the HIR to our internal notion of a type.
    ///
    /// Early-bound type parameters get lowered to [`ty::Param`]
    /// and late-bound ones to [`ty::Bound`].
    pub(crate) fn lower_ty_param(&self, hir_id: HirId) -> Ty<'tcx> {
        let tcx = self.tcx();
        match tcx.named_bound_var(hir_id) {
            Some(rbv::ResolvedArg::LateBound(debruijn, index, def_id)) => {
                let name = tcx.item_name(def_id.to_def_id());
                let br = ty::BoundTy {
                    var: ty::BoundVar::from_u32(index),
                    kind: ty::BoundTyKind::Param(def_id.to_def_id(), name),
                };
                Ty::new_bound(tcx, debruijn, br)
            }
            Some(rbv::ResolvedArg::EarlyBound(def_id)) => {
                let item_def_id = tcx.hir().ty_param_owner(def_id);
                let generics = tcx.generics_of(item_def_id);
                let index = generics.param_def_id_to_index[&def_id.to_def_id()];
                Ty::new_param(tcx, index, tcx.hir().ty_param_name(def_id))
            }
            Some(rbv::ResolvedArg::Error(guar)) => Ty::new_error(tcx, guar),
            arg => bug!("unexpected bound var resolution for {hir_id:?}: {arg:?}"),
        }
    }

    /// Lower a const parameter from the HIR to our internal notion of a constant.
    ///
    /// Early-bound const parameters get lowered to [`ty::ConstKind::Param`]
    /// and late-bound ones to [`ty::ConstKind::Bound`].
    pub(crate) fn lower_const_param(&self, param_def_id: DefId, path_hir_id: HirId) -> Const<'tcx> {
        let tcx = self.tcx();

        match tcx.named_bound_var(path_hir_id) {
            Some(rbv::ResolvedArg::EarlyBound(_)) => {
                // Find the name and index of the const parameter by indexing the generics of
                // the parent item and construct a `ParamConst`.
                let item_def_id = tcx.parent(param_def_id);
                let generics = tcx.generics_of(item_def_id);
                let index = generics.param_def_id_to_index[&param_def_id];
                let name = tcx.item_name(param_def_id);
                ty::Const::new_param(tcx, ty::ParamConst::new(index, name))
            }
            Some(rbv::ResolvedArg::LateBound(debruijn, index, _)) => {
                ty::Const::new_bound(tcx, debruijn, ty::BoundVar::from_u32(index))
            }
            Some(rbv::ResolvedArg::Error(guar)) => ty::Const::new_error(tcx, guar),
            arg => bug!("unexpected bound var resolution for {:?}: {arg:?}", path_hir_id),
        }
    }

    /// Convert a [`hir::ConstArg`] to a [`ty::Const`](Const).
    #[instrument(skip(self), level = "debug")]
    pub fn lower_const_arg(
        &self,
        const_arg: &hir::ConstArg<'tcx>,
        feed: FeedConstTy,
    ) -> Const<'tcx> {
        let tcx = self.tcx();

        if let FeedConstTy::Param(param_def_id) = feed
            && let hir::ConstArgKind::Anon(anon) = &const_arg.kind
        {
            tcx.feed_anon_const_type(anon.def_id, tcx.type_of(param_def_id));
        }

        let hir_id = const_arg.hir_id;
        match const_arg.kind {
            hir::ConstArgKind::Path(hir::QPath::Resolved(maybe_qself, path)) => {
                debug!(?maybe_qself, ?path);
                let opt_self_ty = maybe_qself.as_ref().map(|qself| self.lower_ty(qself));
                self.lower_const_path_resolved(opt_self_ty, path, hir_id)
            }
            hir::ConstArgKind::Path(qpath) => ty::Const::new_error_with_message(
                tcx,
                qpath.span(),
                format!("Const::lower_const_arg: invalid qpath {qpath:?}"),
            ),
            hir::ConstArgKind::Anon(anon) => self.lower_anon_const(anon),
            hir::ConstArgKind::Infer(span) => self.ct_infer(None, span),
        }
    }

    fn lower_const_path_resolved(
        &self,
        opt_self_ty: Option<Ty<'tcx>>,
        path: &hir::Path<'tcx>,
        hir_id: HirId,
    ) -> Const<'tcx> {
        let tcx = self.tcx();
        let span = path.span;
        match path.res {
            Res::Def(DefKind::ConstParam, def_id) => {
                assert_eq!(opt_self_ty, None);
                let _ = self.prohibit_generic_args(
                    path.segments.iter(),
                    GenericsArgsErrExtend::Param(def_id),
                );
                self.lower_const_param(def_id, hir_id)
            }
            Res::Def(DefKind::Const | DefKind::Ctor(_, CtorKind::Const), did) => {
                assert_eq!(opt_self_ty, None);
                let _ = self.prohibit_generic_args(
                    path.segments.split_last().unwrap().1.iter(),
                    GenericsArgsErrExtend::None,
                );
                let args = self.lower_generic_args_of_path_segment(
                    span,
                    did,
                    path.segments.last().unwrap(),
                );
                ty::Const::new_unevaluated(tcx, ty::UnevaluatedConst::new(did, args))
            }
            Res::Def(DefKind::Static { .. }, _) => {
                span_bug!(span, "use of bare `static` ConstArgKind::Path's not yet supported")
            }
            // FIXME(const_generics): create real const to allow fn items as const paths
            Res::Def(DefKind::Fn | DefKind::AssocFn, _) => ty::Const::new_error_with_message(
                tcx,
                span,
                "fn items cannot be used as const args",
            ),

            // Exhaustive match to be clear about what exactly we're considering to be
            // an invalid Res for a const path.
            Res::Def(
                DefKind::Mod
                | DefKind::Enum
                | DefKind::Variant
                | DefKind::Ctor(CtorOf::Variant, CtorKind::Fn)
                | DefKind::Struct
                | DefKind::Ctor(CtorOf::Struct, CtorKind::Fn)
                | DefKind::OpaqueTy
                | DefKind::TyAlias
                | DefKind::TraitAlias
                | DefKind::AssocTy
                | DefKind::Union
                | DefKind::Trait
                | DefKind::ForeignTy
                | DefKind::AssocConst
                | DefKind::TyParam
                | DefKind::Macro(_)
                | DefKind::LifetimeParam
                | DefKind::Use
                | DefKind::ForeignMod
                | DefKind::AnonConst
                | DefKind::InlineConst
                | DefKind::Field
                | DefKind::Impl { .. }
                | DefKind::Closure
                | DefKind::ExternCrate
                | DefKind::GlobalAsm
                | DefKind::SyntheticCoroutineBody,
                _,
            )
            | Res::PrimTy(_)
            | Res::SelfTyParam { .. }
            | Res::SelfTyAlias { .. }
            | Res::SelfCtor(_)
            | Res::Local(_)
            | Res::ToolMod
            | Res::NonMacroAttr(_)
            | Res::Err => Const::new_error_with_message(tcx, span, "invalid Res for const path"),
        }
    }

    /// Literals and const generic parameters are eagerly converted to a constant, everything else
    /// becomes `Unevaluated`.
    #[instrument(skip(self), level = "debug")]
    fn lower_anon_const(&self, anon: &AnonConst) -> Const<'tcx> {
        let tcx = self.tcx();

        let expr = &tcx.hir().body(anon.body).value;
        debug!(?expr);

        let ty = tcx
            .type_of(anon.def_id)
            .no_bound_vars()
            .expect("const parameter types cannot be generic");

        match self.try_lower_anon_const_lit(ty, expr) {
            Some(v) => v,
            None => ty::Const::new_unevaluated(tcx, ty::UnevaluatedConst {
                def: anon.def_id.to_def_id(),
                args: ty::GenericArgs::identity_for_item(tcx, anon.def_id.to_def_id()),
            }),
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn try_lower_anon_const_lit(
        &self,
        ty: Ty<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Option<Const<'tcx>> {
        let tcx = self.tcx();

        // Unwrap a block, so that e.g. `{ P }` is recognised as a parameter. Const arguments
        // currently have to be wrapped in curly brackets, so it's necessary to special-case.
        let expr = match &expr.kind {
            hir::ExprKind::Block(block, _) if block.stmts.is_empty() && block.expr.is_some() => {
                block.expr.as_ref().unwrap()
            }
            _ => expr,
        };

        if let hir::ExprKind::Path(hir::QPath::Resolved(
            _,
            &hir::Path { res: Res::Def(DefKind::ConstParam, _), .. },
        )) = expr.kind
        {
            span_bug!(
                expr.span,
                "try_lower_anon_const_lit: received const param which shouldn't be possible"
            );
        };

        let lit_input = match expr.kind {
            hir::ExprKind::Lit(lit) => Some(LitToConstInput { lit: &lit.node, ty, neg: false }),
            hir::ExprKind::Unary(hir::UnOp::Neg, expr) => match expr.kind {
                hir::ExprKind::Lit(lit) => Some(LitToConstInput { lit: &lit.node, ty, neg: true }),
                _ => None,
            },
            _ => None,
        };

        if let Some(lit_input) = lit_input {
            // If an error occurred, ignore that it's a literal and leave reporting the error up to
            // mir.
            match tcx.at(expr.span).lit_to_const(lit_input) {
                Ok(c) => return Some(c),
                Err(_) if lit_input.ty.has_aliases() => {
                    // allow the `ty` to be an alias type, though we cannot handle it here
                    return None;
                }
                Err(e) => {
                    tcx.dcx().span_delayed_bug(
                        expr.span,
                        format!("try_lower_anon_const_lit: couldn't lit_to_const {e:?}"),
                    );
                }
            }
        }

        None
    }

    fn lower_delegation_ty(&self, idx: hir::InferDelegationKind) -> Ty<'tcx> {
        let delegation_sig = self.tcx().inherit_sig_for_delegation_item(self.item_def_id());
        match idx {
            hir::InferDelegationKind::Input(idx) => delegation_sig[idx],
            hir::InferDelegationKind::Output => *delegation_sig.last().unwrap(),
        }
    }

    /// Lower a type from the HIR to our internal notion of a type.
    #[instrument(level = "debug", skip(self), ret)]
    pub fn lower_ty(&self, hir_ty: &hir::Ty<'tcx>) -> Ty<'tcx> {
        let tcx = self.tcx();

        let result_ty = match &hir_ty.kind {
            hir::TyKind::InferDelegation(_, idx) => self.lower_delegation_ty(*idx),
            hir::TyKind::Slice(ty) => Ty::new_slice(tcx, self.lower_ty(ty)),
            hir::TyKind::Ptr(mt) => Ty::new_ptr(tcx, self.lower_ty(mt.ty), mt.mutbl),
            hir::TyKind::Ref(region, mt) => {
                let r = self.lower_lifetime(region, RegionInferReason::Reference);
                debug!(?r);
                let t = self.lower_ty(mt.ty);
                Ty::new_ref(tcx, r, t, mt.mutbl)
            }
            hir::TyKind::Never => tcx.types.never,
            hir::TyKind::Tup(fields) => {
                Ty::new_tup_from_iter(tcx, fields.iter().map(|t| self.lower_ty(t)))
            }
            hir::TyKind::BareFn(bf) => {
                require_c_abi_if_c_variadic(tcx, bf.decl, bf.abi, hir_ty.span);

                Ty::new_fn_ptr(
                    tcx,
                    self.lower_fn_ty(hir_ty.hir_id, bf.safety, bf.abi, bf.decl, None, Some(hir_ty)),
                )
            }
            hir::TyKind::UnsafeBinder(_binder) => {
                let guar = self
                    .dcx()
                    .struct_span_err(hir_ty.span, "unsafe binders are not yet implemented")
                    .emit();
                Ty::new_error(tcx, guar)
            }
            hir::TyKind::TraitObject(bounds, lifetime, repr) => {
                if let Some(guar) = self.prohibit_or_lint_bare_trait_object_ty(hir_ty) {
                    // Don't continue with type analysis if the `dyn` keyword is missing
                    // It generates confusing errors, especially if the user meant to use another
                    // keyword like `impl`
                    Ty::new_error(tcx, guar)
                } else {
                    let repr = match repr {
                        TraitObjectSyntax::Dyn | TraitObjectSyntax::None => ty::Dyn,
                        TraitObjectSyntax::DynStar => ty::DynStar,
                    };
                    self.lower_trait_object_ty(hir_ty.span, hir_ty.hir_id, bounds, lifetime, repr)
                }
            }
            // If we encounter a fully qualified path with RTN generics, then it must have
            // *not* gone through `lower_ty_maybe_return_type_notation`, and therefore
            // it's certainly in an illegal position.
            hir::TyKind::Path(hir::QPath::Resolved(_, path))
                if path.segments.last().and_then(|segment| segment.args).is_some_and(|args| {
                    matches!(args.parenthesized, hir::GenericArgsParentheses::ReturnTypeNotation)
                }) =>
            {
                let guar = self.dcx().emit_err(BadReturnTypeNotation { span: hir_ty.span });
                Ty::new_error(tcx, guar)
            }
            hir::TyKind::Path(hir::QPath::Resolved(maybe_qself, path)) => {
                debug!(?maybe_qself, ?path);
                let opt_self_ty = maybe_qself.as_ref().map(|qself| self.lower_ty(qself));
                self.lower_path(opt_self_ty, path, hir_ty.hir_id, false)
            }
            &hir::TyKind::OpaqueDef(opaque_ty) => {
                // If this is an RPITIT and we are using the new RPITIT lowering scheme, we
                // generate the def_id of an associated type for the trait and return as
                // type a projection.
                let in_trait = match opaque_ty.origin {
                    hir::OpaqueTyOrigin::FnReturn {
                        in_trait_or_impl: Some(hir::RpitContext::Trait),
                        ..
                    }
                    | hir::OpaqueTyOrigin::AsyncFn {
                        in_trait_or_impl: Some(hir::RpitContext::Trait),
                        ..
                    } => true,
                    hir::OpaqueTyOrigin::FnReturn {
                        in_trait_or_impl: None | Some(hir::RpitContext::TraitImpl),
                        ..
                    }
                    | hir::OpaqueTyOrigin::AsyncFn {
                        in_trait_or_impl: None | Some(hir::RpitContext::TraitImpl),
                        ..
                    }
                    | hir::OpaqueTyOrigin::TyAlias { .. } => false,
                };

                self.lower_opaque_ty(opaque_ty.def_id, in_trait)
            }
            hir::TyKind::TraitAscription(hir_bounds) => {
                // Impl trait in bindings lower as an infer var with additional
                // set of type bounds.
                let self_ty = self.ty_infer(None, hir_ty.span);
                let mut bounds = Bounds::default();
                self.lower_bounds(
                    self_ty,
                    hir_bounds.iter(),
                    &mut bounds,
                    ty::List::empty(),
                    PredicateFilter::All,
                );
                self.register_trait_ascription_bounds(
                    bounds.clauses().collect(),
                    hir_ty.hir_id,
                    hir_ty.span,
                );
                self_ty
            }
            // If we encounter a type relative path with RTN generics, then it must have
            // *not* gone through `lower_ty_maybe_return_type_notation`, and therefore
            // it's certainly in an illegal position.
            hir::TyKind::Path(hir::QPath::TypeRelative(_, segment))
                if segment.args.is_some_and(|args| {
                    matches!(args.parenthesized, hir::GenericArgsParentheses::ReturnTypeNotation)
                }) =>
            {
                let guar = self.dcx().emit_err(BadReturnTypeNotation { span: hir_ty.span });
                Ty::new_error(tcx, guar)
            }
            hir::TyKind::Path(hir::QPath::TypeRelative(qself, segment)) => {
                debug!(?qself, ?segment);
                let ty = self.lower_ty(qself);
                self.lower_assoc_path(hir_ty.hir_id, hir_ty.span, ty, qself, segment, false)
                    .map(|(ty, _, _)| ty)
                    .unwrap_or_else(|guar| Ty::new_error(tcx, guar))
            }
            &hir::TyKind::Path(hir::QPath::LangItem(lang_item, span)) => {
                let def_id = tcx.require_lang_item(lang_item, Some(span));
                let (args, _) = self.lower_generic_args_of_path(
                    span,
                    def_id,
                    &[],
                    &hir::PathSegment::invalid(),
                    None,
                );
                tcx.at(span).type_of(def_id).instantiate(tcx, args)
            }
            hir::TyKind::Array(ty, length) => {
                let length = self.lower_const_arg(length, FeedConstTy::No);
                Ty::new_array_with_const_len(tcx, self.lower_ty(ty), length)
            }
            hir::TyKind::Typeof(e) => tcx.type_of(e.def_id).instantiate_identity(),
            hir::TyKind::Infer => {
                // Infer also appears as the type of arguments or return
                // values in an ExprKind::Closure, or as
                // the type of local variables. Both of these cases are
                // handled specially and will not descend into this routine.
                self.ty_infer(None, hir_ty.span)
            }
            hir::TyKind::Pat(ty, pat) => {
                let ty = self.lower_ty(ty);
                let pat_ty = match pat.kind {
                    hir::PatKind::Wild => {
                        let err = self.dcx().emit_err(WildPatTy { span: pat.span });
                        Ty::new_error(tcx, err)
                    }
                    hir::PatKind::Range(start, end, include_end) => {
                        let expr_to_const = |expr: &'tcx hir::Expr<'tcx>| -> ty::Const<'tcx> {
                            let (expr, neg) = match expr.kind {
                                hir::ExprKind::Unary(hir::UnOp::Neg, negated) => {
                                    (negated, Some((expr.hir_id, expr.span)))
                                }
                                _ => (expr, None),
                            };
                            let (c, c_ty) = match &expr.kind {
                                hir::ExprKind::Lit(lit) => {
                                    let lit_input =
                                        LitToConstInput { lit: &lit.node, ty, neg: neg.is_some() };
                                    let ct = match tcx.lit_to_const(lit_input) {
                                        Ok(c) => c,
                                        Err(LitToConstError::Reported(err)) => {
                                            ty::Const::new_error(tcx, err)
                                        }
                                        Err(LitToConstError::TypeError) => todo!(),
                                    };
                                    (ct, ty)
                                }

                                hir::ExprKind::Path(hir::QPath::Resolved(
                                    _,
                                    path @ &hir::Path {
                                        res: Res::Def(DefKind::ConstParam, def_id),
                                        ..
                                    },
                                )) => {
                                    let _ = self.prohibit_generic_args(
                                        path.segments.iter(),
                                        GenericsArgsErrExtend::Param(def_id),
                                    );
                                    let ty = tcx
                                        .type_of(def_id)
                                        .no_bound_vars()
                                        .expect("const parameter types cannot be generic");
                                    let ct = self.lower_const_param(def_id, expr.hir_id);
                                    (ct, ty)
                                }

                                _ => {
                                    let err = tcx
                                        .dcx()
                                        .emit_err(crate::errors::NonConstRange { span: expr.span });
                                    (ty::Const::new_error(tcx, err), Ty::new_error(tcx, err))
                                }
                            };
                            self.record_ty(expr.hir_id, c_ty, expr.span);
                            if let Some((id, span)) = neg {
                                self.record_ty(id, c_ty, span);
                            }
                            c
                        };

                        let start = start.map(expr_to_const);
                        let end = end.map(expr_to_const);

                        let include_end = match include_end {
                            hir::RangeEnd::Included => true,
                            hir::RangeEnd::Excluded => false,
                        };

                        let pat = tcx.mk_pat(ty::PatternKind::Range { start, end, include_end });
                        Ty::new_pat(tcx, ty, pat)
                    }
                    hir::PatKind::Err(e) => Ty::new_error(tcx, e),
                    _ => Ty::new_error_with_message(
                        tcx,
                        pat.span,
                        format!("unsupported pattern for pattern type: {pat:#?}"),
                    ),
                };
                self.record_ty(pat.hir_id, ty, pat.span);
                pat_ty
            }
            hir::TyKind::Err(guar) => Ty::new_error(tcx, *guar),
        };

        self.record_ty(hir_ty.hir_id, result_ty, hir_ty.span);
        result_ty
    }

    /// Lower an opaque type (i.e., an existential impl-Trait type) from the HIR.
    #[instrument(level = "debug", skip(self), ret)]
    fn lower_opaque_ty(&self, def_id: LocalDefId, in_trait: bool) -> Ty<'tcx> {
        let tcx = self.tcx();

        let lifetimes = tcx.opaque_captured_lifetimes(def_id);
        debug!(?lifetimes);

        // If this is an RPITIT and we are using the new RPITIT lowering scheme, we
        // generate the def_id of an associated type for the trait and return as
        // type a projection.
        let def_id = if in_trait {
            tcx.associated_type_for_impl_trait_in_trait(def_id).to_def_id()
        } else {
            def_id.to_def_id()
        };

        let generics = tcx.generics_of(def_id);
        debug!(?generics);

        // We use `generics.count() - lifetimes.len()` here instead of `generics.parent_count`
        // since return-position impl trait in trait squashes all of the generics from its source fn
        // into its own generics, so the opaque's "own" params isn't always just lifetimes.
        let offset = generics.count() - lifetimes.len();

        let args = ty::GenericArgs::for_item(tcx, def_id, |param, _| {
            if let Some(i) = (param.index as usize).checked_sub(offset) {
                let (lifetime, _) = lifetimes[i];
                self.lower_resolved_lifetime(lifetime).into()
            } else {
                tcx.mk_param_from_def(param)
            }
        });
        debug!(?args);

        if in_trait {
            Ty::new_projection_from_args(tcx, def_id, args)
        } else {
            Ty::new_opaque(tcx, def_id, args)
        }
    }

    pub fn lower_arg_ty(&self, ty: &hir::Ty<'tcx>, expected_ty: Option<Ty<'tcx>>) -> Ty<'tcx> {
        match ty.kind {
            hir::TyKind::Infer if let Some(expected_ty) = expected_ty => {
                self.record_ty(ty.hir_id, expected_ty, ty.span);
                expected_ty
            }
            _ => self.lower_ty(ty),
        }
    }

    /// Lower a function type from the HIR to our internal notion of a function signature.
    #[instrument(level = "debug", skip(self, hir_id, safety, abi, decl, generics, hir_ty), ret)]
    pub fn lower_fn_ty(
        &self,
        hir_id: HirId,
        safety: hir::Safety,
        abi: rustc_abi::ExternAbi,
        decl: &hir::FnDecl<'tcx>,
        generics: Option<&hir::Generics<'_>>,
        hir_ty: Option<&hir::Ty<'_>>,
    ) -> ty::PolyFnSig<'tcx> {
        let tcx = self.tcx();
        let bound_vars = tcx.late_bound_vars(hir_id);
        debug!(?bound_vars);

        let (input_tys, output_ty) = self.lower_fn_sig(decl, generics, hir_id, hir_ty);

        debug!(?output_ty);

        let fn_ty = tcx.mk_fn_sig(input_tys, output_ty, decl.c_variadic, safety, abi);
        let bare_fn_ty = ty::Binder::bind_with_vars(fn_ty, bound_vars);

        if let hir::Node::Ty(hir::Ty { kind: hir::TyKind::BareFn(bare_fn_ty), span, .. }) =
            tcx.hir_node(hir_id)
        {
            check_abi_fn_ptr(tcx, hir_id, *span, bare_fn_ty.abi);
        }

        // reject function types that violate cmse ABI requirements
        cmse::validate_cmse_abi(self.tcx(), self.dcx(), hir_id, abi, bare_fn_ty);

        // Find any late-bound regions declared in return type that do
        // not appear in the arguments. These are not well-formed.
        //
        // Example:
        //     for<'a> fn() -> &'a str <-- 'a is bad
        //     for<'a> fn(&'a String) -> &'a str <-- 'a is ok
        let inputs = bare_fn_ty.inputs();
        let late_bound_in_args =
            tcx.collect_constrained_late_bound_regions(inputs.map_bound(|i| i.to_owned()));
        let output = bare_fn_ty.output();
        let late_bound_in_ret = tcx.collect_referenced_late_bound_regions(output);

        self.validate_late_bound_regions(late_bound_in_args, late_bound_in_ret, |br_name| {
            struct_span_code_err!(
                self.dcx(),
                decl.output.span(),
                E0581,
                "return type references {}, which is not constrained by the fn input types",
                br_name
            )
        });

        bare_fn_ty
    }

    /// Given a fn_hir_id for a impl function, suggest the type that is found on the
    /// corresponding function in the trait that the impl implements, if it exists.
    /// If arg_idx is Some, then it corresponds to an input type index, otherwise it
    /// corresponds to the return type.
    pub(super) fn suggest_trait_fn_ty_for_impl_fn_infer(
        &self,
        fn_hir_id: HirId,
        arg_idx: Option<usize>,
    ) -> Option<Ty<'tcx>> {
        let tcx = self.tcx();
        let hir::Node::ImplItem(hir::ImplItem { kind: hir::ImplItemKind::Fn(..), ident, .. }) =
            tcx.hir_node(fn_hir_id)
        else {
            return None;
        };
        let i = tcx.parent_hir_node(fn_hir_id).expect_item().expect_impl();

        let trait_ref = self.lower_impl_trait_ref(i.of_trait.as_ref()?, self.lower_ty(i.self_ty));

        let assoc = tcx.associated_items(trait_ref.def_id).find_by_name_and_kind(
            tcx,
            *ident,
            ty::AssocKind::Fn,
            trait_ref.def_id,
        )?;

        let fn_sig = tcx.fn_sig(assoc.def_id).instantiate(
            tcx,
            trait_ref.args.extend_to(tcx, assoc.def_id, |param, _| tcx.mk_param_from_def(param)),
        );
        let fn_sig = tcx.liberate_late_bound_regions(fn_hir_id.expect_owner().to_def_id(), fn_sig);

        Some(if let Some(arg_idx) = arg_idx {
            *fn_sig.inputs().get(arg_idx)?
        } else {
            fn_sig.output()
        })
    }

    #[instrument(level = "trace", skip(self, generate_err))]
    fn validate_late_bound_regions<'cx>(
        &'cx self,
        constrained_regions: FxIndexSet<ty::BoundRegionKind>,
        referenced_regions: FxIndexSet<ty::BoundRegionKind>,
        generate_err: impl Fn(&str) -> Diag<'cx>,
    ) {
        for br in referenced_regions.difference(&constrained_regions) {
            let br_name = match *br {
                ty::BoundRegionKind::Named(_, kw::UnderscoreLifetime)
                | ty::BoundRegionKind::Anon
                | ty::BoundRegionKind::ClosureEnv => "an anonymous lifetime".to_string(),
                ty::BoundRegionKind::Named(_, name) => format!("lifetime `{name}`"),
            };

            let mut err = generate_err(&br_name);

            if let ty::BoundRegionKind::Named(_, kw::UnderscoreLifetime)
            | ty::BoundRegionKind::Anon = *br
            {
                // The only way for an anonymous lifetime to wind up
                // in the return type but **also** be unconstrained is
                // if it only appears in "associated types" in the
                // input. See #47511 and #62200 for examples. In this case,
                // though we can easily give a hint that ought to be
                // relevant.
                err.note(
                    "lifetimes appearing in an associated or opaque type are not considered constrained",
                );
                err.note("consider introducing a named lifetime parameter");
            }

            err.emit();
        }
    }

    /// Given the bounds on an object, determines what single region bound (if any) we can
    /// use to summarize this type.
    ///
    /// The basic idea is that we will use the bound the user
    /// provided, if they provided one, and otherwise search the supertypes of trait bounds
    /// for region bounds. It may be that we can derive no bound at all, in which case
    /// we return `None`.
    #[instrument(level = "debug", skip(self, span), ret)]
    fn compute_object_lifetime_bound(
        &self,
        span: Span,
        existential_predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) -> Option<ty::Region<'tcx>> // if None, use the default
    {
        let tcx = self.tcx();

        // No explicit region bound specified. Therefore, examine trait
        // bounds and see if we can derive region bounds from those.
        let derived_region_bounds = object_region_bounds(tcx, existential_predicates);

        // If there are no derived region bounds, then report back that we
        // can find no region bound. The caller will use the default.
        if derived_region_bounds.is_empty() {
            return None;
        }

        // If any of the derived region bounds are 'static, that is always
        // the best choice.
        if derived_region_bounds.iter().any(|r| r.is_static()) {
            return Some(tcx.lifetimes.re_static);
        }

        // Determine whether there is exactly one unique region in the set
        // of derived region bounds. If so, use that. Otherwise, report an
        // error.
        let r = derived_region_bounds[0];
        if derived_region_bounds[1..].iter().any(|r1| r != *r1) {
            self.dcx().emit_err(AmbiguousLifetimeBound { span });
        }
        Some(r)
    }
}
