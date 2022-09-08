//! Conversion from AST representation of types to the `ty.rs` representation.
//! The main routine here is `ast_ty_to_ty()`; each use is parameterized by an
//! instance of `AstConv`.

mod errors;
mod generics;

use crate::bounds::Bounds;
use crate::collect::HirPlaceholderCollector;
use crate::errors::{
    AmbiguousLifetimeBound, MultipleRelaxedDefaultBounds, TraitObjectDeclaredWithNoTraits,
    TypeofReservedKeywordUsed, ValueOfAssociatedStructAlreadySpecified,
};
use crate::middle::resolve_lifetime as rl;
use crate::require_c_abi_if_c_variadic;
use rustc_ast::TraitObjectSyntax;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{
    struct_span_err, Applicability, Diagnostic, DiagnosticBuilder, ErrorGuaranteed, FatalError,
    MultiSpan,
};
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Namespace, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{walk_generics, Visitor as _};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{GenericArg, GenericArgs, OpaqueTyOrigin};
use rustc_middle::middle::stability::AllowUnstable;
use rustc_middle::ty::subst::{self, GenericArgKind, InternalSubsts, Subst, SubstsRef};
use rustc_middle::ty::GenericParamDefKind;
use rustc_middle::ty::{
    self, Const, DefIdTree, EarlyBinder, IsSuggestable, Ty, TyCtxt, TypeVisitable,
};
use rustc_session::lint::builtin::{AMBIGUOUS_ASSOCIATED_ITEMS, BARE_TRAIT_OBJECTS};
use rustc_span::edition::Edition;
use rustc_span::lev_distance::find_best_match_for_name;
use rustc_span::symbol::{kw, Ident, Symbol};
use rustc_span::Span;
use rustc_target::spec::abi;
use rustc_trait_selection::traits;
use rustc_trait_selection::traits::astconv_object_safety_violations;
use rustc_trait_selection::traits::error_reporting::{
    report_object_safety_error, suggestions::NextTypeParamName,
};
use rustc_trait_selection::traits::wf::object_region_bounds;

use smallvec::{smallvec, SmallVec};
use std::collections::BTreeSet;
use std::slice;

#[derive(Debug)]
pub struct PathSeg(pub DefId, pub usize);

pub trait AstConv<'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'tcx>;

    fn item_def_id(&self) -> Option<DefId>;

    /// Returns predicates in scope of the form `X: Foo<T>`, where `X`
    /// is a type parameter `X` with the given id `def_id` and T
    /// matches `assoc_name`. This is a subset of the full set of
    /// predicates.
    ///
    /// This is used for one specific purpose: resolving "short-hand"
    /// associated type references like `T::Item`. In principle, we
    /// would do that by first getting the full set of predicates in
    /// scope and then filtering down to find those that apply to `T`,
    /// but this can lead to cycle errors. The problem is that we have
    /// to do this resolution *in order to create the predicates in
    /// the first place*. Hence, we have this "special pass".
    fn get_type_parameter_bounds(
        &self,
        span: Span,
        def_id: DefId,
        assoc_name: Ident,
    ) -> ty::GenericPredicates<'tcx>;

    /// Returns the lifetime to use when a lifetime is omitted (and not elided).
    fn re_infer(&self, param: Option<&ty::GenericParamDef>, span: Span)
    -> Option<ty::Region<'tcx>>;

    /// Returns the type to use when a type is omitted.
    fn ty_infer(&self, param: Option<&ty::GenericParamDef>, span: Span) -> Ty<'tcx>;

    /// Returns `true` if `_` is allowed in type signatures in the current context.
    fn allow_ty_infer(&self) -> bool;

    /// Returns the const to use when a const is omitted.
    fn ct_infer(
        &self,
        ty: Ty<'tcx>,
        param: Option<&ty::GenericParamDef>,
        span: Span,
    ) -> Const<'tcx>;

    /// Projecting an associated type from a (potentially)
    /// higher-ranked trait reference is more complicated, because of
    /// the possibility of late-bound regions appearing in the
    /// associated type binding. This is not legal in function
    /// signatures for that reason. In a function body, we can always
    /// handle it because we can use inference variables to remove the
    /// late-bound regions.
    fn projected_ty_from_poly_trait_ref(
        &self,
        span: Span,
        item_def_id: DefId,
        item_segment: &hir::PathSegment<'_>,
        poly_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Ty<'tcx>;

    /// Normalize an associated type coming from the user.
    fn normalize_ty(&self, span: Span, ty: Ty<'tcx>) -> Ty<'tcx>;

    /// Invoked when we encounter an error from some prior pass
    /// (e.g., resolve) that is translated into a ty-error. This is
    /// used to help suppress derived errors typeck might otherwise
    /// report.
    fn set_tainted_by_errors(&self);

    fn record_ty(&self, hir_id: hir::HirId, ty: Ty<'tcx>, span: Span);
}

#[derive(Debug)]
struct ConvertedBinding<'a, 'tcx> {
    hir_id: hir::HirId,
    item_name: Ident,
    kind: ConvertedBindingKind<'a, 'tcx>,
    gen_args: &'a GenericArgs<'a>,
    span: Span,
}

#[derive(Debug)]
enum ConvertedBindingKind<'a, 'tcx> {
    Equality(ty::Term<'tcx>),
    Constraint(&'a [hir::GenericBound<'a>]),
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
#[derive(Clone, Default, Debug)]
pub struct GenericArgCountMismatch {
    /// Indicates whether a fatal error was reported (`Some`), or just a lint (`None`).
    pub reported: Option<ErrorGuaranteed>,
    /// A list of spans of arguments provided that were not valid.
    pub invalid_args: Vec<Span>,
}

/// Decorates the result of a generic argument count mismatch
/// check with whether explicit late bounds were provided.
#[derive(Clone, Debug)]
pub struct GenericArgCountResult {
    pub explicit_late_bound: ExplicitLateBound,
    pub correct: Result<(), GenericArgCountMismatch>,
}

pub trait CreateSubstsForGenericArgsCtxt<'a, 'tcx> {
    fn args_for_def_id(&mut self, def_id: DefId) -> (Option<&'a GenericArgs<'a>>, bool);

    fn provided_kind(
        &mut self,
        param: &ty::GenericParamDef,
        arg: &GenericArg<'_>,
    ) -> subst::GenericArg<'tcx>;

    fn inferred_kind(
        &mut self,
        substs: Option<&[subst::GenericArg<'tcx>]>,
        param: &ty::GenericParamDef,
        infer_args: bool,
    ) -> subst::GenericArg<'tcx>;
}

impl<'o, 'tcx> dyn AstConv<'tcx> + 'o {
    #[instrument(level = "debug", skip(self), ret)]
    pub fn ast_region_to_region(
        &self,
        lifetime: &hir::Lifetime,
        def: Option<&ty::GenericParamDef>,
    ) -> ty::Region<'tcx> {
        let tcx = self.tcx();
        let lifetime_name = |def_id| tcx.hir().name(tcx.hir().local_def_id_to_hir_id(def_id));

        match tcx.named_region(lifetime.hir_id) {
            Some(rl::Region::Static) => tcx.lifetimes.re_static,

            Some(rl::Region::LateBound(debruijn, index, def_id)) => {
                let name = lifetime_name(def_id.expect_local());
                let br = ty::BoundRegion {
                    var: ty::BoundVar::from_u32(index),
                    kind: ty::BrNamed(def_id, name),
                };
                tcx.mk_region(ty::ReLateBound(debruijn, br))
            }

            Some(rl::Region::EarlyBound(def_id)) => {
                let name = tcx.hir().ty_param_name(def_id.expect_local());
                let item_def_id = tcx.hir().ty_param_owner(def_id.expect_local());
                let generics = tcx.generics_of(item_def_id);
                let index = generics.param_def_id_to_index[&def_id];
                tcx.mk_region(ty::ReEarlyBound(ty::EarlyBoundRegion { def_id, index, name }))
            }

            Some(rl::Region::Free(scope, id)) => {
                let name = lifetime_name(id.expect_local());
                tcx.mk_region(ty::ReFree(ty::FreeRegion {
                    scope,
                    bound_region: ty::BrNamed(id, name),
                }))

                // (*) -- not late-bound, won't change
            }

            None => {
                self.re_infer(def, lifetime.span).unwrap_or_else(|| {
                    debug!(?lifetime, "unelided lifetime in signature");

                    // This indicates an illegal lifetime
                    // elision. `resolve_lifetime` should have
                    // reported an error in this case -- but if
                    // not, let's error out.
                    tcx.sess.delay_span_bug(lifetime.span, "unelided lifetime in signature");

                    // Supply some dummy value. We don't have an
                    // `re_error`, annoyingly, so use `'static`.
                    tcx.lifetimes.re_static
                })
            }
        }
    }

    /// Given a path `path` that refers to an item `I` with the declared generics `decl_generics`,
    /// returns an appropriate set of substitutions for this particular reference to `I`.
    pub fn ast_path_substs_for_ty(
        &self,
        span: Span,
        def_id: DefId,
        item_segment: &hir::PathSegment<'_>,
    ) -> SubstsRef<'tcx> {
        let (substs, _) = self.create_substs_for_ast_path(
            span,
            def_id,
            &[],
            item_segment,
            item_segment.args(),
            item_segment.infer_args,
            None,
        );
        let assoc_bindings = self.create_assoc_bindings_for_generic_args(item_segment.args());

        if let Some(b) = assoc_bindings.first() {
            Self::prohibit_assoc_ty_binding(self.tcx(), b.span);
        }

        substs
    }

    /// Given the type/lifetime/const arguments provided to some path (along with
    /// an implicit `Self`, if this is a trait reference), returns the complete
    /// set of substitutions. This may involve applying defaulted type parameters.
    /// Constraints on associated types are created from `create_assoc_bindings_for_generic_args`.
    ///
    /// Example:
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
    ///    parameters are returned in the `SubstsRef`, the associated type bindings like
    ///    `Output = u32` are returned from `create_assoc_bindings_for_generic_args`.
    ///
    /// Note that the type listing given here is *exactly* what the user provided.
    ///
    /// For (generic) associated types
    ///
    /// ```ignore (illustrative)
    /// <Vec<u8> as Iterable<u8>>::Iter::<'a>
    /// ```
    ///
    /// We have the parent substs are the substs for the parent trait:
    /// `[Vec<u8>, u8]` and `generic_args` are the arguments for the associated
    /// type itself: `['a]`. The returned `SubstsRef` concatenates these two
    /// lists: `[Vec<u8>, u8, 'a]`.
    #[instrument(level = "debug", skip(self, span), ret)]
    fn create_substs_for_ast_path<'a>(
        &self,
        span: Span,
        def_id: DefId,
        parent_substs: &[subst::GenericArg<'tcx>],
        seg: &hir::PathSegment<'_>,
        generic_args: &'a hir::GenericArgs<'_>,
        infer_args: bool,
        self_ty: Option<Ty<'tcx>>,
    ) -> (SubstsRef<'tcx>, GenericArgCountResult) {
        // If the type is parameterized by this region, then replace this
        // region with the current anon region binding (in other words,
        // whatever & would get replaced with).

        let tcx = self.tcx();
        let generics = tcx.generics_of(def_id);
        debug!("generics: {:?}", generics);

        if generics.has_self {
            if generics.parent.is_some() {
                // The parent is a trait so it should have at least one subst
                // for the `Self` type.
                assert!(!parent_substs.is_empty())
            } else {
                // This item (presumably a trait) needs a self-type.
                assert!(self_ty.is_some());
            }
        } else {
            assert!(self_ty.is_none() && parent_substs.is_empty());
        }

        let arg_count = Self::check_generic_arg_count(
            tcx,
            span,
            def_id,
            seg,
            generics,
            generic_args,
            GenericArgPosition::Type,
            self_ty.is_some(),
            infer_args,
        );

        // Skip processing if type has no generic parameters.
        // Traits always have `Self` as a generic parameter, which means they will not return early
        // here and so associated type bindings will be handled regardless of whether there are any
        // non-`Self` generic parameters.
        if generics.params.is_empty() {
            return (tcx.intern_substs(&[]), arg_count);
        }

        struct SubstsForAstPathCtxt<'a, 'tcx> {
            astconv: &'a (dyn AstConv<'tcx> + 'a),
            def_id: DefId,
            generic_args: &'a GenericArgs<'a>,
            span: Span,
            inferred_params: Vec<Span>,
            infer_args: bool,
        }

        impl<'a, 'tcx> CreateSubstsForGenericArgsCtxt<'a, 'tcx> for SubstsForAstPathCtxt<'a, 'tcx> {
            fn args_for_def_id(&mut self, did: DefId) -> (Option<&'a GenericArgs<'a>>, bool) {
                if did == self.def_id {
                    (Some(self.generic_args), self.infer_args)
                } else {
                    // The last component of this tuple is unimportant.
                    (None, false)
                }
            }

            fn provided_kind(
                &mut self,
                param: &ty::GenericParamDef,
                arg: &GenericArg<'_>,
            ) -> subst::GenericArg<'tcx> {
                let tcx = self.astconv.tcx();

                let mut handle_ty_args = |has_default, ty: &hir::Ty<'_>| {
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
                    if let (hir::TyKind::Infer, false) = (&ty.kind, self.astconv.allow_ty_infer()) {
                        self.inferred_params.push(ty.span);
                        tcx.ty_error().into()
                    } else {
                        self.astconv.ast_ty_to_ty(ty).into()
                    }
                };

                match (&param.kind, arg) {
                    (GenericParamDefKind::Lifetime, GenericArg::Lifetime(lt)) => {
                        self.astconv.ast_region_to_region(lt, Some(param)).into()
                    }
                    (&GenericParamDefKind::Type { has_default, .. }, GenericArg::Type(ty)) => {
                        handle_ty_args(has_default, ty)
                    }
                    (&GenericParamDefKind::Type { has_default, .. }, GenericArg::Infer(inf)) => {
                        handle_ty_args(has_default, &inf.to_ty())
                    }
                    (GenericParamDefKind::Const { .. }, GenericArg::Const(ct)) => {
                        ty::Const::from_opt_const_arg_anon_const(
                            tcx,
                            ty::WithOptConstParam {
                                did: tcx.hir().local_def_id(ct.value.hir_id),
                                const_param_did: Some(param.def_id),
                            },
                        )
                        .into()
                    }
                    (&GenericParamDefKind::Const { .. }, hir::GenericArg::Infer(inf)) => {
                        let ty = tcx.at(self.span).type_of(param.def_id);
                        if self.astconv.allow_ty_infer() {
                            self.astconv.ct_infer(ty, Some(param), inf.span).into()
                        } else {
                            self.inferred_params.push(inf.span);
                            tcx.const_error(ty).into()
                        }
                    }
                    _ => unreachable!(),
                }
            }

            fn inferred_kind(
                &mut self,
                substs: Option<&[subst::GenericArg<'tcx>]>,
                param: &ty::GenericParamDef,
                infer_args: bool,
            ) -> subst::GenericArg<'tcx> {
                let tcx = self.astconv.tcx();
                match param.kind {
                    GenericParamDefKind::Lifetime => self
                        .astconv
                        .re_infer(Some(param), self.span)
                        .unwrap_or_else(|| {
                            debug!(?param, "unelided lifetime in signature");

                            // This indicates an illegal lifetime in a non-assoc-trait position
                            tcx.sess.delay_span_bug(self.span, "unelided lifetime in signature");

                            // Supply some dummy value. We don't have an
                            // `re_error`, annoyingly, so use `'static`.
                            tcx.lifetimes.re_static
                        })
                        .into(),
                    GenericParamDefKind::Type { has_default, .. } => {
                        if !infer_args && has_default {
                            // No type parameter provided, but a default exists.
                            let substs = substs.unwrap();
                            if substs.iter().any(|arg| match arg.unpack() {
                                GenericArgKind::Type(ty) => ty.references_error(),
                                _ => false,
                            }) {
                                // Avoid ICE #86756 when type error recovery goes awry.
                                return tcx.ty_error().into();
                            }
                            self.astconv
                                .normalize_ty(
                                    self.span,
                                    EarlyBinder(tcx.at(self.span).type_of(param.def_id))
                                        .subst(tcx, substs),
                                )
                                .into()
                        } else if infer_args {
                            self.astconv.ty_infer(Some(param), self.span).into()
                        } else {
                            // We've already errored above about the mismatch.
                            tcx.ty_error().into()
                        }
                    }
                    GenericParamDefKind::Const { has_default } => {
                        let ty = tcx.at(self.span).type_of(param.def_id);
                        if !infer_args && has_default {
                            tcx.bound_const_param_default(param.def_id)
                                .subst(tcx, substs.unwrap())
                                .into()
                        } else {
                            if infer_args {
                                self.astconv.ct_infer(ty, Some(param), self.span).into()
                            } else {
                                // We've already errored above about the mismatch.
                                tcx.const_error(ty).into()
                            }
                        }
                    }
                }
            }
        }

        let mut substs_ctx = SubstsForAstPathCtxt {
            astconv: self,
            def_id,
            span,
            generic_args,
            inferred_params: vec![],
            infer_args,
        };
        let substs = Self::create_substs_for_generic_args(
            tcx,
            def_id,
            parent_substs,
            self_ty.is_some(),
            self_ty,
            &arg_count,
            &mut substs_ctx,
        );

        (substs, arg_count)
    }

    fn create_assoc_bindings_for_generic_args<'a>(
        &self,
        generic_args: &'a hir::GenericArgs<'_>,
    ) -> Vec<ConvertedBinding<'a, 'tcx>> {
        // Convert associated-type bindings or constraints into a separate vector.
        // Example: Given this:
        //
        //     T: Iterator<Item = u32>
        //
        // The `T` is passed in as a self-type; the `Item = u32` is
        // not a "type parameter" of the `Iterator` trait, but rather
        // a restriction on `<T as Iterator>::Item`, so it is passed
        // back separately.
        let assoc_bindings = generic_args
            .bindings
            .iter()
            .map(|binding| {
                let kind = match binding.kind {
                    hir::TypeBindingKind::Equality { ref term } => match term {
                        hir::Term::Ty(ref ty) => {
                            ConvertedBindingKind::Equality(self.ast_ty_to_ty(ty).into())
                        }
                        hir::Term::Const(ref c) => {
                            let local_did = self.tcx().hir().local_def_id(c.hir_id);
                            let c = Const::from_anon_const(self.tcx(), local_did);
                            ConvertedBindingKind::Equality(c.into())
                        }
                    },
                    hir::TypeBindingKind::Constraint { ref bounds } => {
                        ConvertedBindingKind::Constraint(bounds)
                    }
                };
                ConvertedBinding {
                    hir_id: binding.hir_id,
                    item_name: binding.ident,
                    kind,
                    gen_args: binding.gen_args,
                    span: binding.span,
                }
            })
            .collect();

        assoc_bindings
    }

    pub(crate) fn create_substs_for_associated_item(
        &self,
        tcx: TyCtxt<'tcx>,
        span: Span,
        item_def_id: DefId,
        item_segment: &hir::PathSegment<'_>,
        parent_substs: SubstsRef<'tcx>,
    ) -> SubstsRef<'tcx> {
        debug!(
            "create_substs_for_associated_item(span: {:?}, item_def_id: {:?}, item_segment: {:?}",
            span, item_def_id, item_segment
        );
        if tcx.generics_of(item_def_id).params.is_empty() {
            self.prohibit_generics(slice::from_ref(item_segment).iter(), |_| {});

            parent_substs
        } else {
            self.create_substs_for_ast_path(
                span,
                item_def_id,
                parent_substs,
                item_segment,
                item_segment.args(),
                item_segment.infer_args,
                None,
            )
            .0
        }
    }

    /// Instantiates the path for the given trait reference, assuming that it's
    /// bound to a valid trait type. Returns the `DefId` of the defining trait.
    /// The type _cannot_ be a type other than a trait type.
    ///
    /// If the `projections` argument is `None`, then assoc type bindings like `Foo<T = X>`
    /// are disallowed. Otherwise, they are pushed onto the vector given.
    pub fn instantiate_mono_trait_ref(
        &self,
        trait_ref: &hir::TraitRef<'_>,
        self_ty: Ty<'tcx>,
    ) -> ty::TraitRef<'tcx> {
        self.prohibit_generics(trait_ref.path.segments.split_last().unwrap().1.iter(), |_| {});

        self.ast_path_to_mono_trait_ref(
            trait_ref.path.span,
            trait_ref.trait_def_id().unwrap_or_else(|| FatalError.raise()),
            self_ty,
            trait_ref.path.segments.last().unwrap(),
            true,
        )
    }

    fn instantiate_poly_trait_ref_inner(
        &self,
        hir_id: hir::HirId,
        span: Span,
        binding_span: Option<Span>,
        constness: ty::BoundConstness,
        bounds: &mut Bounds<'tcx>,
        speculative: bool,
        trait_ref_span: Span,
        trait_def_id: DefId,
        trait_segment: &hir::PathSegment<'_>,
        args: &GenericArgs<'_>,
        infer_args: bool,
        self_ty: Ty<'tcx>,
    ) -> GenericArgCountResult {
        let (substs, arg_count) = self.create_substs_for_ast_path(
            trait_ref_span,
            trait_def_id,
            &[],
            trait_segment,
            args,
            infer_args,
            Some(self_ty),
        );

        let tcx = self.tcx();
        let bound_vars = tcx.late_bound_vars(hir_id);
        debug!(?bound_vars);

        let assoc_bindings = self.create_assoc_bindings_for_generic_args(args);

        let poly_trait_ref =
            ty::Binder::bind_with_vars(ty::TraitRef::new(trait_def_id, substs), bound_vars);

        debug!(?poly_trait_ref, ?assoc_bindings);
        bounds.trait_bounds.push((poly_trait_ref, span, constness));

        let mut dup_bindings = FxHashMap::default();
        for binding in &assoc_bindings {
            // Specify type to assert that error was already reported in `Err` case.
            let _: Result<_, ErrorGuaranteed> = self.add_predicates_for_ast_type_binding(
                hir_id,
                poly_trait_ref,
                binding,
                bounds,
                speculative,
                &mut dup_bindings,
                binding_span.unwrap_or(binding.span),
            );
            // Okay to ignore `Err` because of `ErrorGuaranteed` (see above).
        }

        arg_count
    }

    /// Given a trait bound like `Debug`, applies that trait bound the given self-type to construct
    /// a full trait reference. The resulting trait reference is returned. This may also generate
    /// auxiliary bounds, which are added to `bounds`.
    ///
    /// Example:
    ///
    /// ```ignore (illustrative)
    /// poly_trait_ref = Iterator<Item = u32>
    /// self_ty = Foo
    /// ```
    ///
    /// this would return `Foo: Iterator` and add `<Foo as Iterator>::Item = u32` into `bounds`.
    ///
    /// **A note on binders:** against our usual convention, there is an implied bounder around
    /// the `self_ty` and `poly_trait_ref` parameters here. So they may reference bound regions.
    /// If for example you had `for<'a> Foo<'a>: Bar<'a>`, then the `self_ty` would be `Foo<'a>`
    /// where `'a` is a bound region at depth 0. Similarly, the `poly_trait_ref` would be
    /// `Bar<'a>`. The returned poly-trait-ref will have this binder instantiated explicitly,
    /// however.
    #[instrument(level = "debug", skip(self, span, constness, bounds, speculative))]
    pub(crate) fn instantiate_poly_trait_ref(
        &self,
        trait_ref: &hir::TraitRef<'_>,
        span: Span,
        constness: ty::BoundConstness,
        self_ty: Ty<'tcx>,
        bounds: &mut Bounds<'tcx>,
        speculative: bool,
    ) -> GenericArgCountResult {
        let hir_id = trait_ref.hir_ref_id;
        let binding_span = None;
        let trait_ref_span = trait_ref.path.span;
        let trait_def_id = trait_ref.trait_def_id().unwrap_or_else(|| FatalError.raise());
        let trait_segment = trait_ref.path.segments.last().unwrap();
        let args = trait_segment.args();
        let infer_args = trait_segment.infer_args;

        self.prohibit_generics(trait_ref.path.segments.split_last().unwrap().1.iter(), |_| {});
        self.complain_about_internal_fn_trait(span, trait_def_id, trait_segment, false);

        self.instantiate_poly_trait_ref_inner(
            hir_id,
            span,
            binding_span,
            constness,
            bounds,
            speculative,
            trait_ref_span,
            trait_def_id,
            trait_segment,
            args,
            infer_args,
            self_ty,
        )
    }

    pub(crate) fn instantiate_lang_item_trait_ref(
        &self,
        lang_item: hir::LangItem,
        span: Span,
        hir_id: hir::HirId,
        args: &GenericArgs<'_>,
        self_ty: Ty<'tcx>,
        bounds: &mut Bounds<'tcx>,
    ) {
        let binding_span = Some(span);
        let constness = ty::BoundConstness::NotConst;
        let speculative = false;
        let trait_ref_span = span;
        let trait_def_id = self.tcx().require_lang_item(lang_item, Some(span));
        let trait_segment = &hir::PathSegment::invalid();
        let infer_args = false;

        self.instantiate_poly_trait_ref_inner(
            hir_id,
            span,
            binding_span,
            constness,
            bounds,
            speculative,
            trait_ref_span,
            trait_def_id,
            trait_segment,
            args,
            infer_args,
            self_ty,
        );
    }

    fn ast_path_to_mono_trait_ref(
        &self,
        span: Span,
        trait_def_id: DefId,
        self_ty: Ty<'tcx>,
        trait_segment: &hir::PathSegment<'_>,
        is_impl: bool,
    ) -> ty::TraitRef<'tcx> {
        let (substs, _) = self.create_substs_for_ast_trait_ref(
            span,
            trait_def_id,
            self_ty,
            trait_segment,
            is_impl,
        );
        let assoc_bindings = self.create_assoc_bindings_for_generic_args(trait_segment.args());
        if let Some(b) = assoc_bindings.first() {
            Self::prohibit_assoc_ty_binding(self.tcx(), b.span);
        }
        ty::TraitRef::new(trait_def_id, substs)
    }

    #[instrument(level = "debug", skip(self, span))]
    fn create_substs_for_ast_trait_ref<'a>(
        &self,
        span: Span,
        trait_def_id: DefId,
        self_ty: Ty<'tcx>,
        trait_segment: &'a hir::PathSegment<'a>,
        is_impl: bool,
    ) -> (SubstsRef<'tcx>, GenericArgCountResult) {
        self.complain_about_internal_fn_trait(span, trait_def_id, trait_segment, is_impl);

        self.create_substs_for_ast_path(
            span,
            trait_def_id,
            &[],
            trait_segment,
            trait_segment.args(),
            trait_segment.infer_args,
            Some(self_ty),
        )
    }

    fn trait_defines_associated_type_named(&self, trait_def_id: DefId, assoc_name: Ident) -> bool {
        self.tcx()
            .associated_items(trait_def_id)
            .find_by_name_and_kind(self.tcx(), assoc_name, ty::AssocKind::Type, trait_def_id)
            .is_some()
    }
    fn trait_defines_associated_const_named(&self, trait_def_id: DefId, assoc_name: Ident) -> bool {
        self.tcx()
            .associated_items(trait_def_id)
            .find_by_name_and_kind(self.tcx(), assoc_name, ty::AssocKind::Const, trait_def_id)
            .is_some()
    }

    // Sets `implicitly_sized` to true on `Bounds` if necessary
    pub(crate) fn add_implicitly_sized<'hir>(
        &self,
        bounds: &mut Bounds<'hir>,
        ast_bounds: &'hir [hir::GenericBound<'hir>],
        self_ty_where_predicates: Option<(hir::HirId, &'hir [hir::WherePredicate<'hir>])>,
        span: Span,
    ) {
        let tcx = self.tcx();

        // Try to find an unbound in bounds.
        let mut unbound = None;
        let mut search_bounds = |ast_bounds: &'hir [hir::GenericBound<'hir>]| {
            for ab in ast_bounds {
                if let hir::GenericBound::Trait(ptr, hir::TraitBoundModifier::Maybe) = ab {
                    if unbound.is_none() {
                        unbound = Some(&ptr.trait_ref);
                    } else {
                        tcx.sess.emit_err(MultipleRelaxedDefaultBounds { span });
                    }
                }
            }
        };
        search_bounds(ast_bounds);
        if let Some((self_ty, where_clause)) = self_ty_where_predicates {
            let self_ty_def_id = tcx.hir().local_def_id(self_ty).to_def_id();
            for clause in where_clause {
                if let hir::WherePredicate::BoundPredicate(pred) = clause {
                    if pred.is_param_bound(self_ty_def_id) {
                        search_bounds(pred.bounds);
                    }
                }
            }
        }

        let sized_def_id = tcx.lang_items().require(LangItem::Sized);
        match (&sized_def_id, unbound) {
            (Ok(sized_def_id), Some(tpb))
                if tpb.path.res == Res::Def(DefKind::Trait, *sized_def_id) =>
            {
                // There was in fact a `?Sized` bound, return without doing anything
                return;
            }
            (_, Some(_)) => {
                // There was a `?Trait` bound, but it was not `?Sized`; warn.
                tcx.sess.span_warn(
                    span,
                    "default bound relaxed for a type parameter, but \
                        this does nothing because the given bound is not \
                        a default; only `?Sized` is supported",
                );
                // Otherwise, add implicitly sized if `Sized` is available.
            }
            _ => {
                // There was no `?Sized` bound; add implicitly sized if `Sized` is available.
            }
        }
        if sized_def_id.is_err() {
            // No lang item for `Sized`, so we can't add it as a bound.
            return;
        }
        bounds.implicitly_sized = Some(span);
    }

    /// This helper takes a *converted* parameter type (`param_ty`)
    /// and an *unconverted* list of bounds:
    ///
    /// ```text
    /// fn foo<T: Debug>
    ///        ^  ^^^^^ `ast_bounds` parameter, in HIR form
    ///        |
    ///        `param_ty`, in ty form
    /// ```
    ///
    /// It adds these `ast_bounds` into the `bounds` structure.
    ///
    /// **A note on binders:** there is an implied binder around
    /// `param_ty` and `ast_bounds`. See `instantiate_poly_trait_ref`
    /// for more details.
    #[instrument(level = "debug", skip(self, ast_bounds, bounds))]
    pub(crate) fn add_bounds<'hir, I: Iterator<Item = &'hir hir::GenericBound<'hir>>>(
        &self,
        param_ty: Ty<'tcx>,
        ast_bounds: I,
        bounds: &mut Bounds<'tcx>,
        bound_vars: &'tcx ty::List<ty::BoundVariableKind>,
    ) {
        for ast_bound in ast_bounds {
            match ast_bound {
                hir::GenericBound::Trait(poly_trait_ref, modifier) => {
                    let constness = match modifier {
                        hir::TraitBoundModifier::MaybeConst => ty::BoundConstness::ConstIfConst,
                        hir::TraitBoundModifier::None => ty::BoundConstness::NotConst,
                        hir::TraitBoundModifier::Maybe => continue,
                    };

                    let _ = self.instantiate_poly_trait_ref(
                        &poly_trait_ref.trait_ref,
                        poly_trait_ref.span,
                        constness,
                        param_ty,
                        bounds,
                        false,
                    );
                }
                &hir::GenericBound::LangItemTrait(lang_item, span, hir_id, args) => {
                    self.instantiate_lang_item_trait_ref(
                        lang_item, span, hir_id, args, param_ty, bounds,
                    );
                }
                hir::GenericBound::Outlives(lifetime) => {
                    let region = self.ast_region_to_region(lifetime, None);
                    bounds
                        .region_bounds
                        .push((ty::Binder::bind_with_vars(region, bound_vars), lifetime.span));
                }
            }
        }
    }

    /// Translates a list of bounds from the HIR into the `Bounds` data structure.
    /// The self-type for the bounds is given by `param_ty`.
    ///
    /// Example:
    ///
    /// ```ignore (illustrative)
    /// fn foo<T: Bar + Baz>() { }
    /// //     ^  ^^^^^^^^^ ast_bounds
    /// //     param_ty
    /// ```
    ///
    /// The `sized_by_default` parameter indicates if, in this context, the `param_ty` should be
    /// considered `Sized` unless there is an explicit `?Sized` bound.  This would be true in the
    /// example above, but is not true in supertrait listings like `trait Foo: Bar + Baz`.
    ///
    /// `span` should be the declaration size of the parameter.
    pub(crate) fn compute_bounds(
        &self,
        param_ty: Ty<'tcx>,
        ast_bounds: &[hir::GenericBound<'_>],
    ) -> Bounds<'tcx> {
        self.compute_bounds_inner(param_ty, ast_bounds)
    }

    /// Convert the bounds in `ast_bounds` that refer to traits which define an associated type
    /// named `assoc_name` into ty::Bounds. Ignore the rest.
    pub(crate) fn compute_bounds_that_match_assoc_type(
        &self,
        param_ty: Ty<'tcx>,
        ast_bounds: &[hir::GenericBound<'_>],
        assoc_name: Ident,
    ) -> Bounds<'tcx> {
        let mut result = Vec::new();

        for ast_bound in ast_bounds {
            if let Some(trait_ref) = ast_bound.trait_ref()
                && let Some(trait_did) = trait_ref.trait_def_id()
                && self.tcx().trait_may_define_assoc_type(trait_did, assoc_name)
            {
                result.push(ast_bound.clone());
            }
        }

        self.compute_bounds_inner(param_ty, &result)
    }

    fn compute_bounds_inner(
        &self,
        param_ty: Ty<'tcx>,
        ast_bounds: &[hir::GenericBound<'_>],
    ) -> Bounds<'tcx> {
        let mut bounds = Bounds::default();

        self.add_bounds(param_ty, ast_bounds.iter(), &mut bounds, ty::List::empty());
        debug!(?bounds);

        bounds
    }

    /// Given an HIR binding like `Item = Foo` or `Item: Foo`, pushes the corresponding predicates
    /// onto `bounds`.
    ///
    /// **A note on binders:** given something like `T: for<'a> Iterator<Item = &'a u32>`, the
    /// `trait_ref` here will be `for<'a> T: Iterator`. The `binding` data however is from *inside*
    /// the binder (e.g., `&'a u32`) and hence may reference bound regions.
    #[instrument(level = "debug", skip(self, bounds, speculative, dup_bindings, path_span))]
    fn add_predicates_for_ast_type_binding(
        &self,
        hir_ref_id: hir::HirId,
        trait_ref: ty::PolyTraitRef<'tcx>,
        binding: &ConvertedBinding<'_, 'tcx>,
        bounds: &mut Bounds<'tcx>,
        speculative: bool,
        dup_bindings: &mut FxHashMap<DefId, Span>,
        path_span: Span,
    ) -> Result<(), ErrorGuaranteed> {
        // Given something like `U: SomeTrait<T = X>`, we want to produce a
        // predicate like `<U as SomeTrait>::T = X`. This is somewhat
        // subtle in the event that `T` is defined in a supertrait of
        // `SomeTrait`, because in that case we need to upcast.
        //
        // That is, consider this case:
        //
        // ```
        // trait SubTrait: SuperTrait<i32> { }
        // trait SuperTrait<A> { type T; }
        //
        // ... B: SubTrait<T = foo> ...
        // ```
        //
        // We want to produce `<B as SuperTrait<i32>>::T == foo`.

        let tcx = self.tcx();

        let candidate =
            if self.trait_defines_associated_type_named(trait_ref.def_id(), binding.item_name) {
                // Simple case: X is defined in the current trait.
                trait_ref
            } else {
                // Otherwise, we have to walk through the supertraits to find
                // those that do.
                self.one_bound_for_assoc_type(
                    || traits::supertraits(tcx, trait_ref),
                    || trait_ref.print_only_trait_path().to_string(),
                    binding.item_name,
                    path_span,
                    || match binding.kind {
                        ConvertedBindingKind::Equality(ty) => Some(ty.to_string()),
                        _ => None,
                    },
                )?
            };

        let (assoc_ident, def_scope) =
            tcx.adjust_ident_and_get_scope(binding.item_name, candidate.def_id(), hir_ref_id);

        // We have already adjusted the item name above, so compare with `ident.normalize_to_macros_2_0()` instead
        // of calling `filter_by_name_and_kind`.
        let find_item_of_kind = |kind| {
            tcx.associated_items(candidate.def_id())
                .filter_by_name_unhygienic(assoc_ident.name)
                .find(|i| i.kind == kind && i.ident(tcx).normalize_to_macros_2_0() == assoc_ident)
        };
        let assoc_item = find_item_of_kind(ty::AssocKind::Type)
            .or_else(|| find_item_of_kind(ty::AssocKind::Const))
            .expect("missing associated type");

        if !assoc_item.visibility(tcx).is_accessible_from(def_scope, tcx) {
            tcx.sess
                .struct_span_err(
                    binding.span,
                    &format!("{} `{}` is private", assoc_item.kind, binding.item_name),
                )
                .span_label(binding.span, &format!("private {}", assoc_item.kind))
                .emit();
        }
        tcx.check_stability(assoc_item.def_id, Some(hir_ref_id), binding.span, None);

        if !speculative {
            dup_bindings
                .entry(assoc_item.def_id)
                .and_modify(|prev_span| {
                    self.tcx().sess.emit_err(ValueOfAssociatedStructAlreadySpecified {
                        span: binding.span,
                        prev_span: *prev_span,
                        item_name: binding.item_name,
                        def_path: tcx.def_path_str(assoc_item.container_id(tcx)),
                    });
                })
                .or_insert(binding.span);
        }

        // Include substitutions for generic parameters of associated types
        let projection_ty = candidate.map_bound(|trait_ref| {
            let ident = Ident::new(assoc_item.name, binding.item_name.span);
            let item_segment = hir::PathSegment {
                ident,
                hir_id: binding.hir_id,
                res: Res::Err,
                args: Some(binding.gen_args),
                infer_args: false,
            };

            let substs_trait_ref_and_assoc_item = self.create_substs_for_associated_item(
                tcx,
                path_span,
                assoc_item.def_id,
                &item_segment,
                trait_ref.substs,
            );

            debug!(
                "add_predicates_for_ast_type_binding: substs for trait-ref and assoc_item: {:?}",
                substs_trait_ref_and_assoc_item
            );

            ty::ProjectionTy {
                item_def_id: assoc_item.def_id,
                substs: substs_trait_ref_and_assoc_item,
            }
        });

        if !speculative {
            // Find any late-bound regions declared in `ty` that are not
            // declared in the trait-ref or assoc_item. These are not well-formed.
            //
            // Example:
            //
            //     for<'a> <T as Iterator>::Item = &'a str // <-- 'a is bad
            //     for<'a> <T as FnMut<(&'a u32,)>>::Output = &'a str // <-- 'a is ok
            if let ConvertedBindingKind::Equality(ty) = binding.kind {
                let late_bound_in_trait_ref =
                    tcx.collect_constrained_late_bound_regions(&projection_ty);
                let late_bound_in_ty =
                    tcx.collect_referenced_late_bound_regions(&trait_ref.rebind(ty));
                debug!("late_bound_in_trait_ref = {:?}", late_bound_in_trait_ref);
                debug!("late_bound_in_ty = {:?}", late_bound_in_ty);

                // FIXME: point at the type params that don't have appropriate lifetimes:
                // struct S1<F: for<'a> Fn(&i32, &i32) -> &'a i32>(F);
                //                         ----  ----     ^^^^^^^
                self.validate_late_bound_regions(
                    late_bound_in_trait_ref,
                    late_bound_in_ty,
                    |br_name| {
                        struct_span_err!(
                            tcx.sess,
                            binding.span,
                            E0582,
                            "binding for associated type `{}` references {}, \
                             which does not appear in the trait input types",
                            binding.item_name,
                            br_name
                        )
                    },
                );
            }
        }

        match binding.kind {
            ConvertedBindingKind::Equality(mut term) => {
                // "Desugar" a constraint like `T: Iterator<Item = u32>` this to
                // the "projection predicate" for:
                //
                // `<T as Iterator>::Item = u32`
                let assoc_item_def_id = projection_ty.skip_binder().item_def_id;
                let def_kind = tcx.def_kind(assoc_item_def_id);
                match (def_kind, term.unpack()) {
                    (hir::def::DefKind::AssocTy, ty::TermKind::Ty(_))
                    | (hir::def::DefKind::AssocConst, ty::TermKind::Const(_)) => (),
                    (_, _) => {
                        let got = if let Some(_) = term.ty() { "type" } else { "constant" };
                        let expected = def_kind.descr(assoc_item_def_id);
                        tcx.sess
                            .struct_span_err(
                                binding.span,
                                &format!("expected {expected} bound, found {got}"),
                            )
                            .span_note(
                                tcx.def_span(assoc_item_def_id),
                                &format!("{expected} defined here"),
                            )
                            .emit();
                        term = match def_kind {
                            hir::def::DefKind::AssocTy => tcx.ty_error().into(),
                            hir::def::DefKind::AssocConst => tcx
                                .const_error(
                                    tcx.bound_type_of(assoc_item_def_id)
                                        .subst(tcx, projection_ty.skip_binder().substs),
                                )
                                .into(),
                            _ => unreachable!(),
                        };
                    }
                }
                bounds.projection_bounds.push((
                    projection_ty.map_bound(|projection_ty| ty::ProjectionPredicate {
                        projection_ty,
                        term: term,
                    }),
                    binding.span,
                ));
            }
            ConvertedBindingKind::Constraint(ast_bounds) => {
                // "Desugar" a constraint like `T: Iterator<Item: Debug>` to
                //
                // `<T as Iterator>::Item: Debug`
                //
                // Calling `skip_binder` is okay, because `add_bounds` expects the `param_ty`
                // parameter to have a skipped binder.
                let param_ty = tcx.mk_ty(ty::Projection(projection_ty.skip_binder()));
                self.add_bounds(param_ty, ast_bounds.iter(), bounds, candidate.bound_vars());
            }
        }
        Ok(())
    }

    fn ast_path_to_ty(
        &self,
        span: Span,
        did: DefId,
        item_segment: &hir::PathSegment<'_>,
    ) -> Ty<'tcx> {
        let substs = self.ast_path_substs_for_ty(span, did, item_segment);
        self.normalize_ty(
            span,
            EarlyBinder(self.tcx().at(span).type_of(did)).subst(self.tcx(), substs),
        )
    }

    fn conv_object_ty_poly_trait_ref(
        &self,
        span: Span,
        trait_bounds: &[hir::PolyTraitRef<'_>],
        lifetime: &hir::Lifetime,
        borrowed: bool,
    ) -> Ty<'tcx> {
        let tcx = self.tcx();

        let mut bounds = Bounds::default();
        let mut potential_assoc_types = Vec::new();
        let dummy_self = self.tcx().types.trait_object_dummy_self;
        for trait_bound in trait_bounds.iter().rev() {
            if let GenericArgCountResult {
                correct:
                    Err(GenericArgCountMismatch { invalid_args: cur_potential_assoc_types, .. }),
                ..
            } = self.instantiate_poly_trait_ref(
                &trait_bound.trait_ref,
                trait_bound.span,
                ty::BoundConstness::NotConst,
                dummy_self,
                &mut bounds,
                false,
            ) {
                potential_assoc_types.extend(cur_potential_assoc_types);
            }
        }

        // Expand trait aliases recursively and check that only one regular (non-auto) trait
        // is used and no 'maybe' bounds are used.
        let expanded_traits =
            traits::expand_trait_aliases(tcx, bounds.trait_bounds.iter().map(|&(a, b, _)| (a, b)));
        let (mut auto_traits, regular_traits): (Vec<_>, Vec<_>) = expanded_traits
            .filter(|i| i.trait_ref().self_ty().skip_binder() == dummy_self)
            .partition(|i| tcx.trait_is_auto(i.trait_ref().def_id()));
        if regular_traits.len() > 1 {
            let first_trait = &regular_traits[0];
            let additional_trait = &regular_traits[1];
            let mut err = struct_span_err!(
                tcx.sess,
                additional_trait.bottom().1,
                E0225,
                "only auto traits can be used as additional traits in a trait object"
            );
            additional_trait.label_with_exp_info(
                &mut err,
                "additional non-auto trait",
                "additional use",
            );
            first_trait.label_with_exp_info(&mut err, "first non-auto trait", "first use");
            err.help(&format!(
                "consider creating a new trait with all of these as supertraits and using that \
                 trait here instead: `trait NewTrait: {} {{}}`",
                regular_traits
                    .iter()
                    .map(|t| t.trait_ref().print_only_trait_path().to_string())
                    .collect::<Vec<_>>()
                    .join(" + "),
            ));
            err.note(
                "auto-traits like `Send` and `Sync` are traits that have special properties; \
                 for more information on them, visit \
                 <https://doc.rust-lang.org/reference/special-types-and-traits.html#auto-traits>",
            );
            err.emit();
        }

        if regular_traits.is_empty() && auto_traits.is_empty() {
            let trait_alias_span = bounds
                .trait_bounds
                .iter()
                .map(|&(trait_ref, _, _)| trait_ref.def_id())
                .find(|&trait_ref| tcx.is_trait_alias(trait_ref))
                .map(|trait_ref| tcx.def_span(trait_ref));
            tcx.sess.emit_err(TraitObjectDeclaredWithNoTraits { span, trait_alias_span });
            return tcx.ty_error();
        }

        // Check that there are no gross object safety violations;
        // most importantly, that the supertraits don't contain `Self`,
        // to avoid ICEs.
        for item in &regular_traits {
            let object_safety_violations =
                astconv_object_safety_violations(tcx, item.trait_ref().def_id());
            if !object_safety_violations.is_empty() {
                report_object_safety_error(
                    tcx,
                    span,
                    item.trait_ref().def_id(),
                    &object_safety_violations,
                )
                .emit();
                return tcx.ty_error();
            }
        }

        // Use a `BTreeSet` to keep output in a more consistent order.
        let mut associated_types: FxHashMap<Span, BTreeSet<DefId>> = FxHashMap::default();

        let regular_traits_refs_spans = bounds
            .trait_bounds
            .into_iter()
            .filter(|(trait_ref, _, _)| !tcx.trait_is_auto(trait_ref.def_id()));

        for (base_trait_ref, span, constness) in regular_traits_refs_spans {
            assert_eq!(constness, ty::BoundConstness::NotConst);

            for obligation in traits::elaborate_trait_ref(tcx, base_trait_ref) {
                debug!(
                    "conv_object_ty_poly_trait_ref: observing object predicate `{:?}`",
                    obligation.predicate
                );

                let bound_predicate = obligation.predicate.kind();
                match bound_predicate.skip_binder() {
                    ty::PredicateKind::Trait(pred) => {
                        let pred = bound_predicate.rebind(pred);
                        associated_types.entry(span).or_default().extend(
                            tcx.associated_items(pred.def_id())
                                .in_definition_order()
                                .filter(|item| item.kind == ty::AssocKind::Type)
                                .map(|item| item.def_id),
                        );
                    }
                    ty::PredicateKind::Projection(pred) => {
                        let pred = bound_predicate.rebind(pred);
                        // A `Self` within the original bound will be substituted with a
                        // `trait_object_dummy_self`, so check for that.
                        let references_self = match pred.skip_binder().term.unpack() {
                            ty::TermKind::Ty(ty) => ty.walk().any(|arg| arg == dummy_self.into()),
                            ty::TermKind::Const(c) => {
                                c.ty().walk().any(|arg| arg == dummy_self.into())
                            }
                        };

                        // If the projection output contains `Self`, force the user to
                        // elaborate it explicitly to avoid a lot of complexity.
                        //
                        // The "classically useful" case is the following:
                        // ```
                        //     trait MyTrait: FnMut() -> <Self as MyTrait>::MyOutput {
                        //         type MyOutput;
                        //     }
                        // ```
                        //
                        // Here, the user could theoretically write `dyn MyTrait<Output = X>`,
                        // but actually supporting that would "expand" to an infinitely-long type
                        // `fix $ τ → dyn MyTrait<MyOutput = X, Output = <τ as MyTrait>::MyOutput`.
                        //
                        // Instead, we force the user to write
                        // `dyn MyTrait<MyOutput = X, Output = X>`, which is uglier but works. See
                        // the discussion in #56288 for alternatives.
                        if !references_self {
                            // Include projections defined on supertraits.
                            bounds.projection_bounds.push((pred, span));
                        }
                    }
                    _ => (),
                }
            }
        }

        for (projection_bound, _) in &bounds.projection_bounds {
            for def_ids in associated_types.values_mut() {
                def_ids.remove(&projection_bound.projection_def_id());
            }
        }

        self.complain_about_missing_associated_types(
            associated_types,
            potential_assoc_types,
            trait_bounds,
        );

        // De-duplicate auto traits so that, e.g., `dyn Trait + Send + Send` is the same as
        // `dyn Trait + Send`.
        // We remove duplicates by inserting into a `FxHashSet` to avoid re-ordering
        // the bounds
        let mut duplicates = FxHashSet::default();
        auto_traits.retain(|i| duplicates.insert(i.trait_ref().def_id()));
        debug!("regular_traits: {:?}", regular_traits);
        debug!("auto_traits: {:?}", auto_traits);

        // Erase the `dummy_self` (`trait_object_dummy_self`) used above.
        let existential_trait_refs = regular_traits.iter().map(|i| {
            i.trait_ref().map_bound(|trait_ref: ty::TraitRef<'tcx>| {
                assert_eq!(trait_ref.self_ty(), dummy_self);

                // Verify that `dummy_self` did not leak inside default type parameters.  This
                // could not be done at path creation, since we need to see through trait aliases.
                let mut missing_type_params = vec![];
                let mut references_self = false;
                let generics = tcx.generics_of(trait_ref.def_id);
                let substs: Vec<_> = trait_ref
                    .substs
                    .iter()
                    .enumerate()
                    .skip(1) // Remove `Self` for `ExistentialPredicate`.
                    .map(|(index, arg)| {
                        if arg == dummy_self.into() {
                            let param = &generics.params[index];
                            missing_type_params.push(param.name);
                            return tcx.ty_error().into();
                        } else if arg.walk().any(|arg| arg == dummy_self.into()) {
                            references_self = true;
                            return tcx.ty_error().into();
                        }
                        arg
                    })
                    .collect();
                let substs = tcx.intern_substs(&substs[..]);

                let span = i.bottom().1;
                let empty_generic_args = trait_bounds.iter().any(|hir_bound| {
                    hir_bound.trait_ref.path.res == Res::Def(DefKind::Trait, trait_ref.def_id)
                        && hir_bound.span.contains(span)
                });
                self.complain_about_missing_type_params(
                    missing_type_params,
                    trait_ref.def_id,
                    span,
                    empty_generic_args,
                );

                if references_self {
                    let def_id = i.bottom().0.def_id();
                    let mut err = struct_span_err!(
                        tcx.sess,
                        i.bottom().1,
                        E0038,
                        "the {} `{}` cannot be made into an object",
                        tcx.def_kind(def_id).descr(def_id),
                        tcx.item_name(def_id),
                    );
                    err.note(
                        rustc_middle::traits::ObjectSafetyViolation::SupertraitSelf(smallvec![])
                            .error_msg(),
                    );
                    err.emit();
                }

                ty::ExistentialTraitRef { def_id: trait_ref.def_id, substs }
            })
        });

        let existential_projections = bounds.projection_bounds.iter().map(|(bound, _)| {
            bound.map_bound(|mut b| {
                assert_eq!(b.projection_ty.self_ty(), dummy_self);

                // Like for trait refs, verify that `dummy_self` did not leak inside default type
                // parameters.
                let references_self = b.projection_ty.substs.iter().skip(1).any(|arg| {
                    if arg.walk().any(|arg| arg == dummy_self.into()) {
                        return true;
                    }
                    false
                });
                if references_self {
                    tcx.sess
                        .delay_span_bug(span, "trait object projection bounds reference `Self`");
                    let substs: Vec<_> = b
                        .projection_ty
                        .substs
                        .iter()
                        .map(|arg| {
                            if arg.walk().any(|arg| arg == dummy_self.into()) {
                                return tcx.ty_error().into();
                            }
                            arg
                        })
                        .collect();
                    b.projection_ty.substs = tcx.intern_substs(&substs[..]);
                }

                ty::ExistentialProjection::erase_self_ty(tcx, b)
            })
        });

        let regular_trait_predicates = existential_trait_refs
            .map(|trait_ref| trait_ref.map_bound(ty::ExistentialPredicate::Trait));
        let auto_trait_predicates = auto_traits.into_iter().map(|trait_ref| {
            ty::Binder::dummy(ty::ExistentialPredicate::AutoTrait(trait_ref.trait_ref().def_id()))
        });
        // N.b. principal, projections, auto traits
        // FIXME: This is actually wrong with multiple principals in regards to symbol mangling
        let mut v = regular_trait_predicates
            .chain(
                existential_projections.map(|x| x.map_bound(ty::ExistentialPredicate::Projection)),
            )
            .chain(auto_trait_predicates)
            .collect::<SmallVec<[_; 8]>>();
        v.sort_by(|a, b| a.skip_binder().stable_cmp(tcx, &b.skip_binder()));
        v.dedup();
        let existential_predicates = tcx.mk_poly_existential_predicates(v.into_iter());

        // Use explicitly-specified region bound.
        let region_bound = if !lifetime.is_elided() {
            self.ast_region_to_region(lifetime, None)
        } else {
            self.compute_object_lifetime_bound(span, existential_predicates).unwrap_or_else(|| {
                if tcx.named_region(lifetime.hir_id).is_some() {
                    self.ast_region_to_region(lifetime, None)
                } else {
                    self.re_infer(None, span).unwrap_or_else(|| {
                        let mut err = struct_span_err!(
                            tcx.sess,
                            span,
                            E0228,
                            "the lifetime bound for this object type cannot be deduced \
                             from context; please supply an explicit bound"
                        );
                        if borrowed {
                            // We will have already emitted an error E0106 complaining about a
                            // missing named lifetime in `&dyn Trait`, so we elide this one.
                            err.delay_as_bug();
                        } else {
                            err.emit();
                        }
                        tcx.lifetimes.re_static
                    })
                }
            })
        };
        debug!("region_bound: {:?}", region_bound);

        let ty = tcx.mk_dynamic(existential_predicates, region_bound);
        debug!("trait_object_type: {:?}", ty);
        ty
    }

    fn report_ambiguous_associated_type(
        &self,
        span: Span,
        type_str: &str,
        trait_str: &str,
        name: Symbol,
    ) -> ErrorGuaranteed {
        let mut err = struct_span_err!(self.tcx().sess, span, E0223, "ambiguous associated type");
        if self
            .tcx()
            .resolutions(())
            .confused_type_with_std_module
            .keys()
            .any(|full_span| full_span.contains(span))
        {
            err.span_suggestion(
                span.shrink_to_lo(),
                "you are looking for the module in `std`, not the primitive type",
                "std::",
                Applicability::MachineApplicable,
            );
        } else {
            err.span_suggestion(
                span,
                "use fully-qualified syntax",
                format!("<{} as {}>::{}", type_str, trait_str, name),
                Applicability::HasPlaceholders,
            );
        }
        err.emit()
    }

    // Search for a bound on a type parameter which includes the associated item
    // given by `assoc_name`. `ty_param_def_id` is the `DefId` of the type parameter
    // This function will fail if there are no suitable bounds or there is
    // any ambiguity.
    fn find_bound_for_assoc_item(
        &self,
        ty_param_def_id: LocalDefId,
        assoc_name: Ident,
        span: Span,
    ) -> Result<ty::PolyTraitRef<'tcx>, ErrorGuaranteed> {
        let tcx = self.tcx();

        debug!(
            "find_bound_for_assoc_item(ty_param_def_id={:?}, assoc_name={:?}, span={:?})",
            ty_param_def_id, assoc_name, span,
        );

        let predicates = &self
            .get_type_parameter_bounds(span, ty_param_def_id.to_def_id(), assoc_name)
            .predicates;

        debug!("find_bound_for_assoc_item: predicates={:#?}", predicates);

        let param_name = tcx.hir().ty_param_name(ty_param_def_id);
        self.one_bound_for_assoc_type(
            || {
                traits::transitive_bounds_that_define_assoc_type(
                    tcx,
                    predicates.iter().filter_map(|(p, _)| {
                        Some(p.to_opt_poly_trait_pred()?.map_bound(|t| t.trait_ref))
                    }),
                    assoc_name,
                )
            },
            || param_name.to_string(),
            assoc_name,
            span,
            || None,
        )
    }

    // Checks that `bounds` contains exactly one element and reports appropriate
    // errors otherwise.
    fn one_bound_for_assoc_type<I>(
        &self,
        all_candidates: impl Fn() -> I,
        ty_param_name: impl Fn() -> String,
        assoc_name: Ident,
        span: Span,
        is_equality: impl Fn() -> Option<String>,
    ) -> Result<ty::PolyTraitRef<'tcx>, ErrorGuaranteed>
    where
        I: Iterator<Item = ty::PolyTraitRef<'tcx>>,
    {
        let mut matching_candidates = all_candidates()
            .filter(|r| self.trait_defines_associated_type_named(r.def_id(), assoc_name));
        let mut const_candidates = all_candidates()
            .filter(|r| self.trait_defines_associated_const_named(r.def_id(), assoc_name));

        let (bound, next_cand) = match (matching_candidates.next(), const_candidates.next()) {
            (Some(bound), _) => (bound, matching_candidates.next()),
            (None, Some(bound)) => (bound, const_candidates.next()),
            (None, None) => {
                let reported = self.complain_about_assoc_type_not_found(
                    all_candidates,
                    &ty_param_name(),
                    assoc_name,
                    span,
                );
                return Err(reported);
            }
        };
        debug!("one_bound_for_assoc_type: bound = {:?}", bound);

        if let Some(bound2) = next_cand {
            debug!("one_bound_for_assoc_type: bound2 = {:?}", bound2);

            let is_equality = is_equality();
            let bounds = IntoIterator::into_iter([bound, bound2]).chain(matching_candidates);
            let mut err = if is_equality.is_some() {
                // More specific Error Index entry.
                struct_span_err!(
                    self.tcx().sess,
                    span,
                    E0222,
                    "ambiguous associated type `{}` in bounds of `{}`",
                    assoc_name,
                    ty_param_name()
                )
            } else {
                struct_span_err!(
                    self.tcx().sess,
                    span,
                    E0221,
                    "ambiguous associated type `{}` in bounds of `{}`",
                    assoc_name,
                    ty_param_name()
                )
            };
            err.span_label(span, format!("ambiguous associated type `{}`", assoc_name));

            let mut where_bounds = vec![];
            for bound in bounds {
                let bound_id = bound.def_id();
                let bound_span = self
                    .tcx()
                    .associated_items(bound_id)
                    .find_by_name_and_kind(self.tcx(), assoc_name, ty::AssocKind::Type, bound_id)
                    .and_then(|item| self.tcx().hir().span_if_local(item.def_id));

                if let Some(bound_span) = bound_span {
                    err.span_label(
                        bound_span,
                        format!(
                            "ambiguous `{}` from `{}`",
                            assoc_name,
                            bound.print_only_trait_path(),
                        ),
                    );
                    if let Some(constraint) = &is_equality {
                        where_bounds.push(format!(
                            "        T: {trait}::{assoc} = {constraint}",
                            trait=bound.print_only_trait_path(),
                            assoc=assoc_name,
                            constraint=constraint,
                        ));
                    } else {
                        err.span_suggestion_verbose(
                            span.with_hi(assoc_name.span.lo()),
                            "use fully qualified syntax to disambiguate",
                            format!(
                                "<{} as {}>::",
                                ty_param_name(),
                                bound.print_only_trait_path(),
                            ),
                            Applicability::MaybeIncorrect,
                        );
                    }
                } else {
                    err.note(&format!(
                        "associated type `{}` could derive from `{}`",
                        ty_param_name(),
                        bound.print_only_trait_path(),
                    ));
                }
            }
            if !where_bounds.is_empty() {
                err.help(&format!(
                    "consider introducing a new type parameter `T` and adding `where` constraints:\
                     \n    where\n        T: {},\n{}",
                    ty_param_name(),
                    where_bounds.join(",\n"),
                ));
            }
            let reported = err.emit();
            if !where_bounds.is_empty() {
                return Err(reported);
            }
        }

        Ok(bound)
    }

    // Create a type from a path to an associated type.
    // For a path `A::B::C::D`, `qself_ty` and `qself_def` are the type and def for `A::B::C`
    // and item_segment is the path segment for `D`. We return a type and a def for
    // the whole path.
    // Will fail except for `T::A` and `Self::A`; i.e., if `qself_ty`/`qself_def` are not a type
    // parameter or `Self`.
    // NOTE: When this function starts resolving `Trait::AssocTy` successfully
    // it should also start reporting the `BARE_TRAIT_OBJECTS` lint.
    pub fn associated_path_to_ty(
        &self,
        hir_ref_id: hir::HirId,
        span: Span,
        qself_ty: Ty<'tcx>,
        qself: &hir::Ty<'_>,
        assoc_segment: &hir::PathSegment<'_>,
        permit_variants: bool,
    ) -> Result<(Ty<'tcx>, DefKind, DefId), ErrorGuaranteed> {
        let tcx = self.tcx();
        let assoc_ident = assoc_segment.ident;
        let qself_res = if let hir::TyKind::Path(hir::QPath::Resolved(_, ref path)) = qself.kind {
            path.res
        } else {
            Res::Err
        };

        debug!("associated_path_to_ty: {:?}::{}", qself_ty, assoc_ident);

        // Check if we have an enum variant.
        let mut variant_resolution = None;
        if let ty::Adt(adt_def, _) = qself_ty.kind() {
            if adt_def.is_enum() {
                let variant_def = adt_def
                    .variants()
                    .iter()
                    .find(|vd| tcx.hygienic_eq(assoc_ident, vd.ident(tcx), adt_def.did()));
                if let Some(variant_def) = variant_def {
                    if permit_variants {
                        tcx.check_stability(variant_def.def_id, Some(hir_ref_id), span, None);
                        self.prohibit_generics(slice::from_ref(assoc_segment).iter(), |err| {
                            err.note("enum variants can't have type parameters");
                            let type_name = tcx.item_name(adt_def.did());
                            let msg = format!(
                                "you might have meant to specity type parameters on enum \
                                 `{type_name}`"
                            );
                            let Some(args) = assoc_segment.args else { return; };
                            // Get the span of the generics args *including* the leading `::`.
                            let args_span = assoc_segment.ident.span.shrink_to_hi().to(args.span_ext);
                            if tcx.generics_of(adt_def.did()).count() == 0 {
                                // FIXME(estebank): we could also verify that the arguments being
                                // work for the `enum`, instead of just looking if it takes *any*.
                                err.span_suggestion_verbose(
                                    args_span,
                                    &format!("{type_name} doesn't have generic parameters"),
                                    "",
                                    Applicability::MachineApplicable,
                                );
                                return;
                            }
                            let Ok(snippet) = tcx.sess.source_map().span_to_snippet(args_span) else {
                                err.note(&msg);
                                return;
                            };
                            let (qself_sugg_span, is_self) = if let hir::TyKind::Path(
                                hir::QPath::Resolved(_, ref path)
                            ) = qself.kind {
                                // If the path segment already has type params, we want to overwrite
                                // them.
                                match &path.segments[..] {
                                    // `segment` is the previous to last element on the path,
                                    // which would normally be the `enum` itself, while the last
                                    // `_` `PathSegment` corresponds to the variant.
                                    [.., hir::PathSegment {
                                        ident,
                                        args,
                                        res: Res::Def(DefKind::Enum, _),
                                        ..
                                    }, _] => (
                                        // We need to include the `::` in `Type::Variant::<Args>`
                                        // to point the span to `::<Args>`, not just `<Args>`.
                                        ident.span.shrink_to_hi().to(args.map_or(
                                            ident.span.shrink_to_hi(),
                                            |a| a.span_ext)),
                                        false,
                                    ),
                                    [segment] => (
                                        // We need to include the `::` in `Type::Variant::<Args>`
                                        // to point the span to `::<Args>`, not just `<Args>`.
                                        segment.ident.span.shrink_to_hi().to(segment.args.map_or(
                                            segment.ident.span.shrink_to_hi(),
                                            |a| a.span_ext)),
                                        kw::SelfUpper == segment.ident.name,
                                    ),
                                    _ => {
                                        err.note(&msg);
                                        return;
                                    }
                                }
                            } else {
                                err.note(&msg);
                                return;
                            };
                            let suggestion = vec![
                                if is_self {
                                    // Account for people writing `Self::Variant::<Args>`, where
                                    // `Self` is the enum, and suggest replacing `Self` with the
                                    // appropriate type: `Type::<Args>::Variant`.
                                    (qself.span, format!("{type_name}{snippet}"))
                                } else {
                                    (qself_sugg_span, snippet)
                                },
                                (args_span, String::new()),
                            ];
                            err.multipart_suggestion_verbose(
                                &msg,
                                suggestion,
                                Applicability::MaybeIncorrect,
                            );
                        });
                        return Ok((qself_ty, DefKind::Variant, variant_def.def_id));
                    } else {
                        variant_resolution = Some(variant_def.def_id);
                    }
                }
            }
        }

        // Find the type of the associated item, and the trait where the associated
        // item is declared.
        let bound = match (&qself_ty.kind(), qself_res) {
            (_, Res::SelfTy { trait_: Some(_), alias_to: Some((impl_def_id, _)) }) => {
                // `Self` in an impl of a trait -- we have a concrete self type and a
                // trait reference.
                let Some(trait_ref) = tcx.impl_trait_ref(impl_def_id) else {
                    // A cycle error occurred, most likely.
                    let guar = tcx.sess.delay_span_bug(span, "expected cycle error");
                    return Err(guar);
                };

                self.one_bound_for_assoc_type(
                    || traits::supertraits(tcx, ty::Binder::dummy(trait_ref)),
                    || "Self".to_string(),
                    assoc_ident,
                    span,
                    || None,
                )?
            }
            (
                &ty::Param(_),
                Res::SelfTy { trait_: Some(param_did), alias_to: None }
                | Res::Def(DefKind::TyParam, param_did),
            ) => self.find_bound_for_assoc_item(param_did.expect_local(), assoc_ident, span)?,
            _ => {
                let reported = if variant_resolution.is_some() {
                    // Variant in type position
                    let msg = format!("expected type, found variant `{}`", assoc_ident);
                    tcx.sess.span_err(span, &msg)
                } else if qself_ty.is_enum() {
                    let mut err = struct_span_err!(
                        tcx.sess,
                        assoc_ident.span,
                        E0599,
                        "no variant named `{}` found for enum `{}`",
                        assoc_ident,
                        qself_ty,
                    );

                    let adt_def = qself_ty.ty_adt_def().expect("enum is not an ADT");
                    if let Some(suggested_name) = find_best_match_for_name(
                        &adt_def
                            .variants()
                            .iter()
                            .map(|variant| variant.name)
                            .collect::<Vec<Symbol>>(),
                        assoc_ident.name,
                        None,
                    ) {
                        err.span_suggestion(
                            assoc_ident.span,
                            "there is a variant with a similar name",
                            suggested_name,
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        err.span_label(
                            assoc_ident.span,
                            format!("variant not found in `{}`", qself_ty),
                        );
                    }

                    if let Some(sp) = tcx.hir().span_if_local(adt_def.did()) {
                        err.span_label(sp, format!("variant `{}` not found here", assoc_ident));
                    }

                    err.emit()
                } else if let Some(reported) = qself_ty.error_reported() {
                    reported
                } else {
                    // Don't print `TyErr` to the user.
                    self.report_ambiguous_associated_type(
                        span,
                        &qself_ty.to_string(),
                        "Trait",
                        assoc_ident.name,
                    )
                };
                return Err(reported);
            }
        };

        let trait_did = bound.def_id();
        let (assoc_ident, def_scope) =
            tcx.adjust_ident_and_get_scope(assoc_ident, trait_did, hir_ref_id);

        // We have already adjusted the item name above, so compare with `ident.normalize_to_macros_2_0()` instead
        // of calling `filter_by_name_and_kind`.
        let item = tcx.associated_items(trait_did).in_definition_order().find(|i| {
            i.kind.namespace() == Namespace::TypeNS
                && i.ident(tcx).normalize_to_macros_2_0() == assoc_ident
        });
        // Assume that if it's not matched, there must be a const defined with the same name
        // but it was used in a type position.
        let Some(item) = item else {
            let msg = format!("found associated const `{assoc_ident}` when type was expected");
            let guar = tcx.sess.struct_span_err(span, &msg).emit();
            return Err(guar);
        };

        let ty = self.projected_ty_from_poly_trait_ref(span, item.def_id, assoc_segment, bound);
        let ty = self.normalize_ty(span, ty);

        let kind = DefKind::AssocTy;
        if !item.visibility(tcx).is_accessible_from(def_scope, tcx) {
            let kind = kind.descr(item.def_id);
            let msg = format!("{} `{}` is private", kind, assoc_ident);
            tcx.sess
                .struct_span_err(span, &msg)
                .span_label(span, &format!("private {}", kind))
                .emit();
        }
        tcx.check_stability(item.def_id, Some(hir_ref_id), span, None);

        if let Some(variant_def_id) = variant_resolution {
            tcx.struct_span_lint_hir(AMBIGUOUS_ASSOCIATED_ITEMS, hir_ref_id, span, |lint| {
                let mut err = lint.build("ambiguous associated item");
                let mut could_refer_to = |kind: DefKind, def_id, also| {
                    let note_msg = format!(
                        "`{}` could{} refer to the {} defined here",
                        assoc_ident,
                        also,
                        kind.descr(def_id)
                    );
                    err.span_note(tcx.def_span(def_id), &note_msg);
                };

                could_refer_to(DefKind::Variant, variant_def_id, "");
                could_refer_to(kind, item.def_id, " also");

                err.span_suggestion(
                    span,
                    "use fully-qualified syntax",
                    format!("<{} as {}>::{}", qself_ty, tcx.item_name(trait_did), assoc_ident),
                    Applicability::MachineApplicable,
                );

                err.emit();
            });
        }
        Ok((ty, kind, item.def_id))
    }

    fn qpath_to_ty(
        &self,
        span: Span,
        opt_self_ty: Option<Ty<'tcx>>,
        item_def_id: DefId,
        trait_segment: &hir::PathSegment<'_>,
        item_segment: &hir::PathSegment<'_>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx();

        let trait_def_id = tcx.parent(item_def_id);

        debug!("qpath_to_ty: trait_def_id={:?}", trait_def_id);

        let Some(self_ty) = opt_self_ty else {
            let path_str = tcx.def_path_str(trait_def_id);

            let def_id = self.item_def_id();

            debug!("qpath_to_ty: self.item_def_id()={:?}", def_id);

            let parent_def_id = def_id
                .and_then(|def_id| {
                    def_id.as_local().map(|def_id| tcx.hir().local_def_id_to_hir_id(def_id))
                })
                .map(|hir_id| tcx.hir().get_parent_item(hir_id).to_def_id());

            debug!("qpath_to_ty: parent_def_id={:?}", parent_def_id);

            // If the trait in segment is the same as the trait defining the item,
            // use the `<Self as ..>` syntax in the error.
            let is_part_of_self_trait_constraints = def_id == Some(trait_def_id);
            let is_part_of_fn_in_self_trait = parent_def_id == Some(trait_def_id);

            let type_name = if is_part_of_self_trait_constraints || is_part_of_fn_in_self_trait {
                "Self"
            } else {
                "Type"
            };

            self.report_ambiguous_associated_type(
                span,
                type_name,
                &path_str,
                item_segment.ident.name,
            );
            return tcx.ty_error();
        };

        debug!("qpath_to_ty: self_type={:?}", self_ty);

        let trait_ref =
            self.ast_path_to_mono_trait_ref(span, trait_def_id, self_ty, trait_segment, false);

        let item_substs = self.create_substs_for_associated_item(
            tcx,
            span,
            item_def_id,
            item_segment,
            trait_ref.substs,
        );

        debug!("qpath_to_ty: trait_ref={:?}", trait_ref);

        self.normalize_ty(span, tcx.mk_projection(item_def_id, item_substs))
    }

    pub fn prohibit_generics<'a>(
        &self,
        segments: impl Iterator<Item = &'a hir::PathSegment<'a>> + Clone,
        extend: impl Fn(&mut Diagnostic),
    ) -> bool {
        let args = segments.clone().flat_map(|segment| segment.args().args);

        let (lt, ty, ct, inf) =
            args.clone().fold((false, false, false, false), |(lt, ty, ct, inf), arg| match arg {
                hir::GenericArg::Lifetime(_) => (true, ty, ct, inf),
                hir::GenericArg::Type(_) => (lt, true, ct, inf),
                hir::GenericArg::Const(_) => (lt, ty, true, inf),
                hir::GenericArg::Infer(_) => (lt, ty, ct, true),
            });
        let mut emitted = false;
        if lt || ty || ct || inf {
            let types_and_spans: Vec<_> = segments
                .clone()
                .flat_map(|segment| {
                    if segment.args().args.is_empty() {
                        None
                    } else {
                        Some((
                            match segment.res {
                                Res::PrimTy(ty) => format!("{} `{}`", segment.res.descr(), ty.name()),
                                Res::Def(_, def_id)
                                if let Some(name) = self.tcx().opt_item_name(def_id) => {
                                    format!("{} `{name}`", segment.res.descr())
                                }
                                Res::Err => "this type".to_string(),
                                _ => segment.res.descr().to_string(),
                            },
                            segment.ident.span,
                        ))
                    }
                })
                .collect();
            let this_type = match &types_and_spans[..] {
                [.., _, (last, _)] => format!(
                    "{} and {last}",
                    types_and_spans[..types_and_spans.len() - 1]
                        .iter()
                        .map(|(x, _)| x.as_str())
                        .intersperse(&", ")
                        .collect::<String>()
                ),
                [(only, _)] => only.to_string(),
                [] => "this type".to_string(),
            };

            let arg_spans: Vec<Span> = args.map(|arg| arg.span()).collect();

            let mut kinds = Vec::with_capacity(4);
            if lt {
                kinds.push("lifetime");
            }
            if ty {
                kinds.push("type");
            }
            if ct {
                kinds.push("const");
            }
            if inf {
                kinds.push("generic");
            }
            let (kind, s) = match kinds[..] {
                [.., _, last] => (
                    format!(
                        "{} and {last}",
                        kinds[..kinds.len() - 1]
                            .iter()
                            .map(|&x| x)
                            .intersperse(", ")
                            .collect::<String>()
                    ),
                    "s",
                ),
                [only] => (format!("{only}"), ""),
                [] => unreachable!(),
            };
            let last_span = *arg_spans.last().unwrap();
            let span: MultiSpan = arg_spans.into();
            let mut err = struct_span_err!(
                self.tcx().sess,
                span,
                E0109,
                "{kind} arguments are not allowed on {this_type}",
            );
            err.span_label(last_span, format!("{kind} argument{s} not allowed"));
            for (what, span) in types_and_spans {
                err.span_label(span, format!("not allowed on {what}"));
            }
            extend(&mut err);
            err.emit();
            emitted = true;
        }

        for segment in segments {
            // Only emit the first error to avoid overloading the user with error messages.
            if let [binding, ..] = segment.args().bindings {
                Self::prohibit_assoc_ty_binding(self.tcx(), binding.span);
                return true;
            }
        }
        emitted
    }

    // FIXME(eddyb, varkor) handle type paths here too, not just value ones.
    pub fn def_ids_for_value_path_segments(
        &self,
        segments: &[hir::PathSegment<'_>],
        self_ty: Option<Ty<'tcx>>,
        kind: DefKind,
        def_id: DefId,
    ) -> Vec<PathSeg> {
        // We need to extract the type parameters supplied by the user in
        // the path `path`. Due to the current setup, this is a bit of a
        // tricky-process; the problem is that resolve only tells us the
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
        //    In this case, the parameters are declared in the type space.
        //
        // 2. Reference to a constructor of an enum variant:
        //
        //        enum E<T> { Foo(...) }
        //
        //    In this case, the parameters are defined in the type space,
        //    but may be specified either on the type or the variant.
        //
        // 3. Reference to a fn item or a free constant:
        //
        //        fn foo<T>() { }
        //
        //    In this case, the path will again always have the form
        //    `a::b::foo::<T>` where only the final segment should have
        //    type parameters. However, in this case, those parameters are
        //    declared on a value, and hence are in the `FnSpace`.
        //
        // 4. Reference to a method or an associated constant:
        //
        //        impl<A> SomeStruct<A> {
        //            fn foo<B>(...)
        //        }
        //
        //    Here we can have a path like
        //    `a::b::SomeStruct::<A>::foo::<B>`, in which case parameters
        //    may appear in two places. The penultimate segment,
        //    `SomeStruct::<A>`, contains parameters in TypeSpace, and the
        //    final segment, `foo::<B>` contains parameters in fn space.
        //
        // The first step then is to categorize the segments appropriately.

        let tcx = self.tcx();

        assert!(!segments.is_empty());
        let last = segments.len() - 1;

        let mut path_segs = vec![];

        match kind {
            // Case 1. Reference to a struct constructor.
            DefKind::Ctor(CtorOf::Struct, ..) => {
                // Everything but the final segment should have no
                // parameters at all.
                let generics = tcx.generics_of(def_id);
                // Variant and struct constructors use the
                // generics of their parent type definition.
                let generics_def_id = generics.parent.unwrap_or(def_id);
                path_segs.push(PathSeg(generics_def_id, last));
            }

            // Case 2. Reference to a variant constructor.
            DefKind::Ctor(CtorOf::Variant, ..) | DefKind::Variant => {
                let adt_def = self_ty.map(|t| t.ty_adt_def().unwrap());
                let (generics_def_id, index) = if let Some(adt_def) = adt_def {
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
                path_segs.push(PathSeg(generics_def_id, index));
            }

            // Case 3. Reference to a top-level value.
            DefKind::Fn | DefKind::Const | DefKind::ConstParam | DefKind::Static(_) => {
                path_segs.push(PathSeg(def_id, last));
            }

            // Case 4. Reference to a method or associated const.
            DefKind::AssocFn | DefKind::AssocConst => {
                if segments.len() >= 2 {
                    let generics = tcx.generics_of(def_id);
                    path_segs.push(PathSeg(generics.parent.unwrap(), last - 1));
                }
                path_segs.push(PathSeg(def_id, last));
            }

            kind => bug!("unexpected definition kind {:?} for {:?}", kind, def_id),
        }

        debug!("path_segs = {:?}", path_segs);

        path_segs
    }

    // Check a type `Path` and convert it to a `Ty`.
    pub fn res_to_ty(
        &self,
        opt_self_ty: Option<Ty<'tcx>>,
        path: &hir::Path<'_>,
        permit_variants: bool,
    ) -> Ty<'tcx> {
        let tcx = self.tcx();

        debug!(
            "res_to_ty(res={:?}, opt_self_ty={:?}, path_segments={:?})",
            path.res, opt_self_ty, path.segments
        );

        let span = path.span;
        match path.res {
            Res::Def(DefKind::OpaqueTy, did) => {
                // Check for desugared `impl Trait`.
                assert!(ty::is_impl_trait_defn(tcx, did).is_none());
                let item_segment = path.segments.split_last().unwrap();
                self.prohibit_generics(item_segment.1.iter(), |err| {
                    err.note("`impl Trait` types can't have type parameters");
                });
                let substs = self.ast_path_substs_for_ty(span, did, item_segment.0);
                self.normalize_ty(span, tcx.mk_opaque(did, substs))
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
                self.prohibit_generics(path.segments.split_last().unwrap().1.iter(), |_| {});
                self.ast_path_to_ty(span, did, path.segments.last().unwrap())
            }
            Res::Def(kind @ DefKind::Variant, def_id) if permit_variants => {
                // Convert "variant type" as if it were a real type.
                // The resulting `Ty` is type of the variant's enum for now.
                assert_eq!(opt_self_ty, None);

                let path_segs =
                    self.def_ids_for_value_path_segments(path.segments, None, kind, def_id);
                let generic_segs: FxHashSet<_> =
                    path_segs.iter().map(|PathSeg(_, index)| index).collect();
                self.prohibit_generics(
                    path.segments.iter().enumerate().filter_map(|(index, seg)| {
                        if !generic_segs.contains(&index) { Some(seg) } else { None }
                    }),
                    |err| {
                        err.note("enum variants can't have type parameters");
                    },
                );

                let PathSeg(def_id, index) = path_segs.last().unwrap();
                self.ast_path_to_ty(span, *def_id, &path.segments[*index])
            }
            Res::Def(DefKind::TyParam, def_id) => {
                assert_eq!(opt_self_ty, None);
                self.prohibit_generics(path.segments.iter(), |err| {
                    if let Some(span) = tcx.def_ident_span(def_id) {
                        let name = tcx.item_name(def_id);
                        err.span_note(span, &format!("type parameter `{name}` defined here"));
                    }
                });

                let def_id = def_id.expect_local();
                let item_def_id = tcx.hir().ty_param_owner(def_id);
                let generics = tcx.generics_of(item_def_id);
                let index = generics.param_def_id_to_index[&def_id.to_def_id()];
                tcx.mk_ty_param(index, tcx.hir().ty_param_name(def_id))
            }
            Res::SelfTy { trait_: Some(_), alias_to: None } => {
                // `Self` in trait or type alias.
                assert_eq!(opt_self_ty, None);
                self.prohibit_generics(path.segments.iter(), |err| {
                    if let [hir::PathSegment { args: Some(args), ident, .. }] = &path.segments[..] {
                        err.span_suggestion_verbose(
                            ident.span.shrink_to_hi().to(args.span_ext),
                            "the `Self` type doesn't accept type parameters",
                            "",
                            Applicability::MaybeIncorrect,
                        );
                    }
                });
                tcx.types.self_param
            }
            Res::SelfTy { trait_: _, alias_to: Some((def_id, forbid_generic)) } => {
                // `Self` in impl (we know the concrete type).
                assert_eq!(opt_self_ty, None);
                // Try to evaluate any array length constants.
                let ty = tcx.at(span).type_of(def_id);
                let span_of_impl = tcx.span_of_impl(def_id);
                self.prohibit_generics(path.segments.iter(), |err| {
                    let def_id = match *ty.kind() {
                        ty::Adt(self_def, _) => self_def.did(),
                        _ => return,
                    };

                    let type_name = tcx.item_name(def_id);
                    let span_of_ty = tcx.def_ident_span(def_id);
                    let generics = tcx.generics_of(def_id).count();

                    let msg = format!("`Self` is of type `{ty}`");
                    if let (Ok(i_sp), Some(t_sp)) = (span_of_impl, span_of_ty) {
                        let mut span: MultiSpan = vec![t_sp].into();
                        span.push_span_label(
                            i_sp,
                            &format!("`Self` is on type `{type_name}` in this `impl`"),
                        );
                        let mut postfix = "";
                        if generics == 0 {
                            postfix = ", which doesn't have generic parameters";
                        }
                        span.push_span_label(
                            t_sp,
                            &format!("`Self` corresponds to this type{postfix}"),
                        );
                        err.span_note(span, &msg);
                    } else {
                        err.note(&msg);
                    }
                    for segment in path.segments {
                        if let Some(args) = segment.args && segment.ident.name == kw::SelfUpper {
                            if generics == 0 {
                                // FIXME(estebank): we could also verify that the arguments being
                                // work for the `enum`, instead of just looking if it takes *any*.
                                err.span_suggestion_verbose(
                                    segment.ident.span.shrink_to_hi().to(args.span_ext),
                                    "the `Self` type doesn't accept type parameters",
                                    "",
                                    Applicability::MachineApplicable,
                                );
                                return;
                            } else {
                                err.span_suggestion_verbose(
                                    segment.ident.span,
                                    format!(
                                        "the `Self` type doesn't accept type parameters, use the \
                                        concrete type's name `{type_name}` instead if you want to \
                                        specify its type parameters"
                                    ),
                                    type_name,
                                    Applicability::MaybeIncorrect,
                                );
                            }
                        }
                    }
                });
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
                if forbid_generic && ty.needs_subst() {
                    let mut err = tcx.sess.struct_span_err(
                        path.span,
                        "generic `Self` types are currently not permitted in anonymous constants",
                    );
                    if let Some(hir::Node::Item(&hir::Item {
                        kind: hir::ItemKind::Impl(ref impl_),
                        ..
                    })) = tcx.hir().get_if_local(def_id)
                    {
                        err.span_note(impl_.self_ty.span, "not a concrete type");
                    }
                    err.emit();
                    tcx.ty_error()
                } else {
                    self.normalize_ty(span, ty)
                }
            }
            Res::Def(DefKind::AssocTy, def_id) => {
                debug_assert!(path.segments.len() >= 2);
                self.prohibit_generics(path.segments[..path.segments.len() - 2].iter(), |_| {});
                self.qpath_to_ty(
                    span,
                    opt_self_ty,
                    def_id,
                    &path.segments[path.segments.len() - 2],
                    path.segments.last().unwrap(),
                )
            }
            Res::PrimTy(prim_ty) => {
                assert_eq!(opt_self_ty, None);
                self.prohibit_generics(path.segments.iter(), |err| {
                    let name = prim_ty.name_str();
                    for segment in path.segments {
                        if let Some(args) = segment.args {
                            err.span_suggestion_verbose(
                                segment.ident.span.shrink_to_hi().to(args.span_ext),
                                &format!("primitive type `{name}` doesn't have generic parameters"),
                                "",
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                });
                match prim_ty {
                    hir::PrimTy::Bool => tcx.types.bool,
                    hir::PrimTy::Char => tcx.types.char,
                    hir::PrimTy::Int(it) => tcx.mk_mach_int(ty::int_ty(it)),
                    hir::PrimTy::Uint(uit) => tcx.mk_mach_uint(ty::uint_ty(uit)),
                    hir::PrimTy::Float(ft) => tcx.mk_mach_float(ty::float_ty(ft)),
                    hir::PrimTy::Str => tcx.types.str_,
                }
            }
            Res::Err => {
                self.set_tainted_by_errors();
                self.tcx().ty_error()
            }
            _ => span_bug!(span, "unexpected resolution: {:?}", path.res),
        }
    }

    /// Parses the programmer's textual representation of a type into our
    /// internal notion of a type.
    pub fn ast_ty_to_ty(&self, ast_ty: &hir::Ty<'_>) -> Ty<'tcx> {
        self.ast_ty_to_ty_inner(ast_ty, false, false)
    }

    /// Parses the programmer's textual representation of a type into our
    /// internal notion of a type.  This is meant to be used within a path.
    pub fn ast_ty_to_ty_in_path(&self, ast_ty: &hir::Ty<'_>) -> Ty<'tcx> {
        self.ast_ty_to_ty_inner(ast_ty, false, true)
    }

    /// Turns a `hir::Ty` into a `Ty`. For diagnostics' purposes we keep track of whether trait
    /// objects are borrowed like `&dyn Trait` to avoid emitting redundant errors.
    #[instrument(level = "debug", skip(self), ret)]
    fn ast_ty_to_ty_inner(&self, ast_ty: &hir::Ty<'_>, borrowed: bool, in_path: bool) -> Ty<'tcx> {
        let tcx = self.tcx();

        let result_ty = match ast_ty.kind {
            hir::TyKind::Slice(ref ty) => tcx.mk_slice(self.ast_ty_to_ty(ty)),
            hir::TyKind::Ptr(ref mt) => {
                tcx.mk_ptr(ty::TypeAndMut { ty: self.ast_ty_to_ty(mt.ty), mutbl: mt.mutbl })
            }
            hir::TyKind::Rptr(ref region, ref mt) => {
                let r = self.ast_region_to_region(region, None);
                debug!(?r);
                let t = self.ast_ty_to_ty_inner(mt.ty, true, false);
                tcx.mk_ref(r, ty::TypeAndMut { ty: t, mutbl: mt.mutbl })
            }
            hir::TyKind::Never => tcx.types.never,
            hir::TyKind::Tup(fields) => tcx.mk_tup(fields.iter().map(|t| self.ast_ty_to_ty(t))),
            hir::TyKind::BareFn(bf) => {
                require_c_abi_if_c_variadic(tcx, bf.decl, bf.abi, ast_ty.span);

                tcx.mk_fn_ptr(self.ty_of_fn(
                    ast_ty.hir_id,
                    bf.unsafety,
                    bf.abi,
                    bf.decl,
                    None,
                    Some(ast_ty),
                ))
            }
            hir::TyKind::TraitObject(bounds, ref lifetime, _) => {
                self.maybe_lint_bare_trait(ast_ty, in_path);
                self.conv_object_ty_poly_trait_ref(ast_ty.span, bounds, lifetime, borrowed)
            }
            hir::TyKind::Path(hir::QPath::Resolved(ref maybe_qself, ref path)) => {
                debug!(?maybe_qself, ?path);
                let opt_self_ty = maybe_qself.as_ref().map(|qself| self.ast_ty_to_ty(qself));
                self.res_to_ty(opt_self_ty, path, false)
            }
            hir::TyKind::OpaqueDef(item_id, lifetimes) => {
                let opaque_ty = tcx.hir().item(item_id);
                let def_id = item_id.def_id.to_def_id();

                match opaque_ty.kind {
                    hir::ItemKind::OpaqueTy(hir::OpaqueTy { origin, .. }) => {
                        self.impl_trait_ty_to_ty(def_id, lifetimes, origin)
                    }
                    ref i => bug!("`impl Trait` pointed to non-opaque type?? {:#?}", i),
                }
            }
            hir::TyKind::Path(hir::QPath::TypeRelative(ref qself, ref segment)) => {
                debug!(?qself, ?segment);
                let ty = self.ast_ty_to_ty_inner(qself, false, true);
                self.associated_path_to_ty(ast_ty.hir_id, ast_ty.span, ty, qself, segment, false)
                    .map(|(ty, _, _)| ty)
                    .unwrap_or_else(|_| tcx.ty_error())
            }
            hir::TyKind::Path(hir::QPath::LangItem(lang_item, span, _)) => {
                let def_id = tcx.require_lang_item(lang_item, Some(span));
                let (substs, _) = self.create_substs_for_ast_path(
                    span,
                    def_id,
                    &[],
                    &hir::PathSegment::invalid(),
                    &GenericArgs::none(),
                    true,
                    None,
                );
                EarlyBinder(self.normalize_ty(span, tcx.at(span).type_of(def_id)))
                    .subst(tcx, substs)
            }
            hir::TyKind::Array(ref ty, ref length) => {
                let length = match length {
                    &hir::ArrayLen::Infer(_, span) => self.ct_infer(tcx.types.usize, None, span),
                    hir::ArrayLen::Body(constant) => {
                        let length_def_id = tcx.hir().local_def_id(constant.hir_id);
                        ty::Const::from_anon_const(tcx, length_def_id)
                    }
                };

                let array_ty = tcx.mk_ty(ty::Array(self.ast_ty_to_ty(ty), length));
                self.normalize_ty(ast_ty.span, array_ty)
            }
            hir::TyKind::Typeof(ref e) => {
                let ty = tcx.type_of(tcx.hir().local_def_id(e.hir_id));
                let span = ast_ty.span;
                tcx.sess.emit_err(TypeofReservedKeywordUsed {
                    span,
                    ty,
                    opt_sugg: Some((span, Applicability::MachineApplicable))
                        .filter(|_| ty.is_suggestable(tcx, false)),
                });

                ty
            }
            hir::TyKind::Infer => {
                // Infer also appears as the type of arguments or return
                // values in an ExprKind::Closure, or as
                // the type of local variables. Both of these cases are
                // handled specially and will not descend into this routine.
                self.ty_infer(None, ast_ty.span)
            }
            hir::TyKind::Err => tcx.ty_error(),
        };

        self.record_ty(ast_ty.hir_id, result_ty, ast_ty.span);
        result_ty
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn impl_trait_ty_to_ty(
        &self,
        def_id: DefId,
        lifetimes: &[hir::GenericArg<'_>],
        origin: OpaqueTyOrigin,
    ) -> Ty<'tcx> {
        debug!("impl_trait_ty_to_ty(def_id={:?}, lifetimes={:?})", def_id, lifetimes);
        let tcx = self.tcx();

        let generics = tcx.generics_of(def_id);

        debug!("impl_trait_ty_to_ty: generics={:?}", generics);
        let substs = InternalSubsts::for_item(tcx, def_id, |param, _| {
            if let Some(i) = (param.index as usize).checked_sub(generics.parent_count) {
                // Our own parameters are the resolved lifetimes.
                if let GenericParamDefKind::Lifetime = param.kind {
                    if let hir::GenericArg::Lifetime(lifetime) = &lifetimes[i] {
                        self.ast_region_to_region(lifetime, None).into()
                    } else {
                        bug!()
                    }
                } else {
                    bug!()
                }
            } else {
                match param.kind {
                    // For RPIT (return position impl trait), only lifetimes
                    // mentioned in the impl Trait predicate are captured by
                    // the opaque type, so the lifetime parameters from the
                    // parent item need to be replaced with `'static`.
                    //
                    // For `impl Trait` in the types of statics, constants,
                    // locals and type aliases. These capture all parent
                    // lifetimes, so they can use their identity subst.
                    GenericParamDefKind::Lifetime
                        if matches!(
                            origin,
                            hir::OpaqueTyOrigin::FnReturn(..) | hir::OpaqueTyOrigin::AsyncFn(..)
                        ) =>
                    {
                        tcx.lifetimes.re_static.into()
                    }
                    _ => tcx.mk_param_from_def(param),
                }
            }
        });
        debug!("impl_trait_ty_to_ty: substs={:?}", substs);

        tcx.mk_opaque(def_id, substs)
    }

    pub fn ty_of_arg(&self, ty: &hir::Ty<'_>, expected_ty: Option<Ty<'tcx>>) -> Ty<'tcx> {
        match ty.kind {
            hir::TyKind::Infer if expected_ty.is_some() => {
                self.record_ty(ty.hir_id, expected_ty.unwrap(), ty.span);
                expected_ty.unwrap()
            }
            _ => self.ast_ty_to_ty(ty),
        }
    }

    pub fn ty_of_fn(
        &self,
        hir_id: hir::HirId,
        unsafety: hir::Unsafety,
        abi: abi::Abi,
        decl: &hir::FnDecl<'_>,
        generics: Option<&hir::Generics<'_>>,
        hir_ty: Option<&hir::Ty<'_>>,
    ) -> ty::PolyFnSig<'tcx> {
        debug!("ty_of_fn");

        let tcx = self.tcx();
        let bound_vars = tcx.late_bound_vars(hir_id);
        debug!(?bound_vars);

        // We proactively collect all the inferred type params to emit a single error per fn def.
        let mut visitor = HirPlaceholderCollector::default();
        let mut infer_replacements = vec![];

        if let Some(generics) = generics {
            walk_generics(&mut visitor, generics);
        }

        let input_tys: Vec<_> = decl
            .inputs
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if let hir::TyKind::Infer = a.kind && !self.allow_ty_infer() {
                    if let Some(suggested_ty) =
                        self.suggest_trait_fn_ty_for_impl_fn_infer(hir_id, Some(i))
                    {
                        infer_replacements.push((a.span, suggested_ty.to_string()));
                        return suggested_ty;
                    }
                }

                // Only visit the type looking for `_` if we didn't fix the type above
                visitor.visit_ty(a);
                self.ty_of_arg(a, None)
            })
            .collect();

        let output_ty = match decl.output {
            hir::FnRetTy::Return(output) => {
                if let hir::TyKind::Infer = output.kind
                    && !self.allow_ty_infer()
                    && let Some(suggested_ty) =
                        self.suggest_trait_fn_ty_for_impl_fn_infer(hir_id, None)
                {
                    infer_replacements.push((output.span, suggested_ty.to_string()));
                    suggested_ty
                } else {
                    visitor.visit_ty(output);
                    self.ast_ty_to_ty(output)
                }
            }
            hir::FnRetTy::DefaultReturn(..) => tcx.mk_unit(),
        };

        debug!("ty_of_fn: output_ty={:?}", output_ty);

        let fn_ty = tcx.mk_fn_sig(input_tys.into_iter(), output_ty, decl.c_variadic, unsafety, abi);
        let bare_fn_ty = ty::Binder::bind_with_vars(fn_ty, bound_vars);

        if !self.allow_ty_infer() && !(visitor.0.is_empty() && infer_replacements.is_empty()) {
            // We always collect the spans for placeholder types when evaluating `fn`s, but we
            // only want to emit an error complaining about them if infer types (`_`) are not
            // allowed. `allow_ty_infer` gates this behavior. We check for the presence of
            // `ident_span` to not emit an error twice when we have `fn foo(_: fn() -> _)`.

            let mut diag = crate::collect::placeholder_type_error_diag(
                tcx,
                generics,
                visitor.0,
                infer_replacements.iter().map(|(s, _)| *s).collect(),
                true,
                hir_ty,
                "function",
            );

            if !infer_replacements.is_empty() {
                diag.multipart_suggestion(
                    &format!(
                    "try replacing `_` with the type{} in the corresponding trait method signature",
                    rustc_errors::pluralize!(infer_replacements.len()),
                ),
                    infer_replacements,
                    Applicability::MachineApplicable,
                );
            }

            diag.emit();
        }

        // Find any late-bound regions declared in return type that do
        // not appear in the arguments. These are not well-formed.
        //
        // Example:
        //     for<'a> fn() -> &'a str <-- 'a is bad
        //     for<'a> fn(&'a String) -> &'a str <-- 'a is ok
        let inputs = bare_fn_ty.inputs();
        let late_bound_in_args =
            tcx.collect_constrained_late_bound_regions(&inputs.map_bound(|i| i.to_owned()));
        let output = bare_fn_ty.output();
        let late_bound_in_ret = tcx.collect_referenced_late_bound_regions(&output);

        self.validate_late_bound_regions(late_bound_in_args, late_bound_in_ret, |br_name| {
            struct_span_err!(
                tcx.sess,
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
    fn suggest_trait_fn_ty_for_impl_fn_infer(
        &self,
        fn_hir_id: hir::HirId,
        arg_idx: Option<usize>,
    ) -> Option<Ty<'tcx>> {
        let tcx = self.tcx();
        let hir = tcx.hir();

        let hir::Node::ImplItem(hir::ImplItem { kind: hir::ImplItemKind::Fn(..), ident, .. }) =
            hir.get(fn_hir_id) else { return None };
        let hir::Node::Item(hir::Item { kind: hir::ItemKind::Impl(i), .. }) =
                hir.get(hir.get_parent_node(fn_hir_id)) else { bug!("ImplItem should have Impl parent") };

        let trait_ref =
            self.instantiate_mono_trait_ref(i.of_trait.as_ref()?, self.ast_ty_to_ty(i.self_ty));

        let assoc = tcx.associated_items(trait_ref.def_id).find_by_name_and_kind(
            tcx,
            *ident,
            ty::AssocKind::Fn,
            trait_ref.def_id,
        )?;

        let fn_sig = tcx.bound_fn_sig(assoc.def_id).subst(
            tcx,
            trait_ref.substs.extend_to(tcx, assoc.def_id, |param, _| tcx.mk_param_from_def(param)),
        );

        let ty = if let Some(arg_idx) = arg_idx { fn_sig.input(arg_idx) } else { fn_sig.output() };

        Some(tcx.liberate_late_bound_regions(fn_hir_id.expect_owner().to_def_id(), ty))
    }

    fn validate_late_bound_regions(
        &self,
        constrained_regions: FxHashSet<ty::BoundRegionKind>,
        referenced_regions: FxHashSet<ty::BoundRegionKind>,
        generate_err: impl Fn(&str) -> DiagnosticBuilder<'tcx, ErrorGuaranteed>,
    ) {
        for br in referenced_regions.difference(&constrained_regions) {
            let br_name = match *br {
                ty::BrNamed(_, kw::UnderscoreLifetime) | ty::BrAnon(_) | ty::BrEnv => {
                    "an anonymous lifetime".to_string()
                }
                ty::BrNamed(_, name) => format!("lifetime `{}`", name),
            };

            let mut err = generate_err(&br_name);

            if let ty::BrNamed(_, kw::UnderscoreLifetime) | ty::BrAnon(_) = *br {
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
    /// use to summarize this type. The basic idea is that we will use the bound the user
    /// provided, if they provided one, and otherwise search the supertypes of trait bounds
    /// for region bounds. It may be that we can derive no bound at all, in which case
    /// we return `None`.
    fn compute_object_lifetime_bound(
        &self,
        span: Span,
        existential_predicates: &'tcx ty::List<ty::Binder<'tcx, ty::ExistentialPredicate<'tcx>>>,
    ) -> Option<ty::Region<'tcx>> // if None, use the default
    {
        let tcx = self.tcx();

        debug!("compute_opt_region_bound(existential_predicates={:?})", existential_predicates);

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
            tcx.sess.emit_err(AmbiguousLifetimeBound { span });
        }
        Some(r)
    }

    /// Make sure that we are in the condition to suggest the blanket implementation.
    fn maybe_lint_blanket_trait_impl(&self, self_ty: &hir::Ty<'_>, diag: &mut Diagnostic) {
        let tcx = self.tcx();
        let parent_id = tcx.hir().get_parent_item(self_ty.hir_id);
        if let hir::Node::Item(hir::Item {
            kind:
                hir::ItemKind::Impl(hir::Impl {
                    self_ty: impl_self_ty, of_trait: Some(of_trait_ref), generics, ..
                }),
            ..
        }) = tcx.hir().get_by_def_id(parent_id) && self_ty.hir_id == impl_self_ty.hir_id
        {
            if !of_trait_ref.trait_def_id().map_or(false, |def_id| def_id.is_local()) {
                return;
            }
            let of_trait_span = of_trait_ref.path.span;
            // make sure that we are not calling unwrap to abort during the compilation
            let Ok(impl_trait_name) = tcx.sess.source_map().span_to_snippet(self_ty.span) else { return; };
            let Ok(of_trait_name) = tcx.sess.source_map().span_to_snippet(of_trait_span) else { return; };
            // check if the trait has generics, to make a correct suggestion
            let param_name = generics.params.next_type_param_name(None);

            let add_generic_sugg = if let Some(span) = generics.span_for_param_suggestion() {
                (span, format!(", {}: {}", param_name, impl_trait_name))
            } else {
                (generics.span, format!("<{}: {}>", param_name, impl_trait_name))
            };
            diag.multipart_suggestion(
            format!("alternatively use a blanket \
                     implementation to implement `{of_trait_name}` for \
                     all types that also implement `{impl_trait_name}`"),
                vec![
                    (self_ty.span, param_name),
                    add_generic_sugg,
                ],
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn maybe_lint_bare_trait(&self, self_ty: &hir::Ty<'_>, in_path: bool) {
        let tcx = self.tcx();
        if let hir::TyKind::TraitObject([poly_trait_ref, ..], _, TraitObjectSyntax::None) =
            self_ty.kind
        {
            let needs_bracket = in_path
                && !tcx
                    .sess
                    .source_map()
                    .span_to_prev_source(self_ty.span)
                    .ok()
                    .map_or(false, |s| s.trim_end().ends_with('<'));

            let is_global = poly_trait_ref.trait_ref.path.is_global();
            let sugg = Vec::from_iter([
                (
                    self_ty.span.shrink_to_lo(),
                    format!(
                        "{}dyn {}",
                        if needs_bracket { "<" } else { "" },
                        if is_global { "(" } else { "" },
                    ),
                ),
                (
                    self_ty.span.shrink_to_hi(),
                    format!(
                        "{}{}",
                        if is_global { ")" } else { "" },
                        if needs_bracket { ">" } else { "" },
                    ),
                ),
            ]);
            if self_ty.span.edition() >= Edition::Edition2021 {
                let msg = "trait objects must include the `dyn` keyword";
                let label = "add `dyn` keyword before this trait";
                let mut diag =
                    rustc_errors::struct_span_err!(tcx.sess, self_ty.span, E0782, "{}", msg);
                diag.multipart_suggestion_verbose(label, sugg, Applicability::MachineApplicable);
                // check if the impl trait that we are considering is a impl of a local trait
                self.maybe_lint_blanket_trait_impl(&self_ty, &mut diag);
                diag.emit();
            } else {
                let msg = "trait objects without an explicit `dyn` are deprecated";
                tcx.struct_span_lint_hir(
                    BARE_TRAIT_OBJECTS,
                    self_ty.hir_id,
                    self_ty.span,
                    |lint| {
                        let mut diag = lint.build(msg);
                        diag.multipart_suggestion_verbose(
                            "use `dyn`",
                            sugg,
                            Applicability::MachineApplicable,
                        );
                        self.maybe_lint_blanket_trait_impl(&self_ty, &mut diag);
                        diag.emit();
                    },
                );
            }
        }
    }
}
