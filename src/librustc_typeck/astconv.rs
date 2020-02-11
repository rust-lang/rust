// ignore-tidy-filelength FIXME(#67418) Split up this file.
//! Conversion from AST representation of types to the `ty.rs` representation.
//! The main routine here is `ast_ty_to_ty()`; each use is parameterized by an
//! instance of `AstConv`.

// ignore-tidy-filelength

use crate::collect::PlaceholderHirTyCollector;
use crate::lint;
use crate::middle::lang_items::SizedTraitLangItem;
use crate::middle::resolve_lifetime as rl;
use crate::require_c_abi_if_c_variadic;
use crate::util::common::ErrorReported;
use rustc::lint::builtin::AMBIGUOUS_ASSOCIATED_ITEMS;
use rustc::session::{parse::feature_err, Session};
use rustc::ty::subst::{self, InternalSubsts, Subst, SubstsRef};
use rustc::ty::{self, Const, DefIdTree, ToPredicate, Ty, TyCtxt, TypeFoldable, WithConstness};
use rustc::ty::{GenericParamDef, GenericParamDefKind};
use rustc_ast::ast;
use rustc_ast::util::lev_distance::find_best_match_for_name;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{pluralize, struct_span_err, Applicability, DiagnosticId};
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Namespace, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::Visitor;
use rustc_hir::print;
use rustc_hir::{Constness, ExprKind, GenericArg, GenericArgs};
use rustc_span::symbol::sym;
use rustc_span::{MultiSpan, Span, DUMMY_SP};
use rustc_target::spec::abi;
use rustc_trait_selection::traits;
use rustc_trait_selection::traits::astconv_object_safety_violations;
use rustc_trait_selection::traits::error_reporting::report_object_safety_error;
use rustc_trait_selection::traits::wf::object_region_bounds;
use smallvec::SmallVec;

use std::collections::BTreeSet;
use std::iter;
use std::slice;

use rustc::mir::interpret::LitToConstInput;

#[derive(Debug)]
pub struct PathSeg(pub DefId, pub usize);

pub trait AstConv<'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'tcx>;

    fn item_def_id(&self) -> Option<DefId>;

    fn default_constness_for_trait_bounds(&self) -> Constness;

    /// Returns predicates in scope of the form `X: Foo`, where `X` is
    /// a type parameter `X` with the given id `def_id`. This is a
    /// subset of the full set of predicates.
    ///
    /// This is used for one specific purpose: resolving "short-hand"
    /// associated type references like `T::Item`. In principle, we
    /// would do that by first getting the full set of predicates in
    /// scope and then filtering down to find those that apply to `T`,
    /// but this can lead to cycle errors. The problem is that we have
    /// to do this resolution *in order to create the predicates in
    /// the first place*. Hence, we have this "special pass".
    fn get_type_parameter_bounds(&self, span: Span, def_id: DefId) -> ty::GenericPredicates<'tcx>;

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
    ) -> &'tcx Const<'tcx>;

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

pub enum SizedByDefault {
    Yes,
    No,
}

struct ConvertedBinding<'a, 'tcx> {
    item_name: ast::Ident,
    kind: ConvertedBindingKind<'a, 'tcx>,
    span: Span,
}

enum ConvertedBindingKind<'a, 'tcx> {
    Equality(Ty<'tcx>),
    Constraint(&'a [hir::GenericBound<'a>]),
}

#[derive(PartialEq)]
enum GenericArgPosition {
    Type,
    Value, // e.g., functions
    MethodCall,
}

/// A marker denoting that the generic arguments that were
/// provided did not match the respective generic parameters.
pub struct GenericArgCountMismatch {
    /// Indicates whether a fatal error was reported (`Some`), or just a lint (`None`).
    pub reported: Option<ErrorReported>,
    /// A list of spans of arguments provided that were not valid.
    pub invalid_args: Vec<Span>,
}

impl<'o, 'tcx> dyn AstConv<'tcx> + 'o {
    pub fn ast_region_to_region(
        &self,
        lifetime: &hir::Lifetime,
        def: Option<&ty::GenericParamDef>,
    ) -> ty::Region<'tcx> {
        let tcx = self.tcx();
        let lifetime_name = |def_id| tcx.hir().name(tcx.hir().as_local_hir_id(def_id).unwrap());

        let r = match tcx.named_region(lifetime.hir_id) {
            Some(rl::Region::Static) => tcx.lifetimes.re_static,

            Some(rl::Region::LateBound(debruijn, id, _)) => {
                let name = lifetime_name(id);
                tcx.mk_region(ty::ReLateBound(debruijn, ty::BrNamed(id, name)))
            }

            Some(rl::Region::LateBoundAnon(debruijn, index)) => {
                tcx.mk_region(ty::ReLateBound(debruijn, ty::BrAnon(index)))
            }

            Some(rl::Region::EarlyBound(index, id, _)) => {
                let name = lifetime_name(id);
                tcx.mk_region(ty::ReEarlyBound(ty::EarlyBoundRegion { def_id: id, index, name }))
            }

            Some(rl::Region::Free(scope, id)) => {
                let name = lifetime_name(id);
                tcx.mk_region(ty::ReFree(ty::FreeRegion {
                    scope,
                    bound_region: ty::BrNamed(id, name),
                }))

                // (*) -- not late-bound, won't change
            }

            None => {
                self.re_infer(def, lifetime.span).unwrap_or_else(|| {
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
        };

        debug!("ast_region_to_region(lifetime={:?}) yields {:?}", lifetime, r);

        r
    }

    /// Given a path `path` that refers to an item `I` with the declared generics `decl_generics`,
    /// returns an appropriate set of substitutions for this particular reference to `I`.
    pub fn ast_path_substs_for_ty(
        &self,
        span: Span,
        def_id: DefId,
        item_segment: &hir::PathSegment<'_>,
    ) -> SubstsRef<'tcx> {
        let (substs, assoc_bindings, _) = self.create_substs_for_ast_path(
            span,
            def_id,
            &[],
            item_segment.generic_args(),
            item_segment.infer_args,
            None,
        );

        assoc_bindings.first().map(|b| Self::prohibit_assoc_ty_binding(self.tcx(), b.span));

        substs
    }

    /// Report error if there is an explicit type parameter when using `impl Trait`.
    fn check_impl_trait(
        tcx: TyCtxt<'_>,
        seg: &hir::PathSegment<'_>,
        generics: &ty::Generics,
    ) -> bool {
        let explicit = !seg.infer_args;
        let impl_trait = generics.params.iter().any(|param| match param.kind {
            ty::GenericParamDefKind::Type {
                synthetic: Some(hir::SyntheticTyParamKind::ImplTrait),
                ..
            } => true,
            _ => false,
        });

        if explicit && impl_trait {
            let spans = seg
                .generic_args()
                .args
                .iter()
                .filter_map(|arg| match arg {
                    GenericArg::Type(_) => Some(arg.span()),
                    _ => None,
                })
                .collect::<Vec<_>>();

            let mut err = struct_span_err! {
                tcx.sess,
                spans.clone(),
                E0632,
                "cannot provide explicit generic arguments when `impl Trait` is \
                used in argument position"
            };

            for span in spans {
                err.span_label(span, "explicit generic argument not allowed");
            }

            err.emit();
        }

        impl_trait
    }

    /// Checks that the correct number of generic arguments have been provided.
    /// Used specifically for function calls.
    pub fn check_generic_arg_count_for_call(
        tcx: TyCtxt<'_>,
        span: Span,
        def: &ty::Generics,
        seg: &hir::PathSegment<'_>,
        is_method_call: bool,
    ) -> Result<(), GenericArgCountMismatch> {
        let empty_args = hir::GenericArgs::none();
        let suppress_mismatch = Self::check_impl_trait(tcx, seg, &def);
        Self::check_generic_arg_count(
            tcx,
            span,
            def,
            if let Some(ref args) = seg.args { args } else { &empty_args },
            if is_method_call { GenericArgPosition::MethodCall } else { GenericArgPosition::Value },
            def.parent.is_none() && def.has_self, // `has_self`
            seg.infer_args || suppress_mismatch,  // `infer_args`
        )
    }

    /// Checks that the correct number of generic arguments have been provided.
    /// This is used both for datatypes and function calls.
    fn check_generic_arg_count(
        tcx: TyCtxt<'_>,
        span: Span,
        def: &ty::Generics,
        args: &hir::GenericArgs<'_>,
        position: GenericArgPosition,
        has_self: bool,
        infer_args: bool,
    ) -> Result<(), GenericArgCountMismatch> {
        // At this stage we are guaranteed that the generic arguments are in the correct order, e.g.
        // that lifetimes will proceed types. So it suffices to check the number of each generic
        // arguments in order to validate them with respect to the generic parameters.
        let param_counts = def.own_counts();
        let arg_counts = args.own_counts();
        let infer_lifetimes = position != GenericArgPosition::Type && arg_counts.lifetimes == 0;

        let mut defaults: ty::GenericParamCount = Default::default();
        for param in &def.params {
            match param.kind {
                GenericParamDefKind::Lifetime => {}
                GenericParamDefKind::Type { has_default, .. } => {
                    defaults.types += has_default as usize
                }
                GenericParamDefKind::Const => {
                    // FIXME(const_generics:defaults)
                }
            };
        }

        if position != GenericArgPosition::Type && !args.bindings.is_empty() {
            AstConv::prohibit_assoc_ty_binding(tcx, args.bindings[0].span);
        }

        // Prohibit explicit lifetime arguments if late-bound lifetime parameters are present.
        let mut explicit_lifetimes = Ok(());
        if !infer_lifetimes {
            if let Some(span_late) = def.has_late_bound_regions {
                let msg = "cannot specify lifetime arguments explicitly \
                           if late bound lifetime parameters are present";
                let note = "the late bound lifetime parameter is introduced here";
                let span = args.args[0].span();
                if position == GenericArgPosition::Value
                    && arg_counts.lifetimes != param_counts.lifetimes
                {
                    explicit_lifetimes = Err(true);
                    let mut err = tcx.sess.struct_span_err(span, msg);
                    err.span_note(span_late, note);
                    err.emit();
                } else {
                    explicit_lifetimes = Err(false);
                    let mut multispan = MultiSpan::from_span(span);
                    multispan.push_span_label(span_late, note.to_string());
                    tcx.struct_span_lint_hir(
                        lint::builtin::LATE_BOUND_LIFETIME_ARGUMENTS,
                        args.args[0].id(),
                        multispan,
                        |lint| lint.build(msg).emit(),
                    );
                }
            }
        }

        let check_kind_count =
            |kind, required, permitted, provided, offset, unexpected_spans: &mut Vec<Span>| {
                debug!(
                    "check_kind_count: kind: {} required: {} permitted: {} provided: {} offset: {}",
                    kind, required, permitted, provided, offset
                );
                // We enforce the following: `required` <= `provided` <= `permitted`.
                // For kinds without defaults (e.g.., lifetimes), `required == permitted`.
                // For other kinds (i.e., types), `permitted` may be greater than `required`.
                if required <= provided && provided <= permitted {
                    return Ok(());
                }

                // Unfortunately lifetime and type parameter mismatches are typically styled
                // differently in diagnostics, which means we have a few cases to consider here.
                let (bound, quantifier) = if required != permitted {
                    if provided < required {
                        (required, "at least ")
                    } else {
                        // provided > permitted
                        (permitted, "at most ")
                    }
                } else {
                    (required, "")
                };

                let (spans, label) = if required == permitted && provided > permitted {
                    // In the case when the user has provided too many arguments,
                    // we want to point to the unexpected arguments.
                    let spans: Vec<Span> = args.args[offset + permitted..offset + provided]
                        .iter()
                        .map(|arg| arg.span())
                        .collect();
                    unexpected_spans.extend(spans.clone());
                    (spans, format!("unexpected {} argument", kind))
                } else {
                    (
                        vec![span],
                        format!(
                            "expected {}{} {} argument{}",
                            quantifier,
                            bound,
                            kind,
                            pluralize!(bound),
                        ),
                    )
                };

                let mut err = tcx.sess.struct_span_err_with_code(
                    spans.clone(),
                    &format!(
                        "wrong number of {} arguments: expected {}{}, found {}",
                        kind, quantifier, bound, provided,
                    ),
                    DiagnosticId::Error("E0107".into()),
                );
                for span in spans {
                    err.span_label(span, label.as_str());
                }
                err.emit();

                Err(true)
            };

        let mut arg_count_correct = explicit_lifetimes;
        let mut unexpected_spans = vec![];

        if arg_count_correct.is_ok()
            && (!infer_lifetimes || arg_counts.lifetimes > param_counts.lifetimes)
        {
            arg_count_correct = check_kind_count(
                "lifetime",
                param_counts.lifetimes,
                param_counts.lifetimes,
                arg_counts.lifetimes,
                0,
                &mut unexpected_spans,
            )
            .and(arg_count_correct);
        }
        // FIXME(const_generics:defaults)
        if !infer_args || arg_counts.consts > param_counts.consts {
            arg_count_correct = check_kind_count(
                "const",
                param_counts.consts,
                param_counts.consts,
                arg_counts.consts,
                arg_counts.lifetimes + arg_counts.types,
                &mut unexpected_spans,
            )
            .and(arg_count_correct);
        }
        // Note that type errors are currently be emitted *after* const errors.
        if !infer_args || arg_counts.types > param_counts.types - defaults.types - has_self as usize
        {
            arg_count_correct = check_kind_count(
                "type",
                param_counts.types - defaults.types - has_self as usize,
                param_counts.types - has_self as usize,
                arg_counts.types,
                arg_counts.lifetimes,
                &mut unexpected_spans,
            )
            .and(arg_count_correct);
        }

        arg_count_correct.map_err(|reported_err| GenericArgCountMismatch {
            reported: if reported_err { Some(ErrorReported) } else { None },
            invalid_args: unexpected_spans,
        })
    }

    /// Report an error that a generic argument did not match the generic parameter that was
    /// expected.
    fn generic_arg_mismatch_err(sess: &Session, arg: &GenericArg<'_>, kind: &'static str) {
        let mut err = struct_span_err!(
            sess,
            arg.span(),
            E0747,
            "{} provided when a {} was expected",
            arg.descr(),
            kind,
        );
        // This note will be true as long as generic parameters are strictly ordered by their kind.
        err.note(&format!("{} arguments must be provided before {} arguments", kind, arg.descr()));
        err.emit();
    }

    /// Creates the relevant generic argument substitutions
    /// corresponding to a set of generic parameters. This is a
    /// rather complex function. Let us try to explain the role
    /// of each of its parameters:
    ///
    /// To start, we are given the `def_id` of the thing we are
    /// creating the substitutions for, and a partial set of
    /// substitutions `parent_substs`. In general, the substitutions
    /// for an item begin with substitutions for all the "parents" of
    /// that item -- e.g., for a method it might include the
    /// parameters from the impl.
    ///
    /// Therefore, the method begins by walking down these parents,
    /// starting with the outermost parent and proceed inwards until
    /// it reaches `def_id`. For each parent `P`, it will check `parent_substs`
    /// first to see if the parent's substitutions are listed in there. If so,
    /// we can append those and move on. Otherwise, it invokes the
    /// three callback functions:
    ///
    /// - `args_for_def_id`: given the `DefId` `P`, supplies back the
    ///   generic arguments that were given to that parent from within
    ///   the path; so e.g., if you have `<T as Foo>::Bar`, the `DefId`
    ///   might refer to the trait `Foo`, and the arguments might be
    ///   `[T]`. The boolean value indicates whether to infer values
    ///   for arguments whose values were not explicitly provided.
    /// - `provided_kind`: given the generic parameter and the value from `args_for_def_id`,
    ///   instantiate a `GenericArg`.
    /// - `inferred_kind`: if no parameter was provided, and inference is enabled, then
    ///   creates a suitable inference variable.
    pub fn create_substs_for_generic_args<'b>(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        parent_substs: &[subst::GenericArg<'tcx>],
        has_self: bool,
        self_ty: Option<Ty<'tcx>>,
        arg_count_correct: bool,
        args_for_def_id: impl Fn(DefId) -> (Option<&'b GenericArgs<'b>>, bool),
        mut provided_kind: impl FnMut(&GenericParamDef, &GenericArg<'_>) -> subst::GenericArg<'tcx>,
        mut inferred_kind: impl FnMut(
            Option<&[subst::GenericArg<'tcx>]>,
            &GenericParamDef,
            bool,
        ) -> subst::GenericArg<'tcx>,
    ) -> SubstsRef<'tcx> {
        // Collect the segments of the path; we need to substitute arguments
        // for parameters throughout the entire path (wherever there are
        // generic parameters).
        let mut parent_defs = tcx.generics_of(def_id);
        let count = parent_defs.count();
        let mut stack = vec![(def_id, parent_defs)];
        while let Some(def_id) = parent_defs.parent {
            parent_defs = tcx.generics_of(def_id);
            stack.push((def_id, parent_defs));
        }

        // We manually build up the substitution, rather than using convenience
        // methods in `subst.rs`, so that we can iterate over the arguments and
        // parameters in lock-step linearly, instead of trying to match each pair.
        let mut substs: SmallVec<[subst::GenericArg<'tcx>; 8]> = SmallVec::with_capacity(count);
        // Iterate over each segment of the path.
        while let Some((def_id, defs)) = stack.pop() {
            let mut params = defs.params.iter().peekable();

            // If we have already computed substitutions for parents, we can use those directly.
            while let Some(&param) = params.peek() {
                if let Some(&kind) = parent_substs.get(param.index as usize) {
                    substs.push(kind);
                    params.next();
                } else {
                    break;
                }
            }

            // `Self` is handled first, unless it's been handled in `parent_substs`.
            if has_self {
                if let Some(&param) = params.peek() {
                    if param.index == 0 {
                        if let GenericParamDefKind::Type { .. } = param.kind {
                            substs.push(
                                self_ty
                                    .map(|ty| ty.into())
                                    .unwrap_or_else(|| inferred_kind(None, param, true)),
                            );
                            params.next();
                        }
                    }
                }
            }

            // Check whether this segment takes generic arguments and the user has provided any.
            let (generic_args, infer_args) = args_for_def_id(def_id);

            let mut args =
                generic_args.iter().flat_map(|generic_args| generic_args.args.iter()).peekable();

            // If we encounter a type or const when we expect a lifetime, we infer the lifetimes.
            // If we later encounter a lifetime, we know that the arguments were provided in the
            // wrong order. `force_infer_lt` records the type or const that forced lifetimes to be
            // inferred, so we can use it for diagnostics later.
            let mut force_infer_lt = None;

            loop {
                // We're going to iterate through the generic arguments that the user
                // provided, matching them with the generic parameters we expect.
                // Mismatches can occur as a result of elided lifetimes, or for malformed
                // input. We try to handle both sensibly.
                match (args.peek(), params.peek()) {
                    (Some(&arg), Some(&param)) => {
                        match (arg, &param.kind) {
                            (GenericArg::Lifetime(_), GenericParamDefKind::Lifetime)
                            | (GenericArg::Type(_), GenericParamDefKind::Type { .. })
                            | (GenericArg::Const(_), GenericParamDefKind::Const) => {
                                substs.push(provided_kind(param, arg));
                                args.next();
                                params.next();
                            }
                            (GenericArg::Type(_), GenericParamDefKind::Lifetime)
                            | (GenericArg::Const(_), GenericParamDefKind::Lifetime) => {
                                // We expected a lifetime argument, but got a type or const
                                // argument. That means we're inferring the lifetimes.
                                substs.push(inferred_kind(None, param, infer_args));
                                force_infer_lt = Some(arg);
                                params.next();
                            }
                            (_, kind) => {
                                // We expected one kind of parameter, but the user provided
                                // another. This is an error. However, if we already know that
                                // the arguments don't match up with the parameters, we won't issue
                                // an additional error, as the user already knows what's wrong.
                                if arg_count_correct {
                                    Self::generic_arg_mismatch_err(tcx.sess, arg, kind.descr());
                                }

                                // We've reported the error, but we want to make sure that this
                                // problem doesn't bubble down and create additional, irrelevant
                                // errors. In this case, we're simply going to ignore the argument
                                // and any following arguments. The rest of the parameters will be
                                // inferred.
                                while args.next().is_some() {}
                            }
                        }
                    }

                    (Some(&arg), None) => {
                        // We should never be able to reach this point with well-formed input.
                        // There are two situations in which we can encounter this issue.
                        //
                        //  1.  The number of arguments is incorrect. In this case, an error
                        //      will already have been emitted, and we can ignore it. This case
                        //      also occurs when late-bound lifetime parameters are present, yet
                        //      the lifetime arguments have also been explicitly specified by the
                        //      user.
                        //  2.  We've inferred some lifetimes, which have been provided later (i.e.
                        //      after a type or const). We want to throw an error in this case.

                        if arg_count_correct {
                            let kind = arg.descr();
                            assert_eq!(kind, "lifetime");
                            let provided =
                                force_infer_lt.expect("lifetimes ought to have been inferred");
                            Self::generic_arg_mismatch_err(tcx.sess, provided, kind);
                        }

                        break;
                    }

                    (None, Some(&param)) => {
                        // If there are fewer arguments than parameters, it means
                        // we're inferring the remaining arguments.
                        substs.push(inferred_kind(Some(&substs), param, infer_args));
                        params.next();
                    }

                    (None, None) => break,
                }
            }
        }

        tcx.intern_substs(&substs)
    }

    /// Given the type/lifetime/const arguments provided to some path (along with
    /// an implicit `Self`, if this is a trait reference), returns the complete
    /// set of substitutions. This may involve applying defaulted type parameters.
    /// Also returns back constraints on associated types.
    ///
    /// Example:
    ///
    /// ```
    /// T: std::ops::Index<usize, Output = u32>
    /// ^1 ^^^^^^^^^^^^^^2 ^^^^3  ^^^^^^^^^^^4
    /// ```
    ///
    /// 1. The `self_ty` here would refer to the type `T`.
    /// 2. The path in question is the path to the trait `std::ops::Index`,
    ///    which will have been resolved to a `def_id`
    /// 3. The `generic_args` contains info on the `<...>` contents. The `usize` type
    ///    parameters are returned in the `SubstsRef`, the associated type bindings like
    ///    `Output = u32` are returned in the `Vec<ConvertedBinding...>` result.
    ///
    /// Note that the type listing given here is *exactly* what the user provided.
    ///
    /// For (generic) associated types
    ///
    /// ```
    /// <Vec<u8> as Iterable<u8>>::Iter::<'a>
    /// ```
    ///
    /// We have the parent substs are the substs for the parent trait:
    /// `[Vec<u8>, u8]` and `generic_args` are the arguments for the associated
    /// type itself: `['a]`. The returned `SubstsRef` concatenates these two
    /// lists: `[Vec<u8>, u8, 'a]`.
    fn create_substs_for_ast_path<'a>(
        &self,
        span: Span,
        def_id: DefId,
        parent_substs: &[subst::GenericArg<'tcx>],
        generic_args: &'a hir::GenericArgs<'_>,
        infer_args: bool,
        self_ty: Option<Ty<'tcx>>,
    ) -> (SubstsRef<'tcx>, Vec<ConvertedBinding<'a, 'tcx>>, Result<(), GenericArgCountMismatch>)
    {
        // If the type is parameterized by this region, then replace this
        // region with the current anon region binding (in other words,
        // whatever & would get replaced with).
        debug!(
            "create_substs_for_ast_path(def_id={:?}, self_ty={:?}, \
                generic_args={:?})",
            def_id, self_ty, generic_args
        );

        let tcx = self.tcx();
        let generic_params = tcx.generics_of(def_id);

        if generic_params.has_self {
            if generic_params.parent.is_some() {
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

        let arg_count_correct = Self::check_generic_arg_count(
            tcx,
            span,
            &generic_params,
            &generic_args,
            GenericArgPosition::Type,
            self_ty.is_some(),
            infer_args,
        );

        let is_object = self_ty.map_or(false, |ty| ty == self.tcx().types.trait_object_dummy_self);
        let default_needs_object_self = |param: &ty::GenericParamDef| {
            if let GenericParamDefKind::Type { has_default, .. } = param.kind {
                if is_object && has_default {
                    let self_param = tcx.types.self_param;
                    if tcx.at(span).type_of(param.def_id).walk().any(|ty| ty == self_param) {
                        // There is no suitable inference default for a type parameter
                        // that references self, in an object type.
                        return true;
                    }
                }
            }

            false
        };

        let mut missing_type_params = vec![];
        let mut inferred_params = vec![];
        let substs = Self::create_substs_for_generic_args(
            tcx,
            def_id,
            parent_substs,
            self_ty.is_some(),
            self_ty,
            arg_count_correct.is_ok(),
            // Provide the generic args, and whether types should be inferred.
            |did| {
                if did == def_id {
                    (Some(generic_args), infer_args)
                } else {
                    // The last component of this tuple is unimportant.
                    (None, false)
                }
            },
            // Provide substitutions for parameters for which (valid) arguments have been provided.
            |param, arg| match (&param.kind, arg) {
                (GenericParamDefKind::Lifetime, GenericArg::Lifetime(lt)) => {
                    self.ast_region_to_region(&lt, Some(param)).into()
                }
                (GenericParamDefKind::Type { .. }, GenericArg::Type(ty)) => {
                    if let (hir::TyKind::Infer, false) = (&ty.kind, self.allow_ty_infer()) {
                        inferred_params.push(ty.span);
                        tcx.types.err.into()
                    } else {
                        self.ast_ty_to_ty(&ty).into()
                    }
                }
                (GenericParamDefKind::Const, GenericArg::Const(ct)) => {
                    self.ast_const_to_const(&ct.value, tcx.type_of(param.def_id)).into()
                }
                _ => unreachable!(),
            },
            // Provide substitutions for parameters for which arguments are inferred.
            |substs, param, infer_args| {
                match param.kind {
                    GenericParamDefKind::Lifetime => tcx.lifetimes.re_static.into(),
                    GenericParamDefKind::Type { has_default, .. } => {
                        if !infer_args && has_default {
                            // No type parameter provided, but a default exists.

                            // If we are converting an object type, then the
                            // `Self` parameter is unknown. However, some of the
                            // other type parameters may reference `Self` in their
                            // defaults. This will lead to an ICE if we are not
                            // careful!
                            if default_needs_object_self(param) {
                                missing_type_params.push(param.name.to_string());
                                tcx.types.err.into()
                            } else {
                                // This is a default type parameter.
                                self.normalize_ty(
                                    span,
                                    tcx.at(span).type_of(param.def_id).subst_spanned(
                                        tcx,
                                        substs.unwrap(),
                                        Some(span),
                                    ),
                                )
                                .into()
                            }
                        } else if infer_args {
                            // No type parameters were provided, we can infer all.
                            let param =
                                if !default_needs_object_self(param) { Some(param) } else { None };
                            self.ty_infer(param, span).into()
                        } else {
                            // We've already errored above about the mismatch.
                            tcx.types.err.into()
                        }
                    }
                    GenericParamDefKind::Const => {
                        // FIXME(const_generics:defaults)
                        if infer_args {
                            // No const parameters were provided, we can infer all.
                            let ty = tcx.at(span).type_of(param.def_id);
                            self.ct_infer(ty, Some(param), span).into()
                        } else {
                            // We've already errored above about the mismatch.
                            tcx.consts.err.into()
                        }
                    }
                }
            },
        );
        if !inferred_params.is_empty() {
            // We always collect the spans for placeholder types when evaluating `fn`s, but we
            // only want to emit an error complaining about them if infer types (`_`) are not
            // allowed. `allow_ty_infer` gates this behavior.
            crate::collect::placeholder_type_error(
                tcx,
                inferred_params[0],
                &[],
                inferred_params,
                false,
            );
        }

        self.complain_about_missing_type_params(
            missing_type_params,
            def_id,
            span,
            generic_args.args.is_empty(),
        );

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
                    hir::TypeBindingKind::Equality { ref ty } => {
                        ConvertedBindingKind::Equality(self.ast_ty_to_ty(ty))
                    }
                    hir::TypeBindingKind::Constraint { ref bounds } => {
                        ConvertedBindingKind::Constraint(bounds)
                    }
                };
                ConvertedBinding { item_name: binding.ident, kind, span: binding.span }
            })
            .collect();

        debug!(
            "create_substs_for_ast_path(generic_params={:?}, self_ty={:?}) -> {:?}",
            generic_params, self_ty, substs
        );

        (substs, assoc_bindings, arg_count_correct)
    }

    crate fn create_substs_for_associated_item(
        &self,
        tcx: TyCtxt<'tcx>,
        span: Span,
        item_def_id: DefId,
        item_segment: &hir::PathSegment<'_>,
        parent_substs: SubstsRef<'tcx>,
    ) -> SubstsRef<'tcx> {
        if tcx.generics_of(item_def_id).params.is_empty() {
            self.prohibit_generics(slice::from_ref(item_segment));

            parent_substs
        } else {
            self.create_substs_for_ast_path(
                span,
                item_def_id,
                parent_substs,
                item_segment.generic_args(),
                item_segment.infer_args,
                None,
            )
            .0
        }
    }

    /// On missing type parameters, emit an E0393 error and provide a structured suggestion using
    /// the type parameter's name as a placeholder.
    fn complain_about_missing_type_params(
        &self,
        missing_type_params: Vec<String>,
        def_id: DefId,
        span: Span,
        empty_generic_args: bool,
    ) {
        if missing_type_params.is_empty() {
            return;
        }
        let display =
            missing_type_params.iter().map(|n| format!("`{}`", n)).collect::<Vec<_>>().join(", ");
        let mut err = struct_span_err!(
            self.tcx().sess,
            span,
            E0393,
            "the type parameter{} {} must be explicitly specified",
            pluralize!(missing_type_params.len()),
            display,
        );
        err.span_label(
            self.tcx().def_span(def_id),
            &format!(
                "type parameter{} {} must be specified for this",
                pluralize!(missing_type_params.len()),
                display,
            ),
        );
        let mut suggested = false;
        if let (Ok(snippet), true) = (
            self.tcx().sess.source_map().span_to_snippet(span),
            // Don't suggest setting the type params if there are some already: the order is
            // tricky to get right and the user will already know what the syntax is.
            empty_generic_args,
        ) {
            if snippet.ends_with('>') {
                // The user wrote `Trait<'a, T>` or similar. To provide an accurate suggestion
                // we would have to preserve the right order. For now, as clearly the user is
                // aware of the syntax, we do nothing.
            } else {
                // The user wrote `Iterator`, so we don't have a type we can suggest, but at
                // least we can clue them to the correct syntax `Iterator<Type>`.
                err.span_suggestion(
                    span,
                    &format!(
                        "set the type parameter{plural} to the desired type{plural}",
                        plural = pluralize!(missing_type_params.len()),
                    ),
                    format!("{}<{}>", snippet, missing_type_params.join(", ")),
                    Applicability::HasPlaceholders,
                );
                suggested = true;
            }
        }
        if !suggested {
            err.span_label(
                span,
                format!(
                    "missing reference{} to {}",
                    pluralize!(missing_type_params.len()),
                    display,
                ),
            );
        }
        err.note(
            "because of the default `Self` reference, type parameters must be \
                  specified on object types",
        );
        err.emit();
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
        self.prohibit_generics(trait_ref.path.segments.split_last().unwrap().1);

        self.ast_path_to_mono_trait_ref(
            trait_ref.path.span,
            trait_ref.trait_def_id(),
            self_ty,
            trait_ref.path.segments.last().unwrap(),
        )
    }

    /// The given trait-ref must actually be a trait.
    pub(super) fn instantiate_poly_trait_ref_inner(
        &self,
        trait_ref: &hir::TraitRef<'_>,
        span: Span,
        constness: Constness,
        self_ty: Ty<'tcx>,
        bounds: &mut Bounds<'tcx>,
        speculative: bool,
    ) -> Result<(), GenericArgCountMismatch> {
        let trait_def_id = trait_ref.trait_def_id();

        debug!("instantiate_poly_trait_ref({:?}, def_id={:?})", trait_ref, trait_def_id);

        self.prohibit_generics(trait_ref.path.segments.split_last().unwrap().1);

        let path_span = if let [segment] = &trait_ref.path.segments[..] {
            // FIXME: `trait_ref.path.span` can point to a full path with multiple
            // segments, even though `trait_ref.path.segments` is of length `1`. Work
            // around that bug here, even though it should be fixed elsewhere.
            // This would otherwise cause an invalid suggestion. For an example, look at
            // `src/test/ui/issues/issue-28344.rs`.
            segment.ident.span
        } else {
            trait_ref.path.span
        };
        let (substs, assoc_bindings, arg_count_correct) = self.create_substs_for_ast_trait_ref(
            path_span,
            trait_def_id,
            self_ty,
            trait_ref.path.segments.last().unwrap(),
        );
        let poly_trait_ref = ty::Binder::bind(ty::TraitRef::new(trait_def_id, substs));

        bounds.trait_bounds.push((poly_trait_ref, span, constness));

        let mut dup_bindings = FxHashMap::default();
        for binding in &assoc_bindings {
            // Specify type to assert that error was already reported in `Err` case.
            let _: Result<_, ErrorReported> = self.add_predicates_for_ast_type_binding(
                trait_ref.hir_ref_id,
                poly_trait_ref,
                binding,
                bounds,
                speculative,
                &mut dup_bindings,
                span,
            );
            // Okay to ignore `Err` because of `ErrorReported` (see above).
        }

        debug!(
            "instantiate_poly_trait_ref({:?}, bounds={:?}) -> {:?}",
            trait_ref, bounds, poly_trait_ref
        );

        arg_count_correct
    }

    /// Given a trait bound like `Debug`, applies that trait bound the given self-type to construct
    /// a full trait reference. The resulting trait reference is returned. This may also generate
    /// auxiliary bounds, which are added to `bounds`.
    ///
    /// Example:
    ///
    /// ```
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
    pub fn instantiate_poly_trait_ref(
        &self,
        poly_trait_ref: &hir::PolyTraitRef<'_>,
        constness: Constness,
        self_ty: Ty<'tcx>,
        bounds: &mut Bounds<'tcx>,
    ) -> Result<(), GenericArgCountMismatch> {
        self.instantiate_poly_trait_ref_inner(
            &poly_trait_ref.trait_ref,
            poly_trait_ref.span,
            constness,
            self_ty,
            bounds,
            false,
        )
    }

    fn ast_path_to_mono_trait_ref(
        &self,
        span: Span,
        trait_def_id: DefId,
        self_ty: Ty<'tcx>,
        trait_segment: &hir::PathSegment<'_>,
    ) -> ty::TraitRef<'tcx> {
        let (substs, assoc_bindings, _) =
            self.create_substs_for_ast_trait_ref(span, trait_def_id, self_ty, trait_segment);
        assoc_bindings.first().map(|b| AstConv::prohibit_assoc_ty_binding(self.tcx(), b.span));
        ty::TraitRef::new(trait_def_id, substs)
    }

    /// When the code is using the `Fn` traits directly, instead of the `Fn(A) -> B` syntax, emit
    /// an error and attempt to build a reasonable structured suggestion.
    fn complain_about_internal_fn_trait(
        &self,
        span: Span,
        trait_def_id: DefId,
        trait_segment: &'a hir::PathSegment<'a>,
    ) {
        let trait_def = self.tcx().trait_def(trait_def_id);

        if !self.tcx().features().unboxed_closures
            && trait_segment.generic_args().parenthesized != trait_def.paren_sugar
        {
            // For now, require that parenthetical notation be used only with `Fn()` etc.
            let (msg, sugg) = if trait_def.paren_sugar {
                (
                    "the precise format of `Fn`-family traits' type parameters is subject to \
                     change",
                    Some(format!(
                        "{}{} -> {}",
                        trait_segment.ident,
                        trait_segment
                            .args
                            .as_ref()
                            .and_then(|args| args.args.get(0))
                            .and_then(|arg| match arg {
                                hir::GenericArg::Type(ty) => {
                                    Some(print::to_string(print::NO_ANN, |s| s.print_type(ty)))
                                }
                                _ => None,
                            })
                            .unwrap_or_else(|| "()".to_string()),
                        trait_segment
                            .generic_args()
                            .bindings
                            .iter()
                            .filter_map(|b| match (b.ident.as_str() == "Output", &b.kind) {
                                (true, hir::TypeBindingKind::Equality { ty }) => {
                                    Some(print::to_string(print::NO_ANN, |s| s.print_type(ty)))
                                }
                                _ => None,
                            })
                            .next()
                            .unwrap_or_else(|| "()".to_string()),
                    )),
                )
            } else {
                ("parenthetical notation is only stable when used with `Fn`-family traits", None)
            };
            let sess = &self.tcx().sess.parse_sess;
            let mut err = feature_err(sess, sym::unboxed_closures, span, msg);
            if let Some(sugg) = sugg {
                let msg = "use parenthetical notation instead";
                err.span_suggestion(span, msg, sugg, Applicability::MaybeIncorrect);
            }
            err.emit();
        }
    }

    fn create_substs_for_ast_trait_ref<'a>(
        &self,
        span: Span,
        trait_def_id: DefId,
        self_ty: Ty<'tcx>,
        trait_segment: &'a hir::PathSegment<'a>,
    ) -> (SubstsRef<'tcx>, Vec<ConvertedBinding<'a, 'tcx>>, Result<(), GenericArgCountMismatch>)
    {
        debug!("create_substs_for_ast_trait_ref(trait_segment={:?})", trait_segment);

        self.complain_about_internal_fn_trait(span, trait_def_id, trait_segment);

        self.create_substs_for_ast_path(
            span,
            trait_def_id,
            &[],
            trait_segment.generic_args(),
            trait_segment.infer_args,
            Some(self_ty),
        )
    }

    fn trait_defines_associated_type_named(
        &self,
        trait_def_id: DefId,
        assoc_name: ast::Ident,
    ) -> bool {
        self.tcx()
            .associated_items(trait_def_id)
            .find_by_name_and_kind(self.tcx(), assoc_name, ty::AssocKind::Type, trait_def_id)
            .is_some()
    }

    // Returns `true` if a bounds list includes `?Sized`.
    pub fn is_unsized(&self, ast_bounds: &[hir::GenericBound<'_>], span: Span) -> bool {
        let tcx = self.tcx();

        // Try to find an unbound in bounds.
        let mut unbound = None;
        for ab in ast_bounds {
            if let &hir::GenericBound::Trait(ref ptr, hir::TraitBoundModifier::Maybe) = ab {
                if unbound.is_none() {
                    unbound = Some(&ptr.trait_ref);
                } else {
                    struct_span_err!(
                        tcx.sess,
                        span,
                        E0203,
                        "type parameter has more than one relaxed default \
                        bound, only one is supported"
                    )
                    .emit();
                }
            }
        }

        let kind_id = tcx.lang_items().require(SizedTraitLangItem);
        match unbound {
            Some(tpb) => {
                // FIXME(#8559) currently requires the unbound to be built-in.
                if let Ok(kind_id) = kind_id {
                    if tpb.path.res != Res::Def(DefKind::Trait, kind_id) {
                        tcx.sess.span_warn(
                            span,
                            "default bound relaxed for a type parameter, but \
                             this does nothing because the given bound is not \
                             a default; only `?Sized` is supported",
                        );
                    }
                }
            }
            _ if kind_id.is_ok() => {
                return false;
            }
            // No lang item for `Sized`, so we can't add it as a bound.
            None => {}
        }

        true
    }

    /// This helper takes a *converted* parameter type (`param_ty`)
    /// and an *unconverted* list of bounds:
    ///
    /// ```
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
    fn add_bounds(
        &self,
        param_ty: Ty<'tcx>,
        ast_bounds: &[hir::GenericBound<'_>],
        bounds: &mut Bounds<'tcx>,
    ) {
        let mut trait_bounds = Vec::new();
        let mut region_bounds = Vec::new();

        let constness = self.default_constness_for_trait_bounds();
        for ast_bound in ast_bounds {
            match *ast_bound {
                hir::GenericBound::Trait(ref b, hir::TraitBoundModifier::None) => {
                    trait_bounds.push((b, constness))
                }
                hir::GenericBound::Trait(ref b, hir::TraitBoundModifier::MaybeConst) => {
                    trait_bounds.push((b, Constness::NotConst))
                }
                hir::GenericBound::Trait(_, hir::TraitBoundModifier::Maybe) => {}
                hir::GenericBound::Outlives(ref l) => region_bounds.push(l),
            }
        }

        for (bound, constness) in trait_bounds {
            let _ = self.instantiate_poly_trait_ref(bound, constness, param_ty, bounds);
        }

        bounds.region_bounds.extend(
            region_bounds.into_iter().map(|r| (self.ast_region_to_region(r, None), r.span)),
        );
    }

    /// Translates a list of bounds from the HIR into the `Bounds` data structure.
    /// The self-type for the bounds is given by `param_ty`.
    ///
    /// Example:
    ///
    /// ```
    /// fn foo<T: Bar + Baz>() { }
    ///        ^  ^^^^^^^^^ ast_bounds
    ///        param_ty
    /// ```
    ///
    /// The `sized_by_default` parameter indicates if, in this context, the `param_ty` should be
    /// considered `Sized` unless there is an explicit `?Sized` bound.  This would be true in the
    /// example above, but is not true in supertrait listings like `trait Foo: Bar + Baz`.
    ///
    /// `span` should be the declaration size of the parameter.
    pub fn compute_bounds(
        &self,
        param_ty: Ty<'tcx>,
        ast_bounds: &[hir::GenericBound<'_>],
        sized_by_default: SizedByDefault,
        span: Span,
    ) -> Bounds<'tcx> {
        let mut bounds = Bounds::default();

        self.add_bounds(param_ty, ast_bounds, &mut bounds);
        bounds.trait_bounds.sort_by_key(|(t, _, _)| t.def_id());

        bounds.implicitly_sized = if let SizedByDefault::Yes = sized_by_default {
            if !self.is_unsized(ast_bounds, span) { Some(span) } else { None }
        } else {
            None
        };

        bounds
    }

    /// Given an HIR binding like `Item = Foo` or `Item: Foo`, pushes the corresponding predicates
    /// onto `bounds`.
    ///
    /// **A note on binders:** given something like `T: for<'a> Iterator<Item = &'a u32>`, the
    /// `trait_ref` here will be `for<'a> T: Iterator`. The `binding` data however is from *inside*
    /// the binder (e.g., `&'a u32`) and hence may reference bound regions.
    fn add_predicates_for_ast_type_binding(
        &self,
        hir_ref_id: hir::HirId,
        trait_ref: ty::PolyTraitRef<'tcx>,
        binding: &ConvertedBinding<'_, 'tcx>,
        bounds: &mut Bounds<'tcx>,
        speculative: bool,
        dup_bindings: &mut FxHashMap<DefId, Span>,
        path_span: Span,
    ) -> Result<(), ErrorReported> {
        let tcx = self.tcx();

        if !speculative {
            // Given something like `U: SomeTrait<T = X>`, we want to produce a
            // predicate like `<U as SomeTrait>::T = X`. This is somewhat
            // subtle in the event that `T` is defined in a supertrait of
            // `SomeTrait`, because in that case we need to upcast.
            //
            // That is, consider this case:
            //
            // ```
            // trait SubTrait: SuperTrait<int> { }
            // trait SuperTrait<A> { type T; }
            //
            // ... B: SubTrait<T = foo> ...
            // ```
            //
            // We want to produce `<B as SuperTrait<int>>::T == foo`.

            // Find any late-bound regions declared in `ty` that are not
            // declared in the trait-ref. These are not well-formed.
            //
            // Example:
            //
            //     for<'a> <T as Iterator>::Item = &'a str // <-- 'a is bad
            //     for<'a> <T as FnMut<(&'a u32,)>>::Output = &'a str // <-- 'a is ok
            if let ConvertedBindingKind::Equality(ty) = binding.kind {
                let late_bound_in_trait_ref =
                    tcx.collect_constrained_late_bound_regions(&trait_ref);
                let late_bound_in_ty =
                    tcx.collect_referenced_late_bound_regions(&ty::Binder::bind(ty));
                debug!("late_bound_in_trait_ref = {:?}", late_bound_in_trait_ref);
                debug!("late_bound_in_ty = {:?}", late_bound_in_ty);
                for br in late_bound_in_ty.difference(&late_bound_in_trait_ref) {
                    let br_name = match *br {
                        ty::BrNamed(_, name) => name,
                        _ => {
                            span_bug!(
                                binding.span,
                                "anonymous bound region {:?} in binding but not trait ref",
                                br
                            );
                        }
                    };
                    // FIXME: point at the type params that don't have appropriate lifetimes:
                    // struct S1<F: for<'a> Fn(&i32, &i32) -> &'a i32>(F);
                    //                         ----  ----     ^^^^^^^
                    struct_span_err!(
                        tcx.sess,
                        binding.span,
                        E0582,
                        "binding for associated type `{}` references lifetime `{}`, \
                         which does not appear in the trait input types",
                        binding.item_name,
                        br_name
                    )
                    .emit();
                }
            }
        }

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

        // We have already adjusted the item name above, so compare with `ident.modern()` instead
        // of calling `filter_by_name_and_kind`.
        let assoc_ty = tcx
            .associated_items(candidate.def_id())
            .filter_by_name_unhygienic(assoc_ident.name)
            .find(|i| i.kind == ty::AssocKind::Type && i.ident.modern() == assoc_ident)
            .expect("missing associated type");

        if !assoc_ty.vis.is_accessible_from(def_scope, tcx) {
            let msg = format!("associated type `{}` is private", binding.item_name);
            tcx.sess.span_err(binding.span, &msg);
        }
        tcx.check_stability(assoc_ty.def_id, Some(hir_ref_id), binding.span);

        if !speculative {
            dup_bindings
                .entry(assoc_ty.def_id)
                .and_modify(|prev_span| {
                    struct_span_err!(
                        self.tcx().sess,
                        binding.span,
                        E0719,
                        "the value of the associated type `{}` (from trait `{}`) \
                         is already specified",
                        binding.item_name,
                        tcx.def_path_str(assoc_ty.container.id())
                    )
                    .span_label(binding.span, "re-bound here")
                    .span_label(*prev_span, format!("`{}` bound here first", binding.item_name))
                    .emit();
                })
                .or_insert(binding.span);
        }

        match binding.kind {
            ConvertedBindingKind::Equality(ref ty) => {
                // "Desugar" a constraint like `T: Iterator<Item = u32>` this to
                // the "projection predicate" for:
                //
                // `<T as Iterator>::Item = u32`
                bounds.projection_bounds.push((
                    candidate.map_bound(|trait_ref| ty::ProjectionPredicate {
                        projection_ty: ty::ProjectionTy::from_ref_and_name(
                            tcx,
                            trait_ref,
                            binding.item_name,
                        ),
                        ty,
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
                let param_ty = tcx.mk_projection(assoc_ty.def_id, candidate.skip_binder().substs);
                self.add_bounds(param_ty, ast_bounds, bounds);
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
        self.normalize_ty(span, self.tcx().at(span).type_of(did).subst(self.tcx(), substs))
    }

    fn conv_object_ty_poly_trait_ref(
        &self,
        span: Span,
        trait_bounds: &[hir::PolyTraitRef<'_>],
        lifetime: &hir::Lifetime,
    ) -> Ty<'tcx> {
        let tcx = self.tcx();

        let mut bounds = Bounds::default();
        let mut potential_assoc_types = Vec::new();
        let dummy_self = self.tcx().types.trait_object_dummy_self;
        for trait_bound in trait_bounds.iter().rev() {
            if let Err(GenericArgCountMismatch {
                invalid_args: cur_potential_assoc_types, ..
            }) = self.instantiate_poly_trait_ref(
                trait_bound,
                Constness::NotConst,
                dummy_self,
                &mut bounds,
            ) {
                potential_assoc_types.extend(cur_potential_assoc_types.into_iter());
            }
        }

        // Expand trait aliases recursively and check that only one regular (non-auto) trait
        // is used and no 'maybe' bounds are used.
        let expanded_traits =
            traits::expand_trait_aliases(tcx, bounds.trait_bounds.iter().map(|&(a, b, _)| (a, b)));
        let (mut auto_traits, regular_traits): (Vec<_>, Vec<_>) =
            expanded_traits.partition(|i| tcx.trait_is_auto(i.trait_ref().def_id()));
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
            err.emit();
        }

        if regular_traits.is_empty() && auto_traits.is_empty() {
            struct_span_err!(
                tcx.sess,
                span,
                E0224,
                "at least one trait is required for an object type"
            )
            .emit();
            return tcx.types.err;
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
                    object_safety_violations,
                )
                .emit();
                return tcx.types.err;
            }
        }

        // Use a `BTreeSet` to keep output in a more consistent order.
        let mut associated_types: FxHashMap<Span, BTreeSet<DefId>> = FxHashMap::default();

        let regular_traits_refs_spans = bounds
            .trait_bounds
            .into_iter()
            .filter(|(trait_ref, _, _)| !tcx.trait_is_auto(trait_ref.def_id()));

        for (base_trait_ref, span, constness) in regular_traits_refs_spans {
            assert_eq!(constness, Constness::NotConst);

            for trait_ref in traits::elaborate_trait_ref(tcx, base_trait_ref) {
                debug!(
                    "conv_object_ty_poly_trait_ref: observing object predicate `{:?}`",
                    trait_ref
                );
                match trait_ref {
                    ty::Predicate::Trait(pred, _) => {
                        associated_types.entry(span).or_default().extend(
                            tcx.associated_items(pred.def_id())
                                .in_definition_order()
                                .filter(|item| item.kind == ty::AssocKind::Type)
                                .map(|item| item.def_id),
                        );
                    }
                    ty::Predicate::Projection(pred) => {
                        // A `Self` within the original bound will be substituted with a
                        // `trait_object_dummy_self`, so check for that.
                        let references_self = pred.skip_binder().ty.walk().any(|t| t == dummy_self);

                        // If the projection output contains `Self`, force the user to
                        // elaborate it explicitly to avoid a lot of complexity.
                        //
                        // The "classicaly useful" case is the following:
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
        auto_traits.sort_by_key(|i| i.trait_ref().def_id());
        auto_traits.dedup_by_key(|i| i.trait_ref().def_id());
        debug!("regular_traits: {:?}", regular_traits);
        debug!("auto_traits: {:?}", auto_traits);

        // Transform a `PolyTraitRef` into a `PolyExistentialTraitRef` by
        // removing the dummy `Self` type (`trait_object_dummy_self`).
        let trait_ref_to_existential = |trait_ref: ty::TraitRef<'tcx>| {
            if trait_ref.self_ty() != dummy_self {
                // FIXME: There appears to be a missing filter on top of `expand_trait_aliases`,
                // which picks up non-supertraits where clauses - but also, the object safety
                // completely ignores trait aliases, which could be object safety hazards. We
                // `delay_span_bug` here to avoid an ICE in stable even when the feature is
                // disabled. (#66420)
                tcx.sess.delay_span_bug(
                    DUMMY_SP,
                    &format!(
                        "trait_ref_to_existential called on {:?} with non-dummy Self",
                        trait_ref,
                    ),
                );
            }
            ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref)
        };

        // Erase the `dummy_self` (`trait_object_dummy_self`) used above.
        let existential_trait_refs = regular_traits
            .iter()
            .map(|i| i.trait_ref().map_bound(|trait_ref| trait_ref_to_existential(trait_ref)));
        let existential_projections = bounds.projection_bounds.iter().map(|(bound, _)| {
            bound.map_bound(|b| {
                let trait_ref = trait_ref_to_existential(b.projection_ty.trait_ref(tcx));
                ty::ExistentialProjection {
                    ty: b.ty,
                    item_def_id: b.projection_ty.item_def_id,
                    substs: trait_ref.substs,
                }
            })
        });

        // Calling `skip_binder` is okay because the predicates are re-bound.
        let regular_trait_predicates = existential_trait_refs
            .map(|trait_ref| ty::ExistentialPredicate::Trait(*trait_ref.skip_binder()));
        let auto_trait_predicates = auto_traits
            .into_iter()
            .map(|trait_ref| ty::ExistentialPredicate::AutoTrait(trait_ref.trait_ref().def_id()));
        let mut v = regular_trait_predicates
            .chain(auto_trait_predicates)
            .chain(
                existential_projections
                    .map(|x| ty::ExistentialPredicate::Projection(*x.skip_binder())),
            )
            .collect::<SmallVec<[_; 8]>>();
        v.sort_by(|a, b| a.stable_cmp(tcx, b));
        v.dedup();
        let existential_predicates = ty::Binder::bind(tcx.mk_existential_predicates(v.into_iter()));

        // Use explicitly-specified region bound.
        let region_bound = if !lifetime.is_elided() {
            self.ast_region_to_region(lifetime, None)
        } else {
            self.compute_object_lifetime_bound(span, existential_predicates).unwrap_or_else(|| {
                if tcx.named_region(lifetime.hir_id).is_some() {
                    self.ast_region_to_region(lifetime, None)
                } else {
                    self.re_infer(None, span).unwrap_or_else(|| {
                        struct_span_err!(
                            tcx.sess,
                            span,
                            E0228,
                            "the lifetime bound for this object type cannot be deduced \
                             from context; please supply an explicit bound"
                        )
                        .emit();
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

    /// When there are any missing associated types, emit an E0191 error and attempt to supply a
    /// reasonable suggestion on how to write it. For the case of multiple associated types in the
    /// same trait bound have the same name (as they come from different super-traits), we instead
    /// emit a generic note suggesting using a `where` clause to constraint instead.
    fn complain_about_missing_associated_types(
        &self,
        associated_types: FxHashMap<Span, BTreeSet<DefId>>,
        potential_assoc_types: Vec<Span>,
        trait_bounds: &[hir::PolyTraitRef<'_>],
    ) {
        if !associated_types.values().any(|v| !v.is_empty()) {
            return;
        }
        let tcx = self.tcx();
        // FIXME: Marked `mut` so that we can replace the spans further below with a more
        // appropriate one, but this should be handled earlier in the span assignment.
        let mut associated_types: FxHashMap<Span, Vec<_>> = associated_types
            .into_iter()
            .map(|(span, def_ids)| {
                (span, def_ids.into_iter().map(|did| tcx.associated_item(did)).collect())
            })
            .collect();
        let mut names = vec![];

        // Account for things like `dyn Foo + 'a`, like in tests `issue-22434.rs` and
        // `issue-22560.rs`.
        let mut trait_bound_spans: Vec<Span> = vec![];
        for (span, items) in &associated_types {
            if !items.is_empty() {
                trait_bound_spans.push(*span);
            }
            for assoc_item in items {
                let trait_def_id = assoc_item.container.id();
                names.push(format!(
                    "`{}` (from trait `{}`)",
                    assoc_item.ident,
                    tcx.def_path_str(trait_def_id),
                ));
            }
        }

        match (&potential_assoc_types[..], &trait_bounds) {
            ([], [bound]) => match &bound.trait_ref.path.segments[..] {
                // FIXME: `trait_ref.path.span` can point to a full path with multiple
                // segments, even though `trait_ref.path.segments` is of length `1`. Work
                // around that bug here, even though it should be fixed elsewhere.
                // This would otherwise cause an invalid suggestion. For an example, look at
                // `src/test/ui/issues/issue-28344.rs` where instead of the following:
                //
                //   error[E0191]: the value of the associated type `Output`
                //                 (from trait `std::ops::BitXor`) must be specified
                //   --> $DIR/issue-28344.rs:4:17
                //    |
                // LL |     let x: u8 = BitXor::bitor(0 as u8, 0 as u8);
                //    |                 ^^^^^^ help: specify the associated type:
                //    |                              `BitXor<Output = Type>`
                //
                // we would output:
                //
                //   error[E0191]: the value of the associated type `Output`
                //                 (from trait `std::ops::BitXor`) must be specified
                //   --> $DIR/issue-28344.rs:4:17
                //    |
                // LL |     let x: u8 = BitXor::bitor(0 as u8, 0 as u8);
                //    |                 ^^^^^^^^^^^^^ help: specify the associated type:
                //    |                                     `BitXor::bitor<Output = Type>`
                [segment] if segment.args.is_none() => {
                    trait_bound_spans = vec![segment.ident.span];
                    associated_types = associated_types
                        .into_iter()
                        .map(|(_, items)| (segment.ident.span, items))
                        .collect();
                }
                _ => {}
            },
            _ => {}
        }
        names.sort();
        trait_bound_spans.sort();
        let mut err = struct_span_err!(
            tcx.sess,
            trait_bound_spans,
            E0191,
            "the value of the associated type{} {} must be specified",
            pluralize!(names.len()),
            names.join(", "),
        );
        let mut suggestions = vec![];
        let mut types_count = 0;
        let mut where_constraints = vec![];
        for (span, assoc_items) in &associated_types {
            let mut names: FxHashMap<_, usize> = FxHashMap::default();
            for item in assoc_items {
                types_count += 1;
                *names.entry(item.ident.name).or_insert(0) += 1;
            }
            let mut dupes = false;
            for item in assoc_items {
                let prefix = if names[&item.ident.name] > 1 {
                    let trait_def_id = item.container.id();
                    dupes = true;
                    format!("{}::", tcx.def_path_str(trait_def_id))
                } else {
                    String::new()
                };
                if let Some(sp) = tcx.hir().span_if_local(item.def_id) {
                    err.span_label(sp, format!("`{}{}` defined here", prefix, item.ident));
                }
            }
            if potential_assoc_types.len() == assoc_items.len() {
                // Only suggest when the amount of missing associated types equals the number of
                // extra type arguments present, as that gives us a relatively high confidence
                // that the user forgot to give the associtated type's name. The canonical
                // example would be trying to use `Iterator<isize>` instead of
                // `Iterator<Item = isize>`.
                for (potential, item) in potential_assoc_types.iter().zip(assoc_items.iter()) {
                    if let Ok(snippet) = tcx.sess.source_map().span_to_snippet(*potential) {
                        suggestions.push((*potential, format!("{} = {}", item.ident, snippet)));
                    }
                }
            } else if let (Ok(snippet), false) =
                (tcx.sess.source_map().span_to_snippet(*span), dupes)
            {
                let types: Vec<_> =
                    assoc_items.iter().map(|item| format!("{} = Type", item.ident)).collect();
                let code = if snippet.ends_with('>') {
                    // The user wrote `Trait<'a>` or similar and we don't have a type we can
                    // suggest, but at least we can clue them to the correct syntax
                    // `Trait<'a, Item = Type>` while accounting for the `<'a>` in the
                    // suggestion.
                    format!("{}, {}>", &snippet[..snippet.len() - 1], types.join(", "))
                } else {
                    // The user wrote `Iterator`, so we don't have a type we can suggest, but at
                    // least we can clue them to the correct syntax `Iterator<Item = Type>`.
                    format!("{}<{}>", snippet, types.join(", "))
                };
                suggestions.push((*span, code));
            } else if dupes {
                where_constraints.push(*span);
            }
        }
        let where_msg = "consider introducing a new type parameter, adding `where` constraints \
                         using the fully-qualified path to the associated types";
        if !where_constraints.is_empty() && suggestions.is_empty() {
            // If there are duplicates associated type names and a single trait bound do not
            // use structured suggestion, it means that there are multiple super-traits with
            // the same associated type name.
            err.help(where_msg);
        }
        if suggestions.len() != 1 {
            // We don't need this label if there's an inline suggestion, show otherwise.
            for (span, assoc_items) in &associated_types {
                let mut names: FxHashMap<_, usize> = FxHashMap::default();
                for item in assoc_items {
                    types_count += 1;
                    *names.entry(item.ident.name).or_insert(0) += 1;
                }
                let mut label = vec![];
                for item in assoc_items {
                    let postfix = if names[&item.ident.name] > 1 {
                        let trait_def_id = item.container.id();
                        format!(" (from trait `{}`)", tcx.def_path_str(trait_def_id))
                    } else {
                        String::new()
                    };
                    label.push(format!("`{}`{}", item.ident, postfix));
                }
                if !label.is_empty() {
                    err.span_label(
                        *span,
                        format!(
                            "associated type{} {} must be specified",
                            pluralize!(label.len()),
                            label.join(", "),
                        ),
                    );
                }
            }
        }
        if !suggestions.is_empty() {
            err.multipart_suggestion(
                &format!("specify the associated type{}", pluralize!(types_count)),
                suggestions,
                Applicability::HasPlaceholders,
            );
            if !where_constraints.is_empty() {
                err.span_help(where_constraints, where_msg);
            }
        }
        err.emit();
    }

    fn report_ambiguous_associated_type(
        &self,
        span: Span,
        type_str: &str,
        trait_str: &str,
        name: ast::Name,
    ) {
        let mut err = struct_span_err!(self.tcx().sess, span, E0223, "ambiguous associated type");
        if let (Some(_), Ok(snippet)) = (
            self.tcx().sess.confused_type_with_std_module.borrow().get(&span),
            self.tcx().sess.source_map().span_to_snippet(span),
        ) {
            err.span_suggestion(
                span,
                "you are looking for the module in `std`, not the primitive type",
                format!("std::{}", snippet),
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
        err.emit();
    }

    // Search for a bound on a type parameter which includes the associated item
    // given by `assoc_name`. `ty_param_def_id` is the `DefId` of the type parameter
    // This function will fail if there are no suitable bounds or there is
    // any ambiguity.
    fn find_bound_for_assoc_item(
        &self,
        ty_param_def_id: DefId,
        assoc_name: ast::Ident,
        span: Span,
    ) -> Result<ty::PolyTraitRef<'tcx>, ErrorReported> {
        let tcx = self.tcx();

        debug!(
            "find_bound_for_assoc_item(ty_param_def_id={:?}, assoc_name={:?}, span={:?})",
            ty_param_def_id, assoc_name, span,
        );

        let predicates = &self.get_type_parameter_bounds(span, ty_param_def_id).predicates;

        debug!("find_bound_for_assoc_item: predicates={:#?}", predicates);

        let param_hir_id = tcx.hir().as_local_hir_id(ty_param_def_id).unwrap();
        let param_name = tcx.hir().ty_param_name(param_hir_id);
        self.one_bound_for_assoc_type(
            || {
                traits::transitive_bounds(
                    tcx,
                    predicates.iter().filter_map(|(p, _)| p.to_opt_poly_trait_ref()),
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
        assoc_name: ast::Ident,
        span: Span,
        is_equality: impl Fn() -> Option<String>,
    ) -> Result<ty::PolyTraitRef<'tcx>, ErrorReported>
    where
        I: Iterator<Item = ty::PolyTraitRef<'tcx>>,
    {
        let mut matching_candidates = all_candidates()
            .filter(|r| self.trait_defines_associated_type_named(r.def_id(), assoc_name));

        let bound = match matching_candidates.next() {
            Some(bound) => bound,
            None => {
                self.complain_about_assoc_type_not_found(
                    all_candidates,
                    &ty_param_name(),
                    assoc_name,
                    span,
                );
                return Err(ErrorReported);
            }
        };

        debug!("one_bound_for_assoc_type: bound = {:?}", bound);

        if let Some(bound2) = matching_candidates.next() {
            debug!("one_bound_for_assoc_type: bound2 = {:?}", bound2);

            let is_equality = is_equality();
            let bounds = iter::once(bound).chain(iter::once(bound2)).chain(matching_candidates);
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
                        err.span_suggestion(
                            span,
                            "use fully qualified syntax to disambiguate",
                            format!(
                                "<{} as {}>::{}",
                                ty_param_name(),
                                bound.print_only_trait_path(),
                                assoc_name,
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
            err.emit();
            if !where_bounds.is_empty() {
                return Err(ErrorReported);
            }
        }
        return Ok(bound);
    }

    fn complain_about_assoc_type_not_found<I>(
        &self,
        all_candidates: impl Fn() -> I,
        ty_param_name: &str,
        assoc_name: ast::Ident,
        span: Span,
    ) where
        I: Iterator<Item = ty::PolyTraitRef<'tcx>>,
    {
        // The fallback span is needed because `assoc_name` might be an `Fn()`'s `Output` without a
        // valid span, so we point at the whole path segment instead.
        let span = if assoc_name.span != DUMMY_SP { assoc_name.span } else { span };
        let mut err = struct_span_err!(
            self.tcx().sess,
            span,
            E0220,
            "associated type `{}` not found for `{}`",
            assoc_name,
            ty_param_name
        );

        let all_candidate_names: Vec<_> = all_candidates()
            .map(|r| self.tcx().associated_items(r.def_id()).in_definition_order())
            .flatten()
            .filter_map(
                |item| if item.kind == ty::AssocKind::Type { Some(item.ident.name) } else { None },
            )
            .collect();

        if let (Some(suggested_name), true) = (
            find_best_match_for_name(all_candidate_names.iter(), &assoc_name.as_str(), None),
            assoc_name.span != DUMMY_SP,
        ) {
            err.span_suggestion(
                assoc_name.span,
                "there is an associated type with a similar name",
                suggested_name.to_string(),
                Applicability::MaybeIncorrect,
            );
        } else {
            err.span_label(span, format!("associated type `{}` not found", assoc_name));
        }

        err.emit();
    }

    // Create a type from a path to an associated type.
    // For a path `A::B::C::D`, `qself_ty` and `qself_def` are the type and def for `A::B::C`
    // and item_segment is the path segment for `D`. We return a type and a def for
    // the whole path.
    // Will fail except for `T::A` and `Self::A`; i.e., if `qself_ty`/`qself_def` are not a type
    // parameter or `Self`.
    pub fn associated_path_to_ty(
        &self,
        hir_ref_id: hir::HirId,
        span: Span,
        qself_ty: Ty<'tcx>,
        qself_res: Res,
        assoc_segment: &hir::PathSegment<'_>,
        permit_variants: bool,
    ) -> Result<(Ty<'tcx>, DefKind, DefId), ErrorReported> {
        let tcx = self.tcx();
        let assoc_ident = assoc_segment.ident;

        debug!("associated_path_to_ty: {:?}::{}", qself_ty, assoc_ident);

        // Check if we have an enum variant.
        let mut variant_resolution = None;
        if let ty::Adt(adt_def, _) = qself_ty.kind {
            if adt_def.is_enum() {
                let variant_def = adt_def
                    .variants
                    .iter()
                    .find(|vd| tcx.hygienic_eq(assoc_ident, vd.ident, adt_def.did));
                if let Some(variant_def) = variant_def {
                    if permit_variants {
                        tcx.check_stability(variant_def.def_id, Some(hir_ref_id), span);
                        self.prohibit_generics(slice::from_ref(assoc_segment));
                        return Ok((qself_ty, DefKind::Variant, variant_def.def_id));
                    } else {
                        variant_resolution = Some(variant_def.def_id);
                    }
                }
            }
        }

        // Find the type of the associated item, and the trait where the associated
        // item is declared.
        let bound = match (&qself_ty.kind, qself_res) {
            (_, Res::SelfTy(Some(_), Some(impl_def_id))) => {
                // `Self` in an impl of a trait -- we have a concrete self type and a
                // trait reference.
                let trait_ref = match tcx.impl_trait_ref(impl_def_id) {
                    Some(trait_ref) => trait_ref,
                    None => {
                        // A cycle error occurred, most likely.
                        return Err(ErrorReported);
                    }
                };

                self.one_bound_for_assoc_type(
                    || traits::supertraits(tcx, ty::Binder::bind(trait_ref)),
                    || "Self".to_string(),
                    assoc_ident,
                    span,
                    || None,
                )?
            }
            (&ty::Param(_), Res::SelfTy(Some(param_did), None))
            | (&ty::Param(_), Res::Def(DefKind::TyParam, param_did)) => {
                self.find_bound_for_assoc_item(param_did, assoc_ident, span)?
            }
            _ => {
                if variant_resolution.is_some() {
                    // Variant in type position
                    let msg = format!("expected type, found variant `{}`", assoc_ident);
                    tcx.sess.span_err(span, &msg);
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
                        adt_def.variants.iter().map(|variant| &variant.ident.name),
                        &assoc_ident.as_str(),
                        None,
                    ) {
                        err.span_suggestion(
                            assoc_ident.span,
                            "there is a variant with a similar name",
                            suggested_name.to_string(),
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        err.span_label(
                            assoc_ident.span,
                            format!("variant not found in `{}`", qself_ty),
                        );
                    }

                    if let Some(sp) = tcx.hir().span_if_local(adt_def.did) {
                        let sp = tcx.sess.source_map().def_span(sp);
                        err.span_label(sp, format!("variant `{}` not found here", assoc_ident));
                    }

                    err.emit();
                } else if !qself_ty.references_error() {
                    // Don't print `TyErr` to the user.
                    self.report_ambiguous_associated_type(
                        span,
                        &qself_ty.to_string(),
                        "Trait",
                        assoc_ident.name,
                    );
                }
                return Err(ErrorReported);
            }
        };

        let trait_did = bound.def_id();
        let (assoc_ident, def_scope) =
            tcx.adjust_ident_and_get_scope(assoc_ident, trait_did, hir_ref_id);

        // We have already adjusted the item name above, so compare with `ident.modern()` instead
        // of calling `filter_by_name_and_kind`.
        let item = tcx
            .associated_items(trait_did)
            .in_definition_order()
            .find(|i| i.kind.namespace() == Namespace::TypeNS && i.ident.modern() == assoc_ident)
            .expect("missing associated type");

        let ty = self.projected_ty_from_poly_trait_ref(span, item.def_id, assoc_segment, bound);
        let ty = self.normalize_ty(span, ty);

        let kind = DefKind::AssocTy;
        if !item.vis.is_accessible_from(def_scope, tcx) {
            let msg = format!("{} `{}` is private", kind.descr(item.def_id), assoc_ident);
            tcx.sess.span_err(span, &msg);
        }
        tcx.check_stability(item.def_id, Some(hir_ref_id), span);

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

        let trait_def_id = tcx.parent(item_def_id).unwrap();

        debug!("qpath_to_ty: trait_def_id={:?}", trait_def_id);

        let self_ty = if let Some(ty) = opt_self_ty {
            ty
        } else {
            let path_str = tcx.def_path_str(trait_def_id);

            let def_id = self.item_def_id();

            debug!("qpath_to_ty: self.item_def_id()={:?}", def_id);

            let parent_def_id = def_id
                .and_then(|def_id| tcx.hir().as_local_hir_id(def_id))
                .map(|hir_id| tcx.hir().get_parent_did(hir_id));

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
            return tcx.types.err;
        };

        debug!("qpath_to_ty: self_type={:?}", self_ty);

        let trait_ref = self.ast_path_to_mono_trait_ref(span, trait_def_id, self_ty, trait_segment);

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

    pub fn prohibit_generics<'a, T: IntoIterator<Item = &'a hir::PathSegment<'a>>>(
        &self,
        segments: T,
    ) -> bool {
        let mut has_err = false;
        for segment in segments {
            let (mut err_for_lt, mut err_for_ty, mut err_for_ct) = (false, false, false);
            for arg in segment.generic_args().args {
                let (span, kind) = match arg {
                    hir::GenericArg::Lifetime(lt) => {
                        if err_for_lt {
                            continue;
                        }
                        err_for_lt = true;
                        has_err = true;
                        (lt.span, "lifetime")
                    }
                    hir::GenericArg::Type(ty) => {
                        if err_for_ty {
                            continue;
                        }
                        err_for_ty = true;
                        has_err = true;
                        (ty.span, "type")
                    }
                    hir::GenericArg::Const(ct) => {
                        if err_for_ct {
                            continue;
                        }
                        err_for_ct = true;
                        (ct.span, "const")
                    }
                };
                let mut err = struct_span_err!(
                    self.tcx().sess,
                    span,
                    E0109,
                    "{} arguments are not allowed for this type",
                    kind,
                );
                err.span_label(span, format!("{} argument not allowed", kind));
                err.emit();
                if err_for_lt && err_for_ty && err_for_ct {
                    break;
                }
            }

            // Only emit the first error to avoid overloading the user with error messages.
            if let [binding, ..] = segment.generic_args().bindings {
                has_err = true;
                Self::prohibit_assoc_ty_binding(self.tcx(), binding.span);
            }
        }
        has_err
    }

    pub fn prohibit_assoc_ty_binding(tcx: TyCtxt<'_>, span: Span) {
        let mut err = struct_span_err!(
            tcx.sess,
            span,
            E0229,
            "associated type bindings are not allowed here"
        );
        err.span_label(span, "associated type not allowed here").emit();
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
                    (adt_def.did, last)
                } else if last >= 1 && segments[last - 1].args.is_some() {
                    // Everything but the penultimate segment should have no
                    // parameters at all.
                    let mut def_id = def_id;

                    // `DefKind::Ctor` -> `DefKind::Variant`
                    if let DefKind::Ctor(..) = kind {
                        def_id = tcx.parent(def_id).unwrap()
                    }

                    // `DefKind::Variant` -> `DefKind::Enum`
                    let enum_def_id = tcx.parent(def_id).unwrap();
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
            DefKind::Fn | DefKind::Const | DefKind::ConstParam | DefKind::Static => {
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
                self.prohibit_generics(item_segment.1);
                let substs = self.ast_path_substs_for_ty(span, did, item_segment.0);
                self.normalize_ty(span, tcx.mk_opaque(did, substs))
            }
            Res::Def(DefKind::Enum, did)
            | Res::Def(DefKind::TyAlias, did)
            | Res::Def(DefKind::Struct, did)
            | Res::Def(DefKind::Union, did)
            | Res::Def(DefKind::ForeignTy, did) => {
                assert_eq!(opt_self_ty, None);
                self.prohibit_generics(path.segments.split_last().unwrap().1);
                self.ast_path_to_ty(span, did, path.segments.last().unwrap())
            }
            Res::Def(kind @ DefKind::Variant, def_id) if permit_variants => {
                // Convert "variant type" as if it were a real type.
                // The resulting `Ty` is type of the variant's enum for now.
                assert_eq!(opt_self_ty, None);

                let path_segs =
                    self.def_ids_for_value_path_segments(&path.segments, None, kind, def_id);
                let generic_segs: FxHashSet<_> =
                    path_segs.iter().map(|PathSeg(_, index)| index).collect();
                self.prohibit_generics(path.segments.iter().enumerate().filter_map(
                    |(index, seg)| {
                        if !generic_segs.contains(&index) { Some(seg) } else { None }
                    },
                ));

                let PathSeg(def_id, index) = path_segs.last().unwrap();
                self.ast_path_to_ty(span, *def_id, &path.segments[*index])
            }
            Res::Def(DefKind::TyParam, def_id) => {
                assert_eq!(opt_self_ty, None);
                self.prohibit_generics(path.segments);

                let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
                let item_id = tcx.hir().get_parent_node(hir_id);
                let item_def_id = tcx.hir().local_def_id(item_id);
                let generics = tcx.generics_of(item_def_id);
                let index = generics.param_def_id_to_index[&def_id];
                tcx.mk_ty_param(index, tcx.hir().name(hir_id))
            }
            Res::SelfTy(Some(_), None) => {
                // `Self` in trait or type alias.
                assert_eq!(opt_self_ty, None);
                self.prohibit_generics(path.segments);
                tcx.types.self_param
            }
            Res::SelfTy(_, Some(def_id)) => {
                // `Self` in impl (we know the concrete type).
                assert_eq!(opt_self_ty, None);
                self.prohibit_generics(path.segments);
                // Try to evaluate any array length constants.
                self.normalize_ty(span, tcx.at(span).type_of(def_id))
            }
            Res::Def(DefKind::AssocTy, def_id) => {
                debug_assert!(path.segments.len() >= 2);
                self.prohibit_generics(&path.segments[..path.segments.len() - 2]);
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
                self.prohibit_generics(path.segments);
                match prim_ty {
                    hir::PrimTy::Bool => tcx.types.bool,
                    hir::PrimTy::Char => tcx.types.char,
                    hir::PrimTy::Int(it) => tcx.mk_mach_int(it),
                    hir::PrimTy::Uint(uit) => tcx.mk_mach_uint(uit),
                    hir::PrimTy::Float(ft) => tcx.mk_mach_float(ft),
                    hir::PrimTy::Str => tcx.mk_str(),
                }
            }
            Res::Err => {
                self.set_tainted_by_errors();
                return self.tcx().types.err;
            }
            _ => span_bug!(span, "unexpected resolution: {:?}", path.res),
        }
    }

    /// Parses the programmer's textual representation of a type into our
    /// internal notion of a type.
    pub fn ast_ty_to_ty(&self, ast_ty: &hir::Ty<'_>) -> Ty<'tcx> {
        debug!("ast_ty_to_ty(id={:?}, ast_ty={:?} ty_ty={:?})", ast_ty.hir_id, ast_ty, ast_ty.kind);

        let tcx = self.tcx();

        let result_ty = match ast_ty.kind {
            hir::TyKind::Slice(ref ty) => tcx.mk_slice(self.ast_ty_to_ty(&ty)),
            hir::TyKind::Ptr(ref mt) => {
                tcx.mk_ptr(ty::TypeAndMut { ty: self.ast_ty_to_ty(&mt.ty), mutbl: mt.mutbl })
            }
            hir::TyKind::Rptr(ref region, ref mt) => {
                let r = self.ast_region_to_region(region, None);
                debug!("ast_ty_to_ty: r={:?}", r);
                let t = self.ast_ty_to_ty(&mt.ty);
                tcx.mk_ref(r, ty::TypeAndMut { ty: t, mutbl: mt.mutbl })
            }
            hir::TyKind::Never => tcx.types.never,
            hir::TyKind::Tup(ref fields) => {
                tcx.mk_tup(fields.iter().map(|t| self.ast_ty_to_ty(&t)))
            }
            hir::TyKind::BareFn(ref bf) => {
                require_c_abi_if_c_variadic(tcx, &bf.decl, bf.abi, ast_ty.span);
                tcx.mk_fn_ptr(self.ty_of_fn(bf.unsafety, bf.abi, &bf.decl, &[], None))
            }
            hir::TyKind::TraitObject(ref bounds, ref lifetime) => {
                self.conv_object_ty_poly_trait_ref(ast_ty.span, bounds, lifetime)
            }
            hir::TyKind::Path(hir::QPath::Resolved(ref maybe_qself, ref path)) => {
                debug!("ast_ty_to_ty: maybe_qself={:?} path={:?}", maybe_qself, path);
                let opt_self_ty = maybe_qself.as_ref().map(|qself| self.ast_ty_to_ty(qself));
                self.res_to_ty(opt_self_ty, path, false)
            }
            hir::TyKind::Def(item_id, ref lifetimes) => {
                let did = tcx.hir().local_def_id(item_id.id);
                self.impl_trait_ty_to_ty(did, lifetimes)
            }
            hir::TyKind::Path(hir::QPath::TypeRelative(ref qself, ref segment)) => {
                debug!("ast_ty_to_ty: qself={:?} segment={:?}", qself, segment);
                let ty = self.ast_ty_to_ty(qself);

                let res = if let hir::TyKind::Path(hir::QPath::Resolved(_, ref path)) = qself.kind {
                    path.res
                } else {
                    Res::Err
                };
                self.associated_path_to_ty(ast_ty.hir_id, ast_ty.span, ty, res, segment, false)
                    .map(|(ty, _, _)| ty)
                    .unwrap_or(tcx.types.err)
            }
            hir::TyKind::Array(ref ty, ref length) => {
                let length = self.ast_const_to_const(length, tcx.types.usize);
                let array_ty = tcx.mk_ty(ty::Array(self.ast_ty_to_ty(&ty), length));
                self.normalize_ty(ast_ty.span, array_ty)
            }
            hir::TyKind::Typeof(ref _e) => {
                struct_span_err!(
                    tcx.sess,
                    ast_ty.span,
                    E0516,
                    "`typeof` is a reserved keyword but unimplemented"
                )
                .span_label(ast_ty.span, "reserved keyword")
                .emit();

                tcx.types.err
            }
            hir::TyKind::Infer => {
                // Infer also appears as the type of arguments or return
                // values in a ExprKind::Closure, or as
                // the type of local variables. Both of these cases are
                // handled specially and will not descend into this routine.
                self.ty_infer(None, ast_ty.span)
            }
            hir::TyKind::Err => tcx.types.err,
        };

        debug!("ast_ty_to_ty: result_ty={:?}", result_ty);

        self.record_ty(ast_ty.hir_id, result_ty, ast_ty.span);
        result_ty
    }

    /// Returns the `DefId` of the constant parameter that the provided expression is a path to.
    pub fn const_param_def_id(&self, expr: &hir::Expr<'_>) -> Option<DefId> {
        // Unwrap a block, so that e.g. `{ P }` is recognised as a parameter. Const arguments
        // currently have to be wrapped in curly brackets, so it's necessary to special-case.
        let expr = match &expr.kind {
            ExprKind::Block(block, _) if block.stmts.is_empty() && block.expr.is_some() => {
                block.expr.as_ref().unwrap()
            }
            _ => expr,
        };

        match &expr.kind {
            ExprKind::Path(hir::QPath::Resolved(_, path)) => match path.res {
                Res::Def(DefKind::ConstParam, did) => Some(did),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn ast_const_to_const(
        &self,
        ast_const: &hir::AnonConst,
        ty: Ty<'tcx>,
    ) -> &'tcx ty::Const<'tcx> {
        debug!("ast_const_to_const(id={:?}, ast_const={:?})", ast_const.hir_id, ast_const);

        let tcx = self.tcx();
        let def_id = tcx.hir().local_def_id(ast_const.hir_id);

        let expr = &tcx.hir().body(ast_const.body).value;

        let lit_input = match expr.kind {
            hir::ExprKind::Lit(ref lit) => Some(LitToConstInput { lit: &lit.node, ty, neg: false }),
            hir::ExprKind::Unary(hir::UnOp::UnNeg, ref expr) => match expr.kind {
                hir::ExprKind::Lit(ref lit) => {
                    Some(LitToConstInput { lit: &lit.node, ty, neg: true })
                }
                _ => None,
            },
            _ => None,
        };

        if let Some(lit_input) = lit_input {
            // If an error occurred, ignore that it's a literal and leave reporting the error up to
            // mir.
            if let Ok(c) = tcx.at(expr.span).lit_to_const(lit_input) {
                return c;
            } else {
                tcx.sess.delay_span_bug(expr.span, "ast_const_to_const: couldn't lit_to_const");
            }
        }

        let kind = if let Some(def_id) = self.const_param_def_id(expr) {
            // Find the name and index of the const parameter by indexing the generics of the
            // parent item and construct a `ParamConst`.
            let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
            let item_id = tcx.hir().get_parent_node(hir_id);
            let item_def_id = tcx.hir().local_def_id(item_id);
            let generics = tcx.generics_of(item_def_id);
            let index = generics.param_def_id_to_index[&tcx.hir().local_def_id(hir_id)];
            let name = tcx.hir().name(hir_id);
            ty::ConstKind::Param(ty::ParamConst::new(index, name))
        } else {
            ty::ConstKind::Unevaluated(def_id, InternalSubsts::identity_for_item(tcx, def_id), None)
        };
        tcx.mk_const(ty::Const { val: kind, ty })
    }

    pub fn impl_trait_ty_to_ty(
        &self,
        def_id: DefId,
        lifetimes: &[hir::GenericArg<'_>],
    ) -> Ty<'tcx> {
        debug!("impl_trait_ty_to_ty(def_id={:?}, lifetimes={:?})", def_id, lifetimes);
        let tcx = self.tcx();

        let generics = tcx.generics_of(def_id);

        debug!("impl_trait_ty_to_ty: generics={:?}", generics);
        let substs = InternalSubsts::for_item(tcx, def_id, |param, _| {
            if let Some(i) = (param.index as usize).checked_sub(generics.parent_count) {
                // Our own parameters are the resolved lifetimes.
                match param.kind {
                    GenericParamDefKind::Lifetime => {
                        if let hir::GenericArg::Lifetime(lifetime) = &lifetimes[i] {
                            self.ast_region_to_region(lifetime, None).into()
                        } else {
                            bug!()
                        }
                    }
                    _ => bug!(),
                }
            } else {
                // Replace all parent lifetimes with `'static`.
                match param.kind {
                    GenericParamDefKind::Lifetime => tcx.lifetimes.re_static.into(),
                    _ => tcx.mk_param_from_def(param),
                }
            }
        });
        debug!("impl_trait_ty_to_ty: substs={:?}", substs);

        let ty = tcx.mk_opaque(def_id, substs);
        debug!("impl_trait_ty_to_ty: {}", ty);
        ty
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
        unsafety: hir::Unsafety,
        abi: abi::Abi,
        decl: &hir::FnDecl<'_>,
        generic_params: &[hir::GenericParam<'_>],
        ident_span: Option<Span>,
    ) -> ty::PolyFnSig<'tcx> {
        debug!("ty_of_fn");

        let tcx = self.tcx();

        // We proactively collect all the inferred type params to emit a single error per fn def.
        let mut visitor = PlaceholderHirTyCollector::default();
        for ty in decl.inputs {
            visitor.visit_ty(ty);
        }
        let input_tys = decl.inputs.iter().map(|a| self.ty_of_arg(a, None));
        let output_ty = match decl.output {
            hir::FnRetTy::Return(ref output) => {
                visitor.visit_ty(output);
                self.ast_ty_to_ty(output)
            }
            hir::FnRetTy::DefaultReturn(..) => tcx.mk_unit(),
        };

        debug!("ty_of_fn: output_ty={:?}", output_ty);

        let bare_fn_ty =
            ty::Binder::bind(tcx.mk_fn_sig(input_tys, output_ty, decl.c_variadic, unsafety, abi));

        if !self.allow_ty_infer() {
            // We always collect the spans for placeholder types when evaluating `fn`s, but we
            // only want to emit an error complaining about them if infer types (`_`) are not
            // allowed. `allow_ty_infer` gates this behavior.
            crate::collect::placeholder_type_error(
                tcx,
                ident_span.map(|sp| sp.shrink_to_hi()).unwrap_or(DUMMY_SP),
                generic_params,
                visitor.0,
                ident_span.is_some(),
            );
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
        for br in late_bound_in_ret.difference(&late_bound_in_args) {
            let lifetime_name = match *br {
                ty::BrNamed(_, name) => format!("lifetime `{}`,", name),
                ty::BrAnon(_) | ty::BrEnv => "an anonymous lifetime".to_string(),
            };
            let mut err = struct_span_err!(
                tcx.sess,
                decl.output.span(),
                E0581,
                "return type references {} \
                                            which is not constrained by the fn input types",
                lifetime_name
            );
            if let ty::BrAnon(_) = *br {
                // The only way for an anonymous lifetime to wind up
                // in the return type but **also** be unconstrained is
                // if it only appears in "associated types" in the
                // input. See #47511 for an example. In this case,
                // though we can easily give a hint that ought to be
                // relevant.
                err.note(
                    "lifetimes appearing in an associated type \
                          are not considered constrained",
                );
            }
            err.emit();
        }

        bare_fn_ty
    }

    /// Given the bounds on an object, determines what single region bound (if any) we can
    /// use to summarize this type. The basic idea is that we will use the bound the user
    /// provided, if they provided one, and otherwise search the supertypes of trait bounds
    /// for region bounds. It may be that we can derive no bound at all, in which case
    /// we return `None`.
    fn compute_object_lifetime_bound(
        &self,
        span: Span,
        existential_predicates: ty::Binder<&'tcx ty::List<ty::ExistentialPredicate<'tcx>>>,
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
        if derived_region_bounds.iter().any(|&r| ty::ReStatic == *r) {
            return Some(tcx.lifetimes.re_static);
        }

        // Determine whether there is exactly one unique region in the set
        // of derived region bounds. If so, use that. Otherwise, report an
        // error.
        let r = derived_region_bounds[0];
        if derived_region_bounds[1..].iter().any(|r1| r != *r1) {
            struct_span_err!(
                tcx.sess,
                span,
                E0227,
                "ambiguous lifetime bound, explicit lifetime bound required"
            )
            .emit();
        }
        return Some(r);
    }
}

/// Collects together a list of bounds that are applied to some type,
/// after they've been converted into `ty` form (from the HIR
/// representations). These lists of bounds occur in many places in
/// Rust's syntax:
///
/// ```
/// trait Foo: Bar + Baz { }
///            ^^^^^^^^^ supertrait list bounding the `Self` type parameter
///
/// fn foo<T: Bar + Baz>() { }
///           ^^^^^^^^^ bounding the type parameter `T`
///
/// impl dyn Bar + Baz
///          ^^^^^^^^^ bounding the forgotten dynamic type
/// ```
///
/// Our representation is a bit mixed here -- in some cases, we
/// include the self type (e.g., `trait_bounds`) but in others we do
#[derive(Default, PartialEq, Eq, Clone, Debug)]
pub struct Bounds<'tcx> {
    /// A list of region bounds on the (implicit) self type. So if you
    /// had `T: 'a + 'b` this might would be a list `['a, 'b]` (but
    /// the `T` is not explicitly included).
    pub region_bounds: Vec<(ty::Region<'tcx>, Span)>,

    /// A list of trait bounds. So if you had `T: Debug` this would be
    /// `T: Debug`. Note that the self-type is explicit here.
    pub trait_bounds: Vec<(ty::PolyTraitRef<'tcx>, Span, Constness)>,

    /// A list of projection equality bounds. So if you had `T:
    /// Iterator<Item = u32>` this would include `<T as
    /// Iterator>::Item => u32`. Note that the self-type is explicit
    /// here.
    pub projection_bounds: Vec<(ty::PolyProjectionPredicate<'tcx>, Span)>,

    /// `Some` if there is *no* `?Sized` predicate. The `span`
    /// is the location in the source of the `T` declaration which can
    /// be cited as the source of the `T: Sized` requirement.
    pub implicitly_sized: Option<Span>,
}

impl<'tcx> Bounds<'tcx> {
    /// Converts a bounds list into a flat set of predicates (like
    /// where-clauses). Because some of our bounds listings (e.g.,
    /// regions) don't include the self-type, you must supply the
    /// self-type here (the `param_ty` parameter).
    pub fn predicates(
        &self,
        tcx: TyCtxt<'tcx>,
        param_ty: Ty<'tcx>,
    ) -> Vec<(ty::Predicate<'tcx>, Span)> {
        // If it could be sized, and is, add the `Sized` predicate.
        let sized_predicate = self.implicitly_sized.and_then(|span| {
            tcx.lang_items().sized_trait().map(|sized| {
                let trait_ref = ty::Binder::bind(ty::TraitRef {
                    def_id: sized,
                    substs: tcx.mk_substs_trait(param_ty, &[]),
                });
                (trait_ref.without_const().to_predicate(), span)
            })
        });

        sized_predicate
            .into_iter()
            .chain(
                self.region_bounds
                    .iter()
                    .map(|&(region_bound, span)| {
                        // Account for the binder being introduced below; no need to shift `param_ty`
                        // because, at present at least, it either only refers to early-bound regions,
                        // or it's a generic associated type that deliberately has escaping bound vars.
                        let region_bound = ty::fold::shift_region(tcx, region_bound, 1);
                        let outlives = ty::OutlivesPredicate(param_ty, region_bound);
                        (ty::Binder::bind(outlives).to_predicate(), span)
                    })
                    .chain(self.trait_bounds.iter().map(|&(bound_trait_ref, span, constness)| {
                        let predicate = bound_trait_ref.with_constness(constness).to_predicate();
                        (predicate, span)
                    }))
                    .chain(
                        self.projection_bounds
                            .iter()
                            .map(|&(projection, span)| (projection.to_predicate(), span)),
                    ),
            )
            .collect()
    }
}
