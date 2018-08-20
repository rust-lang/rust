// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Conversion from AST representation of types to the ty.rs
//! representation.  The main routine here is `ast_ty_to_ty()`: each use
//! is parameterized by an instance of `AstConv`.

use rustc_data_structures::accumulate_vec::AccumulateVec;
use rustc_data_structures::array_vec::ArrayVec;
use hir::{self, GenericArg, GenericArgs};
use hir::def::Def;
use hir::def_id::DefId;
use hir::HirVec;
use middle::resolve_lifetime as rl;
use namespace::Namespace;
use rustc::ty::subst::{Kind, Subst, Substs};
use rustc::traits;
use rustc::ty::{self, Ty, TyCtxt, ToPredicate, TypeFoldable};
use rustc::ty::{GenericParamDef, GenericParamDefKind};
use rustc::ty::wf::object_region_bounds;
use rustc_target::spec::abi;
use std::slice;
use require_c_abi_if_variadic;
use util::common::ErrorReported;
use util::nodemap::{FxHashSet, FxHashMap};
use errors::{FatalError, DiagnosticId};
use lint;

use std::iter;
use syntax::ast;
use syntax::ptr::P;
use syntax::feature_gate::{GateIssue, emit_feature_err};
use syntax_pos::{Span, MultiSpan};

pub trait AstConv<'gcx, 'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'a, 'gcx, 'tcx>;

    /// Returns the set of bounds in scope for the type parameter with
    /// the given id.
    fn get_type_parameter_bounds(&self, span: Span, def_id: DefId)
                                 -> ty::GenericPredicates<'tcx>;

    /// What lifetime should we use when a lifetime is omitted (and not elided)?
    fn re_infer(&self, span: Span, _def: Option<&ty::GenericParamDef>)
                -> Option<ty::Region<'tcx>>;

    /// What type should we use when a type is omitted?
    fn ty_infer(&self, span: Span) -> Ty<'tcx>;

    /// Same as ty_infer, but with a known type parameter definition.
    fn ty_infer_for_def(&self,
                        _def: &ty::GenericParamDef,
                        span: Span) -> Ty<'tcx> {
        self.ty_infer(span)
    }

    /// Projecting an associated type from a (potentially)
    /// higher-ranked trait reference is more complicated, because of
    /// the possibility of late-bound regions appearing in the
    /// associated type binding. This is not legal in function
    /// signatures for that reason. In a function body, we can always
    /// handle it because we can use inference variables to remove the
    /// late-bound regions.
    fn projected_ty_from_poly_trait_ref(&self,
                                        span: Span,
                                        item_def_id: DefId,
                                        poly_trait_ref: ty::PolyTraitRef<'tcx>)
                                        -> Ty<'tcx>;

    /// Normalize an associated type coming from the user.
    fn normalize_ty(&self, span: Span, ty: Ty<'tcx>) -> Ty<'tcx>;

    /// Invoked when we encounter an error from some prior pass
    /// (e.g. resolve) that is translated into a ty-error. This is
    /// used to help suppress derived errors typeck might otherwise
    /// report.
    fn set_tainted_by_errors(&self);

    fn record_ty(&self, hir_id: hir::HirId, ty: Ty<'tcx>, span: Span);
}

struct ConvertedBinding<'tcx> {
    item_name: ast::Ident,
    ty: Ty<'tcx>,
    span: Span,
}

#[derive(PartialEq)]
enum GenericArgPosition {
    Type,
    Value, // e.g. functions
    MethodCall,
}

// FIXME(#53525): these error codes should all be unified.
struct GenericArgMismatchErrorCode {
    lifetimes: (&'static str, &'static str),
    types: (&'static str, &'static str),
}

/// Dummy type used for the `Self` of a `TraitRef` created for converting
/// a trait object, and which gets removed in `ExistentialTraitRef`.
/// This type must not appear anywhere in other converted types.
const TRAIT_OBJECT_DUMMY_SELF: ty::TypeVariants<'static> = ty::TyInfer(ty::FreshTy(0));

impl<'o, 'gcx: 'tcx, 'tcx> dyn AstConv<'gcx, 'tcx>+'o {
    pub fn ast_region_to_region(&self,
        lifetime: &hir::Lifetime,
        def: Option<&ty::GenericParamDef>)
        -> ty::Region<'tcx>
    {
        let tcx = self.tcx();
        let lifetime_name = |def_id| {
            tcx.hir.name(tcx.hir.as_local_node_id(def_id).unwrap()).as_interned_str()
        };

        let hir_id = tcx.hir.node_to_hir_id(lifetime.id);
        let r = match tcx.named_region(hir_id) {
            Some(rl::Region::Static) => {
                tcx.types.re_static
            }

            Some(rl::Region::LateBound(debruijn, id, _)) => {
                let name = lifetime_name(id);
                tcx.mk_region(ty::ReLateBound(debruijn,
                    ty::BrNamed(id, name)))
            }

            Some(rl::Region::LateBoundAnon(debruijn, index)) => {
                tcx.mk_region(ty::ReLateBound(debruijn, ty::BrAnon(index)))
            }

            Some(rl::Region::EarlyBound(index, id, _)) => {
                let name = lifetime_name(id);
                tcx.mk_region(ty::ReEarlyBound(ty::EarlyBoundRegion {
                    def_id: id,
                    index,
                    name,
                }))
            }

            Some(rl::Region::Free(scope, id)) => {
                let name = lifetime_name(id);
                tcx.mk_region(ty::ReFree(ty::FreeRegion {
                    scope,
                    bound_region: ty::BrNamed(id, name)
                }))

                    // (*) -- not late-bound, won't change
            }

            None => {
                self.re_infer(lifetime.span, def)
                    .unwrap_or_else(|| {
                        // This indicates an illegal lifetime
                        // elision. `resolve_lifetime` should have
                        // reported an error in this case -- but if
                        // not, let's error out.
                        tcx.sess.delay_span_bug(lifetime.span, "unelided lifetime in signature");

                        // Supply some dummy value. We don't have an
                        // `re_error`, annoyingly, so use `'static`.
                        tcx.types.re_static
                    })
            }
        };

        debug!("ast_region_to_region(lifetime={:?}) yields {:?}",
                lifetime,
                r);

        r
    }

    /// Given a path `path` that refers to an item `I` with the declared generics `decl_generics`,
    /// returns an appropriate set of substitutions for this particular reference to `I`.
    pub fn ast_path_substs_for_ty(&self,
        span: Span,
        def_id: DefId,
        item_segment: &hir::PathSegment)
        -> &'tcx Substs<'tcx>
    {

        let (substs, assoc_bindings) = item_segment.with_generic_args(|generic_args| {
            self.create_substs_for_ast_path(
                span,
                def_id,
                generic_args,
                item_segment.infer_types,
                None,
            )
        });

        assoc_bindings.first().map(|b| Self::prohibit_assoc_ty_binding(self.tcx(), b.span));

        substs
    }

    /// Report error if there is an explicit type parameter when using `impl Trait`.
    fn check_impl_trait(
        tcx: TyCtxt,
        span: Span,
        seg: &hir::PathSegment,
        generics: &ty::Generics,
    ) -> bool {
        let explicit = !seg.infer_types;
        let impl_trait = generics.params.iter().any(|param| match param.kind {
            ty::GenericParamDefKind::Type {
                synthetic: Some(hir::SyntheticTyParamKind::ImplTrait), ..
            } => true,
            _ => false,
        });

        if explicit && impl_trait {
            let mut err = struct_span_err! {
                tcx.sess,
                span,
                E0632,
                "cannot provide explicit type parameters when `impl Trait` is \
                used in argument position."
            };

            err.emit();
        }

        impl_trait
    }

    /// Check that the correct number of generic arguments have been provided.
    /// Used specifically for function calls.
    pub fn check_generic_arg_count_for_call(
        tcx: TyCtxt,
        span: Span,
        def: &ty::Generics,
        seg: &hir::PathSegment,
        is_method_call: bool,
    ) -> bool {
        let empty_args = P(hir::GenericArgs {
            args: HirVec::new(), bindings: HirVec::new(), parenthesized: false,
        });
        let suppress_mismatch = Self::check_impl_trait(tcx, span, seg, &def);
        Self::check_generic_arg_count(
            tcx,
            span,
            def,
            if let Some(ref args) = seg.args {
                args
            } else {
                &empty_args
            },
            if is_method_call {
                GenericArgPosition::MethodCall
            } else {
                GenericArgPosition::Value
            },
            def.parent.is_none() && def.has_self, // `has_self`
            seg.infer_types || suppress_mismatch, // `infer_types`
            GenericArgMismatchErrorCode {
                lifetimes: ("E0090", "E0088"),
                types: ("E0089", "E0087"),
            },
        )
    }

    /// Check that the correct number of generic arguments have been provided.
    /// This is used both for datatypes and function calls.
    fn check_generic_arg_count(
        tcx: TyCtxt,
        span: Span,
        def: &ty::Generics,
        args: &hir::GenericArgs,
        position: GenericArgPosition,
        has_self: bool,
        infer_types: bool,
        error_codes: GenericArgMismatchErrorCode,
    ) -> bool {
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
            };
        }

        if position != GenericArgPosition::Type && !args.bindings.is_empty() {
            AstConv::prohibit_assoc_ty_binding(tcx, args.bindings[0].span);
        }

        // Prohibit explicit lifetime arguments if late-bound lifetime parameters are present.
        if !infer_lifetimes {
            if let Some(span_late) = def.has_late_bound_regions {
                let msg = "cannot specify lifetime arguments explicitly \
                           if late bound lifetime parameters are present";
                let note = "the late bound lifetime parameter is introduced here";
                let span = args.args[0].span();
                if position == GenericArgPosition::Value
                    && arg_counts.lifetimes != param_counts.lifetimes {
                    let mut err = tcx.sess.struct_span_err(span, msg);
                    err.span_note(span_late, note);
                    err.emit();
                    return true;
                } else {
                    let mut multispan = MultiSpan::from_span(span);
                    multispan.push_span_label(span_late, note.to_string());
                    tcx.lint_node(lint::builtin::LATE_BOUND_LIFETIME_ARGUMENTS,
                                  args.args[0].id(), multispan, msg);
                    return false;
                }
            }
        }

        let check_kind_count = |error_code: (&str, &str),
                                kind,
                                required,
                                permitted,
                                provided,
                                offset| {
            // We enforce the following: `required` <= `provided` <= `permitted`.
            // For kinds without defaults (i.e. lifetimes), `required == permitted`.
            // For other kinds (i.e. types), `permitted` may be greater than `required`.
            if required <= provided && provided <= permitted {
                return false;
            }

            // Unfortunately lifetime and type parameter mismatches are typically styled
            // differently in diagnostics, which means we have a few cases to consider here.
            let (bound, quantifier) = if required != permitted {
                if provided < required {
                    (required, "at least ")
                } else { // provided > permitted
                    (permitted, "at most ")
                }
            } else {
                (required, "")
            };

            let mut span = span;
            let label = if required == permitted && provided > permitted {
                let diff = provided - permitted;
                if diff == 1 {
                    // In the case when the user has provided too many arguments,
                    // we want to point to the first unexpected argument.
                    let first_superfluous_arg: &GenericArg = &args.args[offset + permitted];
                    span = first_superfluous_arg.span();
                }
                format!(
                    "{}unexpected {} argument{}",
                    if diff != 1 { format!("{} ", diff) } else { String::new() },
                    kind,
                    if diff != 1 { "s" } else { "" },
                )
            } else {
                format!(
                    "expected {}{} {} argument{}",
                    quantifier,
                    bound,
                    kind,
                    if required != 1 { "s" } else { "" },
                )
            };

            tcx.sess.struct_span_err_with_code(
                span,
                &format!(
                    "wrong number of {} arguments: expected {}{}, found {}",
                    kind,
                    quantifier,
                    bound,
                    provided,
                ),
                DiagnosticId::Error({
                    if provided <= permitted {
                        error_code.0
                    } else {
                        error_code.1
                    }
                }.into())
            ).span_label(span, label).emit();

            provided > required // `suppress_error`
        };

        if !infer_lifetimes || arg_counts.lifetimes > param_counts.lifetimes {
            check_kind_count(
                error_codes.lifetimes,
                "lifetime",
                param_counts.lifetimes,
                param_counts.lifetimes,
                arg_counts.lifetimes,
                0,
            );
        }
        if !infer_types
            || arg_counts.types > param_counts.types - defaults.types - has_self as usize {
            check_kind_count(
                error_codes.types,
                "type",
                param_counts.types - defaults.types - has_self as usize,
                param_counts.types - has_self as usize,
                arg_counts.types,
                arg_counts.lifetimes,
            )
        } else {
            false
        }
    }

    /// Creates the relevant generic argument substitutions
    /// corresponding to a set of generic parameters.
    pub fn create_substs_for_generic_args<'a, 'b, A, P, I>(
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        def_id: DefId,
        parent_substs: &[Kind<'tcx>],
        has_self: bool,
        self_ty: Option<Ty<'tcx>>,
        args_for_def_id: A,
        provided_kind: P,
        inferred_kind: I,
    ) -> &'tcx Substs<'tcx> where
        A: Fn(DefId) -> (Option<&'b GenericArgs>, bool),
        P: Fn(&GenericParamDef, &GenericArg) -> Kind<'tcx>,
        I: Fn(Option<&[Kind<'tcx>]>, &GenericParamDef, bool) -> Kind<'tcx>
    {
        // Collect the segments of the path: we need to substitute arguments
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
        // methods in subst.rs so that we can iterate over the arguments and
        // parameters in lock-step linearly, rather than trying to match each pair.
        let mut substs: AccumulateVec<[Kind<'tcx>; 8]> = if count <= 8 {
            AccumulateVec::Array(ArrayVec::new())
        } else {
            AccumulateVec::Heap(Vec::with_capacity(count))
        };

        fn push_kind<'tcx>(substs: &mut AccumulateVec<[Kind<'tcx>; 8]>, kind: Kind<'tcx>) {
            match substs {
                AccumulateVec::Array(ref mut arr) => arr.push(kind),
                AccumulateVec::Heap(ref mut vec) => vec.push(kind),
            }
        }

        // Iterate over each segment of the path.
        while let Some((def_id, defs)) = stack.pop() {
            let mut params = defs.params.iter().peekable();

            // If we have already computed substitutions for parents, we can use those directly.
            while let Some(&param) = params.peek() {
                if let Some(&kind) = parent_substs.get(param.index as usize) {
                    push_kind(&mut substs, kind);
                    params.next();
                } else {
                    break;
                }
            }

            // (Unless it's been handled in `parent_substs`) `Self` is handled first.
            if has_self {
                if let Some(&param) = params.peek() {
                    if param.index == 0 {
                        if let GenericParamDefKind::Type { .. } = param.kind {
                            push_kind(&mut substs, self_ty.map(|ty| ty.into())
                                .unwrap_or_else(|| inferred_kind(None, param, true)));
                            params.next();
                        }
                    }
                }
            }

            // Check whether this segment takes generic arguments and the user has provided any.
            let (generic_args, infer_types) = args_for_def_id(def_id);

            let mut args = generic_args.iter().flat_map(|generic_args| generic_args.args.iter())
                .peekable();

            loop {
                // We're going to iterate through the generic arguments that the user
                // provided, matching them with the generic parameters we expect.
                // Mismatches can occur as a result of elided lifetimes, or for malformed
                // input. We try to handle both sensibly.
                match (args.peek(), params.peek()) {
                    (Some(&arg), Some(&param)) => {
                        match (arg, &param.kind) {
                            (GenericArg::Lifetime(_), GenericParamDefKind::Lifetime)
                            | (GenericArg::Type(_), GenericParamDefKind::Type { .. }) => {
                                push_kind(&mut substs, provided_kind(param, arg));
                                args.next();
                                params.next();
                            }
                            (GenericArg::Lifetime(_), GenericParamDefKind::Type { .. }) => {
                                // We expected a type argument, but got a lifetime
                                // argument. This is an error, but we need to handle it
                                // gracefully so we can report sensible errors. In this
                                // case, we're simply going to infer this argument.
                                args.next();
                            }
                            (GenericArg::Type(_), GenericParamDefKind::Lifetime) => {
                                // We expected a lifetime argument, but got a type
                                // argument. That means we're inferring the lifetimes.
                                push_kind(&mut substs, inferred_kind(None, param, infer_types));
                                params.next();
                            }
                        }
                    }
                    (Some(_), None) => {
                        // We should never be able to reach this point with well-formed input.
                        // Getting to this point means the user supplied more arguments than
                        // there are parameters.
                        args.next();
                    }
                    (None, Some(&param)) => {
                        // If there are fewer arguments than parameters, it means
                        // we're inferring the remaining arguments.
                        match param.kind {
                            GenericParamDefKind::Lifetime | GenericParamDefKind::Type { .. } => {
                                let kind = inferred_kind(Some(&substs), param, infer_types);
                                push_kind(&mut substs, kind);
                            }
                        }
                        args.next();
                        params.next();
                    }
                    (None, None) => break,
                }
            }
        }

        tcx.intern_substs(&substs)
    }

    /// Given the type/region arguments provided to some path (along with
    /// an implicit Self, if this is a trait reference) returns the complete
    /// set of substitutions. This may involve applying defaulted type parameters.
    ///
    /// Note that the type listing given here is *exactly* what the user provided.
    fn create_substs_for_ast_path(&self,
        span: Span,
        def_id: DefId,
        generic_args: &hir::GenericArgs,
        infer_types: bool,
        self_ty: Option<Ty<'tcx>>)
        -> (&'tcx Substs<'tcx>, Vec<ConvertedBinding<'tcx>>)
    {
        // If the type is parameterized by this region, then replace this
        // region with the current anon region binding (in other words,
        // whatever & would get replaced with).
        debug!("create_substs_for_ast_path(def_id={:?}, self_ty={:?}, \
               generic_args={:?})",
               def_id, self_ty, generic_args);

        let tcx = self.tcx();
        let generic_params = tcx.generics_of(def_id);

        // If a self-type was declared, one should be provided.
        assert_eq!(generic_params.has_self, self_ty.is_some());

        let has_self = generic_params.has_self;
        Self::check_generic_arg_count(
            self.tcx(),
            span,
            &generic_params,
            &generic_args,
            GenericArgPosition::Type,
            has_self,
            infer_types,
            GenericArgMismatchErrorCode {
                lifetimes: ("E0107", "E0107"),
                types: ("E0243", "E0244"),
            },
        );

        let is_object = self_ty.map_or(false, |ty| ty.sty == TRAIT_OBJECT_DUMMY_SELF);
        let default_needs_object_self = |param: &ty::GenericParamDef| {
            if let GenericParamDefKind::Type { has_default, .. } = param.kind {
                if is_object && has_default {
                    if tcx.at(span).type_of(param.def_id).has_self_ty() {
                        // There is no suitable inference default for a type parameter
                        // that references self, in an object type.
                        return true;
                    }
                }
            }

            false
        };

        let substs = Self::create_substs_for_generic_args(
            self.tcx(),
            def_id,
            &[][..],
            self_ty.is_some(),
            self_ty,
            // Provide the generic args, and whether types should be inferred.
            |_| (Some(generic_args), infer_types),
            // Provide substitutions for parameters for which (valid) arguments have been provided.
            |param, arg| {
                match (&param.kind, arg) {
                    (GenericParamDefKind::Lifetime, GenericArg::Lifetime(lt)) => {
                        self.ast_region_to_region(&lt, Some(param)).into()
                    }
                    (GenericParamDefKind::Type { .. }, GenericArg::Type(ty)) => {
                        self.ast_ty_to_ty(&ty).into()
                    }
                    _ => unreachable!(),
                }
            },
            // Provide substitutions for parameters for which arguments are inferred.
            |substs, param, infer_types| {
                match param.kind {
                    GenericParamDefKind::Lifetime => tcx.types.re_static.into(),
                    GenericParamDefKind::Type { has_default, .. } => {
                        if !infer_types && has_default {
                            // No type parameter provided, but a default exists.

                            // If we are converting an object type, then the
                            // `Self` parameter is unknown. However, some of the
                            // other type parameters may reference `Self` in their
                            // defaults. This will lead to an ICE if we are not
                            // careful!
                            if default_needs_object_self(param) {
                                struct_span_err!(tcx.sess, span, E0393,
                                                    "the type parameter `{}` must be explicitly \
                                                    specified",
                                                    param.name)
                                    .span_label(span,
                                                format!("missing reference to `{}`", param.name))
                                    .note(&format!("because of the default `Self` reference, \
                                                    type parameters must be specified on object \
                                                    types"))
                                    .emit();
                                tcx.types.err.into()
                            } else {
                                // This is a default type parameter.
                                self.normalize_ty(
                                    span,
                                    tcx.at(span).type_of(param.def_id)
                                        .subst_spanned(tcx, substs.unwrap(), Some(span))
                                ).into()
                            }
                        } else if infer_types {
                            // No type parameters were provided, we can infer all.
                            if !default_needs_object_self(param) {
                                self.ty_infer_for_def(param, span).into()
                            } else {
                                self.ty_infer(span).into()
                            }
                        } else {
                            // We've already errored above about the mismatch.
                            tcx.types.err.into()
                        }
                    }
                }
            },
        );

        let assoc_bindings = generic_args.bindings.iter().map(|binding| {
            ConvertedBinding {
                item_name: binding.ident,
                ty: self.ast_ty_to_ty(&binding.ty),
                span: binding.span,
            }
        }).collect();

        debug!("create_substs_for_ast_path(generic_params={:?}, self_ty={:?}) -> {:?}",
               generic_params, self_ty, substs);

        (substs, assoc_bindings)
    }

    /// Instantiates the path for the given trait reference, assuming that it's
    /// bound to a valid trait type. Returns the def_id for the defining trait.
    /// The type _cannot_ be a type other than a trait type.
    ///
    /// If the `projections` argument is `None`, then assoc type bindings like `Foo<T=X>`
    /// are disallowed. Otherwise, they are pushed onto the vector given.
    pub fn instantiate_mono_trait_ref(&self,
        trait_ref: &hir::TraitRef,
        self_ty: Ty<'tcx>)
        -> ty::TraitRef<'tcx>
    {
        self.prohibit_generics(trait_ref.path.segments.split_last().unwrap().1);

        let trait_def_id = self.trait_def_id(trait_ref);
        self.ast_path_to_mono_trait_ref(trait_ref.path.span,
                                        trait_def_id,
                                        self_ty,
                                        trait_ref.path.segments.last().unwrap())
    }

    /// Get the DefId of the given trait ref. It _must_ actually be a trait.
    fn trait_def_id(&self, trait_ref: &hir::TraitRef) -> DefId {
        let path = &trait_ref.path;
        match path.def {
            Def::Trait(trait_def_id) => trait_def_id,
            Def::TraitAlias(alias_def_id) => alias_def_id,
            Def::Err => {
                FatalError.raise();
            }
            _ => unreachable!(),
        }
    }

    /// The given `trait_ref` must actually be trait.
    pub(super) fn instantiate_poly_trait_ref_inner(&self,
        trait_ref: &hir::TraitRef,
        self_ty: Ty<'tcx>,
        poly_projections: &mut Vec<ty::PolyProjectionPredicate<'tcx>>,
        speculative: bool)
        -> ty::PolyTraitRef<'tcx>
    {
        let trait_def_id = self.trait_def_id(trait_ref);

        debug!("ast_path_to_poly_trait_ref({:?}, def_id={:?})", trait_ref, trait_def_id);

        self.prohibit_generics(trait_ref.path.segments.split_last().unwrap().1);

        let (substs, assoc_bindings) =
            self.create_substs_for_ast_trait_ref(trait_ref.path.span,
                                                 trait_def_id,
                                                 self_ty,
                                                 trait_ref.path.segments.last().unwrap());
        let poly_trait_ref = ty::Binder::bind(ty::TraitRef::new(trait_def_id, substs));

        let mut dup_bindings = FxHashMap::default();
        poly_projections.extend(assoc_bindings.iter().filter_map(|binding| {
            // specify type to assert that error was already reported in Err case:
            let predicate: Result<_, ErrorReported> =
                self.ast_type_binding_to_poly_projection_predicate(
                    trait_ref.ref_id, poly_trait_ref, binding, speculative, &mut dup_bindings);
            predicate.ok() // ok to ignore Err() because ErrorReported (see above)
        }));

        debug!("ast_path_to_poly_trait_ref({:?}, projections={:?}) -> {:?}",
               trait_ref, poly_projections, poly_trait_ref);
        poly_trait_ref
    }

    pub fn instantiate_poly_trait_ref(&self,
        poly_trait_ref: &hir::PolyTraitRef,
        self_ty: Ty<'tcx>,
        poly_projections: &mut Vec<ty::PolyProjectionPredicate<'tcx>>)
        -> ty::PolyTraitRef<'tcx>
    {
        self.instantiate_poly_trait_ref_inner(&poly_trait_ref.trait_ref, self_ty,
                                              poly_projections, false)
    }

    fn ast_path_to_mono_trait_ref(&self,
                                  span: Span,
                                  trait_def_id: DefId,
                                  self_ty: Ty<'tcx>,
                                  trait_segment: &hir::PathSegment)
                                  -> ty::TraitRef<'tcx>
    {
        let (substs, assoc_bindings) =
            self.create_substs_for_ast_trait_ref(span,
                                                 trait_def_id,
                                                 self_ty,
                                                 trait_segment);
        assoc_bindings.first().map(|b| AstConv::prohibit_assoc_ty_binding(self.tcx(), b.span));
        ty::TraitRef::new(trait_def_id, substs)
    }

    fn create_substs_for_ast_trait_ref(&self,
                                       span: Span,
                                       trait_def_id: DefId,
                                       self_ty: Ty<'tcx>,
                                       trait_segment: &hir::PathSegment)
                                       -> (&'tcx Substs<'tcx>, Vec<ConvertedBinding<'tcx>>)
    {
        debug!("create_substs_for_ast_trait_ref(trait_segment={:?})",
               trait_segment);

        let trait_def = self.tcx().trait_def(trait_def_id);

        if !self.tcx().features().unboxed_closures &&
            trait_segment.with_generic_args(|generic_args| generic_args.parenthesized)
            != trait_def.paren_sugar {
            // For now, require that parenthetical notation be used only with `Fn()` etc.
            let msg = if trait_def.paren_sugar {
                "the precise format of `Fn`-family traits' type parameters is subject to change. \
                 Use parenthetical notation (Fn(Foo, Bar) -> Baz) instead"
            } else {
                "parenthetical notation is only stable when used with `Fn`-family traits"
            };
            emit_feature_err(&self.tcx().sess.parse_sess, "unboxed_closures",
                             span, GateIssue::Language, msg);
        }

        trait_segment.with_generic_args(|generic_args| {
            self.create_substs_for_ast_path(span,
                                            trait_def_id,
                                            generic_args,
                                            trait_segment.infer_types,
                                            Some(self_ty))
        })
    }

    fn trait_defines_associated_type_named(&self,
                                           trait_def_id: DefId,
                                           assoc_name: ast::Ident)
                                           -> bool
    {
        self.tcx().associated_items(trait_def_id).any(|item| {
            item.kind == ty::AssociatedKind::Type &&
            self.tcx().hygienic_eq(assoc_name, item.ident, trait_def_id)
        })
    }

    fn ast_type_binding_to_poly_projection_predicate(
        &self,
        ref_id: ast::NodeId,
        trait_ref: ty::PolyTraitRef<'tcx>,
        binding: &ConvertedBinding<'tcx>,
        speculative: bool,
        dup_bindings: &mut FxHashMap<DefId, Span>)
        -> Result<ty::PolyProjectionPredicate<'tcx>, ErrorReported>
    {
        let tcx = self.tcx();

        if !speculative {
            // Given something like `U : SomeTrait<T=X>`, we want to produce a
            // predicate like `<U as SomeTrait>::T = X`. This is somewhat
            // subtle in the event that `T` is defined in a supertrait of
            // `SomeTrait`, because in that case we need to upcast.
            //
            // That is, consider this case:
            //
            // ```
            // trait SubTrait : SuperTrait<int> { }
            // trait SuperTrait<A> { type T; }
            //
            // ... B : SubTrait<T=foo> ...
            // ```
            //
            // We want to produce `<B as SuperTrait<int>>::T == foo`.

            // Find any late-bound regions declared in `ty` that are not
            // declared in the trait-ref. These are not wellformed.
            //
            // Example:
            //
            //     for<'a> <T as Iterator>::Item = &'a str // <-- 'a is bad
            //     for<'a> <T as FnMut<(&'a u32,)>>::Output = &'a str // <-- 'a is ok
            let late_bound_in_trait_ref = tcx.collect_constrained_late_bound_regions(&trait_ref);
            let late_bound_in_ty =
                tcx.collect_referenced_late_bound_regions(&ty::Binder::bind(binding.ty));
            debug!("late_bound_in_trait_ref = {:?}", late_bound_in_trait_ref);
            debug!("late_bound_in_ty = {:?}", late_bound_in_ty);
            for br in late_bound_in_ty.difference(&late_bound_in_trait_ref) {
                let br_name = match *br {
                    ty::BrNamed(_, name) => name,
                    _ => {
                        span_bug!(
                            binding.span,
                            "anonymous bound region {:?} in binding but not trait ref",
                            br);
                    }
                };
                struct_span_err!(tcx.sess,
                                binding.span,
                                E0582,
                                "binding for associated type `{}` references lifetime `{}`, \
                                which does not appear in the trait input types",
                                binding.item_name, br_name)
                    .emit();
            }
        }

        let candidate = if self.trait_defines_associated_type_named(trait_ref.def_id(),
                                                                    binding.item_name) {
            // Simple case: X is defined in the current trait.
            Ok(trait_ref)
        } else {
            // Otherwise, we have to walk through the supertraits to find
            // those that do.
            let candidates = traits::supertraits(tcx, trait_ref).filter(|r| {
                self.trait_defines_associated_type_named(r.def_id(), binding.item_name)
            });
            self.one_bound_for_assoc_type(candidates, &trait_ref.to_string(),
                                          binding.item_name, binding.span)
        }?;

        let (assoc_ident, def_scope) =
            tcx.adjust_ident(binding.item_name, candidate.def_id(), ref_id);
        let assoc_ty = tcx.associated_items(candidate.def_id()).find(|i| {
            i.kind == ty::AssociatedKind::Type && i.ident.modern() == assoc_ident
        }).expect("missing associated type");

        if !assoc_ty.vis.is_accessible_from(def_scope, tcx) {
            let msg = format!("associated type `{}` is private", binding.item_name);
            tcx.sess.span_err(binding.span, &msg);
        }
        tcx.check_stability(assoc_ty.def_id, Some(ref_id), binding.span);

        if !speculative {
            dup_bindings.entry(assoc_ty.def_id)
                .and_modify(|prev_span| {
                    let mut err = self.tcx().struct_span_lint_node(
                        ::rustc::lint::builtin::DUPLICATE_ASSOCIATED_TYPE_BINDINGS,
                        ref_id,
                        binding.span,
                        &format!("associated type binding `{}` specified more than once",
                                binding.item_name)
                    );
                    err.span_label(binding.span, "used more than once");
                    err.span_label(*prev_span, format!("first use of `{}`", binding.item_name));
                    err.emit();
                })
                .or_insert(binding.span);
        }

        Ok(candidate.map_bound(|trait_ref| {
            ty::ProjectionPredicate {
                projection_ty: ty::ProjectionTy::from_ref_and_name(
                    tcx,
                    trait_ref,
                    binding.item_name,
                ),
                ty: binding.ty,
            }
        }))
    }

    fn ast_path_to_ty(&self,
        span: Span,
        did: DefId,
        item_segment: &hir::PathSegment)
        -> Ty<'tcx>
    {
        let substs = self.ast_path_substs_for_ty(span, did, item_segment);
        self.normalize_ty(
            span,
            self.tcx().at(span).type_of(did).subst(self.tcx(), substs)
        )
    }

    /// Transform a PolyTraitRef into a PolyExistentialTraitRef by
    /// removing the dummy Self type (TRAIT_OBJECT_DUMMY_SELF).
    fn trait_ref_to_existential(&self, trait_ref: ty::TraitRef<'tcx>)
                                -> ty::ExistentialTraitRef<'tcx> {
        assert_eq!(trait_ref.self_ty().sty, TRAIT_OBJECT_DUMMY_SELF);
        ty::ExistentialTraitRef::erase_self_ty(self.tcx(), trait_ref)
    }

    fn conv_object_ty_poly_trait_ref(&self,
        span: Span,
        trait_bounds: &[hir::PolyTraitRef],
        lifetime: &hir::Lifetime)
        -> Ty<'tcx>
    {
        let tcx = self.tcx();

        if trait_bounds.is_empty() {
            span_err!(tcx.sess, span, E0224,
                      "at least one non-builtin trait is required for an object type");
            return tcx.types.err;
        }

        let mut projection_bounds = vec![];
        let dummy_self = tcx.mk_ty(TRAIT_OBJECT_DUMMY_SELF);
        let principal = self.instantiate_poly_trait_ref(&trait_bounds[0],
                                                        dummy_self,
                                                        &mut projection_bounds);

        for trait_bound in trait_bounds[1..].iter() {
            // Sanity check for non-principal trait bounds
            self.instantiate_poly_trait_ref(trait_bound,
                                            dummy_self,
                                            &mut vec![]);
        }

        let (mut auto_traits, trait_bounds) = split_auto_traits(tcx, &trait_bounds[1..]);

        if !trait_bounds.is_empty() {
            let b = &trait_bounds[0];
            let span = b.trait_ref.path.span;
            struct_span_err!(self.tcx().sess, span, E0225,
                "only auto traits can be used as additional traits in a trait object")
                .span_label(span, "non-auto additional trait")
                .emit();
        }

        // Erase the dummy_self (TRAIT_OBJECT_DUMMY_SELF) used above.
        let existential_principal = principal.map_bound(|trait_ref| {
            self.trait_ref_to_existential(trait_ref)
        });
        let existential_projections = projection_bounds.iter().map(|bound| {
            bound.map_bound(|b| {
                let trait_ref = self.trait_ref_to_existential(b.projection_ty.trait_ref(tcx));
                ty::ExistentialProjection {
                    ty: b.ty,
                    item_def_id: b.projection_ty.item_def_id,
                    substs: trait_ref.substs,
                }
            })
        });

        // check that there are no gross object safety violations,
        // most importantly, that the supertraits don't contain Self,
        // to avoid ICE-s.
        let object_safety_violations =
            tcx.astconv_object_safety_violations(principal.def_id());
        if !object_safety_violations.is_empty() {
            tcx.report_object_safety_error(
                span, principal.def_id(), object_safety_violations)
                .emit();
            return tcx.types.err;
        }

        let mut associated_types = FxHashSet::default();
        for tr in traits::supertraits(tcx, principal) {
            associated_types.extend(tcx.associated_items(tr.def_id())
                .filter(|item| item.kind == ty::AssociatedKind::Type)
                .map(|item| item.def_id));
        }

        for projection_bound in &projection_bounds {
            associated_types.remove(&projection_bound.projection_def_id());
        }

        for item_def_id in associated_types {
            let assoc_item = tcx.associated_item(item_def_id);
            let trait_def_id = assoc_item.container.id();
            struct_span_err!(tcx.sess, span, E0191,
                "the value of the associated type `{}` (from the trait `{}`) must be specified",
                        assoc_item.ident,
                        tcx.item_path_str(trait_def_id))
                        .span_label(span, format!(
                            "missing associated type `{}` value", assoc_item.ident))
                        .emit();
        }

        // Dedup auto traits so that `dyn Trait + Send + Send` is the same as `dyn Trait + Send`.
        auto_traits.sort();
        auto_traits.dedup();

        // skip_binder is okay, because the predicates are re-bound.
        let mut v =
            iter::once(ty::ExistentialPredicate::Trait(*existential_principal.skip_binder()))
            .chain(auto_traits.into_iter().map(ty::ExistentialPredicate::AutoTrait))
            .chain(existential_projections
                   .map(|x| ty::ExistentialPredicate::Projection(*x.skip_binder())))
            .collect::<AccumulateVec<[_; 8]>>();
        v.sort_by(|a, b| a.stable_cmp(tcx, b));
        let existential_predicates = ty::Binder::bind(tcx.mk_existential_predicates(v.into_iter()));


        // Explicitly specified region bound. Use that.
        let region_bound = if !lifetime.is_elided() {
            self.ast_region_to_region(lifetime, None)
        } else {
            self.compute_object_lifetime_bound(span, existential_predicates).unwrap_or_else(|| {
                let hir_id = tcx.hir.node_to_hir_id(lifetime.id);
                if tcx.named_region(hir_id).is_some() {
                    self.ast_region_to_region(lifetime, None)
                } else {
                    self.re_infer(span, None).unwrap_or_else(|| {
                        span_err!(tcx.sess, span, E0228,
                                  "the lifetime bound for this object type cannot be deduced \
                                   from context; please supply an explicit bound");
                        tcx.types.re_static
                    })
                }
            })
        };

        debug!("region_bound: {:?}", region_bound);

        let ty = tcx.mk_dynamic(existential_predicates, region_bound);
        debug!("trait_object_type: {:?}", ty);
        ty
    }

    fn report_ambiguous_associated_type(&self,
                                        span: Span,
                                        type_str: &str,
                                        trait_str: &str,
                                        name: &str) {
        struct_span_err!(self.tcx().sess, span, E0223, "ambiguous associated type")
            .span_label(span, "ambiguous associated type")
            .note(&format!("specify the type using the syntax `<{} as {}>::{}`",
                  type_str, trait_str, name))
            .emit();

    }

    // Search for a bound on a type parameter which includes the associated item
    // given by `assoc_name`. `ty_param_def_id` is the `DefId` for the type parameter
    // This function will fail if there are no suitable bounds or there is
    // any ambiguity.
    fn find_bound_for_assoc_item(&self,
                                 ty_param_def_id: DefId,
                                 assoc_name: ast::Ident,
                                 span: Span)
                                 -> Result<ty::PolyTraitRef<'tcx>, ErrorReported>
    {
        let tcx = self.tcx();

        let bounds: Vec<_> = self.get_type_parameter_bounds(span, ty_param_def_id)
            .predicates.into_iter().filter_map(|p| p.to_opt_poly_trait_ref()).collect();

        // Check that there is exactly one way to find an associated type with the
        // correct name.
        let suitable_bounds =
            traits::transitive_bounds(tcx, &bounds)
            .filter(|b| self.trait_defines_associated_type_named(b.def_id(), assoc_name));

        let param_node_id = tcx.hir.as_local_node_id(ty_param_def_id).unwrap();
        let param_name = tcx.hir.ty_param_name(param_node_id);
        self.one_bound_for_assoc_type(suitable_bounds,
                                      &param_name.as_str(),
                                      assoc_name,
                                      span)
    }


    // Checks that bounds contains exactly one element and reports appropriate
    // errors otherwise.
    fn one_bound_for_assoc_type<I>(&self,
                                mut bounds: I,
                                ty_param_name: &str,
                                assoc_name: ast::Ident,
                                span: Span)
        -> Result<ty::PolyTraitRef<'tcx>, ErrorReported>
        where I: Iterator<Item=ty::PolyTraitRef<'tcx>>
    {
        let bound = match bounds.next() {
            Some(bound) => bound,
            None => {
                struct_span_err!(self.tcx().sess, span, E0220,
                          "associated type `{}` not found for `{}`",
                          assoc_name,
                          ty_param_name)
                  .span_label(span, format!("associated type `{}` not found", assoc_name))
                  .emit();
                return Err(ErrorReported);
            }
        };

        if let Some(bound2) = bounds.next() {
            let bounds = iter::once(bound).chain(iter::once(bound2)).chain(bounds);
            let mut err = struct_span_err!(
                self.tcx().sess, span, E0221,
                "ambiguous associated type `{}` in bounds of `{}`",
                assoc_name,
                ty_param_name);
            err.span_label(span, format!("ambiguous associated type `{}`", assoc_name));

            for bound in bounds {
                let bound_span = self.tcx().associated_items(bound.def_id()).find(|item| {
                    item.kind == ty::AssociatedKind::Type &&
                    self.tcx().hygienic_eq(assoc_name, item.ident, bound.def_id())
                })
                .and_then(|item| self.tcx().hir.span_if_local(item.def_id));

                if let Some(span) = bound_span {
                    err.span_label(span, format!("ambiguous `{}` from `{}`",
                                                  assoc_name,
                                                  bound));
                } else {
                    span_note!(&mut err, span,
                               "associated type `{}` could derive from `{}`",
                               ty_param_name,
                               bound);
                }
            }
            err.emit();
        }

        return Ok(bound);
    }

    // Create a type from a path to an associated type.
    // For a path A::B::C::D, ty and ty_path_def are the type and def for A::B::C
    // and item_segment is the path segment for D. We return a type and a def for
    // the whole path.
    // Will fail except for T::A and Self::A; i.e., if ty/ty_path_def are not a type
    // parameter or Self.
    pub fn associated_path_def_to_ty(&self,
                                     ref_id: ast::NodeId,
                                     span: Span,
                                     ty: Ty<'tcx>,
                                     ty_path_def: Def,
                                     item_segment: &hir::PathSegment)
                                     -> (Ty<'tcx>, Def)
    {
        let tcx = self.tcx();
        let assoc_name = item_segment.ident;

        debug!("associated_path_def_to_ty: {:?}::{}", ty, assoc_name);

        self.prohibit_generics(slice::from_ref(item_segment));

        // Find the type of the associated item, and the trait where the associated
        // item is declared.
        let bound = match (&ty.sty, ty_path_def) {
            (_, Def::SelfTy(Some(_), Some(impl_def_id))) => {
                // `Self` in an impl of a trait - we have a concrete self type and a
                // trait reference.
                let trait_ref = match tcx.impl_trait_ref(impl_def_id) {
                    Some(trait_ref) => trait_ref,
                    None => {
                        // A cycle error occurred, most likely.
                        return (tcx.types.err, Def::Err);
                    }
                };

                let candidates =
                    traits::supertraits(tcx, ty::Binder::bind(trait_ref))
                    .filter(|r| self.trait_defines_associated_type_named(r.def_id(), assoc_name));

                match self.one_bound_for_assoc_type(candidates, "Self", assoc_name, span) {
                    Ok(bound) => bound,
                    Err(ErrorReported) => return (tcx.types.err, Def::Err),
                }
            }
            (&ty::TyParam(_), Def::SelfTy(Some(param_did), None)) |
            (&ty::TyParam(_), Def::TyParam(param_did)) => {
                match self.find_bound_for_assoc_item(param_did, assoc_name, span) {
                    Ok(bound) => bound,
                    Err(ErrorReported) => return (tcx.types.err, Def::Err),
                }
            }
            _ => {
                // Don't print TyErr to the user.
                if !ty.references_error() {
                    self.report_ambiguous_associated_type(span,
                                                          &ty.to_string(),
                                                          "Trait",
                                                          &assoc_name.as_str());
                }
                return (tcx.types.err, Def::Err);
            }
        };

        let trait_did = bound.def_id();
        let (assoc_ident, def_scope) = tcx.adjust_ident(assoc_name, trait_did, ref_id);
        let item = tcx.associated_items(trait_did).find(|i| {
            Namespace::from(i.kind) == Namespace::Type &&
            i.ident.modern() == assoc_ident
        })
        .expect("missing associated type");

        let ty = self.projected_ty_from_poly_trait_ref(span, item.def_id, bound);
        let ty = self.normalize_ty(span, ty);

        let def = Def::AssociatedTy(item.def_id);
        if !item.vis.is_accessible_from(def_scope, tcx) {
            let msg = format!("{} `{}` is private", def.kind_name(), assoc_name);
            tcx.sess.span_err(span, &msg);
        }
        tcx.check_stability(item.def_id, Some(ref_id), span);

        (ty, def)
    }

    fn qpath_to_ty(&self,
                   span: Span,
                   opt_self_ty: Option<Ty<'tcx>>,
                   item_def_id: DefId,
                   trait_segment: &hir::PathSegment,
                   item_segment: &hir::PathSegment)
                   -> Ty<'tcx>
    {
        let tcx = self.tcx();
        let trait_def_id = tcx.parent_def_id(item_def_id).unwrap();

        self.prohibit_generics(slice::from_ref(item_segment));

        let self_ty = if let Some(ty) = opt_self_ty {
            ty
        } else {
            let path_str = tcx.item_path_str(trait_def_id);
            self.report_ambiguous_associated_type(span,
                                                  "Type",
                                                  &path_str,
                                                  &item_segment.ident.as_str());
            return tcx.types.err;
        };

        debug!("qpath_to_ty: self_type={:?}", self_ty);

        let trait_ref = self.ast_path_to_mono_trait_ref(span,
                                                        trait_def_id,
                                                        self_ty,
                                                        trait_segment);

        debug!("qpath_to_ty: trait_ref={:?}", trait_ref);

        self.normalize_ty(span, tcx.mk_projection(item_def_id, trait_ref.substs))
    }

    pub fn prohibit_generics<'a, T: IntoIterator<Item = &'a hir::PathSegment>>(&self, segments: T) {
        for segment in segments {
            segment.with_generic_args(|generic_args| {
                let (mut err_for_lt, mut err_for_ty) = (false, false);
                for arg in &generic_args.args {
                    let (mut span_err, span, kind) = match arg {
                        hir::GenericArg::Lifetime(lt) => {
                            if err_for_lt { continue }
                            err_for_lt = true;
                            (struct_span_err!(self.tcx().sess, lt.span, E0110,
                                            "lifetime parameters are not allowed on \
                                                this type"),
                             lt.span,
                             "lifetime")
                        }
                        hir::GenericArg::Type(ty) => {
                            if err_for_ty { continue }
                            err_for_ty = true;
                            (struct_span_err!(self.tcx().sess, ty.span, E0109,
                                            "type parameters are not allowed on this type"),
                             ty.span,
                             "type")
                        }
                    };
                    span_err.span_label(span, format!("{} parameter not allowed", kind))
                            .emit();
                    if err_for_lt && err_for_ty {
                        break;
                    }
                }
                for binding in &generic_args.bindings {
                    Self::prohibit_assoc_ty_binding(self.tcx(), binding.span);
                    break;
                }
            })
        }
    }

    pub fn prohibit_assoc_ty_binding(tcx: TyCtxt, span: Span) {
        let mut err = struct_span_err!(tcx.sess, span, E0229,
                                       "associated type bindings are not allowed here");
        err.span_label(span, "associated type not allowed here").emit();
    }

    // Check a type Path and convert it to a Ty.
    pub fn def_to_ty(&self,
                     opt_self_ty: Option<Ty<'tcx>>,
                     path: &hir::Path,
                     permit_variants: bool)
                     -> Ty<'tcx> {
        let tcx = self.tcx();

        debug!("base_def_to_ty(def={:?}, opt_self_ty={:?}, path_segments={:?})",
               path.def, opt_self_ty, path.segments);

        let span = path.span;
        match path.def {
            Def::Existential(did) => {
                // check for desugared impl trait
                if ty::is_impl_trait_defn(tcx, did).is_some() {
                    let lifetimes = &path.segments[0].args.as_ref().unwrap().args;
                    return self.impl_trait_ty_to_ty(did, lifetimes);
                }
                let item_segment = path.segments.split_last().unwrap();
                self.prohibit_generics(item_segment.1);
                let substs = self.ast_path_substs_for_ty(span, did, item_segment.0);
                self.normalize_ty(
                    span,
                    tcx.mk_anon(did, substs),
                )
            }
            Def::Enum(did) | Def::TyAlias(did) | Def::Struct(did) |
            Def::Union(did) | Def::TyForeign(did) => {
                assert_eq!(opt_self_ty, None);
                self.prohibit_generics(path.segments.split_last().unwrap().1);
                self.ast_path_to_ty(span, did, path.segments.last().unwrap())
            }
            Def::Variant(did) if permit_variants => {
                // Convert "variant type" as if it were a real type.
                // The resulting `Ty` is type of the variant's enum for now.
                assert_eq!(opt_self_ty, None);
                self.prohibit_generics(path.segments.split_last().unwrap().1);
                self.ast_path_to_ty(span,
                                    tcx.parent_def_id(did).unwrap(),
                                    path.segments.last().unwrap())
            }
            Def::TyParam(did) => {
                assert_eq!(opt_self_ty, None);
                self.prohibit_generics(&path.segments);

                let node_id = tcx.hir.as_local_node_id(did).unwrap();
                let item_id = tcx.hir.get_parent_node(node_id);
                let item_def_id = tcx.hir.local_def_id(item_id);
                let generics = tcx.generics_of(item_def_id);
                let index = generics.param_def_id_to_index[&tcx.hir.local_def_id(node_id)];
                tcx.mk_ty_param(index, tcx.hir.name(node_id).as_interned_str())
            }
            Def::SelfTy(_, Some(def_id)) => {
                // Self in impl (we know the concrete type).

                assert_eq!(opt_self_ty, None);
                self.prohibit_generics(&path.segments);

                tcx.at(span).type_of(def_id)
            }
            Def::SelfTy(Some(_), None) => {
                // Self in trait.
                assert_eq!(opt_self_ty, None);
                self.prohibit_generics(&path.segments);
                tcx.mk_self_type()
            }
            Def::AssociatedTy(def_id) => {
                self.prohibit_generics(&path.segments[..path.segments.len()-2]);
                self.qpath_to_ty(span,
                                 opt_self_ty,
                                 def_id,
                                 &path.segments[path.segments.len()-2],
                                 path.segments.last().unwrap())
            }
            Def::PrimTy(prim_ty) => {
                assert_eq!(opt_self_ty, None);
                self.prohibit_generics(&path.segments);
                match prim_ty {
                    hir::TyBool => tcx.types.bool,
                    hir::TyChar => tcx.types.char,
                    hir::TyInt(it) => tcx.mk_mach_int(it),
                    hir::TyUint(uit) => tcx.mk_mach_uint(uit),
                    hir::TyFloat(ft) => tcx.mk_mach_float(ft),
                    hir::TyStr => tcx.mk_str()
                }
            }
            Def::Err => {
                self.set_tainted_by_errors();
                return self.tcx().types.err;
            }
            _ => span_bug!(span, "unexpected definition: {:?}", path.def)
        }
    }

    /// Parses the programmer's textual representation of a type into our
    /// internal notion of a type.
    pub fn ast_ty_to_ty(&self, ast_ty: &hir::Ty) -> Ty<'tcx> {
        debug!("ast_ty_to_ty(id={:?}, ast_ty={:?})",
               ast_ty.id, ast_ty);

        let tcx = self.tcx();

        let result_ty = match ast_ty.node {
            hir::TyKind::Slice(ref ty) => {
                tcx.mk_slice(self.ast_ty_to_ty(&ty))
            }
            hir::TyKind::Ptr(ref mt) => {
                tcx.mk_ptr(ty::TypeAndMut {
                    ty: self.ast_ty_to_ty(&mt.ty),
                    mutbl: mt.mutbl
                })
            }
            hir::TyKind::Rptr(ref region, ref mt) => {
                let r = self.ast_region_to_region(region, None);
                debug!("TyRef r={:?}", r);
                let t = self.ast_ty_to_ty(&mt.ty);
                tcx.mk_ref(r, ty::TypeAndMut {ty: t, mutbl: mt.mutbl})
            }
            hir::TyKind::Never => {
                tcx.types.never
            },
            hir::TyKind::Tup(ref fields) => {
                tcx.mk_tup(fields.iter().map(|t| self.ast_ty_to_ty(&t)))
            }
            hir::TyKind::BareFn(ref bf) => {
                require_c_abi_if_variadic(tcx, &bf.decl, bf.abi, ast_ty.span);
                tcx.mk_fn_ptr(self.ty_of_fn(bf.unsafety, bf.abi, &bf.decl))
            }
            hir::TyKind::TraitObject(ref bounds, ref lifetime) => {
                self.conv_object_ty_poly_trait_ref(ast_ty.span, bounds, lifetime)
            }
            hir::TyKind::Path(hir::QPath::Resolved(ref maybe_qself, ref path)) => {
                debug!("ast_ty_to_ty: maybe_qself={:?} path={:?}", maybe_qself, path);
                let opt_self_ty = maybe_qself.as_ref().map(|qself| {
                    self.ast_ty_to_ty(qself)
                });
                self.def_to_ty(opt_self_ty, path, false)
            }
            hir::TyKind::Path(hir::QPath::TypeRelative(ref qself, ref segment)) => {
                debug!("ast_ty_to_ty: qself={:?} segment={:?}", qself, segment);
                let ty = self.ast_ty_to_ty(qself);

                let def = if let hir::TyKind::Path(hir::QPath::Resolved(_, ref path)) = qself.node {
                    path.def
                } else {
                    Def::Err
                };
                self.associated_path_def_to_ty(ast_ty.id, ast_ty.span, ty, def, segment).0
            }
            hir::TyKind::Array(ref ty, ref length) => {
                let length_def_id = tcx.hir.local_def_id(length.id);
                let substs = Substs::identity_for_item(tcx, length_def_id);
                let length = ty::Const::unevaluated(tcx, length_def_id, substs, tcx.types.usize);
                let array_ty = tcx.mk_ty(ty::TyArray(self.ast_ty_to_ty(&ty), length));
                self.normalize_ty(ast_ty.span, array_ty)
            }
            hir::TyKind::Typeof(ref _e) => {
                struct_span_err!(tcx.sess, ast_ty.span, E0516,
                                 "`typeof` is a reserved keyword but unimplemented")
                    .span_label(ast_ty.span, "reserved keyword")
                    .emit();

                tcx.types.err
            }
            hir::TyKind::Infer => {
                // TyInfer also appears as the type of arguments or return
                // values in a ExprKind::Closure, or as
                // the type of local variables. Both of these cases are
                // handled specially and will not descend into this routine.
                self.ty_infer(ast_ty.span)
            }
            hir::TyKind::Err => {
                tcx.types.err
            }
        };

        self.record_ty(ast_ty.hir_id, result_ty, ast_ty.span);
        result_ty
    }

    pub fn impl_trait_ty_to_ty(
        &self,
        def_id: DefId,
        lifetimes: &[hir::GenericArg],
    ) -> Ty<'tcx> {
        debug!("impl_trait_ty_to_ty(def_id={:?}, lifetimes={:?})", def_id, lifetimes);
        let tcx = self.tcx();

        let generics = tcx.generics_of(def_id);

        debug!("impl_trait_ty_to_ty: generics={:?}", generics);
        let substs = Substs::for_item(tcx, def_id, |param, _| {
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
                    _ => bug!()
                }
            } else {
                // Replace all parent lifetimes with 'static.
                match param.kind {
                    GenericParamDefKind::Lifetime => {
                        tcx.types.re_static.into()
                    }
                    _ => tcx.mk_param_from_def(param)
                }
            }
        });
        debug!("impl_trait_ty_to_ty: final substs = {:?}", substs);

        let ty = tcx.mk_anon(def_id, substs);
        debug!("impl_trait_ty_to_ty: {}", ty);
        ty
    }

    pub fn ty_of_arg(&self,
                     ty: &hir::Ty,
                     expected_ty: Option<Ty<'tcx>>)
                     -> Ty<'tcx>
    {
        match ty.node {
            hir::TyKind::Infer if expected_ty.is_some() => {
                self.record_ty(ty.hir_id, expected_ty.unwrap(), ty.span);
                expected_ty.unwrap()
            }
            _ => self.ast_ty_to_ty(ty),
        }
    }

    pub fn ty_of_fn(&self,
                    unsafety: hir::Unsafety,
                    abi: abi::Abi,
                    decl: &hir::FnDecl)
                    -> ty::PolyFnSig<'tcx> {
        debug!("ty_of_fn");

        let tcx = self.tcx();
        let input_tys: Vec<Ty> =
            decl.inputs.iter().map(|a| self.ty_of_arg(a, None)).collect();

        let output_ty = match decl.output {
            hir::Return(ref output) => self.ast_ty_to_ty(output),
            hir::DefaultReturn(..) => tcx.mk_nil(),
        };

        debug!("ty_of_fn: output_ty={:?}", output_ty);

        let bare_fn_ty = ty::Binder::bind(tcx.mk_fn_sig(
            input_tys.into_iter(),
            output_ty,
            decl.variadic,
            unsafety,
            abi
        ));

        // Find any late-bound regions declared in return type that do
        // not appear in the arguments. These are not wellformed.
        //
        // Example:
        //     for<'a> fn() -> &'a str <-- 'a is bad
        //     for<'a> fn(&'a String) -> &'a str <-- 'a is ok
        let inputs = bare_fn_ty.inputs();
        let late_bound_in_args = tcx.collect_constrained_late_bound_regions(
            &inputs.map_bound(|i| i.to_owned()));
        let output = bare_fn_ty.output();
        let late_bound_in_ret = tcx.collect_referenced_late_bound_regions(&output);
        for br in late_bound_in_ret.difference(&late_bound_in_args) {
            let lifetime_name = match *br {
                ty::BrNamed(_, name) => format!("lifetime `{}`,", name),
                ty::BrAnon(_) | ty::BrFresh(_) | ty::BrEnv => "an anonymous lifetime".to_string(),
            };
            let mut err = struct_span_err!(tcx.sess,
                                           decl.output.span(),
                                           E0581,
                                           "return type references {} \
                                            which is not constrained by the fn input types",
                                           lifetime_name);
            if let ty::BrAnon(_) = *br {
                // The only way for an anonymous lifetime to wind up
                // in the return type but **also** be unconstrained is
                // if it only appears in "associated types" in the
                // input. See #47511 for an example. In this case,
                // though we can easily give a hint that ought to be
                // relevant.
                err.note("lifetimes appearing in an associated type \
                          are not considered constrained");
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
    fn compute_object_lifetime_bound(&self,
        span: Span,
        existential_predicates: ty::Binder<&'tcx ty::Slice<ty::ExistentialPredicate<'tcx>>>)
        -> Option<ty::Region<'tcx>> // if None, use the default
    {
        let tcx = self.tcx();

        debug!("compute_opt_region_bound(existential_predicates={:?})",
               existential_predicates);

        // No explicit region bound specified. Therefore, examine trait
        // bounds and see if we can derive region bounds from those.
        let derived_region_bounds =
            object_region_bounds(tcx, existential_predicates);

        // If there are no derived region bounds, then report back that we
        // can find no region bound. The caller will use the default.
        if derived_region_bounds.is_empty() {
            return None;
        }

        // If any of the derived region bounds are 'static, that is always
        // the best choice.
        if derived_region_bounds.iter().any(|&r| ty::ReStatic == *r) {
            return Some(tcx.types.re_static);
        }

        // Determine whether there is exactly one unique region in the set
        // of derived region bounds. If so, use that. Otherwise, report an
        // error.
        let r = derived_region_bounds[0];
        if derived_region_bounds[1..].iter().any(|r1| r != *r1) {
            span_err!(tcx.sess, span, E0227,
                      "ambiguous lifetime bound, explicit lifetime bound required");
        }
        return Some(r);
    }
}

/// Divides a list of general trait bounds into two groups: auto traits (e.g. Sync and Send) and the
/// remaining general trait bounds.
fn split_auto_traits<'a, 'b, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                         trait_bounds: &'b [hir::PolyTraitRef])
    -> (Vec<DefId>, Vec<&'b hir::PolyTraitRef>)
{
    let (auto_traits, trait_bounds): (Vec<_>, _) = trait_bounds.iter().partition(|bound| {
        // Checks whether `trait_did` is an auto trait and adds it to `auto_traits` if so.
        match bound.trait_ref.path.def {
            Def::Trait(trait_did) if tcx.trait_is_auto(trait_did) => {
                true
            }
            _ => false
        }
    });

    let auto_traits = auto_traits.into_iter().map(|tr| {
        if let Def::Trait(trait_did) = tr.trait_ref.path.def {
            trait_did
        } else {
            unreachable!()
        }
    }).collect::<Vec<_>>();

    (auto_traits, trait_bounds)
}

// A helper struct for conveniently grouping a set of bounds which we pass to
// and return from functions in multiple places.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Bounds<'tcx> {
    pub region_bounds: Vec<ty::Region<'tcx>>,
    pub implicitly_sized: bool,
    pub trait_bounds: Vec<ty::PolyTraitRef<'tcx>>,
    pub projection_bounds: Vec<ty::PolyProjectionPredicate<'tcx>>,
}

impl<'a, 'gcx, 'tcx> Bounds<'tcx> {
    pub fn predicates(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, param_ty: Ty<'tcx>)
                      -> Vec<ty::Predicate<'tcx>>
    {
        // If it could be sized, and is, add the sized predicate
        let sized_predicate = if self.implicitly_sized {
            tcx.lang_items().sized_trait().map(|sized| {
                let trait_ref = ty::TraitRef {
                    def_id: sized,
                    substs: tcx.mk_substs_trait(param_ty, &[])
                };
                trait_ref.to_predicate()
            })
        } else {
            None
        };

        sized_predicate.into_iter().chain(
            self.region_bounds.iter().map(|&region_bound| {
                // account for the binder being introduced below; no need to shift `param_ty`
                // because, at present at least, it can only refer to early-bound regions
                let region_bound = tcx.mk_region(ty::fold::shift_region(*region_bound, 1));
                ty::Binder::dummy(ty::OutlivesPredicate(param_ty, region_bound)).to_predicate()
            }).chain(
                self.trait_bounds.iter().map(|bound_trait_ref| {
                    bound_trait_ref.to_predicate()
                })
            ).chain(
                self.projection_bounds.iter().map(|projection| {
                    projection.to_predicate()
                })
            )
        ).collect()
    }
}
