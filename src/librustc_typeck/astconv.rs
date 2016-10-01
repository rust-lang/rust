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
//! is parameterized by an instance of `AstConv` and a `RegionScope`.
//!
//! The parameterization of `ast_ty_to_ty()` is because it behaves
//! somewhat differently during the collect and check phases,
//! particularly with respect to looking up the types of top-level
//! items.  In the collect phase, the crate context is used as the
//! `AstConv` instance; in this phase, the `get_item_type_scheme()`
//! function triggers a recursive call to `type_scheme_of_item()`
//! (note that `ast_ty_to_ty()` will detect recursive types and report
//! an error).  In the check phase, when the FnCtxt is used as the
//! `AstConv`, `get_item_type_scheme()` just looks up the item type in
//! `tcx.tcache` (using `ty::lookup_item_type`).
//!
//! The `RegionScope` trait controls what happens when the user does
//! not specify a region in some location where a region is required
//! (e.g., if the user writes `&Foo` as a type rather than `&'a Foo`).
//! See the `rscope` module for more details.
//!
//! Unlike the `AstConv` trait, the region scope can change as we descend
//! the type.  This is to accommodate the fact that (a) fn types are binding
//! scopes and (b) the default region may change.  To understand case (a),
//! consider something like:
//!
//!   type foo = { x: &a.int, y: |&a.int| }
//!
//! The type of `x` is an error because there is no region `a` in scope.
//! In the type of `y`, however, region `a` is considered a bound region
//! as it does not already appear in scope.
//!
//! Case (b) says that if you have a type:
//!   type foo<'a> = ...;
//!   type bar = fn(&foo, &a.foo)
//! The fully expanded version of type bar is:
//!   type bar = fn(&'foo &, &a.foo<'a>)
//! Note that the self region for the `foo` defaulted to `&` in the first
//! case but `&a` in the second.  Basically, defaults that appear inside
//! an rptr (`&r.T`) use the region `r` that appears in the rptr.

use rustc_const_eval::eval_length;
use hir::{self, SelfKind};
use hir::def::{Def, PathResolution};
use hir::def_id::DefId;
use hir::print as pprust;
use middle::resolve_lifetime as rl;
use rustc::lint;
use rustc::ty::subst::{Subst, Substs};
use rustc::traits;
use rustc::ty::{self, Ty, TyCtxt, ToPredicate, TypeFoldable};
use rustc::ty::wf::object_region_bounds;
use rustc_back::slice;
use require_c_abi_if_variadic;
use rscope::{self, UnelidableRscope, RegionScope, ElidableRscope,
             ObjectLifetimeDefaultRscope, ShiftedRscope, BindingRscope,
             ElisionFailureInfo, ElidedLifetime};
use rscope::{AnonTypeScope, MaybeWithAnonTypes};
use util::common::{ErrorReported, FN_OUTPUT_NAME};
use util::nodemap::{NodeMap, FnvHashSet};

use std::cell::RefCell;
use syntax::{abi, ast};
use syntax::feature_gate::{GateIssue, emit_feature_err};
use syntax::parse::token::{self, keywords};
use syntax_pos::{Span, Pos};
use errors::DiagnosticBuilder;

pub trait AstConv<'gcx, 'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'a, 'gcx, 'tcx>;

    /// A cache used for the result of `ast_ty_to_ty_cache`
    fn ast_ty_to_ty_cache(&self) -> &RefCell<NodeMap<Ty<'tcx>>>;

    /// Returns the generic type and lifetime parameters for an item.
    fn get_generics(&self, span: Span, id: DefId)
                    -> Result<&'tcx ty::Generics<'tcx>, ErrorReported>;

    /// Identify the type scheme for an item with a type, like a type
    /// alias, fn, or struct. This allows you to figure out the set of
    /// type parameters defined on the item.
    fn get_item_type_scheme(&self, span: Span, id: DefId)
                            -> Result<ty::TypeScheme<'tcx>, ErrorReported>;

    /// Returns the `TraitDef` for a given trait. This allows you to
    /// figure out the set of type parameters defined on the trait.
    fn get_trait_def(&self, span: Span, id: DefId)
                     -> Result<&'tcx ty::TraitDef<'tcx>, ErrorReported>;

    /// Ensure that the super-predicates for the trait with the given
    /// id are available and also for the transitive set of
    /// super-predicates.
    fn ensure_super_predicates(&self, span: Span, id: DefId)
                               -> Result<(), ErrorReported>;

    /// Returns the set of bounds in scope for the type parameter with
    /// the given id.
    fn get_type_parameter_bounds(&self, span: Span, def_id: ast::NodeId)
                                 -> Result<Vec<ty::PolyTraitRef<'tcx>>, ErrorReported>;

    /// Returns true if the trait with id `trait_def_id` defines an
    /// associated type with the name `name`.
    fn trait_defines_associated_type_named(&self, trait_def_id: DefId, name: ast::Name)
                                           -> bool;

    /// Return an (optional) substitution to convert bound type parameters that
    /// are in scope into free ones. This function should only return Some
    /// within a fn body.
    /// See ParameterEnvironment::free_substs for more information.
    fn get_free_substs(&self) -> Option<&Substs<'tcx>>;

    /// What type should we use when a type is omitted?
    fn ty_infer(&self, span: Span) -> Ty<'tcx>;

    /// Same as ty_infer, but with a known type parameter definition.
    fn ty_infer_for_def(&self,
                        _def: &ty::TypeParameterDef<'tcx>,
                        _substs: &Substs<'tcx>,
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
                                        poly_trait_ref: ty::PolyTraitRef<'tcx>,
                                        item_name: ast::Name)
                                        -> Ty<'tcx>;

    /// Project an associated type from a non-higher-ranked trait reference.
    /// This is fairly straightforward and can be accommodated in any context.
    fn projected_ty(&self,
                    span: Span,
                    _trait_ref: ty::TraitRef<'tcx>,
                    _item_name: ast::Name)
                    -> Ty<'tcx>;

    /// Invoked when we encounter an error from some prior pass
    /// (e.g. resolve) that is translated into a ty-error. This is
    /// used to help suppress derived errors typeck might otherwise
    /// report.
    fn set_tainted_by_errors(&self);
}

#[derive(PartialEq, Eq)]
pub enum PathParamMode {
    // Any path in a type context.
    Explicit,
    // The `module::Type` in `module::Type::method` in an expression.
    Optional
}

struct ConvertedBinding<'tcx> {
    item_name: ast::Name,
    ty: Ty<'tcx>,
    span: Span,
}

type TraitAndProjections<'tcx> = (ty::PolyTraitRef<'tcx>, Vec<ty::PolyProjectionPredicate<'tcx>>);

/// Dummy type used for the `Self` of a `TraitRef` created for converting
/// a trait object, and which gets removed in `ExistentialTraitRef`.
/// This type must not appear anywhere in other converted types.
const TRAIT_OBJECT_DUMMY_SELF: ty::TypeVariants<'static> = ty::TyInfer(ty::FreshTy(0));

pub fn ast_region_to_region<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                            lifetime: &hir::Lifetime)
                                            -> &'tcx ty::Region {
    let r = match tcx.named_region_map.defs.get(&lifetime.id) {
        None => {
            // should have been recorded by the `resolve_lifetime` pass
            span_bug!(lifetime.span, "unresolved lifetime");
        }

        Some(&rl::DefStaticRegion) => {
            ty::ReStatic
        }

        Some(&rl::DefLateBoundRegion(debruijn, id)) => {
            // If this region is declared on a function, it will have
            // an entry in `late_bound`, but if it comes from
            // `for<'a>` in some type or something, it won't
            // necessarily have one. In that case though, we won't be
            // changed from late to early bound, so we can just
            // substitute false.
            let issue_32330 = tcx.named_region_map
                                 .late_bound
                                 .get(&id)
                                 .cloned()
                                 .unwrap_or(ty::Issue32330::WontChange);
            ty::ReLateBound(debruijn, ty::BrNamed(tcx.map.local_def_id(id),
                                                  lifetime.name,
                                                  issue_32330))
        }

        Some(&rl::DefEarlyBoundRegion(index, _)) => {
            ty::ReEarlyBound(ty::EarlyBoundRegion {
                index: index,
                name: lifetime.name
            })
        }

        Some(&rl::DefFreeRegion(scope, id)) => {
            // As in DefLateBoundRegion above, could be missing for some late-bound
            // regions, but also for early-bound regions.
            let issue_32330 = tcx.named_region_map
                                 .late_bound
                                 .get(&id)
                                 .cloned()
                                 .unwrap_or(ty::Issue32330::WontChange);
            ty::ReFree(ty::FreeRegion {
                    scope: scope.to_code_extent(&tcx.region_maps),
                    bound_region: ty::BrNamed(tcx.map.local_def_id(id),
                                              lifetime.name,
                                              issue_32330)
            })

                // (*) -- not late-bound, won't change
        }
    };

    debug!("ast_region_to_region(lifetime={:?} id={}) yields {:?}",
           lifetime,
           lifetime.id,
           r);

    tcx.mk_region(r)
}

fn report_elision_failure(
    db: &mut DiagnosticBuilder,
    params: Vec<ElisionFailureInfo>)
{
    let mut m = String::new();
    let len = params.len();

    let elided_params: Vec<_> = params.into_iter()
                                       .filter(|info| info.lifetime_count > 0)
                                       .collect();

    let elided_len = elided_params.len();

    for (i, info) in elided_params.into_iter().enumerate() {
        let ElisionFailureInfo {
            name, lifetime_count: n, have_bound_regions
        } = info;

        let help_name = if name.is_empty() {
            format!("argument {}", i + 1)
        } else {
            format!("`{}`", name)
        };

        m.push_str(&(if n == 1 {
            help_name
        } else {
            format!("one of {}'s {} elided {}lifetimes", help_name, n,
                    if have_bound_regions { "free " } else { "" } )
        })[..]);

        if elided_len == 2 && i == 0 {
            m.push_str(" or ");
        } else if i + 2 == elided_len {
            m.push_str(", or ");
        } else if i != elided_len - 1 {
            m.push_str(", ");
        }

    }

    if len == 0 {
        help!(db,
                   "this function's return type contains a borrowed value, but \
                    there is no value for it to be borrowed from");
        help!(db,
                   "consider giving it a 'static lifetime");
    } else if elided_len == 0 {
        help!(db,
                   "this function's return type contains a borrowed value with \
                    an elided lifetime, but the lifetime cannot be derived from \
                    the arguments");
        help!(db,
                   "consider giving it an explicit bounded or 'static \
                    lifetime");
    } else if elided_len == 1 {
        help!(db,
                   "this function's return type contains a borrowed value, but \
                    the signature does not say which {} it is borrowed from",
                   m);
    } else {
        help!(db,
                   "this function's return type contains a borrowed value, but \
                    the signature does not say whether it is borrowed from {}",
                   m);
    }
}

impl<'o, 'gcx: 'tcx, 'tcx> AstConv<'gcx, 'tcx>+'o {
    pub fn opt_ast_region_to_region(&self,
        rscope: &RegionScope,
        default_span: Span,
        opt_lifetime: &Option<hir::Lifetime>) -> &'tcx ty::Region
    {
        let r = match *opt_lifetime {
            Some(ref lifetime) => {
                ast_region_to_region(self.tcx(), lifetime)
            }

            None => self.tcx().mk_region(match rscope.anon_regions(default_span, 1) {
                Ok(rs) => rs[0],
                Err(params) => {
                    let ampersand_span = Span { hi: default_span.lo, ..default_span};

                    let mut err = struct_span_err!(self.tcx().sess, ampersand_span, E0106,
                                                 "missing lifetime specifier");
                    err.span_label(ampersand_span, &format!("expected lifetime parameter"));

                    if let Some(params) = params {
                        report_elision_failure(&mut err, params);
                    }
                    err.emit();
                    ty::ReStatic
                }
            })
        };

        debug!("opt_ast_region_to_region(opt_lifetime={:?}) yields {:?}",
                opt_lifetime,
                r);

        r
    }

    /// Given a path `path` that refers to an item `I` with the declared generics `decl_generics`,
    /// returns an appropriate set of substitutions for this particular reference to `I`.
    pub fn ast_path_substs_for_ty(&self,
        rscope: &RegionScope,
        span: Span,
        param_mode: PathParamMode,
        def_id: DefId,
        item_segment: &hir::PathSegment)
        -> &'tcx Substs<'tcx>
    {
        let tcx = self.tcx();

        match item_segment.parameters {
            hir::AngleBracketedParameters(_) => {}
            hir::ParenthesizedParameters(..) => {
                struct_span_err!(tcx.sess, span, E0214,
                          "parenthesized parameters may only be used with a trait")
                    .span_label(span, &format!("only traits may use parentheses"))
                    .emit();

                return Substs::for_item(tcx, def_id, |_, _| {
                    tcx.mk_region(ty::ReStatic)
                }, |_, _| {
                    tcx.types.err
                });
            }
        }

        let (substs, assoc_bindings) =
            self.create_substs_for_ast_path(rscope,
                                            span,
                                            param_mode,
                                            def_id,
                                            &item_segment.parameters,
                                            None);

        assoc_bindings.first().map(|b| self.tcx().prohibit_projection(b.span));

        substs
    }

    /// Given the type/region arguments provided to some path (along with
    /// an implicit Self, if this is a trait reference) returns the complete
    /// set of substitutions. This may involve applying defaulted type parameters.
    ///
    /// Note that the type listing given here is *exactly* what the user provided.
    fn create_substs_for_ast_path(&self,
        rscope: &RegionScope,
        span: Span,
        param_mode: PathParamMode,
        def_id: DefId,
        parameters: &hir::PathParameters,
        self_ty: Option<Ty<'tcx>>)
        -> (&'tcx Substs<'tcx>, Vec<ConvertedBinding<'tcx>>)
    {
        let tcx = self.tcx();

        debug!("create_substs_for_ast_path(def_id={:?}, self_ty={:?}, \
               parameters={:?})",
               def_id, self_ty, parameters);

        let (lifetimes, num_types_provided) = match *parameters {
            hir::AngleBracketedParameters(ref data) => {
                if param_mode == PathParamMode::Optional && data.types.is_empty() {
                    (&data.lifetimes[..], None)
                } else {
                    (&data.lifetimes[..], Some(data.types.len()))
                }
            }
            hir::ParenthesizedParameters(_) => (&[][..], Some(1))
        };

        // If the type is parameterized by this region, then replace this
        // region with the current anon region binding (in other words,
        // whatever & would get replaced with).
        let decl_generics = match self.get_generics(span, def_id) {
            Ok(generics) => generics,
            Err(ErrorReported) => {
                // No convenient way to recover from a cycle here. Just bail. Sorry!
                self.tcx().sess.abort_if_errors();
                bug!("ErrorReported returned, but no errors reports?")
            }
        };
        let expected_num_region_params = decl_generics.regions.len();
        let supplied_num_region_params = lifetimes.len();
        let regions = if expected_num_region_params == supplied_num_region_params {
            lifetimes.iter().map(|l| *ast_region_to_region(tcx, l)).collect()
        } else {
            let anon_regions =
                rscope.anon_regions(span, expected_num_region_params);

            if supplied_num_region_params != 0 || anon_regions.is_err() {
                report_lifetime_number_error(tcx, span,
                                             supplied_num_region_params,
                                             expected_num_region_params);
            }

            match anon_regions {
                Ok(anon_regions) => anon_regions,
                Err(_) => (0..expected_num_region_params).map(|_| ty::ReStatic).collect()
            }
        };

        // If a self-type was declared, one should be provided.
        assert_eq!(decl_generics.has_self, self_ty.is_some());

        // Check the number of type parameters supplied by the user.
        if let Some(num_provided) = num_types_provided {
            let ty_param_defs = &decl_generics.types[self_ty.is_some() as usize..];
            check_type_argument_count(tcx, span, num_provided, ty_param_defs);
        }

        let is_object = self_ty.map_or(false, |ty| ty.sty == TRAIT_OBJECT_DUMMY_SELF);
        let default_needs_object_self = |p: &ty::TypeParameterDef<'tcx>| {
            if let Some(ref default) = p.default {
                if is_object && default.has_self_ty() {
                    // There is no suitable inference default for a type parameter
                    // that references self, in an object type.
                    return true;
                }
            }

            false
        };

        let mut output_assoc_binding = None;
        let substs = Substs::for_item(tcx, def_id, |def, _| {
            let i = def.index as usize - self_ty.is_some() as usize;
            tcx.mk_region(regions[i])
        }, |def, substs| {
            let i = def.index as usize;

            // Handle Self first, so we can adjust the index to match the AST.
            if let (0, Some(ty)) = (i, self_ty) {
                return ty;
            }

            let i = i - self_ty.is_some() as usize - decl_generics.regions.len();
            if num_types_provided.map_or(false, |n| i < n) {
                // A provided type parameter.
                match *parameters {
                    hir::AngleBracketedParameters(ref data) => {
                        self.ast_ty_arg_to_ty(rscope, Some(def), substs, &data.types[i])
                    }
                    hir::ParenthesizedParameters(ref data) => {
                        assert_eq!(i, 0);
                        let (ty, assoc) =
                            self.convert_parenthesized_parameters(rscope, substs, data);
                        output_assoc_binding = Some(assoc);
                        ty
                    }
                }
            } else if num_types_provided.is_none() {
                // No type parameters were provided, we can infer all.
                let ty_var = if !default_needs_object_self(def) {
                    self.ty_infer_for_def(def, substs, span)
                } else {
                    self.ty_infer(span)
                };
                ty_var
            } else if let Some(default) = def.default {
                // No type parameter provided, but a default exists.

                // If we are converting an object type, then the
                // `Self` parameter is unknown. However, some of the
                // other type parameters may reference `Self` in their
                // defaults. This will lead to an ICE if we are not
                // careful!
                if default_needs_object_self(def) {
                    struct_span_err!(tcx.sess, span, E0393,
                                     "the type parameter `{}` must be explicitly specified",
                                     def.name)
                        .span_label(span, &format!("missing reference to `{}`", def.name))
                        .note(&format!("because of the default `Self` reference, \
                                        type parameters must be specified on object types"))
                        .emit();
                    tcx.types.err
                } else {
                    // This is a default type parameter.
                    default.subst_spanned(tcx, substs, Some(span))
                }
            } else {
                // We've already errored above about the mismatch.
                tcx.types.err
            }
        });

        let assoc_bindings = match *parameters {
            hir::AngleBracketedParameters(ref data) => {
                data.bindings.iter().map(|b| {
                    ConvertedBinding {
                        item_name: b.name,
                        ty: self.ast_ty_to_ty(rscope, &b.ty),
                        span: b.span
                    }
                }).collect()
            }
            hir::ParenthesizedParameters(ref data) => {
                vec![output_assoc_binding.unwrap_or_else(|| {
                    // This is an error condition, but we should
                    // get the associated type binding anyway.
                    self.convert_parenthesized_parameters(rscope, substs, data).1
                })]
            }
        };

        debug!("create_substs_for_ast_path(decl_generics={:?}, self_ty={:?}) -> {:?}",
               decl_generics, self_ty, substs);

        (substs, assoc_bindings)
    }

    /// Returns the appropriate lifetime to use for any output lifetimes
    /// (if one exists) and a vector of the (pattern, number of lifetimes)
    /// corresponding to each input type/pattern.
    fn find_implied_output_region(&self,
                                  input_tys: &[Ty<'tcx>],
                                  input_pats: Vec<String>) -> ElidedLifetime
    {
        let tcx = self.tcx();
        let mut lifetimes_for_params = Vec::new();
        let mut possible_implied_output_region = None;

        for (input_type, input_pat) in input_tys.iter().zip(input_pats) {
            let mut regions = FnvHashSet();
            let have_bound_regions = tcx.collect_regions(input_type, &mut regions);

            debug!("find_implied_output_regions: collected {:?} from {:?} \
                    have_bound_regions={:?}", &regions, input_type, have_bound_regions);

            if regions.len() == 1 {
                // there's a chance that the unique lifetime of this
                // iteration will be the appropriate lifetime for output
                // parameters, so lets store it.
                possible_implied_output_region = regions.iter().cloned().next();
            }

            lifetimes_for_params.push(ElisionFailureInfo {
                name: input_pat,
                lifetime_count: regions.len(),
                have_bound_regions: have_bound_regions
            });
        }

        if lifetimes_for_params.iter().map(|e| e.lifetime_count).sum::<usize>() == 1 {
            Ok(*possible_implied_output_region.unwrap())
        } else {
            Err(Some(lifetimes_for_params))
        }
    }

    fn convert_ty_with_lifetime_elision(&self,
                                        elided_lifetime: ElidedLifetime,
                                        ty: &hir::Ty,
                                        anon_scope: Option<AnonTypeScope>)
                                        -> Ty<'tcx>
    {
        match elided_lifetime {
            Ok(implied_output_region) => {
                let rb = ElidableRscope::new(implied_output_region);
                self.ast_ty_to_ty(&MaybeWithAnonTypes::new(rb, anon_scope), ty)
            }
            Err(param_lifetimes) => {
                // All regions must be explicitly specified in the output
                // if the lifetime elision rules do not apply. This saves
                // the user from potentially-confusing errors.
                let rb = UnelidableRscope::new(param_lifetimes);
                self.ast_ty_to_ty(&MaybeWithAnonTypes::new(rb, anon_scope), ty)
            }
        }
    }

    fn convert_parenthesized_parameters(&self,
                                        rscope: &RegionScope,
                                        region_substs: &Substs<'tcx>,
                                        data: &hir::ParenthesizedParameterData)
                                        -> (Ty<'tcx>, ConvertedBinding<'tcx>)
    {
        let anon_scope = rscope.anon_type_scope();
        let binding_rscope = MaybeWithAnonTypes::new(BindingRscope::new(), anon_scope);
        let inputs: Vec<_> = data.inputs.iter().map(|a_t| {
            self.ast_ty_arg_to_ty(&binding_rscope, None, region_substs, a_t)
        }).collect();
        let input_params = vec![String::new(); inputs.len()];
        let implied_output_region = self.find_implied_output_region(&inputs, input_params);

        let (output, output_span) = match data.output {
            Some(ref output_ty) => {
                (self.convert_ty_with_lifetime_elision(implied_output_region,
                                                       &output_ty,
                                                       anon_scope),
                 output_ty.span)
            }
            None => {
                (self.tcx().mk_nil(), data.span)
            }
        };

        let output_binding = ConvertedBinding {
            item_name: token::intern(FN_OUTPUT_NAME),
            ty: output,
            span: output_span
        };

        (self.tcx().mk_tup(inputs), output_binding)
    }

    pub fn instantiate_poly_trait_ref(&self,
        rscope: &RegionScope,
        ast_trait_ref: &hir::PolyTraitRef,
        self_ty: Ty<'tcx>,
        poly_projections: &mut Vec<ty::PolyProjectionPredicate<'tcx>>)
        -> ty::PolyTraitRef<'tcx>
    {
        let trait_ref = &ast_trait_ref.trait_ref;
        let trait_def_id = self.trait_def_id(trait_ref);
        self.ast_path_to_poly_trait_ref(rscope,
                                        trait_ref.path.span,
                                        PathParamMode::Explicit,
                                        trait_def_id,
                                        self_ty,
                                        trait_ref.ref_id,
                                        trait_ref.path.segments.last().unwrap(),
                                        poly_projections)
    }

    /// Instantiates the path for the given trait reference, assuming that it's
    /// bound to a valid trait type. Returns the def_id for the defining trait.
    /// Fails if the type is a type other than a trait type.
    ///
    /// If the `projections` argument is `None`, then assoc type bindings like `Foo<T=X>`
    /// are disallowed. Otherwise, they are pushed onto the vector given.
    pub fn instantiate_mono_trait_ref(&self,
        rscope: &RegionScope,
        trait_ref: &hir::TraitRef,
        self_ty: Ty<'tcx>)
        -> ty::TraitRef<'tcx>
    {
        let trait_def_id = self.trait_def_id(trait_ref);
        self.ast_path_to_mono_trait_ref(rscope,
                                        trait_ref.path.span,
                                        PathParamMode::Explicit,
                                        trait_def_id,
                                        self_ty,
                                        trait_ref.path.segments.last().unwrap())
    }

    fn trait_def_id(&self, trait_ref: &hir::TraitRef) -> DefId {
        let path = &trait_ref.path;
        match self.tcx().expect_def(trait_ref.ref_id) {
            Def::Trait(trait_def_id) => trait_def_id,
            Def::Err => {
                self.tcx().sess.fatal("cannot continue compilation due to previous error");
            }
            _ => {
                span_fatal!(self.tcx().sess, path.span, E0245, "`{}` is not a trait",
                            path);
            }
        }
    }

    fn ast_path_to_poly_trait_ref(&self,
        rscope: &RegionScope,
        span: Span,
        param_mode: PathParamMode,
        trait_def_id: DefId,
        self_ty: Ty<'tcx>,
        path_id: ast::NodeId,
        trait_segment: &hir::PathSegment,
        poly_projections: &mut Vec<ty::PolyProjectionPredicate<'tcx>>)
        -> ty::PolyTraitRef<'tcx>
    {
        debug!("ast_path_to_poly_trait_ref(trait_segment={:?})", trait_segment);
        // The trait reference introduces a binding level here, so
        // we need to shift the `rscope`. It'd be nice if we could
        // do away with this rscope stuff and work this knowledge
        // into resolve_lifetimes, as we do with non-omitted
        // lifetimes. Oh well, not there yet.
        let shifted_rscope = &ShiftedRscope::new(rscope);

        let (substs, assoc_bindings) =
            self.create_substs_for_ast_trait_ref(shifted_rscope,
                                                 span,
                                                 param_mode,
                                                 trait_def_id,
                                                 self_ty,
                                                 trait_segment);
        let poly_trait_ref = ty::Binder(ty::TraitRef::new(trait_def_id, substs));

        poly_projections.extend(assoc_bindings.iter().filter_map(|binding| {
            // specify type to assert that error was already reported in Err case:
            let predicate: Result<_, ErrorReported> =
                self.ast_type_binding_to_poly_projection_predicate(path_id,
                                                                   poly_trait_ref,
                                                                   binding);
            predicate.ok() // ok to ignore Err() because ErrorReported (see above)
        }));

        debug!("ast_path_to_poly_trait_ref(trait_segment={:?}, projections={:?}) -> {:?}",
               trait_segment, poly_projections, poly_trait_ref);
        poly_trait_ref
    }

    fn ast_path_to_mono_trait_ref(&self,
                                  rscope: &RegionScope,
                                  span: Span,
                                  param_mode: PathParamMode,
                                  trait_def_id: DefId,
                                  self_ty: Ty<'tcx>,
                                  trait_segment: &hir::PathSegment)
                                  -> ty::TraitRef<'tcx>
    {
        let (substs, assoc_bindings) =
            self.create_substs_for_ast_trait_ref(rscope,
                                                 span,
                                                 param_mode,
                                                 trait_def_id,
                                                 self_ty,
                                                 trait_segment);
        assoc_bindings.first().map(|b| self.tcx().prohibit_projection(b.span));
        ty::TraitRef::new(trait_def_id, substs)
    }

    fn create_substs_for_ast_trait_ref(&self,
                                       rscope: &RegionScope,
                                       span: Span,
                                       param_mode: PathParamMode,
                                       trait_def_id: DefId,
                                       self_ty: Ty<'tcx>,
                                       trait_segment: &hir::PathSegment)
                                       -> (&'tcx Substs<'tcx>, Vec<ConvertedBinding<'tcx>>)
    {
        debug!("create_substs_for_ast_trait_ref(trait_segment={:?})",
               trait_segment);

        let trait_def = match self.get_trait_def(span, trait_def_id) {
            Ok(trait_def) => trait_def,
            Err(ErrorReported) => {
                // No convenient way to recover from a cycle here. Just bail. Sorry!
                self.tcx().sess.abort_if_errors();
                bug!("ErrorReported returned, but no errors reports?")
            }
        };

        match trait_segment.parameters {
            hir::AngleBracketedParameters(_) => {
                // For now, require that parenthetical notation be used
                // only with `Fn()` etc.
                if !self.tcx().sess.features.borrow().unboxed_closures && trait_def.paren_sugar {
                    emit_feature_err(&self.tcx().sess.parse_sess,
                                     "unboxed_closures", span, GateIssue::Language,
                                     "\
                        the precise format of `Fn`-family traits' \
                        type parameters is subject to change. \
                        Use parenthetical notation (Fn(Foo, Bar) -> Baz) instead");
                }
            }
            hir::ParenthesizedParameters(_) => {
                // For now, require that parenthetical notation be used
                // only with `Fn()` etc.
                if !self.tcx().sess.features.borrow().unboxed_closures && !trait_def.paren_sugar {
                    emit_feature_err(&self.tcx().sess.parse_sess,
                                     "unboxed_closures", span, GateIssue::Language,
                                     "\
                        parenthetical notation is only stable when used with `Fn`-family traits");
                }
            }
        }

        self.create_substs_for_ast_path(rscope,
                                        span,
                                        param_mode,
                                        trait_def_id,
                                        &trait_segment.parameters,
                                        Some(self_ty))
    }

    fn ast_type_binding_to_poly_projection_predicate(
        &self,
        path_id: ast::NodeId,
        trait_ref: ty::PolyTraitRef<'tcx>,
        binding: &ConvertedBinding<'tcx>)
        -> Result<ty::PolyProjectionPredicate<'tcx>, ErrorReported>
    {
        let tcx = self.tcx();

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
        let late_bound_in_ty = tcx.collect_referenced_late_bound_regions(&ty::Binder(binding.ty));
        debug!("late_bound_in_trait_ref = {:?}", late_bound_in_trait_ref);
        debug!("late_bound_in_ty = {:?}", late_bound_in_ty);
        for br in late_bound_in_ty.difference(&late_bound_in_trait_ref) {
            let br_name = match *br {
                ty::BrNamed(_, name, _) => name,
                _ => {
                    span_bug!(
                        binding.span,
                        "anonymous bound region {:?} in binding but not trait ref",
                        br);
                }
            };
            tcx.sess.add_lint(
                lint::builtin::HR_LIFETIME_IN_ASSOC_TYPE,
                path_id,
                binding.span,
                format!("binding for associated type `{}` references lifetime `{}`, \
                         which does not appear in the trait input types",
                        binding.item_name, br_name));
        }

        // Simple case: X is defined in the current trait.
        if self.trait_defines_associated_type_named(trait_ref.def_id(), binding.item_name) {
            return Ok(trait_ref.map_bound(|trait_ref| {
                ty::ProjectionPredicate {
                    projection_ty: ty::ProjectionTy {
                        trait_ref: trait_ref,
                        item_name: binding.item_name,
                    },
                    ty: binding.ty,
                }
            }));
        }

        // Otherwise, we have to walk through the supertraits to find
        // those that do.
        self.ensure_super_predicates(binding.span, trait_ref.def_id())?;

        let candidates: Vec<ty::PolyTraitRef> =
            traits::supertraits(tcx, trait_ref.clone())
            .filter(|r| self.trait_defines_associated_type_named(r.def_id(), binding.item_name))
            .collect();

        let candidate = self.one_bound_for_assoc_type(candidates,
                                                      &trait_ref.to_string(),
                                                      &binding.item_name.as_str(),
                                                      binding.span)?;

        Ok(candidate.map_bound(|trait_ref| {
            ty::ProjectionPredicate {
                projection_ty: ty::ProjectionTy {
                    trait_ref: trait_ref,
                    item_name: binding.item_name,
                },
                ty: binding.ty,
            }
        }))
    }

    fn ast_path_to_ty(&self,
        rscope: &RegionScope,
        span: Span,
        param_mode: PathParamMode,
        did: DefId,
        item_segment: &hir::PathSegment)
        -> Ty<'tcx>
    {
        let tcx = self.tcx();
        let decl_ty = match self.get_item_type_scheme(span, did) {
            Ok(type_scheme) => type_scheme.ty,
            Err(ErrorReported) => {
                return tcx.types.err;
            }
        };

        let substs = self.ast_path_substs_for_ty(rscope,
                                                 span,
                                                 param_mode,
                                                 did,
                                                 item_segment);

        // FIXME(#12938): This is a hack until we have full support for DST.
        if Some(did) == self.tcx().lang_items.owned_box() {
            assert_eq!(substs.types().count(), 1);
            return self.tcx().mk_box(substs.type_at(0));
        }

        decl_ty.subst(self.tcx(), substs)
    }

    fn ast_ty_to_object_trait_ref(&self,
                                  rscope: &RegionScope,
                                  span: Span,
                                  ty: &hir::Ty,
                                  bounds: &[hir::TyParamBound])
                                  -> Ty<'tcx>
    {
        /*!
         * In a type like `Foo + Send`, we want to wait to collect the
         * full set of bounds before we make the object type, because we
         * need them to infer a region bound.  (For example, if we tried
         * made a type from just `Foo`, then it wouldn't be enough to
         * infer a 'static bound, and hence the user would get an error.)
         * So this function is used when we're dealing with a sum type to
         * convert the LHS. It only accepts a type that refers to a trait
         * name, and reports an error otherwise.
         */

        let tcx = self.tcx();
        match ty.node {
            hir::TyPath(None, ref path) => {
                let resolution = tcx.expect_resolution(ty.id);
                match resolution.base_def {
                    Def::Trait(trait_def_id) if resolution.depth == 0 => {
                        self.trait_path_to_object_type(rscope,
                                                       path.span,
                                                       PathParamMode::Explicit,
                                                       trait_def_id,
                                                       ty.id,
                                                       path.segments.last().unwrap(),
                                                       span,
                                                       partition_bounds(tcx, span, bounds))
                    }
                    _ => {
                        struct_span_err!(tcx.sess, ty.span, E0172,
                                  "expected a reference to a trait")
                            .span_label(ty.span, &format!("expected a trait"))
                            .emit();
                        tcx.types.err
                    }
                }
            }
            _ => {
                let mut err = struct_span_err!(tcx.sess, ty.span, E0178,
                                               "expected a path on the left-hand side \
                                                of `+`, not `{}`",
                                               pprust::ty_to_string(ty));
                err.span_label(ty.span, &format!("expected a path"));
                let hi = bounds.iter().map(|x| match *x {
                    hir::TraitTyParamBound(ref tr, _) => tr.span.hi,
                    hir::RegionTyParamBound(ref r) => r.span.hi,
                }).max_by_key(|x| x.to_usize());
                let full_span = hi.map(|hi| Span {
                    lo: ty.span.lo,
                    hi: hi,
                    expn_id: ty.span.expn_id,
                });
                match (&ty.node, full_span) {
                    (&hir::TyRptr(None, ref mut_ty), Some(full_span)) => {
                        let mutbl_str = if mut_ty.mutbl == hir::MutMutable { "mut " } else { "" };
                        err.span_suggestion(full_span, "try adding parentheses (per RFC 438):",
                                            format!("&{}({} +{})",
                                                    mutbl_str,
                                                    pprust::ty_to_string(&mut_ty.ty),
                                                    pprust::bounds_to_string(bounds)));
                    }
                    (&hir::TyRptr(Some(ref lt), ref mut_ty), Some(full_span)) => {
                        let mutbl_str = if mut_ty.mutbl == hir::MutMutable { "mut " } else { "" };
                        err.span_suggestion(full_span, "try adding parentheses (per RFC 438):",
                                            format!("&{} {}({} +{})",
                                                    pprust::lifetime_to_string(lt),
                                                    mutbl_str,
                                                    pprust::ty_to_string(&mut_ty.ty),
                                                    pprust::bounds_to_string(bounds)));
                    }

                    _ => {
                        help!(&mut err,
                                   "perhaps you forgot parentheses? (per RFC 438)");
                    }
                }
                err.emit();
                tcx.types.err
            }
        }
    }

    /// Transform a PolyTraitRef into a PolyExistentialTraitRef by
    /// removing the dummy Self type (TRAIT_OBJECT_DUMMY_SELF).
    fn trait_ref_to_existential(&self, trait_ref: ty::TraitRef<'tcx>)
                                -> ty::ExistentialTraitRef<'tcx> {
        assert_eq!(trait_ref.self_ty().sty, TRAIT_OBJECT_DUMMY_SELF);
        ty::ExistentialTraitRef::erase_self_ty(self.tcx(), trait_ref)
    }

    fn trait_path_to_object_type(&self,
                                 rscope: &RegionScope,
                                 path_span: Span,
                                 param_mode: PathParamMode,
                                 trait_def_id: DefId,
                                 trait_path_ref_id: ast::NodeId,
                                 trait_segment: &hir::PathSegment,
                                 span: Span,
                                 partitioned_bounds: PartitionedBounds)
                                 -> Ty<'tcx> {
        let tcx = self.tcx();

        let mut projection_bounds = vec![];
        let dummy_self = tcx.mk_ty(TRAIT_OBJECT_DUMMY_SELF);
        let principal = self.ast_path_to_poly_trait_ref(rscope,
                                                        path_span,
                                                        param_mode,
                                                        trait_def_id,
                                                        dummy_self,
                                                        trait_path_ref_id,
                                                        trait_segment,
                                                        &mut projection_bounds);

        let PartitionedBounds { builtin_bounds,
                                trait_bounds,
                                region_bounds } =
            partitioned_bounds;

        if !trait_bounds.is_empty() {
            let b = &trait_bounds[0];
            let span = b.trait_ref.path.span;
            struct_span_err!(self.tcx().sess, span, E0225,
                             "only the builtin traits can be used as closure or object bounds")
                .span_label(span, &format!("non-builtin trait used as bounds"))
                .emit();
        }

        // Erase the dummy_self (TRAIT_OBJECT_DUMMY_SELF) used above.
        let existential_principal = principal.map_bound(|trait_ref| {
            self.trait_ref_to_existential(trait_ref)
        });
        let existential_projections = projection_bounds.iter().map(|bound| {
            bound.map_bound(|b| {
                let p = b.projection_ty;
                ty::ExistentialProjection {
                    trait_ref: self.trait_ref_to_existential(p.trait_ref),
                    item_name: p.item_name,
                    ty: b.ty
                }
            })
        }).collect();

        let region_bound =
            self.compute_object_lifetime_bound(span,
                                               &region_bounds,
                                               existential_principal,
                                               builtin_bounds);

        let region_bound = match region_bound {
            Some(r) => r,
            None => {
                tcx.mk_region(match rscope.object_lifetime_default(span) {
                    Some(r) => r,
                    None => {
                        span_err!(self.tcx().sess, span, E0228,
                                  "the lifetime bound for this object type cannot be deduced \
                                   from context; please supply an explicit bound");
                        ty::ReStatic
                    }
                })
            }
        };

        debug!("region_bound: {:?}", region_bound);

        // ensure the super predicates and stop if we encountered an error
        if self.ensure_super_predicates(span, principal.def_id()).is_err() {
            return tcx.types.err;
        }

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

        let mut associated_types = FnvHashSet::default();
        for tr in traits::supertraits(tcx, principal) {
            if let Some(trait_id) = tcx.map.as_local_node_id(tr.def_id()) {
                use collect::trait_associated_type_names;

                associated_types.extend(trait_associated_type_names(tcx, trait_id)
                    .map(|name| (tr.def_id(), name)))
            } else {
                let trait_items = tcx.impl_or_trait_items(tr.def_id());
                associated_types.extend(trait_items.iter().filter_map(|&def_id| {
                    match tcx.impl_or_trait_item(def_id) {
                        ty::TypeTraitItem(ref item) => Some(item.name),
                        _ => None
                    }
                }).map(|name| (tr.def_id(), name)));
            }
        }

        for projection_bound in &projection_bounds {
            let pair = (projection_bound.0.projection_ty.trait_ref.def_id,
                        projection_bound.0.projection_ty.item_name);
            associated_types.remove(&pair);
        }

        for (trait_def_id, name) in associated_types {
            struct_span_err!(tcx.sess, span, E0191,
                "the value of the associated type `{}` (from the trait `{}`) must be specified",
                        name,
                        tcx.item_path_str(trait_def_id))
                        .span_label(span, &format!(
                            "missing associated type `{}` value", name))
                        .emit();
        }

        let ty = tcx.mk_trait(ty::TraitObject {
            principal: existential_principal,
            region_bound: region_bound,
            builtin_bounds: builtin_bounds,
            projection_bounds: existential_projections
        });
        debug!("trait_object_type: {:?}", ty);
        ty
    }

    fn report_ambiguous_associated_type(&self,
                                        span: Span,
                                        type_str: &str,
                                        trait_str: &str,
                                        name: &str) {
        struct_span_err!(self.tcx().sess, span, E0223, "ambiguous associated type")
            .span_label(span, &format!("ambiguous associated type"))
            .note(&format!("specify the type using the syntax `<{} as {}>::{}`",
                  type_str, trait_str, name))
            .emit();

    }

    // Search for a bound on a type parameter which includes the associated item
    // given by assoc_name. ty_param_node_id is the node id for the type parameter
    // (which might be `Self`, but only if it is the `Self` of a trait, not an
    // impl). This function will fail if there are no suitable bounds or there is
    // any ambiguity.
    fn find_bound_for_assoc_item(&self,
                                 ty_param_node_id: ast::NodeId,
                                 ty_param_name: ast::Name,
                                 assoc_name: ast::Name,
                                 span: Span)
                                 -> Result<ty::PolyTraitRef<'tcx>, ErrorReported>
    {
        let tcx = self.tcx();

        let bounds = match self.get_type_parameter_bounds(span, ty_param_node_id) {
            Ok(v) => v,
            Err(ErrorReported) => {
                return Err(ErrorReported);
            }
        };

        // Ensure the super predicates and stop if we encountered an error.
        if bounds.iter().any(|b| self.ensure_super_predicates(span, b.def_id()).is_err()) {
            return Err(ErrorReported);
        }

        // Check that there is exactly one way to find an associated type with the
        // correct name.
        let suitable_bounds: Vec<_> =
            traits::transitive_bounds(tcx, &bounds)
            .filter(|b| self.trait_defines_associated_type_named(b.def_id(), assoc_name))
            .collect();

        self.one_bound_for_assoc_type(suitable_bounds,
                                      &ty_param_name.as_str(),
                                      &assoc_name.as_str(),
                                      span)
    }


    // Checks that bounds contains exactly one element and reports appropriate
    // errors otherwise.
    fn one_bound_for_assoc_type(&self,
                                bounds: Vec<ty::PolyTraitRef<'tcx>>,
                                ty_param_name: &str,
                                assoc_name: &str,
                                span: Span)
        -> Result<ty::PolyTraitRef<'tcx>, ErrorReported>
    {
        if bounds.is_empty() {
            span_err!(self.tcx().sess, span, E0220,
                      "associated type `{}` not found for `{}`",
                      assoc_name,
                      ty_param_name);
            return Err(ErrorReported);
        }

        if bounds.len() > 1 {
            let mut err = struct_span_err!(
                self.tcx().sess, span, E0221,
                "ambiguous associated type `{}` in bounds of `{}`",
                assoc_name,
                ty_param_name);
            err.span_label(span, &format!("ambiguous associated type `{}`", assoc_name));

            for bound in &bounds {
                span_note!(&mut err, span,
                           "associated type `{}` could derive from `{}`",
                           ty_param_name,
                           bound);
            }
            err.emit();
        }

        Ok(bounds[0].clone())
    }

    // Create a type from a path to an associated type.
    // For a path A::B::C::D, ty and ty_path_def are the type and def for A::B::C
    // and item_segment is the path segment for D. We return a type and a def for
    // the whole path.
    // Will fail except for T::A and Self::A; i.e., if ty/ty_path_def are not a type
    // parameter or Self.
    fn associated_path_def_to_ty(&self,
                                 span: Span,
                                 ty: Ty<'tcx>,
                                 ty_path_def: Def,
                                 item_segment: &hir::PathSegment)
                                 -> (Ty<'tcx>, Def)
    {
        let tcx = self.tcx();
        let assoc_name = item_segment.name;

        debug!("associated_path_def_to_ty: {:?}::{}", ty, assoc_name);

        tcx.prohibit_type_params(slice::ref_slice(item_segment));

        // Find the type of the associated item, and the trait where the associated
        // item is declared.
        let bound = match (&ty.sty, ty_path_def) {
            (_, Def::SelfTy(Some(_), Some(impl_def_id))) => {
                // `Self` in an impl of a trait - we have a concrete self type and a
                // trait reference.
                let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
                let trait_ref = if let Some(free_substs) = self.get_free_substs() {
                    trait_ref.subst(tcx, free_substs)
                } else {
                    trait_ref
                };

                if self.ensure_super_predicates(span, trait_ref.def_id).is_err() {
                    return (tcx.types.err, Def::Err);
                }

                let candidates: Vec<ty::PolyTraitRef> =
                    traits::supertraits(tcx, ty::Binder(trait_ref))
                    .filter(|r| self.trait_defines_associated_type_named(r.def_id(),
                                                                         assoc_name))
                    .collect();

                match self.one_bound_for_assoc_type(candidates,
                                                    "Self",
                                                    &assoc_name.as_str(),
                                                    span) {
                    Ok(bound) => bound,
                    Err(ErrorReported) => return (tcx.types.err, Def::Err),
                }
            }
            (&ty::TyParam(_), Def::SelfTy(Some(trait_did), None)) => {
                let trait_node_id = tcx.map.as_local_node_id(trait_did).unwrap();
                match self.find_bound_for_assoc_item(trait_node_id,
                                                     keywords::SelfType.name(),
                                                     assoc_name,
                                                     span) {
                    Ok(bound) => bound,
                    Err(ErrorReported) => return (tcx.types.err, Def::Err),
                }
            }
            (&ty::TyParam(_), Def::TyParam(param_did)) => {
                let param_node_id = tcx.map.as_local_node_id(param_did).unwrap();
                let param_name = tcx.type_parameter_def(param_node_id).name;
                match self.find_bound_for_assoc_item(param_node_id,
                                                     param_name,
                                                     assoc_name,
                                                     span) {
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

        let trait_did = bound.0.def_id;
        let ty = self.projected_ty_from_poly_trait_ref(span, bound, assoc_name);

        let item_did = if let Some(trait_id) = tcx.map.as_local_node_id(trait_did) {
            // `ty::trait_items` used below requires information generated
            // by type collection, which may be in progress at this point.
            match tcx.map.expect_item(trait_id).node {
                hir::ItemTrait(.., ref trait_items) => {
                    let item = trait_items.iter()
                                          .find(|i| i.name == assoc_name)
                                          .expect("missing associated type");
                    tcx.map.local_def_id(item.id)
                }
                _ => bug!()
            }
        } else {
            let trait_items = tcx.trait_items(trait_did);
            let item = trait_items.iter().find(|i| i.name() == assoc_name);
            item.expect("missing associated type").def_id()
        };

        (ty, Def::AssociatedTy(item_did))
    }

    fn qpath_to_ty(&self,
                   rscope: &RegionScope,
                   span: Span,
                   param_mode: PathParamMode,
                   opt_self_ty: Option<Ty<'tcx>>,
                   trait_def_id: DefId,
                   trait_segment: &hir::PathSegment,
                   item_segment: &hir::PathSegment)
                   -> Ty<'tcx>
    {
        let tcx = self.tcx();

        tcx.prohibit_type_params(slice::ref_slice(item_segment));

        let self_ty = if let Some(ty) = opt_self_ty {
            ty
        } else {
            let path_str = tcx.item_path_str(trait_def_id);
            self.report_ambiguous_associated_type(span,
                                                  "Type",
                                                  &path_str,
                                                  &item_segment.name.as_str());
            return tcx.types.err;
        };

        debug!("qpath_to_ty: self_type={:?}", self_ty);

        let trait_ref = self.ast_path_to_mono_trait_ref(rscope,
                                                        span,
                                                        param_mode,
                                                        trait_def_id,
                                                        self_ty,
                                                        trait_segment);

        debug!("qpath_to_ty: trait_ref={:?}", trait_ref);

        self.projected_ty(span, trait_ref, item_segment.name)
    }

    /// Convert a type supplied as value for a type argument from AST into our
    /// our internal representation. This is the same as `ast_ty_to_ty` but that
    /// it applies the object lifetime default.
    ///
    /// # Parameters
    ///
    /// * `this`, `rscope`: the surrounding context
    /// * `def`: the type parameter being instantiated (if available)
    /// * `region_substs`: a partial substitution consisting of
    ///   only the region type parameters being supplied to this type.
    /// * `ast_ty`: the ast representation of the type being supplied
    fn ast_ty_arg_to_ty(&self,
                        rscope: &RegionScope,
                        def: Option<&ty::TypeParameterDef<'tcx>>,
                        region_substs: &Substs<'tcx>,
                        ast_ty: &hir::Ty)
                        -> Ty<'tcx>
    {
        let tcx = self.tcx();

        if let Some(def) = def {
            let object_lifetime_default = def.object_lifetime_default.subst(tcx, region_substs);
            let rscope1 = &ObjectLifetimeDefaultRscope::new(rscope, object_lifetime_default);
            self.ast_ty_to_ty(rscope1, ast_ty)
        } else {
            self.ast_ty_to_ty(rscope, ast_ty)
        }
    }

    // Check the base def in a PathResolution and convert it to a Ty. If there are
    // associated types in the PathResolution, these will need to be separately
    // resolved.
    fn base_def_to_ty(&self,
                      rscope: &RegionScope,
                      span: Span,
                      param_mode: PathParamMode,
                      def: Def,
                      opt_self_ty: Option<Ty<'tcx>>,
                      base_path_ref_id: ast::NodeId,
                      base_segments: &[hir::PathSegment])
                      -> Ty<'tcx> {
        let tcx = self.tcx();

        debug!("base_def_to_ty(def={:?}, opt_self_ty={:?}, base_segments={:?})",
               def, opt_self_ty, base_segments);

        match def {
            Def::Trait(trait_def_id) => {
                // N.B. this case overlaps somewhat with
                // TyObjectSum, see that fn for details

                tcx.prohibit_type_params(base_segments.split_last().unwrap().1);

                self.trait_path_to_object_type(rscope,
                                               span,
                                               param_mode,
                                               trait_def_id,
                                               base_path_ref_id,
                                               base_segments.last().unwrap(),
                                               span,
                                               partition_bounds(tcx, span, &[]))
            }
            Def::Enum(did) | Def::TyAlias(did) | Def::Struct(did) | Def::Union(did) => {
                tcx.prohibit_type_params(base_segments.split_last().unwrap().1);
                self.ast_path_to_ty(rscope,
                                    span,
                                    param_mode,
                                    did,
                                    base_segments.last().unwrap())
            }
            Def::TyParam(did) => {
                tcx.prohibit_type_params(base_segments);

                let node_id = tcx.map.as_local_node_id(did).unwrap();
                let param = tcx.ty_param_defs.borrow().get(&node_id)
                               .map(ty::ParamTy::for_def);
                if let Some(p) = param {
                    p.to_ty(tcx)
                } else {
                    // Only while computing defaults of earlier type
                    // parameters can a type parameter be missing its def.
                    struct_span_err!(tcx.sess, span, E0128,
                                     "type parameters with a default cannot use \
                                      forward declared identifiers")
                        .span_label(span, &format!("defaulted type parameters \
                                                    cannot be forward declared"))
                        .emit();
                    tcx.types.err
                }
            }
            Def::SelfTy(_, Some(def_id)) => {
                // Self in impl (we know the concrete type).

                tcx.prohibit_type_params(base_segments);
                let impl_id = tcx.map.as_local_node_id(def_id).unwrap();
                let ty = tcx.node_id_to_type(impl_id);
                if let Some(free_substs) = self.get_free_substs() {
                    ty.subst(tcx, free_substs)
                } else {
                    ty
                }
            }
            Def::SelfTy(Some(_), None) => {
                // Self in trait.
                tcx.prohibit_type_params(base_segments);
                tcx.mk_self_type()
            }
            Def::AssociatedTy(def_id) => {
                tcx.prohibit_type_params(&base_segments[..base_segments.len()-2]);
                let trait_did = tcx.parent_def_id(def_id).unwrap();
                self.qpath_to_ty(rscope,
                                 span,
                                 param_mode,
                                 opt_self_ty,
                                 trait_did,
                                 &base_segments[base_segments.len()-2],
                                 base_segments.last().unwrap())
            }
            Def::Mod(..) => {
                // Used as sentinel by callers to indicate the `<T>::A::B::C` form.
                // FIXME(#22519) This part of the resolution logic should be
                // avoided entirely for that form, once we stop needed a Def
                // for `associated_path_def_to_ty`.
                // Fixing this will also let use resolve <Self>::Foo the same way we
                // resolve Self::Foo, at the moment we can't resolve the former because
                // we don't have the trait information around, which is just sad.

                assert!(base_segments.is_empty());

                opt_self_ty.expect("missing T in <T>::a::b::c")
            }
            Def::PrimTy(prim_ty) => {
                tcx.prim_ty_to_ty(base_segments, prim_ty)
            }
            Def::Err => {
                self.set_tainted_by_errors();
                return self.tcx().types.err;
            }
            _ => {
                struct_span_err!(tcx.sess, span, E0248,
                           "found value `{}` used as a type",
                            tcx.item_path_str(def.def_id()))
                           .span_label(span, &format!("value used as a type"))
                           .emit();
                return self.tcx().types.err;
            }
        }
    }

    // Resolve possibly associated type path into a type and final definition.
    // Note that both base_segments and assoc_segments may be empty, although not at same time.
    pub fn finish_resolving_def_to_ty(&self,
                                      rscope: &RegionScope,
                                      span: Span,
                                      param_mode: PathParamMode,
                                      base_def: Def,
                                      opt_self_ty: Option<Ty<'tcx>>,
                                      base_path_ref_id: ast::NodeId,
                                      base_segments: &[hir::PathSegment],
                                      assoc_segments: &[hir::PathSegment])
                                      -> (Ty<'tcx>, Def) {
        // Convert the base type.
        debug!("finish_resolving_def_to_ty(base_def={:?}, \
                base_segments={:?}, \
                assoc_segments={:?})",
               base_def,
               base_segments,
               assoc_segments);
        let base_ty = self.base_def_to_ty(rscope,
                                          span,
                                          param_mode,
                                          base_def,
                                          opt_self_ty,
                                          base_path_ref_id,
                                          base_segments);
        debug!("finish_resolving_def_to_ty: base_def_to_ty returned {:?}", base_ty);

        // If any associated type segments remain, attempt to resolve them.
        let (mut ty, mut def) = (base_ty, base_def);
        for segment in assoc_segments {
            debug!("finish_resolving_def_to_ty: segment={:?}", segment);
            // This is pretty bad (it will fail except for T::A and Self::A).
            let (new_ty, new_def) = self.associated_path_def_to_ty(span, ty, def, segment);
            ty = new_ty;
            def = new_def;

            if def == Def::Err {
                break;
            }
        }
        (ty, def)
    }

    /// Parses the programmer's textual representation of a type into our
    /// internal notion of a type.
    pub fn ast_ty_to_ty(&self, rscope: &RegionScope, ast_ty: &hir::Ty) -> Ty<'tcx> {
        debug!("ast_ty_to_ty(id={:?}, ast_ty={:?})",
               ast_ty.id, ast_ty);

        let tcx = self.tcx();

        let cache = self.ast_ty_to_ty_cache();
        match cache.borrow().get(&ast_ty.id) {
            Some(ty) => { return ty; }
            None => { }
        }

        let result_ty = match ast_ty.node {
            hir::TySlice(ref ty) => {
                tcx.mk_slice(self.ast_ty_to_ty(rscope, &ty))
            }
            hir::TyObjectSum(ref ty, ref bounds) => {
                self.ast_ty_to_object_trait_ref(rscope, ast_ty.span, ty, bounds)
            }
            hir::TyPtr(ref mt) => {
                tcx.mk_ptr(ty::TypeAndMut {
                    ty: self.ast_ty_to_ty(rscope, &mt.ty),
                    mutbl: mt.mutbl
                })
            }
            hir::TyRptr(ref region, ref mt) => {
                let r = self.opt_ast_region_to_region(rscope, ast_ty.span, region);
                debug!("TyRef r={:?}", r);
                let rscope1 =
                    &ObjectLifetimeDefaultRscope::new(
                        rscope,
                        ty::ObjectLifetimeDefault::Specific(r));
                let t = self.ast_ty_to_ty(rscope1, &mt.ty);
                tcx.mk_ref(r, ty::TypeAndMut {ty: t, mutbl: mt.mutbl})
            }
            hir::TyNever => {
                tcx.types.never
            },
            hir::TyTup(ref fields) => {
                let flds = fields.iter()
                                 .map(|t| self.ast_ty_to_ty(rscope, &t))
                                 .collect();
                tcx.mk_tup(flds)
            }
            hir::TyBareFn(ref bf) => {
                require_c_abi_if_variadic(tcx, &bf.decl, bf.abi, ast_ty.span);
                let anon_scope = rscope.anon_type_scope();
                let (bare_fn_ty, _) =
                    self.ty_of_method_or_bare_fn(bf.unsafety,
                                                 bf.abi,
                                                 None,
                                                 &bf.decl,
                                                 anon_scope,
                                                 anon_scope);

                // Find any late-bound regions declared in return type that do
                // not appear in the arguments. These are not wellformed.
                //
                // Example:
                //
                //     for<'a> fn() -> &'a str <-- 'a is bad
                //     for<'a> fn(&'a String) -> &'a str <-- 'a is ok
                //
                // Note that we do this check **here** and not in
                // `ty_of_bare_fn` because the latter is also used to make
                // the types for fn items, and we do not want to issue a
                // warning then. (Once we fix #32330, the regions we are
                // checking for here would be considered early bound
                // anyway.)
                let inputs = bare_fn_ty.sig.inputs();
                let late_bound_in_args = tcx.collect_constrained_late_bound_regions(&inputs);
                let output = bare_fn_ty.sig.output();
                let late_bound_in_ret = tcx.collect_referenced_late_bound_regions(&output);
                for br in late_bound_in_ret.difference(&late_bound_in_args) {
                    let br_name = match *br {
                        ty::BrNamed(_, name, _) => name,
                        _ => {
                            span_bug!(
                                bf.decl.output.span(),
                                "anonymous bound region {:?} in return but not args",
                                br);
                        }
                    };
                    tcx.sess.add_lint(
                        lint::builtin::HR_LIFETIME_IN_ASSOC_TYPE,
                        ast_ty.id,
                        ast_ty.span,
                        format!("return type references lifetime `{}`, \
                                 which does not appear in the trait input types",
                                br_name));
                }
                tcx.mk_fn_ptr(bare_fn_ty)
            }
            hir::TyPolyTraitRef(ref bounds) => {
                self.conv_object_ty_poly_trait_ref(rscope, ast_ty.span, bounds)
            }
            hir::TyImplTrait(ref bounds) => {
                use collect::{compute_bounds, SizedByDefault};

                // Create the anonymized type.
                let def_id = tcx.map.local_def_id(ast_ty.id);
                if let Some(anon_scope) = rscope.anon_type_scope() {
                    let substs = anon_scope.fresh_substs(self, ast_ty.span);
                    let ty = tcx.mk_anon(tcx.map.local_def_id(ast_ty.id), substs);

                    // Collect the bounds, i.e. the `A+B+'c` in `impl A+B+'c`.
                    let bounds = compute_bounds(self, ty, bounds,
                                                SizedByDefault::Yes,
                                                Some(anon_scope),
                                                ast_ty.span);
                    let predicates = bounds.predicates(tcx, ty);
                    let predicates = tcx.lift_to_global(&predicates).unwrap();
                    tcx.predicates.borrow_mut().insert(def_id, ty::GenericPredicates {
                        parent: None,
                        predicates: predicates
                    });

                    ty
                } else {
                    span_err!(tcx.sess, ast_ty.span, E0562,
                              "`impl Trait` not allowed outside of function \
                               and inherent method return types");
                    tcx.types.err
                }
            }
            hir::TyPath(ref maybe_qself, ref path) => {
                debug!("ast_ty_to_ty: maybe_qself={:?} path={:?}", maybe_qself, path);
                let path_res = tcx.expect_resolution(ast_ty.id);
                let base_ty_end = path.segments.len() - path_res.depth;
                let opt_self_ty = maybe_qself.as_ref().map(|qself| {
                    self.ast_ty_to_ty(rscope, &qself.ty)
                });
                let (ty, def) = self.finish_resolving_def_to_ty(rscope,
                                                                ast_ty.span,
                                                                PathParamMode::Explicit,
                                                                path_res.base_def,
                                                                opt_self_ty,
                                                                ast_ty.id,
                                                                &path.segments[..base_ty_end],
                                                                &path.segments[base_ty_end..]);

                // Write back the new resolution.
                if path_res.depth != 0 {
                    tcx.def_map.borrow_mut().insert(ast_ty.id, PathResolution::new(def));
                }

                ty
            }
            hir::TyArray(ref ty, ref e) => {
                if let Ok(length) = eval_length(tcx.global_tcx(), &e, "array length") {
                    tcx.mk_array(self.ast_ty_to_ty(rscope, &ty), length)
                } else {
                    self.tcx().types.err
                }
            }
            hir::TyTypeof(ref _e) => {
                struct_span_err!(tcx.sess, ast_ty.span, E0516,
                                 "`typeof` is a reserved keyword but unimplemented")
                    .span_label(ast_ty.span, &format!("reserved keyword"))
                    .emit();

                tcx.types.err
            }
            hir::TyInfer => {
                // TyInfer also appears as the type of arguments or return
                // values in a ExprClosure, or as
                // the type of local variables. Both of these cases are
                // handled specially and will not descend into this routine.
                self.ty_infer(ast_ty.span)
            }
        };

        cache.borrow_mut().insert(ast_ty.id, result_ty);

        result_ty
    }

    pub fn ty_of_arg(&self,
                     rscope: &RegionScope,
                     a: &hir::Arg,
                     expected_ty: Option<Ty<'tcx>>)
                     -> Ty<'tcx>
    {
        match a.ty.node {
            hir::TyInfer if expected_ty.is_some() => expected_ty.unwrap(),
            hir::TyInfer => self.ty_infer(a.ty.span),
            _ => self.ast_ty_to_ty(rscope, &a.ty),
        }
    }

    pub fn ty_of_method(&self,
                        sig: &hir::MethodSig,
                        untransformed_self_ty: Ty<'tcx>,
                        anon_scope: Option<AnonTypeScope>)
                        -> (&'tcx ty::BareFnTy<'tcx>, ty::ExplicitSelfCategory<'tcx>) {
        self.ty_of_method_or_bare_fn(sig.unsafety,
                                     sig.abi,
                                     Some(untransformed_self_ty),
                                     &sig.decl,
                                     None,
                                     anon_scope)
    }

    pub fn ty_of_bare_fn(&self,
                         unsafety: hir::Unsafety,
                         abi: abi::Abi,
                         decl: &hir::FnDecl,
                         anon_scope: Option<AnonTypeScope>)
                         -> &'tcx ty::BareFnTy<'tcx> {
        self.ty_of_method_or_bare_fn(unsafety, abi, None, decl, None, anon_scope).0
    }

    fn ty_of_method_or_bare_fn(&self,
                               unsafety: hir::Unsafety,
                               abi: abi::Abi,
                               opt_untransformed_self_ty: Option<Ty<'tcx>>,
                               decl: &hir::FnDecl,
                               arg_anon_scope: Option<AnonTypeScope>,
                               ret_anon_scope: Option<AnonTypeScope>)
                               -> (&'tcx ty::BareFnTy<'tcx>, ty::ExplicitSelfCategory<'tcx>)
    {
        debug!("ty_of_method_or_bare_fn");

        // New region names that appear inside of the arguments of the function
        // declaration are bound to that function type.
        let rb = MaybeWithAnonTypes::new(BindingRscope::new(), arg_anon_scope);

        // `implied_output_region` is the region that will be assumed for any
        // region parameters in the return type. In accordance with the rules for
        // lifetime elision, we can determine it in two ways. First (determined
        // here), if self is by-reference, then the implied output region is the
        // region of the self parameter.
        let (self_ty, explicit_self_category) = match (opt_untransformed_self_ty, decl.get_self()) {
            (Some(untransformed_self_ty), Some(explicit_self)) => {
                let self_type = self.determine_self_type(&rb, untransformed_self_ty,
                                                         &explicit_self);
                (Some(self_type.0), self_type.1)
            }
            _ => (None, ty::ExplicitSelfCategory::Static),
        };

        // HACK(eddyb) replace the fake self type in the AST with the actual type.
        let arg_params = if self_ty.is_some() {
            &decl.inputs[1..]
        } else {
            &decl.inputs[..]
        };
        let arg_tys: Vec<Ty> =
            arg_params.iter().map(|a| self.ty_of_arg(&rb, a, None)).collect();
        let arg_pats: Vec<String> =
            arg_params.iter().map(|a| pprust::pat_to_string(&a.pat)).collect();

        // Second, if there was exactly one lifetime (either a substitution or a
        // reference) in the arguments, then any anonymous regions in the output
        // have that lifetime.
        let implied_output_region = match explicit_self_category {
            ty::ExplicitSelfCategory::ByReference(region, _) => Ok(*region),
            _ => self.find_implied_output_region(&arg_tys, arg_pats)
        };

        let output_ty = match decl.output {
            hir::Return(ref output) =>
                self.convert_ty_with_lifetime_elision(implied_output_region,
                                                      &output,
                                                      ret_anon_scope),
            hir::DefaultReturn(..) => self.tcx().mk_nil(),
        };

        let input_tys = self_ty.into_iter().chain(arg_tys).collect();

        debug!("ty_of_method_or_bare_fn: input_tys={:?}", input_tys);
        debug!("ty_of_method_or_bare_fn: output_ty={:?}", output_ty);

        (self.tcx().mk_bare_fn(ty::BareFnTy {
            unsafety: unsafety,
            abi: abi,
            sig: ty::Binder(ty::FnSig {
                inputs: input_tys,
                output: output_ty,
                variadic: decl.variadic
            }),
        }), explicit_self_category)
    }

    fn determine_self_type<'a>(&self,
                               rscope: &RegionScope,
                               untransformed_self_ty: Ty<'tcx>,
                               explicit_self: &hir::ExplicitSelf)
                               -> (Ty<'tcx>, ty::ExplicitSelfCategory<'tcx>)
    {
        return match explicit_self.node {
            SelfKind::Value(..) => {
                (untransformed_self_ty, ty::ExplicitSelfCategory::ByValue)
            }
            SelfKind::Region(ref lifetime, mutability) => {
                let region =
                    self.opt_ast_region_to_region(
                                             rscope,
                                             explicit_self.span,
                                             lifetime);
                (self.tcx().mk_ref(region,
                    ty::TypeAndMut {
                        ty: untransformed_self_ty,
                        mutbl: mutability
                    }),
                 ty::ExplicitSelfCategory::ByReference(region, mutability))
            }
            SelfKind::Explicit(ref ast_type, _) => {
                let explicit_type = self.ast_ty_to_ty(rscope, &ast_type);

                // We wish to (for now) categorize an explicit self
                // declaration like `self: SomeType` into either `self`,
                // `&self`, `&mut self`, or `Box<self>`. We do this here
                // by some simple pattern matching. A more precise check
                // is done later in `check_method_self_type()`.
                //
                // Examples:
                //
                // ```
                // impl Foo for &T {
                //     // Legal declarations:
                //     fn method1(self: &&T); // ExplicitSelfCategory::ByReference
                //     fn method2(self: &T); // ExplicitSelfCategory::ByValue
                //     fn method3(self: Box<&T>); // ExplicitSelfCategory::ByBox
                //
                //     // Invalid cases will be caught later by `check_method_self_type`:
                //     fn method_err1(self: &mut T); // ExplicitSelfCategory::ByReference
                // }
                // ```
                //
                // To do the check we just count the number of "modifiers"
                // on each type and compare them. If they are the same or
                // the impl has more, we call it "by value". Otherwise, we
                // look at the outermost modifier on the method decl and
                // call it by-ref, by-box as appropriate. For method1, for
                // example, the impl type has one modifier, but the method
                // type has two, so we end up with
                // ExplicitSelfCategory::ByReference.

                let impl_modifiers = count_modifiers(untransformed_self_ty);
                let method_modifiers = count_modifiers(explicit_type);

                debug!("determine_explicit_self_category(self_info.untransformed_self_ty={:?} \
                       explicit_type={:?} \
                       modifiers=({},{})",
                       untransformed_self_ty,
                       explicit_type,
                       impl_modifiers,
                       method_modifiers);

                let category = if impl_modifiers >= method_modifiers {
                    ty::ExplicitSelfCategory::ByValue
                } else {
                    match explicit_type.sty {
                        ty::TyRef(r, mt) => ty::ExplicitSelfCategory::ByReference(r, mt.mutbl),
                        ty::TyBox(_) => ty::ExplicitSelfCategory::ByBox,
                        _ => ty::ExplicitSelfCategory::ByValue,
                    }
                };

                (explicit_type, category)
            }
        };

        fn count_modifiers(ty: Ty) -> usize {
            match ty.sty {
                ty::TyRef(_, mt) => count_modifiers(mt.ty) + 1,
                ty::TyBox(t) => count_modifiers(t) + 1,
                _ => 0,
            }
        }
    }

    pub fn ty_of_closure(&self,
        unsafety: hir::Unsafety,
        decl: &hir::FnDecl,
        abi: abi::Abi,
        expected_sig: Option<ty::FnSig<'tcx>>)
        -> ty::ClosureTy<'tcx>
    {
        debug!("ty_of_closure(expected_sig={:?})",
               expected_sig);

        // new region names that appear inside of the fn decl are bound to
        // that function type
        let rb = rscope::BindingRscope::new();

        let input_tys: Vec<_> = decl.inputs.iter().enumerate().map(|(i, a)| {
            let expected_arg_ty = expected_sig.as_ref().and_then(|e| {
                // no guarantee that the correct number of expected args
                // were supplied
                if i < e.inputs.len() {
                    Some(e.inputs[i])
                } else {
                    None
                }
            });
            self.ty_of_arg(&rb, a, expected_arg_ty)
        }).collect();

        let expected_ret_ty = expected_sig.map(|e| e.output);

        let is_infer = match decl.output {
            hir::Return(ref output) if output.node == hir::TyInfer => true,
            hir::DefaultReturn(..) => true,
            _ => false
        };

        let output_ty = match decl.output {
            _ if is_infer && expected_ret_ty.is_some() =>
                expected_ret_ty.unwrap(),
            _ if is_infer => self.ty_infer(decl.output.span()),
            hir::Return(ref output) =>
                self.ast_ty_to_ty(&rb, &output),
            hir::DefaultReturn(..) => bug!(),
        };

        debug!("ty_of_closure: input_tys={:?}", input_tys);
        debug!("ty_of_closure: output_ty={:?}", output_ty);

        ty::ClosureTy {
            unsafety: unsafety,
            abi: abi,
            sig: ty::Binder(ty::FnSig {inputs: input_tys,
                                       output: output_ty,
                                       variadic: decl.variadic}),
        }
    }

    fn conv_object_ty_poly_trait_ref(&self,
        rscope: &RegionScope,
        span: Span,
        ast_bounds: &[hir::TyParamBound])
        -> Ty<'tcx>
    {
        let mut partitioned_bounds = partition_bounds(self.tcx(), span, &ast_bounds[..]);

        let trait_bound = if !partitioned_bounds.trait_bounds.is_empty() {
            partitioned_bounds.trait_bounds.remove(0)
        } else {
            span_err!(self.tcx().sess, span, E0224,
                      "at least one non-builtin trait is required for an object type");
            return self.tcx().types.err;
        };

        let trait_ref = &trait_bound.trait_ref;
        let trait_def_id = self.trait_def_id(trait_ref);
        self.trait_path_to_object_type(rscope,
                                       trait_ref.path.span,
                                       PathParamMode::Explicit,
                                       trait_def_id,
                                       trait_ref.ref_id,
                                       trait_ref.path.segments.last().unwrap(),
                                       span,
                                       partitioned_bounds)
    }

    /// Given the bounds on an object, determines what single region bound (if any) we can
    /// use to summarize this type. The basic idea is that we will use the bound the user
    /// provided, if they provided one, and otherwise search the supertypes of trait bounds
    /// for region bounds. It may be that we can derive no bound at all, in which case
    /// we return `None`.
    fn compute_object_lifetime_bound(&self,
        span: Span,
        explicit_region_bounds: &[&hir::Lifetime],
        principal_trait_ref: ty::PolyExistentialTraitRef<'tcx>,
        builtin_bounds: ty::BuiltinBounds)
        -> Option<&'tcx ty::Region> // if None, use the default
    {
        let tcx = self.tcx();

        debug!("compute_opt_region_bound(explicit_region_bounds={:?}, \
               principal_trait_ref={:?}, builtin_bounds={:?})",
               explicit_region_bounds,
               principal_trait_ref,
               builtin_bounds);

        if explicit_region_bounds.len() > 1 {
            span_err!(tcx.sess, explicit_region_bounds[1].span, E0226,
                "only a single explicit lifetime bound is permitted");
        }

        if !explicit_region_bounds.is_empty() {
            // Explicitly specified region bound. Use that.
            let r = explicit_region_bounds[0];
            return Some(ast_region_to_region(tcx, r));
        }

        if let Err(ErrorReported) =
                self.ensure_super_predicates(span, principal_trait_ref.def_id()) {
            return Some(tcx.mk_region(ty::ReStatic));
        }

        // No explicit region bound specified. Therefore, examine trait
        // bounds and see if we can derive region bounds from those.
        let derived_region_bounds =
            object_region_bounds(tcx, principal_trait_ref, builtin_bounds);

        // If there are no derived region bounds, then report back that we
        // can find no region bound. The caller will use the default.
        if derived_region_bounds.is_empty() {
            return None;
        }

        // If any of the derived region bounds are 'static, that is always
        // the best choice.
        if derived_region_bounds.iter().any(|&r| ty::ReStatic == *r) {
            return Some(tcx.mk_region(ty::ReStatic));
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

pub struct PartitionedBounds<'a> {
    pub builtin_bounds: ty::BuiltinBounds,
    pub trait_bounds: Vec<&'a hir::PolyTraitRef>,
    pub region_bounds: Vec<&'a hir::Lifetime>,
}

/// Divides a list of bounds from the AST into three groups: builtin bounds (Copy, Sized etc),
/// general trait bounds, and region bounds.
pub fn partition_bounds<'a, 'b, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                            _span: Span,
                                            ast_bounds: &'b [hir::TyParamBound])
                                            -> PartitionedBounds<'b>
{
    let mut builtin_bounds = ty::BuiltinBounds::empty();
    let mut region_bounds = Vec::new();
    let mut trait_bounds = Vec::new();
    for ast_bound in ast_bounds {
        match *ast_bound {
            hir::TraitTyParamBound(ref b, hir::TraitBoundModifier::None) => {
                match tcx.expect_def(b.trait_ref.ref_id) {
                    Def::Trait(trait_did) => {
                        if tcx.try_add_builtin_trait(trait_did,
                                                     &mut builtin_bounds) {
                            let segments = &b.trait_ref.path.segments;
                            let parameters = &segments[segments.len() - 1].parameters;
                            if !parameters.types().is_empty() {
                                check_type_argument_count(tcx, b.trait_ref.path.span,
                                                          parameters.types().len(), &[]);
                            }
                            if !parameters.lifetimes().is_empty() {
                                report_lifetime_number_error(tcx, b.trait_ref.path.span,
                                                             parameters.lifetimes().len(), 0);
                            }
                            continue; // success
                        }
                    }
                    _ => {
                        // Not a trait? that's an error, but it'll get
                        // reported later.
                    }
                }
                trait_bounds.push(b);
            }
            hir::TraitTyParamBound(_, hir::TraitBoundModifier::Maybe) => {}
            hir::RegionTyParamBound(ref l) => {
                region_bounds.push(l);
            }
        }
    }

    PartitionedBounds {
        builtin_bounds: builtin_bounds,
        trait_bounds: trait_bounds,
        region_bounds: region_bounds,
    }
}

fn check_type_argument_count(tcx: TyCtxt, span: Span, supplied: usize,
                             ty_param_defs: &[ty::TypeParameterDef]) {
    let accepted = ty_param_defs.len();
    let required = ty_param_defs.iter().take_while(|x| x.default.is_none()) .count();
    if supplied < required {
        let expected = if required < accepted {
            "expected at least"
        } else {
            "expected"
        };
        struct_span_err!(tcx.sess, span, E0243, "wrong number of type arguments")
            .span_label(
                span,
                &format!("{} {} type arguments, found {}", expected, required, supplied)
            )
            .emit();
    } else if supplied > accepted {
        let expected = if required == 0 {
            "expected no".to_string()
        } else if required < accepted {
            format!("expected at most {}", accepted)
        } else {
            format!("expected {}", accepted)
        };

        struct_span_err!(tcx.sess, span, E0244, "wrong number of type arguments")
            .span_label(
                span,
                &format!("{} type arguments, found {}", expected, supplied)
            )
            .emit();
    }
}

fn report_lifetime_number_error(tcx: TyCtxt, span: Span, number: usize, expected: usize) {
    let label = if number < expected {
        if expected == 1 {
            format!("expected {} lifetime parameter", expected)
        } else {
            format!("expected {} lifetime parameters", expected)
        }
    } else {
        let additional = number - expected;
        if additional == 1 {
            "unexpected lifetime parameter".to_string()
        } else {
            format!("{} unexpected lifetime parameters", additional)
        }
    };
    struct_span_err!(tcx.sess, span, E0107,
                     "wrong number of lifetime parameters: expected {}, found {}",
                     expected, number)
        .span_label(span, &label)
        .emit();
}

// A helper struct for conveniently grouping a set of bounds which we pass to
// and return from functions in multiple places.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Bounds<'tcx> {
    pub region_bounds: Vec<&'tcx ty::Region>,
    pub builtin_bounds: ty::BuiltinBounds,
    pub trait_bounds: Vec<ty::PolyTraitRef<'tcx>>,
    pub projection_bounds: Vec<ty::PolyProjectionPredicate<'tcx>>,
}

impl<'a, 'gcx, 'tcx> Bounds<'tcx> {
    pub fn predicates(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, param_ty: Ty<'tcx>)
                      -> Vec<ty::Predicate<'tcx>>
    {
        let mut vec = Vec::new();

        for builtin_bound in &self.builtin_bounds {
            match tcx.trait_ref_for_builtin_bound(builtin_bound, param_ty) {
                Ok(trait_ref) => { vec.push(trait_ref.to_predicate()); }
                Err(ErrorReported) => { }
            }
        }

        for &region_bound in &self.region_bounds {
            // account for the binder being introduced below; no need to shift `param_ty`
            // because, at present at least, it can only refer to early-bound regions
            let region_bound = tcx.mk_region(ty::fold::shift_region(*region_bound, 1));
            vec.push(ty::Binder(ty::OutlivesPredicate(param_ty, region_bound)).to_predicate());
        }

        for bound_trait_ref in &self.trait_bounds {
            vec.push(bound_trait_ref.to_predicate());
        }

        for projection in &self.projection_bounds {
            vec.push(projection.to_predicate());
        }

        vec
    }
}
