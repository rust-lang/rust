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

use middle::astconv_util::{ast_ty_to_prim_ty, check_path_args, NO_TPS, NO_REGIONS};
use middle::const_eval;
use middle::def;
use middle::resolve_lifetime as rl;
use middle::subst::{FnSpace, TypeSpace, SelfSpace, Subst, Substs};
use middle::traits;
use middle::ty::{self, RegionEscape, ToPolyTraitRef, Ty};
use rscope::{self, UnelidableRscope, RegionScope, ElidableRscope,
             ObjectLifetimeDefaultRscope, ShiftedRscope, BindingRscope};
use TypeAndSubsts;
use util::common::{ErrorReported, FN_OUTPUT_NAME};
use util::nodemap::DefIdMap;
use util::ppaux::{self, Repr, UserString};

use std::rc::Rc;
use std::iter::{repeat, AdditiveIterator};
use syntax::{abi, ast, ast_util};
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::print::pprust;

pub trait AstConv<'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx>;

    fn get_item_type_scheme(&self, id: ast::DefId) -> ty::TypeScheme<'tcx>;

    fn get_trait_def(&self, id: ast::DefId) -> Rc<ty::TraitDef<'tcx>>;

    /// Return an (optional) substitution to convert bound type parameters that
    /// are in scope into free ones. This function should only return Some
    /// within a fn body.
    /// See ParameterEnvironment::free_substs for more information.
    fn get_free_substs(&self) -> Option<&Substs<'tcx>> {
        None
    }

    /// What type should we use when a type is omitted?
    fn ty_infer(&self, span: Span) -> Ty<'tcx>;

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
                                        -> Ty<'tcx>
    {
        if ty::binds_late_bound_regions(self.tcx(), &poly_trait_ref) {
            span_err!(self.tcx().sess, span, E0212,
                "cannot extract an associated type from a higher-ranked trait bound \
                 in this context");
            self.tcx().types.err
        } else {
            // no late-bound regions, we can just ignore the binder
            self.projected_ty(span, poly_trait_ref.0.clone(), item_name)
        }
    }

    /// Project an associated type from a non-higher-ranked trait reference.
    /// This is fairly straightforward and can be accommodated in any context.
    fn projected_ty(&self,
                    span: Span,
                    _trait_ref: Rc<ty::TraitRef<'tcx>>,
                    _item_name: ast::Name)
                    -> Ty<'tcx>
    {
        span_err!(self.tcx().sess, span, E0213,
            "associated types are not accepted in this context");

        self.tcx().types.err
    }
}

pub fn ast_region_to_region(tcx: &ty::ctxt, lifetime: &ast::Lifetime)
                            -> ty::Region {
    let r = match tcx.named_region_map.get(&lifetime.id) {
        None => {
            // should have been recorded by the `resolve_lifetime` pass
            tcx.sess.span_bug(lifetime.span, "unresolved lifetime");
        }

        Some(&rl::DefStaticRegion) => {
            ty::ReStatic
        }

        Some(&rl::DefLateBoundRegion(debruijn, id)) => {
            ty::ReLateBound(debruijn, ty::BrNamed(ast_util::local_def(id), lifetime.name))
        }

        Some(&rl::DefEarlyBoundRegion(space, index, id)) => {
            ty::ReEarlyBound(id, space, index, lifetime.name)
        }

        Some(&rl::DefFreeRegion(scope, id)) => {
            ty::ReFree(ty::FreeRegion {
                    scope: scope,
                    bound_region: ty::BrNamed(ast_util::local_def(id),
                                              lifetime.name)
                })
        }
    };

    debug!("ast_region_to_region(lifetime={} id={}) yields {}",
           lifetime.repr(tcx),
           lifetime.id,
           r.repr(tcx));

    r
}

pub fn opt_ast_region_to_region<'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    default_span: Span,
    opt_lifetime: &Option<ast::Lifetime>) -> ty::Region
{
    let r = match *opt_lifetime {
        Some(ref lifetime) => {
            ast_region_to_region(this.tcx(), lifetime)
        }

        None => {
            match rscope.anon_regions(default_span, 1) {
                Err(v) => {
                    debug!("optional region in illegal location");
                    span_err!(this.tcx().sess, default_span, E0106,
                        "missing lifetime specifier");
                    match v {
                        Some(v) => {
                            let mut m = String::new();
                            let len = v.len();
                            for (i, (name, n)) in v.into_iter().enumerate() {
                                let help_name = if name.is_empty() {
                                    format!("argument {}", i + 1)
                                } else {
                                    format!("`{}`", name)
                                };

                                m.push_str(&(if n == 1 {
                                    help_name
                                } else {
                                    format!("one of {}'s {} elided lifetimes", help_name, n)
                                })[]);

                                if len == 2 && i == 0 {
                                    m.push_str(" or ");
                                } else if i == len - 2 {
                                    m.push_str(", or ");
                                } else if i != len - 1 {
                                    m.push_str(", ");
                                }
                            }
                            if len == 1 {
                                span_help!(this.tcx().sess, default_span,
                                    "this function's return type contains a borrowed value, but \
                                     the signature does not say which {} it is borrowed from",
                                    m);
                            } else if len == 0 {
                                span_help!(this.tcx().sess, default_span,
                                    "this function's return type contains a borrowed value, but \
                                     there is no value for it to be borrowed from");
                                span_help!(this.tcx().sess, default_span,
                                    "consider giving it a 'static lifetime");
                            } else {
                                span_help!(this.tcx().sess, default_span,
                                    "this function's return type contains a borrowed value, but \
                                     the signature does not say whether it is borrowed from {}",
                                    m);
                            }
                        }
                        None => {},
                    }
                    ty::ReStatic
                }

                Ok(rs) => rs[0],
            }
        }
    };

    debug!("opt_ast_region_to_region(opt_lifetime={}) yields {}",
            opt_lifetime.repr(this.tcx()),
            r.repr(this.tcx()));

    r
}

/// Given a path `path` that refers to an item `I` with the declared generics `decl_generics`,
/// returns an appropriate set of substitutions for this particular reference to `I`.
pub fn ast_path_substs_for_ty<'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    decl_generics: &ty::Generics<'tcx>,
    path: &ast::Path)
    -> Substs<'tcx>
{
    let tcx = this.tcx();

    // ast_path_substs() is only called to convert paths that are
    // known to refer to traits, types, or structs. In these cases,
    // all type parameters defined for the item being referenced will
    // be in the TypeSpace or SelfSpace.
    //
    // Note: in the case of traits, the self parameter is also
    // defined, but we don't currently create a `type_param_def` for
    // `Self` because it is implicit.
    assert!(decl_generics.regions.all(|d| d.space == TypeSpace));
    assert!(decl_generics.types.all(|d| d.space != FnSpace));

    let (regions, types, assoc_bindings) = match path.segments.last().unwrap().parameters {
        ast::AngleBracketedParameters(ref data) => {
            convert_angle_bracketed_parameters(this, rscope, path.span, decl_generics, data)
        }
        ast::ParenthesizedParameters(ref data) => {
            span_err!(tcx.sess, path.span, E0214,
                "parenthesized parameters may only be used with a trait");
            convert_parenthesized_parameters(this, rscope, path.span, decl_generics, data)
        }
    };

    prohibit_projections(this.tcx(), &assoc_bindings);

    create_substs_for_ast_path(this,
                               path.span,
                               decl_generics,
                               None,
                               types,
                               regions)
}

fn create_region_substs<'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    span: Span,
    decl_generics: &ty::Generics<'tcx>,
    regions_provided: Vec<ty::Region>)
    -> Substs<'tcx>
{
    let tcx = this.tcx();

    // If the type is parameterized by the this region, then replace this
    // region with the current anon region binding (in other words,
    // whatever & would get replaced with).
    let expected_num_region_params = decl_generics.regions.len(TypeSpace);
    let supplied_num_region_params = regions_provided.len();
    let regions = if expected_num_region_params == supplied_num_region_params {
        regions_provided
    } else {
        let anon_regions =
            rscope.anon_regions(span, expected_num_region_params);

        if supplied_num_region_params != 0 || anon_regions.is_err() {
            span_err!(tcx.sess, span, E0107,
                      "wrong number of lifetime parameters: expected {}, found {}",
                      expected_num_region_params, supplied_num_region_params);
        }

        match anon_regions {
            Ok(anon_regions) => anon_regions,
            Err(_) => (0..expected_num_region_params).map(|_| ty::ReStatic).collect()
        }
    };
    Substs::new_type(vec![], regions)
}

/// Given the type/region arguments provided to some path (along with
/// an implicit Self, if this is a trait reference) returns the complete
/// set of substitutions. This may involve applying defaulted type parameters.
///
/// Note that the type listing given here is *exactly* what the user provided.
///
/// The `region_substs` should be the result of `create_region_substs`
/// -- that is, a substitution with no types but the correct number of
/// regions.
fn create_substs_for_ast_path<'tcx>(
    this: &AstConv<'tcx>,
    span: Span,
    decl_generics: &ty::Generics<'tcx>,
    self_ty: Option<Ty<'tcx>>,
    types_provided: Vec<Ty<'tcx>>,
    region_substs: Substs<'tcx>)
    -> Substs<'tcx>
{
    let tcx = this.tcx();

    debug!("create_substs_for_ast_path(decl_generics={}, self_ty={}, \
           types_provided={}, region_substs={}",
           decl_generics.repr(tcx), self_ty.repr(tcx), types_provided.repr(tcx),
           region_substs.repr(tcx));

    assert_eq!(region_substs.regions().len(TypeSpace), decl_generics.regions.len(TypeSpace));
    assert!(region_substs.types.is_empty());

    // Convert the type parameters supplied by the user.
    let ty_param_defs = decl_generics.types.get_slice(TypeSpace);
    let supplied_ty_param_count = types_provided.len();
    let formal_ty_param_count = ty_param_defs.len();
    let required_ty_param_count = ty_param_defs.iter()
                                               .take_while(|x| x.default.is_none())
                                               .count();

    let mut type_substs = types_provided;
    if supplied_ty_param_count < required_ty_param_count {
        let expected = if required_ty_param_count < formal_ty_param_count {
            "expected at least"
        } else {
            "expected"
        };
        span_err!(this.tcx().sess, span, E0243,
                  "wrong number of type arguments: {} {}, found {}",
                  expected,
                  required_ty_param_count,
                  supplied_ty_param_count);
        while type_substs.len() < required_ty_param_count {
            type_substs.push(tcx.types.err);
        }
    } else if supplied_ty_param_count > formal_ty_param_count {
        let expected = if required_ty_param_count < formal_ty_param_count {
            "expected at most"
        } else {
            "expected"
        };
        span_err!(this.tcx().sess, span, E0244,
                  "wrong number of type arguments: {} {}, found {}",
                  expected,
                  formal_ty_param_count,
                  supplied_ty_param_count);
        type_substs.truncate(formal_ty_param_count);
    }
    assert!(type_substs.len() >= required_ty_param_count &&
            type_substs.len() <= formal_ty_param_count);

    let mut substs = region_substs;
    substs.types.extend(TypeSpace, type_substs.into_iter());

    match self_ty {
        None => {
            // If no self-type is provided, it's still possible that
            // one was declared, because this could be an object type.
        }
        Some(ty) => {
            // If a self-type is provided, one should have been
            // "declared" (in other words, this should be a
            // trait-ref).
            assert!(decl_generics.types.get_self().is_some());
            substs.types.push(SelfSpace, ty);
        }
    }

    let actual_supplied_ty_param_count = substs.types.len(TypeSpace);
    for param in &ty_param_defs[actual_supplied_ty_param_count..] {
        if let Some(default) = param.default {
            // If we are converting an object type, then the
            // `Self` parameter is unknown. However, some of the
            // other type parameters may reference `Self` in their
            // defaults. This will lead to an ICE if we are not
            // careful!
            if self_ty.is_none() && ty::type_has_self(default) {
                tcx.sess.span_err(
                    span,
                    &format!("the type parameter `{}` must be explicitly specified \
                              in an object type because its default value `{}` references \
                              the type `Self`",
                             param.name.user_string(tcx),
                             default.user_string(tcx)));
                substs.types.push(TypeSpace, tcx.types.err);
            } else {
                // This is a default type parameter.
                let default = default.subst_spanned(tcx,
                                                    &substs,
                                                    Some(span));
                substs.types.push(TypeSpace, default);
            }
        } else {
            tcx.sess.span_bug(span, "extra parameter without default");
        }
    }

    return substs;
}

struct ConvertedBinding<'tcx> {
    item_name: ast::Name,
    ty: Ty<'tcx>,
    span: Span,
}

fn convert_angle_bracketed_parameters<'tcx>(this: &AstConv<'tcx>,
                                            rscope: &RegionScope,
                                            span: Span,
                                            decl_generics: &ty::Generics<'tcx>,
                                            data: &ast::AngleBracketedParameterData)
                                            -> (Substs<'tcx>,
                                                Vec<Ty<'tcx>>,
                                                Vec<ConvertedBinding<'tcx>>)
{
    let regions: Vec<_> =
        data.lifetimes.iter()
                      .map(|l| ast_region_to_region(this.tcx(), l))
                      .collect();

    let region_substs =
        create_region_substs(this, rscope, span, decl_generics, regions);

    let types: Vec<_> =
        data.types.iter()
                  .enumerate()
                  .map(|(i,t)| ast_ty_arg_to_ty(this, rscope, decl_generics,
                                                i, &region_substs, t))
                  .collect();

    let assoc_bindings: Vec<_> =
        data.bindings.iter()
                     .map(|b| ConvertedBinding { item_name: b.ident.name,
                                                 ty: ast_ty_to_ty(this, rscope, &*b.ty),
                                                 span: b.span })
                     .collect();

    (region_substs, types, assoc_bindings)
}

/// Returns the appropriate lifetime to use for any output lifetimes
/// (if one exists) and a vector of the (pattern, number of lifetimes)
/// corresponding to each input type/pattern.
fn find_implied_output_region(input_tys: &[Ty], input_pats: Vec<String>)
                              -> (Option<ty::Region>, Vec<(String, uint)>)
{
    let mut lifetimes_for_params: Vec<(String, uint)> = Vec::new();
    let mut possible_implied_output_region = None;

    for (input_type, input_pat) in input_tys.iter().zip(input_pats.into_iter()) {
        let mut accumulator = Vec::new();
        ty::accumulate_lifetimes_in_type(&mut accumulator, *input_type);

        if accumulator.len() == 1 {
            // there's a chance that the unique lifetime of this
            // iteration will be the appropriate lifetime for output
            // parameters, so lets store it.
            possible_implied_output_region = Some(accumulator[0])
        }

        lifetimes_for_params.push((input_pat, accumulator.len()));
    }

    let implied_output_region = if lifetimes_for_params.iter().map(|&(_, n)| n).sum() == 1 {
        assert!(possible_implied_output_region.is_some());
        possible_implied_output_region
    } else {
        None
    };
    (implied_output_region, lifetimes_for_params)
}

fn convert_ty_with_lifetime_elision<'tcx>(this: &AstConv<'tcx>,
                                          implied_output_region: Option<ty::Region>,
                                          param_lifetimes: Vec<(String, uint)>,
                                          ty: &ast::Ty)
                                          -> Ty<'tcx>
{
    match implied_output_region {
        Some(implied_output_region) => {
            let rb = ElidableRscope::new(implied_output_region);
            ast_ty_to_ty(this, &rb, ty)
        }
        None => {
            // All regions must be explicitly specified in the output
            // if the lifetime elision rules do not apply. This saves
            // the user from potentially-confusing errors.
            let rb = UnelidableRscope::new(param_lifetimes);
            ast_ty_to_ty(this, &rb, ty)
        }
    }
}

fn convert_parenthesized_parameters<'tcx>(this: &AstConv<'tcx>,
                                          rscope: &RegionScope,
                                          span: Span,
                                          decl_generics: &ty::Generics<'tcx>,
                                          data: &ast::ParenthesizedParameterData)
                                          -> (Substs<'tcx>,
                                              Vec<Ty<'tcx>>,
                                              Vec<ConvertedBinding<'tcx>>)
{
    let region_substs =
        create_region_substs(this, rscope, span, decl_generics, Vec::new());

    let binding_rscope = BindingRscope::new();
    let inputs =
        data.inputs.iter()
                   .map(|a_t| ast_ty_arg_to_ty(this, &binding_rscope, decl_generics,
                                               0, &region_substs, a_t))
                   .collect::<Vec<Ty<'tcx>>>();

    let input_params: Vec<_> = repeat(String::new()).take(inputs.len()).collect();
    let (implied_output_region,
         params_lifetimes) = find_implied_output_region(&*inputs, input_params);

    let input_ty = ty::mk_tup(this.tcx(), inputs);

    let (output, output_span) = match data.output {
        Some(ref output_ty) => {
            (convert_ty_with_lifetime_elision(this,
                                              implied_output_region,
                                              params_lifetimes,
                                              &**output_ty),
             output_ty.span)
        }
        None => {
            (ty::mk_nil(this.tcx()), data.span)
        }
    };

    let output_binding = ConvertedBinding {
        item_name: token::intern(FN_OUTPUT_NAME),
        ty: output,
        span: output_span
    };

    (region_substs, vec![input_ty], vec![output_binding])
}

pub fn instantiate_poly_trait_ref<'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    ast_trait_ref: &ast::PolyTraitRef,
    self_ty: Option<Ty<'tcx>>,
    poly_projections: &mut Vec<ty::PolyProjectionPredicate<'tcx>>)
    -> ty::PolyTraitRef<'tcx>
{
    let mut projections = Vec::new();

    // The trait reference introduces a binding level here, so
    // we need to shift the `rscope`. It'd be nice if we could
    // do away with this rscope stuff and work this knowledge
    // into resolve_lifetimes, as we do with non-omitted
    // lifetimes. Oh well, not there yet.
    let shifted_rscope = ShiftedRscope::new(rscope);

    let trait_ref =
        instantiate_trait_ref(this, &shifted_rscope, &ast_trait_ref.trait_ref,
                              self_ty, Some(&mut projections));

    for projection in projections {
        poly_projections.push(ty::Binder(projection));
    }

    ty::Binder(trait_ref)
}

/// Instantiates the path for the given trait reference, assuming that it's
/// bound to a valid trait type. Returns the def_id for the defining trait.
/// Fails if the type is a type other than a trait type.
///
/// If the `projections` argument is `None`, then assoc type bindings like `Foo<T=X>`
/// are disallowed. Otherwise, they are pushed onto the vector given.
pub fn instantiate_trait_ref<'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    ast_trait_ref: &ast::TraitRef,
    self_ty: Option<Ty<'tcx>>,
    projections: Option<&mut Vec<ty::ProjectionPredicate<'tcx>>>)
    -> Rc<ty::TraitRef<'tcx>>
{
    match ::lookup_def_tcx(this.tcx(), ast_trait_ref.path.span, ast_trait_ref.ref_id) {
        def::DefTrait(trait_def_id) => {
            let trait_ref = ast_path_to_trait_ref(this,
                                                  rscope,
                                                  trait_def_id,
                                                  self_ty,
                                                  &ast_trait_ref.path,
                                                  projections);
            this.tcx().trait_refs.borrow_mut().insert(ast_trait_ref.ref_id, trait_ref.clone());
            trait_ref
        }
        _ => {
            span_fatal!(this.tcx().sess, ast_trait_ref.path.span, E0245,
                "`{}` is not a trait",
                        ast_trait_ref.path.user_string(this.tcx()));
        }
    }
}

fn object_path_to_poly_trait_ref<'a,'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    trait_def_id: ast::DefId,
    path: &ast::Path,
    mut projections: &mut Vec<ty::PolyProjectionPredicate<'tcx>>)
    -> ty::PolyTraitRef<'tcx>
{
    // we are introducing a binder here, so shift the
    // anonymous regions depth to account for that
    let shifted_rscope = ShiftedRscope::new(rscope);

    let mut tmp = Vec::new();
    let trait_ref = ty::Binder(ast_path_to_trait_ref(this,
                                                     &shifted_rscope,
                                                     trait_def_id,
                                                     None,
                                                     path,
                                                     Some(&mut tmp)));
    projections.extend(tmp.into_iter().map(ty::Binder));
    trait_ref
}

fn ast_path_to_trait_ref<'a,'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    trait_def_id: ast::DefId,
    self_ty: Option<Ty<'tcx>>,
    path: &ast::Path,
    mut projections: Option<&mut Vec<ty::ProjectionPredicate<'tcx>>>)
    -> Rc<ty::TraitRef<'tcx>>
{
    debug!("ast_path_to_trait_ref {:?}", path);
    let trait_def = this.get_trait_def(trait_def_id);

    let (regions, types, assoc_bindings) = match path.segments.last().unwrap().parameters {
        ast::AngleBracketedParameters(ref data) => {
            // For now, require that parenthetical notation be used
            // only with `Fn()` etc.
            if !this.tcx().sess.features.borrow().unboxed_closures && trait_def.paren_sugar {
                span_err!(this.tcx().sess, path.span, E0215,
                                         "angle-bracket notation is not stable when \
                                         used with the `Fn` family of traits, use parentheses");
                span_help!(this.tcx().sess, path.span,
                           "add `#![feature(unboxed_closures)]` to \
                            the crate attributes to enable");
            }

            convert_angle_bracketed_parameters(this, rscope, path.span, &trait_def.generics, data)
        }
        ast::ParenthesizedParameters(ref data) => {
            // For now, require that parenthetical notation be used
            // only with `Fn()` etc.
            if !this.tcx().sess.features.borrow().unboxed_closures && !trait_def.paren_sugar {
                span_err!(this.tcx().sess, path.span, E0216,
                                         "parenthetical notation is only stable when \
                                         used with the `Fn` family of traits");
                span_help!(this.tcx().sess, path.span,
                           "add `#![feature(unboxed_closures)]` to \
                            the crate attributes to enable");
            }

            convert_parenthesized_parameters(this, rscope, path.span, &trait_def.generics, data)
        }
    };

    let substs = create_substs_for_ast_path(this,
                                            path.span,
                                            &trait_def.generics,
                                            self_ty,
                                            types,
                                            regions);
    let substs = this.tcx().mk_substs(substs);

    let trait_ref = Rc::new(ty::TraitRef::new(trait_def_id, substs));

    match projections {
        None => {
            prohibit_projections(this.tcx(), &assoc_bindings);
        }
        Some(ref mut v) => {
            for binding in &assoc_bindings {
                match ast_type_binding_to_projection_predicate(this, trait_ref.clone(),
                                                               self_ty, binding) {
                    Ok(pp) => { v.push(pp); }
                    Err(ErrorReported) => { }
                }
            }
        }
    }

    trait_ref
}

fn ast_type_binding_to_projection_predicate<'tcx>(
    this: &AstConv<'tcx>,
    mut trait_ref: Rc<ty::TraitRef<'tcx>>,
    self_ty: Option<Ty<'tcx>>,
    binding: &ConvertedBinding<'tcx>)
    -> Result<ty::ProjectionPredicate<'tcx>, ErrorReported>
{
    let tcx = this.tcx();

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

    // Simple case: X is defined in the current trait.
    if trait_defines_associated_type_named(this, trait_ref.def_id, binding.item_name) {
        return Ok(ty::ProjectionPredicate {
            projection_ty: ty::ProjectionTy {
                trait_ref: trait_ref,
                item_name: binding.item_name,
            },
            ty: binding.ty,
        });
    }

    // Otherwise, we have to walk through the supertraits to find
    // those that do.  This is complicated by the fact that, for an
    // object type, the `Self` type is not present in the
    // substitutions (after all, it's being constructed right now),
    // but the `supertraits` iterator really wants one. To handle
    // this, we currently insert a dummy type and then remove it
    // later. Yuck.

    let dummy_self_ty = ty::mk_infer(tcx, ty::FreshTy(0));
    if self_ty.is_none() { // if converting for an object type
        let mut dummy_substs = trait_ref.substs.clone();
        assert!(dummy_substs.self_ty().is_none());
        dummy_substs.types.push(SelfSpace, dummy_self_ty);
        trait_ref = Rc::new(ty::TraitRef::new(trait_ref.def_id,
                                              tcx.mk_substs(dummy_substs)));
    }

    let mut candidates: Vec<ty::PolyTraitRef> =
        traits::supertraits(tcx, trait_ref.to_poly_trait_ref())
        .filter(|r| trait_defines_associated_type_named(this, r.def_id(), binding.item_name))
        .collect();

    // If converting for an object type, then remove the dummy-ty from `Self` now.
    // Yuckety yuck.
    if self_ty.is_none() {
        for candidate in &mut candidates {
            let mut dummy_substs = candidate.0.substs.clone();
            assert!(dummy_substs.self_ty() == Some(dummy_self_ty));
            dummy_substs.types.pop(SelfSpace);
            *candidate = ty::Binder(Rc::new(ty::TraitRef::new(candidate.def_id(),
                                                              tcx.mk_substs(dummy_substs))));
        }
    }

    if candidates.len() > 1 {
        span_err!(tcx.sess, binding.span, E0217,
            "ambiguous associated type: `{}` defined in multiple supertraits `{}`",
                    token::get_name(binding.item_name),
                    candidates.user_string(tcx));
        return Err(ErrorReported);
    }

    let candidate = match candidates.pop() {
        Some(c) => c,
        None => {
            span_err!(tcx.sess, binding.span, E0218,
                "no associated type `{}` defined in `{}`",
                        token::get_name(binding.item_name),
                        trait_ref.user_string(tcx));
            return Err(ErrorReported);
        }
    };

    if ty::binds_late_bound_regions(tcx, &candidate) {
        span_err!(tcx.sess, binding.span, E0219,
            "associated type `{}` defined in higher-ranked supertrait `{}`",
                    token::get_name(binding.item_name),
                    candidate.user_string(tcx));
        return Err(ErrorReported);
    }

    Ok(ty::ProjectionPredicate {
        projection_ty: ty::ProjectionTy {
            trait_ref: candidate.0,
            item_name: binding.item_name,
        },
        ty: binding.ty,
    })
}

pub fn ast_path_to_ty<'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    did: ast::DefId,
    path: &ast::Path)
    -> TypeAndSubsts<'tcx>
{
    let tcx = this.tcx();
    let ty::TypeScheme {
        generics,
        ty: decl_ty
    } = this.get_item_type_scheme(did);

    let substs = ast_path_substs_for_ty(this,
                                        rscope,
                                        &generics,
                                        path);
    let ty = decl_ty.subst(tcx, &substs);
    TypeAndSubsts { substs: substs, ty: ty }
}

/// Converts the given AST type to a built-in type. A "built-in type" is, at
/// present, either a core numeric type, a string, or `Box`.
pub fn ast_ty_to_builtin_ty<'tcx>(
        this: &AstConv<'tcx>,
        rscope: &RegionScope,
        ast_ty: &ast::Ty)
        -> Option<Ty<'tcx>> {
    match ast_ty_to_prim_ty(this.tcx(), ast_ty) {
        Some(typ) => return Some(typ),
        None => {}
    }

    match ast_ty.node {
        ast::TyPath(ref path, id) => {
            let a_def = match this.tcx().def_map.borrow().get(&id) {
                None => {
                    this.tcx()
                        .sess
                        .span_bug(ast_ty.span,
                                  &format!("unbound path {}",
                                          path.repr(this.tcx()))[])
                }
                Some(&d) => d
            };

            // FIXME(#12938): This is a hack until we have full support for
            // DST.
            match a_def {
                def::DefTy(did, _) |
                def::DefStruct(did) if Some(did) == this.tcx().lang_items.owned_box() => {
                    let ty = ast_path_to_ty(this, rscope, did, path).ty;
                    match ty.sty {
                        ty::ty_struct(struct_def_id, ref substs) => {
                            assert_eq!(struct_def_id, did);
                            assert_eq!(substs.types.len(TypeSpace), 1);
                            let referent_ty = *substs.types.get(TypeSpace, 0);
                            Some(ty::mk_uniq(this.tcx(), referent_ty))
                        }
                        _ => {
                            this.tcx().sess.span_bug(
                                path.span,
                                &format!("converting `Box` to `{}`",
                                        ty.repr(this.tcx()))[]);
                        }
                    }
                }
                _ => None
            }
        }
        _ => None
    }
}

type TraitAndProjections<'tcx> = (ty::PolyTraitRef<'tcx>, Vec<ty::PolyProjectionPredicate<'tcx>>);

fn ast_ty_to_trait_ref<'tcx>(this: &AstConv<'tcx>,
                             rscope: &RegionScope,
                             ty: &ast::Ty,
                             bounds: &[ast::TyParamBound])
                             -> Result<TraitAndProjections<'tcx>, ErrorReported>
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

    match ty.node {
        ast::TyPath(ref path, id) => {
            match this.tcx().def_map.borrow().get(&id) {
                Some(&def::DefTrait(trait_def_id)) => {
                    let mut projection_bounds = Vec::new();
                    let trait_ref = object_path_to_poly_trait_ref(this,
                                                                  rscope,
                                                                  trait_def_id,
                                                                  path,
                                                                  &mut projection_bounds);
                    Ok((trait_ref, projection_bounds))
                }
                _ => {
                    span_err!(this.tcx().sess, ty.span, E0172, "expected a reference to a trait");
                    Err(ErrorReported)
                }
            }
        }
        _ => {
            span_err!(this.tcx().sess, ty.span, E0178,
                      "expected a path on the left-hand side of `+`, not `{}`",
                      pprust::ty_to_string(ty));
            match ty.node {
                ast::TyRptr(None, ref mut_ty) => {
                    span_help!(this.tcx().sess, ty.span,
                               "perhaps you meant `&{}({} +{})`? (per RFC 438)",
                               ppaux::mutability_to_string(mut_ty.mutbl),
                               pprust::ty_to_string(&*mut_ty.ty),
                               pprust::bounds_to_string(bounds));
                }
               ast::TyRptr(Some(ref lt), ref mut_ty) => {
                    span_help!(this.tcx().sess, ty.span,
                               "perhaps you meant `&{} {}({} +{})`? (per RFC 438)",
                               pprust::lifetime_to_string(lt),
                               ppaux::mutability_to_string(mut_ty.mutbl),
                               pprust::ty_to_string(&*mut_ty.ty),
                               pprust::bounds_to_string(bounds));
                }

                _ => {
                    span_help!(this.tcx().sess, ty.span,
                               "perhaps you forgot parentheses? (per RFC 438)");
                }
            }
            Err(ErrorReported)
        }
    }
}

fn trait_ref_to_object_type<'tcx>(this: &AstConv<'tcx>,
                                  rscope: &RegionScope,
                                  span: Span,
                                  trait_ref: ty::PolyTraitRef<'tcx>,
                                  projection_bounds: Vec<ty::PolyProjectionPredicate<'tcx>>,
                                  bounds: &[ast::TyParamBound])
                                  -> Ty<'tcx>
{
    let existential_bounds = conv_existential_bounds(this,
                                                     rscope,
                                                     span,
                                                     trait_ref.clone(),
                                                     projection_bounds,
                                                     bounds);

    let result = ty::mk_trait(this.tcx(), trait_ref, existential_bounds);
    debug!("trait_ref_to_object_type: result={}",
           result.repr(this.tcx()));

    result
}

fn associated_path_def_to_ty<'tcx>(this: &AstConv<'tcx>,
                                   ast_ty: &ast::Ty,
                                   provenance: def::TyParamProvenance,
                                   assoc_name: ast::Name)
                                   -> Ty<'tcx>
{
    let tcx = this.tcx();
    let ty_param_def_id = provenance.def_id();

    let mut suitable_bounds: Vec<_>;
    let ty_param_name: ast::Name;
    { // contain scope of refcell:
        let ty_param_defs = tcx.ty_param_defs.borrow();
        let ty_param_def = &ty_param_defs[ty_param_def_id.node];
        ty_param_name = ty_param_def.name;

        // FIXME(#20300) -- search where clauses, not bounds
        suitable_bounds =
            traits::transitive_bounds(tcx, &ty_param_def.bounds.trait_bounds)
            .filter(|b| trait_defines_associated_type_named(this, b.def_id(), assoc_name))
            .collect();
    }

    if suitable_bounds.len() == 0 {
        span_err!(tcx.sess, ast_ty.span, E0220,
                          "associated type `{}` not found for type parameter `{}`",
                                  token::get_name(assoc_name),
                                  token::get_name(ty_param_name));
        return this.tcx().types.err;
    }

    if suitable_bounds.len() > 1 {
        span_err!(tcx.sess, ast_ty.span, E0221,
                          "ambiguous associated type `{}` in bounds of `{}`",
                                  token::get_name(assoc_name),
                                  token::get_name(ty_param_name));

        for suitable_bound in &suitable_bounds {
            span_note!(this.tcx().sess, ast_ty.span,
                       "associated type `{}` could derive from `{}`",
                       token::get_name(ty_param_name),
                       suitable_bound.user_string(this.tcx()));
        }
    }

    let suitable_bound = suitable_bounds.pop().unwrap().clone();
    return this.projected_ty_from_poly_trait_ref(ast_ty.span, suitable_bound, assoc_name);
}

fn trait_defines_associated_type_named(this: &AstConv,
                                       trait_def_id: ast::DefId,
                                       assoc_name: ast::Name)
                                       -> bool
{
    let tcx = this.tcx();
    let trait_def = ty::lookup_trait_def(tcx, trait_def_id);
    trait_def.associated_type_names.contains(&assoc_name)
}

fn qpath_to_ty<'tcx>(this: &AstConv<'tcx>,
                     rscope: &RegionScope,
                     ast_ty: &ast::Ty, // the TyQPath
                     qpath: &ast::QPath)
                     -> Ty<'tcx>
{
    debug!("qpath_to_ty(ast_ty={})",
           ast_ty.repr(this.tcx()));

    let self_type = ast_ty_to_ty(this, rscope, &*qpath.self_type);

    debug!("qpath_to_ty: self_type={}", self_type.repr(this.tcx()));

    let trait_ref = instantiate_trait_ref(this,
                                          rscope,
                                          &*qpath.trait_ref,
                                          Some(self_type),
                                          None);

    debug!("qpath_to_ty: trait_ref={}", trait_ref.repr(this.tcx()));

    // `<T as Trait>::U<V>` shouldn't parse right now.
    assert!(qpath.item_path.parameters.is_empty());

    return this.projected_ty(ast_ty.span,
                             trait_ref,
                             qpath.item_path.identifier.name);
}

/// Convert a type supplied as value for a type argument from AST into our
/// our internal representation. This is the same as `ast_ty_to_ty` but that
/// it applies the object lifetime default.
///
/// # Parameters
///
/// * `this`, `rscope`: the surrounding context
/// * `decl_generics`: the generics of the struct/enum/trait declaration being
///   referenced
/// * `index`: the index of the type parameter being instantiated from the list
///   (we assume it is in the `TypeSpace`)
/// * `region_substs`: a partial substitution consisting of
///   only the region type parameters being supplied to this type.
/// * `ast_ty`: the ast representation of the type being supplied
pub fn ast_ty_arg_to_ty<'tcx>(this: &AstConv<'tcx>,
                              rscope: &RegionScope,
                              decl_generics: &ty::Generics<'tcx>,
                              index: usize,
                              region_substs: &Substs<'tcx>,
                              ast_ty: &ast::Ty)
                              -> Ty<'tcx>
{
    let tcx = this.tcx();

    if let Some(def) = decl_generics.types.opt_get(TypeSpace, index) {
        let object_lifetime_default = def.object_lifetime_default.subst(tcx, region_substs);
        let rscope1 = &ObjectLifetimeDefaultRscope::new(rscope, object_lifetime_default);
        ast_ty_to_ty(this, rscope1, ast_ty)
    } else {
        ast_ty_to_ty(this, rscope, ast_ty)
    }
}

/// Parses the programmer's textual representation of a type into our
/// internal notion of a type.
pub fn ast_ty_to_ty<'tcx>(this: &AstConv<'tcx>,
                          rscope: &RegionScope,
                          ast_ty: &ast::Ty)
                          -> Ty<'tcx>
{
    debug!("ast_ty_to_ty(ast_ty={})",
           ast_ty.repr(this.tcx()));

    let tcx = this.tcx();

    let mut ast_ty_to_ty_cache = tcx.ast_ty_to_ty_cache.borrow_mut();
    match ast_ty_to_ty_cache.get(&ast_ty.id) {
        Some(&ty::atttce_resolved(ty)) => return ty,
        Some(&ty::atttce_unresolved) => {
            span_fatal!(tcx.sess, ast_ty.span, E0246,
                                "illegal recursive type; insert an enum \
                                 or struct in the cycle, if this is \
                                 desired");
        }
        None => { /* go on */ }
    }
    ast_ty_to_ty_cache.insert(ast_ty.id, ty::atttce_unresolved);
    drop(ast_ty_to_ty_cache);

    let typ = ast_ty_to_builtin_ty(this, rscope, ast_ty).unwrap_or_else(|| {
        match ast_ty.node {
            ast::TyVec(ref ty) => {
                ty::mk_vec(tcx, ast_ty_to_ty(this, rscope, &**ty), None)
            }
            ast::TyObjectSum(ref ty, ref bounds) => {
                match ast_ty_to_trait_ref(this, rscope, &**ty, &bounds[..]) {
                    Ok((trait_ref, projection_bounds)) => {
                        trait_ref_to_object_type(this,
                                                 rscope,
                                                 ast_ty.span,
                                                 trait_ref,
                                                 projection_bounds,
                                                 &bounds[..])
                    }
                    Err(ErrorReported) => {
                        this.tcx().types.err
                    }
                }
            }
            ast::TyPtr(ref mt) => {
                ty::mk_ptr(tcx, ty::mt {
                    ty: ast_ty_to_ty(this, rscope, &*mt.ty),
                    mutbl: mt.mutbl
                })
            }
            ast::TyRptr(ref region, ref mt) => {
                let r = opt_ast_region_to_region(this, rscope, ast_ty.span, region);
                debug!("ty_rptr r={}", r.repr(this.tcx()));
                let rscope1 =
                    &ObjectLifetimeDefaultRscope::new(
                        rscope,
                        Some(ty::ObjectLifetimeDefault::Specific(r)));
                let t = ast_ty_to_ty(this, rscope1, &*mt.ty);
                ty::mk_rptr(tcx, tcx.mk_region(r), ty::mt {ty: t, mutbl: mt.mutbl})
            }
            ast::TyTup(ref fields) => {
                let flds = fields.iter()
                                 .map(|t| ast_ty_to_ty(this, rscope, &**t))
                                 .collect();
                ty::mk_tup(tcx, flds)
            }
            ast::TyParen(ref typ) => ast_ty_to_ty(this, rscope, &**typ),
            ast::TyBareFn(ref bf) => {
                if bf.decl.variadic && bf.abi != abi::C {
                    span_err!(tcx.sess, ast_ty.span, E0222,
                                      "variadic function must have C calling convention");
                }
                let bare_fn = ty_of_bare_fn(this, bf.unsafety, bf.abi, &*bf.decl);
                ty::mk_bare_fn(tcx, None, tcx.mk_bare_fn(bare_fn))
            }
            ast::TyPolyTraitRef(ref bounds) => {
                conv_ty_poly_trait_ref(this, rscope, ast_ty.span, &bounds[..])
            }
            ast::TyPath(ref path, id) => {
                let a_def = match tcx.def_map.borrow().get(&id) {
                    None => {
                        tcx.sess
                           .span_bug(ast_ty.span,
                                     &format!("unbound path {}",
                                             path.repr(tcx))[])
                    }
                    Some(&d) => d
                };
                match a_def {
                    def::DefTrait(trait_def_id) => {
                        // N.B. this case overlaps somewhat with
                        // TyObjectSum, see that fn for details
                        let mut projection_bounds = Vec::new();

                        let trait_ref = object_path_to_poly_trait_ref(this,
                                                                      rscope,
                                                                      trait_def_id,
                                                                      path,
                                                                      &mut projection_bounds);

                        trait_ref_to_object_type(this, rscope, path.span,
                                                 trait_ref, projection_bounds, &[])
                    }
                    def::DefTy(did, _) | def::DefStruct(did) => {
                        ast_path_to_ty(this, rscope, did, path).ty
                    }
                    def::DefTyParam(space, index, _, name) => {
                        check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                        ty::mk_param(tcx, space, index, name)
                    }
                    def::DefSelfTy(_) => {
                        // n.b.: resolve guarantees that the this type only appears in a
                        // trait, which we rely upon in various places when creating
                        // substs
                        check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                        ty::mk_self_type(tcx)
                    }
                    def::DefMod(id) => {
                        span_fatal!(tcx.sess, ast_ty.span, E0247,
                            "found module name used as a type: {}",
                                    tcx.map.node_to_string(id.node));
                    }
                    def::DefPrimTy(_) => {
                        panic!("DefPrimTy arm missed in previous ast_ty_to_prim_ty call");
                    }
                    def::DefAssociatedTy(trait_type_id) => {
                        let path_str = tcx.map.path_to_string(
                            tcx.map.get_parent(trait_type_id.node));
                        span_err!(tcx.sess, ast_ty.span, E0223,
                                          "ambiguous associated \
                                                   type; specify the type \
                                                   using the syntax `<Type \
                                                   as {}>::{}`",
                                                  path_str,
                                                  &token::get_ident(
                                                      path.segments
                                                          .last()
                                                          .unwrap()
                                                          .identifier));
                        this.tcx().types.err
                    }
                    def::DefAssociatedPath(provenance, assoc_ident) => {
                        associated_path_def_to_ty(this, ast_ty, provenance, assoc_ident.name)
                    }
                    _ => {
                        span_fatal!(tcx.sess, ast_ty.span, E0248,
                                            "found value name used \
                                                     as a type: {:?}",
                                                    a_def);
                    }
                }
            }
            ast::TyQPath(ref qpath) => {
                qpath_to_ty(this, rscope, ast_ty, &**qpath)
            }
            ast::TyFixedLengthVec(ref ty, ref e) => {
                match const_eval::eval_const_expr_partial(tcx, &**e, Some(tcx.types.uint)) {
                    Ok(ref r) => {
                        match *r {
                            const_eval::const_int(i) =>
                                ty::mk_vec(tcx, ast_ty_to_ty(this, rscope, &**ty),
                                           Some(i as uint)),
                            const_eval::const_uint(i) =>
                                ty::mk_vec(tcx, ast_ty_to_ty(this, rscope, &**ty),
                                           Some(i as uint)),
                            _ => {
                                span_fatal!(tcx.sess, ast_ty.span, E0249,
                                            "expected constant expr for array length");
                            }
                        }
                    }
                    Err(ref r) => {
                        span_fatal!(tcx.sess, ast_ty.span, E0250,
                            "expected constant expr for array \
                                     length: {}",
                                    *r);
                    }
                }
            }
            ast::TyTypeof(ref _e) => {
                tcx.sess.span_bug(ast_ty.span, "typeof is reserved but unimplemented");
            }
            ast::TyInfer => {
                // TyInfer also appears as the type of arguments or return
                // values in a ExprClosure, or as
                // the type of local variables. Both of these cases are
                // handled specially and will not descend into this routine.
                this.ty_infer(ast_ty.span)
            }
        }
    });

    tcx.ast_ty_to_ty_cache.borrow_mut().insert(ast_ty.id, ty::atttce_resolved(typ));
    return typ;
}

pub fn ty_of_arg<'tcx>(this: &AstConv<'tcx>,
                       rscope: &RegionScope,
                       a: &ast::Arg,
                       expected_ty: Option<Ty<'tcx>>)
                       -> Ty<'tcx>
{
    match a.ty.node {
        ast::TyInfer if expected_ty.is_some() => expected_ty.unwrap(),
        ast::TyInfer => this.ty_infer(a.ty.span),
        _ => ast_ty_to_ty(this, rscope, &*a.ty),
    }
}

struct SelfInfo<'a, 'tcx> {
    untransformed_self_ty: Ty<'tcx>,
    explicit_self: &'a ast::ExplicitSelf,
}

pub fn ty_of_method<'tcx>(this: &AstConv<'tcx>,
                          unsafety: ast::Unsafety,
                          untransformed_self_ty: Ty<'tcx>,
                          explicit_self: &ast::ExplicitSelf,
                          decl: &ast::FnDecl,
                          abi: abi::Abi)
                          -> (ty::BareFnTy<'tcx>, ty::ExplicitSelfCategory) {
    let self_info = Some(SelfInfo {
        untransformed_self_ty: untransformed_self_ty,
        explicit_self: explicit_self,
    });
    let (bare_fn_ty, optional_explicit_self_category) =
        ty_of_method_or_bare_fn(this,
                                unsafety,
                                abi,
                                self_info,
                                decl);
    (bare_fn_ty, optional_explicit_self_category.unwrap())
}

pub fn ty_of_bare_fn<'tcx>(this: &AstConv<'tcx>, unsafety: ast::Unsafety, abi: abi::Abi,
                                              decl: &ast::FnDecl) -> ty::BareFnTy<'tcx> {
    let (bare_fn_ty, _) = ty_of_method_or_bare_fn(this, unsafety, abi, None, decl);
    bare_fn_ty
}

fn ty_of_method_or_bare_fn<'a, 'tcx>(this: &AstConv<'tcx>,
                                     unsafety: ast::Unsafety,
                                     abi: abi::Abi,
                                     opt_self_info: Option<SelfInfo<'a, 'tcx>>,
                                     decl: &ast::FnDecl)
                                     -> (ty::BareFnTy<'tcx>, Option<ty::ExplicitSelfCategory>)
{
    debug!("ty_of_method_or_bare_fn");

    // New region names that appear inside of the arguments of the function
    // declaration are bound to that function type.
    let rb = rscope::BindingRscope::new();

    // `implied_output_region` is the region that will be assumed for any
    // region parameters in the return type. In accordance with the rules for
    // lifetime elision, we can determine it in two ways. First (determined
    // here), if self is by-reference, then the implied output region is the
    // region of the self parameter.
    let mut explicit_self_category_result = None;
    let (self_ty, mut implied_output_region) = match opt_self_info {
        None => (None, None),
        Some(self_info) => {
            // This type comes from an impl or trait; no late-bound
            // regions should be present.
            assert!(!self_info.untransformed_self_ty.has_escaping_regions());

            // Figure out and record the explicit self category.
            let explicit_self_category =
                determine_explicit_self_category(this, &rb, &self_info);
            explicit_self_category_result = Some(explicit_self_category);
            match explicit_self_category {
                ty::StaticExplicitSelfCategory => {
                    (None, None)
                }
                ty::ByValueExplicitSelfCategory => {
                    (Some(self_info.untransformed_self_ty), None)
                }
                ty::ByReferenceExplicitSelfCategory(region, mutability) => {
                    (Some(ty::mk_rptr(this.tcx(),
                                      this.tcx().mk_region(region),
                                      ty::mt {
                                        ty: self_info.untransformed_self_ty,
                                        mutbl: mutability
                                      })),
                     Some(region))
                }
                ty::ByBoxExplicitSelfCategory => {
                    (Some(ty::mk_uniq(this.tcx(), self_info.untransformed_self_ty)), None)
                }
            }
        }
    };

    // HACK(eddyb) replace the fake self type in the AST with the actual type.
    let input_params = if self_ty.is_some() {
        &decl.inputs[1..]
    } else {
        &decl.inputs[]
    };
    let input_tys = input_params.iter().map(|a| ty_of_arg(this, &rb, a, None));
    let input_pats: Vec<String> = input_params.iter()
                                              .map(|a| pprust::pat_to_string(&*a.pat))
                                              .collect();
    let self_and_input_tys: Vec<Ty> =
        self_ty.into_iter().chain(input_tys).collect();


    // Second, if there was exactly one lifetime (either a substitution or a
    // reference) in the arguments, then any anonymous regions in the output
    // have that lifetime.
    let lifetimes_for_params = if implied_output_region.is_none() {
        let input_tys = if self_ty.is_some() {
            // Skip the first argument if `self` is present.
            &self_and_input_tys[1..]
        } else {
            &self_and_input_tys[..]
        };

        let (ior, lfp) = find_implied_output_region(input_tys, input_pats);
        implied_output_region = ior;
        lfp
    } else {
        vec![]
    };

    let output_ty = match decl.output {
        ast::Return(ref output) if output.node == ast::TyInfer =>
            ty::FnConverging(this.ty_infer(output.span)),
        ast::Return(ref output) =>
            ty::FnConverging(convert_ty_with_lifetime_elision(this,
                                                              implied_output_region,
                                                              lifetimes_for_params,
                                                              &**output)),
        ast::DefaultReturn(..) => ty::FnConverging(ty::mk_nil(this.tcx())),
        ast::NoReturn(..) => ty::FnDiverging
    };

    (ty::BareFnTy {
        unsafety: unsafety,
        abi: abi,
        sig: ty::Binder(ty::FnSig {
            inputs: self_and_input_tys,
            output: output_ty,
            variadic: decl.variadic
        }),
    }, explicit_self_category_result)
}

fn determine_explicit_self_category<'a, 'tcx>(this: &AstConv<'tcx>,
                                              rscope: &RegionScope,
                                              self_info: &SelfInfo<'a, 'tcx>)
                                              -> ty::ExplicitSelfCategory
{
    return match self_info.explicit_self.node {
        ast::SelfStatic => ty::StaticExplicitSelfCategory,
        ast::SelfValue(_) => ty::ByValueExplicitSelfCategory,
        ast::SelfRegion(ref lifetime, mutability, _) => {
            let region =
                opt_ast_region_to_region(this,
                                         rscope,
                                         self_info.explicit_self.span,
                                         lifetime);
            ty::ByReferenceExplicitSelfCategory(region, mutability)
        }
        ast::SelfExplicit(ref ast_type, _) => {
            let explicit_type = ast_ty_to_ty(this, rscope, &**ast_type);

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
            //     fn method1(self: &&T); // ByReferenceExplicitSelfCategory
            //     fn method2(self: &T); // ByValueExplicitSelfCategory
            //     fn method3(self: Box<&T>); // ByBoxExplicitSelfCategory
            //
            //     // Invalid cases will be caught later by `check_method_self_type`:
            //     fn method_err1(self: &mut T); // ByReferenceExplicitSelfCategory
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
            // ByReferenceExplicitSelfCategory.

            let impl_modifiers = count_modifiers(self_info.untransformed_self_ty);
            let method_modifiers = count_modifiers(explicit_type);

            debug!("determine_explicit_self_category(self_info.untransformed_self_ty={} \
                   explicit_type={} \
                   modifiers=({},{})",
                   self_info.untransformed_self_ty.repr(this.tcx()),
                   explicit_type.repr(this.tcx()),
                   impl_modifiers,
                   method_modifiers);

            if impl_modifiers >= method_modifiers {
                ty::ByValueExplicitSelfCategory
            } else {
                match explicit_type.sty {
                    ty::ty_rptr(r, mt) => ty::ByReferenceExplicitSelfCategory(*r, mt.mutbl),
                    ty::ty_uniq(_) => ty::ByBoxExplicitSelfCategory,
                    _ => ty::ByValueExplicitSelfCategory,
                }
            }
        }
    };

    fn count_modifiers(ty: Ty) -> uint {
        match ty.sty {
            ty::ty_rptr(_, mt) => count_modifiers(mt.ty) + 1,
            ty::ty_uniq(t) => count_modifiers(t) + 1,
            _ => 0,
        }
    }
}

pub fn ty_of_closure<'tcx>(
    this: &AstConv<'tcx>,
    unsafety: ast::Unsafety,
    decl: &ast::FnDecl,
    abi: abi::Abi,
    expected_sig: Option<ty::FnSig<'tcx>>)
    -> ty::ClosureTy<'tcx>
{
    debug!("ty_of_closure(expected_sig={})",
           expected_sig.repr(this.tcx()));

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
        ty_of_arg(this, &rb, a, expected_arg_ty)
    }).collect();

    let expected_ret_ty = expected_sig.map(|e| e.output);

    let is_infer = match decl.output {
        ast::Return(ref output) if output.node == ast::TyInfer => true,
        ast::DefaultReturn(..) => true,
        _ => false
    };

    let output_ty = match decl.output {
        _ if is_infer && expected_ret_ty.is_some() =>
            expected_ret_ty.unwrap(),
        _ if is_infer =>
            ty::FnConverging(this.ty_infer(decl.output.span())),
        ast::Return(ref output) =>
            ty::FnConverging(ast_ty_to_ty(this, &rb, &**output)),
        ast::DefaultReturn(..) => unreachable!(),
        ast::NoReturn(..) => ty::FnDiverging
    };

    debug!("ty_of_closure: input_tys={}", input_tys.repr(this.tcx()));
    debug!("ty_of_closure: output_ty={}", output_ty.repr(this.tcx()));

    ty::ClosureTy {
        unsafety: unsafety,
        abi: abi,
        sig: ty::Binder(ty::FnSig {inputs: input_tys,
                                   output: output_ty,
                                   variadic: decl.variadic}),
    }
}

/// Given an existential type like `Foo+'a+Bar`, this routine converts the `'a` and `Bar` intos an
/// `ExistentialBounds` struct. The `main_trait_refs` argument specifies the `Foo` -- it is absent
/// for closures. Eventually this should all be normalized, I think, so that there is no "main
/// trait ref" and instead we just have a flat list of bounds as the existential type.
fn conv_existential_bounds<'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    span: Span,
    principal_trait_ref: ty::PolyTraitRef<'tcx>,
    projection_bounds: Vec<ty::PolyProjectionPredicate<'tcx>>,
    ast_bounds: &[ast::TyParamBound])
    -> ty::ExistentialBounds<'tcx>
{
    let partitioned_bounds =
        partition_bounds(this.tcx(), span, ast_bounds);

    conv_existential_bounds_from_partitioned_bounds(
        this, rscope, span, principal_trait_ref, projection_bounds, partitioned_bounds)
}

fn conv_ty_poly_trait_ref<'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    span: Span,
    ast_bounds: &[ast::TyParamBound])
    -> Ty<'tcx>
{
    let mut partitioned_bounds = partition_bounds(this.tcx(), span, &ast_bounds[..]);

    let mut projection_bounds = Vec::new();
    let main_trait_bound = if !partitioned_bounds.trait_bounds.is_empty() {
        let trait_bound = partitioned_bounds.trait_bounds.remove(0);
        instantiate_poly_trait_ref(this,
                                   rscope,
                                   trait_bound,
                                   None,
                                   &mut projection_bounds)
    } else {
        span_err!(this.tcx().sess, span, E0224,
                  "at least one non-builtin trait is required for an object type");
        return this.tcx().types.err;
    };

    let bounds =
        conv_existential_bounds_from_partitioned_bounds(this,
                                                        rscope,
                                                        span,
                                                        main_trait_bound.clone(),
                                                        projection_bounds,
                                                        partitioned_bounds);

    ty::mk_trait(this.tcx(), main_trait_bound, bounds)
}

pub fn conv_existential_bounds_from_partitioned_bounds<'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    span: Span,
    principal_trait_ref: ty::PolyTraitRef<'tcx>,
    mut projection_bounds: Vec<ty::PolyProjectionPredicate<'tcx>>, // Empty for boxed closures
    partitioned_bounds: PartitionedBounds)
    -> ty::ExistentialBounds<'tcx>
{
    let PartitionedBounds { builtin_bounds,
                            trait_bounds,
                            region_bounds } =
        partitioned_bounds;

    if !trait_bounds.is_empty() {
        let b = &trait_bounds[0];
        span_err!(this.tcx().sess, b.trait_ref.path.span, E0225,
                  "only the builtin traits can be used as closure or object bounds");
    }

    let region_bound = compute_object_lifetime_bound(this,
                                                     rscope,
                                                     span,
                                                     &region_bounds,
                                                     principal_trait_ref,
                                                     builtin_bounds);

    ty::sort_bounds_list(&mut projection_bounds);

    ty::ExistentialBounds {
        region_bound: region_bound,
        builtin_bounds: builtin_bounds,
        projection_bounds: projection_bounds,
    }
}

/// Given the bounds on an object, determines what single region bound
/// (if any) we can use to summarize this type. The basic idea is that we will use the bound the
/// user provided, if they provided one, and otherwise search the supertypes of trait bounds for
/// region bounds. It may be that we can derive no bound at all, in which case we return `None`.
fn compute_object_lifetime_bound<'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    span: Span,
    explicit_region_bounds: &[&ast::Lifetime],
    principal_trait_ref: ty::PolyTraitRef<'tcx>,
    builtin_bounds: ty::BuiltinBounds)
    -> ty::Region
{
    let tcx = this.tcx();

    debug!("compute_opt_region_bound(explicit_region_bounds={:?}, \
           principal_trait_ref={}, builtin_bounds={})",
           explicit_region_bounds,
           principal_trait_ref.repr(tcx),
           builtin_bounds.repr(tcx));

    if explicit_region_bounds.len() > 1 {
        span_err!(tcx.sess, explicit_region_bounds[1].span, E0226,
            "only a single explicit lifetime bound is permitted");
    }

    if explicit_region_bounds.len() != 0 {
        // Explicitly specified region bound. Use that.
        let r = explicit_region_bounds[0];
        return ast_region_to_region(tcx, r);
    }

    // No explicit region bound specified. Therefore, examine trait
    // bounds and see if we can derive region bounds from those.
    let derived_region_bounds =
        object_region_bounds(tcx, &principal_trait_ref, builtin_bounds);

    // If there are no derived region bounds, then report back that we
    // can find no region bound.
    if derived_region_bounds.len() == 0 {
        match rscope.object_lifetime_default(span) {
            Some(r) => { return r; }
            None => {
                span_err!(this.tcx().sess, span, E0228,
                          "the lifetime bound for this object type cannot be deduced \
                           from context; please supply an explicit bound");
                return ty::ReStatic;
            }
        }
    }

    // If any of the derived region bounds are 'static, that is always
    // the best choice.
    if derived_region_bounds.iter().any(|r| ty::ReStatic == *r) {
        return ty::ReStatic;
    }

    // Determine whether there is exactly one unique region in the set
    // of derived region bounds. If so, use that. Otherwise, report an
    // error.
    let r = derived_region_bounds[0];
    if derived_region_bounds[1..].iter().any(|r1| r != *r1) {
        span_err!(tcx.sess, span, E0227,
                  "ambiguous lifetime bound, explicit lifetime bound required");
    }
    return r;
}

/// Given an object type like `SomeTrait+Send`, computes the lifetime
/// bounds that must hold on the elided self type. These are derived
/// from the declarations of `SomeTrait`, `Send`, and friends -- if
/// they declare `trait SomeTrait : 'static`, for example, then
/// `'static` would appear in the list. The hard work is done by
/// `ty::required_region_bounds`, see that for more information.
pub fn object_region_bounds<'tcx>(
    tcx: &ty::ctxt<'tcx>,
    principal: &ty::PolyTraitRef<'tcx>,
    others: ty::BuiltinBounds)
    -> Vec<ty::Region>
{
    // Since we don't actually *know* the self type for an object,
    // this "open(err)" serves as a kind of dummy standin -- basically
    // a skolemized type.
    let open_ty = ty::mk_infer(tcx, ty::FreshTy(0));

    // Note that we preserve the overall binding levels here.
    assert!(!open_ty.has_escaping_regions());
    let substs = tcx.mk_substs(principal.0.substs.with_self_ty(open_ty));
    let trait_refs = vec!(ty::Binder(Rc::new(ty::TraitRef::new(principal.0.def_id, substs))));

    let param_bounds = ty::ParamBounds {
        region_bounds: Vec::new(),
        builtin_bounds: others,
        trait_bounds: trait_refs,
        projection_bounds: Vec::new(), // not relevant to computing region bounds
    };

    let predicates = ty::predicates(tcx, open_ty, &param_bounds);
    ty::required_region_bounds(tcx, open_ty, predicates)
}

pub struct PartitionedBounds<'a> {
    pub builtin_bounds: ty::BuiltinBounds,
    pub trait_bounds: Vec<&'a ast::PolyTraitRef>,
    pub region_bounds: Vec<&'a ast::Lifetime>,
}

/// Divides a list of bounds from the AST into three groups: builtin bounds (Copy, Sized etc),
/// general trait bounds, and region bounds.
pub fn partition_bounds<'a>(tcx: &ty::ctxt,
                            _span: Span,
                            ast_bounds: &'a [ast::TyParamBound])
                            -> PartitionedBounds<'a>
{
    let mut builtin_bounds = ty::empty_builtin_bounds();
    let mut region_bounds = Vec::new();
    let mut trait_bounds = Vec::new();
    let mut trait_def_ids = DefIdMap();
    for ast_bound in ast_bounds {
        match *ast_bound {
            ast::TraitTyParamBound(ref b, ast::TraitBoundModifier::None) => {
                match ::lookup_def_tcx(tcx, b.trait_ref.path.span, b.trait_ref.ref_id) {
                    def::DefTrait(trait_did) => {
                        match trait_def_ids.get(&trait_did) {
                            // Already seen this trait. We forbid
                            // duplicates in the list (for some
                            // reason).
                            Some(span) => {
                                span_err!(
                                    tcx.sess, b.trait_ref.path.span, E0127,
                                    "trait `{}` already appears in the \
                                     list of bounds",
                                    b.trait_ref.path.user_string(tcx));
                                tcx.sess.span_note(
                                    *span,
                                    "previous appearance is here");

                                continue;
                            }

                            None => { }
                        }

                        trait_def_ids.insert(trait_did, b.trait_ref.path.span);

                        if ty::try_add_builtin_trait(tcx,
                                                     trait_did,
                                                     &mut builtin_bounds) {
                            // FIXME(#20302) -- we should check for things like Copy<T>
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
            ast::TraitTyParamBound(_, ast::TraitBoundModifier::Maybe) => {}
            ast::RegionTyParamBound(ref l) => {
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

fn prohibit_projections<'tcx>(tcx: &ty::ctxt<'tcx>,
                              bindings: &[ConvertedBinding<'tcx>])
{
    for binding in bindings.iter().take(1) {
        span_err!(tcx.sess, binding.span, E0229,
            "associated type bindings are not allowed here");
    }
}
