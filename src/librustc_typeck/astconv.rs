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

use middle::astconv_util::{prim_ty_to_ty, check_path_args, NO_TPS, NO_REGIONS};
use middle::const_eval;
use middle::def;
use middle::implicator::object_region_bounds;
use middle::resolve_lifetime as rl;
use middle::privacy::{AllPublic, LastMod};
use middle::subst::{FnSpace, TypeSpace, SelfSpace, Subst, Substs};
use middle::traits;
use middle::ty::{self, RegionEscape, Ty};
use rscope::{self, UnelidableRscope, RegionScope, ElidableRscope, ExplicitRscope,
             ObjectLifetimeDefaultRscope, ShiftedRscope, BindingRscope};
use util::common::{ErrorReported, FN_OUTPUT_NAME};
use util::nodemap::FnvHashSet;
use util::ppaux::{self, Repr, UserString};

use std::iter::repeat;
use std::rc::Rc;
use std::slice;
use syntax::{abi, ast, ast_util};
use syntax::codemap::{Span, Pos};
use syntax::parse::token;
use syntax::print::pprust;

pub trait AstConv<'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx>;

    /// Identify the type scheme for an item with a type, like a type
    /// alias, fn, or struct. This allows you to figure out the set of
    /// type parameters defined on the item.
    fn get_item_type_scheme(&self, span: Span, id: ast::DefId)
                            -> Result<ty::TypeScheme<'tcx>, ErrorReported>;

    /// Returns the `TraitDef` for a given trait. This allows you to
    /// figure out the set of type parameters defined on the trait.
    fn get_trait_def(&self, span: Span, id: ast::DefId)
                     -> Result<Rc<ty::TraitDef<'tcx>>, ErrorReported>;

    /// Ensure that the super-predicates for the trait with the given
    /// id are available and also for the transitive set of
    /// super-predicates.
    fn ensure_super_predicates(&self, span: Span, id: ast::DefId)
                               -> Result<(), ErrorReported>;

    /// Returns the set of bounds in scope for the type parameter with
    /// the given id.
    fn get_type_parameter_bounds(&self, span: Span, def_id: ast::NodeId)
                                 -> Result<Vec<ty::PolyTraitRef<'tcx>>, ErrorReported>;

    /// Returns true if the trait with id `trait_def_id` defines an
    /// associated type with the name `name`.
    fn trait_defines_associated_type_named(&self, trait_def_id: ast::DefId, name: ast::Name)
                                           -> bool;

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
                    -> Ty<'tcx>;
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
            ty::ReEarlyBound(ty::EarlyBoundRegion {
                param_id: id,
                space: space,
                index: index,
                name: lifetime.name
            })
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
                                })[..]);

                                if len == 2 && i == 0 {
                                    m.push_str(" or ");
                                } else if i + 2 == len {
                                    m.push_str(", or ");
                                } else if i + 1 != len {
                                    m.push_str(", ");
                                }
                            }
                            if len == 1 {
                                fileline_help!(this.tcx().sess, default_span,
                                    "this function's return type contains a borrowed value, but \
                                     the signature does not say which {} it is borrowed from",
                                    m);
                            } else if len == 0 {
                                fileline_help!(this.tcx().sess, default_span,
                                    "this function's return type contains a borrowed value, but \
                                     there is no value for it to be borrowed from");
                                fileline_help!(this.tcx().sess, default_span,
                                    "consider giving it a 'static lifetime");
                            } else {
                                fileline_help!(this.tcx().sess, default_span,
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
    span: Span,
    param_mode: PathParamMode,
    decl_generics: &ty::Generics<'tcx>,
    item_segment: &ast::PathSegment)
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

    let (regions, types, assoc_bindings) = match item_segment.parameters {
        ast::AngleBracketedParameters(ref data) => {
            convert_angle_bracketed_parameters(this, rscope, span, decl_generics, data)
        }
        ast::ParenthesizedParameters(ref data) => {
            span_err!(tcx.sess, span, E0214,
                "parenthesized parameters may only be used with a trait");
            convert_parenthesized_parameters(this, rscope, span, decl_generics, data)
        }
    };

    prohibit_projections(this.tcx(), &assoc_bindings);

    create_substs_for_ast_path(this,
                               span,
                               param_mode,
                               decl_generics,
                               None,
                               types,
                               regions)
}

#[derive(PartialEq, Eq)]
pub enum PathParamMode {
    // Any path in a type context.
    Explicit,
    // The `module::Type` in `module::Type::method` in an expression.
    Optional
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
            report_lifetime_number_error(tcx, span,
                                         supplied_num_region_params,
                                         expected_num_region_params);
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
    param_mode: PathParamMode,
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
    let formal_ty_param_count = ty_param_defs.len();
    let required_ty_param_count = ty_param_defs.iter()
                                               .take_while(|x| x.default.is_none())
                                               .count();

    // Fill with `ty_infer` if no params were specified, as long as
    // they were optional (e.g. paths inside expressions).
    let mut type_substs = if param_mode == PathParamMode::Optional &&
                             types_provided.is_empty() {
        (0..formal_ty_param_count).map(|_| this.ty_infer(span)).collect()
    } else {
        types_provided
    };

    let supplied_ty_param_count = type_substs.len();
    check_type_argument_count(this.tcx(), span, supplied_ty_param_count,
                              required_ty_param_count, formal_ty_param_count);

    if supplied_ty_param_count < required_ty_param_count {
        while type_substs.len() < required_ty_param_count {
            type_substs.push(tcx.types.err);
        }
    } else if supplied_ty_param_count > formal_ty_param_count {
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

    substs
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
                              -> (Option<ty::Region>, Vec<(String, usize)>)
{
    let mut lifetimes_for_params: Vec<(String, usize)> = Vec::new();
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

    let implied_output_region =
        if lifetimes_for_params.iter().map(|&(_, n)| n).sum::<usize>() == 1 {
            assert!(possible_implied_output_region.is_some());
            possible_implied_output_region
        } else {
            None
        };
    (implied_output_region, lifetimes_for_params)
}

fn convert_ty_with_lifetime_elision<'tcx>(this: &AstConv<'tcx>,
                                          implied_output_region: Option<ty::Region>,
                                          param_lifetimes: Vec<(String, usize)>,
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
    let trait_ref = &ast_trait_ref.trait_ref;
    let trait_def_id = trait_def_id(this, trait_ref);
    ast_path_to_poly_trait_ref(this,
                               rscope,
                               trait_ref.path.span,
                               PathParamMode::Explicit,
                               trait_def_id,
                               self_ty,
                               trait_ref.path.segments.last().unwrap(),
                               poly_projections)
}

/// Instantiates the path for the given trait reference, assuming that it's
/// bound to a valid trait type. Returns the def_id for the defining trait.
/// Fails if the type is a type other than a trait type.
///
/// If the `projections` argument is `None`, then assoc type bindings like `Foo<T=X>`
/// are disallowed. Otherwise, they are pushed onto the vector given.
pub fn instantiate_mono_trait_ref<'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    trait_ref: &ast::TraitRef,
    self_ty: Option<Ty<'tcx>>)
    -> Rc<ty::TraitRef<'tcx>>
{
    let trait_def_id = trait_def_id(this, trait_ref);
    ast_path_to_mono_trait_ref(this,
                               rscope,
                               trait_ref.path.span,
                               PathParamMode::Explicit,
                               trait_def_id,
                               self_ty,
                               trait_ref.path.segments.last().unwrap())
}

fn trait_def_id<'tcx>(this: &AstConv<'tcx>, trait_ref: &ast::TraitRef) -> ast::DefId {
    let path = &trait_ref.path;
    match ::lookup_full_def(this.tcx(), path.span, trait_ref.ref_id) {
        def::DefTrait(trait_def_id) => trait_def_id,
        _ => {
            span_fatal!(this.tcx().sess, path.span, E0245, "`{}` is not a trait",
                        path.user_string(this.tcx()));
        }
    }
}

fn object_path_to_poly_trait_ref<'a,'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    span: Span,
    param_mode: PathParamMode,
    trait_def_id: ast::DefId,
    trait_segment: &ast::PathSegment,
    mut projections: &mut Vec<ty::PolyProjectionPredicate<'tcx>>)
    -> ty::PolyTraitRef<'tcx>
{
    ast_path_to_poly_trait_ref(this,
                               rscope,
                               span,
                               param_mode,
                               trait_def_id,
                               None,
                               trait_segment,
                               projections)
}

fn ast_path_to_poly_trait_ref<'a,'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    span: Span,
    param_mode: PathParamMode,
    trait_def_id: ast::DefId,
    self_ty: Option<Ty<'tcx>>,
    trait_segment: &ast::PathSegment,
    poly_projections: &mut Vec<ty::PolyProjectionPredicate<'tcx>>)
    -> ty::PolyTraitRef<'tcx>
{
    // The trait reference introduces a binding level here, so
    // we need to shift the `rscope`. It'd be nice if we could
    // do away with this rscope stuff and work this knowledge
    // into resolve_lifetimes, as we do with non-omitted
    // lifetimes. Oh well, not there yet.
    let shifted_rscope = &ShiftedRscope::new(rscope);

    let (substs, assoc_bindings) =
        create_substs_for_ast_trait_ref(this,
                                        shifted_rscope,
                                        span,
                                        param_mode,
                                        trait_def_id,
                                        self_ty,
                                        trait_segment);
    let poly_trait_ref = ty::Binder(Rc::new(ty::TraitRef::new(trait_def_id, substs)));

    {
        let converted_bindings =
            assoc_bindings
            .iter()
            .filter_map(|binding| {
                // specify type to assert that error was already reported in Err case:
                let predicate: Result<_, ErrorReported> =
                    ast_type_binding_to_poly_projection_predicate(this,
                                                                  poly_trait_ref.clone(),
                                                                  self_ty,
                                                                  binding);
                predicate.ok() // ok to ignore Err() because ErrorReported (see above)
            });
        poly_projections.extend(converted_bindings);
    }

    poly_trait_ref
}

fn ast_path_to_mono_trait_ref<'a,'tcx>(this: &AstConv<'tcx>,
                                       rscope: &RegionScope,
                                       span: Span,
                                       param_mode: PathParamMode,
                                       trait_def_id: ast::DefId,
                                       self_ty: Option<Ty<'tcx>>,
                                       trait_segment: &ast::PathSegment)
                                       -> Rc<ty::TraitRef<'tcx>>
{
    let (substs, assoc_bindings) =
        create_substs_for_ast_trait_ref(this,
                                        rscope,
                                        span,
                                        param_mode,
                                        trait_def_id,
                                        self_ty,
                                        trait_segment);
    prohibit_projections(this.tcx(), &assoc_bindings);
    Rc::new(ty::TraitRef::new(trait_def_id, substs))
}

fn create_substs_for_ast_trait_ref<'a,'tcx>(this: &AstConv<'tcx>,
                                            rscope: &RegionScope,
                                            span: Span,
                                            param_mode: PathParamMode,
                                            trait_def_id: ast::DefId,
                                            self_ty: Option<Ty<'tcx>>,
                                            trait_segment: &ast::PathSegment)
                                            -> (&'tcx Substs<'tcx>, Vec<ConvertedBinding<'tcx>>)
{
    debug!("create_substs_for_ast_trait_ref(trait_segment={:?})",
           trait_segment);

    let trait_def = match this.get_trait_def(span, trait_def_id) {
        Ok(trait_def) => trait_def,
        Err(ErrorReported) => {
            // No convenient way to recover from a cycle here. Just bail. Sorry!
            this.tcx().sess.abort_if_errors();
            this.tcx().sess.bug("ErrorReported returned, but no errors reports?")
        }
    };

    let (regions, types, assoc_bindings) = match trait_segment.parameters {
        ast::AngleBracketedParameters(ref data) => {
            // For now, require that parenthetical notation be used
            // only with `Fn()` etc.
            if !this.tcx().sess.features.borrow().unboxed_closures && trait_def.paren_sugar {
                span_err!(this.tcx().sess, span, E0215,
                                         "angle-bracket notation is not stable when \
                                         used with the `Fn` family of traits, use parentheses");
                fileline_help!(this.tcx().sess, span,
                           "add `#![feature(unboxed_closures)]` to \
                            the crate attributes to enable");
            }

            convert_angle_bracketed_parameters(this, rscope, span, &trait_def.generics, data)
        }
        ast::ParenthesizedParameters(ref data) => {
            // For now, require that parenthetical notation be used
            // only with `Fn()` etc.
            if !this.tcx().sess.features.borrow().unboxed_closures && !trait_def.paren_sugar {
                span_err!(this.tcx().sess, span, E0216,
                                         "parenthetical notation is only stable when \
                                         used with the `Fn` family of traits");
                fileline_help!(this.tcx().sess, span,
                           "add `#![feature(unboxed_closures)]` to \
                            the crate attributes to enable");
            }

            convert_parenthesized_parameters(this, rscope, span, &trait_def.generics, data)
        }
    };

    let substs = create_substs_for_ast_path(this,
                                            span,
                                            param_mode,
                                            &trait_def.generics,
                                            self_ty,
                                            types,
                                            regions);

    (this.tcx().mk_substs(substs), assoc_bindings)
}

fn ast_type_binding_to_poly_projection_predicate<'tcx>(
    this: &AstConv<'tcx>,
    mut trait_ref: ty::PolyTraitRef<'tcx>,
    self_ty: Option<Ty<'tcx>>,
    binding: &ConvertedBinding<'tcx>)
    -> Result<ty::PolyProjectionPredicate<'tcx>, ErrorReported>
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
    if this.trait_defines_associated_type_named(trait_ref.def_id(), binding.item_name) {
        return Ok(ty::Binder(ty::ProjectionPredicate {      // <-------------------+
            projection_ty: ty::ProjectionTy {               //                     |
                trait_ref: trait_ref.skip_binder().clone(), // Binder moved here --+
                item_name: binding.item_name,
            },
            ty: binding.ty,
        }));
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
        let mut dummy_substs = trait_ref.skip_binder().substs.clone(); // binder moved here -+
        assert!(dummy_substs.self_ty().is_none());                     //                    |
        dummy_substs.types.push(SelfSpace, dummy_self_ty);             //                    |
        trait_ref = ty::Binder(Rc::new(ty::TraitRef::new(trait_ref.def_id(), // <------------+
                                                         tcx.mk_substs(dummy_substs))));
    }

    try!(this.ensure_super_predicates(binding.span, trait_ref.def_id()));

    let mut candidates: Vec<ty::PolyTraitRef> =
        traits::supertraits(tcx, trait_ref.clone())
        .filter(|r| this.trait_defines_associated_type_named(r.def_id(), binding.item_name))
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

    let candidate = try!(one_bound_for_assoc_type(tcx,
                                                  candidates,
                                                  &trait_ref.user_string(tcx),
                                                  &token::get_name(binding.item_name),
                                                  binding.span));

    Ok(ty::Binder(ty::ProjectionPredicate {             // <-------------------------+
        projection_ty: ty::ProjectionTy {               //                           |
            trait_ref: candidate.skip_binder().clone(), // binder is moved up here --+
            item_name: binding.item_name,
        },
        ty: binding.ty,
    }))
}

fn ast_path_to_ty<'tcx>(
    this: &AstConv<'tcx>,
    rscope: &RegionScope,
    span: Span,
    param_mode: PathParamMode,
    did: ast::DefId,
    item_segment: &ast::PathSegment)
    -> Ty<'tcx>
{
    let tcx = this.tcx();
    let (generics, decl_ty) = match this.get_item_type_scheme(span, did) {
        Ok(ty::TypeScheme { generics,  ty: decl_ty }) => {
            (generics, decl_ty)
        }
        Err(ErrorReported) => {
            return tcx.types.err;
        }
    };

    let substs = ast_path_substs_for_ty(this,
                                        rscope,
                                        span,
                                        param_mode,
                                        &generics,
                                        item_segment);

    // FIXME(#12938): This is a hack until we have full support for DST.
    if Some(did) == this.tcx().lang_items.owned_box() {
        assert_eq!(substs.types.len(TypeSpace), 1);
        return ty::mk_uniq(this.tcx(), *substs.types.get(TypeSpace, 0));
    }

    decl_ty.subst(this.tcx(), &substs)
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
        ast::TyPath(None, ref path) => {
            let def = match this.tcx().def_map.borrow().get(&ty.id) {
                Some(&def::PathResolution { base_def, depth: 0, .. }) => Some(base_def),
                _ => None
            };
            match def {
                Some(def::DefTrait(trait_def_id)) => {
                    let mut projection_bounds = Vec::new();
                    let trait_ref = object_path_to_poly_trait_ref(this,
                                                                  rscope,
                                                                  path.span,
                                                                  PathParamMode::Explicit,
                                                                  trait_def_id,
                                                                  path.segments.last().unwrap(),
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
            let hi = bounds.iter().map(|x| match *x {
                ast::TraitTyParamBound(ref tr, _) => tr.span.hi,
                ast::RegionTyParamBound(ref r) => r.span.hi,
            }).max_by(|x| x.to_usize());
            let full_span = hi.map(|hi| Span {
                lo: ty.span.lo,
                hi: hi,
                expn_id: ty.span.expn_id,
            });
            match (&ty.node, full_span) {
                (&ast::TyRptr(None, ref mut_ty), Some(full_span)) => {
                    this.tcx().sess
                        .span_suggestion(full_span, "try adding parentheses (per RFC 438):",
                                         format!("&{}({} +{})",
                                                 ppaux::mutability_to_string(mut_ty.mutbl),
                                                 pprust::ty_to_string(&*mut_ty.ty),
                                                 pprust::bounds_to_string(bounds)));
                }
                (&ast::TyRptr(Some(ref lt), ref mut_ty), Some(full_span)) => {
                    this.tcx().sess
                        .span_suggestion(full_span, "try adding parentheses (per RFC 438):",
                                         format!("&{} {}({} +{})",
                                                 pprust::lifetime_to_string(lt),
                                                 ppaux::mutability_to_string(mut_ty.mutbl),
                                                 pprust::ty_to_string(&*mut_ty.ty),
                                                 pprust::bounds_to_string(bounds)));
                }

                _ => {
                    fileline_help!(this.tcx().sess, ty.span,
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

    let result = make_object_type(this, span, trait_ref, existential_bounds);
    debug!("trait_ref_to_object_type: result={}",
           result.repr(this.tcx()));

    result
}

fn make_object_type<'tcx>(this: &AstConv<'tcx>,
                          span: Span,
                          principal: ty::PolyTraitRef<'tcx>,
                          bounds: ty::ExistentialBounds<'tcx>)
                          -> Ty<'tcx> {
    let tcx = this.tcx();
    let object = ty::TyTrait {
        principal: principal,
        bounds: bounds
    };
    let object_trait_ref =
        object.principal_trait_ref_with_self_ty(tcx, tcx.types.err);

    // ensure the super predicates and stop if we encountered an error
    if this.ensure_super_predicates(span, object.principal_def_id()).is_err() {
        return tcx.types.err;
    }

    let mut associated_types: FnvHashSet<(ast::DefId, ast::Name)> =
        traits::supertraits(tcx, object_trait_ref)
        .flat_map(|tr| {
            let trait_def = ty::lookup_trait_def(tcx, tr.def_id());
            trait_def.associated_type_names
                .clone()
                .into_iter()
                .map(move |associated_type_name| (tr.def_id(), associated_type_name))
        })
        .collect();

    for projection_bound in &object.bounds.projection_bounds {
        let pair = (projection_bound.0.projection_ty.trait_ref.def_id,
                    projection_bound.0.projection_ty.item_name);
        associated_types.remove(&pair);
    }

    for (trait_def_id, name) in associated_types {
        span_err!(tcx.sess, span, E0191,
            "the value of the associated type `{}` (from the trait `{}`) must be specified",
                    name.user_string(tcx),
                    ty::item_path_str(tcx, trait_def_id));
    }

    ty::mk_trait(tcx, object.principal, object.bounds)
}

fn report_ambiguous_associated_type(tcx: &ty::ctxt,
                                    span: Span,
                                    type_str: &str,
                                    trait_str: &str,
                                    name: &str) {
    span_err!(tcx.sess, span, E0223,
              "ambiguous associated type; specify the type using the syntax \
               `<{} as {}>::{}`",
              type_str, trait_str, name);
}

// Search for a bound on a type parameter which includes the associated item
// given by assoc_name. ty_param_node_id is the node id for the type parameter
// (which might be `Self`, but only if it is the `Self` of a trait, not an
// impl). This function will fail if there are no suitable bounds or there is
// any ambiguity.
fn find_bound_for_assoc_item<'tcx>(this: &AstConv<'tcx>,
                                   ty_param_node_id: ast::NodeId,
                                   assoc_name: ast::Name,
                                   span: Span)
                                   -> Result<ty::PolyTraitRef<'tcx>, ErrorReported>
{
    let tcx = this.tcx();

    let bounds = match this.get_type_parameter_bounds(span, ty_param_node_id) {
        Ok(v) => v,
        Err(ErrorReported) => {
            return Err(ErrorReported);
        }
    };

    // Ensure the super predicates and stop if we encountered an error.
    if bounds.iter().any(|b| this.ensure_super_predicates(span, b.def_id()).is_err()) {
        return Err(ErrorReported);
    }

    // Check that there is exactly one way to find an associated type with the
    // correct name.
    let suitable_bounds: Vec<_> =
        traits::transitive_bounds(tcx, &bounds)
        .filter(|b| this.trait_defines_associated_type_named(b.def_id(), assoc_name))
        .collect();

    let ty_param_name = tcx.type_parameter_def(ty_param_node_id).name;
    one_bound_for_assoc_type(tcx,
                             suitable_bounds,
                             &token::get_name(ty_param_name),
                             &token::get_name(assoc_name),
                             span)
}


// Checks that bounds contains exactly one element and reports appropriate
// errors otherwise.
fn one_bound_for_assoc_type<'tcx>(tcx: &ty::ctxt<'tcx>,
                                  bounds: Vec<ty::PolyTraitRef<'tcx>>,
                                  ty_param_name: &str,
                                  assoc_name: &str,
                                  span: Span)
    -> Result<ty::PolyTraitRef<'tcx>, ErrorReported>
{
    if bounds.is_empty() {
        span_err!(tcx.sess, span, E0220,
                  "associated type `{}` not found for `{}`",
                  assoc_name,
                  ty_param_name);
        return Err(ErrorReported);
    }

    if bounds.len() > 1 {
        span_err!(tcx.sess, span, E0221,
                  "ambiguous associated type `{}` in bounds of `{}`",
                  assoc_name,
                  ty_param_name);

        for bound in &bounds {
            span_note!(tcx.sess, span,
                       "associated type `{}` could derive from `{}`",
                       ty_param_name,
                       bound.user_string(tcx));
        }
    }

    Ok(bounds[0].clone())
}

// Create a type from a a path to an associated type.
// For a path A::B::C::D, ty and ty_path_def are the type and def for A::B::C
// and item_segment is the path segment for D. We return a type and a def for
// the whole path.
// Will fail except for T::A and Self::A; i.e., if ty/ty_path_def are not a type
// parameter or Self.
fn associated_path_def_to_ty<'tcx>(this: &AstConv<'tcx>,
                                   span: Span,
                                   ty: Ty<'tcx>,
                                   ty_path_def: def::Def,
                                   item_segment: &ast::PathSegment)
                                   -> (Ty<'tcx>, def::Def)
{
    let tcx = this.tcx();
    let assoc_name = item_segment.identifier.name;

    debug!("associated_path_def_to_ty: {}::{}", ty.repr(tcx), token::get_name(assoc_name));

    check_path_args(tcx, slice::ref_slice(item_segment), NO_TPS | NO_REGIONS);

    // Find the type of the associated item, and the trait where the associated
    // item is declared.
    let bound = match (&ty.sty, ty_path_def) {
        (_, def::DefSelfTy(Some(trait_did), Some((impl_id, _)))) => {
            // `Self` in an impl of a trait - we have a concrete self type and a
            // trait reference.
            match tcx.map.expect_item(impl_id).node {
                ast::ItemImpl(_, _, _, Some(ref trait_ref), _, _) => {
                    if this.ensure_super_predicates(span, trait_did).is_err() {
                        return (tcx.types.err, ty_path_def);
                    }

                    let trait_segment = &trait_ref.path.segments.last().unwrap();
                    let trait_ref = ast_path_to_mono_trait_ref(this,
                                                               &ExplicitRscope,
                                                               span,
                                                               PathParamMode::Explicit,
                                                               trait_did,
                                                               Some(ty),
                                                               trait_segment);

                    let candidates: Vec<ty::PolyTraitRef> =
                        traits::supertraits(tcx, ty::Binder(trait_ref.clone()))
                        .filter(|r| this.trait_defines_associated_type_named(r.def_id(),
                                                                             assoc_name))
                        .collect();

                    match one_bound_for_assoc_type(tcx,
                                                   candidates,
                                                   "Self",
                                                   &token::get_name(assoc_name),
                                                   span) {
                        Ok(bound) => bound,
                        Err(ErrorReported) => return (tcx.types.err, ty_path_def),
                    }
                }
                _ => unreachable!()
            }
        }
        (&ty::ty_param(_), def::DefTyParam(..)) |
        (&ty::ty_param(_), def::DefSelfTy(Some(_), None)) => {
            // A type parameter or Self, we need to find the associated item from
            // a bound.
            let ty_param_node_id = ty_path_def.local_node_id();
            match find_bound_for_assoc_item(this, ty_param_node_id, assoc_name, span) {
                Ok(bound) => bound,
                Err(ErrorReported) => return (tcx.types.err, ty_path_def),
            }
        }
        _ => {
            report_ambiguous_associated_type(tcx,
                                             span,
                                             &ty.user_string(tcx),
                                             "Trait",
                                             &token::get_name(assoc_name));
            return (tcx.types.err, ty_path_def);
        }
    };

    let trait_did = bound.0.def_id;
    let ty = this.projected_ty_from_poly_trait_ref(span, bound, assoc_name);

    let item_did = if trait_did.krate == ast::LOCAL_CRATE {
        // `ty::trait_items` used below requires information generated
        // by type collection, which may be in progress at this point.
        match tcx.map.expect_item(trait_did.node).node {
            ast::ItemTrait(_, _, _, ref trait_items) => {
                let item = trait_items.iter()
                                      .find(|i| i.ident.name == assoc_name)
                                      .expect("missing associated type");
                ast_util::local_def(item.id)
            }
            _ => unreachable!()
        }
    } else {
        let trait_items = ty::trait_items(tcx, trait_did);
        let item = trait_items.iter().find(|i| i.name() == assoc_name);
        item.expect("missing associated type").def_id()
    };

    (ty, def::DefAssociatedTy(trait_did, item_did))
}

fn qpath_to_ty<'tcx>(this: &AstConv<'tcx>,
                     rscope: &RegionScope,
                     span: Span,
                     param_mode: PathParamMode,
                     opt_self_ty: Option<Ty<'tcx>>,
                     trait_def_id: ast::DefId,
                     trait_segment: &ast::PathSegment,
                     item_segment: &ast::PathSegment)
                     -> Ty<'tcx>
{
    let tcx = this.tcx();

    check_path_args(tcx, slice::ref_slice(item_segment), NO_TPS | NO_REGIONS);

    let self_ty = if let Some(ty) = opt_self_ty {
        ty
    } else {
        let path_str = ty::item_path_str(tcx, trait_def_id);
        report_ambiguous_associated_type(tcx,
                                         span,
                                         "Type",
                                         &path_str,
                                         &token::get_ident(item_segment.identifier));
        return tcx.types.err;
    };

    debug!("qpath_to_ty: self_type={}", self_ty.repr(tcx));

    let trait_ref = ast_path_to_mono_trait_ref(this,
                                               rscope,
                                               span,
                                               param_mode,
                                               trait_def_id,
                                               Some(self_ty),
                                               trait_segment);

    debug!("qpath_to_ty: trait_ref={}", trait_ref.repr(tcx));

    this.projected_ty(span, trait_ref, item_segment.identifier.name)
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

// Check the base def in a PathResolution and convert it to a Ty. If there are
// associated types in the PathResolution, these will need to be seperately
// resolved.
fn base_def_to_ty<'tcx>(this: &AstConv<'tcx>,
                        rscope: &RegionScope,
                        span: Span,
                        param_mode: PathParamMode,
                        def: &def::Def,
                        opt_self_ty: Option<Ty<'tcx>>,
                        base_segments: &[ast::PathSegment])
                        -> Ty<'tcx> {
    let tcx = this.tcx();

    match *def {
        def::DefTrait(trait_def_id) => {
            // N.B. this case overlaps somewhat with
            // TyObjectSum, see that fn for details
            let mut projection_bounds = Vec::new();

            let trait_ref = object_path_to_poly_trait_ref(this,
                                                          rscope,
                                                          span,
                                                          param_mode,
                                                          trait_def_id,
                                                          base_segments.last().unwrap(),
                                                          &mut projection_bounds);

            check_path_args(tcx, base_segments.init(), NO_TPS | NO_REGIONS);
            trait_ref_to_object_type(this,
                                     rscope,
                                     span,
                                     trait_ref,
                                     projection_bounds,
                                     &[])
        }
        def::DefTy(did, _) | def::DefStruct(did) => {
            check_path_args(tcx, base_segments.init(), NO_TPS | NO_REGIONS);
            ast_path_to_ty(this,
                           rscope,
                           span,
                           param_mode,
                           did,
                           base_segments.last().unwrap())
        }
        def::DefTyParam(space, index, _, name) => {
            check_path_args(tcx, base_segments, NO_TPS | NO_REGIONS);
            ty::mk_param(tcx, space, index, name)
        }
        def::DefSelfTy(_, Some((_, self_ty_id))) => {
            // Self in impl (we know the concrete type).
            check_path_args(tcx, base_segments, NO_TPS | NO_REGIONS);
            if let Some(&ty) = tcx.ast_ty_to_ty_cache.borrow().get(&self_ty_id) {
                ty
            } else {
                tcx.sess.span_bug(span, "self type has not been fully resolved")
            }
        }
        def::DefSelfTy(Some(_), None) => {
            // Self in trait.
            check_path_args(tcx, base_segments, NO_TPS | NO_REGIONS);
            ty::mk_self_type(tcx)
        }
        def::DefAssociatedTy(trait_did, _) => {
            check_path_args(tcx, &base_segments[..base_segments.len()-2], NO_TPS | NO_REGIONS);
            qpath_to_ty(this,
                        rscope,
                        span,
                        param_mode,
                        opt_self_ty,
                        trait_did,
                        &base_segments[base_segments.len()-2],
                        base_segments.last().unwrap())
        }
        def::DefMod(id) => {
            // Used as sentinel by callers to indicate the `<T>::A::B::C` form.
            // FIXME(#22519) This part of the resolution logic should be
            // avoided entirely for that form, once we stop needed a Def
            // for `associated_path_def_to_ty`.
            // Fixing this will also let use resolve <Self>::Foo the same way we
            // resolve Self::Foo, at the moment we can't resolve the former because
            // we don't have the trait information around, which is just sad.

            if !base_segments.is_empty() {
                span_err!(tcx.sess,
                          span,
                          E0247,
                          "found module name used as a type: {}",
                          tcx.map.node_to_string(id.node));
                return this.tcx().types.err;
            }

            opt_self_ty.expect("missing T in <T>::a::b::c")
        }
        def::DefPrimTy(prim_ty) => {
            prim_ty_to_ty(tcx, base_segments, prim_ty)
        }
        _ => {
            span_err!(tcx.sess, span, E0248,
                      "found value name used as a type: {:?}", *def);
            return this.tcx().types.err;
        }
    }
}

// Note that both base_segments and assoc_segments may be empty, although not at
// the same time.
pub fn finish_resolving_def_to_ty<'tcx>(this: &AstConv<'tcx>,
                                        rscope: &RegionScope,
                                        span: Span,
                                        param_mode: PathParamMode,
                                        def: &def::Def,
                                        opt_self_ty: Option<Ty<'tcx>>,
                                        base_segments: &[ast::PathSegment],
                                        assoc_segments: &[ast::PathSegment])
                                        -> Ty<'tcx> {
    let mut ty = base_def_to_ty(this,
                                rscope,
                                span,
                                param_mode,
                                def,
                                opt_self_ty,
                                base_segments);
    let mut def = *def;
    // If any associated type segments remain, attempt to resolve them.
    for segment in assoc_segments {
        if ty.sty == ty::ty_err {
            break;
        }
        // This is pretty bad (it will fail except for T::A and Self::A).
        let (a_ty, a_def) = associated_path_def_to_ty(this,
                                                      span,
                                                      ty,
                                                      def,
                                                      segment);
        ty = a_ty;
        def = a_def;
    }
    ty
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

    if let Some(&ty) = tcx.ast_ty_to_ty_cache.borrow().get(&ast_ty.id) {
        return ty;
    }

    let typ = match ast_ty.node {
        ast::TyVec(ref ty) => {
            ty::mk_vec(tcx, ast_ty_to_ty(this, rscope, &**ty), None)
        }
        ast::TyObjectSum(ref ty, ref bounds) => {
            match ast_ty_to_trait_ref(this, rscope, &**ty, bounds) {
                Ok((trait_ref, projection_bounds)) => {
                    trait_ref_to_object_type(this,
                                             rscope,
                                             ast_ty.span,
                                             trait_ref,
                                             projection_bounds,
                                             bounds)
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
            conv_ty_poly_trait_ref(this, rscope, ast_ty.span, bounds)
        }
        ast::TyPath(ref maybe_qself, ref path) => {
            let path_res = if let Some(&d) = tcx.def_map.borrow().get(&ast_ty.id) {
                d
            } else if let Some(ast::QSelf { position: 0, .. }) = *maybe_qself {
                // Create some fake resolution that can't possibly be a type.
                def::PathResolution {
                    base_def: def::DefMod(ast_util::local_def(ast::CRATE_NODE_ID)),
                    last_private: LastMod(AllPublic),
                    depth: path.segments.len()
                }
            } else {
                tcx.sess.span_bug(ast_ty.span,
                                  &format!("unbound path {}", ast_ty.repr(tcx)))
            };
            let def = path_res.base_def;
            let base_ty_end = path.segments.len() - path_res.depth;
            let opt_self_ty = maybe_qself.as_ref().map(|qself| {
                ast_ty_to_ty(this, rscope, &qself.ty)
            });
            let ty = finish_resolving_def_to_ty(this,
                                                rscope,
                                                ast_ty.span,
                                                PathParamMode::Explicit,
                                                &def,
                                                opt_self_ty,
                                                &path.segments[..base_ty_end],
                                                &path.segments[base_ty_end..]);

            if path_res.depth != 0 && ty.sty != ty::ty_err {
                // Write back the new resolution.
                tcx.def_map.borrow_mut().insert(ast_ty.id, def::PathResolution {
                    base_def: def,
                    last_private: path_res.last_private,
                    depth: 0
                });
            }

            ty
        }
        ast::TyFixedLengthVec(ref ty, ref e) => {
            match const_eval::eval_const_expr_partial(tcx, &**e, Some(tcx.types.usize)) {
                Ok(r) => {
                    match r {
                        const_eval::const_int(i) =>
                            ty::mk_vec(tcx, ast_ty_to_ty(this, rscope, &**ty),
                                        Some(i as usize)),
                        const_eval::const_uint(i) =>
                            ty::mk_vec(tcx, ast_ty_to_ty(this, rscope, &**ty),
                                        Some(i as usize)),
                        _ => {
                            span_err!(tcx.sess, ast_ty.span, E0249,
                                      "expected constant expr for array length");
                            this.tcx().types.err
                        }
                    }
                }
                Err(ref r) => {
                    let subspan  =
                        ast_ty.span.lo <= r.span.lo && r.span.hi <= ast_ty.span.hi;
                    span_err!(tcx.sess, r.span, E0250,
                              "array length constant evaluation error: {}",
                              r.description());
                    if !subspan {
                        span_note!(tcx.sess, ast_ty.span, "for array length here")
                    }
                    this.tcx().types.err
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
    };

    tcx.ast_ty_to_ty_cache.borrow_mut().insert(ast_ty.id, typ);
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
                          sig: &ast::MethodSig,
                          untransformed_self_ty: Ty<'tcx>)
                          -> (ty::BareFnTy<'tcx>, ty::ExplicitSelfCategory) {
    let self_info = Some(SelfInfo {
        untransformed_self_ty: untransformed_self_ty,
        explicit_self: &sig.explicit_self,
    });
    let (bare_fn_ty, optional_explicit_self_category) =
        ty_of_method_or_bare_fn(this,
                                sig.unsafety,
                                sig.abi,
                                self_info,
                                &sig.decl);
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
        &decl.inputs[..]
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

    fn count_modifiers(ty: Ty) -> usize {
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

    make_object_type(this, span, main_trait_bound, bounds)
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

    if !explicit_region_bounds.is_empty() {
        // Explicitly specified region bound. Use that.
        let r = explicit_region_bounds[0];
        return ast_region_to_region(tcx, r);
    }

    if let Err(ErrorReported) = this.ensure_super_predicates(span,principal_trait_ref.def_id()) {
        return ty::ReStatic;
    }

    // No explicit region bound specified. Therefore, examine trait
    // bounds and see if we can derive region bounds from those.
    let derived_region_bounds =
        object_region_bounds(tcx, &principal_trait_ref, builtin_bounds);

    // If there are no derived region bounds, then report back that we
    // can find no region bound.
    if derived_region_bounds.is_empty() {
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
    for ast_bound in ast_bounds {
        match *ast_bound {
            ast::TraitTyParamBound(ref b, ast::TraitBoundModifier::None) => {
                match ::lookup_full_def(tcx, b.trait_ref.path.span, b.trait_ref.ref_id) {
                    def::DefTrait(trait_did) => {
                        if ty::try_add_builtin_trait(tcx,
                                                     trait_did,
                                                     &mut builtin_bounds) {
                            let segments = &b.trait_ref.path.segments;
                            let parameters = &segments[segments.len() - 1].parameters;
                            if !parameters.types().is_empty() {
                                check_type_argument_count(tcx, b.trait_ref.path.span,
                                                          parameters.types().len(), 0, 0);
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

fn check_type_argument_count(tcx: &ty::ctxt, span: Span, supplied: usize,
                             required: usize, accepted: usize) {
    if supplied < required {
        let expected = if required < accepted {
            "expected at least"
        } else {
            "expected"
        };
        span_err!(tcx.sess, span, E0243,
                  "wrong number of type arguments: {} {}, found {}",
                  expected, required, supplied);
    } else if supplied > accepted {
        let expected = if required < accepted {
            "expected at most"
        } else {
            "expected"
        };
        span_err!(tcx.sess, span, E0244,
                  "wrong number of type arguments: {} {}, found {}",
                  expected,
                  accepted,
                  supplied);
    }
}

fn report_lifetime_number_error(tcx: &ty::ctxt, span: Span, number: usize, expected: usize) {
    span_err!(tcx.sess, span, E0107,
              "wrong number of lifetime parameters: expected {}, found {}",
              expected, number);
}
