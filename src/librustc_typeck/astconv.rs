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
//! `AstConv` instance; in this phase, the `get_item_type()`
//! function triggers a recursive call to `type_of_item()`
//! (note that `ast_ty_to_ty()` will detect recursive types and report
//! an error).  In the check phase, when the FnCtxt is used as the
//! `AstConv`, `get_item_type()` just looks up the item type in
//! `tcx.types` (using `TyCtxt::item_type`).
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
use rustc_data_structures::accumulate_vec::AccumulateVec;
use hir;
use hir::def::Def;
use hir::def_id::DefId;
use middle::resolve_lifetime as rl;
use rustc::lint;
use rustc::ty::subst::{Kind, Subst, Substs};
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
use util::nodemap::{NodeMap, FxHashSet};

use std::cell::RefCell;
use std::iter;
use syntax::{abi, ast};
use syntax::feature_gate::{GateIssue, emit_feature_err};
use syntax::symbol::{Symbol, keywords};
use syntax_pos::Span;
use errors::DiagnosticBuilder;

pub trait AstConv<'gcx, 'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'a, 'gcx, 'tcx>;

    /// A cache used for the result of `ast_ty_to_ty_cache`
    fn ast_ty_to_ty_cache(&self) -> &RefCell<NodeMap<Ty<'tcx>>>;

    /// Returns the generic type and lifetime parameters for an item.
    fn get_generics(&self, span: Span, id: DefId)
                    -> Result<&'tcx ty::Generics<'tcx>, ErrorReported>;

    /// Identify the type for an item, like a type alias, fn, or struct.
    fn get_item_type(&self, span: Span, id: DefId) -> Result<Ty<'tcx>, ErrorReported>;

    /// Returns the `TraitDef` for a given trait. This allows you to
    /// figure out the set of type parameters defined on the trait.
    fn get_trait_def(&self, span: Span, id: DefId)
                     -> Result<&'tcx ty::TraitDef, ErrorReported>;

    /// Ensure that the super-predicates for the trait with the given
    /// id are available and also for the transitive set of
    /// super-predicates.
    fn ensure_super_predicates(&self, span: Span, id: DefId)
                               -> Result<(), ErrorReported>;

    /// Returns the set of bounds in scope for the type parameter with
    /// the given id.
    fn get_type_parameter_bounds(&self, span: Span, def_id: ast::NodeId)
                                 -> Result<Vec<ty::PolyTraitRef<'tcx>>, ErrorReported>;

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
                        _substs: &[Kind<'tcx>],
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

struct ConvertedBinding<'tcx> {
    item_name: ast::Name,
    ty: Ty<'tcx>,
    span: Span,
}

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
    tcx: TyCtxt,
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
            parent, index, lifetime_count: n, have_bound_regions
        } = info;

        let help_name = if let Some(body) = parent {
            let arg = &tcx.map.body(body).arguments[index];
            format!("`{}`", tcx.map.node_to_pretty_string(arg.pat.id))
        } else {
            format!("argument {}", index + 1)
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
                        report_elision_failure(self.tcx(), &mut err, params);
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
        def_id: DefId,
        parameters: &hir::PathParameters,
        self_ty: Option<Ty<'tcx>>)
        -> (&'tcx Substs<'tcx>, Vec<ConvertedBinding<'tcx>>)
    {
        let tcx = self.tcx();

        debug!("create_substs_for_ast_path(def_id={:?}, self_ty={:?}, \
               parameters={:?})",
               def_id, self_ty, parameters);

        let (lifetimes, num_types_provided, infer_types) = match *parameters {
            hir::AngleBracketedParameters(ref data) => {
                (&data.lifetimes[..], data.types.len(), data.infer_types)
            }
            hir::ParenthesizedParameters(_) => (&[][..], 1, false)
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
        let ty_param_defs = &decl_generics.types[self_ty.is_some() as usize..];
        if !infer_types || num_types_provided > ty_param_defs.len() {
            check_type_argument_count(tcx, span, num_types_provided, ty_param_defs);
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
            if i < num_types_provided {
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
            } else if infer_types {
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
    fn find_implied_output_region<I>(&self,
                                     input_tys: &[Ty<'tcx>],
                                     parent: Option<hir::BodyId>,
                                     input_indices: I) -> ElidedLifetime
        where I: Iterator<Item=usize>
    {
        let tcx = self.tcx();
        let mut lifetimes_for_params = Vec::with_capacity(input_tys.len());
        let mut possible_implied_output_region = None;
        let mut lifetimes = 0;

        for (input_type, index) in input_tys.iter().zip(input_indices) {
            let mut regions = FxHashSet();
            let have_bound_regions = tcx.collect_regions(input_type, &mut regions);

            debug!("find_implied_output_regions: collected {:?} from {:?} \
                    have_bound_regions={:?}", &regions, input_type, have_bound_regions);

            lifetimes += regions.len();

            if lifetimes == 1 && regions.len() == 1 {
                // there's a chance that the unique lifetime of this
                // iteration will be the appropriate lifetime for output
                // parameters, so lets store it.
                possible_implied_output_region = regions.iter().cloned().next();
            }

            lifetimes_for_params.push(ElisionFailureInfo {
                parent: parent,
                index: index,
                lifetime_count: regions.len(),
                have_bound_regions: have_bound_regions
            });
        }

        if lifetimes == 1 {
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
                                        region_substs: &[Kind<'tcx>],
                                        data: &hir::ParenthesizedParameterData)
                                        -> (Ty<'tcx>, ConvertedBinding<'tcx>)
    {
        let anon_scope = rscope.anon_type_scope();
        let binding_rscope = MaybeWithAnonTypes::new(BindingRscope::new(), anon_scope);
        let inputs = self.tcx().mk_type_list(data.inputs.iter().map(|a_t| {
            self.ast_ty_arg_to_ty(&binding_rscope, None, region_substs, a_t)
        }));
        let input_params = 0..inputs.len();
        let implied_output_region = self.find_implied_output_region(&inputs, None, input_params);

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
            item_name: Symbol::intern(FN_OUTPUT_NAME),
            ty: output,
            span: output_span
        };

        (self.tcx().mk_ty(ty::TyTuple(inputs)), output_binding)
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
                                        trait_def_id,
                                        self_ty,
                                        trait_ref.path.segments.last().unwrap())
    }

    fn trait_def_id(&self, trait_ref: &hir::TraitRef) -> DefId {
        let path = &trait_ref.path;
        match path.def {
            Def::Trait(trait_def_id) => trait_def_id,
            Def::Err => {
                self.tcx().sess.fatal("cannot continue compilation due to previous error");
            }
            _ => {
                span_fatal!(self.tcx().sess, path.span, E0245, "`{}` is not a trait",
                            self.tcx().map.node_to_pretty_string(trait_ref.ref_id));
            }
        }
    }

    fn ast_path_to_poly_trait_ref(&self,
        rscope: &RegionScope,
        span: Span,
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
                                  trait_def_id: DefId,
                                  self_ty: Ty<'tcx>,
                                  trait_segment: &hir::PathSegment)
                                  -> ty::TraitRef<'tcx>
    {
        let (substs, assoc_bindings) =
            self.create_substs_for_ast_trait_ref(rscope,
                                                 span,
                                                 trait_def_id,
                                                 self_ty,
                                                 trait_segment);
        assoc_bindings.first().map(|b| self.tcx().prohibit_projection(b.span));
        ty::TraitRef::new(trait_def_id, substs)
    }

    fn create_substs_for_ast_trait_ref(&self,
                                       rscope: &RegionScope,
                                       span: Span,
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
                                        trait_def_id,
                                        &trait_segment.parameters,
                                        Some(self_ty))
    }

    fn trait_defines_associated_type_named(&self,
                                           trait_def_id: DefId,
                                           assoc_name: ast::Name)
                                           -> bool
    {
        self.tcx().associated_items(trait_def_id).any(|item| {
            item.kind == ty::AssociatedKind::Type && item.name == assoc_name
        })
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

        let candidates =
            traits::supertraits(tcx, trait_ref.clone())
            .filter(|r| self.trait_defines_associated_type_named(r.def_id(), binding.item_name));

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
        did: DefId,
        item_segment: &hir::PathSegment)
        -> Ty<'tcx>
    {
        let tcx = self.tcx();
        let decl_ty = match self.get_item_type(span, did) {
            Ok(ty) => ty,
            Err(ErrorReported) => {
                return tcx.types.err;
            }
        };

        let substs = self.ast_path_substs_for_ty(rscope,
                                                 span,
                                                 did,
                                                 item_segment);

        // FIXME(#12938): This is a hack until we have full support for DST.
        if Some(did) == self.tcx().lang_items.owned_box() {
            assert_eq!(substs.types().count(), 1);
            return self.tcx().mk_box(substs.type_at(0));
        }

        decl_ty.subst(self.tcx(), substs)
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
                                                        trait_def_id,
                                                        dummy_self,
                                                        trait_path_ref_id,
                                                        trait_segment,
                                                        &mut projection_bounds);

        let PartitionedBounds { trait_bounds,
                                region_bounds } =
            partitioned_bounds;

        let (auto_traits, trait_bounds) = split_auto_traits(tcx, trait_bounds);

        if !trait_bounds.is_empty() {
            let b = &trait_bounds[0];
            let span = b.trait_ref.path.span;
            struct_span_err!(self.tcx().sess, span, E0225,
                "only Send/Sync traits can be used as additional traits in a trait object")
                .span_label(span, &format!("non-Send/Sync additional trait"))
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
        });

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

        let mut associated_types = FxHashSet::default();
        for tr in traits::supertraits(tcx, principal) {
            associated_types.extend(tcx.associated_items(tr.def_id())
                .filter(|item| item.kind == ty::AssociatedKind::Type)
                .map(|item| (tr.def_id(), item.name)));
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

        let mut v =
            iter::once(ty::ExistentialPredicate::Trait(*existential_principal.skip_binder()))
            .chain(auto_traits.into_iter().map(ty::ExistentialPredicate::AutoTrait))
            .chain(existential_projections
                   .map(|x| ty::ExistentialPredicate::Projection(*x.skip_binder())))
            .collect::<AccumulateVec<[_; 8]>>();
        v.sort_by(|a, b| a.cmp(tcx, b));
        let existential_predicates = ty::Binder(tcx.mk_existential_predicates(v.into_iter()));

        let region_bound = self.compute_object_lifetime_bound(span,
                                                              &region_bounds,
                                                              existential_predicates);

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
        let suitable_bounds =
            traits::transitive_bounds(tcx, &bounds)
            .filter(|b| self.trait_defines_associated_type_named(b.def_id(), assoc_name));

        self.one_bound_for_assoc_type(suitable_bounds,
                                      &ty_param_name.as_str(),
                                      &assoc_name.as_str(),
                                      span)
    }


    // Checks that bounds contains exactly one element and reports appropriate
    // errors otherwise.
    fn one_bound_for_assoc_type<I>(&self,
                                mut bounds: I,
                                ty_param_name: &str,
                                assoc_name: &str,
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
                  .span_label(span, &format!("associated type `{}` not found", assoc_name))
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
            err.span_label(span, &format!("ambiguous associated type `{}`", assoc_name));

            for bound in bounds {
                let bound_span = self.tcx().associated_items(bound.def_id()).find(|item| {
                    item.kind == ty::AssociatedKind::Type && item.name == assoc_name
                })
                .and_then(|item| self.tcx().map.span_if_local(item.def_id));

                if let Some(span) = bound_span {
                    err.span_label(span, &format!("ambiguous `{}` from `{}`",
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

                let candidates =
                    traits::supertraits(tcx, ty::Binder(trait_ref))
                    .filter(|r| self.trait_defines_associated_type_named(r.def_id(),
                                                                         assoc_name));

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

        let item = tcx.associated_items(trait_did).find(|i| i.name == assoc_name);
        let def_id = item.expect("missing associated type").def_id;
        tcx.check_stability(def_id, ref_id, span);
        (ty, Def::AssociatedTy(def_id))
    }

    fn qpath_to_ty(&self,
                   rscope: &RegionScope,
                   span: Span,
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
                        region_substs: &[Kind<'tcx>],
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

    // Check a type Path and convert it to a Ty.
    pub fn def_to_ty(&self,
                     rscope: &RegionScope,
                     opt_self_ty: Option<Ty<'tcx>>,
                     path: &hir::Path,
                     path_id: ast::NodeId,
                     permit_variants: bool)
                     -> Ty<'tcx> {
        let tcx = self.tcx();

        debug!("base_def_to_ty(def={:?}, opt_self_ty={:?}, path_segments={:?})",
               path.def, opt_self_ty, path.segments);

        let span = path.span;
        match path.def {
            Def::Trait(trait_def_id) => {
                // N.B. this case overlaps somewhat with
                // TyTraitObject, see that fn for details

                assert_eq!(opt_self_ty, None);
                tcx.prohibit_type_params(path.segments.split_last().unwrap().1);

                self.trait_path_to_object_type(rscope,
                                               span,
                                               trait_def_id,
                                               path_id,
                                               path.segments.last().unwrap(),
                                               span,
                                               partition_bounds(&[]))
            }
            Def::Enum(did) | Def::TyAlias(did) | Def::Struct(did) | Def::Union(did) => {
                assert_eq!(opt_self_ty, None);
                tcx.prohibit_type_params(path.segments.split_last().unwrap().1);
                self.ast_path_to_ty(rscope, span, did, path.segments.last().unwrap())
            }
            Def::Variant(did) if permit_variants => {
                // Convert "variant type" as if it were a real type.
                // The resulting `Ty` is type of the variant's enum for now.
                assert_eq!(opt_self_ty, None);
                tcx.prohibit_type_params(path.segments.split_last().unwrap().1);
                self.ast_path_to_ty(rscope,
                                    span,
                                    tcx.parent_def_id(did).unwrap(),
                                    path.segments.last().unwrap())
            }
            Def::TyParam(did) => {
                assert_eq!(opt_self_ty, None);
                tcx.prohibit_type_params(&path.segments);

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

                assert_eq!(opt_self_ty, None);
                tcx.prohibit_type_params(&path.segments);
                let ty = tcx.item_type(def_id);
                if let Some(free_substs) = self.get_free_substs() {
                    ty.subst(tcx, free_substs)
                } else {
                    ty
                }
            }
            Def::SelfTy(Some(_), None) => {
                // Self in trait.
                assert_eq!(opt_self_ty, None);
                tcx.prohibit_type_params(&path.segments);
                tcx.mk_self_type()
            }
            Def::AssociatedTy(def_id) => {
                tcx.prohibit_type_params(&path.segments[..path.segments.len()-2]);
                let trait_did = tcx.parent_def_id(def_id).unwrap();
                self.qpath_to_ty(rscope,
                                 span,
                                 opt_self_ty,
                                 trait_did,
                                 &path.segments[path.segments.len()-2],
                                 path.segments.last().unwrap())
            }
            Def::PrimTy(prim_ty) => {
                assert_eq!(opt_self_ty, None);
                tcx.prim_ty_to_ty(&path.segments, prim_ty)
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
    pub fn ast_ty_to_ty(&self, rscope: &RegionScope, ast_ty: &hir::Ty) -> Ty<'tcx> {
        debug!("ast_ty_to_ty(id={:?}, ast_ty={:?})",
               ast_ty.id, ast_ty);

        let tcx = self.tcx();

        let cache = self.ast_ty_to_ty_cache();
        if let Some(ty) = cache.borrow().get(&ast_ty.id) {
            return ty;
        }

        let result_ty = match ast_ty.node {
            hir::TySlice(ref ty) => {
                tcx.mk_slice(self.ast_ty_to_ty(rscope, &ty))
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
                tcx.mk_tup(fields.iter().map(|t| self.ast_ty_to_ty(rscope, &t)))
            }
            hir::TyBareFn(ref bf) => {
                require_c_abi_if_variadic(tcx, &bf.decl, bf.abi, ast_ty.span);
                let anon_scope = rscope.anon_type_scope();
                let bare_fn_ty = self.ty_of_method_or_bare_fn(bf.unsafety,
                                                              bf.abi,
                                                              None,
                                                              &bf.decl,
                                                              None,
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
                let late_bound_in_args = tcx.collect_constrained_late_bound_regions(
                    &inputs.map_bound(|i| i.to_owned()));
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
            hir::TyTraitObject(ref bounds) => {
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
            hir::TyPath(hir::QPath::Resolved(ref maybe_qself, ref path)) => {
                debug!("ast_ty_to_ty: maybe_qself={:?} path={:?}", maybe_qself, path);
                let opt_self_ty = maybe_qself.as_ref().map(|qself| {
                    self.ast_ty_to_ty(rscope, qself)
                });
                self.def_to_ty(rscope, opt_self_ty, path, ast_ty.id, false)
            }
            hir::TyPath(hir::QPath::TypeRelative(ref qself, ref segment)) => {
                debug!("ast_ty_to_ty: qself={:?} segment={:?}", qself, segment);
                let ty = self.ast_ty_to_ty(rscope, qself);

                let def = if let hir::TyPath(hir::QPath::Resolved(_, ref path)) = qself.node {
                    path.def
                } else {
                    Def::Err
                };
                self.associated_path_def_to_ty(ast_ty.id, ast_ty.span, ty, def, segment).0
            }
            hir::TyArray(ref ty, length) => {
                if let Ok(length) = eval_length(tcx.global_tcx(), length, "array length") {
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
                     ty: &hir::Ty,
                     expected_ty: Option<Ty<'tcx>>)
                     -> Ty<'tcx>
    {
        match ty.node {
            hir::TyInfer if expected_ty.is_some() => expected_ty.unwrap(),
            hir::TyInfer => self.ty_infer(ty.span),
            _ => self.ast_ty_to_ty(rscope, ty),
        }
    }

    pub fn ty_of_method(&self,
                        sig: &hir::MethodSig,
                        opt_self_value_ty: Option<Ty<'tcx>>,
                        body: Option<hir::BodyId>,
                        anon_scope: Option<AnonTypeScope>)
                        -> &'tcx ty::BareFnTy<'tcx> {
        self.ty_of_method_or_bare_fn(sig.unsafety,
                                     sig.abi,
                                     opt_self_value_ty,
                                     &sig.decl,
                                     body,
                                     None,
                                     anon_scope)
    }

    pub fn ty_of_bare_fn(&self,
                         unsafety: hir::Unsafety,
                         abi: abi::Abi,
                         decl: &hir::FnDecl,
                         body: hir::BodyId,
                         anon_scope: Option<AnonTypeScope>)
                         -> &'tcx ty::BareFnTy<'tcx> {
        self.ty_of_method_or_bare_fn(unsafety, abi, None, decl, Some(body), None, anon_scope)
    }

    fn ty_of_method_or_bare_fn(&self,
                               unsafety: hir::Unsafety,
                               abi: abi::Abi,
                               opt_self_value_ty: Option<Ty<'tcx>>,
                               decl: &hir::FnDecl,
                               body: Option<hir::BodyId>,
                               arg_anon_scope: Option<AnonTypeScope>,
                               ret_anon_scope: Option<AnonTypeScope>)
                               -> &'tcx ty::BareFnTy<'tcx>
    {
        debug!("ty_of_method_or_bare_fn");

        // New region names that appear inside of the arguments of the function
        // declaration are bound to that function type.
        let rb = MaybeWithAnonTypes::new(BindingRscope::new(), arg_anon_scope);

        let input_tys: Vec<Ty> =
            decl.inputs.iter().map(|a| self.ty_of_arg(&rb, a, None)).collect();

        let has_self = opt_self_value_ty.is_some();
        let explicit_self = opt_self_value_ty.map(|self_value_ty| {
            ExplicitSelf::determine(self_value_ty, input_tys[0])
        });

        let implied_output_region = match explicit_self {
            // `implied_output_region` is the region that will be assumed for any
            // region parameters in the return type. In accordance with the rules for
            // lifetime elision, we can determine it in two ways. First (determined
            // here), if self is by-reference, then the implied output region is the
            // region of the self parameter.
            Some(ExplicitSelf::ByReference(region, _)) => Ok(*region),

            // Second, if there was exactly one lifetime (either a substitution or a
            // reference) in the arguments, then any anonymous regions in the output
            // have that lifetime.
            _ => {
                let arg_tys = &input_tys[has_self as usize..];
                let arg_params = has_self as usize..input_tys.len();
                self.find_implied_output_region(arg_tys, body, arg_params)

            }
        };

        let output_ty = match decl.output {
            hir::Return(ref output) =>
                self.convert_ty_with_lifetime_elision(implied_output_region,
                                                      &output,
                                                      ret_anon_scope),
            hir::DefaultReturn(..) => self.tcx().mk_nil(),
        };

        debug!("ty_of_method_or_bare_fn: output_ty={:?}", output_ty);

        self.tcx().mk_bare_fn(ty::BareFnTy {
            unsafety: unsafety,
            abi: abi,
            sig: ty::Binder(self.tcx().mk_fn_sig(
                input_tys.into_iter(),
                output_ty,
                decl.variadic
            )),
        })
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

        let input_tys = decl.inputs.iter().enumerate().map(|(i, a)| {
            let expected_arg_ty = expected_sig.as_ref().and_then(|e| {
                // no guarantee that the correct number of expected args
                // were supplied
                if i < e.inputs().len() {
                    Some(e.inputs()[i])
                } else {
                    None
                }
            });
            self.ty_of_arg(&rb, a, expected_arg_ty)
        });

        let expected_ret_ty = expected_sig.as_ref().map(|e| e.output());

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

        debug!("ty_of_closure: output_ty={:?}", output_ty);

        ty::ClosureTy {
            unsafety: unsafety,
            abi: abi,
            sig: ty::Binder(self.tcx().mk_fn_sig(input_tys, output_ty, decl.variadic)),
        }
    }

    fn conv_object_ty_poly_trait_ref(&self,
        rscope: &RegionScope,
        span: Span,
        ast_bounds: &[hir::TyParamBound])
        -> Ty<'tcx>
    {
        let mut partitioned_bounds = partition_bounds(ast_bounds);

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
        existential_predicates: ty::Binder<&'tcx ty::Slice<ty::ExistentialPredicate<'tcx>>>)
        -> Option<&'tcx ty::Region> // if None, use the default
    {
        let tcx = self.tcx();

        debug!("compute_opt_region_bound(explicit_region_bounds={:?}, \
               existential_predicates={:?})",
               explicit_region_bounds,
               existential_predicates);

        if explicit_region_bounds.len() > 1 {
            span_err!(tcx.sess, explicit_region_bounds[1].span, E0226,
                "only a single explicit lifetime bound is permitted");
        }

        if let Some(&r) = explicit_region_bounds.get(0) {
            // Explicitly specified region bound. Use that.
            return Some(ast_region_to_region(tcx, r));
        }

        if let Some(principal) = existential_predicates.principal() {
            if let Err(ErrorReported) = self.ensure_super_predicates(span, principal.def_id()) {
                return Some(tcx.mk_region(ty::ReStatic));
            }
        }

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
    pub trait_bounds: Vec<&'a hir::PolyTraitRef>,
    pub region_bounds: Vec<&'a hir::Lifetime>,
}

/// Divides a list of general trait bounds into two groups: builtin bounds (Sync/Send) and the
/// remaining general trait bounds.
fn split_auto_traits<'a, 'b, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                         trait_bounds: Vec<&'b hir::PolyTraitRef>)
    -> (Vec<DefId>, Vec<&'b hir::PolyTraitRef>)
{
    let (auto_traits, trait_bounds): (Vec<_>, _) = trait_bounds.into_iter().partition(|bound| {
        match bound.trait_ref.path.def {
            Def::Trait(trait_did) => {
                // Checks whether `trait_did` refers to one of the builtin
                // traits, like `Send`, and adds it to `auto_traits` if so.
                if Some(trait_did) == tcx.lang_items.send_trait() ||
                    Some(trait_did) == tcx.lang_items.sync_trait() {
                    let segments = &bound.trait_ref.path.segments;
                    let parameters = &segments[segments.len() - 1].parameters;
                    if !parameters.types().is_empty() {
                        check_type_argument_count(tcx, bound.trait_ref.path.span,
                                                  parameters.types().len(), &[]);
                    }
                    if !parameters.lifetimes().is_empty() {
                        report_lifetime_number_error(tcx, bound.trait_ref.path.span,
                                                     parameters.lifetimes().len(), 0);
                    }
                    true
                } else {
                    false
                }
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

/// Divides a list of bounds from the AST into two groups: general trait bounds and region bounds
pub fn partition_bounds<'a, 'b, 'gcx, 'tcx>(ast_bounds: &'b [hir::TyParamBound])
    -> PartitionedBounds<'b>
{
    let mut region_bounds = Vec::new();
    let mut trait_bounds = Vec::new();
    for ast_bound in ast_bounds {
        match *ast_bound {
            hir::TraitTyParamBound(ref b, hir::TraitBoundModifier::None) => {
                trait_bounds.push(b);
            }
            hir::TraitTyParamBound(_, hir::TraitBoundModifier::Maybe) => {}
            hir::RegionTyParamBound(ref l) => {
                region_bounds.push(l);
            }
        }
    }

    PartitionedBounds {
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
        let arguments_plural = if required == 1 { "" } else { "s" };

        struct_span_err!(tcx.sess, span, E0243,
                "wrong number of type arguments: {} {}, found {}",
                expected, required, supplied)
            .span_label(span,
                &format!("{} {} type argument{}",
                    expected,
                    required,
                    arguments_plural))
            .emit();
    } else if supplied > accepted {
        let expected = if required < accepted {
            format!("expected at most {}", accepted)
        } else {
            format!("expected {}", accepted)
        };
        let arguments_plural = if accepted == 1 { "" } else { "s" };

        struct_span_err!(tcx.sess, span, E0244,
                "wrong number of type arguments: {}, found {}",
                expected, supplied)
            .span_label(
                span,
                &format!("{} type argument{}",
                    if accepted == 0 { "expected no" } else { &expected },
                    arguments_plural)
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
    pub implicitly_sized: bool,
    pub trait_bounds: Vec<ty::PolyTraitRef<'tcx>>,
    pub projection_bounds: Vec<ty::PolyProjectionPredicate<'tcx>>,
}

impl<'a, 'gcx, 'tcx> Bounds<'tcx> {
    pub fn predicates(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, param_ty: Ty<'tcx>)
                      -> Vec<ty::Predicate<'tcx>>
    {
        let mut vec = Vec::new();

        // If it could be sized, and is, add the sized predicate
        if self.implicitly_sized {
            if let Some(sized) = tcx.lang_items.sized_trait() {
                let trait_ref = ty::TraitRef {
                    def_id: sized,
                    substs: tcx.mk_substs_trait(param_ty, &[])
                };
                vec.push(trait_ref.to_predicate());
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

pub enum ExplicitSelf<'tcx> {
    ByValue,
    ByReference(&'tcx ty::Region, hir::Mutability),
    ByBox
}

impl<'tcx> ExplicitSelf<'tcx> {
    /// We wish to (for now) categorize an explicit self
    /// declaration like `self: SomeType` into either `self`,
    /// `&self`, `&mut self`, or `Box<self>`. We do this here
    /// by some simple pattern matching. A more precise check
    /// is done later in `check_method_self_type()`.
    ///
    /// Examples:
    ///
    /// ```
    /// impl Foo for &T {
    ///     // Legal declarations:
    ///     fn method1(self: &&T); // ExplicitSelf::ByReference
    ///     fn method2(self: &T); // ExplicitSelf::ByValue
    ///     fn method3(self: Box<&T>); // ExplicitSelf::ByBox
    ///
    ///     // Invalid cases will be caught later by `check_method_self_type`:
    ///     fn method_err1(self: &mut T); // ExplicitSelf::ByReference
    /// }
    /// ```
    ///
    /// To do the check we just count the number of "modifiers"
    /// on each type and compare them. If they are the same or
    /// the impl has more, we call it "by value". Otherwise, we
    /// look at the outermost modifier on the method decl and
    /// call it by-ref, by-box as appropriate. For method1, for
    /// example, the impl type has one modifier, but the method
    /// type has two, so we end up with
    /// ExplicitSelf::ByReference.
    pub fn determine(untransformed_self_ty: Ty<'tcx>,
                     self_arg_ty: Ty<'tcx>)
                     -> ExplicitSelf<'tcx> {
        fn count_modifiers(ty: Ty) -> usize {
            match ty.sty {
                ty::TyRef(_, mt) => count_modifiers(mt.ty) + 1,
                ty::TyBox(t) => count_modifiers(t) + 1,
                _ => 0,
            }
        }

        let impl_modifiers = count_modifiers(untransformed_self_ty);
        let method_modifiers = count_modifiers(self_arg_ty);

        if impl_modifiers >= method_modifiers {
            ExplicitSelf::ByValue
        } else {
            match self_arg_ty.sty {
                ty::TyRef(r, mt) => ExplicitSelf::ByReference(r, mt.mutbl),
                ty::TyBox(_) => ExplicitSelf::ByBox,
                _ => ExplicitSelf::ByValue,
            }
        }
    }
}
