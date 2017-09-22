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

use rustc::middle::const_val::ConstVal;
use rustc_data_structures::accumulate_vec::AccumulateVec;
use hir;
use hir::def::Def;
use hir::def_id::DefId;
use middle::resolve_lifetime as rl;
use rustc::ty::subst::{Kind, Subst, Substs};
use rustc::traits;
use rustc::ty::{self, Ty, TyCtxt, ToPredicate, TypeFoldable};
use rustc::ty::wf::object_region_bounds;
use rustc_back::slice;
use require_c_abi_if_variadic;
use util::common::ErrorReported;
use util::nodemap::FxHashSet;

use std::iter;
use syntax::{abi, ast};
use syntax::feature_gate::{GateIssue, emit_feature_err};
use syntax_pos::Span;

pub trait AstConv<'gcx, 'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'a, 'gcx, 'tcx>;

    /// Returns the set of bounds in scope for the type parameter with
    /// the given id.
    fn get_type_parameter_bounds(&self, span: Span, def_id: DefId)
                                 -> ty::GenericPredicates<'tcx>;

    /// What lifetime should we use when a lifetime is omitted (and not elided)?
    fn re_infer(&self, span: Span, _def: Option<&ty::RegionParameterDef>)
                -> Option<ty::Region<'tcx>>;

    /// What type should we use when a type is omitted?
    fn ty_infer(&self, span: Span) -> Ty<'tcx>;

    /// Same as ty_infer, but with a known type parameter definition.
    fn ty_infer_for_def(&self,
                        _def: &ty::TypeParameterDef,
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

impl<'o, 'gcx: 'tcx, 'tcx> AstConv<'gcx, 'tcx>+'o {
    pub fn ast_region_to_region(&self,
        lifetime: &hir::Lifetime,
        def: Option<&ty::RegionParameterDef>)
        -> ty::Region<'tcx>
    {
        let tcx = self.tcx();
        let lifetime_name = |def_id| {
            tcx.hir.name(tcx.hir.as_local_node_id(def_id).unwrap())
        };

        let hir_id = tcx.hir.node_to_hir_id(lifetime.id);
        let r = match tcx.named_region(hir_id) {
            Some(rl::Region::Static) => {
                tcx.types.re_static
            }

            Some(rl::Region::LateBound(debruijn, id)) => {
                let name = lifetime_name(id);
                tcx.mk_region(ty::ReLateBound(debruijn,
                    ty::BrNamed(id, name)))
            }

            Some(rl::Region::LateBoundAnon(debruijn, index)) => {
                tcx.mk_region(ty::ReLateBound(debruijn, ty::BrAnon(index)))
            }

            Some(rl::Region::EarlyBound(index, id)) => {
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
                self.re_infer(lifetime.span, def).expect("unelided lifetime in signature")
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
        let (substs, assoc_bindings) =
            self.create_substs_for_ast_path(span,
                                            def_id,
                                            &item_segment.parameters,
                                            None);

        assoc_bindings.first().map(|b| self.prohibit_projection(b.span));

        substs
    }

    /// Given the type/region arguments provided to some path (along with
    /// an implicit Self, if this is a trait reference) returns the complete
    /// set of substitutions. This may involve applying defaulted type parameters.
    ///
    /// Note that the type listing given here is *exactly* what the user provided.
    fn create_substs_for_ast_path(&self,
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

        // If the type is parameterized by this region, then replace this
        // region with the current anon region binding (in other words,
        // whatever & would get replaced with).
        let decl_generics = tcx.generics_of(def_id);
        let num_types_provided = parameters.types.len();
        let expected_num_region_params = decl_generics.regions.len();
        let supplied_num_region_params = parameters.lifetimes.len();
        if expected_num_region_params != supplied_num_region_params {
            report_lifetime_number_error(tcx, span,
                                         supplied_num_region_params,
                                         expected_num_region_params);
        }

        // If a self-type was declared, one should be provided.
        assert_eq!(decl_generics.has_self, self_ty.is_some());

        // Check the number of type parameters supplied by the user.
        let ty_param_defs = &decl_generics.types[self_ty.is_some() as usize..];
        if !parameters.infer_types || num_types_provided > ty_param_defs.len() {
            check_type_argument_count(tcx, span, num_types_provided, ty_param_defs);
        }

        let is_object = self_ty.map_or(false, |ty| ty.sty == TRAIT_OBJECT_DUMMY_SELF);
        let default_needs_object_self = |p: &ty::TypeParameterDef| {
            if is_object && p.has_default {
                if tcx.at(span).type_of(p.def_id).has_self_ty() {
                    // There is no suitable inference default for a type parameter
                    // that references self, in an object type.
                    return true;
                }
            }

            false
        };

        let substs = Substs::for_item(tcx, def_id, |def, _| {
            let i = def.index as usize - self_ty.is_some() as usize;
            if let Some(lifetime) = parameters.lifetimes.get(i) {
                self.ast_region_to_region(lifetime, Some(def))
            } else {
                tcx.types.re_static
            }
        }, |def, substs| {
            let i = def.index as usize;

            // Handle Self first, so we can adjust the index to match the AST.
            if let (0, Some(ty)) = (i, self_ty) {
                return ty;
            }

            let i = i - self_ty.is_some() as usize - decl_generics.regions.len();
            if i < num_types_provided {
                // A provided type parameter.
                self.ast_ty_to_ty(&parameters.types[i])
            } else if parameters.infer_types {
                // No type parameters were provided, we can infer all.
                let ty_var = if !default_needs_object_self(def) {
                    self.ty_infer_for_def(def, substs, span)
                } else {
                    self.ty_infer(span)
                };
                ty_var
            } else if def.has_default {
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
                        .span_label(span, format!("missing reference to `{}`", def.name))
                        .note(&format!("because of the default `Self` reference, \
                                        type parameters must be specified on object types"))
                        .emit();
                    tcx.types.err
                } else {
                    // This is a default type parameter.
                    self.normalize_ty(
                        span,
                        tcx.at(span).type_of(def.def_id)
                            .subst_spanned(tcx, substs, Some(span))
                    )
                }
            } else {
                // We've already errored above about the mismatch.
                tcx.types.err
            }
        });

        let assoc_bindings = parameters.bindings.iter().map(|binding| {
            ConvertedBinding {
                item_name: binding.name,
                ty: self.ast_ty_to_ty(&binding.ty),
                span: binding.span,
            }
        }).collect();

        debug!("create_substs_for_ast_path(decl_generics={:?}, self_ty={:?}) -> {:?}",
               decl_generics, self_ty, substs);

        (substs, assoc_bindings)
    }

    /// Instantiates the path for the given trait reference, assuming that it's
    /// bound to a valid trait type. Returns the def_id for the defining trait.
    /// Fails if the type is a type other than a trait type.
    ///
    /// If the `projections` argument is `None`, then assoc type bindings like `Foo<T=X>`
    /// are disallowed. Otherwise, they are pushed onto the vector given.
    pub fn instantiate_mono_trait_ref(&self,
        trait_ref: &hir::TraitRef,
        self_ty: Ty<'tcx>)
        -> ty::TraitRef<'tcx>
    {
        self.prohibit_type_params(trait_ref.path.segments.split_last().unwrap().1);

        let trait_def_id = self.trait_def_id(trait_ref);
        self.ast_path_to_mono_trait_ref(trait_ref.path.span,
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
                            self.tcx().hir.node_to_pretty_string(trait_ref.ref_id));
            }
        }
    }

    pub fn instantiate_poly_trait_ref(&self,
        ast_trait_ref: &hir::PolyTraitRef,
        self_ty: Ty<'tcx>,
        poly_projections: &mut Vec<ty::PolyProjectionPredicate<'tcx>>)
        -> ty::PolyTraitRef<'tcx>
    {
        let trait_ref = &ast_trait_ref.trait_ref;
        let trait_def_id = self.trait_def_id(trait_ref);

        debug!("ast_path_to_poly_trait_ref({:?}, def_id={:?})", trait_ref, trait_def_id);

        self.prohibit_type_params(trait_ref.path.segments.split_last().unwrap().1);

        let (substs, assoc_bindings) =
            self.create_substs_for_ast_trait_ref(trait_ref.path.span,
                                                 trait_def_id,
                                                 self_ty,
                                                 trait_ref.path.segments.last().unwrap());
        let poly_trait_ref = ty::Binder(ty::TraitRef::new(trait_def_id, substs));

        poly_projections.extend(assoc_bindings.iter().filter_map(|binding| {
            // specify type to assert that error was already reported in Err case:
            let predicate: Result<_, ErrorReported> =
                self.ast_type_binding_to_poly_projection_predicate(trait_ref.ref_id,
                                                                   poly_trait_ref,
                                                                   binding);
            predicate.ok() // ok to ignore Err() because ErrorReported (see above)
        }));

        debug!("ast_path_to_poly_trait_ref({:?}, projections={:?}) -> {:?}",
               trait_ref, poly_projections, poly_trait_ref);
        poly_trait_ref
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
        assoc_bindings.first().map(|b| self.prohibit_projection(b.span));
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

        if !self.tcx().sess.features.borrow().unboxed_closures &&
           trait_segment.parameters.parenthesized != trait_def.paren_sugar {
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

        self.create_substs_for_ast_path(span,
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
        _path_id: ast::NodeId,
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

        // Simple case: X is defined in the current trait.
        if self.trait_defines_associated_type_named(trait_ref.def_id(), binding.item_name) {
            return Ok(trait_ref.map_bound(|trait_ref| {
                ty::ProjectionPredicate {
                    projection_ty: ty::ProjectionTy::from_ref_and_name(
                        tcx,
                        trait_ref,
                        binding.item_name,
                    ),
                    ty: binding.ty,
                }
            }));
        }

        // Otherwise, we have to walk through the supertraits to find
        // those that do.
        let candidates =
            traits::supertraits(tcx, trait_ref.clone())
            .filter(|r| self.trait_defines_associated_type_named(r.def_id(), binding.item_name));

        let candidate = self.one_bound_for_assoc_type(candidates,
                                                      &trait_ref.to_string(),
                                                      &binding.item_name.as_str(),
                                                      binding.span)?;

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

        let (auto_traits, trait_bounds) = split_auto_traits(tcx, &trait_bounds[1..]);

        if !trait_bounds.is_empty() {
            let b = &trait_bounds[0];
            let span = b.trait_ref.path.span;
            struct_span_err!(self.tcx().sess, span, E0225,
                "only Send/Sync traits can be used as additional traits in a trait object")
                .span_label(span, "non-Send/Sync additional trait")
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
            associated_types.remove(&projection_bound.0.projection_ty.item_def_id);
        }

        for item_def_id in associated_types {
            let assoc_item = tcx.associated_item(item_def_id);
            let trait_def_id = assoc_item.container.id();
            struct_span_err!(tcx.sess, span, E0191,
                "the value of the associated type `{}` (from the trait `{}`) must be specified",
                        assoc_item.name,
                        tcx.item_path_str(trait_def_id))
                        .span_label(span, format!(
                            "missing associated type `{}` value", assoc_item.name))
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
                                 assoc_name: ast::Name,
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
                    item.kind == ty::AssociatedKind::Type && item.name == assoc_name
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
        let assoc_name = item_segment.name;

        debug!("associated_path_def_to_ty: {:?}::{}", ty, assoc_name);

        self.prohibit_type_params(slice::ref_slice(item_segment));

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

        let trait_did = bound.0.def_id;
        let item = tcx.associated_items(trait_did).find(|i| i.name == assoc_name)
                                                  .expect("missing associated type");

        let ty = self.projected_ty_from_poly_trait_ref(span, item.def_id, bound);
        let ty = self.normalize_ty(span, ty);

        let def = Def::AssociatedTy(item.def_id);
        let def_scope = tcx.adjust(assoc_name, item.container.id(), ref_id).1;
        if !item.vis.is_accessible_from(def_scope, tcx) {
            let msg = format!("{} `{}` is private", def.kind_name(), assoc_name);
            tcx.sess.span_err(span, &msg);
        }
        tcx.check_stability(item.def_id, ref_id, span);

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

        self.prohibit_type_params(slice::ref_slice(item_segment));

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

        let trait_ref = self.ast_path_to_mono_trait_ref(span,
                                                        trait_def_id,
                                                        self_ty,
                                                        trait_segment);

        debug!("qpath_to_ty: trait_ref={:?}", trait_ref);

        self.normalize_ty(span, tcx.mk_projection(item_def_id, trait_ref.substs))
    }

    pub fn prohibit_type_params(&self, segments: &[hir::PathSegment]) {
        for segment in segments {
            for typ in &segment.parameters.types {
                struct_span_err!(self.tcx().sess, typ.span, E0109,
                                 "type parameters are not allowed on this type")
                    .span_label(typ.span, "type parameter not allowed")
                    .emit();
                break;
            }
            for lifetime in &segment.parameters.lifetimes {
                struct_span_err!(self.tcx().sess, lifetime.span, E0110,
                                 "lifetime parameters are not allowed on this type")
                    .span_label(lifetime.span,
                                "lifetime parameter not allowed on this type")
                    .emit();
                break;
            }
            for binding in &segment.parameters.bindings {
                self.prohibit_projection(binding.span);
                break;
            }
        }
    }

    pub fn prohibit_projection(&self, span: Span) {
        let mut err = struct_span_err!(self.tcx().sess, span, E0229,
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
            Def::Enum(did) | Def::TyAlias(did) | Def::Struct(did) | Def::Union(did) => {
                assert_eq!(opt_self_ty, None);
                self.prohibit_type_params(path.segments.split_last().unwrap().1);
                self.ast_path_to_ty(span, did, path.segments.last().unwrap())
            }
            Def::Variant(did) if permit_variants => {
                // Convert "variant type" as if it were a real type.
                // The resulting `Ty` is type of the variant's enum for now.
                assert_eq!(opt_self_ty, None);
                self.prohibit_type_params(path.segments.split_last().unwrap().1);
                self.ast_path_to_ty(span,
                                    tcx.parent_def_id(did).unwrap(),
                                    path.segments.last().unwrap())
            }
            Def::TyParam(did) => {
                assert_eq!(opt_self_ty, None);
                self.prohibit_type_params(&path.segments);

                let node_id = tcx.hir.as_local_node_id(did).unwrap();
                let item_id = tcx.hir.get_parent_node(node_id);
                let item_def_id = tcx.hir.local_def_id(item_id);
                let generics = tcx.generics_of(item_def_id);
                let index = generics.type_param_to_index[&tcx.hir.local_def_id(node_id).index];
                tcx.mk_param(index, tcx.hir.name(node_id))
            }
            Def::SelfTy(_, Some(def_id)) => {
                // Self in impl (we know the concrete type).

                assert_eq!(opt_self_ty, None);
                self.prohibit_type_params(&path.segments);

                tcx.at(span).type_of(def_id)
            }
            Def::SelfTy(Some(_), None) => {
                // Self in trait.
                assert_eq!(opt_self_ty, None);
                self.prohibit_type_params(&path.segments);
                tcx.mk_self_type()
            }
            Def::AssociatedTy(def_id) => {
                self.prohibit_type_params(&path.segments[..path.segments.len()-2]);
                self.qpath_to_ty(span,
                                 opt_self_ty,
                                 def_id,
                                 &path.segments[path.segments.len()-2],
                                 path.segments.last().unwrap())
            }
            Def::PrimTy(prim_ty) => {
                assert_eq!(opt_self_ty, None);
                self.prohibit_type_params(&path.segments);
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
            hir::TySlice(ref ty) => {
                tcx.mk_slice(self.ast_ty_to_ty(&ty))
            }
            hir::TyPtr(ref mt) => {
                tcx.mk_ptr(ty::TypeAndMut {
                    ty: self.ast_ty_to_ty(&mt.ty),
                    mutbl: mt.mutbl
                })
            }
            hir::TyRptr(ref region, ref mt) => {
                let r = self.ast_region_to_region(region, None);
                debug!("TyRef r={:?}", r);
                let t = self.ast_ty_to_ty(&mt.ty);
                tcx.mk_ref(r, ty::TypeAndMut {ty: t, mutbl: mt.mutbl})
            }
            hir::TyNever => {
                tcx.types.never
            },
            hir::TyTup(ref fields) => {
                tcx.mk_tup(fields.iter().map(|t| self.ast_ty_to_ty(&t)), false)
            }
            hir::TyBareFn(ref bf) => {
                require_c_abi_if_variadic(tcx, &bf.decl, bf.abi, ast_ty.span);
                tcx.mk_fn_ptr(self.ty_of_fn(bf.unsafety, bf.abi, &bf.decl))
            }
            hir::TyTraitObject(ref bounds, ref lifetime) => {
                self.conv_object_ty_poly_trait_ref(ast_ty.span, bounds, lifetime)
            }
            hir::TyImplTrait(_) => {
                // Figure out if we can allow an `impl Trait` here, by walking up
                // to a `fn` or inherent `impl` method, going only through `Ty`
                // or `TraitRef` nodes (as nothing else should be in types) and
                // ensuring that we reach the `fn`/method signature's return type.
                let mut node_id = ast_ty.id;
                let fn_decl = loop {
                    let parent = tcx.hir.get_parent_node(node_id);
                    match tcx.hir.get(parent) {
                        hir::map::NodeItem(&hir::Item {
                            node: hir::ItemFn(ref fn_decl, ..), ..
                        }) => break Some(fn_decl),

                        hir::map::NodeImplItem(&hir::ImplItem {
                            node: hir::ImplItemKind::Method(ref sig, _), ..
                        }) => {
                            match tcx.hir.expect_item(tcx.hir.get_parent(parent)).node {
                                hir::ItemImpl(.., None, _, _) => {
                                    break Some(&sig.decl)
                                }
                                _ => break None
                            }
                        }

                        hir::map::NodeTy(_) | hir::map::NodeTraitRef(_) => {}

                        _ => break None
                    }
                    node_id = parent;
                };
                let allow = fn_decl.map_or(false, |fd| {
                    match fd.output {
                        hir::DefaultReturn(_) => false,
                        hir::Return(ref ty) => ty.id == node_id
                    }
                });

                // Create the anonymized type.
                if allow {
                    let def_id = tcx.hir.local_def_id(ast_ty.id);
                    tcx.mk_anon(def_id, Substs::identity_for_item(tcx, def_id))
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
                    self.ast_ty_to_ty(qself)
                });
                self.def_to_ty(opt_self_ty, path, false)
            }
            hir::TyPath(hir::QPath::TypeRelative(ref qself, ref segment)) => {
                debug!("ast_ty_to_ty: qself={:?} segment={:?}", qself, segment);
                let ty = self.ast_ty_to_ty(qself);

                let def = if let hir::TyPath(hir::QPath::Resolved(_, ref path)) = qself.node {
                    path.def
                } else {
                    Def::Err
                };
                self.associated_path_def_to_ty(ast_ty.id, ast_ty.span, ty, def, segment).0
            }
            hir::TyArray(ref ty, length) => {
                let length_def_id = tcx.hir.body_owner_def_id(length);
                let substs = Substs::identity_for_item(tcx, length_def_id);
                let length = tcx.mk_const(ty::Const {
                    val: ConstVal::Unevaluated(length_def_id, substs),
                    ty: tcx.types.usize
                });
                let array_ty = tcx.mk_ty(ty::TyArray(self.ast_ty_to_ty(&ty), length));
                self.normalize_ty(ast_ty.span, array_ty)
            }
            hir::TyTypeof(ref _e) => {
                struct_span_err!(tcx.sess, ast_ty.span, E0516,
                                 "`typeof` is a reserved keyword but unimplemented")
                    .span_label(ast_ty.span, "reserved keyword")
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
            hir::TyErr => {
                tcx.types.err
            }
        };

        result_ty
    }

    pub fn ty_of_arg(&self,
                     ty: &hir::Ty,
                     expected_ty: Option<Ty<'tcx>>)
                     -> Ty<'tcx>
    {
        match ty.node {
            hir::TyInfer if expected_ty.is_some() => expected_ty.unwrap(),
            hir::TyInfer => self.ty_infer(ty.span),
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

        let bare_fn_ty = ty::Binder(tcx.mk_fn_sig(
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
            let br_name = match *br {
                ty::BrNamed(_, name) => name,
                _ => {
                    span_bug!(
                        decl.output.span(),
                        "anonymous bound region {:?} in return but not args",
                        br);
                }
            };
            struct_span_err!(tcx.sess,
                             decl.output.span(),
                             E0581,
                             "return type references lifetime `{}`, \
                             which does not appear in the fn input types",
                             br_name)
                .emit();
        }

        bare_fn_ty
    }

    pub fn ty_of_closure(&self,
        unsafety: hir::Unsafety,
        decl: &hir::FnDecl,
        abi: abi::Abi,
        expected_sig: Option<ty::FnSig<'tcx>>)
        -> ty::PolyFnSig<'tcx>
    {
        debug!("ty_of_closure(expected_sig={:?})",
               expected_sig);

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
            self.ty_of_arg(a, expected_arg_ty)
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
                self.ast_ty_to_ty(&output),
            hir::DefaultReturn(..) => bug!(),
        };

        debug!("ty_of_closure: output_ty={:?}", output_ty);

        ty::Binder(self.tcx().mk_fn_sig(
            input_tys,
            output_ty,
            decl.variadic,
            unsafety,
            abi
        ))
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

/// Divides a list of general trait bounds into two groups: builtin bounds (Sync/Send) and the
/// remaining general trait bounds.
fn split_auto_traits<'a, 'b, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                         trait_bounds: &'b [hir::PolyTraitRef])
    -> (Vec<DefId>, Vec<&'b hir::PolyTraitRef>)
{
    let (auto_traits, trait_bounds): (Vec<_>, _) = trait_bounds.iter().partition(|bound| {
        match bound.trait_ref.path.def {
            Def::Trait(trait_did) => {
                // Checks whether `trait_did` refers to one of the builtin
                // traits, like `Send`, and adds it to `auto_traits` if so.
                if Some(trait_did) == tcx.lang_items().send_trait() ||
                    Some(trait_did) == tcx.lang_items().sync_trait() {
                    let segments = &bound.trait_ref.path.segments;
                    let parameters = &segments[segments.len() - 1].parameters;
                    if !parameters.types.is_empty() {
                        check_type_argument_count(tcx, bound.trait_ref.path.span,
                                                  parameters.types.len(), &[]);
                    }
                    if !parameters.lifetimes.is_empty() {
                        report_lifetime_number_error(tcx, bound.trait_ref.path.span,
                                                     parameters.lifetimes.len(), 0);
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

fn check_type_argument_count(tcx: TyCtxt, span: Span, supplied: usize,
                             ty_param_defs: &[ty::TypeParameterDef]) {
    let accepted = ty_param_defs.len();
    let required = ty_param_defs.iter().take_while(|x| !x.has_default).count();
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
                format!("{} {} type argument{}",
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
                format!("{} type argument{}",
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
        .span_label(span, label)
        .emit();
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
        let mut vec = Vec::new();

        // If it could be sized, and is, add the sized predicate
        if self.implicitly_sized {
            if let Some(sized) = tcx.lang_items().sized_trait() {
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
    ByReference(ty::Region<'tcx>, hir::Mutability),
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
                ty::TyAdt(def, _) if def.is_box() => count_modifiers(ty.boxed_ty()) + 1,
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
                ty::TyAdt(def, _) if def.is_box() => ExplicitSelf::ByBox,
                _ => ExplicitSelf::ByValue,
            }
        }
    }
}
