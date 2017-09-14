// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Constraint construction and representation
//!
//! The second pass over the AST determines the set of constraints.
//! We walk the set of items and, for each member, generate new constraints.

use hir::def_id::DefId;
use rustc::dep_graph::{DepGraphSafe, DepKind};
use rustc::ich::StableHashingContext;
use rustc::ty::subst::Substs;
use rustc::ty::{self, Ty, TyCtxt};
use syntax::ast;
use rustc::hir;
use rustc::hir::itemlikevisit::ItemLikeVisitor;

use rustc_data_structures::transitive_relation::TransitiveRelation;
use rustc_data_structures::stable_hasher::StableHashingContextProvider;

use super::terms::*;
use super::terms::VarianceTerm::*;

pub struct ConstraintContext<'a, 'tcx: 'a> {
    pub terms_cx: TermsContext<'a, 'tcx>,

    // These are pointers to common `ConstantTerm` instances
    covariant: VarianceTermPtr<'a>,
    contravariant: VarianceTermPtr<'a>,
    invariant: VarianceTermPtr<'a>,
    bivariant: VarianceTermPtr<'a>,

    pub constraints: Vec<Constraint<'a>>,

    /// This relation tracks the dependencies between the variance of
    /// various items. In particular, if `a < b`, then the variance of
    /// `a` depends on the sources of `b`.
    pub dependencies: TransitiveRelation<DefId>,
}

/// Declares that the variable `decl_id` appears in a location with
/// variance `variance`.
#[derive(Copy, Clone)]
pub struct Constraint<'a> {
    pub inferred: InferredIndex,
    pub variance: &'a VarianceTerm<'a>,
}

/// To build constraints, we visit one item (type, trait) at a time
/// and look at its contents. So e.g. if we have
///
///     struct Foo<T> {
///         b: Bar<T>
///     }
///
/// then while we are visiting `Bar<T>`, the `CurrentItem` would have
/// the def-id and the start of `Foo`'s inferreds.
pub struct CurrentItem {
    def_id: DefId,
    inferred_start: InferredIndex,
}

pub fn add_constraints_from_crate<'a, 'tcx>(terms_cx: TermsContext<'a, 'tcx>)
                                            -> ConstraintContext<'a, 'tcx> {
    let tcx = terms_cx.tcx;
    let covariant = terms_cx.arena.alloc(ConstantTerm(ty::Covariant));
    let contravariant = terms_cx.arena.alloc(ConstantTerm(ty::Contravariant));
    let invariant = terms_cx.arena.alloc(ConstantTerm(ty::Invariant));
    let bivariant = terms_cx.arena.alloc(ConstantTerm(ty::Bivariant));
    let mut constraint_cx = ConstraintContext {
        terms_cx,
        covariant,
        contravariant,
        invariant,
        bivariant,
        constraints: Vec::new(),
        dependencies: TransitiveRelation::new(),
    };

    tcx.hir.krate().visit_all_item_likes(&mut constraint_cx);

    constraint_cx
}

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for ConstraintContext<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        match item.node {
            hir::ItemStruct(ref struct_def, _) |
            hir::ItemUnion(ref struct_def, _) => {
                self.visit_node_helper(item.id);

                if let hir::VariantData::Tuple(..) = *struct_def {
                    self.visit_node_helper(struct_def.id());
                }
            }

            hir::ItemEnum(ref enum_def, _) => {
                self.visit_node_helper(item.id);

                for variant in &enum_def.variants {
                    if let hir::VariantData::Tuple(..) = variant.node.data {
                        self.visit_node_helper(variant.node.data.id());
                    }
                }
            }

            hir::ItemFn(..) => {
                self.visit_node_helper(item.id);
            }

            hir::ItemForeignMod(ref foreign_mod) => {
                for foreign_item in &foreign_mod.items {
                    if let hir::ForeignItemFn(..) = foreign_item.node {
                        self.visit_node_helper(foreign_item.id);
                    }
                }
            }

            _ => {}
        }
    }

    fn visit_trait_item(&mut self, trait_item: &hir::TraitItem) {
        if let hir::TraitItemKind::Method(..) = trait_item.node {
            self.visit_node_helper(trait_item.id);
        }
    }

    fn visit_impl_item(&mut self, impl_item: &hir::ImplItem) {
        if let hir::ImplItemKind::Method(..) = impl_item.node {
            self.visit_node_helper(impl_item.id);
        }
    }
}

impl<'a, 'tcx> StableHashingContextProvider for ConstraintContext<'a, 'tcx> {
    type ContextType = StableHashingContext<'tcx>;

    fn create_stable_hashing_context(&self) -> Self::ContextType {
         self.terms_cx.tcx.create_stable_hashing_context()
    }
}

impl<'a, 'tcx> DepGraphSafe for ConstraintContext<'a, 'tcx> {}

impl<'a, 'tcx> ConstraintContext<'a, 'tcx> {
    fn visit_node_helper(&mut self, id: ast::NodeId) {
        let tcx = self.terms_cx.tcx;
        let def_id = tcx.hir.local_def_id(id);

        // Encapsulate constructing the constraints into a task we can
        // reference later. This can go away once the red-green
        // algorithm is in place.
        //
        // See README.md for a detailed discussion
        // on dep-graph management.
        let dep_node = def_id.to_dep_node(tcx, DepKind::ItemVarianceConstraints);
        tcx.dep_graph.with_task(dep_node,
                                self,
                                def_id,
                                visit_item_task);

        fn visit_item_task<'a, 'tcx>(ccx: &mut ConstraintContext<'a, 'tcx>,
                                     def_id: DefId)
        {
            ccx.build_constraints_for_item(def_id);
        }
    }

    fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx> {
        self.terms_cx.tcx
    }

    fn build_constraints_for_item(&mut self, def_id: DefId) {
        let tcx = self.tcx();
        debug!("build_constraints_for_item({})", tcx.item_path_str(def_id));

        // Skip items with no generics - there's nothing to infer in them.
        if tcx.generics_of(def_id).count() == 0 {
            return;
        }

        let id = tcx.hir.as_local_node_id(def_id).unwrap();
        let inferred_start = self.terms_cx.inferred_starts[&id];
        let current_item = &CurrentItem { def_id, inferred_start };
        match tcx.type_of(def_id).sty {
            ty::TyAdt(def, _) => {
                // Not entirely obvious: constraints on structs/enums do not
                // affect the variance of their type parameters. See discussion
                // in comment at top of module.
                //
                // self.add_constraints_from_generics(generics);

                for field in def.all_fields() {
                    self.add_constraints_from_ty(current_item,
                                                 tcx.type_of(field.did),
                                                 self.covariant);
                }
            }

            ty::TyFnDef(..) => {
                self.add_constraints_from_sig(current_item,
                                              tcx.fn_sig(def_id),
                                              self.covariant);
            }

            _ => {
                span_bug!(tcx.def_span(def_id),
                          "`build_constraints_for_item` unsupported for this item");
            }
        }
    }

    fn add_constraint(&mut self,
                      current: &CurrentItem,
                      index: u32,
                      variance: VarianceTermPtr<'a>) {
        debug!("add_constraint(index={}, variance={:?})", index, variance);
        self.constraints.push(Constraint {
            inferred: InferredIndex(current.inferred_start.0 + index as usize),
            variance,
        });
    }

    fn contravariant(&mut self, variance: VarianceTermPtr<'a>) -> VarianceTermPtr<'a> {
        self.xform(variance, self.contravariant)
    }

    fn invariant(&mut self, variance: VarianceTermPtr<'a>) -> VarianceTermPtr<'a> {
        self.xform(variance, self.invariant)
    }

    fn constant_term(&self, v: ty::Variance) -> VarianceTermPtr<'a> {
        match v {
            ty::Covariant => self.covariant,
            ty::Invariant => self.invariant,
            ty::Contravariant => self.contravariant,
            ty::Bivariant => self.bivariant,
        }
    }

    fn xform(&mut self, v1: VarianceTermPtr<'a>, v2: VarianceTermPtr<'a>) -> VarianceTermPtr<'a> {
        match (*v1, *v2) {
            (_, ConstantTerm(ty::Covariant)) => {
                // Applying a "covariant" transform is always a no-op
                v1
            }

            (ConstantTerm(c1), ConstantTerm(c2)) => self.constant_term(c1.xform(c2)),

            _ => &*self.terms_cx.arena.alloc(TransformTerm(v1, v2)),
        }
    }

    fn add_constraints_from_trait_ref(&mut self,
                                      current: &CurrentItem,
                                      trait_ref: ty::TraitRef<'tcx>,
                                      variance: VarianceTermPtr<'a>) {
        debug!("add_constraints_from_trait_ref: trait_ref={:?} variance={:?}",
               trait_ref,
               variance);
        self.add_constraints_from_invariant_substs(current, trait_ref.substs, variance);
    }

    fn add_constraints_from_invariant_substs(&mut self,
                                             current: &CurrentItem,
                                             substs: &Substs<'tcx>,
                                             variance: VarianceTermPtr<'a>) {
        debug!("add_constraints_from_invariant_substs: substs={:?} variance={:?}",
               substs,
               variance);

        // Trait are always invariant so we can take advantage of that.
        let variance_i = self.invariant(variance);
        for ty in substs.types() {
            self.add_constraints_from_ty(current, ty, variance_i);
        }

        for region in substs.regions() {
            self.add_constraints_from_region(current, region, variance_i);
        }
    }

    /// Adds constraints appropriate for an instance of `ty` appearing
    /// in a context with the generics defined in `generics` and
    /// ambient variance `variance`
    fn add_constraints_from_ty(&mut self,
                               current: &CurrentItem,
                               ty: Ty<'tcx>,
                               variance: VarianceTermPtr<'a>) {
        debug!("add_constraints_from_ty(ty={:?}, variance={:?})",
               ty,
               variance);

        match ty.sty {
            ty::TyBool | ty::TyChar | ty::TyInt(_) | ty::TyUint(_) | ty::TyFloat(_) |
            ty::TyStr | ty::TyNever => {
                // leaf type -- noop
            }

            ty::TyFnDef(..) |
            ty::TyGenerator(..) |
            ty::TyClosure(..) => {
                bug!("Unexpected closure type in variance computation");
            }

            ty::TyRef(region, ref mt) => {
                let contra = self.contravariant(variance);
                self.add_constraints_from_region(current, region, contra);
                self.add_constraints_from_mt(current, mt, variance);
            }

            ty::TyArray(typ, _) |
            ty::TySlice(typ) => {
                self.add_constraints_from_ty(current, typ, variance);
            }

            ty::TyRawPtr(ref mt) => {
                self.add_constraints_from_mt(current, mt, variance);
            }

            ty::TyTuple(subtys, _) => {
                for &subty in subtys {
                    self.add_constraints_from_ty(current, subty, variance);
                }
            }

            ty::TyAdt(def, substs) => {
                self.add_constraints_from_substs(current, def.did, substs, variance);
            }

            ty::TyProjection(ref data) => {
                let tcx = self.tcx();
                self.add_constraints_from_trait_ref(current, data.trait_ref(tcx), variance);
            }

            ty::TyAnon(_, substs) => {
                self.add_constraints_from_invariant_substs(current, substs, variance);
            }

            ty::TyDynamic(ref data, r) => {
                // The type `Foo<T+'a>` is contravariant w/r/t `'a`:
                let contra = self.contravariant(variance);
                self.add_constraints_from_region(current, r, contra);

                if let Some(p) = data.principal() {
                    let poly_trait_ref = p.with_self_ty(self.tcx(), self.tcx().types.err);
                    self.add_constraints_from_trait_ref(current, poly_trait_ref.0, variance);
                }

                for projection in data.projection_bounds() {
                    self.add_constraints_from_ty(current, projection.0.ty, self.invariant);
                }
            }

            ty::TyParam(ref data) => {
                self.add_constraint(current, data.idx, variance);
            }

            ty::TyFnPtr(sig) => {
                self.add_constraints_from_sig(current, sig, variance);
            }

            ty::TyError => {
                // we encounter this when walking the trait references for object
                // types, where we use TyError as the Self type
            }

            ty::TyInfer(..) => {
                bug!("unexpected type encountered in \
                      variance inference: {}",
                     ty);
            }
        }
    }

    /// Adds constraints appropriate for a nominal type (enum, struct,
    /// object, etc) appearing in a context with ambient variance `variance`
    fn add_constraints_from_substs(&mut self,
                                   current: &CurrentItem,
                                   def_id: DefId,
                                   substs: &Substs<'tcx>,
                                   variance: VarianceTermPtr<'a>) {
        debug!("add_constraints_from_substs(def_id={:?}, substs={:?}, variance={:?})",
               def_id,
               substs,
               variance);

        // We don't record `inferred_starts` entries for empty generics.
        if substs.is_empty() {
            return;
        }

        // Add a corresponding relation into the dependencies to
        // indicate that the variance for `current` relies on `def_id`.
        if self.tcx().dep_graph.is_fully_enabled() {
            self.dependencies.add(current.def_id, def_id);
        }

        let (local, remote) = if let Some(id) = self.tcx().hir.as_local_node_id(def_id) {
            (Some(self.terms_cx.inferred_starts[&id]), None)
        } else {
            (None, Some(self.tcx().variances_of(def_id)))
        };

        for (i, k) in substs.iter().enumerate() {
            let variance_decl = if let Some(InferredIndex(start)) = local {
                // Parameter on an item defined within current crate:
                // variance not yet inferred, so return a symbolic
                // variance.
                self.terms_cx.inferred_terms[start + i]
            } else {
                // Parameter on an item defined within another crate:
                // variance already inferred, just look it up.
                self.constant_term(remote.as_ref().unwrap()[i])
            };
            let variance_i = self.xform(variance, variance_decl);
            debug!("add_constraints_from_substs: variance_decl={:?} variance_i={:?}",
                   variance_decl,
                   variance_i);
            if let Some(ty) = k.as_type() {
                self.add_constraints_from_ty(current, ty, variance_i);
            } else if let Some(r) = k.as_region() {
                self.add_constraints_from_region(current, r, variance_i);
            } else {
                bug!();
            }
        }
    }

    /// Adds constraints appropriate for a function with signature
    /// `sig` appearing in a context with ambient variance `variance`
    fn add_constraints_from_sig(&mut self,
                                current: &CurrentItem,
                                sig: ty::PolyFnSig<'tcx>,
                                variance: VarianceTermPtr<'a>) {
        let contra = self.contravariant(variance);
        for &input in sig.0.inputs() {
            self.add_constraints_from_ty(current, input, contra);
        }
        self.add_constraints_from_ty(current, sig.0.output(), variance);
    }

    /// Adds constraints appropriate for a region appearing in a
    /// context with ambient variance `variance`
    fn add_constraints_from_region(&mut self,
                                   current: &CurrentItem,
                                   region: ty::Region<'tcx>,
                                   variance: VarianceTermPtr<'a>) {
        match *region {
            ty::ReEarlyBound(ref data) => {
                self.add_constraint(current, data.index, variance);
            }

            ty::ReStatic => {}

            ty::ReLateBound(..) => {
                // Late-bound regions do not get substituted the same
                // way early-bound regions do, so we skip them here.
            }

            ty::ReFree(..) |
            ty::ReScope(..) |
            ty::ReVar(..) |
            ty::ReSkolemized(..) |
            ty::ReEmpty |
            ty::ReErased => {
                // We don't expect to see anything but 'static or bound
                // regions when visiting member types or method types.
                bug!("unexpected region encountered in variance \
                      inference: {:?}",
                     region);
            }
        }
    }

    /// Adds constraints appropriate for a mutability-type pair
    /// appearing in a context with ambient variance `variance`
    fn add_constraints_from_mt(&mut self,
                               current: &CurrentItem,
                               mt: &ty::TypeAndMut<'tcx>,
                               variance: VarianceTermPtr<'a>) {
        match mt.mutbl {
            hir::MutMutable => {
                let invar = self.invariant(variance);
                self.add_constraints_from_ty(current, mt.ty, invar);
            }

            hir::MutImmutable => {
                self.add_constraints_from_ty(current, mt.ty, variance);
            }
        }
    }
}
