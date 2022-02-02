//! Constraint construction and representation
//!
//! The second pass over the AST determines the set of constraints.
//! We walk the set of items and, for each member, generate new constraints.

use hir::def_id::{DefId, LocalDefId};
use rustc_hir as hir;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_middle::ty::subst::{GenericArgKind, SubstsRef};
use rustc_middle::ty::{self, Ty, TyCtxt};

use super::terms::VarianceTerm::*;
use super::terms::*;

pub struct ConstraintContext<'a, 'tcx> {
    pub terms_cx: TermsContext<'a, 'tcx>,

    // These are pointers to common `ConstantTerm` instances
    covariant: VarianceTermPtr<'a>,
    contravariant: VarianceTermPtr<'a>,
    invariant: VarianceTermPtr<'a>,
    bivariant: VarianceTermPtr<'a>,

    pub constraints: Vec<Constraint<'a>>,
}

/// Declares that the variable `decl_id` appears in a location with
/// variance `variance`.
#[derive(Copy, Clone)]
pub struct Constraint<'a> {
    pub inferred: InferredIndex,
    pub variance: &'a VarianceTerm<'a>,
}

/// To build constraints, we visit one item (type, trait) at a time
/// and look at its contents. So e.g., if we have
///
///     struct Foo<T> {
///         b: Bar<T>
///     }
///
/// then while we are visiting `Bar<T>`, the `CurrentItem` would have
/// the `DefId` and the start of `Foo`'s inferreds.
pub struct CurrentItem {
    inferred_start: InferredIndex,
}

pub fn add_constraints_from_crate<'a, 'tcx>(
    terms_cx: TermsContext<'a, 'tcx>,
) -> ConstraintContext<'a, 'tcx> {
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
    };

    tcx.hir().visit_all_item_likes(&mut constraint_cx);

    constraint_cx
}

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for ConstraintContext<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        match item.kind {
            hir::ItemKind::Struct(ref struct_def, _) | hir::ItemKind::Union(ref struct_def, _) => {
                self.visit_node_helper(item.hir_id());

                if let hir::VariantData::Tuple(..) = *struct_def {
                    self.visit_node_helper(struct_def.ctor_hir_id().unwrap());
                }
            }

            hir::ItemKind::Enum(ref enum_def, _) => {
                self.visit_node_helper(item.hir_id());

                for variant in enum_def.variants {
                    if let hir::VariantData::Tuple(..) = variant.data {
                        self.visit_node_helper(variant.data.ctor_hir_id().unwrap());
                    }
                }
            }

            hir::ItemKind::Fn(..) => {
                self.visit_node_helper(item.hir_id());
            }

            _ => {}
        }
    }

    fn visit_trait_item(&mut self, trait_item: &hir::TraitItem<'_>) {
        if let hir::TraitItemKind::Fn(..) = trait_item.kind {
            self.visit_node_helper(trait_item.hir_id());
        }
    }

    fn visit_impl_item(&mut self, impl_item: &hir::ImplItem<'_>) {
        if let hir::ImplItemKind::Fn(..) = impl_item.kind {
            self.visit_node_helper(impl_item.hir_id());
        }
    }

    fn visit_foreign_item(&mut self, foreign_item: &hir::ForeignItem<'_>) {
        if let hir::ForeignItemKind::Fn(..) = foreign_item.kind {
            self.visit_node_helper(foreign_item.hir_id());
        }
    }
}

impl<'a, 'tcx> ConstraintContext<'a, 'tcx> {
    fn visit_node_helper(&mut self, id: hir::HirId) {
        let tcx = self.terms_cx.tcx;
        let def_id = tcx.hir().local_def_id(id);
        self.build_constraints_for_item(def_id);
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.terms_cx.tcx
    }

    fn build_constraints_for_item(&mut self, def_id: LocalDefId) {
        let tcx = self.tcx();
        debug!("build_constraints_for_item({})", tcx.def_path_str(def_id.to_def_id()));

        // Skip items with no generics - there's nothing to infer in them.
        if tcx.generics_of(def_id).count() == 0 {
            return;
        }

        let id = tcx.hir().local_def_id_to_hir_id(def_id);
        let inferred_start = self.terms_cx.inferred_starts[&id];
        let current_item = &CurrentItem { inferred_start };
        match tcx.type_of(def_id).kind() {
            ty::Adt(def, _) => {
                // Not entirely obvious: constraints on structs/enums do not
                // affect the variance of their type parameters. See discussion
                // in comment at top of module.
                //
                // self.add_constraints_from_generics(generics);

                for field in def.all_fields() {
                    self.add_constraints_from_ty(
                        current_item,
                        tcx.type_of(field.did),
                        self.covariant,
                    );
                }
            }

            ty::FnDef(..) => {
                self.add_constraints_from_sig(current_item, tcx.fn_sig(def_id), self.covariant);
            }

            ty::Error(_) => {}
            _ => {
                span_bug!(
                    tcx.def_span(def_id),
                    "`build_constraints_for_item` unsupported for this item"
                );
            }
        }
    }

    fn add_constraint(&mut self, current: &CurrentItem, index: u32, variance: VarianceTermPtr<'a>) {
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

    #[instrument(level = "debug", skip(self, current))]
    fn add_constraints_from_invariant_substs(
        &mut self,
        current: &CurrentItem,
        substs: SubstsRef<'tcx>,
        variance: VarianceTermPtr<'a>,
    ) {
        // Trait are always invariant so we can take advantage of that.
        let variance_i = self.invariant(variance);

        for k in substs {
            match k.unpack() {
                GenericArgKind::Lifetime(lt) => {
                    self.add_constraints_from_region(current, lt, variance_i)
                }
                GenericArgKind::Type(ty) => self.add_constraints_from_ty(current, ty, variance_i),
                GenericArgKind::Const(val) => {
                    self.add_constraints_from_const(current, val, variance_i)
                }
            }
        }
    }

    /// Adds constraints appropriate for an instance of `ty` appearing
    /// in a context with the generics defined in `generics` and
    /// ambient variance `variance`
    fn add_constraints_from_ty(
        &mut self,
        current: &CurrentItem,
        ty: Ty<'tcx>,
        variance: VarianceTermPtr<'a>,
    ) {
        debug!("add_constraints_from_ty(ty={:?}, variance={:?})", ty, variance);

        match *ty.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Never
            | ty::Foreign(..) => {
                // leaf type -- noop
            }

            ty::FnDef(..) | ty::Generator(..) | ty::Closure(..) => {
                bug!("Unexpected closure type in variance computation");
            }

            ty::Ref(region, ty, mutbl) => {
                let contra = self.contravariant(variance);
                self.add_constraints_from_region(current, region, contra);
                self.add_constraints_from_mt(current, &ty::TypeAndMut { ty, mutbl }, variance);
            }

            ty::Array(typ, len) => {
                self.add_constraints_from_const(current, len, variance);
                self.add_constraints_from_ty(current, typ, variance);
            }

            ty::Slice(typ) => {
                self.add_constraints_from_ty(current, typ, variance);
            }

            ty::RawPtr(ref mt) => {
                self.add_constraints_from_mt(current, mt, variance);
            }

            ty::Tuple(subtys) => {
                for subty in subtys {
                    self.add_constraints_from_ty(current, subty.expect_ty(), variance);
                }
            }

            ty::Adt(def, substs) => {
                self.add_constraints_from_substs(current, def.did, substs, variance);
            }

            ty::Projection(ref data) => {
                self.add_constraints_from_invariant_substs(current, data.substs, variance);
            }

            ty::Opaque(_, substs) => {
                self.add_constraints_from_invariant_substs(current, substs, variance);
            }

            ty::Dynamic(data, r) => {
                // The type `Foo<T+'a>` is contravariant w/r/t `'a`:
                let contra = self.contravariant(variance);
                self.add_constraints_from_region(current, r, contra);

                if let Some(poly_trait_ref) = data.principal() {
                    self.add_constraints_from_invariant_substs(
                        current,
                        poly_trait_ref.skip_binder().substs,
                        variance,
                    );
                }

                for projection in data.projection_bounds() {
                    match projection.skip_binder().term {
                        ty::Term::Ty(ty) => {
                            self.add_constraints_from_ty(current, ty, self.invariant);
                        }
                        ty::Term::Const(c) => {
                            self.add_constraints_from_const(current, c, self.invariant)
                        }
                    }
                }
            }

            ty::Param(ref data) => {
                self.add_constraint(current, data.index, variance);
            }

            ty::FnPtr(sig) => {
                self.add_constraints_from_sig(current, sig, variance);
            }

            ty::Error(_) => {
                // we encounter this when walking the trait references for object
                // types, where we use Error as the Self type
            }

            ty::Placeholder(..) | ty::GeneratorWitness(..) | ty::Bound(..) | ty::Infer(..) => {
                bug!(
                    "unexpected type encountered in \
                      variance inference: {}",
                    ty
                );
            }
        }
    }

    /// Adds constraints appropriate for a nominal type (enum, struct,
    /// object, etc) appearing in a context with ambient variance `variance`
    fn add_constraints_from_substs(
        &mut self,
        current: &CurrentItem,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
        variance: VarianceTermPtr<'a>,
    ) {
        debug!(
            "add_constraints_from_substs(def_id={:?}, substs={:?}, variance={:?})",
            def_id, substs, variance
        );

        // We don't record `inferred_starts` entries for empty generics.
        if substs.is_empty() {
            return;
        }

        let (local, remote) = if let Some(def_id) = def_id.as_local() {
            let id = self.tcx().hir().local_def_id_to_hir_id(def_id);
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
            debug!(
                "add_constraints_from_substs: variance_decl={:?} variance_i={:?}",
                variance_decl, variance_i
            );
            match k.unpack() {
                GenericArgKind::Lifetime(lt) => {
                    self.add_constraints_from_region(current, lt, variance_i)
                }
                GenericArgKind::Type(ty) => self.add_constraints_from_ty(current, ty, variance_i),
                GenericArgKind::Const(val) => {
                    self.add_constraints_from_const(current, val, variance)
                }
            }
        }
    }

    /// Adds constraints appropriate for a const expression `val`
    /// in a context with ambient variance `variance`
    fn add_constraints_from_const(
        &mut self,
        current: &CurrentItem,
        val: ty::Const<'tcx>,
        variance: VarianceTermPtr<'a>,
    ) {
        debug!("add_constraints_from_const(val={:?}, variance={:?})", val, variance);

        match &val.val() {
            ty::ConstKind::Unevaluated(uv) => {
                self.add_constraints_from_invariant_substs(current, uv.substs, variance);
            }
            _ => {}
        }
    }

    /// Adds constraints appropriate for a function with signature
    /// `sig` appearing in a context with ambient variance `variance`
    fn add_constraints_from_sig(
        &mut self,
        current: &CurrentItem,
        sig: ty::PolyFnSig<'tcx>,
        variance: VarianceTermPtr<'a>,
    ) {
        let contra = self.contravariant(variance);
        for &input in sig.skip_binder().inputs() {
            self.add_constraints_from_ty(current, input, contra);
        }
        self.add_constraints_from_ty(current, sig.skip_binder().output(), variance);
    }

    /// Adds constraints appropriate for a region appearing in a
    /// context with ambient variance `variance`
    fn add_constraints_from_region(
        &mut self,
        current: &CurrentItem,
        region: ty::Region<'tcx>,
        variance: VarianceTermPtr<'a>,
    ) {
        match *region {
            ty::ReEarlyBound(ref data) => {
                self.add_constraint(current, data.index, variance);
            }

            ty::ReStatic => {}

            ty::ReLateBound(..) => {
                // Late-bound regions do not get substituted the same
                // way early-bound regions do, so we skip them here.
            }

            ty::ReFree(..)
            | ty::ReVar(..)
            | ty::RePlaceholder(..)
            | ty::ReEmpty(_)
            | ty::ReErased => {
                // We don't expect to see anything but 'static or bound
                // regions when visiting member types or method types.
                bug!(
                    "unexpected region encountered in variance \
                      inference: {:?}",
                    region
                );
            }
        }
    }

    /// Adds constraints appropriate for a mutability-type pair
    /// appearing in a context with ambient variance `variance`
    fn add_constraints_from_mt(
        &mut self,
        current: &CurrentItem,
        mt: &ty::TypeAndMut<'tcx>,
        variance: VarianceTermPtr<'a>,
    ) {
        match mt.mutbl {
            hir::Mutability::Mut => {
                let invar = self.invariant(variance);
                self.add_constraints_from_ty(current, mt.ty, invar);
            }

            hir::Mutability::Not => {
                self.add_constraints_from_ty(current, mt.ty, variance);
            }
        }
    }
}
