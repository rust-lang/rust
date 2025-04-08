//! Constraint construction and representation
//!
//! The second pass over the HIR determines the set of constraints.
//! We walk the set of items and, for each member, generate new constraints.

use hir::def_id::{DefId, LocalDefId};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_middle::ty::{self, GenericArgKind, GenericArgsRef, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use tracing::{debug, instrument};

use super::terms::VarianceTerm::*;
use super::terms::*;

pub(crate) struct ConstraintContext<'a, 'tcx> {
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
pub(crate) struct Constraint<'a> {
    pub inferred: InferredIndex,
    pub variance: &'a VarianceTerm<'a>,
}

/// To build constraints, we visit one item (type, trait) at a time
/// and look at its contents. So e.g., if we have
/// ```ignore (illustrative)
/// struct Foo<T> {
///     b: Bar<T>
/// }
/// ```
/// then while we are visiting `Bar<T>`, the `CurrentItem` would have
/// the `DefId` and the start of `Foo`'s inferreds.
struct CurrentItem {
    inferred_start: InferredIndex,
}

pub(crate) fn add_constraints_from_crate<'a, 'tcx>(
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

    let crate_items = tcx.hir_crate_items(());

    for def_id in crate_items.definitions() {
        let def_kind = tcx.def_kind(def_id);
        match def_kind {
            DefKind::Struct | DefKind::Union | DefKind::Enum => {
                constraint_cx.build_constraints_for_item(def_id);

                let adt = tcx.adt_def(def_id);
                for variant in adt.variants() {
                    if let Some(ctor_def_id) = variant.ctor_def_id() {
                        constraint_cx.build_constraints_for_item(ctor_def_id.expect_local());
                    }
                }
            }
            DefKind::Fn | DefKind::AssocFn => constraint_cx.build_constraints_for_item(def_id),
            DefKind::TyAlias if tcx.type_alias_is_lazy(def_id) => {
                constraint_cx.build_constraints_for_item(def_id)
            }
            _ => {}
        }
    }

    constraint_cx
}

impl<'a, 'tcx> ConstraintContext<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.terms_cx.tcx
    }

    fn build_constraints_for_item(&mut self, def_id: LocalDefId) {
        let tcx = self.tcx();
        debug!("build_constraints_for_item({})", tcx.def_path_str(def_id));

        // Skip items with no generics - there's nothing to infer in them.
        if tcx.generics_of(def_id).is_empty() {
            return;
        }

        let inferred_start = self.terms_cx.inferred_starts[&def_id];
        let current_item = &CurrentItem { inferred_start };
        let ty = tcx.type_of(def_id).instantiate_identity();

        // The type as returned by `type_of` is the underlying type and generally not a weak projection.
        // Therefore we need to check the `DefKind` first.
        if let DefKind::TyAlias = tcx.def_kind(def_id)
            && tcx.type_alias_is_lazy(def_id)
        {
            self.add_constraints_from_ty(current_item, ty, self.covariant);
            return;
        }

        match ty.kind() {
            ty::Adt(def, _) => {
                // Not entirely obvious: constraints on structs/enums do not
                // affect the variance of their type parameters. See discussion
                // in comment at top of module.
                //
                // self.add_constraints_from_generics(generics);

                for field in def.all_fields() {
                    self.add_constraints_from_ty(
                        current_item,
                        tcx.type_of(field.did).instantiate_identity(),
                        self.covariant,
                    );
                }
            }

            ty::FnDef(..) => {
                self.add_constraints_from_sig(
                    current_item,
                    tcx.fn_sig(def_id).instantiate_identity(),
                    self.covariant,
                );
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
    fn add_constraints_from_invariant_args(
        &mut self,
        current: &CurrentItem,
        args: GenericArgsRef<'tcx>,
        variance: VarianceTermPtr<'a>,
    ) {
        // Trait are always invariant so we can take advantage of that.
        let variance_i = self.invariant(variance);

        for k in args {
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

            ty::FnDef(..) | ty::Coroutine(..) | ty::Closure(..) | ty::CoroutineClosure(..) => {
                bug!("Unexpected unnameable type in variance computation: {ty}");
            }

            ty::Ref(region, ty, mutbl) => {
                self.add_constraints_from_region(current, region, variance);
                self.add_constraints_from_mt(current, &ty::TypeAndMut { ty, mutbl }, variance);
            }

            ty::Array(typ, len) => {
                self.add_constraints_from_const(current, len, variance);
                self.add_constraints_from_ty(current, typ, variance);
            }

            ty::Pat(typ, pat) => {
                match *pat {
                    ty::PatternKind::Range { start, end } => {
                        self.add_constraints_from_const(current, start, variance);
                        self.add_constraints_from_const(current, end, variance);
                    }
                }
                self.add_constraints_from_ty(current, typ, variance);
            }

            ty::Slice(typ) => {
                self.add_constraints_from_ty(current, typ, variance);
            }

            ty::RawPtr(ty, mutbl) => {
                self.add_constraints_from_mt(current, &ty::TypeAndMut { ty, mutbl }, variance);
            }

            ty::Tuple(subtys) => {
                for subty in subtys {
                    self.add_constraints_from_ty(current, subty, variance);
                }
            }

            ty::Adt(def, args) => {
                self.add_constraints_from_args(current, def.did(), args, variance);
            }

            ty::Alias(ty::Projection | ty::Inherent | ty::Opaque, ref data) => {
                self.add_constraints_from_invariant_args(current, data.args, variance);
            }

            ty::Alias(ty::Weak, ref data) => {
                self.add_constraints_from_args(current, data.def_id, data.args, variance);
            }

            ty::Dynamic(data, r, _) => {
                // The type `dyn Trait<T> +'a` is covariant w/r/t `'a`:
                self.add_constraints_from_region(current, r, variance);

                if let Some(poly_trait_ref) = data.principal() {
                    self.add_constraints_from_invariant_args(
                        current,
                        poly_trait_ref.skip_binder().args,
                        variance,
                    );
                }

                for projection in data.projection_bounds() {
                    match projection.skip_binder().term.unpack() {
                        ty::TermKind::Ty(ty) => {
                            self.add_constraints_from_ty(current, ty, self.invariant);
                        }
                        ty::TermKind::Const(c) => {
                            self.add_constraints_from_const(current, c, self.invariant)
                        }
                    }
                }
            }

            ty::Param(ref data) => {
                self.add_constraint(current, data.index, variance);
            }

            ty::FnPtr(sig_tys, hdr) => {
                self.add_constraints_from_sig(current, sig_tys.with(hdr), variance);
            }

            ty::UnsafeBinder(ty) => {
                // FIXME(unsafe_binders): This is covariant, right?
                self.add_constraints_from_ty(current, ty.skip_binder(), variance);
            }

            ty::Error(_) => {
                // we encounter this when walking the trait references for object
                // types, where we use Error as the Self type
            }

            ty::Placeholder(..) | ty::CoroutineWitness(..) | ty::Bound(..) | ty::Infer(..) => {
                bug!("unexpected type encountered in variance inference: {}", ty);
            }
        }
    }

    /// Adds constraints appropriate for a nominal type (enum, struct,
    /// object, etc) appearing in a context with ambient variance `variance`
    fn add_constraints_from_args(
        &mut self,
        current: &CurrentItem,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
        variance: VarianceTermPtr<'a>,
    ) {
        debug!(
            "add_constraints_from_args(def_id={:?}, args={:?}, variance={:?})",
            def_id, args, variance
        );

        // We don't record `inferred_starts` entries for empty generics.
        if args.is_empty() {
            return;
        }

        let (local, remote) = if let Some(def_id) = def_id.as_local() {
            (Some(self.terms_cx.inferred_starts[&def_id]), None)
        } else {
            (None, Some(self.tcx().variances_of(def_id)))
        };

        for (i, k) in args.iter().enumerate() {
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
                "add_constraints_from_args: variance_decl={:?} variance_i={:?}",
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
        c: ty::Const<'tcx>,
        variance: VarianceTermPtr<'a>,
    ) {
        debug!("add_constraints_from_const(c={:?}, variance={:?})", c, variance);

        match &c.kind() {
            ty::ConstKind::Unevaluated(uv) => {
                self.add_constraints_from_invariant_args(current, uv.args, variance);
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
        match region.kind() {
            ty::ReEarlyParam(ref data) => {
                self.add_constraint(current, data.index, variance);
            }

            ty::ReStatic => {}

            ty::ReBound(..) => {
                // Either a higher-ranked region inside of a type or a
                // late-bound function parameter.
                //
                // We do not compute constraints for either of these.
            }

            ty::ReError(_) => {}

            ty::ReLateParam(..) | ty::ReVar(..) | ty::RePlaceholder(..) | ty::ReErased => {
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
