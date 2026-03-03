//! Constraint solving
//!
//! The final phase iterates over the constraints, refining the variance
//! for each inferred until a fixed point is reached. This will be the
//! optimal solution to the constraints. The final variance for each
//! inferred is then written into the `variance_map` in the tcx.

use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, DefIdMap};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitor};
use tracing::debug;

use super::constraints::*;
use super::terms::VarianceTerm::*;
use super::terms::*;

fn glb(v1: ty::Variance, v2: ty::Variance) -> ty::Variance {
    // Greatest lower bound of the variance lattice as defined in The Paper:
    //
    //       *
    //    -     +
    //       o
    match (v1, v2) {
        (ty::Invariant, _) | (_, ty::Invariant) => ty::Invariant,

        (ty::Covariant, ty::Contravariant) => ty::Invariant,
        (ty::Contravariant, ty::Covariant) => ty::Invariant,

        (ty::Covariant, ty::Covariant) => ty::Covariant,

        (ty::Contravariant, ty::Contravariant) => ty::Contravariant,

        (x, ty::Bivariant) | (ty::Bivariant, x) => x,
    }
}
struct SolveContext<'a, 'tcx> {
    terms_cx: TermsContext<'a, 'tcx>,
    constraints: Vec<Constraint<'a>>,

    // Maps from an InferredIndex to the inferred value for that variable.
    solutions: Vec<ty::Variance>,
}

struct GroundingUseVisitor {
    item_def_id: DefId,
    grounded_params: Vec<u32>,
    in_self: bool,
    in_alias: bool,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for GroundingUseVisitor {
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> () {
        match ty.kind() {
            // Self-reference: visit args in a self-recursive context.
            ty::Adt(def, _) if def.did() == self.item_def_id => {
                let was_in_self = self.in_self;
                self.in_self = true;
                ty.super_visit_with(self);
                self.in_self = was_in_self;
                ()
            }
            // Projections/aliases: treat parameter uses as grounding.
            ty::Alias(..) => {
                let was_in_alias = self.in_alias;
                self.in_alias = true;
                ty.super_visit_with(self);
                self.in_alias = was_in_alias;
                ()
            }
            // Found a direct parameter use, record it
            ty::Param(data) => {
                if !self.in_self || self.in_alias {
                    self.grounded_params.push(data.index);
                }
                ()
            }
            // Everything else: recurse normally via super_visit_with,
            // which visits generic args of ADTs, elements of tuples, etc.
            _ => ty.super_visit_with(self),
        }
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> () {
        if let ty::ReEarlyParam(ref data) = r.kind() {
            if !self.in_self || self.in_alias {
                self.grounded_params.push(data.index);
            }
        }
        ()
    }
}

pub(crate) fn solve_constraints<'tcx>(
    constraints_cx: ConstraintContext<'_, 'tcx>,
) -> ty::CrateVariancesMap<'tcx> {
    let ConstraintContext { terms_cx, constraints, .. } = constraints_cx;

    let mut solutions = vec![ty::Bivariant; terms_cx.inferred_terms.len()];
    for (id, variances) in &terms_cx.lang_items {
        let InferredIndex(start) = terms_cx.inferred_starts[id];
        for (i, &variance) in variances.iter().enumerate() {
            solutions[start + i] = variance;
        }
    }

    let mut solutions_cx = SolveContext { terms_cx, constraints, solutions };
    solutions_cx.solve();
    solutions_cx.fix_purely_recursive_params();
    let variances = solutions_cx.create_map();

    ty::CrateVariancesMap { variances }
}

impl<'a, 'tcx> SolveContext<'a, 'tcx> {
    fn solve(&mut self) {
        // Propagate constraints until a fixed point is reached. Note
        // that the maximum number of iterations is 2C where C is the
        // number of constraints (each variable can change values at most
        // twice). Since number of constraints is linear in size of the
        // input, so is the inference process.
        let mut changed = true;
        while changed {
            changed = false;

            for constraint in &self.constraints {
                let Constraint { inferred, variance: term } = *constraint;
                let InferredIndex(inferred) = inferred;
                let variance = self.evaluate(term);
                let old_value = self.solutions[inferred];
                let new_value = glb(variance, old_value);
                if old_value != new_value {
                    debug!(
                        "updating inferred {} \
                            from {:?} to {:?} due to {:?}",
                        inferred, old_value, new_value, term
                    );

                    self.solutions[inferred] = new_value;
                    changed = true;
                }
            }
        }
    }

    /// After the fixed-point solver, check for parameters whose non-bivariance
    /// is solely due to self-referential cycles (e.g. `struct Thing<T>(*mut Thing<T>)`).
    /// Those parameters have no "grounding" use and should be bivariant.
    fn fix_purely_recursive_params(&mut self) {
        let tcx = self.terms_cx.tcx;

        // First pass: identify which solution slots need to be reset to Bivariant.
        // We use a RefCell so the Fn closure required by `UnordItems::all` can
        // accumulate results via interior mutability.
        let to_reset: std::cell::RefCell<Vec<usize>> = std::cell::RefCell::new(Vec::new());

        self.terms_cx.inferred_starts.items().all(|(&def_id, &InferredIndex(start))| {
            // Skip lang items with hardcoded variance (e.g., PhantomData).
            if self.terms_cx.lang_items.iter().any(|(id, _)| *id == def_id) {
                return true;
            }

            // Only check ADTs (structs, enums, unions).
            if !matches!(tcx.def_kind(def_id), DefKind::Struct | DefKind::Enum | DefKind::Union) {
                return true;
            }

            let generics = tcx.generics_of(def_id);
            let count = generics.count();

            // Quick check: if all params are already bivariant, nothing to do.
            if (0..count).all(|i| self.solutions[start + i] == ty::Bivariant) {
                return true;
            }

            // Walk all fields to find grounding uses.
            let mut visitor = GroundingUseVisitor {
                item_def_id: def_id.to_def_id(),
                grounded_params: Default::default(),
                in_self: false,
                in_alias: false,
            };
            let adt = tcx.adt_def(def_id);
            for field in adt.all_fields() {
                tcx.type_of(field.did).instantiate_identity().visit_with(&mut visitor);
            }

            // Collect solution indices with no grounding use.
            for i in 0..count {
                if !matches!(generics.param_at(i, tcx).kind, ty::GenericParamDefKind::Type { .. }) {
                    continue;
                }
                if self.solutions[start + i] != ty::Bivariant
                    && !visitor.grounded_params.contains(&(i as u32))
                {
                    debug!(
                        "fix_purely_recursive_params: param {} of {:?} has no grounding use \
                        (was {:?}), will reset to Bivariant",
                        i,
                        def_id,
                        self.solutions[start + i]
                    );
                    to_reset.borrow_mut().push(start + i);
                }
            }
            true
        });

        // Second pass: apply the resets.
        for idx in to_reset.into_inner() {
            self.solutions[idx] = ty::Bivariant;
        }
    }

    fn enforce_const_invariance(&self, generics: &ty::Generics, variances: &mut [ty::Variance]) {
        let tcx = self.terms_cx.tcx;

        // Make all const parameters invariant.
        for param in generics.own_params.iter() {
            if let ty::GenericParamDefKind::Const { .. } = param.kind {
                variances[param.index as usize] = ty::Invariant;
            }
        }

        // Make all the const parameters in the parent invariant (recursively).
        if let Some(def_id) = generics.parent {
            self.enforce_const_invariance(tcx.generics_of(def_id), variances);
        }
    }

    fn create_map(&self) -> DefIdMap<&'tcx [ty::Variance]> {
        let tcx = self.terms_cx.tcx;

        let solutions = &self.solutions;
        DefIdMap::from(self.terms_cx.inferred_starts.items().map(
            |(&def_id, &InferredIndex(start))| {
                let generics = tcx.generics_of(def_id);
                let count = generics.count();

                let variances = tcx.arena.alloc_slice(&solutions[start..(start + count)]);

                // Const parameters are always invariant.
                self.enforce_const_invariance(generics, variances);

                // Functions are permitted to have unused generic parameters: make those invariant.
                if let ty::FnDef(..) = tcx.type_of(def_id).instantiate_identity().kind() {
                    for variance in variances.iter_mut() {
                        if *variance == ty::Bivariant {
                            *variance = ty::Invariant;
                        }
                    }
                }

                (def_id.to_def_id(), &*variances)
            },
        ))
    }

    fn evaluate(&self, term: VarianceTermPtr<'a>) -> ty::Variance {
        match *term {
            ConstantTerm(v) => v,

            TransformTerm(t1, t2) => {
                let v1 = self.evaluate(t1);
                let v2 = self.evaluate(t2);
                v1.xform(v2)
            }

            InferredTerm(InferredIndex(index)) => self.solutions[index],
        }
    }
}
