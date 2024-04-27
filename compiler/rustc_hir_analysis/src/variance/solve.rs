//! Constraint solving
//!
//! The final phase iterates over the constraints, refining the variance
//! for each inferred until a fixed point is reached. This will be the
//! optimal solution to the constraints. The final variance for each
//! inferred is then written into the `variance_map` in the tcx.

use rustc_hir::def_id::DefIdMap;
use rustc_middle::ty;

use super::constraints::*;
use super::terms::VarianceTerm::*;
use super::terms::*;
use super::xform::*;

struct SolveContext<'a, 'tcx> {
    terms_cx: TermsContext<'a, 'tcx>,
    constraints: Vec<Constraint<'a>>,

    // Maps from an InferredIndex to the inferred value for that variable.
    solutions: Vec<ty::Variance>,
}

pub fn solve_constraints<'tcx>(
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

    fn enforce_const_invariance(&self, generics: &ty::Generics, variances: &mut [ty::Variance]) {
        let tcx = self.terms_cx.tcx;

        // Make all const parameters invariant.
        for param in generics.params.iter() {
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
