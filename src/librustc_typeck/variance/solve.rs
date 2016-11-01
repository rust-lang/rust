// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Constraint solving
//!
//! The final phase iterates over the constraints, refining the variance
//! for each inferred until a fixed point is reached. This will be the
//! optimal solution to the constraints. The final variance for each
//! inferred is then written into the `variance_map` in the tcx.

use rustc::ty;
use std::rc::Rc;

use super::constraints::*;
use super::terms::*;
use super::terms::VarianceTerm::*;
use super::xform::*;

struct SolveContext<'a, 'tcx: 'a> {
    terms_cx: TermsContext<'a, 'tcx>,
    constraints: Vec<Constraint<'a>>,

    // Maps from an InferredIndex to the inferred value for that variable.
    solutions: Vec<ty::Variance>,
}

pub fn solve_constraints(constraints_cx: ConstraintContext) {
    let ConstraintContext { terms_cx, constraints, .. } = constraints_cx;

    let solutions = terms_cx.inferred_infos
        .iter()
        .map(|ii| ii.initial_variance)
        .collect();

    let mut solutions_cx = SolveContext {
        terms_cx: terms_cx,
        constraints: constraints,
        solutions: solutions,
    };
    solutions_cx.solve();
    solutions_cx.write();
}

impl<'a, 'tcx> SolveContext<'a, 'tcx> {
    fn solve(&mut self) {
        // Propagate constraints until a fixed point is reached.  Note
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
                    debug!("Updating inferred {} (node {}) \
                            from {:?} to {:?} due to {:?}",
                           inferred,
                           self.terms_cx
                                   .inferred_infos[inferred]
                               .param_id,
                           old_value,
                           new_value,
                           term);

                    self.solutions[inferred] = new_value;
                    changed = true;
                }
            }
        }
    }

    fn write(&self) {
        // Collect all the variances for a particular item and stick
        // them into the variance map. We rely on the fact that we
        // generate all the inferreds for a particular item
        // consecutively (that is, we collect solutions for an item
        // until we see a new item id, and we assume (1) the solutions
        // are in the same order as the type parameters were declared
        // and (2) all solutions or a given item appear before a new
        // item id).

        let tcx = self.terms_cx.tcx;

        // Ignore the writes here because the relevant edges were
        // already accounted for in `constraints.rs`. See the section
        // on dependency graph management in README.md for more
        // information.
        let _ignore = tcx.dep_graph.in_ignore();

        let solutions = &self.solutions;
        let inferred_infos = &self.terms_cx.inferred_infos;
        let mut index = 0;
        let num_inferred = self.terms_cx.num_inferred();
        while index < num_inferred {
            let item_id = inferred_infos[index].item_id;

            let mut item_variances = vec![];

            while index < num_inferred && inferred_infos[index].item_id == item_id {
                let info = &inferred_infos[index];
                let variance = solutions[index];
                debug!("Index {} Info {} Variance {:?}",
                       index,
                       info.index,
                       variance);

                assert_eq!(item_variances.len(), info.index);
                item_variances.push(variance);
                index += 1;
            }

            debug!("item_id={} item_variances={:?}", item_id, item_variances);

            let item_def_id = tcx.map.local_def_id(item_id);

            // For unit testing: check for a special "rustc_variance"
            // attribute and report an error with various results if found.
            if tcx.has_attr(item_def_id, "rustc_variance") {
                span_err!(tcx.sess,
                          tcx.map.span(item_id),
                          E0208,
                          "{:?}",
                          item_variances);
            }

            let newly_added = tcx.item_variance_map
                .borrow_mut()
                .insert(item_def_id, Rc::new(item_variances))
                .is_none();
            assert!(newly_added);
        }
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
