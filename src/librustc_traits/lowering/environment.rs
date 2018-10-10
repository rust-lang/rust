// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::traits::{
    Clause,
    Clauses,
    DomainGoal,
    FromEnv,
    Goal,
    ProgramClause,
    Environment,
};
use rustc::ty::{TyCtxt, Ty};
use rustc_data_structures::fx::FxHashSet;

struct ClauseVisitor<'set, 'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    round: &'set mut FxHashSet<Clause<'tcx>>,
}

impl ClauseVisitor<'set, 'a, 'tcx> {
    fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>, round: &'set mut FxHashSet<Clause<'tcx>>) -> Self {
        ClauseVisitor {
            tcx,
            round,
        }
    }

    fn visit_ty(&mut self, _ty: Ty<'tcx>) {

    }

    fn visit_from_env(&mut self, from_env: FromEnv<'tcx>) {
        match from_env {
            FromEnv::Trait(predicate) => {
                self.round.extend(
                    self.tcx.program_clauses_for(predicate.def_id())
                        .iter()
                        .cloned()
                );
            }

            FromEnv::Ty(ty) => self.visit_ty(ty),
        }
    }

    fn visit_domain_goal(&mut self, domain_goal: DomainGoal<'tcx>) {
        if let DomainGoal::FromEnv(from_env) = domain_goal {
            self.visit_from_env(from_env);
        }
    }

    fn visit_goal(&mut self, goal: Goal<'tcx>) {
        match goal {
            Goal::Implies(clauses, goal) => {
                for clause in clauses {
                    self.visit_clause(*clause);
                }
                self.visit_goal(*goal);
            }

            Goal::And(goal1, goal2) => {
                self.visit_goal(*goal1);
                self.visit_goal(*goal2);
            }

            Goal::Not(goal) => self.visit_goal(*goal),
            Goal::DomainGoal(domain_goal) => self.visit_domain_goal(domain_goal),
            Goal::Quantified(_, goal) => self.visit_goal(**goal.skip_binder()),
            Goal::CannotProve => (),
        }
    }

    fn visit_program_clause(&mut self, clause: ProgramClause<'tcx>) {
        self.visit_domain_goal(clause.goal);
        for goal in clause.hypotheses {
            self.visit_goal(*goal);
        }
    }

    fn visit_clause(&mut self, clause: Clause<'tcx>) {
        match clause {
            Clause::Implies(clause) => self.visit_program_clause(clause),
            Clause::ForAll(clause) => self.visit_program_clause(*clause.skip_binder()),
        }
    }
}

crate fn program_clauses_for_env<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    environment: Environment<'tcx>,
) -> Clauses<'tcx> {
    debug!("program_clauses_for_env(environment={:?})", environment);

    let mut last_round = FxHashSet();
    {
        let mut visitor = ClauseVisitor::new(tcx, &mut last_round);
        for clause in environment.clauses {
            visitor.visit_clause(*clause);
        }
    }

    let mut closure = last_round.clone();
    let mut next_round = FxHashSet();
    while !last_round.is_empty() {
        let mut visitor = ClauseVisitor::new(tcx, &mut next_round);
        for clause in last_round {
            visitor.visit_clause(clause);
        }
        last_round = next_round.drain()
            .filter(|&clause| closure.insert(clause))
            .collect();
    }

    debug!("program_clauses_for_env: closure = {:#?}", closure);

    return tcx.mk_clauses(
        closure.into_iter()
    );
}
