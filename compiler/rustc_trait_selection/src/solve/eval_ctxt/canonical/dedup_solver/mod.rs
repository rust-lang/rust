#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unreachable_code)]
#![allow(unused_imports)]
use crate::infer::canonical::{Canonical, CanonicalVarInfos};
use crate::infer::region_constraints::MemberConstraint;
use crate::solve::{ExternalConstraintsData, Response};
use rustc_middle::ty::{TyCtxt, UniverseIndex};

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use std::hash::Hash;
use std::ops::Deref;

mod constraint_walker;
mod solver;
use constraint_walker::{ConstraintWalker, Outlives};

pub struct Deduper<'tcx> {
    tcx: TyCtxt<'tcx>,
    rule_vars: Vec<Vec<usize>>,
    rule_cats: FxIndexMap<ConstraintType<'tcx>, Vec<usize>>,
    /// Maps a constraint index (the index inside constraint_vars) back to its index in outlives
    indx_to_outlives: FxHashMap<usize, usize>,
    /// Maps a constraint index (the index inside constraint_vars) back to its index in member_constraints
    indx_to_members: FxHashMap<usize, usize>,
}
#[derive(Debug, PartialEq, Eq, Hash)]
enum ConstraintType<'tcx> {
    Outlives(Outlives<'tcx>),
    Member(MemberConstraint<'tcx>),
}

impl<'tcx> Deduper<'tcx> {
    pub fn dedup(tcx: TyCtxt<'tcx>, input: &mut Canonical<'tcx, Response<'tcx>>) {
        let mut constraints = input.value.external_constraints.deref().clone();
        let mut deduper = Self {
            tcx,
            rule_vars: Vec::new(),
            rule_cats: FxIndexMap::default(),
            indx_to_outlives: FxHashMap::default(),
            indx_to_members: FxHashMap::default(),
        };
        deduper.dedup_internal(&mut constraints, &mut input.variables, &mut input.max_universe);
        input.value.external_constraints = tcx.mk_external_constraints(constraints);
    }
    fn dedup_internal(
        &mut self,
        constraints: &mut ExternalConstraintsData<'tcx>,
        variables: &mut CanonicalVarInfos<'tcx>,
        max_universe: &mut UniverseIndex,
    ) {
        dedup_exact_eq(&mut constraints.region_constraints.outlives);
        dedup_exact_eq(&mut constraints.region_constraints.member_constraints);

        let dedupable_vars: FxIndexSet<usize> = variables
            .iter()
            .enumerate()
            .filter(|(_, var)| var.universe() > UniverseIndex::ROOT)
            .map(|(indx, _)| indx)
            .collect();

        self.extract_constraint_data(&dedupable_vars, constraints, variables);

        let rule_vars = std::mem::take(&mut self.rule_vars);
        let rule_cats = std::mem::take(&mut self.rule_cats).into_values().collect::<Vec<_>>();
        let unremovable_vars: FxIndexSet<usize> =
            (0..variables.len()).filter(|x| !dedupable_vars.contains(x)).collect();

        let solve_result = solver::DedupSolver::dedup(rule_vars, rule_cats, unremovable_vars);
        self.remove_duplicate_constraints(&solve_result.removed_constraints, constraints);
        self.compress_variables(&solve_result.removed_vars, constraints, variables, max_universe);
    }
    // Extracts data about each constraint, i.e. the variables present, as well as the constraint
    // categories
    fn extract_constraint_data(
        &mut self,
        dedupable_vars: &FxIndexSet<usize>,
        constraints: &mut ExternalConstraintsData<'tcx>,
        variables: &mut CanonicalVarInfos<'tcx>,
    ) {
        let num_vars = variables.len();
        // dummy_var_rewriter is the fetch_var function that will be given to ConstraintWalker
        // it re-writes all variables with a dummy value (num_vars - guaranteed to NOT be a var index),
        // allowing us to compare constraints based solely on their structure, not on the variables present
        // Used to compute constraint categories
        let mut dummy_var_rewriter = |var| num_vars;
        /*
        let mut dummy_var_rewriter = |var| {
            if dedupable_vars.contains(&var) {
                return num_vars;
            }
            var
        };
        */
        for (indx, outlives) in constraints.region_constraints.outlives.iter().enumerate() {
            let mut extractor = ConstraintWalker::new(self.tcx, &mut dummy_var_rewriter);
            let erased = ConstraintType::Outlives(extractor.walk_outlives(&outlives.0));
            let vars = std::mem::take(&mut extractor.vars);
            if vars.is_empty() {
                continue;
            }
            self.process_constraint_data(indx, erased, vars);
        }
        for (indx, member) in constraints.region_constraints.member_constraints.iter().enumerate() {
            let mut extractor = ConstraintWalker::new(self.tcx, &mut dummy_var_rewriter);
            let erased = ConstraintType::Member(extractor.walk_members(member));
            let vars = std::mem::take(&mut extractor.vars);
            if vars.is_empty() {
                continue;
            }
            self.process_constraint_data(indx, erased, vars);
        }
    }
    fn process_constraint_data(
        &mut self,
        input_indx: usize,
        erased: ConstraintType<'tcx>,
        vars: Vec<usize>,
    ) {
        self.rule_vars.push(vars);
        let constraint_indx = self.rule_vars.len() - 1;
        match &erased {
            ConstraintType::Outlives(_) => &mut self.indx_to_outlives,
            ConstraintType::Member(_) => &mut self.indx_to_members,
        }
        .insert(constraint_indx, input_indx);
        self.rule_cats.entry(erased).or_insert_with(Vec::new).push(constraint_indx);
    }
    fn remove_duplicate_constraints(
        &mut self,
        to_remove: &FxIndexSet<usize>,
        constraints: &mut ExternalConstraintsData<'tcx>,
    ) {
        let mut remove_outlives: FxIndexSet<usize> =
            to_remove.iter().filter_map(|x| self.indx_to_outlives.get(x)).cloned().collect();
        let mut remove_members: FxIndexSet<usize> =
            to_remove.iter().filter_map(|x| self.indx_to_members.get(x)).cloned().collect();
        remove_outlives.sort();
        remove_members.sort();

        for indx in remove_outlives.into_iter().rev() {
            constraints.region_constraints.outlives.swap_remove(indx);
        }
        for indx in remove_members.into_iter().rev() {
            constraints.region_constraints.member_constraints.swap_remove(indx);
        }
    }
    fn compress_variables(
        &mut self,
        removed_vars: &FxIndexSet<usize>,
        constraints: &mut ExternalConstraintsData<'tcx>,
        variables: &mut CanonicalVarInfos<'tcx>,
        max_universe: &mut UniverseIndex,
    ) {
        let mut vars = variables.as_slice().to_vec();
        let mut universes_available: FxIndexSet<UniverseIndex> =
            vars.iter().map(|x| x.universe()).collect();
        universes_available.sort();

        let mut compressed_vars: FxHashMap<usize, usize> = FxHashMap::default();
        let mut universes_used: FxIndexSet<UniverseIndex> = FxIndexSet::default();

        let mut num_removed = 0;
        let mut var_indx = 0;
        while var_indx < vars.len() {
            let original_var_indx = var_indx + num_removed;
            if removed_vars.contains(&original_var_indx) {
                num_removed += 1;
                vars.remove(var_indx);
                continue;
            }
            compressed_vars.insert(original_var_indx, var_indx);
            universes_used.insert(vars[var_indx].universe());
            var_indx += 1;
        }
        universes_used.sort();

        for var in vars.iter_mut() {
            *var = var.with_updated_universe(
                *universes_available
                    .get_index(universes_used.get_index_of(&var.universe()).unwrap())
                    .unwrap(),
            );
        }

        let mut var_rewriter = |var| compressed_vars.get(&var).cloned().unwrap_or(var);
        for outlives in constraints.region_constraints.outlives.iter_mut() {
            let mut walker = ConstraintWalker::new(self.tcx, &mut var_rewriter);
            outlives.0 = walker.walk_outlives(&outlives.0);
        }
        for member in constraints.region_constraints.member_constraints.iter_mut() {
            let mut walker = ConstraintWalker::new(self.tcx, &mut var_rewriter);
            *member = walker.walk_members(member);
        }

        *variables = self.tcx.mk_canonical_var_infos(&vars);
        *max_universe = UniverseIndex::from(universes_used.len().saturating_sub(1));
    }
}

fn dedup_exact_eq<T: PartialEq>(input: &mut Vec<T>) {
    let mut indx = 0;
    while indx < input.len() {
        if input.iter().skip(indx + 1).any(|x| x == &input[indx]) {
            input.swap_remove(indx);
            continue;
        }
        indx += 1;
    }
}
