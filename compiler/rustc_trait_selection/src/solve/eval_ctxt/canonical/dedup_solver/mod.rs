use crate::infer::canonical::{Canonical, CanonicalVarInfos};
use crate::infer::region_constraints::MemberConstraint;
use crate::solve::{ExternalConstraintsData, Response};
use rustc_middle::ty::{TyCtxt, UniverseIndex};

use rustc_data_structures::fx::{FxHashMap, FxIndexMap, FxIndexSet};
use rustc_index::{Idx, IndexVec};
use std::hash::Hash;
use std::ops::Deref;

mod constraint_walker;
mod solver;
use constraint_walker::{ConstraintWalker, Outlives};
use solver::{ConstraintIndex, DedupSolver, VarIndex};

pub struct Deduper<'tcx> {
    tcx: TyCtxt<'tcx>,
    constraint_vars: IndexVec<ConstraintIndex, Vec<VarIndex>>,
    constraint_cliques: FxIndexMap<ConstraintType<'tcx>, Vec<ConstraintIndex>>,
    /// Maps a constraint index (the index inside constraint_vars) back to its index in outlives
    indx_to_outlives: FxHashMap<ConstraintIndex, usize>,
    /// Maps a constraint index (the index inside constraint_vars) back to its index in member_constraints
    indx_to_members: FxHashMap<ConstraintIndex, usize>,
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
            constraint_vars: IndexVec::new(),
            constraint_cliques: FxIndexMap::default(),
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
        constraints.region_constraints.outlives =
            FxIndexSet::<_>::from_iter(constraints.region_constraints.outlives.clone().into_iter())
                .into_iter()
                .collect();
        constraints.region_constraints.member_constraints = FxIndexSet::<_>::from_iter(
            constraints.region_constraints.member_constraints.clone().into_iter(),
        )
        .into_iter()
        .collect();

        let dedupable_vars: FxIndexSet<VarIndex> = variables
            .iter()
            .enumerate()
            .filter(|(_, var)| var.universe() > UniverseIndex::ROOT)
            .map(|(indx, _)| VarIndex::new(indx))
            .collect();

        self.extract_constraint_data(constraints, variables);

        let constraint_vars = std::mem::take(&mut self.constraint_vars);
        let constraint_cliques =
            std::mem::take(&mut self.constraint_cliques).into_values().collect::<IndexVec<_, _>>();
        let unremovable_vars: FxIndexSet<_> = (0..variables.len())
            .map(VarIndex::new)
            .filter(|x| !dedupable_vars.contains(x))
            .collect();

        let solve_result =
            DedupSolver::dedup(constraint_vars, constraint_cliques, unremovable_vars);
        self.remove_duplicate_constraints(&solve_result.removed_constraints, constraints);
        self.compress_variables(&solve_result.removed_vars, constraints, variables, max_universe);
    }
    // Extracts data about each constraint, i.e. the variables present, as well as the constraint
    // categories
    fn extract_constraint_data(
        &mut self,
        constraints: &mut ExternalConstraintsData<'tcx>,
        variables: &mut CanonicalVarInfos<'tcx>,
    ) {
        let num_vars = variables.len();
        // dummy_var_rewriter is the fetch_var function that will be given to ConstraintWalker
        // it re-writes all variables with a dummy value (num_vars - guaranteed to NOT be a var index),
        // allowing us to compare constraints based solely on their structure, not on the variables present
        // Used to compute constraint categories
        let mut dummy_var_rewriter = |_| num_vars;

        for (indx, outlives) in constraints.region_constraints.outlives.iter().enumerate() {
            let mut extractor = ConstraintWalker::new(self.tcx, &mut dummy_var_rewriter);
            let erased = ConstraintType::Outlives(extractor.walk_outlives(&outlives.0));
            let vars: Vec<_> = extractor.vars.iter().map(|x| VarIndex::new(*x)).collect();
            if vars.is_empty() {
                continue;
            }
            self.process_constraint_data(indx, erased, vars);
        }
        for (indx, member) in constraints.region_constraints.member_constraints.iter().enumerate() {
            let mut extractor = ConstraintWalker::new(self.tcx, &mut dummy_var_rewriter);
            let erased = ConstraintType::Member(extractor.walk_members(member));
            let vars: Vec<_> = extractor.vars.iter().map(|x| VarIndex::new(*x)).collect();
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
        vars: Vec<VarIndex>,
    ) {
        let constraint_indx = self.constraint_vars.next_index();
        self.constraint_vars.push(vars);
        match &erased {
            ConstraintType::Outlives(_) => &mut self.indx_to_outlives,
            ConstraintType::Member(_) => &mut self.indx_to_members,
        }
        .insert(constraint_indx, input_indx);
        self.constraint_cliques.entry(erased).or_insert_with(Vec::new).push(constraint_indx);
    }
    fn remove_duplicate_constraints(
        &mut self,
        to_remove: &FxIndexSet<ConstraintIndex>,
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
        removed_vars: &FxIndexSet<VarIndex>,
        constraints: &mut ExternalConstraintsData<'tcx>,
        variables: &mut CanonicalVarInfos<'tcx>,
        max_universe: &mut UniverseIndex,
    ) {
        let mut vars = variables.as_slice().to_vec();
        let mut universes_available: FxIndexSet<UniverseIndex> =
            vars.iter().map(|x| x.universe()).collect();
        universes_available.sort();

        let mut compressed_vars: FxHashMap<VarIndex, usize> = FxHashMap::default();
        let mut universes_used: FxIndexSet<UniverseIndex> = FxIndexSet::default();

        let mut num_removed = 0;
        let mut var_indx = VarIndex::new(0);
        while var_indx.index() < vars.len() {
            let original_var_indx = var_indx.plus(num_removed);
            if removed_vars.contains(&original_var_indx) {
                num_removed += 1;
                vars.remove(var_indx.index());
                continue;
            }
            compressed_vars.insert(original_var_indx, var_indx.index());
            universes_used.insert(vars[var_indx.index()].universe());
            var_indx.increment_by(1);
        }
        universes_used.sort();

        for var in vars.iter_mut() {
            *var = var.with_updated_universe(
                *universes_available
                    .get_index(universes_used.get_index_of(&var.universe()).unwrap())
                    .unwrap(),
            );
        }

        let mut var_rewriter =
            |var| compressed_vars.get(&VarIndex::from(var)).cloned().unwrap_or(var);
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
