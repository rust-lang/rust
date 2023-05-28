use crate::infer::canonical::QueryRegionConstraints;
use crate::infer::region_constraints::MemberConstraint;
use rustc_infer::infer::InferCtxt;
use rustc_middle::ty;
use ty::{subst::GenericArg, Region, UniverseIndex};

use rustc_data_structures::fx::{FxHashMap, FxIndexMap, FxIndexSet};
use rustc_index::IndexVec;
use std::hash::Hash;

mod constraint_walker;
mod solver;
use constraint_walker::{DedupWalker, DedupableIndexer};
use solver::{ConstraintIndex, DedupSolver, VarIndex};

pub struct Deduper<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    max_nameable_universe: UniverseIndex,

    var_indexer: DedupableIndexer<'tcx>,
    constraint_vars: IndexVec<ConstraintIndex, Vec<VarIndex>>,
    /// Constraints that are identical except for the value of their variables are grouped into the same clique
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
pub type Outlives<'tcx> = ty::OutlivesPredicate<GenericArg<'tcx>, Region<'tcx>>;

impl<'a, 'tcx> Deduper<'a, 'tcx> {
    pub fn dedup(
        infcx: &'a InferCtxt<'tcx>,
        max_nameable_universe: UniverseIndex,
        constraints: &mut QueryRegionConstraints<'tcx>,
    ) {
        let mut deduper = Self {
            infcx,
            max_nameable_universe,

            var_indexer: DedupableIndexer::new(),
            constraint_vars: IndexVec::default(),
            constraint_cliques: FxIndexMap::default(),
            indx_to_outlives: FxHashMap::default(),
            indx_to_members: FxHashMap::default(),
        };
        deduper.dedup_internal(constraints);
    }
    fn dedup_internal(&mut self, constraints: &mut QueryRegionConstraints<'tcx>) {
        fn dedup_exact<T: Clone + Hash + Eq>(input: &mut Vec<T>) {
            *input = FxIndexSet::<T>::from_iter(input.clone()).into_iter().collect();
        }
        dedup_exact(&mut constraints.outlives);
        dedup_exact(&mut constraints.member_constraints);

        self.lower_constraints_into_solver(constraints);
        let constraint_vars = std::mem::take(&mut self.constraint_vars);
        let constraint_cliques =
            std::mem::take(&mut self.constraint_cliques).into_iter().map(|x| x.1).collect();
        let var_universes = std::mem::take(&mut self.var_indexer.var_universes)
            .into_iter()
            .map(|(var, uni)| (VarIndex::from(var), uni.index()))
            .collect();
        let removed = DedupSolver::dedup(constraint_vars, constraint_cliques, var_universes)
            .removed_constraints;

        let mut removed_outlives =
            removed.iter().filter_map(|x| self.indx_to_outlives.get(x)).collect::<Vec<_>>();
        let mut removed_members =
            removed.iter().filter_map(|x| self.indx_to_members.get(x)).collect::<Vec<_>>();
        removed_outlives.sort();
        removed_members.sort();

        for removed_outlive in removed_outlives.into_iter().rev() {
            constraints.outlives.swap_remove(*removed_outlive);
        }
        for removed_member in removed_members.into_iter().rev() {
            constraints.member_constraints.swap_remove(*removed_member);
        }
    }
    fn lower_constraints_into_solver(&mut self, constraints: &QueryRegionConstraints<'tcx>) {
        for (outlives_indx, outlives) in constraints.outlives.iter().enumerate() {
            let (erased, vars) = DedupWalker::erase_dedupables(
                self.infcx,
                &mut self.var_indexer,
                self.max_nameable_universe,
                outlives.0.clone(),
            );
            self.insert_constraint(vars, ConstraintType::Outlives(erased), outlives_indx);
        }
        for (member_indx, member) in constraints.member_constraints.iter().enumerate() {
            let (erased, vars) = DedupWalker::erase_dedupables(
                self.infcx,
                &mut self.var_indexer,
                self.max_nameable_universe,
                member.clone(),
            );
            self.insert_constraint(vars, ConstraintType::Member(erased), member_indx);
        }
    }
    fn insert_constraint(
        &mut self,
        vars: Vec<usize>,
        erased: ConstraintType<'tcx>,
        original_indx: usize,
    ) {
        if vars.is_empty() {
            return;
        }
        let constraint_indx = self.constraint_vars.next_index();
        match erased {
            ConstraintType::Outlives(_) => {
                self.indx_to_outlives.insert(constraint_indx, original_indx)
            }
            ConstraintType::Member(_) => {
                self.indx_to_members.insert(constraint_indx, original_indx)
            }
        };
        self.constraint_vars.push(vars.into_iter().map(VarIndex::from).collect());
        self.constraint_cliques.entry(erased).or_insert_with(Vec::new).push(constraint_indx);
    }
}
