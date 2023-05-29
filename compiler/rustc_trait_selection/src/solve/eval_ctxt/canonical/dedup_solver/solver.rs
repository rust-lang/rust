const MAX_DFS_DEPTH: usize = 6; // lol

use rustc_data_structures::fx::{FxHashMap, FxIndexMap, FxIndexSet};
use rustc_index::{Idx, IndexVec};
use std::cell::RefCell;
use std::collections::BTreeMap;

#[cfg(test)]
mod tests;

rustc_index::newtype_index! { pub struct VarIndex {} }
rustc_index::newtype_index! { pub struct ConstraintIndex {} }
rustc_index::newtype_index! {
    /// Identifies a clique in the dedup graph by an index
    /// Two constraints can potentially be merged if and only if they belong to the same clique
    pub struct CliqueIndex {}
}
rustc_index::newtype_index! { pub struct MappingIndex {} }

#[derive(Debug)]
pub struct DedupSolver {
    /// The variables present in each constraint - the inner vec contains the variables, in the order
    /// that they appear in the constraint. See solver/tests.rs for examples on how constraints are lowered to this format
    constraint_vars: IndexVec<ConstraintIndex, Vec<VarIndex>>,
    /// The cliques that constraints are partitioned into. Constraints can only be merged if they belong to the same clique,
    /// and it's impossible for a constraint to be in more than one clique
    constraint_cliques: IndexVec<CliqueIndex, Vec<ConstraintIndex>>,
    /// The universes each var resides in. This is used because deduping prioritizes the removal of constraints
    /// that involve the highest universe indices
    var_universes: FxHashMap<VarIndex, usize>,

    /// The below are internal variables used in the solving process:

    /// All the mappings that can possibly be taken
    mappings: FxIndexMap<Mapping, MappingEval>,
    /// Constraints that have already been removed by deduplication
    removed_constraints: RefCell<FxIndexSet<ConstraintIndex>>,
    /// All of the currently applied var mappings, summed together
    applied_mappings: RefCell<Mapping>,
}
#[derive(Debug)]
pub struct DedupResult {
    pub removed_constraints: FxIndexSet<ConstraintIndex>,
}
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct Mapping(BTreeMap<VarIndex, VarIndex>);
#[derive(Debug, Clone, PartialEq, Eq)]
enum MappingEval {
    /// If a mapping can be applied just as-is, the mapping eval result is of variant `Removes`, which
    /// contains the set of constraints that the mapping will remove. The meaning of "as-is" is explained
    /// in the doc for the next variant
    Removes(FxIndexSet<ConstraintIndex>),
    /// A mapping has dependencies if it only maps a subset of the variables in a constraint, and therefore
    /// depends on another mapping to complete the full mapping. For example, if we have the constraints
    /// [1, 2] (index 1), [11, 12] (index 2), [2, 3] (index 3), and [12, 13] (index 4), the mapping
    /// that merges the 1st constraint into the 2nd (mapping vars 1 to 11, 2 to 12) will also "partially"
    /// map the 3rd constraint (because the mapping maps var 2, which the 3rd constraint contains).
    /// This will partially map the 3rd constraint into [12, 3], which isn't a pre-existing constraint - HOWEVER,
    /// if we also apply the mapping var2->var12,var3->var13, then it maps the constraint to [12, 13] which *is*
    /// a preexisting constraint. Therefore, the "mappabiilty" of the two constraints depends on that of the other.
    /// `DependsOn` is a set of vecs because a constraint can depend on multiple other cnostraints being mapped - each
    /// constraint that has to be mapped gets its own Vec that stores the list of Mappings that will satisfy the dependency
    DependsOn(FxIndexSet<Vec<MappingIndex>>),
    /// A temporary, intermediate variant used to calculate the actual contents of the `DependsOn` variant
    InterimDependencies {
        removes: FxIndexSet<ConstraintIndex>,
        depends_on: FxIndexSet<ConstraintIndex>,
    },
}

impl DedupSolver {
    pub fn dedup(
        constraint_vars: IndexVec<ConstraintIndex, Vec<VarIndex>>,
        constraint_cliques: IndexVec<CliqueIndex, Vec<ConstraintIndex>>,
        var_universes: FxHashMap<VarIndex, usize>,
    ) -> DedupResult {
        let mut deduper = Self {
            constraint_vars,
            constraint_cliques,
            var_universes,

            mappings: FxIndexMap::default(),
            removed_constraints: RefCell::new(FxIndexSet::default()),
            applied_mappings: RefCell::new(Mapping::from(&[], &[]).unwrap()),
        };
        deduper.compute_mappings();
        deduper.resolve_dependencies();

        DedupResult { removed_constraints: deduper.removed_constraints.into_inner() }
    }

    /// Computes the set of all possible mappings
    /// If a mapping has no dependencies, then it's eagerly taken, to increase performance
    /// If a mapping has dependencies, then it's added to `self.mappings`
    /// Deduplication can be done greedily because if two constraints can be merged, then they're
    /// equivalent in every way, including in relations to other constraints
    fn compute_mappings(&mut self) {
        for clique in self.constraint_cliques.iter() {
            for (n, constraint_1) in clique
                .iter()
                .enumerate()
                .filter(|x| !self.removed_constraints.borrow().contains(x.1))
            {
                let constraint_1_vars = &self.constraint_vars[*constraint_1];
                for constraint_2 in clique
                    .iter()
                    .skip(n + 1)
                    .filter(|x| !self.removed_constraints.borrow().contains(*x))
                {
                    let constraint_2_vars = &self.constraint_vars[*constraint_2];
                    // Maps constraint_1 to constraint_2
                    let forward = Mapping::from(constraint_1_vars, constraint_2_vars);
                    // Maps constraint_2 to constraint_1
                    let reverse = Mapping::from(constraint_2_vars, constraint_1_vars);
                    let (Ok(forward), Ok(reverse)) = (forward, reverse) else {
                        continue;
                    };

                    // If constraint_1 and constraint_2 can be merged, this relation should be
                    // bidirectional, i.e. we can merge 1 into 2 or 2 into 1
                    // For example, if a clique contains constraints [1, 2] and [11, 12] and another
                    // clique contains constraint [1], then we **cannot** merge vars 1 and 11 - the
                    // reverse direction (merging 11 into 1) works (if we map 11 to 1 and 12 to 2),
                    // but the forward direction (merging 1 into 11) doesn't work, as this would
                    // map the constraint [1] into [11], which is a constraint that doesn't exist
                    let (eval_forward, eval_reverse) =
                        (self.eval_mapping(&forward), self.eval_mapping(&reverse));
                    let (Ok(eval_forward), Ok(eval_reverse)) = (eval_forward, eval_reverse) else {
                        continue;
                    };

                    // Prioritize the removal of constraints referencing higher universes, because
                    // those are newer constraints (right?)
                    let max_forward_universe = forward.max_removed_universe(&self.var_universes);
                    let max_reverse_universe = reverse.max_removed_universe(&self.var_universes);
                    let (chosen_mapping, chosen_eval) =
                        if max_forward_universe >= max_reverse_universe {
                            (forward, eval_forward)
                        } else {
                            (reverse, eval_reverse)
                        };

                    // If there's dependencies, we add it to mappings and will think about it later
                    if matches!(&chosen_eval, MappingEval::InterimDependencies { .. }) {
                        self.mappings.insert(chosen_mapping, chosen_eval);
                        continue;
                    }
                    // Otherwise apply
                    let _ = self.try_apply_mapping(&chosen_mapping, &chosen_eval);
                }
            }
        }
        self.mappings.retain(|mapping, _| !self.applied_mappings.borrow().conflicts_with(mapping));
        self.resolve_interim_dependencies();
    }
    /// Evaluates the mapping. Can return Err if the mapping is invalid (i.e. it maps
    /// some constraints onto a constraint that doesn't exist, or conflicts with the
    /// mappings that were already greedily applied). Otherwise, returns a MappingEval
    fn eval_mapping(&self, mapping: &Mapping) -> Result<MappingEval, ()> {
        let mut eval = MappingEval::new();
        for clique in self.constraint_cliques.iter() {
            for constraint_1 in clique {
                let vars_1 = &self.constraint_vars[*constraint_1];
                if !mapping.affects_constraint(vars_1) {
                    continue;
                }
                let mut found_non_conflicting = false;
                for constraint_2 in clique.iter() {
                    let vars_2 = &self.constraint_vars[*constraint_2];
                    let Ok(trial_mapping) = Mapping::from(vars_1, vars_2) else {
                        continue;
                    };
                    if mapping.conflicts_with(&trial_mapping) {
                        continue;
                    }
                    found_non_conflicting = true;
                    // Only maps a subset of variables in constraint_1
                    if !mapping.contains_fully(&trial_mapping) {
                        // The input mapping can be applied only if there's another mapping that
                        // maps every variable in constraint_1 (and doesn't conflict with the input mapping)
                        eval.add_constraint_dependency(*constraint_1);
                        continue;
                    }
                    if *constraint_1 != *constraint_2 {
                        eval.add_constraint_removal(*constraint_1);
                    }
                }
                if !found_non_conflicting {
                    return Err(());
                }
            }
        }
        Ok(eval)
    }
    /// Currently, all dependencies are InterimDependencies(dep_set), where dep_set is the set of constraints
    /// that the mapping depends on. This function will convert those InterimDependencies into DependsOns,
    /// which contain information about which mappings can satisfy the dependencies
    fn resolve_interim_dependencies(&mut self) {
        self.mappings.retain(|mapping, mapping_eval| {
            // Remove mappings that conflict with already-applied mappings
            if self.applied_mappings.borrow().conflicts_with(mapping) {
                return false;
            }
            // If a constraint has already been removed by a pre-existing mapping, this current
            // mapping's dependency on this constraint has been resolved
            if let MappingEval::InterimDependencies { depends_on, .. } = mapping_eval {
                depends_on.retain(|x| !self.removed_constraints.borrow().contains(x));
            }
            true
        });
        // A map from a constraint to the mappings that will eliminate it (i.e. map it fully)
        let mut constraint_mappings: FxIndexMap<ConstraintIndex, Vec<MappingIndex>> =
            FxIndexMap::default();
        for (mapping_index, (_, mapping_info)) in self.mappings.iter().enumerate() {
            let Some(removes_constraints) = mapping_info.removes_constraints() else {
                continue;
            };
            for eliminated_constraint in removes_constraints {
                constraint_mappings
                    .entry(*eliminated_constraint)
                    .or_insert_with(Vec::new)
                    .push(MappingIndex::new(mapping_index));
            }
        }
        for (_, mapping_eval) in self.mappings.iter_mut() {
            mapping_eval.resolve_from_interim(&constraint_mappings);
        }
    }

    /// Resolves dependencies on mappings - i.e. we find sets of mappings that mutually satisfy each other's
    /// dependencies, and don't conflict, then apply these mappings
    fn resolve_dependencies(&mut self) {
        let mut used_mappings = FxIndexSet::default();
        for indx in (0..self.mappings.len()).map(MappingIndex::new) {
            if used_mappings.contains(&indx)
                || self.applied_mappings.borrow().conflicts_with(self.get_mapping(indx))
            {
                continue;
            }

            let applied_mappings = self.applied_mappings.borrow().clone();
            let starting_set = FxIndexSet::from_iter([indx]);
            if let Some(applied_mappings) = self.dfs_search(
                used_mappings.clone(),
                applied_mappings,
                starting_set,
                MAX_DFS_DEPTH,
            ) {
                let mut summed_mapping = self.applied_mappings.borrow().clone();
                for mapping in applied_mappings {
                    summed_mapping.join_mappings(&self.get_mapping(mapping));
                    used_mappings.insert(mapping);
                }
                let summed_eval = self.eval_mapping(&summed_mapping);
                assert!(summed_eval.is_ok());
                let application_result =
                    self.try_apply_mapping(&summed_mapping, &summed_eval.unwrap());
                assert!(application_result.is_ok());
            }
        }
    }
    /// Finds a set of mappings that mutually satisfy each other's dependencies without conflicting with each
    /// other or the mappings that have already been applied. It does this through depth first search -
    /// it takes a FxIndexSet of the mappings that have already been presumed to be part of the mapping set as well
    /// as a FxIndexSet of mappings that we are trying to add to this set (`from`). These mappings still may have
    /// dependencies that might be unresolved, so dfs_search attempts to resolve these dependencies, recursively calling
    /// itself if necessary. If `from` has no unresolved dependencies, then a set of mappings is found, and we return
    fn dfs_search(
        &self,
        mut used_mappings: FxIndexSet<MappingIndex>,
        mut applied_mappings: Mapping,
        mut from: FxIndexSet<MappingIndex>,
        remaining_depth: usize,
    ) -> Option<FxIndexSet<MappingIndex>> {
        if remaining_depth == 0 {
            return None;
        }

        // If we already applied a mapping, we now remove it from `from`, as its dependencies have
        // been resolved and therefore we don't need to worry about it
        from.retain(|x| !used_mappings.contains(x));
        used_mappings.extend(from.iter());

        // Apply the mappings that we're trying to apply (in `from`), aborting if there's any conflicts
        for mapping_indx in from.iter() {
            let (mapping, _) = self.mappings.get_index(mapping_indx.index()).unwrap();
            if applied_mappings.conflicts_with(mapping) {
                return None;
            }
            applied_mappings.join_mappings(&mapping);
        }

        // For each unresolved dependency, we have a list of Mappings that can resolve it
        let unresolved_dependencies: FxIndexSet<Vec<MappingIndex>> = FxIndexSet::from_iter(
            from.iter()
                .flat_map(|x| self.get_mapping_info(*x).get_mapping_dependencies().unwrap())
                .map(|mapping_choices| {
                    let mut new_choices = mapping_choices.clone();
                    new_choices.retain(|x| !self.get_mapping(*x).conflicts_with(&applied_mappings));
                    new_choices
                }),
        );

        // Wooo, no dependencies left to resolve!
        if unresolved_dependencies.is_empty() {
            return Some(used_mappings);
        }
        // At least one dependency has no viable resolution options (the Vec is empty) - failed
        if unresolved_dependencies.iter().any(|x| x.is_empty()) {
            return None;
        }
        // For each unresolved dependency, we have a list of Mappings that can resolve it. The idea is that
        // we need to pick one mapping from each list to create the set of mappings we're going to try adding
        // to the set of mappings, and right now, it's done incredibly naively - for each list of mappings,
        // we store the value that denotes the index of the mapping in that list that we're going to try
        // adding next. We start out with a vec of all zeroes, i.e. we try taking the first element of each
        // list, and we sweep the indices in order, e.g. going [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
        // [0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0], [0, 2, 0], so on and so forth
        let mut trial_indices = vec![0; unresolved_dependencies.len()];
        while *trial_indices.last().unwrap() < unresolved_dependencies.last().unwrap().len() {
            // The set of mappings that were chosen to be added next
            let choice: FxIndexSet<MappingIndex> =
                trial_indices.iter().zip(&unresolved_dependencies).map(|(x, y)| y[*x]).collect();
            let search_result = self.dfs_search(
                used_mappings.clone(),
                applied_mappings.clone(),
                choice,
                remaining_depth - 1,
            );
            if search_result.is_some() {
                return search_result;
            }

            // Advance the indices to the next possibility
            for (indx, options) in trial_indices.iter_mut().zip(&unresolved_dependencies) {
                *indx += 1;
                if *indx >= options.len() {
                    *indx = 0;
                    continue;
                }
                break;
            }
        }
        None
    }

    /// Tries to apply a mapping, returning Ok if it works, otherwise Err
    fn try_apply_mapping(&self, mapping: &Mapping, info: &MappingEval) -> Result<(), ()> {
        let MappingEval::Removes(removes) = info else {
            return Err(());
        };
        if self.applied_mappings.borrow().conflicts_with(mapping) {
            return Err(());
        }
        self.removed_constraints.borrow_mut().extend(removes.clone());
        self.applied_mappings.borrow_mut().join_mappings(&mapping);
        Ok(())
    }
    fn get_mapping(&self, index: MappingIndex) -> &Mapping {
        &self.mappings.get_index(index.index()).unwrap().0
    }
    fn get_mapping_info(&self, index: MappingIndex) -> &MappingEval {
        &self.mappings.get_index(index.index()).unwrap().1
    }
}

impl Mapping {
    /// Creates a mapping between two constraints. If the resulting mapping is invalid,
    /// an Err is returned
    fn from(from: &[VarIndex], to: &[VarIndex]) -> Result<Self, ()> {
        if from.len() != to.len() {
            return Err(());
        }
        let mut mapping_set = BTreeMap::new();
        for (from_var, to_var) in from.iter().zip(to) {
            if let Some(previous_map) = mapping_set.get(from_var) {
                if previous_map != to_var {
                    return Err(());
                }
                continue;
            }
            mapping_set.insert(*from_var, *to_var);
        }
        // We impose a constraint that a variable cannot be both a key and a value of
        // a mapping, as that would mean [1, 2] can be mapped onto [2, 1] - however,
        // these are fundamentally different constraints that can't be merged.
        // The only exception is if a var maps to itself - that's fine, as all it's saying
        // is that we want to fix a variable and don't map it
        if mapping_set.values().any(|x| mapping_set.get(x).unwrap_or(x) != x) {
            return Err(());
        }
        Ok(Self(mapping_set))
    }
    fn maps_var(&self, constraint: VarIndex) -> Option<VarIndex> {
        self.0.get(&constraint).map(|x| *x)
    }
    /// Returns whether the mapping will change the given constraint if applied
    fn affects_constraint(&self, constraint: &[VarIndex]) -> bool {
        constraint.iter().any(|x| self.maps_var(*x).unwrap_or(*x) != *x)
    }
    /// Returns whether a mapping is a superset of another mapping
    fn contains_fully(&self, other: &Self) -> bool {
        other.0.iter().all(|(from, to)| self.maps_var(*from) == Some(*to))
    }
    /// Returns whether a mapping conflicts with another mapping, i.e. they can't be applied together
    fn conflicts_with(&self, other: &Self) -> bool {
        for (from_a, to_a) in self.0.iter() {
            for (from_b, to_b) in other.0.iter() {
                // Maps the same key to different values - conflicts!
                let map_conflicts = from_a == from_b && to_a != to_b;
                // Map A maps var v to w, but map B maps w to v. Applying both maps together doesn't
                // remove any variables, but just shuffles them around, so we call this a conflict
                let not_productive =
                    to_b == from_a && from_a != to_a || to_a == from_b && from_b != to_b;
                if map_conflicts || not_productive {
                    return true;
                }
            }
        }
        false
    }
    fn max_removed_universe(&self, var_universes: &FxHashMap<VarIndex, usize>) -> usize {
        self.0.keys().map(|x| *var_universes.get(x).unwrap()).max().unwrap_or(0)
    }
    fn join_mappings(&mut self, other: &Self) {
        assert!(!self.conflicts_with(other));
        self.0.extend(&other.0);
    }
}
impl MappingEval {
    fn new() -> Self {
        Self::Removes(FxIndexSet::default())
    }
    fn add_constraint_removal(&mut self, constraint: ConstraintIndex) {
        match self {
            Self::Removes(removes) => {
                removes.insert(constraint);
            }
            Self::InterimDependencies { removes, .. } => {
                removes.insert(constraint);
            }
            _ => {}
        }
    }
    fn add_constraint_dependency(&mut self, constraint: ConstraintIndex) {
        if let Self::InterimDependencies { depends_on, .. } = self {
            depends_on.insert(constraint);
            return;
        }
        if let Self::Removes(removes) = self.clone() {
            *self = Self::InterimDependencies {
                removes: removes,
                depends_on: FxIndexSet::from_iter([constraint]),
            };
        }
    }
    fn removes_constraints<'a>(&'a self) -> Option<&'a FxIndexSet<ConstraintIndex>> {
        match self {
            Self::Removes(removes) => Some(removes),
            Self::InterimDependencies { removes, .. } => Some(removes),
            _ => None,
        }
    }
    fn resolve_from_interim(
        &mut self,
        constraint_mappings: &FxIndexMap<ConstraintIndex, Vec<MappingIndex>>,
    ) {
        match self.clone() {
            Self::InterimDependencies { depends_on, .. } => {
                let resolution_set = depends_on
                    .into_iter()
                    .map(|x| {
                        let mut maps =
                            constraint_mappings.get(&x).cloned().unwrap_or_else(Vec::new);
                        maps.sort();
                        maps
                    })
                    .collect::<FxIndexSet<_>>();
                *self = Self::DependsOn(resolution_set);
            }
            _ => {}
        }
    }
    fn get_mapping_dependencies<'a>(&'a self) -> Option<&'a FxIndexSet<Vec<MappingIndex>>> {
        match self {
            Self::DependsOn(set) => Some(set),
            _ => None,
        }
    }
}
