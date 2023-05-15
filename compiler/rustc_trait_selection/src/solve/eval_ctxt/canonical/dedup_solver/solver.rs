use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_index::{Idx, IndexVec};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

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
    /// The cliques that constraints are partitioned into. These are determined as follows: imagine a graph where
    /// each constraint is a vertex, and an edge exists between a pair of constraints if there is a many-to-one mapping of
    /// variables that perfectly maps one constraint onto the other. The constraint cliques are just cliques within this graph.
    /// By nature of this problem, it is impossible for a constraint to be in two cliques
    constraint_cliques: IndexVec<CliqueIndex, Vec<ConstraintIndex>>,
    /// A set of variables we cannot remove, i.e. they belong to a universe that the caller can name. We keep track of these
    /// to determine if there's a variable that we **can** remove that behaves like one of these, where in that case we just
    /// remove the unremovable var and keep the removable ones
    unremovable_vars: FxIndexSet<VarIndex>,

    /// The below are internal variables used in the solving process:

    /// All the mappings that can possibly be taken
    mappings: FxIndexMap<Mapping, MappingInfo>,
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
struct MappingInfo {
    /// The constraints that a mapping will eliminate. For example, if we have the constraints
    /// [1, 2] (with ConstraintIndex of 0) and [3, 4], the mapping 1:3,2:4 will eliminate constraint 0
    eliminated_constraints: FxIndexSet<ConstraintIndex>,
    /// A mapping has dependencies if it only maps a subset of the variables in a constraint, and therefore
    /// depends on another mapping to complete the full mapping. For example, if we have the constraints
    /// [1, 2] (index 1), [11, 12] (index 2), [2, 3] (index 3), and [12, 13] (index 4), the mapping
    /// that merges the 1st constraint into the 2nd (mapping vars 1 to 11, 2 to 12) will also "partially"
    /// map the 3rd constraint (because the mapping maps var 2, which the 3rd constraint contains).
    /// This will partially map the 3rd constraint into [12, 3], which isn't a pre-existing constraint - HOWEVER,
    /// if we also apply the mapping var2->var12,var3->var13, then it maps the constraint to [12, 13] which *is*
    /// a preexisting constraint. Therefore, the two constraints depend on each other
    dependencies: FxIndexMap<ConstraintIndex, BTreeSet<MappingIndex>>,
}
#[derive(Debug, PartialEq, Eq)]
enum MapEvalErr {
    Conflicts,
    Unremovable,
}

impl DedupSolver {
    pub fn dedup(
        constraint_vars: IndexVec<ConstraintIndex, Vec<VarIndex>>,
        constraint_cliques: IndexVec<CliqueIndex, Vec<ConstraintIndex>>,
        unremovable_vars: FxIndexSet<VarIndex>,
    ) -> DedupResult {
        let mut deduper = Self {
            constraint_vars,
            constraint_cliques,
            unremovable_vars,

            mappings: FxIndexMap::default(),
            removed_constraints: RefCell::new(FxIndexSet::default()),
            applied_mappings: RefCell::new(Mapping::from(&[], &[])),
        };
        deduper.refine_cliques();
        deduper.compute_mappings();
        deduper.resolve_dependencies();

        DedupResult { removed_constraints: deduper.removed_constraints.into_inner() }
    }
    /// The input cliques are provided just on a basis of the structure of the constraints, i.e.
    /// "are they the same if we ignore variables unnameable from the caller". However, just because
    /// this is the case doesn't mean that two constraints can be merged - for example, the constraint
    /// involving vars [1, 3] can't be merged with a constraint involving vars [2, 2].
    /// This function refines the cliques such that if we create a graph with constraints as vertices
    /// and edges if they can be merged, constraint_cliques now represents the **true** cliques of the
    /// graph, i.e. any two constrains in the same clique can now create a valid mapping
    fn refine_cliques(&mut self) {
        // Refine categories based on shape - see canonicalize_constraint_shape for more info
        for clique_indx in (0..self.constraint_cliques.len()).map(CliqueIndex::new) {
            let mut shape_cliques: FxIndexMap<Vec<usize>, CliqueIndex> = FxIndexMap::default();
            let mut constraint_indx = 0;
            while constraint_indx < self.constraint_cliques[clique_indx].len() {
                let constraint = self.constraint_cliques[clique_indx][constraint_indx];
                let shape =
                    Self::canonicalize_constraint_shape(&mut self.constraint_vars[constraint]);
                let is_first_entry = shape_cliques.is_empty();
                let new_clique = *shape_cliques.entry(shape).or_insert_with(|| {
                    if is_first_entry {
                        clique_indx
                    } else {
                        self.constraint_cliques.push(Vec::new());
                        CliqueIndex::new(self.constraint_cliques.len() - 1)
                    }
                });
                if new_clique == clique_indx {
                    constraint_indx += 1;
                    continue;
                }
                self.constraint_cliques[clique_indx].swap_remove(constraint_indx);
                self.constraint_cliques[new_clique].push(constraint);
            }
        }
        // Refine categories based on indices of variables. This is based on the observation that
        // if a variable V is present in a constraint C1 at some set of indices I, then a constraint
        // C2 can be merged with C1 only if one of the following cases are satisfied:
        //    a. V is present in constraint C2 at the **same** set of indices I, where in that case
        //       the variable mapping that merges these two constraints would just map V onto V
        //    b. V is not present in constraint C2 at all, in which case some other variable would
        //       be mapped onto V
        // If none of these above cases are true, that means we have a situation where we map V
        // to another variable U, and a variable W would be mapped onto V - in this case, we're just
        // shuffling variables around without actually eliminating any, which is unproductive and
        // hence an "invalid mapping"
        for clique_indx in (0..self.constraint_cliques.len()).map(CliqueIndex::new) {
            // First element of tuple (the FxIndexMap) maps a variable to
            // the index it occurs in
            let mut index_cliques: Vec<(FxIndexMap<VarIndex, usize>, CliqueIndex)> = Vec::new();
            let mut constraint_indx = 0;
            while constraint_indx < self.constraint_cliques[clique_indx].len() {
                let constraint = self.constraint_cliques[clique_indx][constraint_indx];
                let constraint_vars = &self.constraint_vars[constraint];
                let constraint_var_indices: FxIndexMap<VarIndex, usize> =
                    constraint_vars.iter().enumerate().map(|(indx, x)| (*x, indx)).collect();

                let mut found_clique = None;
                for (clique_vars, new_clique_indx) in index_cliques.iter_mut() {
                    let is_clique_member = constraint_vars
                        .iter()
                        .enumerate()
                        .all(|(indx, x)| *clique_vars.get(x).unwrap_or(&indx) == indx);
                    if !is_clique_member {
                        continue;
                    }
                    found_clique = Some(*new_clique_indx);
                    clique_vars.extend(&constraint_var_indices);
                    break;
                }
                let new_clique = found_clique.unwrap_or_else(|| {
                    if index_cliques.is_empty() {
                        clique_indx
                    } else {
                        let new_clique = self.constraint_cliques.next_index();
                        self.constraint_cliques.push(Vec::new());
                        index_cliques.push((constraint_var_indices, new_clique));
                        new_clique
                    }
                });
                if new_clique == clique_indx {
                    constraint_indx += 1;
                    continue;
                }
                self.constraint_cliques[clique_indx].swap_remove(constraint_indx);
                self.constraint_cliques[new_clique].push(constraint);
            }
        }
    }
    /// Returns the "shape" of a constraint, which captures information about the location(s) and
    /// multiplicity of variables in the constraint, irrespective of the actual variable indices
    /// For example, a constraint involving the vars [1, 1, 2, 3] has a shape of [0, 0, 1, 2],
    /// and a constraint involving vars [3, 4] has a shape of [0, 1]
    /// It takes a mutable reference to the vars because it also removes duplicates from
    /// the input vector after computing the shape
    /// Clearly, two constraints can be mapped onto each other only if they have the
    /// same shape
    fn canonicalize_constraint_shape(vars: &mut Vec<VarIndex>) -> Vec<usize> {
        let mut shape = Vec::new();
        let mut num_vars = 0;
        let mut indx = 0;
        while indx < vars.len() {
            if let Some(val) = shape.iter().find(|y| vars[**y] == vars[indx]) {
                shape.push(*val);
                vars.remove(indx);
            } else {
                shape.push(num_vars);
                num_vars += 1;
                indx += 1;
            }
        }
        shape
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
                for constraint_2 in clique
                    .iter()
                    .skip(n + 1)
                    .filter(|x| !self.removed_constraints.borrow().contains(*x))
                {
                    // Maps constraint_1 to constraint_2
                    let forward = Mapping::from(
                        &self.constraint_vars[*constraint_1],
                        &self.constraint_vars[*constraint_2],
                    );
                    // Maps constraint_2 to constraint_1
                    let reverse = Mapping::from(
                        &self.constraint_vars[*constraint_2],
                        &self.constraint_vars[*constraint_1],
                    );
                    if self.mappings.contains_key(&forward) || self.mappings.contains_key(&reverse)
                    {
                        continue;
                    }

                    // if constraint_1 and constraint_2 can be merged, this relation should be
                    // bidirectional, i.e. we can merge 1 into 2 or 2 into 1
                    // For example, if a clique contains constraints [1, 2] and [11, 12] and another
                    // clique contains constraint [1], then we **cannot** merge vars 1 and 11 - the
                    // reverse direction (merging 11 into 1) works (if we map 11 to 1 and 12 to 2),
                    // but the forward direction (merging 1 into 11) doesn't work, as this would
                    // map the constraint [1] into [11], which is a constraint that doesn't exist
                    let (eval_forward, eval_reverse) =
                        (self.eval_mapping(&forward), self.eval_mapping(&reverse));
                    if eval_forward == Err(MapEvalErr::Conflicts)
                        || eval_reverse == Err(MapEvalErr::Conflicts)
                    {
                        continue;
                    }
                    if let Ok(eval_forward) = eval_forward {
                        if self.try_apply_mapping(&forward, &eval_forward, false) == Err(true) {
                            self.mappings.insert(forward, eval_forward);
                        }
                    }
                    if let Ok(eval_reverse) = eval_reverse {
                        if self.try_apply_mapping(&reverse, &eval_reverse, false) == Err(true) {
                            self.mappings.insert(reverse, eval_reverse);
                        }
                    }
                }
            }
        }
        self.resolve_dependencies_to_mapping();
    }
    /// Evaluates the mapping. Can return None if the mapping is invalid (i.e. it maps
    /// some constraints onto a constraint that doesn't exist, or conflicts with the
    /// mappings that were already greedily applied). Otherwise, returns MappingInfo.
    /// MappingInfo can contain dependencies - these occur if a mapping *partially* maps
    /// a constraint onto another, so the mapping isn't immediately invalid, but we do need
    /// another mapping to complete that partial map for it to actually be valid
    fn eval_mapping(&self, mapping: &Mapping) -> Result<MappingInfo, MapEvalErr> {
        let maps_unremovable_var =
            mapping.0.iter().any(|(from, to)| self.unremovable_vars.contains(from) && from != to);

        let mut info = MappingInfo::new();
        for clique in self.constraint_cliques.iter() {
            for constraint_1 in clique {
                let vars_1 = &self.constraint_vars[*constraint_1];
                if !mapping.affects_constraint(vars_1) {
                    continue;
                }
                let mut found_non_conflicting = false;
                for constraint_2 in clique.iter() {
                    let vars_2 = &self.constraint_vars[*constraint_2];
                    let trial_mapping = Mapping::from(vars_1, vars_2);
                    if mapping.conflicts_with(&trial_mapping) {
                        continue;
                    }
                    found_non_conflicting = true;
                    // Only maps a subset of variables in constraint_1
                    if !mapping.contains_fully(&trial_mapping) {
                        // The input mapping can be applied only if there's another mapping that
                        // maps every variable in constraint_1 (and doesn't conflict with the input mapping)
                        info.dependencies.insert(*constraint_1, BTreeSet::default());
                        continue;
                    }
                    if *constraint_1 != *constraint_2 {
                        info.eliminated_constraints.insert(*constraint_1);
                    }
                }
                if !found_non_conflicting {
                    return Err(MapEvalErr::Conflicts);
                }
            }
        }
        if maps_unremovable_var {
            return Err(MapEvalErr::Unremovable);
        }
        Ok(info)
    }
    /// Currently, dependencies are in the form FxIndexMap<ConstraintIndex, Empty FxIndexSet>,
    /// where ConstraintIndex is the constraint we must *also* map in order to apply this mapping.
    /// We must populate the Empty FxIndexSet with a set of mappings that can map the constraint without
    /// conflicting with the current mapping
    fn resolve_dependencies_to_mapping(&mut self) {
        self.mappings.retain(|mapping, mapping_info| {
            // Remove mappings that conflict with already-applied mappings
            if self.applied_mappings.borrow().conflicts_with(mapping) {
                return false;
            }
            // If a constraint has already been removed by a pre-existing mapping, this current
            // mapping's dependency on this constraint has been resolved
            mapping_info
                .dependencies
                .retain(|dependency, _| !self.removed_constraints.borrow().contains(dependency));
            true
        });
        // A map from a constraint to the mappings that will eliminate it (i.e. map it fully)
        let mut constraint_mappings: FxIndexMap<ConstraintIndex, BTreeSet<MappingIndex>> =
            FxIndexMap::default();
        for (indx, (_, mapping_info)) in self.mappings.iter().enumerate() {
            for eliminated_constraint in mapping_info.eliminated_constraints.iter() {
                constraint_mappings
                    .entry(*eliminated_constraint)
                    .or_insert_with(BTreeSet::default)
                    .insert(MappingIndex::new(indx));
            }
        }
        for indx in (0..self.mappings.len()).map(MappingIndex::new) {
            let mapping = self.get_mapping(indx);
            let input_dependencies = &self.get_mapping_info(indx).dependencies;
            let mut dependencies = FxIndexSet::default();
            for (dependency, _) in input_dependencies.iter() {
                // The set of mappings that can resolve this dependency
                let mut resolve_options =
                    constraint_mappings.get(dependency).cloned().unwrap_or_else(BTreeSet::new);
                resolve_options.retain(|x| !mapping.conflicts_with(&self.get_mapping(*x)));
                dependencies.insert(resolve_options);
            }
            // After this point, the actual constraints that a dependency maps
            // stops mattering - all that matters is that the dependency *exists*
            let old_dependencies =
                &mut self.mappings.get_index_mut(indx.index()).unwrap().1.dependencies;
            *old_dependencies = dependencies
                .into_iter()
                .enumerate()
                .map(|(indx, x)| (ConstraintIndex::from(indx), x))
                .collect();
        }
    }

    /// Resolves dependencies on mappings - i.e. we find sets of mappings that mutually satisfy each other's
    /// dependencies, and don't conflict, then apply these mappings
    fn resolve_dependencies(&mut self) {
        let mut used_mappings = FxIndexSet::default();
        for indx in (0..self.mappings.len()).map(MappingIndex::new) {
            if used_mappings.contains(&indx) {
                continue;
            }
            if self.applied_mappings.borrow().conflicts_with(self.get_mapping(indx)) {
                continue;
            }
            let applied_mappings = self.applied_mappings.borrow().clone();
            let mut starting_set = FxIndexSet::default();
            starting_set.insert(indx);
            if let Some(applied_mappings) =
                self.dfs_search(used_mappings.clone(), applied_mappings, starting_set)
            {
                for mapping in applied_mappings {
                    if used_mappings.contains(&mapping) {
                        continue;
                    }
                    let application_result = self.try_apply_mapping(
                        self.get_mapping(mapping),
                        self.get_mapping_info(mapping),
                        true,
                    );
                    assert!(application_result.is_ok());
                    used_mappings.insert(mapping);
                }
            }
        }
    }
    /// Finds a set of mappings that mutually satisfy each other's dependencies without conflicting with each
    /// other, or the mappings that have already been applied. It does this through depth first search -
    /// it takes a FxIndexSet of the mappings that have already been presumed to be part of the mapping set as well
    /// as a FxIndexSet of mappings that we are trying to add to this set (`from`). These mappings still may have
    /// dependencies that might be unresolved, so dfs_search attempts to resolve these dependencies, recursively calling
    /// itself if necessary. If `from` has no unresolved dependencies, then a set of mappings is found, and we return
    fn dfs_search(
        &self,
        mut used_mappings: FxIndexSet<MappingIndex>,
        mut applied_mappings: Mapping,
        mut from: FxIndexSet<MappingIndex>,
    ) -> Option<FxIndexSet<MappingIndex>> {
        // Apply the mappings that we're trying to apply (in `from`), aborting if there's any conflicts
        for mapping_indx in from.iter() {
            let (mapping, _) = self.mappings.get_index(mapping_indx.index()).unwrap();
            if applied_mappings.conflicts_with(mapping) {
                return None;
            }
            applied_mappings.0.extend(&mapping.0);
        }
        // If we already applied a mapping, we now remove it from `from`, as its dependencies have
        // been resolved and therefore we don't need to worry about it
        from.retain(|x| !used_mappings.contains(x));
        if from.is_empty() {
            return Some(used_mappings);
        }
        used_mappings.extend(from.iter());

        // For each unresolved dependency, we have a list of Mappings that can resolve it
        let mut unresolved_dependencies: FxIndexSet<Vec<MappingIndex>> = FxIndexSet::default();
        for from_mapping in from.iter() {
            let resolve_options = self.get_mapping_info(*from_mapping).dependencies.values();
            let resolve_options = resolve_options.map(|x| {
                Vec::from_iter(
                    x.iter()
                        .cloned()
                        // Throw out mappings that conflict with the current `used_mappings` we're
                        // trying to satisfy the dependencies of
                        .filter(|x| !self.get_mapping(*x).conflicts_with(&applied_mappings)),
                )
            });
            unresolved_dependencies.extend(resolve_options);
        }
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
            let search_result =
                self.dfs_search(used_mappings.clone(), applied_mappings.clone(), choice);
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

    /// Tries to apply a mapping, returning Ok(()) if the application was a success, Err(true) if the
    /// application failed but only because of unresolved dependencies, and Err(false) if the application
    /// fails because of conflicts
    fn try_apply_mapping(
        &self,
        mapping: &Mapping,
        info: &MappingInfo,
        allow_dependencies: bool,
    ) -> Result<(), bool> {
        if !allow_dependencies && !info.dependencies.is_empty() {
            return Err(true);
        }
        if self.applied_mappings.borrow().conflicts_with(mapping) {
            return Err(false);
        }
        self.removed_constraints.borrow_mut().extend(info.eliminated_constraints.iter());
        self.applied_mappings.borrow_mut().0.extend(&mapping.0);
        Ok(())
    }
    fn get_mapping(&self, index: MappingIndex) -> &Mapping {
        &self.mappings.get_index(index.index()).unwrap().0
    }
    fn get_mapping_info(&self, index: MappingIndex) -> &MappingInfo {
        &self.mappings.get_index(index.index()).unwrap().1
    }
}

impl Mapping {
    fn from(from: &[VarIndex], to: &[VarIndex]) -> Self {
        Self(from.iter().zip(to).map(|(x, y)| (*x, *y)).collect())
    }
    fn maps_var(&self, constraint: VarIndex) -> Option<VarIndex> {
        self.0.get(&constraint).map(|x| *x)
    }
    fn affects_constraint(&self, constraint: &[VarIndex]) -> bool {
        constraint.iter().any(|x| self.maps_var(*x).unwrap_or(*x) != *x)
    }
    fn contains_fully(&self, other: &Self) -> bool {
        other.0.iter().all(|(from, to)| self.maps_var(*from) == Some(*to))
    }
    fn conflicts_with(&self, other: &Self) -> bool {
        for (from_a, to_a) in self.0.iter() {
            for (from_b, to_b) in other.0.iter() {
                let map_conflicts = from_a == from_b && to_a != to_b;
                let not_productive =
                    to_b == from_a && from_a != to_a || to_a == from_b && from_b != to_b;
                if map_conflicts || not_productive {
                    return true;
                }
            }
        }
        false
    }
}
impl MappingInfo {
    fn new() -> Self {
        Self { dependencies: FxIndexMap::default(), eliminated_constraints: FxIndexSet::default() }
    }
}
