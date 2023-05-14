use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
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
    dependencies: FxIndexMap<ConstraintIndex, FxIndexSet<MappingIndex>>,
    eliminated_constraints: FxIndexSet<ConstraintIndex>,
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
            applied_mappings: RefCell::new(Mapping::map_constraints(&[], &[])),
        };
        deduper.refine_cliques();
        deduper.compute_mappings();
        deduper.resolve_dependencies();

        DedupResult { removed_constraints: deduper.removed_constraints.into_inner() }
    }
    fn refine_cliques(&mut self) {
        // Refine categories based on shape
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
        // Refine categories based on indices of variables
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
        let mut invalid_maps: FxIndexSet<Mapping> = FxIndexSet::default();
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
                    let forward = Mapping::map_constraints(
                        &self.constraint_vars[*constraint_1],
                        &self.constraint_vars[*constraint_2],
                    );
                    if invalid_maps.contains(&forward) || self.mappings.contains_key(&forward) {
                        continue;
                    }
                    // Maps constraint_2 to constraint_1
                    let reverse = Mapping::map_constraints(
                        &self.constraint_vars[*constraint_2],
                        &self.constraint_vars[*constraint_1],
                    );
                    if invalid_maps.contains(&reverse) || self.mappings.contains_key(&reverse) {
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
                        invalid_maps.insert(forward);
                        invalid_maps.insert(reverse);
                        continue;
                    }
                    if let Ok(eval_forward) = eval_forward {
                        if !self.try_apply_mapping(&forward, &eval_forward, false) {
                            self.mappings.insert(forward, eval_forward);
                        }
                    }
                    if let Ok(eval_reverse) = eval_reverse {
                        if !self.try_apply_mapping(&reverse, &eval_reverse, false) {
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
                    let trial_mapping = Mapping::map_constraints(vars_1, vars_2);
                    if mapping.conflicts_with(&trial_mapping) {
                        continue;
                    }
                    found_non_conflicting = true;
                    // Only maps a subset of variables in constraint_1
                    if !mapping.contains_fully(&trial_mapping) {
                        // The input mapping can be applied only if there's another mapping that
                        // maps every variable in constraint_1 (and doesn't conflict with the input mapping)
                        info.dependencies.insert(*constraint_1, FxIndexSet::default());
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
        let mut constraint_mappings: FxIndexMap<ConstraintIndex, FxIndexSet<MappingIndex>> =
            FxIndexMap::default();
        for (indx, (_, mapping_info)) in self.mappings.iter().enumerate() {
            for eliminated_constraint in mapping_info.eliminated_constraints.iter() {
                constraint_mappings
                    .entry(*eliminated_constraint)
                    .or_insert_with(FxIndexSet::default)
                    .insert(MappingIndex::new(indx));
            }
        }
        for indx in (0..self.mappings.len()).map(MappingIndex::new) {
            let mapping = self.get_mapping(indx);
            let input_dependencies = &self.get_mapping_info(indx).dependencies;
            let mut dependencies = IndexVec::new();
            for (dependency, _) in input_dependencies.iter() {
                // The set of mappings that can resolve this dependency
                let mut resolve_options = FxIndexSet::default();
                if let Some(resolve_mappings) = constraint_mappings.get(dependency) {
                    resolve_options.extend(
                        resolve_mappings
                            .iter()
                            .filter(|x| !mapping.conflicts_with(&self.get_mapping(**x)))
                            .cloned(),
                    )
                }
                // Don't duplicate dependency groups
                if dependencies.iter().any(|x| x == &resolve_options) {
                    continue;
                }
                dependencies.push(resolve_options);
            }
            // After this point, the actual constraints that a dependency maps
            // stops mattering - all that matters is that the dependency *exists*
            self.mappings.get_index_mut(indx.index()).unwrap().1.dependencies =
                dependencies.into_iter_enumerated().collect();
        }
    }

    /// Resolves dependencies on mappings - i.e. we find a set of mappings that mutually satisfy each other's
    /// dependencies, and don't conflict
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
                    assert!(application_result);
                    used_mappings.insert(mapping);
                }
            }
        }
    }
    /// Performs a depth-first search on the dependencies graph of mappings. Each call to this functino
    /// takes in 3 arguments - the mappings that are already presumed to be part of the set of mappings
    /// that should be applied together, these mappings aggregated into a single mapping, and a `from` set,
    /// i.e. a set of mappings that we just added, and thus might have unresolved dependencies
    /// There's quite a few heuristics that will probably yield *significant* speedups
    /// I'll look into that later, if the rest of this approach is sound
    fn dfs_search(
        &self,
        mut used_mappings: FxIndexSet<MappingIndex>,
        mut applied_mappings: Mapping,
        from: FxIndexSet<MappingIndex>,
    ) -> Option<FxIndexSet<MappingIndex>> {
        for mapping_indx in from.iter() {
            let (mapping, _) = self.mappings.get_index(mapping_indx.index()).unwrap();
            if applied_mappings.conflicts_with(mapping) {
                return None;
            }
            applied_mappings.0.extend(&mapping.0);
        }
        if from.iter().all(|x| used_mappings.contains(x)) {
            return Some(used_mappings);
        }
        used_mappings.extend(from.iter());

        // For each unresolved dependency, we have a set of Mappings that can resolve it
        let unresolved_dependencies: Vec<&FxIndexSet<MappingIndex>> =
            from.iter().flat_map(|x| self.get_mapping_info(*x).dependencies.values()).collect();
        if unresolved_dependencies.is_empty() {
            return Some(used_mappings);
        }
        // For each unresolved dependency, we have an index denoting the index in the FxIndexSet that
        // we will try applying next
        // Essentially, we just sweep through all the combinations exhaustively, e.g. if we have 3
        // unresolved dependencies, each with 3 options, we would go "[0, 0, 0]", "[1, 0, 0]", "[2, 0, 0]",
        // "[0, 1, 0]", "[1, 1, 0]", "[2, 1, 0]", so on and so forth until we find one that succeeds
        let mut trial_indices = vec![0; unresolved_dependencies.len()];
        while *trial_indices.last().unwrap() < unresolved_dependencies.last().unwrap().len() {
            let choice: FxIndexSet<MappingIndex> = trial_indices
                .iter()
                .zip(&unresolved_dependencies)
                .map(|(x, y)| *y.get_index(*x).unwrap())
                .collect();
            let search_result =
                self.dfs_search(used_mappings.clone(), applied_mappings.clone(), choice);
            if search_result.is_some() {
                return search_result;
            }

            for (val, limit) in
                trial_indices.iter_mut().zip(unresolved_dependencies.iter().map(|x| x.len()))
            {
                *val += 1;
                if *val >= limit {
                    *val = 0;
                    continue;
                }
                break;
            }
        }
        None
    }

    fn try_apply_mapping(
        &self,
        mapping: &Mapping,
        info: &MappingInfo,
        allow_dependencies: bool,
    ) -> bool {
        if !allow_dependencies && !info.dependencies.is_empty() {
            return false;
        }
        if self.applied_mappings.borrow().conflicts_with(mapping) {
            return false;
        }
        self.removed_constraints.borrow_mut().extend(info.eliminated_constraints.iter());
        self.applied_mappings.borrow_mut().0.extend(&mapping.0);
        true
    }
    fn get_mapping(&self, index: MappingIndex) -> &Mapping {
        &self.mappings.get_index(index.index()).unwrap().0
    }
    fn get_mapping_info(&self, index: MappingIndex) -> &MappingInfo {
        &self.mappings.get_index(index.index()).unwrap().1
    }
}

impl Mapping {
    fn map_constraints(from: &[VarIndex], to: &[VarIndex]) -> Self {
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
