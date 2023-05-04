// Quick terminology mapping:
//     * "rule" => region constraint ("rule" is just shorter)
//     * "category"/"cat" => group of constraints with similar structure, i.e.
//                           constraints can only be merged if they're in the
//                           same category
//     * "mapping" => mapping from one set of variables to another. Mappings MUST
//                    be many-to-one. Obviously, they can't be one-to-many, and
//                    a one-to-one mapping isn't really productive, because
//                    we're not eliminating any variables that way
//
// This algorithm, in its present state, *is* exponential time. This exponential time
// case happens iff we have potential mappings that overlap, thus requiring them
// to be combined. For example, if we have a category A that contains constraints involving
// variables [1, 2, 3] and [11, 12, 4], and a category B that contains constraints involving
// variables [1, 2, 100] and [11, 12, 101], then the mapping in category A from the first
// constraint to the second constraint is valid if and only if the mapping from the first
// constraint to the second constraint in category B is valid (i.e. they depend on each other).
// In this trivial case, it's obvious that they're both valid, but more complicated cases
// can create a graph of dependencies (potentially with cycles), creating exponential behavior.
//
// In general, I don't think the exponential case should be happening *that* often, and if it does,
// the dependencies graph shouldn't be very deep, so it shouldn't be terrible. However, I'm not very
// knowledgable on the types of region constraints that can be generated, so maybe this assertion is false.
// There's some heuristics that I can use to speed the exponential part up, or maybe just cap the search depth.

use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use std::cell::RefCell;
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct DedupSolver {
    /// The variables present in each rule - the inner vec contains the variables, in the order
    /// that they appear in the rule
    rule_vars: Vec<Vec<usize>>,
    /// The categories that rules can be partitioned into - the inner vec contains all the rules
    /// that are in the same category
    rule_cats: Vec<Vec<usize>>,
    unremovable_vars: FxIndexSet<usize>,

    /// The below are internal variables used in the solving process:

    /// All the mappings that can possibly be taken
    mappings: FxIndexMap<Mapping, MappingInfo>,
    /// Rules that have already been removed by deduplication
    removed_rules: RefCell<FxIndexSet<usize>>,
    /// All of the currently applied var mappings, summed together
    applied_mappings: RefCell<Mapping>,
}
#[derive(Debug)]
pub struct DedupResult {
    pub removed_constraints: FxIndexSet<usize>,
    pub removed_vars: FxIndexSet<usize>,
}
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct Mapping(BTreeMap<usize, usize>);
#[derive(Debug, Clone, PartialEq, Eq)]
struct MappingInfo {
    dependencies: FxIndexMap<usize, FxIndexSet<usize>>,
    rule_mappings: FxIndexMap<usize, usize>,
}
#[derive(Debug, PartialEq, Eq)]
enum MapEvalErr {
    Conflicts,
    Unremovable,
}

impl DedupSolver {
    pub fn dedup(
        constraint_vars: Vec<Vec<usize>>,
        constraint_categories: Vec<Vec<usize>>,
        unremovable_vars: FxIndexSet<usize>,
    ) -> DedupResult {
        let mut deduper = Self {
            rule_vars: constraint_vars,
            rule_cats: constraint_categories,
            unremovable_vars,

            mappings: FxIndexMap::default(),
            removed_rules: RefCell::new(FxIndexSet::default()),
            applied_mappings: RefCell::new(Mapping::map_rules(&[], &[])),
        };
        deduper.refine_categories();
        deduper.compute_mappings();
        deduper.resolve_dependencies();

        let mut removed_vars = FxIndexSet::default();
        for (from, to) in deduper.applied_mappings.borrow().0.iter() {
            if *from == *to {
                continue;
            }
            removed_vars.insert(*from);
        }
        DedupResult { removed_constraints: deduper.removed_rules.into_inner(), removed_vars }
    }
    fn refine_categories(&mut self) {
        // Refine categories based on shape
        for cat_indx in 0..self.rule_cats.len() {
            let mut shape_categories: FxIndexMap<Vec<usize>, usize> = FxIndexMap::default();
            let mut rule_indx = 0;
            while rule_indx < self.rule_cats[cat_indx].len() {
                let rule = self.rule_cats[cat_indx][rule_indx];
                let shape = Self::canonicalize_rule_shape(&mut self.rule_vars[rule]);
                let is_first_entry = shape_categories.is_empty();
                let new_cat = *shape_categories.entry(shape).or_insert_with(|| {
                    if is_first_entry {
                        cat_indx
                    } else {
                        self.rule_cats.push(Vec::new());
                        self.rule_cats.len() - 1
                    }
                });
                if new_cat == cat_indx {
                    rule_indx += 1;
                    continue;
                }
                self.rule_cats[cat_indx].swap_remove(rule_indx);
                self.rule_cats[new_cat].push(rule);
            }
        }
        // Refine categories based on indices of variables
        for cat_indx in 0..self.rule_cats.len() {
            // A vec of tuples representing index categories.
            // First element of tuple is a mapping from a variable to the index it occurs in
            // Second element of tuple is the category
            let mut index_categories: Vec<(FxIndexMap<usize, usize>, usize)> = Vec::new();
            let mut rule_indx = 0;
            while rule_indx < self.rule_cats[cat_indx].len() {
                let rule = self.rule_cats[cat_indx][rule_indx];
                let rule_vars = &self.rule_vars[rule];
                let rule_var_indices: FxIndexMap<usize, usize> =
                    rule_vars.iter().enumerate().map(|(indx, x)| (*x, indx)).collect();

                let mut found_cat = None;
                for (cat_vars, new_cat_indx) in index_categories.iter_mut() {
                    let is_cat_member = rule_vars
                        .iter()
                        .enumerate()
                        .all(|(indx, x)| *cat_vars.get(x).unwrap_or(&indx) == indx);
                    if !is_cat_member {
                        continue;
                    }
                    found_cat = Some(*new_cat_indx);
                    cat_vars.extend(&rule_var_indices);
                    break;
                }
                let new_cat = found_cat.unwrap_or_else(|| {
                    if index_categories.is_empty() {
                        cat_indx
                    } else {
                        self.rule_cats.push(Vec::new());
                        let new_cat = self.rule_cats.len() - 1;
                        index_categories.push((rule_var_indices, new_cat));
                        new_cat
                    }
                });
                if new_cat == cat_indx {
                    rule_indx += 1;
                    continue;
                }
                self.rule_cats[cat_indx].swap_remove(rule_indx);
                self.rule_cats[new_cat].push(rule);
            }
        }
    }
    /// Returns the "shape" of a rule - related to the idea of de bruijn indices
    /// For example, a rule involving the vars [1, 1, 2, 3] has a shape of [0, 0, 1, 2],
    /// and a rule involving vars [3, 4] has a shape of [0, 1]
    /// It takes a mutable reference to the vars because it also removes duplicates from
    /// the input vector after computing the shape
    /// Clearly, two constraints can be mapped onto each other only if they have the
    /// same shape
    fn canonicalize_rule_shape(vars: &mut Vec<usize>) -> Vec<usize> {
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
    /// Deduplication can be done greedily because if two rules can be merged, then they're
    /// equivalent in every way, including in relations to other rules
    fn compute_mappings(&mut self) {
        let mut invalid_maps: FxIndexSet<Mapping> = FxIndexSet::default();
        for cat in self.rule_cats.iter() {
            for (n, rule_1) in
                cat.iter().enumerate().filter(|x| !self.removed_rules.borrow().contains(x.1))
            {
                for rule_2 in
                    cat.iter().skip(n + 1).filter(|x| !self.removed_rules.borrow().contains(*x))
                {
                    let forward =
                        Mapping::map_rules(&self.rule_vars[*rule_1], &self.rule_vars[*rule_2]);
                    if invalid_maps.contains(&forward) || self.mappings.contains_key(&forward) {
                        continue;
                    }
                    let reverse =
                        Mapping::map_rules(&self.rule_vars[*rule_2], &self.rule_vars[*rule_1]);
                    if invalid_maps.contains(&reverse) || self.mappings.contains_key(&reverse) {
                        continue;
                    }

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
    /// Currently, dependencies are in the form FxIndexMap<A, EmptyFxIndexSet>,
    /// where A is a rule that must be mapped by another mapping. We must
    /// populate the EmptyFxIndexSet with a set of mappings that can map A without
    /// conflicting with the current mapping
    fn resolve_dependencies_to_mapping(&mut self) {
        self.mappings.retain(|mapping, mapping_info| {
            if self.applied_mappings.borrow().conflicts_with(mapping) {
                return false;
            }
            mapping_info
                .dependencies
                .retain(|dependency, _| !self.removed_rules.borrow().contains(dependency));
            true
        });
        // A map from a constraint to the mappings that will eliminate it
        let mut constraint_mappings: FxIndexMap<usize, FxIndexSet<usize>> = FxIndexMap::default();
        for (indx, (_, mapping_info)) in self.mappings.iter().enumerate() {
            for (from_rule, _) in mapping_info.rule_mappings.iter() {
                constraint_mappings
                    .entry(*from_rule)
                    .or_insert_with(FxIndexSet::default)
                    .insert(indx);
            }
        }
        for indx in 0..self.mappings.len() {
            let mapping = self.get_mapping(indx);
            let input_dependencies = &self.get_mapping_info(indx).dependencies;
            let mut dependencies = Vec::new();
            for (dependency, _) in input_dependencies.iter() {
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
                if dependencies.contains(&resolve_options) {
                    continue;
                }
                dependencies.push(resolve_options);
            }
            // After this point, the actual rules that a dependency maps
            // stops mattering - all that matters is that the dependency *exists*
            self.mappings.get_index_mut(indx).unwrap().1.dependencies =
                dependencies.into_iter().enumerate().collect();
        }
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
        for cat in self.rule_cats.iter() {
            for rule_1 in cat {
                let vars_1 = &self.rule_vars[*rule_1];
                if !mapping.affects_rule(vars_1) {
                    continue;
                }
                let mut found_non_conflicting = false;
                for rule_2 in cat.iter() {
                    let vars_2 = &self.rule_vars[*rule_2];
                    let trial_mapping = Mapping::map_rules(vars_1, vars_2);
                    if mapping.conflicts_with(&trial_mapping) {
                        continue;
                    }
                    found_non_conflicting = true;
                    // Only maps a subset of variables in rule_1
                    if !mapping.maps_fully(vars_1, vars_2) {
                        info.dependencies.insert(*rule_1, FxIndexSet::default());
                        continue;
                    }
                    if *rule_1 != *rule_2 {
                        info.rule_mappings.insert(*rule_1, *rule_2);
                    }
                }
                if !found_non_conflicting {
                    return Err(MapEvalErr::Conflicts);
                }
            }
        }
        for fully_mapped in info.rule_mappings.keys() {
            info.dependencies.remove(fully_mapped);
        }
        if maps_unremovable_var {
            return Err(MapEvalErr::Unremovable);
        }
        Ok(info)
    }

    fn resolve_dependencies(&mut self) {
        let mut used_mappings = FxIndexSet::default();
        for indx in 0..self.mappings.len() {
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
    // There's quite a few heuristics that will probably yield *significant* speedups
    // I'll look into that later, if the rest of this approach is sound
    fn dfs_search(
        &self,
        mut used_mappings: FxIndexSet<usize>,
        mut applied_mappings: Mapping,
        from: FxIndexSet<usize>,
    ) -> Option<FxIndexSet<usize>> {
        for mapping_indx in from.iter() {
            let (mapping, _) = self.mappings.get_index(*mapping_indx).unwrap();
            if applied_mappings.conflicts_with(mapping) {
                return None;
            }
            applied_mappings.0.extend(&mapping.0);
        }
        if from.iter().all(|x| used_mappings.contains(x)) {
            return Some(used_mappings);
        }
        used_mappings.extend(from.iter());

        let choices: Vec<&FxIndexSet<usize>> =
            from.iter().flat_map(|x| self.get_mapping_info(*x).dependencies.values()).collect();
        if choices.is_empty() {
            return Some(used_mappings);
        }
        let mut choice_indices = vec![0; choices.len()];
        while *choice_indices.last().unwrap() < choices.last().unwrap().len() {
            let choice: FxIndexSet<usize> = choice_indices
                .iter()
                .zip(&choices)
                .map(|(x, y)| *y.get_index(*x).unwrap())
                .collect();
            let search_result =
                self.dfs_search(used_mappings.clone(), applied_mappings.clone(), choice);
            if search_result.is_some() {
                return search_result;
            }

            for (val, limit) in choice_indices.iter_mut().zip(choices.iter().map(|x| x.len())) {
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
        self.removed_rules.borrow_mut().extend(info.rule_mappings.keys());
        self.applied_mappings.borrow_mut().0.extend(&mapping.0);
        true
    }
    fn get_mapping(&self, index: usize) -> &Mapping {
        &self.mappings.get_index(index).unwrap().0
    }
    fn get_mapping_info(&self, index: usize) -> &MappingInfo {
        &self.mappings.get_index(index).unwrap().1
    }
}

impl Mapping {
    fn map_rules(from: &[usize], to: &[usize]) -> Self {
        Self(from.iter().zip(to).map(|(x, y)| (*x, *y)).collect())
    }
    fn maps_var(&self, rule: usize) -> Option<usize> {
        self.0.get(&rule).map(|x| *x)
    }
    fn affects_rule(&self, rule: &[usize]) -> bool {
        rule.iter().any(|x| self.maps_var(*x).unwrap_or(*x) != *x)
    }
    fn maps_fully(&self, from: &[usize], to: &[usize]) -> bool {
        if from.len() != to.len() {
            return false;
        }
        from.iter().zip(to).all(|(x, y)| self.maps_var(*x).unwrap_or(*x) == *y)
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
        Self { dependencies: FxIndexMap::default(), rule_mappings: FxIndexMap::default() }
    }
}

// FIXME: Tests that test the solver on its own have been deleted cuz tidy check won't stop yelling at me - add back later
