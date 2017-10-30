// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{Region, RegionIndex};
use super::free_regions::FreeRegions;
use rustc::infer::InferCtxt;
use rustc::mir::{Location, Mir};
use rustc::ty;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc_data_structures::fx::FxHashSet;

pub struct RegionInferenceContext<'tcx> {
    /// Contains the definition for every region variable.  Region
    /// variables are identified by their index (`RegionIndex`). The
    /// definition contains information about where the region came
    /// from as well as its final inferred value.
    definitions: IndexVec<RegionIndex, RegionDefinition<'tcx>>,

    /// The indices of all "free regions" in scope. These are the
    /// lifetime parameters (anonymous and named) declared in the
    /// function signature:
    ///
    ///     fn foo<'a, 'b>(x: &Foo<'a, 'b>)
    ///            ^^  ^^     ^
    ///
    /// These indices will be from 0..N, as it happens, but we collect
    /// them into a vector for convenience.
    free_regions: Vec<RegionIndex>,

    /// The constraints we have accumulated and used during solving.
    constraints: Vec<Constraint>,
}

#[derive(Default)]
struct RegionDefinition<'tcx> {
    /// If this is a free-region, then this is `Some(X)` where `X` is
    /// the name of the region.
    name: Option<ty::Region<'tcx>>,

    /// If true, this is a constant region which cannot grow larger.
    /// This is used for named regions as well as `'static`.
    constant: bool,

    /// The current value of this inference variable. This starts out
    /// empty, but grows as we add constraints. The final value is
    /// determined when `solve()` is executed.
    value: Region,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Constraint {
    sub: RegionIndex,
    sup: RegionIndex,
    point: Location,
}

impl<'a, 'gcx, 'tcx> RegionInferenceContext<'tcx> {
    /// Creates a new region inference context with a total of
    /// `num_region_variables` valid inference variables; the first N
    /// of those will be constant regions representing the free
    /// regions defined in `free_regions`.
    pub fn new(
        free_regions: &FreeRegions<'tcx>,
        num_region_variables: usize,
        mir: &Mir<'tcx>,
    ) -> Self {
        let mut result = Self {
            definitions: (0..num_region_variables)
                .map(|_| RegionDefinition::default())
                .collect(),
            constraints: Vec::new(),
            free_regions: Vec::new(),
        };

        result.init_free_regions(free_regions, mir);

        result
    }

    /// Initializes the region variables for each free region
    /// (lifetime parameter). The first N variables always correspond
    /// to the free regions appearing in the function signature (both
    /// named and anonymous) and where clauses. This function iterates
    /// over those regions and initializes them with minimum values.
    ///
    /// For example:
    ///
    ///     fn foo<'a, 'b>(..) where 'a: 'b
    ///
    /// would initialize two variables like so:
    ///
    ///     R0 = { CFG, R0 } // 'a
    ///     R1 = { CFG, R0, R1 } // 'b
    ///
    /// Here, R0 represents `'a`, and it contains (a) the entire CFG
    /// and (b) any free regions that it outlives, which in this case
    /// is just itself. R1 (`'b`) in contrast also outlives `'a` and
    /// hence contains R0 and R1.
    fn init_free_regions(&mut self, free_regions: &FreeRegions<'tcx>, mir: &Mir<'tcx>) {
        let &FreeRegions {
            ref indices,
            ref free_region_map,
        } = free_regions;

        // For each free region X:
        for (free_region, index) in indices {
            let variable = RegionIndex::new(*index);

            self.free_regions.push(variable);

            // Initialize the name and a few other details.
            self.definitions[variable].name = Some(free_region);
            self.definitions[variable].constant = true;

            // Add all nodes in the CFG to `definition.value`.
            for (block, block_data) in mir.basic_blocks().iter_enumerated() {
                let definition = &mut self.definitions[variable];
                for statement_index in 0..block_data.statements.len() + 1 {
                    let location = Location {
                        block,
                        statement_index,
                    };
                    definition.value.add_point(location);
                }
            }

            // Add `end(X)` into the set for X.
            self.definitions[variable].value.add_free_region(variable);

            // Go through each region Y that outlives X (i.e., where
            // Y: X is true). Add `end(X)` into the set for `Y`.
            for superregion in free_region_map.regions_that_outlive(&free_region) {
                let superregion_index = RegionIndex::new(indices[superregion]);
                self.definitions[superregion_index]
                    .value
                    .add_free_region(variable);
            }

            debug!(
                "init_free_regions: region variable for `{:?}` is `{:?}` with value `{:?}`",
                free_region,
                variable,
                self.definitions[variable].value
            );
        }
    }

    /// Returns an iterator over all the region indices.
    pub fn regions(&self) -> impl Iterator<Item = RegionIndex> {
        self.definitions.indices()
    }

    /// Returns the inferred value for the region `r`.
    ///
    /// Until `solve()` executes, this value is not particularly meaningful.
    pub fn region_value(&self, r: RegionIndex) -> &Region {
        &self.definitions[r].value
    }

    /// Indicates that the region variable `v` is live at the point `point`.
    pub(super) fn add_live_point(&mut self, v: RegionIndex, point: Location) {
        debug!("add_live_point({:?}, {:?})", v, point);
        let definition = &mut self.definitions[v];
        definition.value.add_point(point);
    }

    /// Indicates that the region variable `sup` must outlive `sub` is live at the point `point`.
    pub(super) fn add_outlives(&mut self, sup: RegionIndex, sub: RegionIndex, point: Location) {
        debug!("add_outlives({:?}: {:?} @ {:?}", sup, sub, point);
        self.constraints.push(Constraint { sup, sub, point });
    }

    /// Perform region inference.
    pub(super) fn solve(&mut self, infcx: &InferCtxt<'a, 'gcx, 'tcx>, mir: &Mir<'tcx>) {
        self.propagate_constraints(infcx, mir);
    }

    /// Propagate the region constraints: this will grow the values
    /// for each region variable until all the constraints are
    /// satisfied. Note that some values may grow **too** large to be
    /// feasible, but we check this later.
    fn propagate_constraints(&mut self, infcx: &InferCtxt<'a, 'gcx, 'tcx>, mir: &Mir<'tcx>) {
        let mut changed = true;
        let mut dfs = Dfs::new(infcx, mir);
        while changed {
            changed = false;
            for constraint in &self.constraints {
                let sub = &self.definitions[constraint.sub].value.clone();
                let sup_def = &mut self.definitions[constraint.sup];
                debug!("constraint: {:?}", constraint);
                debug!("    sub (before): {:?}", sub);
                debug!("    sup (before): {:?}", sup_def.value);

                if dfs.copy(sub, &mut sup_def.value, constraint.point) {
                    changed = true;
                }

                debug!("    sup (after) : {:?}", sup_def.value);
                debug!("    changed     : {:?}", changed);
            }
            debug!("\n");
        }
    }
}

struct Dfs<'a, 'gcx: 'tcx + 'a, 'tcx: 'a> {
    #[allow(dead_code)] infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    mir: &'a Mir<'tcx>,
}

impl<'a, 'gcx: 'tcx, 'tcx: 'a> Dfs<'a, 'gcx, 'tcx> {
    fn new(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>, mir: &'a Mir<'tcx>) -> Self {
        Self { infcx, mir }
    }

    fn copy(
        &mut self,
        from_region: &Region,
        to_region: &mut Region,
        start_point: Location,
    ) -> bool {
        let mut changed = false;

        let mut stack = vec![];
        let mut visited = FxHashSet();

        stack.push(start_point);
        while let Some(p) = stack.pop() {
            debug!("        dfs: p={:?}", p);

            if !from_region.may_contain_point(p) {
                debug!("            not in from-region");
                continue;
            }

            if !visited.insert(p) {
                debug!("            already visited");
                continue;
            }

            changed |= to_region.add_point(p);

            let block_data = &self.mir[p.block];
            let successor_points = if p.statement_index < block_data.statements.len() {
                vec![
                    Location {
                        statement_index: p.statement_index + 1,
                        ..p
                    },
                ]
            } else {
                block_data
                    .terminator()
                    .successors()
                    .iter()
                    .map(|&basic_block| {
                        Location {
                            statement_index: 0,
                            block: basic_block,
                        }
                    })
                    .collect::<Vec<_>>()
            };

            if successor_points.is_empty() {
                // If we reach the END point in the graph, then copy
                // over any skolemized end points in the `from_region`
                // and make sure they are included in the `to_region`.

                to_region.free_regions.extend(&from_region.free_regions);
            } else {
                stack.extend(successor_points);
            }
        }

        changed
    }
}
