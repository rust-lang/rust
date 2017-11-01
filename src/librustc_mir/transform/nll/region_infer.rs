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
use std::mem;
use rustc::infer::InferCtxt;
use rustc::mir::{Location, Mir};
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::fx::FxHashSet;

pub struct RegionInferenceContext {
    /// Contains the definition for every region variable.  Region
    /// variables are identified by their index (`RegionIndex`). The
    /// definition contains information about where the region came
    /// from as well as its final inferred value.
    definitions: IndexVec<RegionIndex, RegionDefinition>,

    /// The constraints we have accumulated and used during solving.
    constraints: Vec<Constraint>,

    /// List of errors we have accumulated as we add constraints.
    /// After solving is done, this is replaced with an empty vector.
    errors: Vec<InferenceError>,
}

pub struct InferenceError {
    pub constraint_point: Location,
    pub name: (), // FIXME(nashenas88) RegionName
}

#[derive(Default)]
struct RegionDefinition {
    name: (), // FIXME(nashenas88) RegionName
    value: Region,
    capped: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Constraint {
    sub: RegionIndex,
    sup: RegionIndex,
    point: Location,
}

impl RegionInferenceContext {
    pub fn new(num_region_variables: usize) -> Self {
        Self {
            definitions: (0..num_region_variables)
                .map(|_| RegionDefinition::default())
                .collect(),
            constraints: Vec::new(),
            errors: Vec::new(),
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

    /// Flags a region as being "capped" -- this means that if its
    /// value is required to grow as a result of some constraint
    /// (e.g., `add_live_point` or `add_outlives`), that indicates an
    /// error. This is used for the regions representing named
    /// lifetime parameters on a function: they get initialized to
    /// their complete value, and then "capped" so that they can no
    /// longer grow.
    #[allow(dead_code)]
    pub(super) fn cap_var(&mut self, v: RegionIndex) {
        self.definitions[v].capped = true;
    }

    pub(super) fn add_live_point(&mut self, v: RegionIndex, point: Location) {
        debug!("add_live_point({:?}, {:?})", v, point);
        let definition = &mut self.definitions[v];
        if definition.value.add_point(point) {
            if definition.capped {
                self.errors.push(InferenceError {
                    constraint_point: point,
                    name: definition.name,
                });
            }
        }
    }

    pub(super) fn add_outlives(&mut self, sup: RegionIndex, sub: RegionIndex, point: Location) {
        debug!("add_outlives({:?}: {:?} @ {:?}", sup, sub, point);
        self.constraints.push(Constraint { sup, sub, point });
    }

    /// Perform region inference.
    pub(super) fn solve<'a, 'gcx, 'tcx>(
        &mut self,
        infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
        mir: &'a Mir<'tcx>,
    ) -> Vec<InferenceError>
    where
        'gcx: 'tcx + 'a,
        'tcx: 'a,
    {
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
                    if sup_def.capped {
                        // This is kind of a hack, but when we add a
                        // constraint, the "point" is always the point
                        // AFTER the action that induced the
                        // constraint. So report the error on the
                        // action BEFORE that.
                        assert!(constraint.point.statement_index > 0);
                        let p = Location {
                            block: constraint.point.block,
                            statement_index: constraint.point.statement_index - 1,
                        };

                        self.errors.push(InferenceError {
                            constraint_point: p,
                            name: sup_def.name,
                        });
                    }
                }

                debug!("    sup (after) : {:?}", sup_def.value);
                debug!("    changed     : {:?}", changed);
            }
            debug!("\n");
        }

        mem::replace(&mut self.errors, Vec::new())
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

            if !from_region.may_contain(p) {
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
                // FIXME handle free regions
                // If we reach the END point in the graph, then copy
                // over any skolemized end points in the `from_region`
                // and make sure they are included in the `to_region`.
                // for region_decl in self.infcx.tcx.tables.borrow().free_region_map() {
                //     // FIXME(nashenas88) figure out skolemized_end points
                //     let block = self.env.graph.skolemized_end(region_decl.name);
                //     let skolemized_end_point = Location {
                //         block,
                //         statement_index: 0,
                //     };
                //     changed |= to_region.add_point(skolemized_end_point);
                // }
            } else {
                stack.extend(successor_points);
            }
        }

        changed
    }
}
