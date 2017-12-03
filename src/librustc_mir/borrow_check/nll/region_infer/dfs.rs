// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Module defining the `dfs` method on `RegionInferenceContext`, along with
//! its associated helper traits.

use rustc::mir::{Location, Mir};
use rustc::ty::RegionVid;
use rustc_data_structures::fx::FxHashSet;
use super::RegionInferenceContext;
use super::values::{RegionElementIndex, RegionValues, RegionValueElements};

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Function used to satisfy or test a `R1: R2 @ P`
    /// constraint. The core idea is that it performs a DFS starting
    /// from `P`. The precise actions *during* that DFS depend on the
    /// `op` supplied, so see (e.g.) `CopyFromSourceToTarget` for more
    /// details.
    ///
    /// Returns:
    ///
    /// - `Ok(true)` if the walk was completed and something changed
    ///   along the way;
    /// - `Ok(false)` if the walk was completed with no changes;
    /// - `Err(early)` if the walk was existed early by `op`. `earlyelem` is the
    ///   value that `op` returned.
    pub(super) fn dfs<C>(&self, mir: &Mir<'tcx>, mut op: C) -> Result<bool, C::Early>
    where
        C: DfsOp,
    {
        let mut changed = false;

        let mut stack = vec![];
        let mut visited = FxHashSet();

        stack.push(op.start_point());
        while let Some(p) = stack.pop() {
            let point_index = self.elements.index(p);

            if !op.source_region_contains(point_index) {
                debug!("            not in from-region");
                continue;
            }

            if !visited.insert(p) {
                debug!("            already visited");
                continue;
            }

            let new = op.add_to_target_region(point_index)?;
            changed |= new;

            let block_data = &mir[p.block];

            let start_stack_len = stack.len();

            if p.statement_index < block_data.statements.len() {
                stack.push(Location {
                    statement_index: p.statement_index + 1,
                    ..p
                });
            } else {
                stack.extend(block_data.terminator().successors().iter().map(
                    |&basic_block| {
                        Location {
                            statement_index: 0,
                            block: basic_block,
                        }
                    },
                ));
            }

            if stack.len() == start_stack_len {
                // If we reach the END point in the graph, then copy
                // over any skolemized end points in the `from_region`
                // and make sure they are included in the `to_region`.
                changed |= op.add_universal_regions_outlived_by_source_to_target()?;
            }
        }

        Ok(changed)
    }
}

/// Customizes the operation of the `dfs` function. This function is
/// used during inference to satisfy a `R1: R2 @ P` constraint.
pub(super) trait DfsOp {
    /// If this op stops the walk early, what type does it propagate?
    type Early;

    /// Returns the point from which to start the DFS.
    fn start_point(&self) -> Location;

    /// Returns true if the source region contains the given point.
    fn source_region_contains(&mut self, point_index: RegionElementIndex) -> bool;

    /// Adds the given point to the target region, returning true if
    /// something has changed. Returns `Err` if we should abort the
    /// walk early.
    fn add_to_target_region(
        &mut self,
        point_index: RegionElementIndex,
    ) -> Result<bool, Self::Early>;

    /// Adds all universal regions in the source region to the target region, returning
    /// true if something has changed.
    fn add_universal_regions_outlived_by_source_to_target(
        &mut self,
    ) -> Result<bool, Self::Early>;
}

/// Used during inference to enforce a `R1: R2 @ P` constraint.  For
/// each point Q we reach along the DFS, we check if Q is in R2 (the
/// "source region"). If not, we stop the walk. Otherwise, we add Q to
/// R1 (the "target region") and continue to Q's successors. If we
/// reach the end of the graph, then we add any universal regions from
/// R2 into R1.
pub(super) struct CopyFromSourceToTarget<'v> {
    pub source_region: RegionVid,
    pub target_region: RegionVid,
    pub inferred_values: &'v mut RegionValues,
    pub constraint_point: Location,
}

impl<'v> DfsOp for CopyFromSourceToTarget<'v> {
    /// We never stop the walk early.
    type Early = !;

    fn start_point(&self) -> Location {
        self.constraint_point
    }

    fn source_region_contains(&mut self, point_index: RegionElementIndex) -> bool {
        self.inferred_values
            .contains(self.source_region, point_index)
    }

    fn add_to_target_region(
        &mut self,
        point_index: RegionElementIndex,
    ) -> Result<bool, !> {
        Ok(self.inferred_values.add(self.target_region, point_index))
    }

    fn add_universal_regions_outlived_by_source_to_target(
        &mut self,
    ) -> Result<bool, !> {
        Ok(
            self.inferred_values
                .add_universal_regions_outlived_by(self.source_region, self.target_region),
        )
    }
}

/// Used after inference to *test* a `R1: R2 @ P` constraint.  For
/// each point Q we reach along the DFS, we check if Q in R2 is also
/// contained in R1. If not, we abort the walk early with an `Err`
/// condition. Similarly, if we reach the end of the graph and find
/// that R1 contains some universal region that R2 does not contain,
/// we abort the walk early.
#[allow(dead_code)] // TODO
pub(super) struct TestTarget<'v> {
    source_region: RegionVid,
    target_region: RegionVid,
    elements: &'v RegionValueElements,
    inferred_values: &'v RegionValues,
    constraint_point: Location,
}

#[allow(dead_code)] // TODO
impl<'v> DfsOp for TestTarget<'v> {
    /// The element that was not found within R2.
    type Early = RegionElementIndex;

    fn start_point(&self) -> Location {
        self.constraint_point
    }

    fn source_region_contains(&mut self, point_index: RegionElementIndex) -> bool {
        self.inferred_values
            .contains(self.source_region, point_index)
    }

    fn add_to_target_region(
        &mut self,
        point_index: RegionElementIndex,
    ) -> Result<bool, RegionElementIndex> {
        if !self.inferred_values
            .contains(self.target_region, point_index)
        {
            return Err(point_index);
        }

        Ok(false)
    }

    fn add_universal_regions_outlived_by_source_to_target(
        &mut self,
    ) -> Result<bool, RegionElementIndex> {
        for ur in self.inferred_values
            .universal_regions_outlived_by(self.source_region)
        {
            if !self.inferred_values.contains(self.target_region, ur) {
                return Err(self.elements.index(ur));
            }
        }

        Ok(false)
    }
}
