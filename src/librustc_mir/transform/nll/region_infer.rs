// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::free_regions::FreeRegions;
use rustc::infer::InferCtxt;
use rustc::infer::RegionVariableOrigin;
use rustc::infer::NLLRegionVariableOrigin;
use rustc::infer::region_constraints::VarOrigins;
use rustc::mir::{Location, Mir};
use rustc::ty::{self, RegionVid};
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::fx::FxHashSet;
use std::collections::BTreeSet;
use std::fmt;
use syntax_pos::Span;

pub struct RegionInferenceContext<'tcx> {
    /// Contains the definition for every region variable.  Region
    /// variables are identified by their index (`RegionVid`). The
    /// definition contains information about where the region came
    /// from as well as its final inferred value.
    definitions: IndexVec<RegionVid, RegionDefinition<'tcx>>,

    /// The constraints we have accumulated and used during solving.
    constraints: Vec<Constraint>,
}

struct RegionDefinition<'tcx> {
    /// Why we created this variable. Mostly these will be
    /// `RegionVariableOrigin::NLL`, but some variables get created
    /// elsewhere in the code with other causes (e.g., instantiation
    /// late-bound-regions).
    origin: RegionVariableOrigin,

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

/// The value of an individual region variable. Region variables
/// consist of a set of points in the CFG as well as a set of "free
/// regions", which are sometimes written as `end(R)`. These
/// correspond to the named lifetimes and refer to portions of the
/// caller's control-flow graph -- specifically some portion that can
/// be reached after we return.
#[derive(Clone, Default, PartialEq, Eq)]
struct Region {
    points: BTreeSet<Location>,
    free_regions: BTreeSet<RegionVid>,
}

impl fmt::Debug for Region {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        formatter
            .debug_set()
            .entries(&self.points)
            .entries(&self.free_regions)
            .finish()
    }
}

impl Region {
    fn add_point(&mut self, point: Location) -> bool {
        self.points.insert(point)
    }

    fn add_free_region(&mut self, region: RegionVid) -> bool {
        self.free_regions.insert(region)
    }

    fn contains_point(&self, point: Location) -> bool {
        self.points.contains(&point)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Constraint {
    // NB. The ordering here is not significant for correctness, but
    // it is for convenience. Before we dump the constraints in the
    // debugging logs, we sort them, and we'd like the "super region"
    // to be first, etc. (In particular, span should remain last.)

    /// The region SUP must outlive SUB...
    sup: RegionVid,

    /// Region that must be outlived.
    sub: RegionVid,

    /// At this location.
    point: Location,

    /// Where did this constraint arise?
    span: Span,
}

impl<'a, 'gcx, 'tcx> RegionInferenceContext<'tcx> {
    /// Creates a new region inference context with a total of
    /// `num_region_variables` valid inference variables; the first N
    /// of those will be constant regions representing the free
    /// regions defined in `free_regions`.
    pub fn new(var_origins: VarOrigins, free_regions: &FreeRegions<'tcx>, mir: &Mir<'tcx>) -> Self {
        // Create a RegionDefinition for each inference variable.
        let definitions = var_origins
            .into_iter()
            .map(|origin| RegionDefinition::new(origin))
            .collect();

        let mut result = Self {
            definitions: definitions,
            constraints: Vec::new(),
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
        let FreeRegions {
            indices,
            free_region_map,
        } = free_regions;

        // For each free region X:
        for (free_region, &variable) in indices {
            // These should be free-region variables.
            assert!(match self.definitions[variable].origin {
                RegionVariableOrigin::NLL(NLLRegionVariableOrigin::FreeRegion) => true,
                _ => false,
            });

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

            // `'static` outlives all other free regions as well.
            if let ty::ReStatic = free_region {
                for &other_variable in indices.values() {
                    self.definitions[variable]
                        .value
                        .add_free_region(other_variable);
                }
            }

            // Go through each region Y that outlives X (i.e., where
            // Y: X is true). Add `end(X)` into the set for `Y`.
            for superregion in free_region_map.regions_that_outlive(&free_region) {
                let superregion_index = indices[superregion];
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
    pub fn regions(&self) -> impl Iterator<Item = RegionVid> {
        self.definitions.indices()
    }

    /// Returns true if the region `r` contains the point `p`.
    ///
    /// Until `solve()` executes, this value is not particularly meaningful.
    pub fn region_contains_point(&self, r: RegionVid, p: Location) -> bool {
        self.definitions[r].value.contains_point(p)
    }

    /// Returns access to the value of `r` for debugging purposes.
    pub(super) fn region_value(&self, r: RegionVid) -> &fmt::Debug {
        &self.definitions[r].value
    }

    /// Indicates that the region variable `v` is live at the point `point`.
    pub(super) fn add_live_point(&mut self, v: RegionVid, point: Location) {
        debug!("add_live_point({:?}, {:?})", v, point);
        let definition = &mut self.definitions[v];
        if !definition.constant {
            definition.value.add_point(point);
        } else {
            // Constants are used for free regions, which already
            // contain all the points in the control-flow graph.
            assert!(definition.value.contains_point(point));
        }
    }

    /// Indicates that the region variable `sup` must outlive `sub` is live at the point `point`.
    pub(super) fn add_outlives(
        &mut self,
        span: Span,
        sup: RegionVid,
        sub: RegionVid,
        point: Location,
    ) {
        debug!("add_outlives({:?}: {:?} @ {:?}", sup, sub, point);
        self.constraints.push(Constraint {
            span,
            sup,
            sub,
            point,
        });
    }

    /// Perform region inference.
    pub(super) fn solve(&mut self, infcx: &InferCtxt<'a, 'gcx, 'tcx>, mir: &Mir<'tcx>) {
        let errors = self.propagate_constraints(mir);

        // worst error msg ever
        for (fr1, span, fr2) in errors {
            infcx.tcx.sess.span_err(
                span,
                &format!(
                    "free region `{}` does not outlive `{}`",
                    self.definitions[fr1].name.unwrap(),
                    self.definitions[fr2].name.unwrap()
                ),
            );
        }
    }

    /// Propagate the region constraints: this will grow the values
    /// for each region variable until all the constraints are
    /// satisfied. Note that some values may grow **too** large to be
    /// feasible, but we check this later.
    fn propagate_constraints(&mut self, mir: &Mir<'tcx>) -> Vec<(RegionVid, Span, RegionVid)> {
        let mut changed = true;
        let mut dfs = Dfs::new(mir);
        let mut error_regions = FxHashSet();
        let mut errors = vec![];

        debug!("propagate_constraints()");
        debug!("propagate_constraints: constraints={:#?}", {
            let mut constraints: Vec<_> = self.constraints.iter().collect();
            constraints.sort();
            constraints
        });

        while changed {
            changed = false;
            for constraint in &self.constraints {
                debug!("propagate_constraints: constraint={:?}", constraint);
                let sub = &self.definitions[constraint.sub].value.clone();
                let sup_def = &mut self.definitions[constraint.sup];

                debug!("propagate_constraints:    sub (before): {:?}", sub);
                debug!("propagate_constraints:    sup (before): {:?}", sup_def.value);

                if !sup_def.constant {
                    // If this is not a constant, then grow the value as needed to
                    // accommodate the outlives constraint.

                    if dfs.copy(sub, &mut sup_def.value, constraint.point) {
                        changed = true;
                    }

                    debug!("propagate_constraints:    sup (after) : {:?}", sup_def.value);
                    debug!("propagate_constraints:    changed     : {:?}", changed);
                } else {
                    // If this is a constant, check whether it *would
                    // have* to grow in order for the constraint to be
                    // satisfied. If so, create an error.

                    let mut sup_value = sup_def.value.clone();
                    if dfs.copy(sub, &mut sup_value, constraint.point) {
                        // Constant values start out with the entire
                        // CFG, so it must be some new free region
                        // that was added. Find one.
                        let &new_region = sup_value
                            .free_regions
                            .difference(&sup_def.value.free_regions)
                            .next()
                            .unwrap();
                        debug!("propagate_constraints:    new_region : {:?}", new_region);
                        if error_regions.insert(constraint.sup) {
                            errors.push((constraint.sup, constraint.span, new_region));
                        }
                    }
                }
            }
            debug!("\n");
        }
        errors
    }
}

struct Dfs<'a, 'tcx: 'a> {
    mir: &'a Mir<'tcx>,
}

impl<'a, 'tcx> Dfs<'a, 'tcx> {
    fn new(mir: &'a Mir<'tcx>) -> Self {
        Self { mir }
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

            if !from_region.contains_point(p) {
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

                debug!("        dfs: free_regions={:?}", from_region.free_regions);
                for &fr in &from_region.free_regions {
                    changed |= to_region.free_regions.insert(fr);
                }
            } else {
                stack.extend(successor_points);
            }
        }

        changed
    }
}

impl<'tcx> RegionDefinition<'tcx> {
    fn new(origin: RegionVariableOrigin) -> Self {
        // Create a new region definition. Note that, for free
        // regions, these fields get updated later in
        // `init_free_regions`.
        Self {
            origin,
            name: None,
            constant: false,
            value: Region::default(),
        }
    }
}

impl fmt::Debug for Constraint {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            formatter,
            "({:?}: {:?} @ {:?}) due to {:?}",
            self.sup,
            self.sub,
            self.point,
            self.span
        )
    }
}
