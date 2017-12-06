// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::universal_regions::UniversalRegions;
use rustc::infer::InferCtxt;
use rustc::infer::RegionVariableOrigin;
use rustc::infer::NLLRegionVariableOrigin;
use rustc::infer::region_constraints::VarOrigins;
use rustc::infer::outlives::free_region_map::FreeRegionMap;
use rustc::mir::{Location, Mir};
use rustc::ty::{self, RegionVid};
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::bitvec::BitMatrix;
use rustc_data_structures::indexed_vec::Idx;
use std::collections::BTreeMap;
use std::fmt;
use syntax_pos::Span;

pub struct RegionInferenceContext<'tcx> {
    /// Contains the definition for every region variable.  Region
    /// variables are identified by their index (`RegionVid`). The
    /// definition contains information about where the region came
    /// from as well as its final inferred value.
    definitions: IndexVec<RegionVid, RegionDefinition<'tcx>>,

    /// The liveness constraints added to each region. For most
    /// regions, these start out empty and steadily grow, though for
    /// each universally quantified region R they start out containing
    /// the entire CFG and `end(R)`.
    ///
    /// In this `BitMatrix` representation, the rows are the region
    /// variables and the columns are the free regions and MIR locations.
    liveness_constraints: BitMatrix,

    /// The final inferred values of the inference variables; `None`
    /// until `solve` is invoked.
    inferred_values: Option<BitMatrix>,

    /// The constraints we have accumulated and used during solving.
    constraints: Vec<Constraint>,

    /// A map from each MIR Location to its column index in
    /// `liveness_constraints`/`inferred_values`. (The first N columns are
    /// the free regions.)
    point_indices: BTreeMap<Location, usize>,

    /// Number of universally quantified regions. This is used to
    /// determine the meaning of the bits in `inferred_values` and
    /// friends.
    num_universal_regions: usize,

    free_region_map: &'tcx FreeRegionMap<'tcx>,
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

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Creates a new region inference context with a total of
    /// `num_region_variables` valid inference variables; the first N
    /// of those will be constant regions representing the free
    /// regions defined in `universal_regions`.
    pub fn new(
        var_origins: VarOrigins,
        universal_regions: &UniversalRegions<'tcx>,
        mir: &Mir<'tcx>,
    ) -> Self {
        let num_region_variables = var_origins.len();
        let num_universal_regions = universal_regions.indices.len();

        let mut num_points = 0;
        let mut point_indices = BTreeMap::new();

        for (block, block_data) in mir.basic_blocks().iter_enumerated() {
            for statement_index in 0..block_data.statements.len() + 1 {
                let location = Location {
                    block,
                    statement_index,
                };
                point_indices.insert(location, num_universal_regions + num_points);
                num_points += 1;
            }
        }

        // Create a RegionDefinition for each inference variable.
        let definitions = var_origins
            .into_iter()
            .map(|origin| RegionDefinition::new(origin))
            .collect();

        let mut result = Self {
            definitions,
            liveness_constraints: BitMatrix::new(
                num_region_variables,
                num_universal_regions + num_points,
            ),
            inferred_values: None,
            constraints: Vec::new(),
            point_indices,
            num_universal_regions,
            free_region_map: universal_regions.free_region_map,
        };

        result.init_universal_regions(universal_regions);

        result
    }

    /// Initializes the region variables for each universally
    /// quantified region (lifetime parameter). The first N variables
    /// always correspond to the regions appearing in the function
    /// signature (both named and anonymous) and where clauses. This
    /// function iterates over those regions and initializes them with
    /// minimum values.
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
    fn init_universal_regions(&mut self, universal_regions: &UniversalRegions<'tcx>) {
        let UniversalRegions {
            indices,
            free_region_map: _,
        } = universal_regions;

        // For each universally quantified region X:
        for (free_region, &variable) in indices {
            // These should be free-region variables.
            assert!(match self.definitions[variable].origin {
                RegionVariableOrigin::NLL(NLLRegionVariableOrigin::FreeRegion) => true,
                _ => false,
            });

            // Initialize the name and a few other details.
            self.definitions[variable].name = Some(free_region);

            // Add all nodes in the CFG to liveness constraints
            for (_location, point_index) in &self.point_indices {
                self.liveness_constraints
                    .add(variable.index(), *point_index);
            }

            // Add `end(X)` into the set for X.
            self.liveness_constraints
                .add(variable.index(), variable.index());
        }
    }

    /// Returns an iterator over all the region indices.
    pub fn regions(&self) -> impl Iterator<Item = RegionVid> {
        self.definitions.indices()
    }

    /// Returns true if the region `r` contains the point `p`.
    ///
    /// Panics if called before `solve()` executes,
    pub fn region_contains_point(&self, r: RegionVid, p: Location) -> bool {
        let inferred_values = self.inferred_values
            .as_ref()
            .expect("region values not yet inferred");
        self.region_contains_point_in_matrix(inferred_values, r, p)
    }

    /// True if given region `r` contains the point `p`, when
    /// evaluated in the set of region values `matrix`.
    fn region_contains_point_in_matrix(
        &self,
        matrix: &BitMatrix,
        r: RegionVid,
        p: Location,
    ) -> bool {
        let point_index = self.point_indices
            .get(&p)
            .expect("point index should be known");
        matrix.contains(r.index(), *point_index)
    }

    /// True if given region `r` contains the `end(s)`, when
    /// evaluated in the set of region values `matrix`.
    fn region_contains_region_in_matrix(
        &self,
        matrix: &BitMatrix,
        r: RegionVid,
        s: RegionVid,
    ) -> bool {
        matrix.contains(r.index(), s.index())
    }

    /// Returns access to the value of `r` for debugging purposes.
    pub(super) fn region_value_str(&self, r: RegionVid) -> String {
        let inferred_values = self.inferred_values
            .as_ref()
            .expect("region values not yet inferred");

        let mut result = String::new();
        result.push_str("{");
        let mut sep = "";

        for &point in self.point_indices.keys() {
            if self.region_contains_point_in_matrix(inferred_values, r, point) {
                result.push_str(&format!("{}{:?}", sep, point));
                sep = ", ";
            }
        }

        for fr in (0..self.num_universal_regions).map(RegionVid::new) {
            if self.region_contains_region_in_matrix(inferred_values, r, fr) {
                result.push_str(&format!("{}{:?}", sep, fr));
                sep = ", ";
            }
        }

        result.push_str("}");

        result
    }

    /// Indicates that the region variable `v` is live at the point `point`.
    pub(super) fn add_live_point(&mut self, v: RegionVid, point: Location) -> bool {
        debug!("add_live_point({:?}, {:?})", v, point);
        assert!(self.inferred_values.is_none(), "values already inferred");
        let point_index = self.point_indices
            .get(&point)
            .expect("point index should be known");
        self.liveness_constraints.add(v.index(), *point_index)
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
        assert!(self.inferred_values.is_none(), "values already inferred");
        self.constraints.push(Constraint {
            span,
            sup,
            sub,
            point,
        });
    }

    /// Perform region inference.
    pub(super) fn solve(&mut self, infcx: &InferCtxt<'_, '_, 'tcx>, mir: &Mir<'tcx>) {
        assert!(self.inferred_values.is_none(), "values already inferred");

        // Find the minimal regions that can solve the constraints. This is infallible.
        self.propagate_constraints(mir);

        // Now, see whether any of the constraints were too strong. In
        // particular, we want to check for a case where a universally
        // quantified region exceeded its bounds.  Consider:
        //
        //     fn foo<'a, 'b>(x: &'a u32) -> &'b u32 { x }
        //
        // In this case, returning `x` requires `&'a u32 <: &'b u32`
        // and hence we establish (transitively) a constraint that
        // `'a: 'b`. The `propagate_constraints` code above will
        // therefore add `end('a)` into the region for `'b` -- but we
        // have no evidence that `'a` outlives `'b`, so we want to report
        // an error.

        // The universal regions are always found in a prefix of the
        // full list.
        let free_region_definitions = self.definitions
            .iter_enumerated()
            .take_while(|(_, fr_definition)| fr_definition.name.is_some());

        for (fr, fr_definition) in free_region_definitions {
            self.check_free_region(infcx, fr, fr_definition);
        }
    }

    fn check_free_region(
        &self,
        infcx: &InferCtxt<'_, '_, 'tcx>,
        longer_fr: RegionVid,
        longer_definition: &RegionDefinition<'tcx>,
    ) {
        let inferred_values = self.inferred_values.as_ref().unwrap();
        let longer_name = longer_definition.name.unwrap();
        let longer_value = inferred_values.iter(longer_fr.index());

        // Find every region `shorter` such that `longer: shorter`
        // (because `longer` includes `end(shorter)`).
        for shorter_fr in longer_value.take_while(|&i| i < self.num_universal_regions) {
            let shorter_fr = RegionVid::new(shorter_fr);

            // `fr` includes `end(fr)`, that's not especially
            // interesting.
            if longer_fr == shorter_fr {
                continue;
            }

            let shorter_definition = &self.definitions[shorter_fr];
            let shorter_name = shorter_definition.name.unwrap();

            // Check that `o <= fr`. If not, report an error.
            if !self.free_region_map
                .sub_free_regions(shorter_name, longer_name)
            {
                // FIXME: worst error msg ever
                let blame_span = self.blame_span(longer_fr, shorter_fr);
                infcx.tcx.sess.span_err(
                    blame_span,
                    &format!(
                        "free region `{}` does not outlive `{}`",
                        longer_name,
                        shorter_name
                    ),
                );
            }
        }
    }

    /// Propagate the region constraints: this will grow the values
    /// for each region variable until all the constraints are
    /// satisfied. Note that some values may grow **too** large to be
    /// feasible, but we check this later.
    fn propagate_constraints(&mut self, mir: &Mir<'tcx>) {
        let mut changed = true;

        debug!("propagate_constraints()");
        debug!("propagate_constraints: constraints={:#?}", {
            let mut constraints: Vec<_> = self.constraints.iter().collect();
            constraints.sort();
            constraints
        });

        // The initial values for each region are derived from the liveness
        // constraints we have accumulated.
        let mut inferred_values = self.liveness_constraints.clone();

        while changed {
            changed = false;
            debug!("propagate_constraints: --------------------");
            for constraint in &self.constraints {
                debug!("propagate_constraints: constraint={:?}", constraint);

                // Grow the value as needed to accommodate the
                // outlives constraint.

                if self.copy(
                    &mut inferred_values,
                    mir,
                    constraint.sub,
                    constraint.sup,
                    constraint.point,
                ) {
                    debug!("propagate_constraints:   sub={:?}", constraint.sub);
                    debug!("propagate_constraints:   sup={:?}", constraint.sup);
                    changed = true;
                }
            }
            debug!("\n");
        }

        self.inferred_values = Some(inferred_values);
    }

    fn copy(
        &self,
        inferred_values: &mut BitMatrix,
        mir: &Mir<'tcx>,
        from_region: RegionVid,
        to_region: RegionVid,
        start_point: Location,
    ) -> bool {
        let mut changed = false;

        let mut stack = vec![];
        let mut visited = FxHashSet();

        stack.push(start_point);
        while let Some(p) = stack.pop() {
            debug!("        copy: p={:?}", p);

            if !self.region_contains_point_in_matrix(inferred_values, from_region, p) {
                debug!("            not in from-region");
                continue;
            }

            if !visited.insert(p) {
                debug!("            already visited");
                continue;
            }

            let point_index = self.point_indices.get(&p).unwrap();
            changed |= inferred_values.add(to_region.index(), *point_index);

            let block_data = &mir[p.block];
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
                let universal_region_indices = inferred_values
                    .iter(from_region.index())
                    .take_while(|&i| i < self.num_universal_regions)
                    .collect::<Vec<_>>();
                for fr in &universal_region_indices {
                    changed |= inferred_values.add(to_region.index(), *fr);
                }
            } else {
                stack.extend(successor_points);
            }
        }

        changed
    }

    /// Tries to finds a good span to blame for the fact that `fr1`
    /// contains `fr2`.
    fn blame_span(&self, fr1: RegionVid, fr2: RegionVid) -> Span {
        // Find everything that influenced final value of `fr`.
        let influenced_fr1 = self.dependencies(fr1);

        // Try to find some outlives constraint `'X: fr2` where `'X`
        // influenced `fr1`. Blame that.
        //
        // NB, this is a pretty bad choice most of the time. In
        // particular, the connection between `'X` and `fr1` may not
        // be obvious to the user -- not to mention the naive notion
        // of dependencies, which doesn't account for the locations of
        // contraints at all. But it will do for now.
        for constraint in &self.constraints {
            if constraint.sub == fr2 && influenced_fr1[constraint.sup] {
                return constraint.span;
            }
        }

        bug!(
            "could not find any constraint to blame for {:?}: {:?}",
            fr1,
            fr2
        );
    }

    /// Finds all regions whose values `'a` may depend on in some way.
    /// Basically if there exists a constraint `'a: 'b @ P`, then `'b`
    /// and `dependencies('b)` will be in the final set.
    ///
    /// Used during error reporting, extremely naive and inefficient.
    fn dependencies(&self, r0: RegionVid) -> IndexVec<RegionVid, bool> {
        let mut result_set = IndexVec::from_elem(false, &self.definitions);
        let mut changed = true;
        result_set[r0] = true;

        while changed {
            changed = false;
            for constraint in &self.constraints {
                if result_set[constraint.sup] {
                    if !result_set[constraint.sub] {
                        result_set[constraint.sub] = true;
                        changed = true;
                    }
                }
            }
        }

        result_set
    }
}

impl<'tcx> RegionDefinition<'tcx> {
    fn new(origin: RegionVariableOrigin) -> Self {
        // Create a new region definition. Note that, for free
        // regions, these fields get updated later in
        // `init_universal_regions`.
        Self { origin, name: None }
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
