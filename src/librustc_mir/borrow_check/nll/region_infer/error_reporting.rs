// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use borrow_check::nll::region_infer::{Cause, ConstraintIndex, RegionInferenceContext};
use borrow_check::nll::region_infer::values::ToElementIndex;
use borrow_check::nll::type_check::Locations;
use rustc::hir::def_id::DefId;
use rustc::infer::InferCtxt;
use rustc::infer::error_reporting::nice_region_error::NiceRegionError;
use rustc::mir::{self, Location, Mir, Place, StatementKind, TerminatorKind, Rvalue};
use rustc::ty::RegionVid;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::indexed_vec::IndexVec;
use syntax_pos::Span;

/// Constraints that are considered interesting can be categorized to
/// determine why they are interesting. Order of variants indicates
/// sort order of the category, thereby influencing diagnostic output.
#[derive(Debug, Eq, PartialEq, PartialOrd, Ord)]
enum ConstraintCategory {
    Cast,
    Assignment,
    Return,
    CallArgument,
    Other,
    Boring,
}

impl fmt::Display for ConstraintCategory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ConstraintCategory::Assignment => write!(f, "assignment"),
            ConstraintCategory::Return => write!(f, "return"),
            ConstraintCategory::Cast => write!(f, "cast"),
            ConstraintCategory::CallArgument => write!(f, "argument"),
            _ => write!(f, "free region"),
        }
    }
}

impl<'tcx> RegionInferenceContext<'tcx> {
    /// When reporting an error, it is useful to be able to determine which constraints influenced
    /// the region being reported as an error. This function finds all of the paths from the
    /// constraint.
    fn find_constraint_paths_from_region(
        &self,
        r0: RegionVid
    ) -> Vec<Vec<ConstraintIndex>> {
        let constraints = self.constraints.clone();

        // Mapping of regions to the previous region and constraint index that led to it.
        let mut previous = FxHashMap();
        // Regions yet to be visited.
        let mut next = vec! [ r0 ];
        // Regions that have been visited.
        let mut visited = FxHashSet();
        // Ends of paths.
        let mut end_regions: Vec<RegionVid> = Vec::new();

        // When we've still got points to visit...
        while let Some(current) = next.pop() {
            // ...take the next point...
            debug!("find_constraint_paths_from_region: current={:?} next={:?}", current, next);

            // ...find the edges containing it...
            let mut upcoming = Vec::new();
            for (index, constraint) in constraints.iter_enumerated() {
                if constraint.sub == current {
                    // ...add the regions that join us with to the path we've taken...
                    debug!("find_constraint_paths_from_region: index={:?} constraint={:?}",
                           index, constraint);
                    let next_region = constraint.sup.clone();

                    // ...unless we've visited it since this was added...
                    if visited.contains(&next_region) {
                        debug!("find_constraint_paths_from_region: skipping as visited");
                        continue;
                    }

                    previous.insert(next_region, (index, Some(current)));
                    upcoming.push(next_region);
                }
            }

            if upcoming.is_empty() {
                // If we didn't find any edges then this is the end of a path...
                debug!("find_constraint_paths_from_region: new end region current={:?}", current);
                end_regions.push(current);
            } else {
                // ...but, if we did find edges, then add these to the regions yet to visit...
                debug!("find_constraint_paths_from_region: extend next upcoming={:?}", upcoming);
                next.extend(upcoming);
            }

            // ...and don't visit it again.
            visited.insert(current.clone());
            debug!("find_constraint_paths_from_region: next={:?} visited={:?}", next, visited);
        }

        // Now we've visited each point, compute the final paths.
        let mut paths: Vec<Vec<ConstraintIndex>> = Vec::new();
        debug!("find_constraint_paths_from_region: end_regions={:?}", end_regions);
        for end_region in end_regions {
            debug!("find_constraint_paths_from_region: end_region={:?}", end_region);

            // Get the constraint and region that led to this end point.
            // We can unwrap as we know if end_point was in the vector that it
            // must also be in our previous map.
            let (mut index, mut region) = previous.get(&end_region).unwrap();
            debug!("find_constraint_paths_from_region: index={:?} region={:?}", index, region);

            // Keep track of the indices.
            let mut path: Vec<ConstraintIndex> = vec![index];

            while region.is_some() && region != Some(r0) {
                let p = previous.get(&region.unwrap()).unwrap();
                index = p.0;
                region = p.1;

                debug!("find_constraint_paths_from_region: index={:?} region={:?}", index, region);
                path.push(index);
            }

            // Add to our paths.
            paths.push(path);
        }

        debug!("find_constraint_paths_from_region: paths={:?}", paths);
        paths
    }

    /// This function will return true if a constraint is interesting and false if a constraint
    /// is not. It is useful in filtering constraint paths to only interesting points.
    fn constraint_is_interesting(&self, index: &ConstraintIndex) -> bool {
        self.constraints.get(*index).filter(|constraint| {
            debug!("constraint_is_interesting: locations={:?} constraint={:?}",
                   constraint.locations, constraint);
            if let Locations::Interesting(_) = constraint.locations { true } else { false }
        }).is_some()
    }

    /// This function classifies a constraint from a location.
    fn classify_constraint(&self, index: &ConstraintIndex,
                           mir: &Mir<'tcx>) -> Option<(ConstraintCategory, Span)> {
        let constraint = self.constraints.get(*index)?;
        let span = constraint.locations.span(mir);
        let location = constraint.locations.from_location()?;

        if !self.constraint_is_interesting(index) {
            return Some((ConstraintCategory::Boring, span));
        }

        let data = &mir[location.block];
        let category = if location.statement_index == data.statements.len() {
            if let Some(ref terminator) = data.terminator {
                match terminator.kind {
                    TerminatorKind::DropAndReplace { .. } => ConstraintCategory::Assignment,
                    TerminatorKind::Call { .. } => ConstraintCategory::CallArgument,
                    _ => ConstraintCategory::Other,
                }
            } else {
                ConstraintCategory::Other
            }
        } else {
            let statement = &data.statements[location.statement_index];
            match statement.kind {
                StatementKind::Assign(ref place, ref rvalue) => {
                    if *place == Place::Local(mir::RETURN_PLACE) {
                        ConstraintCategory::Return
                    } else {
                        match rvalue {
                            Rvalue::Cast(..) => ConstraintCategory::Cast,
                            Rvalue::Use(..) => ConstraintCategory::Assignment,
                            _ => ConstraintCategory::Other,
                        }
                    }
                },
                _ => ConstraintCategory::Other,
            }
        };

        Some((category, span))
    }

    /// Report an error because the universal region `fr` was required to outlive
    /// `outlived_fr` but it is not known to do so. For example:
    ///
    /// ```
    /// fn foo<'a, 'b>(x: &'a u32) -> &'b u32 { x }
    /// ```
    ///
    /// Here we would be invoked with `fr = 'a` and `outlived_fr = `'b`.
    pub(super) fn report_error(
        &self,
        mir: &Mir<'tcx>,
        infcx: &InferCtxt<'_, '_, 'tcx>,
        mir_def_id: DefId,
        fr: RegionVid,
        outlived_fr: RegionVid,
        blame_span: Span,
    ) {
        // Obviously uncool error reporting.

        let fr_name = self.to_error_region(fr);
        let outlived_fr_name = self.to_error_region(outlived_fr);

        if let (Some(f), Some(o)) = (fr_name, outlived_fr_name) {
            let tables = infcx.tcx.typeck_tables_of(mir_def_id);
            let nice = NiceRegionError::new_from_span(infcx.tcx, blame_span, o, f, Some(tables));
            if let Some(_error_reported) = nice.try_report() {
                return;
            }
        }

        let fr_string = match fr_name {
            Some(r) => format!("free region `{}`", r),
            None => format!("free region `{:?}`", fr),
        };

        let outlived_fr_string = match outlived_fr_name {
            Some(r) => format!("free region `{}`", r),
            None => format!("free region `{:?}`", outlived_fr),
        };

        let constraints = self.find_constraint_paths_from_region(fr.clone());
        let path = constraints.iter().min_by_key(|p| p.len()).unwrap();
        debug!("report_error: shortest_path={:?}", path);

        let mut categorized_path = path.iter().filter_map(|index| {
            self.classify_constraint(index, mir)
        }).collect::<Vec<(ConstraintCategory, Span)>>();
        debug!("report_error: categorized_path={:?}", categorized_path);

        categorized_path.sort_by(|p0, p1| p0.0.cmp(&p1.0));
        debug!("report_error: sorted_path={:?}", categorized_path);

        if let Some((category, span)) = &categorized_path.first() {
            let mut diag = infcx.tcx.sess.struct_span_err(
                *span, &format!("{} requires that data must outlive {}",
                                category, outlived_fr_string),
            );

            diag.emit();
        } else {
            let mut diag = infcx.tcx.sess.struct_span_err(
                blame_span,
                &format!("{} does not outlive {}", fr_string, outlived_fr_string,),
            );

            diag.emit();
        }
    }

    crate fn why_region_contains_point(&self, fr1: RegionVid, elem: Location) -> Option<Cause> {
        // Find some constraint `X: Y` where:
        // - `fr1: X` transitively
        // - and `Y` is live at `elem`
        let index = self.blame_constraint(fr1, elem);
        let region_sub = self.constraints[index].sub;

        // then return why `Y` was live at `elem`
        self.liveness_constraints.cause(region_sub, elem)
    }

    /// Tries to finds a good span to blame for the fact that `fr1`
    /// contains `fr2`.
    pub(super) fn blame_constraint(&self, fr1: RegionVid,
                                   elem: impl ToElementIndex) -> ConstraintIndex {
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
        let relevant_constraint = self.constraints
            .iter_enumerated()
            .filter_map(|(i, constraint)| {
                if !self.liveness_constraints.contains(constraint.sub, elem) {
                    None
                } else {
                    influenced_fr1[constraint.sup]
                        .map(|distance| (distance, i))
                }
            })
            .min() // constraining fr1 with fewer hops *ought* to be more obvious
            .map(|(_dist, i)| i);

        relevant_constraint.unwrap_or_else(|| {
            bug!(
                "could not find any constraint to blame for {:?}: {:?}",
                fr1,
                elem,
            );
        })
    }

    /// Finds all regions whose values `'a` may depend on in some way.
    /// For each region, returns either `None` (does not influence
    /// `'a`) or `Some(d)` which indicates that it influences `'a`
    /// with distinct `d` (minimum number of edges that must be
    /// traversed).
    ///
    /// Used during error reporting, extremely naive and inefficient.
    fn dependencies(&self, r0: RegionVid) -> IndexVec<RegionVid, Option<usize>> {
        let mut result_set = IndexVec::from_elem(None, &self.definitions);
        let mut changed = true;
        result_set[r0] = Some(0); // distance 0 from `r0`

        while changed {
            changed = false;
            for constraint in &*self.constraints {
                if let Some(n) = result_set[constraint.sup] {
                    let m = n + 1;
                    if result_set[constraint.sub]
                        .map(|distance| m < distance)
                        .unwrap_or(true)
                    {
                        result_set[constraint.sub] = Some(m);
                        changed = true;
                    }
                }
            }
        }

        result_set
    }
}
