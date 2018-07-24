// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::nll::region_infer::values::ToElementIndex;
use borrow_check::nll::region_infer::{ConstraintIndex, RegionInferenceContext};
use borrow_check::nll::type_check::Locations;
use rustc::hir::def_id::DefId;
use rustc::infer::error_reporting::nice_region_error::NiceRegionError;
use rustc::infer::InferCtxt;
use rustc::mir::{self, Location, Mir, Place, Rvalue, StatementKind, TerminatorKind};
use rustc::ty::RegionVid;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_errors::Diagnostic;
use std::fmt;
use syntax_pos::Span;

mod region_name;
mod var_name;

/// Constraints that are considered interesting can be categorized to
/// determine why they are interesting. Order of variants indicates
/// sort order of the category, thereby influencing diagnostic output.
#[derive(Debug, Eq, PartialEq, PartialOrd, Ord)]
enum ConstraintCategory {
    Cast,
    Assignment,
    AssignmentToUpvar,
    Return,
    CallArgumentToUpvar,
    CallArgument,
    Other,
    Boring,
}

impl fmt::Display for ConstraintCategory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ConstraintCategory::Assignment |
            ConstraintCategory::AssignmentToUpvar => write!(f, "assignment"),
            ConstraintCategory::Return => write!(f, "return"),
            ConstraintCategory::Cast => write!(f, "cast"),
            ConstraintCategory::CallArgument |
            ConstraintCategory::CallArgumentToUpvar => write!(f, "argument"),
            _ => write!(f, "free region"),
        }
    }
}

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Walks the graph of constraints (where `'a: 'b` is considered
    /// an edge `'a -> 'b`) to find all paths from `from_region` to
    /// `to_region`. The paths are accumulated into the vector
    /// `results`. The paths are stored as a series of
    /// `ConstraintIndex` values -- in other words, a list of *edges*.
    fn find_constraint_paths_between_regions(
        &self,
        from_region: RegionVid,
        target_test: impl Fn(RegionVid) -> bool,
    ) -> Vec<Vec<ConstraintIndex>> {
        let mut results = vec![];
        self.find_constraint_paths_between_regions_helper(
            from_region,
            from_region,
            &target_test,
            &mut FxHashSet::default(),
            &mut vec![],
            &mut results,
        );
        results
    }

    /// Helper for `find_constraint_paths_between_regions`.
    fn find_constraint_paths_between_regions_helper(
        &self,
        from_region: RegionVid,
        current_region: RegionVid,
        target_test: &impl Fn(RegionVid) -> bool,
        visited: &mut FxHashSet<RegionVid>,
        stack: &mut Vec<ConstraintIndex>,
        results: &mut Vec<Vec<ConstraintIndex>>,
    ) {
        // Check if we already visited this region.
        if !visited.insert(current_region) {
            return;
        }

        // Check if we reached the region we were looking for.
        if target_test(current_region) {
            if !stack.is_empty() {
                assert_eq!(self.constraints[stack[0]].sup, from_region);
                results.push(stack.clone());
            }
            return;
        }

        for constraint in self.constraint_graph.outgoing_edges(current_region) {
            assert_eq!(self.constraints[constraint].sup, current_region);
            stack.push(constraint);
            self.find_constraint_paths_between_regions_helper(
                from_region,
                self.constraints[constraint].sub,
                target_test,
                visited,
                stack,
                results,
            );
            stack.pop();
        }
    }

    /// This function will return true if a constraint is interesting and false if a constraint
    /// is not. It is useful in filtering constraint paths to only interesting points.
    fn constraint_is_interesting(&self, index: ConstraintIndex) -> bool {
        let constraint = self.constraints[index];
        debug!(
            "constraint_is_interesting: locations={:?} constraint={:?}",
            constraint.locations, constraint
        );
        if let Locations::Interesting(_) = constraint.locations {
            true
        } else {
            false
        }
    }

    /// This function classifies a constraint from a location.
    fn classify_constraint(
        &self,
        index: ConstraintIndex,
        mir: &Mir<'tcx>,
        _infcx: &InferCtxt<'_, '_, 'tcx>,
    ) -> (ConstraintCategory, Span) {
        let constraint = self.constraints[index];
        debug!("classify_constraint: constraint={:?}", constraint);
        let span = constraint.locations.span(mir);
        let location = constraint.locations.from_location().unwrap_or(Location::START);

        if !self.constraint_is_interesting(index) {
            return (ConstraintCategory::Boring, span);
        }

        let data = &mir[location.block];
        debug!("classify_constraint: location={:?} data={:?}", location, data);
        let category = if location.statement_index == data.statements.len() {
            if let Some(ref terminator) = data.terminator {
                debug!("classify_constraint: terminator.kind={:?}", terminator.kind);
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
            debug!("classify_constraint: statement.kind={:?}", statement.kind);
            match statement.kind {
                StatementKind::Assign(ref place, ref rvalue) => {
                    debug!("classify_constraint: place={:?} rvalue={:?}", place, rvalue);
                    if *place == Place::Local(mir::RETURN_PLACE) {
                        ConstraintCategory::Return
                    } else {
                        match rvalue {
                            Rvalue::Cast(..) => ConstraintCategory::Cast,
                            Rvalue::Use(..) |
                            Rvalue::Aggregate(..) => ConstraintCategory::Assignment,
                            _ => ConstraintCategory::Other,
                        }
                    }
                }
                _ => ConstraintCategory::Other,
            }
        };

        (category, span)
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
        errors_buffer: &mut Vec<Diagnostic>,
    ) {
        debug!("report_error(fr={:?}, outlived_fr={:?})", fr, outlived_fr);

        if let (Some(f), Some(o)) = (self.to_error_region(fr), self.to_error_region(outlived_fr)) {
            let tables = infcx.tcx.typeck_tables_of(mir_def_id);
            let nice = NiceRegionError::new_from_span(infcx.tcx, blame_span, o, f, Some(tables));
            if let Some(_error_reported) = nice.try_report() {
                return;
            }
        }

        // Find all paths
        let constraint_paths = self.find_constraint_paths_between_regions(fr, |r| r == outlived_fr);
        debug!("report_error: constraint_paths={:#?}", constraint_paths);

        // Find the shortest such path.
        let path = constraint_paths.iter().min_by_key(|p| p.len()).unwrap();
        debug!("report_error: shortest_path={:?}", path);

        // Classify each of the constraints along the path.
        let mut categorized_path: Vec<(ConstraintCategory, Span)> = path.iter()
            .map(|&index| self.classify_constraint(index, mir, infcx))
            .collect();
        debug!("report_error: categorized_path={:?}", categorized_path);

        // Find what appears to be the most interesting path to report to the user.
        categorized_path.sort_unstable_by(|p0, p1| p0.0.cmp(&p1.0));
        debug!("report_error: sorted_path={:?}", categorized_path);

        // Get a span
        let (category, span) = categorized_path.first().unwrap();

        let category = match (
            category,
            self.universal_regions.is_local_free_region(fr),
            self.universal_regions.is_local_free_region(outlived_fr),
        ) {
            (ConstraintCategory::Assignment, true, false) =>
                &ConstraintCategory::AssignmentToUpvar,
            (ConstraintCategory::CallArgument, true, false) =>
                &ConstraintCategory::CallArgumentToUpvar,
            (category, _, _) => category,
        };

        debug!("report_error: category={:?}", category);
        match category {
            ConstraintCategory::AssignmentToUpvar |
            ConstraintCategory::CallArgumentToUpvar =>
                self.report_closure_error(
                    mir, infcx, mir_def_id, fr, outlived_fr, category, span, errors_buffer),
            _ =>
                self.report_general_error(
                    mir, infcx, mir_def_id, fr, outlived_fr, category, span, errors_buffer),
        }
    }

    fn report_closure_error(
        &self,
        mir: &Mir<'tcx>,
        infcx: &InferCtxt<'_, '_, 'tcx>,
        mir_def_id: DefId,
        fr: RegionVid,
        outlived_fr: RegionVid,
        category: &ConstraintCategory,
        span: &Span,
        errors_buffer: &mut Vec<Diagnostic>,
    ) {
        let fr_name_and_span  = self.get_var_name_and_span_for_region(
            infcx.tcx, mir, fr);
        let outlived_fr_name_and_span = self.get_var_name_and_span_for_region(
            infcx.tcx, mir,outlived_fr);

        if fr_name_and_span.is_none() && outlived_fr_name_and_span.is_none() {
            return self.report_general_error(
                mir, infcx, mir_def_id, fr, outlived_fr, category, span, errors_buffer);
        }

        let mut diag = infcx.tcx.sess.struct_span_err(
            *span, &format!("borrowed data escapes outside of closure"),
        );

        if let Some((outlived_fr_name, outlived_fr_span)) = outlived_fr_name_and_span {
            if let Some(name) = outlived_fr_name {
                diag.span_label(
                    outlived_fr_span,
                    format!("`{}` is declared here, outside of the closure body", name),
                );
            }
        }

        if let Some((fr_name, fr_span)) = fr_name_and_span {
            if let Some(name) = fr_name {
                diag.span_label(
                    fr_span,
                    format!("`{}` is a reference that is only valid in the closure body", name),
                );

                diag.span_label(*span, format!("`{}` escapes the closure body here", name));
            }
        }

        diag.buffer(errors_buffer);
    }

    fn report_general_error(
        &self,
        mir: &Mir<'tcx>,
        infcx: &InferCtxt<'_, '_, 'tcx>,
        mir_def_id: DefId,
        fr: RegionVid,
        outlived_fr: RegionVid,
        category: &ConstraintCategory,
        span: &Span,
        errors_buffer: &mut Vec<Diagnostic>,
    ) {
        let mut diag = infcx.tcx.sess.struct_span_err(
            *span, &format!("unsatisfied lifetime constraints"), // FIXME
        );

        let counter = &mut 1;
        let fr_name = self.give_region_a_name(
            infcx.tcx, mir, mir_def_id, fr, counter, &mut diag);
        let outlived_fr_name = self.give_region_a_name(
            infcx.tcx, mir, mir_def_id, outlived_fr, counter, &mut diag);

        diag.span_label(*span, format!(
            "{} requires that `{}` must outlive `{}`",
            category, fr_name, outlived_fr_name,
        ));

        diag.buffer(errors_buffer);
    }

    // Find some constraint `X: Y` where:
    // - `fr1: X` transitively
    // - and `Y` is live at `elem`
    crate fn find_constraint(&self, fr1: RegionVid, elem: Location) -> RegionVid {
        let index = self.blame_constraint(fr1, elem);
        self.constraints[index].sub
    }

    /// Tries to finds a good span to blame for the fact that `fr1`
    /// contains `fr2`.
    pub(super) fn blame_constraint(
        &self,
        fr1: RegionVid,
        elem: impl ToElementIndex,
    ) -> ConstraintIndex {
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
            for constraint in self.constraints.iter() {
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
