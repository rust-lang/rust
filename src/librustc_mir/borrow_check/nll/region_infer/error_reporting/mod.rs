// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::nll::region_infer::{ConstraintIndex, RegionInferenceContext};
use borrow_check::nll::type_check::Locations;
use rustc::hir::def_id::DefId;
use rustc::infer::error_reporting::nice_region_error::NiceRegionError;
use rustc::infer::InferCtxt;
use rustc::mir::{self, Location, Mir, Place, Rvalue, StatementKind, TerminatorKind};
use rustc::ty::RegionVid;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_errors::Diagnostic;
use std::collections::VecDeque;
use std::fmt;
use syntax_pos::Span;

mod region_name;
mod var_name;

/// Constraints that are considered interesting can be categorized to
/// determine why they are interesting. Order of variants indicates
/// sort order of the category, thereby influencing diagnostic output.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
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
        // Must end with a space. Allows for empty names to be provided.
        match self {
            ConstraintCategory::Assignment => write!(f, "assignment "),
            ConstraintCategory::Return => write!(f, "return "),
            ConstraintCategory::Cast => write!(f, "cast "),
            ConstraintCategory::CallArgument => write!(f, "argument "),
            _ => write!(f, ""),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Trace {
    StartRegion,
    FromConstraint(ConstraintIndex),
    NotVisited,
}

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Tries to find the best constraint to blame for the fact that
    /// `R: from_region`, where `R` is some region that meets
    /// `target_test`. This works by following the constraint graph,
    /// creating a constraint path that forces `R` to outlive
    /// `from_region`, and then finding the best choices within that
    /// path to blame.
    fn best_blame_constraint(
        &self,
        mir: &Mir<'tcx>,
        from_region: RegionVid,
        target_test: impl Fn(RegionVid) -> bool,
    ) -> (ConstraintCategory, Span, RegionVid) {
        debug!("best_blame_constraint(from_region={:?})", from_region);

        // Find all paths
        let (path, target_region) = self
            .find_constraint_paths_between_regions(from_region, target_test)
            .unwrap();
        debug!(
            "best_blame_constraint: path={:#?}",
            path.iter()
                .map(|&ci| format!(
                    "{:?}: {:?} ({:?}: {:?})",
                    ci,
                    &self.constraints[ci],
                    self.constraint_sccs.scc(self.constraints[ci].sup),
                    self.constraint_sccs.scc(self.constraints[ci].sub),
                ))
                .collect::<Vec<_>>()
        );

        // Classify each of the constraints along the path.
        let mut categorized_path: Vec<(ConstraintCategory, Span)> = path
            .iter()
            .map(|&index| self.classify_constraint(index, mir))
            .collect();
        debug!(
            "best_blame_constraint: categorized_path={:#?}",
            categorized_path
        );

        // To find the best span to cite, we first try to look for the
        // final constraint that is interesting and where the `sup` is
        // not unified with the ultimate target region. The reason
        // for this is that we have a chain of constraints that lead
        // from the source to the target region, something like:
        //
        //    '0: '1 ('0 is the source)
        //    '1: '2
        //    '2: '3
        //    '3: '4
        //    '4: '5
        //    '5: '6 ('6 is the target)
        //
        // Some of those regions are unified with `'6` (in the same
        // SCC).  We want to screen those out. After that point, the
        // "closest" constraint we have to the end is going to be the
        // most likely to be the point where the value escapes -- but
        // we still want to screen for an "interesting" point to
        // highlight (e.g., a call site or something).
        let target_scc = self.constraint_sccs.scc(target_region);
        let best_choice = (0..path.len()).rev().find(|&i| {
            let constraint = &self.constraints[path[i]];

            let constraint_sup_scc = self.constraint_sccs.scc(constraint.sup);
            if constraint_sup_scc == target_scc {
                return false;
            }

            match categorized_path[i].0 {
                ConstraintCategory::Boring => false,
                _ => true,
            }
        });
        if let Some(i) = best_choice {
            let (category, span) = categorized_path[i];
            return (category, span, target_region);
        }

        // If that search fails, that is.. unusual. Maybe everything
        // is in the same SCC or something. In that case, find what
        // appears to be the most interesting point to report to the
        // user via an even more ad-hoc guess.
        categorized_path.sort_by(|p0, p1| p0.0.cmp(&p1.0));
        debug!("best_blame_constraint: sorted_path={:#?}", categorized_path);

        let &(category, span) = categorized_path.first().unwrap();

        (category, span, target_region)
    }

    /// Walks the graph of constraints (where `'a: 'b` is considered
    /// an edge `'a -> 'b`) to find all paths from `from_region` to
    /// `to_region`. The paths are accumulated into the vector
    /// `results`. The paths are stored as a series of
    /// `ConstraintIndex` values -- in other words, a list of *edges*.
    ///
    /// Returns: a series of constraints as well as the region `R`
    /// that passed the target test.
    fn find_constraint_paths_between_regions(
        &self,
        from_region: RegionVid,
        target_test: impl Fn(RegionVid) -> bool,
    ) -> Option<(Vec<ConstraintIndex>, RegionVid)> {
        let mut context = IndexVec::from_elem(Trace::NotVisited, &self.definitions);
        context[from_region] = Trace::StartRegion;

        // Use a deque so that we do a breadth-first search. We will
        // stop at the first match, which ought to be the shortest
        // path (fewest constraints).
        let mut deque = VecDeque::new();
        deque.push_back(from_region);

        while let Some(r) = deque.pop_front() {
            // Check if we reached the region we were looking for. If so,
            // we can reconstruct the path that led to it and return it.
            if target_test(r) {
                let mut result = vec![];
                let mut p = r;
                loop {
                    match context[p] {
                        Trace::NotVisited => {
                            bug!("found unvisited region {:?} on path to {:?}", p, r)
                        }
                        Trace::FromConstraint(c) => {
                            result.push(c);
                            p = self.constraints[c].sup;
                        }

                        Trace::StartRegion => {
                            result.reverse();
                            return Some((result, r));
                        }
                    }
                }
            }

            // Otherwise, walk over the outgoing constraints and
            // enqueue any regions we find, keeping track of how we
            // reached them.
            for constraint in self.constraint_graph.outgoing_edges(r) {
                assert_eq!(self.constraints[constraint].sup, r);
                let sub_region = self.constraints[constraint].sub;
                if let Trace::NotVisited = context[sub_region] {
                    context[sub_region] = Trace::FromConstraint(constraint);
                    deque.push_back(sub_region);
                }
            }
        }

        None
    }

    /// This function will return true if a constraint is interesting and false if a constraint
    /// is not. It is useful in filtering constraint paths to only interesting points.
    fn constraint_is_interesting(&self, index: ConstraintIndex) -> bool {
        let constraint = self.constraints[index];
        debug!(
            "constraint_is_interesting: locations={:?} constraint={:?}",
            constraint.locations, constraint
        );

        match constraint.locations {
            Locations::Interesting(_) | Locations::All => true,
            _ => false,
        }
    }

    /// This function classifies a constraint from a location.
    fn classify_constraint(
        &self,
        index: ConstraintIndex,
        mir: &Mir<'tcx>,
    ) -> (ConstraintCategory, Span) {
        let constraint = self.constraints[index];
        debug!("classify_constraint: constraint={:?}", constraint);
        let span = constraint.locations.span(mir);
        let location = constraint
            .locations
            .from_location()
            .unwrap_or(Location::START);

        if !self.constraint_is_interesting(index) {
            return (ConstraintCategory::Boring, span);
        }

        let data = &mir[location.block];
        debug!(
            "classify_constraint: location={:?} data={:?}",
            location, data
        );
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
                            Rvalue::Use(..) | Rvalue::Aggregate(..) => {
                                ConstraintCategory::Assignment
                            }
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
        errors_buffer: &mut Vec<Diagnostic>,
    ) {
        debug!("report_error(fr={:?}, outlived_fr={:?})", fr, outlived_fr);

        let (category, span, _) = self.best_blame_constraint(mir, fr, |r| r == outlived_fr);

        // Check if we can use one of the "nice region errors".
        if let (Some(f), Some(o)) = (self.to_error_region(fr), self.to_error_region(outlived_fr)) {
            let tables = infcx.tcx.typeck_tables_of(mir_def_id);
            let nice = NiceRegionError::new_from_span(infcx.tcx, span, o, f, Some(tables));
            if let Some(_error_reported) = nice.try_report() {
                return;
            }
        }

        let (fr_is_local, outlived_fr_is_local): (bool, bool) = (
            self.universal_regions.is_local_free_region(fr),
            self.universal_regions.is_local_free_region(outlived_fr),
        );
        debug!("report_error: fr_is_local={:?} outlived_fr_is_local={:?} category={:?}",
               fr_is_local, outlived_fr_is_local, category);

        match (category, fr_is_local, outlived_fr_is_local) {
            (ConstraintCategory::Assignment, true, false) |
            (ConstraintCategory::CallArgument, true, false) =>
                self.report_escaping_data_error(mir, infcx, mir_def_id, fr, outlived_fr,
                                                category, span, errors_buffer),
            _ =>
                self.report_general_error(mir, infcx, mir_def_id, fr, fr_is_local,
                                          outlived_fr, outlived_fr_is_local,
                                          category, span, errors_buffer),
        };
    }

    fn report_escaping_data_error(
        &self,
        mir: &Mir<'tcx>,
        infcx: &InferCtxt<'_, '_, 'tcx>,
        mir_def_id: DefId,
        fr: RegionVid,
        outlived_fr: RegionVid,
        category: ConstraintCategory,
        span: Span,
        errors_buffer: &mut Vec<Diagnostic>,
    ) {
        let fr_name_and_span = self.get_var_name_and_span_for_region(infcx.tcx, mir, fr);
        let outlived_fr_name_and_span =
            self.get_var_name_and_span_for_region(infcx.tcx, mir, outlived_fr);

        let escapes_from = if infcx.tcx.is_closure(mir_def_id) { "closure" } else { "function" };

        if fr_name_and_span.is_none() && outlived_fr_name_and_span.is_none() {
            return self.report_general_error(mir, infcx, mir_def_id,
                                             fr, true, outlived_fr, false,
                                             category, span, errors_buffer);
        }

        let mut diag = infcx.tcx.sess.struct_span_err(
            span, &format!("borrowed data escapes outside of {}", escapes_from),
        );

        if let Some((outlived_fr_name, outlived_fr_span)) = outlived_fr_name_and_span {
            if let Some(name) = outlived_fr_name {
                diag.span_label(
                    outlived_fr_span,
                    format!("`{}` is declared here, outside of the {} body", name, escapes_from),
                );
            }
        }

        if let Some((fr_name, fr_span)) = fr_name_and_span {
            if let Some(name) = fr_name {
                diag.span_label(
                    fr_span,
                    format!("`{}` is a reference that is only valid in the {} body",
                            name, escapes_from),
                );

                diag.span_label(span, format!("`{}` escapes the {} body here",
                                               name, escapes_from));
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
        fr_is_local: bool,
        outlived_fr: RegionVid,
        outlived_fr_is_local: bool,
        category: ConstraintCategory,
        span: Span,
        errors_buffer: &mut Vec<Diagnostic>,
    ) {
        let mut diag = infcx.tcx.sess.struct_span_err(
            span,
            "unsatisfied lifetime constraints", // FIXME
        );

        let counter = &mut 1;
        let fr_name = self.give_region_a_name(
            infcx, mir, mir_def_id, fr, counter, &mut diag);
        let outlived_fr_name = self.give_region_a_name(
            infcx, mir, mir_def_id, outlived_fr, counter, &mut diag);

        let mir_def_name = if infcx.tcx.is_closure(mir_def_id) { "closure" } else { "function" };

        match (category, outlived_fr_is_local, fr_is_local) {
            (ConstraintCategory::Return, true, _) => {
                diag.span_label(span, format!(
                    "{} was supposed to return data with lifetime `{}` but it is returning \
                    data with lifetime `{}`",
                    mir_def_name, fr_name, outlived_fr_name,
                ));
            },
            _ => {
                diag.span_label(span, format!(
                    "{}requires that `{}` must outlive `{}`",
                    category, fr_name, outlived_fr_name,
                ));
            },
        }

        diag.buffer(errors_buffer);
    }

    // Finds some region R such that `fr1: R` and `R` is live at
    // `elem`.
    crate fn find_sub_region_live_at(&self, fr1: RegionVid, elem: Location) -> RegionVid {
        // Find all paths
        let (_path, r) =
            self.find_constraint_paths_between_regions(fr1, |r| {
                self.liveness_constraints.contains(r, elem)
            }).unwrap();
        r
    }

    // Finds a good span to blame for the fact that `fr1` outlives `fr2`.
    crate fn find_outlives_blame_span(
        &self,
        mir: &Mir<'tcx>,
        fr1: RegionVid,
        fr2: RegionVid,
    ) -> Span {
        let (_, span, _) = self.best_blame_constraint(mir, fr1, |r| r == fr2);
        span
    }
}
