use crate::borrow_check::nll::constraints::OutlivesConstraint;
use crate::borrow_check::nll::region_infer::AppliedMemberConstraint;
use crate::borrow_check::nll::region_infer::RegionInferenceContext;
use crate::borrow_check::nll::type_check::Locations;
use crate::borrow_check::nll::universal_regions::DefiningTy;
use crate::borrow_check::nll::ConstraintDescription;
use crate::util::borrowck_errors::{BorrowckErrors, Origin};
use crate::borrow_check::Upvar;
use rustc::hir::def_id::DefId;
use rustc::infer::error_reporting::nice_region_error::NiceRegionError;
use rustc::infer::InferCtxt;
use rustc::infer::NLLRegionVariableOrigin;
use rustc::mir::{ConstraintCategory, Location, Body};
use rustc::ty::{self, RegionVid};
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_errors::{Diagnostic, DiagnosticBuilder};
use std::collections::VecDeque;
use syntax::errors::Applicability;
use syntax::symbol::kw;
use syntax_pos::Span;

mod region_name;
mod var_name;

crate use self::region_name::{RegionName, RegionNameSource};

impl ConstraintDescription for ConstraintCategory {
    fn description(&self) -> &'static str {
        // Must end with a space. Allows for empty names to be provided.
        match self {
            ConstraintCategory::Assignment => "assignment ",
            ConstraintCategory::Return => "returning this value ",
            ConstraintCategory::Yield => "yielding this value ",
            ConstraintCategory::UseAsConst => "using this value as a constant ",
            ConstraintCategory::UseAsStatic => "using this value as a static ",
            ConstraintCategory::Cast => "cast ",
            ConstraintCategory::CallArgument => "argument ",
            ConstraintCategory::TypeAnnotation => "type annotation ",
            ConstraintCategory::ClosureBounds => "closure body ",
            ConstraintCategory::SizedBound => "proving this value is `Sized` ",
            ConstraintCategory::CopyBound => "copying this value ",
            ConstraintCategory::OpaqueType => "opaque type ",
            ConstraintCategory::Boring
            | ConstraintCategory::BoringNoLocation
            | ConstraintCategory::Internal => "",
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Trace {
    StartRegion,
    FromOutlivesConstraint(OutlivesConstraint),
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
        body: &Body<'tcx>,
        from_region: RegionVid,
        target_test: impl Fn(RegionVid) -> bool,
    ) -> (ConstraintCategory, bool, Span) {
        debug!("best_blame_constraint(from_region={:?})", from_region);

        // Find all paths
        let (path, target_region) =
            self.find_constraint_paths_between_regions(from_region, target_test)
                .unwrap();
        debug!(
            "best_blame_constraint: path={:#?}",
            path.iter()
                .map(|&c| format!(
                    "{:?} ({:?}: {:?})",
                    c,
                    self.constraint_sccs.scc(c.sup),
                    self.constraint_sccs.scc(c.sub),
                ))
                .collect::<Vec<_>>()
        );

        // Classify each of the constraints along the path.
        let mut categorized_path: Vec<(ConstraintCategory, bool, Span)> = path.iter()
            .map(|constraint| {
                if constraint.category == ConstraintCategory::ClosureBounds {
                    self.retrieve_closure_constraint_info(body, &constraint)
                } else {
                    (constraint.category, false, constraint.locations.span(body))
                }
            })
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
            let constraint = path[i];

            let constraint_sup_scc = self.constraint_sccs.scc(constraint.sup);

            match categorized_path[i].0 {
                ConstraintCategory::OpaqueType | ConstraintCategory::Boring |
                ConstraintCategory::BoringNoLocation | ConstraintCategory::Internal => false,
                ConstraintCategory::TypeAnnotation | ConstraintCategory::Return |
                ConstraintCategory::Yield => true,
                _ => constraint_sup_scc != target_scc,
            }
        });
        if let Some(i) = best_choice {
            if let Some(next) = categorized_path.get(i + 1) {
                if categorized_path[i].0 == ConstraintCategory::Return
                    && next.0 == ConstraintCategory::OpaqueType
                {
                    // The return expression is being influenced by the return type being
                    // impl Trait, point at the return type and not the return expr.
                    return *next;
                }
            }
            return categorized_path[i];
        }

        // If that search fails, that is.. unusual. Maybe everything
        // is in the same SCC or something. In that case, find what
        // appears to be the most interesting point to report to the
        // user via an even more ad-hoc guess.
        categorized_path.sort_by(|p0, p1| p0.0.cmp(&p1.0));
        debug!("`: sorted_path={:#?}", categorized_path);

        *categorized_path.first().unwrap()
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
    ) -> Option<(Vec<OutlivesConstraint>, RegionVid)> {
        let mut context = IndexVec::from_elem(Trace::NotVisited, &self.definitions);
        context[from_region] = Trace::StartRegion;

        // Use a deque so that we do a breadth-first search. We will
        // stop at the first match, which ought to be the shortest
        // path (fewest constraints).
        let mut deque = VecDeque::new();
        deque.push_back(from_region);

        while let Some(r) = deque.pop_front() {
            debug!(
                "find_constraint_paths_between_regions: from_region={:?} r={:?} value={}",
                from_region,
                r,
                self.region_value_str(r),
            );

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

                        Trace::FromOutlivesConstraint(c) => {
                            result.push(c);
                            p = c.sup;
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

            // A constraint like `'r: 'x` can come from our constraint
            // graph.
            let fr_static = self.universal_regions.fr_static;
            let outgoing_edges_from_graph = self.constraint_graph
                .outgoing_edges(r, &self.constraints, fr_static);


            // But member constraints can also give rise to `'r: 'x`
            // edges that were not part of the graph initially, so
            // watch out for those.
            let outgoing_edges_from_picks = self.applied_member_constraints(r)
                .iter()
                .map(|&AppliedMemberConstraint { min_choice, member_constraint_index, .. }| {
                    let p_c = &self.member_constraints[member_constraint_index];
                    OutlivesConstraint {
                        sup: r,
                        sub: min_choice,
                        locations: Locations::All(p_c.definition_span),
                        category: ConstraintCategory::OpaqueType,
                    }
                });

            for constraint in outgoing_edges_from_graph.chain(outgoing_edges_from_picks) {
                debug_assert_eq!(constraint.sup, r);
                let sub_region = constraint.sub;
                if let Trace::NotVisited = context[sub_region] {
                    context[sub_region] = Trace::FromOutlivesConstraint(constraint);
                    deque.push_back(sub_region);
                }
            }
        }

        None
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
        body: &Body<'tcx>,
        upvars: &[Upvar],
        infcx: &InferCtxt<'_, 'tcx>,
        mir_def_id: DefId,
        fr: RegionVid,
        outlived_fr: RegionVid,
        errors_buffer: &mut Vec<Diagnostic>,
    ) {
        debug!("report_error(fr={:?}, outlived_fr={:?})", fr, outlived_fr);

        let (category, _, span) = self.best_blame_constraint(body, fr, |r| {
            self.provides_universal_region(r, fr, outlived_fr)
        });

        debug!("report_error: category={:?} {:?}", category, span);
        // Check if we can use one of the "nice region errors".
        if let (Some(f), Some(o)) = (self.to_error_region(fr), self.to_error_region(outlived_fr)) {
            let tables = infcx.tcx.typeck_tables_of(mir_def_id);
            let nice = NiceRegionError::new_from_span(infcx, span, o, f, Some(tables));
            if let Some(diag) = nice.try_report_from_nll() {
                diag.buffer(errors_buffer);
                return;
            }
        }

        let (fr_is_local, outlived_fr_is_local): (bool, bool) = (
            self.universal_regions.is_local_free_region(fr),
            self.universal_regions.is_local_free_region(outlived_fr),
        );

        debug!(
            "report_error: fr_is_local={:?} outlived_fr_is_local={:?} category={:?}",
            fr_is_local, outlived_fr_is_local, category
        );
        match (category, fr_is_local, outlived_fr_is_local) {
            (ConstraintCategory::Return, true, false) if self.is_closure_fn_mut(infcx, fr) => {
                self.report_fnmut_error(
                    body,
                    upvars,
                    infcx,
                    mir_def_id,
                    fr,
                    outlived_fr,
                    span,
                    errors_buffer,
                )
            }
            (ConstraintCategory::Assignment, true, false)
            | (ConstraintCategory::CallArgument, true, false) => self.report_escaping_data_error(
                body,
                upvars,
                infcx,
                mir_def_id,
                fr,
                outlived_fr,
                category,
                span,
                errors_buffer,
            ),
            _ => self.report_general_error(
                body,
                upvars,
                infcx,
                mir_def_id,
                fr,
                fr_is_local,
                outlived_fr,
                outlived_fr_is_local,
                category,
                span,
                errors_buffer,
            ),
        };
    }

    /// We have a constraint `fr1: fr2` that is not satisfied, where
    /// `fr2` represents some universal region. Here, `r` is some
    /// region where we know that `fr1: r` and this function has the
    /// job of determining whether `r` is "to blame" for the fact that
    /// `fr1: fr2` is required.
    ///
    /// This is true under two conditions:
    ///
    /// - `r == fr2`
    /// - `fr2` is `'static` and `r` is some placeholder in a universe
    ///   that cannot be named by `fr1`; in that case, we will require
    ///   that `fr1: 'static` because it is the only way to `fr1: r` to
    ///   be satisfied. (See `add_incompatible_universe`.)
    fn provides_universal_region(&self, r: RegionVid, fr1: RegionVid, fr2: RegionVid) -> bool {
        debug!(
            "provides_universal_region(r={:?}, fr1={:?}, fr2={:?})",
            r, fr1, fr2
        );
        let result = {
            r == fr2 || {
                fr2 == self.universal_regions.fr_static && self.cannot_name_placeholder(fr1, r)
            }
        };
        debug!("provides_universal_region: result = {:?}", result);
        result
    }

    /// Report a specialized error when `FnMut` closures return a reference to a captured variable.
    /// This function expects `fr` to be local and `outlived_fr` to not be local.
    ///
    /// ```text
    /// error: captured variable cannot escape `FnMut` closure body
    ///   --> $DIR/issue-53040.rs:15:8
    ///    |
    /// LL |     || &mut v;
    ///    |     -- ^^^^^^ creates a reference to a captured variable which escapes the closure body
    ///    |     |
    ///    |     inferred to be a `FnMut` closure
    ///    |
    ///    = note: `FnMut` closures only have access to their captured variables while they are
    ///            executing...
    ///    = note: ...therefore, returned references to captured variables will escape the closure
    /// ```
    fn report_fnmut_error(
        &self,
        body: &Body<'tcx>,
        upvars: &[Upvar],
        infcx: &InferCtxt<'_, 'tcx>,
        mir_def_id: DefId,
        _fr: RegionVid,
        outlived_fr: RegionVid,
        span: Span,
        errors_buffer: &mut Vec<Diagnostic>,
    ) {
        let mut diag = infcx
            .tcx
            .sess
            .struct_span_err(span, "captured variable cannot escape `FnMut` closure body");

        // We should check if the return type of this closure is in fact a closure - in that
        // case, we can special case the error further.
        let return_type_is_closure = self.universal_regions.unnormalized_output_ty.is_closure();
        let message = if return_type_is_closure {
            "returns a closure that contains a reference to a captured variable, which then \
             escapes the closure body"
        } else {
            "returns a reference to a captured variable which escapes the closure body"
        };

        diag.span_label(span, message);

        match self.give_region_a_name(infcx, body, upvars, mir_def_id, outlived_fr, &mut 1)
            .unwrap().source
        {
            RegionNameSource::NamedEarlyBoundRegion(fr_span)
            | RegionNameSource::NamedFreeRegion(fr_span)
            | RegionNameSource::SynthesizedFreeEnvRegion(fr_span, _)
            | RegionNameSource::CannotMatchHirTy(fr_span, _)
            | RegionNameSource::MatchedHirTy(fr_span)
            | RegionNameSource::MatchedAdtAndSegment(fr_span)
            | RegionNameSource::AnonRegionFromUpvar(fr_span, _)
            | RegionNameSource::AnonRegionFromOutput(fr_span, _, _) => {
                diag.span_label(fr_span, "inferred to be a `FnMut` closure");
            }
            _ => {}
        }

        diag.note(
            "`FnMut` closures only have access to their captured variables while they are \
             executing...",
        );
        diag.note("...therefore, they cannot allow references to captured variables to escape");

        diag.buffer(errors_buffer);
    }

    /// Reports a error specifically for when data is escaping a closure.
    ///
    /// ```text
    /// error: borrowed data escapes outside of function
    ///   --> $DIR/lifetime-bound-will-change-warning.rs:44:5
    ///    |
    /// LL | fn test2<'a>(x: &'a Box<Fn()+'a>) {
    ///    |              - `x` is a reference that is only valid in the function body
    /// LL |     // but ref_obj will not, so warn.
    /// LL |     ref_obj(x)
    ///    |     ^^^^^^^^^^ `x` escapes the function body here
    /// ```
    fn report_escaping_data_error(
        &self,
        body: &Body<'tcx>,
        upvars: &[Upvar],
        infcx: &InferCtxt<'_, 'tcx>,
        mir_def_id: DefId,
        fr: RegionVid,
        outlived_fr: RegionVid,
        category: ConstraintCategory,
        span: Span,
        errors_buffer: &mut Vec<Diagnostic>,
    ) {
        let fr_name_and_span =
            self.get_var_name_and_span_for_region(infcx.tcx, body, upvars, fr);
        let outlived_fr_name_and_span =
            self.get_var_name_and_span_for_region(infcx.tcx, body, upvars, outlived_fr);

        let escapes_from = match self.universal_regions.defining_ty {
            DefiningTy::Closure(..) => "closure",
            DefiningTy::Generator(..) => "generator",
            DefiningTy::FnDef(..) => "function",
            DefiningTy::Const(..) => "const",
        };

        // Revert to the normal error in these cases.
        // Assignments aren't "escapes" in function items.
        if (fr_name_and_span.is_none() && outlived_fr_name_and_span.is_none())
            || (category == ConstraintCategory::Assignment && escapes_from == "function")
            || escapes_from == "const"
        {
            return self.report_general_error(
                body,
                upvars,
                infcx,
                mir_def_id,
                fr,
                true,
                outlived_fr,
                false,
                category,
                span,
                errors_buffer,
            );
        }

        let mut diag = infcx
            .tcx
            .borrowed_data_escapes_closure(span, escapes_from, Origin::Mir);

        if let Some((Some(outlived_fr_name), outlived_fr_span)) = outlived_fr_name_and_span {
            diag.span_label(
                outlived_fr_span,
                format!(
                    "`{}` is declared here, outside of the {} body",
                    outlived_fr_name, escapes_from
                ),
            );
        }

        if let Some((Some(fr_name), fr_span)) = fr_name_and_span {
            diag.span_label(
                fr_span,
                format!(
                    "`{}` is a reference that is only valid in the {} body",
                    fr_name, escapes_from
                ),
            );

            diag.span_label(
                span,
                format!("`{}` escapes the {} body here", fr_name, escapes_from),
            );
        }

        diag.buffer(errors_buffer);
    }

    /// Reports a region inference error for the general case with named/synthesized lifetimes to
    /// explain what is happening.
    ///
    /// ```text
    /// error: unsatisfied lifetime constraints
    ///   --> $DIR/regions-creating-enums3.rs:17:5
    ///    |
    /// LL | fn mk_add_bad1<'a,'b>(x: &'a ast<'a>, y: &'b ast<'b>) -> ast<'a> {
    ///    |                -- -- lifetime `'b` defined here
    ///    |                |
    ///    |                lifetime `'a` defined here
    /// LL |     ast::add(x, y)
    ///    |     ^^^^^^^^^^^^^^ function was supposed to return data with lifetime `'a` but it
    ///    |                    is returning data with lifetime `'b`
    /// ```
    fn report_general_error(
        &self,
        body: &Body<'tcx>,
        upvars: &[Upvar],
        infcx: &InferCtxt<'_, 'tcx>,
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
            "lifetime may not live long enough"
        );

        let counter = &mut 1;
        let fr_name = self.give_region_a_name(
            infcx, body, upvars, mir_def_id, fr, counter).unwrap();
        fr_name.highlight_region_name(&mut diag);
        let outlived_fr_name =
            self.give_region_a_name(infcx, body, upvars, mir_def_id, outlived_fr, counter).unwrap();
        outlived_fr_name.highlight_region_name(&mut diag);

        let mir_def_name = if infcx.tcx.is_closure(mir_def_id) {
            "closure"
        } else {
            "function"
        };

        match (category, outlived_fr_is_local, fr_is_local) {
            (ConstraintCategory::Return, true, _) => {
                diag.span_label(
                    span,
                    format!(
                        "{} was supposed to return data with lifetime `{}` but it is returning \
                         data with lifetime `{}`",
                        mir_def_name, outlived_fr_name, fr_name
                    ),
                );
            }
            _ => {
                diag.span_label(
                    span,
                    format!(
                        "{}requires that `{}` must outlive `{}`",
                        category.description(),
                        fr_name,
                        outlived_fr_name,
                    ),
                );
            }
        }

        self.add_static_impl_trait_suggestion(infcx, &mut diag, fr, fr_name, outlived_fr);

        diag.buffer(errors_buffer);
    }

    /// Adds a suggestion to errors where a `impl Trait` is returned.
    ///
    /// ```text
    /// help: to allow this `impl Trait` to capture borrowed data with lifetime `'1`, add `'_` as
    ///       a constraint
    ///    |
    /// LL |     fn iter_values_anon(&self) -> impl Iterator<Item=u32> + 'a {
    ///    |                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    /// ```
    fn add_static_impl_trait_suggestion(
        &self,
        infcx: &InferCtxt<'_, 'tcx>,
        diag: &mut DiagnosticBuilder<'_>,
        fr: RegionVid,
        // We need to pass `fr_name` - computing it again will label it twice.
        fr_name: RegionName,
        outlived_fr: RegionVid,
    ) {
        if let (Some(f), Some(ty::RegionKind::ReStatic)) =
            (self.to_error_region(fr), self.to_error_region(outlived_fr))
        {
            if let Some(ty::TyS {
                sty: ty::Opaque(did, substs),
                ..
            }) = infcx
                .tcx
                .is_suitable_region(f)
                .map(|r| r.def_id)
                .map(|id| infcx.tcx.return_type_impl_trait(id))
                .unwrap_or(None)
            {
                // Check whether or not the impl trait return type is intended to capture
                // data with the static lifetime.
                //
                // eg. check for `impl Trait + 'static` instead of `impl Trait`.
                let has_static_predicate = {
                    let predicates_of = infcx.tcx.predicates_of(*did);
                    let bounds = predicates_of.instantiate(infcx.tcx, substs);

                    let mut found = false;
                    for predicate in bounds.predicates {
                        if let ty::Predicate::TypeOutlives(binder) = predicate {
                            if let ty::OutlivesPredicate(_, ty::RegionKind::ReStatic) =
                                binder.skip_binder()
                            {
                                found = true;
                                break;
                            }
                        }
                    }

                    found
                };

                debug!(
                    "add_static_impl_trait_suggestion: has_static_predicate={:?}",
                    has_static_predicate
                );
                let static_str = kw::StaticLifetime;
                // If there is a static predicate, then the only sensible suggestion is to replace
                // fr with `'static`.
                if has_static_predicate {
                    diag.help(&format!(
                        "consider replacing `{}` with `{}`",
                        fr_name, static_str,
                    ));
                } else {
                    // Otherwise, we should suggest adding a constraint on the return type.
                    let span = infcx.tcx.def_span(*did);
                    if let Ok(snippet) = infcx.tcx.sess.source_map().span_to_snippet(span) {
                        let suggestable_fr_name = if fr_name.was_named() {
                            fr_name.to_string()
                        } else {
                            "'_".to_string()
                        };

                        diag.span_suggestion(
                            span,
                            &format!(
                                "to allow this `impl Trait` to capture borrowed data with lifetime \
                                 `{}`, add `{}` as a constraint",
                                fr_name, suggestable_fr_name,
                            ),
                            format!("{} + {}", snippet, suggestable_fr_name),
                            Applicability::MachineApplicable,
                        );
                    }
                }
            }
        }
    }

    crate fn free_region_constraint_info(
        &self,
        body: &Body<'tcx>,
        upvars: &[Upvar],
        mir_def_id: DefId,
        infcx: &InferCtxt<'_, 'tcx>,
        borrow_region: RegionVid,
        outlived_region: RegionVid,
    ) -> (ConstraintCategory, bool, Span, Option<RegionName>) {
        let (category, from_closure, span) = self.best_blame_constraint(
            body,
            borrow_region,
            |r| self.provides_universal_region(r, borrow_region, outlived_region)
        );
        let outlived_fr_name =
            self.give_region_a_name(infcx, body, upvars, mir_def_id, outlived_region, &mut 1);
        (category, from_closure, span, outlived_fr_name)
    }

    // Finds some region R such that `fr1: R` and `R` is live at
    // `elem`.
    crate fn find_sub_region_live_at(
        &self,
        fr1: RegionVid,
        elem: Location,
    ) -> RegionVid {
        debug!("find_sub_region_live_at(fr1={:?}, elem={:?})", fr1, elem);
        self.find_constraint_paths_between_regions(fr1, |r| {
            // First look for some `r` such that `fr1: r` and `r` is live at `elem`
            debug!(
                "find_sub_region_live_at: liveness_constraints for {:?} are {:?}",
                r,
                self.liveness_constraints.region_value_str(r),
            );
            self.liveness_constraints.contains(r, elem)
        }).or_else(|| {
                // If we fail to find that, we may find some `r` such that
                // `fr1: r` and `r` is a placeholder from some universe
                // `fr1` cannot name. This would force `fr1` to be
                // `'static`.
                self.find_constraint_paths_between_regions(fr1, |r| {
                    self.cannot_name_placeholder(fr1, r)
                })
            })
            .or_else(|| {
                // If we fail to find THAT, it may be that `fr1` is a
                // placeholder that cannot "fit" into its SCC. In that
                // case, there should be some `r` where `fr1: r`, both
                // `fr1` and `r` are in the same SCC, and `fr1` is a
                // placeholder that `r` cannot name. We can blame that
                // edge.
                self.find_constraint_paths_between_regions(fr1, |r| {
                    self.constraint_sccs.scc(fr1) == self.constraint_sccs.scc(r)
                        && self.cannot_name_placeholder(r, fr1)
                })
            })
            .map(|(_path, r)| r)
            .unwrap()
    }

    // Finds a good span to blame for the fact that `fr1` outlives `fr2`.
    crate fn find_outlives_blame_span(
        &self,
        body: &Body<'tcx>,
        fr1: RegionVid,
        fr2: RegionVid,
    ) -> (ConstraintCategory, Span) {
        let (category, _, span) = self.best_blame_constraint(
            body,
            fr1,
            |r| self.provides_universal_region(r, fr1, fr2),
        );
        (category, span)
    }

    fn retrieve_closure_constraint_info(
        &self,
        body: &Body<'tcx>,
        constraint: &OutlivesConstraint,
    ) -> (ConstraintCategory, bool, Span) {
        let loc = match constraint.locations {
            Locations::All(span) => return (constraint.category, false, span),
            Locations::Single(loc) => loc,
        };

        let opt_span_category =
            self.closure_bounds_mapping[&loc].get(&(constraint.sup, constraint.sub));
        opt_span_category
            .map(|&(category, span)| (category, true, span))
            .unwrap_or((constraint.category, false, body.source_info(loc).span))
    }

    /// Returns `true` if a closure is inferred to be an `FnMut` closure.
    crate fn is_closure_fn_mut(&self, infcx: &InferCtxt<'_, 'tcx>, fr: RegionVid) -> bool {
        if let Some(ty::ReFree(free_region)) = self.to_error_region(fr) {
            if let ty::BoundRegion::BrEnv = free_region.bound_region {
                if let DefiningTy::Closure(def_id, substs) = self.universal_regions.defining_ty {
                    let closure_kind_ty = substs.closure_kind_ty(def_id, infcx.tcx);
                    return Some(ty::ClosureKind::FnMut) == closure_kind_ty.to_opt_closure_kind();
                }
            }
        }

        false
    }

    /// If `r2` represents a placeholder region, then this returns
    /// `true` if `r1` cannot name that placeholder in its
    /// value; otherwise, returns `false`.
    fn cannot_name_placeholder(&self, r1: RegionVid, r2: RegionVid) -> bool {
        debug!("cannot_name_value_of(r1={:?}, r2={:?})", r1, r2);

        match self.definitions[r2].origin {
            NLLRegionVariableOrigin::Placeholder(placeholder) => {
                let universe1 = self.definitions[r1].universe;
                debug!(
                    "cannot_name_value_of: universe1={:?} placeholder={:?}",
                    universe1, placeholder
                );
                universe1.cannot_name(placeholder.universe)
            }

            NLLRegionVariableOrigin::FreeRegion | NLLRegionVariableOrigin::Existential => false,
        }
    }
}
