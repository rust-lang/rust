use rustc_data_structures::fx::{FxHashSet, FxIndexMap, IndexEntry};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_index::bit_set::BitSet;
use rustc_index::IndexVec;
use rustc_middle::mir::visit::{
    MutatingUseContext, NonMutatingUseContext, NonUseContext, PlaceContext, Visitor,
};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_mir_dataflow::{
    fmt::DebugWithContext, Analysis, AnalysisDomain, Backward, GenKill, GenKillAnalysis,
    ResultsCursor,
};
use rustc_session::lint;
use rustc_span::symbol::sym;
use rustc_span::Span;
use rustc_target::abi::FieldIdx;

use crate::errors;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum AccessKind {
    Param,
    Assign,
    Capture,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CaptureKind {
    Closure(ty::ClosureKind),
    Coroutine,
    CoroutineClosure,
    None,
}

struct AssignmentResult {
    /// Set of locals that are live at least once. This is used to report fully unused locals.
    ever_live: BitSet<PlaceIndex>,
    /// Set of locals that have a non-trivial drop. This is used to skip reporting unused
    /// assignment if it would be used by the `Drop` impl.
    ever_dropped: BitSet<PlaceIndex>,
    /// Set of assignments for each local. Here, assignment is understood in the AST sense. Any
    /// MIR that may look like an assignment (Assign, DropAndReplace, Yield, Call) are considered.
    ///
    /// For each local, we return a map: for each source position, whether the statement is live
    /// and which kind of access it performs. When we encounter multiple statements at the same
    /// location, we only increase the liveness, in order to avoid false positives.
    assignments: IndexVec<PlaceIndex, FxIndexMap<SourceInfo, (bool, AccessKind)>>,
}

#[tracing::instrument(level = "debug", skip(tcx))]
pub fn check_liveness<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> BitSet<FieldIdx> {
    // Don't run unused pass for #[naked]
    if tcx.has_attr(def_id.to_def_id(), sym::naked) {
        return BitSet::new_empty(0);
    }

    // Don't run unused pass for #[derive]
    let parent = tcx.parent(tcx.typeck_root_def_id(def_id.to_def_id()));
    if let DefKind::Impl { of_trait: true } = tcx.def_kind(parent)
        && tcx.has_attr(parent, sym::automatically_derived)
    {
        return BitSet::new_empty(0);
    }

    let mut body = &*tcx.mir_promoted(def_id).0.borrow();
    let mut body_mem;

    // Don't run if there are errors.
    if body.tainted_by_errors.is_some() {
        return BitSet::new_empty(0);
    }

    let mut checked_places = PlaceSet::default();
    checked_places.insert_locals(&body.local_decls);

    // The body is the one of a closure or generator, so we also want to analyse captures.
    let (capture_kind, num_captures) = if tcx.is_closure_like(def_id.to_def_id()) {
        let mut self_ty = body.local_decls[ty::CAPTURE_STRUCT_LOCAL].ty;
        let mut self_is_ref = false;
        if let ty::Ref(_, ty, _) = self_ty.kind() {
            self_ty = *ty;
            self_is_ref = true;
        }

        let (capture_kind, args) = match self_ty.kind() {
            ty::Closure(_, args) => {
                (CaptureKind::Closure(args.as_closure().kind()), ty::UpvarArgs::Closure(args))
            }
            &ty::Coroutine(_, args) => (CaptureKind::Coroutine, ty::UpvarArgs::Coroutine(args)),
            &ty::CoroutineClosure(_, args) => {
                (CaptureKind::CoroutineClosure, ty::UpvarArgs::CoroutineClosure(args))
            }
            _ => bug!("expected closure or generator, found {:?}", self_ty),
        };

        let captures = tcx.closure_captures(def_id);
        checked_places.insert_captures(tcx, self_is_ref, captures, args.upvar_tys());

        // `FnMut` closures can modify captured values and carry those
        // modified values with them in subsequent calls. To model this behaviour,
        // we consider the `FnMut` closure as jumping to `bb0` upon return.
        if let CaptureKind::Closure(ty::ClosureKind::FnMut) = capture_kind
            && checked_places.captures.iter().any(|(_, by_ref)| !by_ref)
        {
            // FIXME: stop cloning the body.
            body_mem = body.clone();
            for bbdata in body_mem.basic_blocks_mut() {
                if let TerminatorKind::Return = bbdata.terminator().kind {
                    bbdata.terminator_mut().kind = TerminatorKind::Goto { target: START_BLOCK };
                }
            }
            body = &body_mem;
        }

        (capture_kind, args.upvar_tys().len())
    } else {
        (CaptureKind::None, 0)
    };

    // Get the remaining variables' names from debuginfo.
    checked_places.record_debuginfo(&body.var_debug_info);

    let self_assignment = find_self_assignments(&checked_places, body);

    let mut live =
        MaybeLivePlaces { tcx, capture_kind, checked_places: &checked_places, self_assignment }
            .into_engine(tcx, body)
            .iterate_to_fixpoint()
            .into_results_cursor(body);

    let AssignmentResult { mut ever_live, ever_dropped, mut assignments } =
        find_dead_assignments(tcx, &checked_places, &mut live, body);

    // Match guards introduce a different local to freeze the guarded value as immutable.
    // Having two locals, we need to make sure that we do not report an unused_variable
    // when the guard local is used but not the arm local, or vice versa, like in this example.
    //
    //    match 5 {
    //      x if x > 2 => {}
    //      ^    ^- This is `local`
    //      +------ This is `arm_local`
    //      _ => {}
    //    }
    //
    for (index, place) in checked_places.iter() {
        let local = place.local;
        if let &LocalInfo::User(BindingForm::RefForGuard(arm_local)) =
            body.local_decls[local].local_info()
        {
            debug_assert!(place.projection.is_empty());

            // Local to use in the arm.
            let Some((arm_index, _proj)) = checked_places.get(arm_local.into()) else { continue };
            debug_assert_ne!(index, arm_index);
            debug_assert_eq!(_proj, &[]);

            // Mark the arm local as used if the guard local is used.
            if ever_live.contains(index) {
                ever_live.insert(arm_index);
            }

            // Some assignments are common to both locals in the source code.
            // Sadly, we can only detect this using the `source_info`.
            // Therefore, we loop over all the assignments we have for the guard local:
            // - if they already appeared for the arm local, the assignment is live if one of the
            //   two versions is live;
            // - if it does not appear for the arm local, it happened inside the guard, so we add
            //   it as-is.
            let guard_assignments = std::mem::take(&mut assignments[index]);
            let arm_assignments = &mut assignments[arm_index];
            for (source_info, (live, kind)) in guard_assignments {
                match arm_assignments.entry(source_info) {
                    IndexEntry::Vacant(v) => {
                        v.insert((live, kind));
                    }
                    IndexEntry::Occupied(mut o) => {
                        o.get_mut().0 |= live;
                    }
                }
            }
        }
    }

    // Report to caller the set of dead captures.
    let mut dead_captures = BitSet::new_empty(num_captures);

    // First, report fully unused locals.
    debug!("report fully unused places");
    for (index, place) in checked_places.iter() {
        if ever_live.contains(index) {
            continue;
        }

        // This is a capture: let the enclosing function report the unused variable.
        if is_capture(*place) {
            debug_assert_eq!(place.local, ty::CAPTURE_STRUCT_LOCAL);
            for p in place.projection {
                if let PlaceElem::Field(f, _) = p {
                    dead_captures.insert(*f);
                    break;
                }
            }
            continue;
        }

        let Some((ref name, def_span)) = checked_places.names[index] else { continue };
        if name.is_empty() || name.starts_with('_') || name == "self" {
            continue;
        }

        let local = place.local;
        let decl = &body.local_decls[local];

        if decl.from_compiler_desugaring() {
            continue;
        }

        // Only report actual user-defined binding from now on.
        let LocalInfo::User(BindingForm::Var(binding)) = decl.local_info() else { continue };
        let Some(hir_id) = decl.source_info.scope.lint_root(&body.source_scopes) else { continue };

        let introductions = &binding.introductions;

        // #117284, when `ident_span` and `def_span` have different contexts
        // we can't provide a good suggestion, instead we pointed out the spans from macro
        let from_macro = def_span.from_expansion()
            && introductions.iter().any(|(ident_span, _)| ident_span.eq_ctxt(def_span));

        let statements = &mut assignments[index];
        if statements.is_empty() {
            let sugg = if from_macro {
                errors::UnusedVariableSugg::NoSugg { span: def_span, name: name.clone() }
            } else {
                errors::UnusedVariableSugg::TryPrefix { spans: vec![def_span], name: name.clone() }
            };
            tcx.emit_node_span_lint(
                lint::builtin::UNUSED_VARIABLES,
                hir_id,
                def_span,
                errors::UnusedVariable {
                    name: name.clone(),
                    string_interp: maybe_suggest_literal_matching_name(body, name),
                    sugg,
                },
            );
            continue;
        }

        // Idiomatic rust assigns a value to a local upon definition. However, we do not want to
        // warn twice, for the unused local and for the unused assignment. Therefore, we remove
        // from the list of assignments the ones that happen at the definition site.
        statements.retain(|source_info, _| {
            source_info.span.find_ancestor_inside(binding.pat_span).is_none()
        });

        // Extra assignments that we recognize thanks to the initialization span. We need to
        // take care of macro contexts here to be accurate.
        if let Some((_, initializer_span)) = binding.opt_match_place {
            statements.retain(|source_info, _| {
                let within = source_info.span.find_ancestor_inside(initializer_span);
                let outer_initializer_span =
                    initializer_span.find_ancestor_in_same_ctxt(source_info.span);
                within.is_none()
                    && outer_initializer_span.map_or(true, |s| !s.contains(source_info.span))
            });
        }

        if !statements.is_empty() {
            // We have a dead local with outstanding assignments and with non-trivial drop.
            // This is probably a drop-guard, so we do not issue a warning there.
            if ever_dropped.contains(index) {
                continue;
            }

            tcx.emit_node_span_lint(
                lint::builtin::UNUSED_VARIABLES,
                hir_id,
                def_span,
                errors::UnusedVarAssignedOnly { name: name.clone() },
            );
            continue;
        }

        // We do not have outstanding assignments, suggest renaming the binding.
        let spans = introductions.iter().map(|(span, _)| *span).collect::<Vec<_>>();

        let any_shorthand = introductions.iter().any(|(_, is_shorthand)| *is_shorthand);

        let sugg = if any_shorthand {
            errors::UnusedVariableSugg::TryIgnore {
                name: name.clone(),
                shorthands: introductions
                    .iter()
                    .filter_map(
                        |&(span, is_shorthand)| {
                            if is_shorthand { Some(span) } else { None }
                        },
                    )
                    .collect(),
                non_shorthands: introductions
                    .iter()
                    .filter_map(
                        |&(span, is_shorthand)| {
                            if !is_shorthand { Some(span) } else { None }
                        },
                    )
                    .collect(),
            }
        } else if from_macro {
            errors::UnusedVariableSugg::NoSugg { span: def_span, name: name.clone() }
        } else if !introductions.is_empty() {
            errors::UnusedVariableSugg::TryPrefix {
                name: name.clone(),
                spans: introductions.iter().map(|&(span, _)| span).collect(),
            }
        } else {
            errors::UnusedVariableSugg::TryPrefix { name: name.clone(), spans: vec![def_span] }
        };

        tcx.emit_node_span_lint(
            lint::builtin::UNUSED_VARIABLES,
            hir_id,
            spans,
            errors::UnusedVariable {
                name: name.clone(),
                string_interp: maybe_suggest_literal_matching_name(body, name),
                sugg,
            },
        );
    }

    // Second, report unused assignments that do not correspond to initialization.
    // Initializations have been removed in the previous loop reporting unused variables.
    debug!("report dead assignments");
    for (index, statements) in assignments.into_iter_enumerated() {
        if statements.is_empty() {
            continue;
        }

        let Some((ref name, decl_span)) = checked_places.names[index] else { continue };
        if name.is_empty() || name.starts_with('_') || name == "self" {
            continue;
        }

        // We have outstanding assignments and with non-trivial drop.
        // This is probably a drop-guard, so we do not issue a warning there.
        if ever_dropped.contains(index) {
            continue;
        }

        // We probed MIR in reverse order for dataflow.
        // We revert the vector to give a consistent order to the user.
        for (source_info, (live, kind)) in statements.into_iter().rev() {
            if live {
                continue;
            }

            // Report the dead assignment.
            let Some(hir_id) = source_info.scope.lint_root(&body.source_scopes) else { continue };

            match kind {
                AccessKind::Assign => tcx.emit_node_span_lint(
                    lint::builtin::UNUSED_ASSIGNMENTS,
                    hir_id,
                    source_info.span,
                    errors::UnusedAssign { name: name.clone() },
                ),
                AccessKind::Param => tcx.emit_node_span_lint(
                    lint::builtin::UNUSED_ASSIGNMENTS,
                    hir_id,
                    source_info.span,
                    errors::UnusedAssignPassed { name: name.clone() },
                ),
                AccessKind::Capture => tcx.emit_node_span_lint(
                    lint::builtin::UNUSED_ASSIGNMENTS,
                    hir_id,
                    decl_span,
                    errors::UnusedCaptureMaybeCaptureRef { name: name.clone() },
                ),
            }
        }
    }

    dead_captures
}

/// Small helper to make semantics easier to read.
#[inline]
fn is_capture(place: PlaceRef<'_>) -> bool {
    if !place.projection.is_empty() {
        debug_assert_eq!(place.local, ty::CAPTURE_STRUCT_LOCAL);
        true
    } else {
        false
    }
}

/// Give a diagnostic when any of the string constants look like a naked format string that would
/// interpolate our dead local.
fn maybe_suggest_literal_matching_name(
    body: &Body<'_>,
    name: &str,
) -> Vec<errors::UnusedVariableStringInterp> {
    struct LiteralFinder<'body, 'tcx> {
        body: &'body Body<'tcx>,
        name: String,
        name_colon: String,
        found: Vec<errors::UnusedVariableStringInterp>,
    }

    impl<'tcx> Visitor<'tcx> for LiteralFinder<'_, 'tcx> {
        fn visit_constant(&mut self, constant: &ConstOperand<'tcx>, loc: Location) {
            if let ty::Ref(_, ref_ty, _) = constant.ty().kind()
                && ref_ty.kind() == &ty::Str
            {
                let rendered_constant = constant.const_.to_string();
                if rendered_constant.contains(&self.name)
                    || rendered_constant.contains(&self.name_colon)
                {
                    let lit = self.body.source_info(loc).span;
                    self.found.push(errors::UnusedVariableStringInterp { lit });
                }
            }
        }
    }

    let mut finder = LiteralFinder {
        body,
        name: format!("{{{name}}}"),
        name_colon: format!("{{{name}:"),
        found: vec![],
    };
    finder.visit_body(body);
    finder.found
}

/// Compute self-assignments of the form `a += b`.
///
/// MIR building generates 2 statements and 1 terminator for such assignments:
/// - _temp = CheckedBinaryOp(a, b)
/// - assert(!_temp.1)
/// - a = _temp.0
///
/// This function tries to detect this pattern in order to avoid marking statement as a definition
/// and use. This will let the analysis be dictated by the next use of `a`.
///
/// Note that we will still need to account for the use of `b`.
fn find_self_assignments<'tcx>(
    checked_places: &PlaceSet<'tcx>,
    body: &Body<'tcx>,
) -> FxHashSet<Location> {
    let mut self_assign = FxHashSet::default();

    const FIELD_0: FieldIdx = FieldIdx::from_u32(0);
    const FIELD_1: FieldIdx = FieldIdx::from_u32(1);

    for (bb, bb_data) in body.basic_blocks.iter_enumerated() {
        for (statement_index, stmt) in bb_data.statements.iter().enumerate() {
            let StatementKind::Assign(box (first_place, rvalue)) = &stmt.kind else { continue };
            match rvalue {
                // Straight self-assignment.
                Rvalue::BinaryOp(op, box (Operand::Copy(lhs), _)) => {
                    if lhs != first_place {
                        continue;
                    }

                    // We ignore indirect self-assignment, because both occurences of `dest` are uses.
                    let is_indirect = checked_places
                        .get(first_place.as_ref())
                        .map_or(false, |(_, projections)| is_indirect(projections));
                    if is_indirect {
                        continue;
                    }

                    self_assign.insert(Location { block: bb, statement_index });

                    // Checked division verifies overflow before performing the division, so we
                    // need to go and ignore this check in the predecessor block.
                    if let BinOp::Div | BinOp::Rem = op
                        && statement_index == 0
                        && let &[pred] = body.basic_blocks.predecessors()[bb].as_slice()
                        && let TerminatorKind::Assert { msg, .. } =
                            &body.basic_blocks[pred].terminator().kind
                        && let AssertKind::Overflow(..) = **msg
                        && let len = body.basic_blocks[pred].statements.len()
                        && len >= 2
                    {
                        // BitAnd of two checks.
                        self_assign.insert(Location { block: pred, statement_index: len - 1 });
                        // `lhs == MIN`.
                        self_assign.insert(Location { block: pred, statement_index: len - 2 });
                    }
                }
                // For checked binary ops, the MIR builder inserts an assertion in between.
                Rvalue::CheckedBinaryOp(_, box (Operand::Copy(lhs), _)) => {
                    // Checked binary ops only appear at the end of the block, before the assertion.
                    if statement_index + 1 != bb_data.statements.len() {
                        continue;
                    }

                    let TerminatorKind::Assert {
                        cond,
                        target,
                        msg: box AssertKind::Overflow(..),
                        ..
                    } = &bb_data.terminator().kind
                    else {
                        continue;
                    };
                    let Some(assign) = body.basic_blocks[*target].statements.first() else {
                        continue;
                    };
                    let StatementKind::Assign(box (dest, Rvalue::Use(Operand::Move(temp)))) =
                        assign.kind
                    else {
                        continue;
                    };

                    if dest != *lhs {
                        continue;
                    }

                    let Operand::Move(cond) = cond else { continue };
                    let [PlaceElem::Field(FIELD_0, _)] = &temp.projection.as_slice() else {
                        continue;
                    };
                    let [PlaceElem::Field(FIELD_1, _)] = &cond.projection.as_slice() else {
                        continue;
                    };

                    // We ignore indirect self-assignment, because both occurences of `dest` are uses.
                    let is_indirect = checked_places
                        .get(dest.as_ref())
                        .map_or(false, |(_, projections)| is_indirect(projections));
                    if is_indirect {
                        continue;
                    }

                    if first_place.local == temp.local
                        && first_place.local == cond.local
                        && first_place.projection.is_empty()
                    {
                        // Original block
                        self_assign.insert(Location {
                            block: bb,
                            statement_index: bb_data.statements.len() - 1,
                        });
                        self_assign.insert(Location {
                            block: bb,
                            statement_index: bb_data.statements.len(),
                        });
                        // Target block
                        self_assign.insert(Location { block: *target, statement_index: 0 });
                    }
                }
                _ => {}
            }
        }
    }

    self_assign
}

#[derive(Default, Debug)]
struct PlaceSet<'tcx> {
    places: IndexVec<PlaceIndex, PlaceRef<'tcx>>,
    names: IndexVec<PlaceIndex, Option<(String, Span)>>,

    /// Places corresponding to locals, common case.
    locals: IndexVec<Local, Option<PlaceIndex>>,

    // Handling of captures.
    /// If `_1` is a reference, we need to add a `Deref` to the matched place.
    capture_field_pos: usize,
    /// Captured fields.
    captures: IndexVec<FieldIdx, (PlaceIndex, bool)>,
}

impl<'tcx> PlaceSet<'tcx> {
    fn insert_locals(&mut self, decls: &IndexVec<Local, LocalDecl<'tcx>>) {
        self.locals = IndexVec::from_elem(None, &decls);
        for (local, decl) in decls.iter_enumerated() {
            // Record all user-written locals for the analysis.
            // We also keep the `RefForGuard` locals (more on that below).
            if let LocalInfo::User(BindingForm::Var(_) | BindingForm::RefForGuard(_)) =
                decl.local_info()
            {
                let index = self.places.push(local.into());
                self.locals[local] = Some(index);
                let _index = self.names.push(None);
                debug_assert_eq!(index, _index);
            }
        }
    }

    fn insert_captures(
        &mut self,
        tcx: TyCtxt<'tcx>,
        self_is_ref: bool,
        captures: &[&'tcx ty::CapturedPlace<'tcx>],
        upvars: &ty::List<Ty<'tcx>>,
    ) {
        // We should not track the environment local separately.
        debug_assert_eq!(self.locals[ty::CAPTURE_STRUCT_LOCAL], None);

        let self_place = Place {
            local: ty::CAPTURE_STRUCT_LOCAL,
            projection: tcx.mk_place_elems(if self_is_ref { &[PlaceElem::Deref] } else { &[] }),
        };
        if self_is_ref {
            self.capture_field_pos = 1;
        }

        for (f, (capture, ty)) in std::iter::zip(captures, upvars).enumerate() {
            let f = FieldIdx::from_usize(f);
            let elem = PlaceElem::Field(f, ty);
            let by_ref = matches!(capture.info.capture_kind, ty::UpvarCapture::ByRef(..));
            let place = if by_ref {
                self_place.project_deeper(&[elem, PlaceElem::Deref], tcx)
            } else {
                self_place.project_deeper(&[elem], tcx)
            };
            let index = self.places.push(place.as_ref());
            let _f = self.captures.push((index, by_ref));
            debug_assert_eq!(_f, f);

            // Record a variable name from the capture, because it is much friendlier than the
            // debuginfo name.
            self.names.insert(index, (capture.to_string(tcx), capture.get_path_span(tcx)));
        }
    }

    fn record_debuginfo(&mut self, var_debug_info: &Vec<VarDebugInfo<'tcx>>) {
        for var_debug_info in var_debug_info {
            if let VarDebugInfoContents::Place(place) = var_debug_info.value
                && let Some(index) = self.locals[place.local]
            {
                self.names.get_or_insert_with(index, || {
                    (var_debug_info.name.to_string(), var_debug_info.source_info.span)
                });
            }
        }
    }

    fn get(&self, place: PlaceRef<'tcx>) -> Option<(PlaceIndex, &'tcx [PlaceElem<'tcx>])> {
        if let Some(index) = self.locals[place.local] {
            return Some((index, place.projection));
        }
        if place.local == ty::CAPTURE_STRUCT_LOCAL
            && !self.captures.is_empty()
            && self.capture_field_pos < place.projection.len()
            && let PlaceElem::Field(f, _) = place.projection[self.capture_field_pos]
            && let Some((index, by_ref)) = self.captures.get(f)
        {
            let mut start = self.capture_field_pos + 1;
            if *by_ref {
                // Account for an extra Deref.
                start += 1;
            }
            // We may have an attempt to access `_1.f` as a shallow reborrow. Just ignore it.
            if start <= place.projection.len() {
                let projection = &place.projection[start..];
                return Some((*index, projection));
            }
        }
        None
    }

    fn iter(&self) -> impl Iterator<Item = (PlaceIndex, &PlaceRef<'tcx>)> {
        self.places.iter_enumerated()
    }

    fn len(&self) -> usize {
        self.places.len()
    }
}

/// Collect all assignments to checked locals.
///
/// Assignments are collected, even if they are live. Dead assignments are reported, and live
/// assignments are used to make diagnostics correct for match guards.
fn find_dead_assignments<'tcx>(
    tcx: TyCtxt<'tcx>,
    checked_places: &PlaceSet<'tcx>,
    cursor: &mut ResultsCursor<'_, 'tcx, MaybeLivePlaces<'_, 'tcx>>,
    body: &Body<'tcx>,
) -> AssignmentResult {
    let mut ever_live = BitSet::new_empty(checked_places.len());
    let mut ever_dropped = BitSet::new_empty(checked_places.len());
    let mut assignments = IndexVec::<PlaceIndex, FxIndexMap<_, _>>::from_elem(
        Default::default(),
        &checked_places.places,
    );

    let mut check_place =
        |place: Place<'tcx>, kind, source_info: SourceInfo, live: &BitSet<PlaceIndex>| {
            if let Some((index, extra_projections)) = checked_places.get(place.as_ref()) {
                if !is_indirect(extra_projections) {
                    match assignments[index].entry(source_info) {
                        IndexEntry::Vacant(v) => {
                            v.insert((live.contains(index), kind));
                        }
                        IndexEntry::Occupied(mut o) => {
                            // There were already a sighting. Mark this statement as live if it was,
                            // to avoid false positives.
                            o.get_mut().0 |= live.contains(index);
                        }
                    }
                }
            }
        };

    let param_env = tcx.param_env(body.source.def_id());
    let mut record_drop = |place: Place<'tcx>| {
        if let Some((index, &[])) = checked_places.get(place.as_ref()) {
            let ty = place.ty(&body.local_decls, tcx).ty;
            let needs_drop = matches!(
                ty.kind(),
                ty::Closure(..)
                    | ty::Coroutine(..)
                    | ty::Tuple(..)
                    | ty::Adt(..)
                    | ty::Dynamic(..)
                    | ty::Array(..)
                    | ty::Slice(..)
                    | ty::Alias(ty::Opaque, ..)
            ) && ty.needs_drop(tcx, param_env);
            if needs_drop {
                ever_dropped.insert(index);
            }
        }
    };

    for (bb, bb_data) in traversal::postorder(body) {
        cursor.seek_to_block_end(bb);
        let live = cursor.get();
        ever_live.union(live);

        let terminator = bb_data.terminator();
        match &terminator.kind {
            TerminatorKind::Call { destination: place, .. }
            | TerminatorKind::Yield { resume_arg: place, .. } => {
                check_place(*place, AccessKind::Assign, terminator.source_info, live);
                record_drop(*place)
            }
            TerminatorKind::Drop { place, .. } => record_drop(*place),
            TerminatorKind::InlineAsm { operands, .. } => {
                for operand in operands {
                    if let InlineAsmOperand::Out { place: Some(place), .. }
                    | InlineAsmOperand::InOut { out_place: Some(place), .. } = operand
                    {
                        check_place(*place, AccessKind::Assign, terminator.source_info, live);
                    }
                }
            }
            _ => {}
        }

        for (statement_index, statement) in bb_data.statements.iter().enumerate().rev() {
            cursor.seek_before_primary_effect(Location { block: bb, statement_index });
            let live = cursor.get();
            ever_live.union(live);
            match &statement.kind {
                StatementKind::Assign(box (place, _))
                | StatementKind::Deinit(box place)
                | StatementKind::SetDiscriminant { box place, .. } => {
                    check_place(*place, AccessKind::Assign, statement.source_info, live);
                }
                StatementKind::Retag(_, _)
                | StatementKind::StorageLive(_)
                | StatementKind::StorageDead(_)
                | StatementKind::Coverage(_)
                | StatementKind::Intrinsic(_)
                | StatementKind::Nop
                | StatementKind::FakeRead(_)
                | StatementKind::PlaceMention(_)
                | StatementKind::ConstEvalCounter
                | StatementKind::AscribeUserType(_, _) => (),
            }
        }
    }

    // Check liveness of function arguments on entry.
    {
        cursor.seek_to_block_start(START_BLOCK);
        let live = cursor.get();
        ever_live.union(live);

        // Verify that arguments and captured values are useful.
        for (index, place) in checked_places.iter() {
            let kind = if is_capture(*place) {
                // This is a by-ref capture, an assignment to it will modify surrounding
                // environment, so we do not report it.
                if place.projection.last() == Some(&PlaceElem::Deref) {
                    continue;
                }

                AccessKind::Capture
            } else if body.local_kind(place.local) == LocalKind::Arg {
                AccessKind::Param
            } else {
                continue;
            };
            let source_info = body.local_decls[place.local].source_info;
            assignments[index].insert(source_info, (live.contains(index), kind));
        }
    }

    AssignmentResult { ever_live, ever_dropped, assignments }
}

rustc_index::newtype_index! {
    pub struct PlaceIndex {}
}

impl DebugWithContext<MaybeLivePlaces<'_, '_>> for PlaceIndex {
    fn fmt_with(
        &self,
        ctxt: &MaybeLivePlaces<'_, '_>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        std::fmt::Debug::fmt(&ctxt.checked_places.places[*self], f)
    }
}

pub struct MaybeLivePlaces<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    checked_places: &'a PlaceSet<'tcx>,
    capture_kind: CaptureKind,
    self_assignment: FxHashSet<Location>,
}

impl<'tcx> MaybeLivePlaces<'_, 'tcx> {
    fn transfer_function<'a, T>(&'a self, trans: &'a mut T) -> TransferFunction<'a, 'tcx, T> {
        TransferFunction {
            tcx: self.tcx,
            checked_places: &self.checked_places,
            capture_kind: self.capture_kind,
            trans,
            self_assignment: &self.self_assignment,
        }
    }
}

impl<'tcx> AnalysisDomain<'tcx> for MaybeLivePlaces<'_, 'tcx> {
    type Domain = BitSet<PlaceIndex>;
    type Direction = Backward;

    const NAME: &'static str = "liveness-lint";

    fn bottom_value(&self, _: &Body<'tcx>) -> Self::Domain {
        // bottom = not live
        BitSet::new_empty(self.checked_places.len())
    }

    fn initialize_start_block(&self, _: &Body<'tcx>, _: &mut Self::Domain) {
        // No variables are live until we observe a use
    }
}

impl<'tcx> GenKillAnalysis<'tcx> for MaybeLivePlaces<'_, 'tcx> {
    type Idx = PlaceIndex;

    fn domain_size(&self, _: &Body<'tcx>) -> usize {
        self.checked_places.len()
    }

    fn statement_effect(
        &mut self,
        trans: &mut impl GenKill<Self::Idx>,
        statement: &Statement<'tcx>,
        location: Location,
    ) {
        self.transfer_function(trans).visit_statement(statement, location);
    }

    fn terminator_effect<'mir>(
        &mut self,
        trans: &mut Self::Domain,
        terminator: &'mir Terminator<'tcx>,
        location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        self.transfer_function(trans).visit_terminator(terminator, location);
        terminator.edges()
    }

    fn call_return_effect(
        &mut self,
        _trans: &mut Self::Domain,
        _block: BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        // FIXME: what should happen here?
    }
}

struct TransferFunction<'a, 'tcx, T> {
    tcx: TyCtxt<'tcx>,
    checked_places: &'a PlaceSet<'tcx>,
    trans: &'a mut T,
    capture_kind: CaptureKind,
    self_assignment: &'a FxHashSet<Location>,
}

impl<'tcx, T> Visitor<'tcx> for TransferFunction<'_, 'tcx, T>
where
    T: GenKill<PlaceIndex>,
{
    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match statement.kind {
            // `ForLet` fake read erroneously marks the just-assigned local as live.
            // This defeats the purpose of the analysis for `let` bindings.
            StatementKind::FakeRead(box (FakeReadCause::ForLet(..), _)) => return,
            // Handle self-assignment by restricting the read/write they do.
            StatementKind::Assign(box (ref dest, ref rvalue))
                if self.self_assignment.contains(&location) =>
            {
                if let Rvalue::CheckedBinaryOp(_, box (_, ref rhs)) = rvalue {
                    // We are computing the binary operation:
                    // - the LHS will be assigned, so we don't read it;
                    // - the RHS still needs to be read.
                    self.visit_operand(rhs, location);
                    self.visit_place(
                        dest,
                        PlaceContext::MutatingUse(MutatingUseContext::Store),
                        location,
                    );
                } else if let Rvalue::BinaryOp(_, box (_, ref rhs)) = rvalue {
                    // We are computing the binary operation:
                    // - the LHS is being updated, so we don't read it;
                    // - the RHS still needs to be read.
                    self.visit_operand(rhs, location);
                } else {
                    // This is the second part of a checked self-assignment,
                    // we are assigning the result.
                    // We do not consider the write to the destination as a `def`.
                    // `self_assignment` must be false if the assignment is indirect.
                    self.visit_rvalue(rvalue, location);
                }
            }
            _ => self.super_statement(statement, location),
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        // By-ref captures could be read by the surrounding environement, so we mark
        // them as live upon yield and return.
        match terminator.kind {
            TerminatorKind::Return
            | TerminatorKind::Yield { .. }
            | TerminatorKind::Goto { target: START_BLOCK } // Inserted for the `FnMut` case.
                if self.capture_kind != CaptureKind::None =>
            {
                // All indirect captures have an effect on the environment, so we mark them as live.
                for (index, place) in self.checked_places.iter() {
                    if place.local == ty::CAPTURE_STRUCT_LOCAL
                        && place.projection.last() == Some(&PlaceElem::Deref)
                    {
                        self.trans.gen(index);
                    }
                }
            }
            // Do not consider a drop to be a use. We whitelist interesting drops elsewhere.
            TerminatorKind::Drop { .. } => {}
            // Ignore assertions since they must be triggered by actual code.
            TerminatorKind::Assert { .. } => {}
            _ => self.super_terminator(terminator, location),
        }
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        match rvalue {
            // When a closure/generator does not use some of its captures, do not consider these
            // captures as live in the surrounding function. This allows to report unused variables,
            // even if they have been (uselessly) captured.
            Rvalue::Aggregate(
                box AggregateKind::Closure(def_id, _) | box AggregateKind::Coroutine(def_id, _),
                ref operands,
            ) => {
                if let Some(def_id) = def_id.as_local() {
                    let dead_captures = self.tcx.check_liveness(def_id);
                    for (field, operand) in
                        operands.iter_enumerated().take(dead_captures.domain_size())
                    {
                        if !dead_captures.contains(field) {
                            self.visit_operand(operand, location);
                        }
                    }
                }
            }
            _ => self.super_rvalue(rvalue, location),
        }
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        if let Some((index, extra_projections)) = self.checked_places.get(place.as_ref()) {
            for i in (extra_projections.len()..=place.projection.len()).rev() {
                let place_part =
                    PlaceRef { local: place.local, projection: &place.projection[..i] };
                let extra_projections = &place.projection[i..];

                if let Some(&elem) = extra_projections.get(0) {
                    self.visit_projection_elem(place_part, elem, context, location);
                }
            }

            match DefUse::for_place(extra_projections, context) {
                Some(DefUse::Def) => self.trans.kill(index),
                Some(DefUse::Use) => self.trans.gen(index),
                None => {}
            }
        } else {
            self.super_place(place, context, location)
        }
    }

    fn visit_local(&mut self, local: Local, context: PlaceContext, _: Location) {
        if let Some((index, _proj)) = self.checked_places.get(local.into()) {
            debug_assert_eq!(_proj, &[]);
            match DefUse::for_place(&[], context) {
                Some(DefUse::Def) => self.trans.kill(index),
                Some(DefUse::Use) => self.trans.gen(index),
                _ => {}
            }
        }
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
enum DefUse {
    Def,
    Use,
}

fn is_indirect(proj: &[PlaceElem<'_>]) -> bool {
    proj.iter().any(|p| p.is_indirect())
}

impl DefUse {
    fn for_place<'tcx>(projection: &[PlaceElem<'tcx>], context: PlaceContext) -> Option<DefUse> {
        let is_indirect = is_indirect(projection);
        match context {
            PlaceContext::MutatingUse(
                MutatingUseContext::Store
                | MutatingUseContext::Deinit
                | MutatingUseContext::SetDiscriminant,
            ) => {
                if is_indirect {
                    // Treat derefs as a use of the base local. `*p = 4` is not a def of `p` but a
                    // use.
                    Some(DefUse::Use)
                } else if projection.is_empty() {
                    Some(DefUse::Def)
                } else {
                    None
                }
            }

            // For the associated terminators, this is only a `Def` when the terminator returns
            // "successfully." As such, we handle this case separately in `call_return_effect`
            // above. However, if the place looks like `*_5`, this is still unconditionally a use of
            // `_5`.
            PlaceContext::MutatingUse(
                MutatingUseContext::Call
                | MutatingUseContext::Yield
                | MutatingUseContext::AsmOutput,
            ) => is_indirect.then_some(DefUse::Use),

            // All other contexts are uses...
            PlaceContext::MutatingUse(
                MutatingUseContext::AddressOf
                | MutatingUseContext::Borrow
                | MutatingUseContext::Drop
                | MutatingUseContext::Retag,
            )
            | PlaceContext::NonMutatingUse(
                NonMutatingUseContext::AddressOf
                | NonMutatingUseContext::Copy
                | NonMutatingUseContext::Inspect
                | NonMutatingUseContext::Move
                | NonMutatingUseContext::FakeBorrow
                | NonMutatingUseContext::SharedBorrow
                | NonMutatingUseContext::PlaceMention,
            ) => Some(DefUse::Use),

            PlaceContext::NonUse(
                NonUseContext::StorageLive
                | NonUseContext::StorageDead
                | NonUseContext::AscribeUserTy(_)
                | NonUseContext::VarDebugInfo,
            ) => None,

            PlaceContext::MutatingUse(MutatingUseContext::Projection)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Projection) => {
                unreachable!("A projection could be a def or a use and must be handled separately")
            }
        }
    }
}
