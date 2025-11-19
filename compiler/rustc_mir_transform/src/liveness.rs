use rustc_abi::FieldIdx;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, IndexEntry};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def::{CtorKind, DefKind};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::find_attr;
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::bug;
use rustc_middle::mir::visit::{
    MutatingUseContext, NonMutatingUseContext, NonUseContext, PlaceContext, Visitor,
};
use rustc_middle::mir::*;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_mir_dataflow::fmt::DebugWithContext;
use rustc_mir_dataflow::{Analysis, Backward, ResultsCursor};
use rustc_session::lint;
use rustc_span::Span;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::symbol::{Symbol, kw, sym};

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

#[derive(Copy, Clone, Debug)]
struct Access {
    /// Describe the current access.
    kind: AccessKind,
    /// Is the accessed place is live at the current statement?
    /// When we encounter multiple statements at the same location, we only increase the liveness,
    /// in order to avoid false positives.
    live: bool,
}

#[tracing::instrument(level = "debug", skip(tcx), ret)]
pub(crate) fn check_liveness<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> DenseBitSet<FieldIdx> {
    // Don't run on synthetic MIR, as that will ICE trying to access HIR.
    if tcx.is_synthetic_mir(def_id) {
        return DenseBitSet::new_empty(0);
    }

    // Don't run unused pass for intrinsics
    if tcx.intrinsic(def_id.to_def_id()).is_some() {
        return DenseBitSet::new_empty(0);
    }

    // Don't run unused pass for #[naked]
    if find_attr!(tcx.get_all_attrs(def_id.to_def_id()), AttributeKind::Naked(..)) {
        return DenseBitSet::new_empty(0);
    }

    // Don't run unused pass for #[derive]
    let parent = tcx.parent(tcx.typeck_root_def_id(def_id.to_def_id()));
    if let DefKind::Impl { of_trait: true } = tcx.def_kind(parent)
        && find_attr!(tcx.get_all_attrs(parent), AttributeKind::AutomaticallyDerived(..))
    {
        return DenseBitSet::new_empty(0);
    }

    let mut body = &*tcx.mir_promoted(def_id).0.borrow();
    let mut body_mem;

    // Don't run if there are errors.
    if body.tainted_by_errors.is_some() {
        return DenseBitSet::new_empty(0);
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
        if let CaptureKind::Closure(ty::ClosureKind::FnMut) = capture_kind {
            // FIXME: stop cloning the body.
            body_mem = body.clone();
            for bbdata in body_mem.basic_blocks_mut() {
                // We can call a closure again, either after a normal return or an unwind.
                if let TerminatorKind::Return | TerminatorKind::UnwindResume =
                    bbdata.terminator().kind
                {
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
            .iterate_to_fixpoint(tcx, body, None)
            .into_results_cursor(body);

    let typing_env = ty::TypingEnv::post_analysis(tcx, body.source.def_id());

    let mut assignments =
        AssignmentResult::find_dead_assignments(tcx, typing_env, &checked_places, &mut live, body);

    assignments.merge_guards();

    let dead_captures = assignments.compute_dead_captures(num_captures);

    assignments.report_fully_unused();
    assignments.report_unused_assignments();

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

/// Give a diagnostic when an unused variable may be a typo of a unit variant or a struct.
fn maybe_suggest_unit_pattern_typo<'tcx>(
    tcx: TyCtxt<'tcx>,
    body_def_id: DefId,
    name: Symbol,
    span: Span,
    ty: Ty<'tcx>,
) -> Option<errors::PatternTypo> {
    if let ty::Adt(adt_def, _) = ty.peel_refs().kind() {
        let variant_names: Vec<_> = adt_def
            .variants()
            .iter()
            .filter(|v| matches!(v.ctor, Some((CtorKind::Const, _))))
            .map(|v| v.name)
            .collect();
        if let Some(name) = find_best_match_for_name(&variant_names, name, None)
            && let Some(variant) = adt_def
                .variants()
                .iter()
                .find(|v| v.name == name && matches!(v.ctor, Some((CtorKind::Const, _))))
        {
            return Some(errors::PatternTypo {
                span,
                code: with_no_trimmed_paths!(tcx.def_path_str(variant.def_id)),
                kind: tcx.def_descr(variant.def_id),
                item_name: variant.name,
            });
        }
    }

    // Look for consts of the same type with similar names as well,
    // not just unit structs and variants.
    let constants = tcx
        .hir_body_owners()
        .filter(|&def_id| {
            matches!(tcx.def_kind(def_id), DefKind::Const)
                && tcx.type_of(def_id).instantiate_identity() == ty
                && tcx.visibility(def_id).is_accessible_from(body_def_id, tcx)
        })
        .collect::<Vec<_>>();
    let names = constants.iter().map(|&def_id| tcx.item_name(def_id)).collect::<Vec<_>>();
    if let Some(item_name) = find_best_match_for_name(&names, name, None)
        && let Some(position) = names.iter().position(|&n| n == item_name)
        && let Some(&def_id) = constants.get(position)
    {
        return Some(errors::PatternTypo {
            span,
            code: with_no_trimmed_paths!(tcx.def_path_str(def_id)),
            kind: "constant",
            item_name,
        });
    }

    None
}

/// Return whether we should consider the current place as a drop guard and skip reporting.
fn maybe_drop_guard<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    index: PlaceIndex,
    ever_dropped: &DenseBitSet<PlaceIndex>,
    checked_places: &PlaceSet<'tcx>,
    body: &Body<'tcx>,
) -> bool {
    if ever_dropped.contains(index) {
        let ty = checked_places.places[index].ty(&body.local_decls, tcx).ty;
        matches!(
            ty.kind(),
            ty::Closure(..)
                | ty::Coroutine(..)
                | ty::Tuple(..)
                | ty::Adt(..)
                | ty::Dynamic(..)
                | ty::Array(..)
                | ty::Slice(..)
                | ty::Alias(ty::Opaque, ..)
        ) && ty.needs_drop(tcx, typing_env)
    } else {
        false
    }
}

/// Detect the following case
///
/// ```text
/// fn change_object(mut a: &Ty) {
///     let a = Ty::new();
///     b = &a;
/// }
/// ```
///
/// where the user likely meant to modify the value behind there reference, use `a` as an out
/// parameter, instead of mutating the local binding. When encountering this we suggest:
///
/// ```text
/// fn change_object(a: &'_ mut Ty) {
///     let a = Ty::new();
///     *b = a;
/// }
/// ```
fn annotate_mut_binding_to_immutable_binding<'tcx>(
    tcx: TyCtxt<'tcx>,
    place: PlaceRef<'tcx>,
    body_def_id: LocalDefId,
    assignment_span: Span,
    body: &Body<'tcx>,
) -> Option<errors::UnusedAssignSuggestion> {
    use rustc_hir as hir;
    use rustc_hir::intravisit::{self, Visitor};

    // Verify we have a mutable argument...
    let local = place.as_local()?;
    let LocalKind::Arg = body.local_kind(local) else { return None };
    let Mutability::Mut = body.local_decls[local].mutability else { return None };

    // ... with reference type...
    let hir_param_index =
        local.as_usize() - if tcx.is_closure_like(body_def_id.to_def_id()) { 2 } else { 1 };
    let fn_decl = tcx.hir_node_by_def_id(body_def_id).fn_decl()?;
    let ty = fn_decl.inputs[hir_param_index];
    let hir::TyKind::Ref(lt, mut_ty) = ty.kind else { return None };

    // ... as a binding pattern.
    let hir_body = tcx.hir_maybe_body_owned_by(body_def_id)?;
    let param = hir_body.params[hir_param_index];
    let hir::PatKind::Binding(hir::BindingMode::MUT, _hir_id, ident, _) = param.pat.kind else {
        return None;
    };

    // Find the assignment to modify.
    let mut finder = ExprFinder { assignment_span, lhs: None, rhs: None };
    finder.visit_body(hir_body);
    let lhs = finder.lhs?;
    let rhs = finder.rhs?;

    let hir::ExprKind::AddrOf(hir::BorrowKind::Ref, _mut, inner) = rhs.kind else { return None };

    // Changes to the parameter's type.
    let pre = if lt.ident.span.is_empty() { "" } else { " " };
    let ty_span = if mut_ty.mutbl.is_mut() {
        // Leave `&'name mut Ty` and `&mut Ty` as they are (#136028).
        None
    } else {
        // `&'name Ty` -> `&'name mut Ty` or `&Ty` -> `&mut Ty`
        Some(mut_ty.ty.span.shrink_to_lo())
    };

    return Some(errors::UnusedAssignSuggestion {
        ty_span,
        pre,
        // Span of the `mut` before the binding.
        ty_ref_span: param.pat.span.until(ident.span),
        // Where to add a `*`.
        pre_lhs_span: lhs.span.shrink_to_lo(),
        // Where to remove the borrow.
        rhs_borrow_span: rhs.span.until(inner.span),
    });

    #[derive(Debug)]
    struct ExprFinder<'hir> {
        assignment_span: Span,
        lhs: Option<&'hir hir::Expr<'hir>>,
        rhs: Option<&'hir hir::Expr<'hir>>,
    }
    impl<'hir> Visitor<'hir> for ExprFinder<'hir> {
        fn visit_expr(&mut self, expr: &'hir hir::Expr<'hir>) {
            if expr.span == self.assignment_span
                && let hir::ExprKind::Assign(lhs, rhs, _) = expr.kind
            {
                self.lhs = Some(lhs);
                self.rhs = Some(rhs);
            } else {
                intravisit::walk_expr(self, expr)
            }
        }
    }
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
                // For checked binary ops, the MIR builder inserts an assertion in between.
                Rvalue::BinaryOp(
                    BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow,
                    box (Operand::Copy(lhs), _),
                ) => {
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

                    // We ignore indirect self-assignment, because both occurrences of `dest` are uses.
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
                // Straight self-assignment.
                Rvalue::BinaryOp(op, box (Operand::Copy(lhs), _)) => {
                    if lhs != first_place {
                        continue;
                    }

                    // We ignore indirect self-assignment, because both occurrences of `dest` are uses.
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
                _ => {}
            }
        }
    }

    self_assign
}

#[derive(Default, Debug)]
struct PlaceSet<'tcx> {
    places: IndexVec<PlaceIndex, PlaceRef<'tcx>>,
    names: IndexVec<PlaceIndex, Option<(Symbol, Span)>>,

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
            self.names.insert(
                index,
                (Symbol::intern(&capture.to_string(tcx)), capture.get_path_span(tcx)),
            );
        }
    }

    fn record_debuginfo(&mut self, var_debug_info: &Vec<VarDebugInfo<'tcx>>) {
        let ignore_name = |name: Symbol| {
            name == sym::empty || name == kw::SelfLower || name.as_str().starts_with('_')
        };
        for var_debug_info in var_debug_info {
            if let VarDebugInfoContents::Place(place) = var_debug_info.value
                && let Some(index) = self.locals[place.local]
                && !ignore_name(var_debug_info.name)
            {
                self.names.get_or_insert_with(index, || {
                    (var_debug_info.name, var_debug_info.source_info.span)
                });
            }
        }

        // Discard places that will not result in a diagnostic.
        for index_opt in self.locals.iter_mut() {
            if let Some(index) = *index_opt {
                let remove = match self.names[index] {
                    None => true,
                    Some((name, _)) => ignore_name(name),
                };
                if remove {
                    *index_opt = None;
                }
            }
        }
    }

    #[inline]
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

struct AssignmentResult<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    checked_places: &'a PlaceSet<'tcx>,
    body: &'a Body<'tcx>,
    /// Set of locals that are live at least once. This is used to report fully unused locals.
    ever_live: DenseBitSet<PlaceIndex>,
    /// Set of locals that have a non-trivial drop. This is used to skip reporting unused
    /// assignment if it would be used by the `Drop` impl.
    ever_dropped: DenseBitSet<PlaceIndex>,
    /// Set of assignments for each local. Here, assignment is understood in the AST sense. Any
    /// MIR that may look like an assignment (Assign, DropAndReplace, Yield, Call) are considered.
    ///
    /// For each local, we return a map: for each source position, whether the statement is live
    /// and which kind of access it performs. When we encounter multiple statements at the same
    /// location, we only increase the liveness, in order to avoid false positives.
    assignments: IndexVec<PlaceIndex, FxIndexMap<SourceInfo, Access>>,
}

impl<'a, 'tcx> AssignmentResult<'a, 'tcx> {
    /// Collect all assignments to checked locals.
    ///
    /// Assignments are collected, even if they are live. Dead assignments are reported, and live
    /// assignments are used to make diagnostics correct for match guards.
    fn find_dead_assignments(
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        checked_places: &'a PlaceSet<'tcx>,
        cursor: &mut ResultsCursor<'_, 'tcx, MaybeLivePlaces<'_, 'tcx>>,
        body: &'a Body<'tcx>,
    ) -> AssignmentResult<'a, 'tcx> {
        let mut ever_live = DenseBitSet::new_empty(checked_places.len());
        let mut ever_dropped = DenseBitSet::new_empty(checked_places.len());
        let mut assignments = IndexVec::<PlaceIndex, FxIndexMap<_, _>>::from_elem(
            Default::default(),
            &checked_places.places,
        );

        let mut check_place =
            |place: Place<'tcx>, kind, source_info: SourceInfo, live: &DenseBitSet<PlaceIndex>| {
                if let Some((index, extra_projections)) = checked_places.get(place.as_ref()) {
                    if !is_indirect(extra_projections) {
                        match assignments[index].entry(source_info) {
                            IndexEntry::Vacant(v) => {
                                let access = Access { kind, live: live.contains(index) };
                                v.insert(access);
                            }
                            IndexEntry::Occupied(mut o) => {
                                // There were already a sighting. Mark this statement as live if it
                                // was, to avoid false positives.
                                o.get_mut().live |= live.contains(index);
                            }
                        }
                    }
                }
            };

        let mut record_drop = |place: Place<'tcx>| {
            if let Some((index, &[])) = checked_places.get(place.as_ref()) {
                ever_dropped.insert(index);
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
                    | StatementKind::BackwardIncompatibleDropHint { .. }
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
                let access = Access { kind, live: live.contains(index) };
                assignments[index].insert(source_info, access);
            }
        }

        AssignmentResult {
            tcx,
            typing_env,
            checked_places,
            ever_live,
            ever_dropped,
            assignments,
            body,
        }
    }

    /// Match guards introduce a different local to freeze the guarded value as immutable.
    /// Having two locals, we need to make sure that we do not report an unused_variable
    /// when the guard local is used but not the arm local, or vice versa, like in this example.
    ///
    ///    match 5 {
    ///      x if x > 2 => {}
    ///      ^    ^- This is `local`
    ///      +------ This is `arm_local`
    ///      _ => {}
    ///    }
    ///
    fn merge_guards(&mut self) {
        for (index, place) in self.checked_places.iter() {
            let local = place.local;
            if let &LocalInfo::User(BindingForm::RefForGuard(arm_local)) =
                self.body.local_decls[local].local_info()
            {
                debug_assert!(place.projection.is_empty());

                // Local to use in the arm.
                let Some((arm_index, _proj)) = self.checked_places.get(arm_local.into()) else {
                    continue;
                };
                debug_assert_ne!(index, arm_index);
                debug_assert_eq!(_proj, &[]);

                // Mark the arm local as used if the guard local is used.
                if self.ever_live.contains(index) {
                    self.ever_live.insert(arm_index);
                }

                // Some assignments are common to both locals in the source code.
                // Sadly, we can only detect this using the `source_info`.
                // Therefore, we loop over all the assignments we have for the guard local:
                // - if they already appeared for the arm local, the assignment is live if one of the
                //   two versions is live;
                // - if it does not appear for the arm local, it happened inside the guard, so we add
                //   it as-is.
                let guard_assignments = std::mem::take(&mut self.assignments[index]);
                let arm_assignments = &mut self.assignments[arm_index];
                for (source_info, access) in guard_assignments {
                    match arm_assignments.entry(source_info) {
                        IndexEntry::Vacant(v) => {
                            v.insert(access);
                        }
                        IndexEntry::Occupied(mut o) => {
                            o.get_mut().live |= access.live;
                        }
                    }
                }
            }
        }
    }

    /// Compute captures that are fully dead.
    fn compute_dead_captures(&self, num_captures: usize) -> DenseBitSet<FieldIdx> {
        // Report to caller the set of dead captures.
        let mut dead_captures = DenseBitSet::new_empty(num_captures);
        for (index, place) in self.checked_places.iter() {
            if self.ever_live.contains(index) {
                continue;
            }

            // This is a capture: pass information to the enclosing function.
            if is_capture(*place) {
                for p in place.projection {
                    if let PlaceElem::Field(f, _) = p {
                        dead_captures.insert(*f);
                        break;
                    }
                }
                continue;
            }
        }

        dead_captures
    }

    /// Report fully unused locals, and forget the corresponding assignments.
    fn report_fully_unused(&mut self) {
        let tcx = self.tcx;

        // Give a diagnostic when any of the string constants look like a naked format string that
        // would interpolate our dead local.
        let mut string_constants_in_body = None;
        let mut maybe_suggest_literal_matching_name = |name: Symbol| {
            // Visiting MIR to enumerate string constants can be expensive, so cache the result.
            let string_constants_in_body = string_constants_in_body.get_or_insert_with(|| {
                struct LiteralFinder {
                    found: Vec<(Span, String)>,
                }

                impl<'tcx> Visitor<'tcx> for LiteralFinder {
                    fn visit_const_operand(&mut self, constant: &ConstOperand<'tcx>, _: Location) {
                        if let ty::Ref(_, ref_ty, _) = constant.ty().kind()
                            && ref_ty.kind() == &ty::Str
                        {
                            let rendered_constant = constant.const_.to_string();
                            self.found.push((constant.span, rendered_constant));
                        }
                    }
                }

                let mut finder = LiteralFinder { found: vec![] };
                finder.visit_body(self.body);
                finder.found
            });

            let brace_name = format!("{{{name}");
            string_constants_in_body
                .iter()
                .filter(|(_, rendered_constant)| {
                    rendered_constant
                        .split(&brace_name)
                        .any(|c| matches!(c.chars().next(), Some('}' | ':')))
                })
                .map(|&(lit, _)| errors::UnusedVariableStringInterp { lit })
                .collect::<Vec<_>>()
        };

        // First, report fully unused locals.
        for (index, place) in self.checked_places.iter() {
            if self.ever_live.contains(index) {
                continue;
            }

            // this is a capture: let the enclosing function report the unused variable.
            if is_capture(*place) {
                continue;
            }

            let local = place.local;
            let decl = &self.body.local_decls[local];

            if decl.from_compiler_desugaring() {
                continue;
            }

            // Only report actual user-defined binding from now on.
            let LocalInfo::User(BindingForm::Var(binding)) = decl.local_info() else { continue };
            let Some(hir_id) = decl.source_info.scope.lint_root(&self.body.source_scopes) else {
                continue;
            };

            let introductions = &binding.introductions;

            let Some((name, def_span)) = self.checked_places.names[index] else { continue };

            // #117284, when `ident_span` and `def_span` have different contexts
            // we can't provide a good suggestion, instead we pointed out the spans from macro
            let from_macro = def_span.from_expansion()
                && introductions.iter().any(|intro| intro.span.eq_ctxt(def_span));

            let maybe_suggest_typo = || {
                if let LocalKind::Arg = self.body.local_kind(local) {
                    None
                } else {
                    maybe_suggest_unit_pattern_typo(
                        tcx,
                        self.body.source.def_id(),
                        name,
                        def_span,
                        decl.ty,
                    )
                }
            };

            let statements = &mut self.assignments[index];
            if statements.is_empty() {
                let sugg = if from_macro {
                    errors::UnusedVariableSugg::NoSugg { span: def_span, name }
                } else {
                    let typo = maybe_suggest_typo();
                    errors::UnusedVariableSugg::TryPrefix { spans: vec![def_span], name, typo }
                };
                tcx.emit_node_span_lint(
                    lint::builtin::UNUSED_VARIABLES,
                    hir_id,
                    def_span,
                    errors::UnusedVariable {
                        name,
                        string_interp: maybe_suggest_literal_matching_name(name),
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
                if maybe_drop_guard(
                    tcx,
                    self.typing_env,
                    index,
                    &self.ever_dropped,
                    self.checked_places,
                    self.body,
                ) {
                    statements.clear();
                    continue;
                }

                let typo = maybe_suggest_typo();
                tcx.emit_node_span_lint(
                    lint::builtin::UNUSED_VARIABLES,
                    hir_id,
                    def_span,
                    errors::UnusedVarAssignedOnly { name, typo },
                );
                continue;
            }

            // We do not have outstanding assignments, suggest renaming the binding.
            let spans = introductions.iter().map(|intro| intro.span).collect::<Vec<_>>();

            let any_shorthand = introductions.iter().any(|intro| intro.is_shorthand);

            let sugg = if any_shorthand {
                errors::UnusedVariableSugg::TryIgnore {
                    name,
                    shorthands: introductions
                        .iter()
                        .filter_map(
                            |intro| if intro.is_shorthand { Some(intro.span) } else { None },
                        )
                        .collect(),
                    non_shorthands: introductions
                        .iter()
                        .filter_map(
                            |intro| {
                                if !intro.is_shorthand { Some(intro.span) } else { None }
                            },
                        )
                        .collect(),
                }
            } else if from_macro {
                errors::UnusedVariableSugg::NoSugg { span: def_span, name }
            } else if !introductions.is_empty() {
                let typo = maybe_suggest_typo();
                errors::UnusedVariableSugg::TryPrefix { name, typo, spans: spans.clone() }
            } else {
                let typo = maybe_suggest_typo();
                errors::UnusedVariableSugg::TryPrefix { name, typo, spans: vec![def_span] }
            };

            tcx.emit_node_span_lint(
                lint::builtin::UNUSED_VARIABLES,
                hir_id,
                spans,
                errors::UnusedVariable {
                    name,
                    string_interp: maybe_suggest_literal_matching_name(name),
                    sugg,
                },
            );
        }
    }

    /// Second, report unused assignments that do not correspond to initialization.
    /// Initializations have been removed in the previous loop reporting unused variables.
    fn report_unused_assignments(self) {
        let tcx = self.tcx;

        for (index, statements) in self.assignments.into_iter_enumerated() {
            if statements.is_empty() {
                continue;
            }

            let Some((name, decl_span)) = self.checked_places.names[index] else { continue };

            // We have outstanding assignments and with non-trivial drop.
            // This is probably a drop-guard, so we do not issue a warning there.
            if maybe_drop_guard(
                tcx,
                self.typing_env,
                index,
                &self.ever_dropped,
                self.checked_places,
                self.body,
            ) {
                continue;
            }

            // We probed MIR in reverse order for dataflow.
            // We revert the vector to give a consistent order to the user.
            for (source_info, Access { live, kind }) in statements.into_iter().rev() {
                if live {
                    continue;
                }

                // Report the dead assignment.
                let Some(hir_id) = source_info.scope.lint_root(&self.body.source_scopes) else {
                    continue;
                };

                match kind {
                    AccessKind::Assign => {
                        let suggestion = annotate_mut_binding_to_immutable_binding(
                            tcx,
                            self.checked_places.places[index],
                            self.body.source.def_id().expect_local(),
                            source_info.span,
                            self.body,
                        );
                        tcx.emit_node_span_lint(
                            lint::builtin::UNUSED_ASSIGNMENTS,
                            hir_id,
                            source_info.span,
                            errors::UnusedAssign { name, help: suggestion.is_none(), suggestion },
                        )
                    }
                    AccessKind::Param => tcx.emit_node_span_lint(
                        lint::builtin::UNUSED_ASSIGNMENTS,
                        hir_id,
                        source_info.span,
                        errors::UnusedAssignPassed { name },
                    ),
                    AccessKind::Capture => tcx.emit_node_span_lint(
                        lint::builtin::UNUSED_ASSIGNMENTS,
                        hir_id,
                        decl_span,
                        errors::UnusedCaptureMaybeCaptureRef { name },
                    ),
                }
            }
        }
    }
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
    fn transfer_function<'a>(
        &'a self,
        trans: &'a mut DenseBitSet<PlaceIndex>,
    ) -> TransferFunction<'a, 'tcx> {
        TransferFunction {
            tcx: self.tcx,
            checked_places: &self.checked_places,
            capture_kind: self.capture_kind,
            trans,
            self_assignment: &self.self_assignment,
        }
    }
}

impl<'tcx> Analysis<'tcx> for MaybeLivePlaces<'_, 'tcx> {
    type Domain = DenseBitSet<PlaceIndex>;
    type Direction = Backward;

    const NAME: &'static str = "liveness-lint";

    fn bottom_value(&self, _: &Body<'tcx>) -> Self::Domain {
        // bottom = not live
        DenseBitSet::new_empty(self.checked_places.len())
    }

    fn initialize_start_block(&self, _: &Body<'tcx>, _: &mut Self::Domain) {
        // No variables are live until we observe a use
    }

    fn apply_primary_statement_effect(
        &self,
        trans: &mut Self::Domain,
        statement: &Statement<'tcx>,
        location: Location,
    ) {
        self.transfer_function(trans).visit_statement(statement, location);
    }

    fn apply_primary_terminator_effect<'mir>(
        &self,
        trans: &mut Self::Domain,
        terminator: &'mir Terminator<'tcx>,
        location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        self.transfer_function(trans).visit_terminator(terminator, location);
        terminator.edges()
    }

    fn apply_call_return_effect(
        &self,
        _trans: &mut Self::Domain,
        _block: BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        // FIXME: what should happen here?
    }
}

struct TransferFunction<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    checked_places: &'a PlaceSet<'tcx>,
    trans: &'a mut DenseBitSet<PlaceIndex>,
    capture_kind: CaptureKind,
    self_assignment: &'a FxHashSet<Location>,
}

impl<'tcx> Visitor<'tcx> for TransferFunction<'_, 'tcx> {
    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match statement.kind {
            // `ForLet(None)` fake read erroneously marks the just-assigned local as live.
            // This defeats the purpose of the analysis for `let` bindings.
            StatementKind::FakeRead(box (FakeReadCause::ForLet(None), _)) => return,
            // Handle self-assignment by restricting the read/write they do.
            StatementKind::Assign(box (ref dest, ref rvalue))
                if self.self_assignment.contains(&location) =>
            {
                if let Rvalue::BinaryOp(
                    BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow,
                    box (_, rhs),
                ) = rvalue
                {
                    // We are computing the binary operation:
                    // - the LHS will be assigned, so we don't read it;
                    // - the RHS still needs to be read.
                    self.visit_operand(rhs, location);
                    self.visit_place(
                        dest,
                        PlaceContext::MutatingUse(MutatingUseContext::Store),
                        location,
                    );
                } else if let Rvalue::BinaryOp(_, box (_, rhs)) = rvalue {
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
        // By-ref captures could be read by the surrounding environment, so we mark
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
                        self.trans.insert(index);
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
                operands,
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
                Some(DefUse::Def) => {
                    self.trans.remove(index);
                }
                Some(DefUse::Use) => {
                    self.trans.insert(index);
                }
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
                Some(DefUse::Def) => {
                    self.trans.remove(index);
                }
                Some(DefUse::Use) => {
                    self.trans.insert(index);
                }
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
                MutatingUseContext::Store | MutatingUseContext::SetDiscriminant,
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
                MutatingUseContext::RawBorrow
                | MutatingUseContext::Borrow
                | MutatingUseContext::Drop
                | MutatingUseContext::Retag,
            )
            | PlaceContext::NonMutatingUse(
                NonMutatingUseContext::RawBorrow
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
                | NonUseContext::BackwardIncompatibleDropHint
                | NonUseContext::VarDebugInfo,
            ) => None,

            PlaceContext::MutatingUse(MutatingUseContext::Projection)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Projection) => {
                unreachable!("A projection could be a def or a use and must be handled separately")
            }
        }
    }
}
