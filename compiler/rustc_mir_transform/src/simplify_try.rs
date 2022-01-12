//! The general point of the optimizations provided here is to simplify something like:
//!
//! ```rust
//! match x {
//!     Ok(x) => Ok(x),
//!     Err(x) => Err(x)
//! }
//! ```
//!
//! into just `x`.

use crate::{simplify, MirPass};
use itertools::Itertools as _;
use rustc_index::{bit_set::BitSet, vec::IndexVec};
use rustc_middle::mir::visit::{NonUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, List, Ty, TyCtxt};
use rustc_target::abi::VariantIdx;
use std::iter::{once, Enumerate, Peekable};
use std::slice::Iter;

/// Simplifies arms of form `Variant(x) => Variant(x)` to just a move.
///
/// This is done by transforming basic blocks where the statements match:
///
/// ```rust
/// _LOCAL_TMP = ((_LOCAL_1 as Variant ).FIELD: TY );
/// _TMP_2 = _LOCAL_TMP;
/// ((_LOCAL_0 as Variant).FIELD: TY) = move _TMP_2;
/// discriminant(_LOCAL_0) = VAR_IDX;
/// ```
///
/// into:
///
/// ```rust
/// _LOCAL_0 = move _LOCAL_1
/// ```
pub struct SimplifyArmIdentity;

#[derive(Debug)]
struct ArmIdentityInfo<'tcx> {
    /// Storage location for the variant's field
    local_temp_0: Local,
    /// Storage location holding the variant being read from
    local_1: Local,
    /// The variant field being read from
    vf_s0: VarField<'tcx>,
    /// Index of the statement which loads the variant being read
    get_variant_field_stmt: usize,

    /// Tracks each assignment to a temporary of the variant's field
    field_tmp_assignments: Vec<(Local, Local)>,

    /// Storage location holding the variant's field that was read from
    local_tmp_s1: Local,
    /// Storage location holding the enum that we are writing to
    local_0: Local,
    /// The variant field being written to
    vf_s1: VarField<'tcx>,

    /// Storage location that the discriminant is being written to
    set_discr_local: Local,
    /// The variant being written
    set_discr_var_idx: VariantIdx,

    /// Index of the statement that should be overwritten as a move
    stmt_to_overwrite: usize,
    /// SourceInfo for the new move
    source_info: SourceInfo,

    /// Indices of matching Storage{Live,Dead} statements encountered.
    /// (StorageLive index,, StorageDead index, Local)
    storage_stmts: Vec<(usize, usize, Local)>,

    /// The statements that should be removed (turned into nops)
    stmts_to_remove: Vec<usize>,

    /// Indices of debug variables that need to be adjusted to point to
    // `{local_0}.{dbg_projection}`.
    dbg_info_to_adjust: Vec<usize>,

    /// The projection used to rewrite debug info.
    dbg_projection: &'tcx List<PlaceElem<'tcx>>,
}

fn get_arm_identity_info<'a, 'tcx>(
    stmts: &'a [Statement<'tcx>],
    locals_count: usize,
    debug_info: &'a [VarDebugInfo<'tcx>],
) -> Option<ArmIdentityInfo<'tcx>> {
    // This can't possibly match unless there are at least 3 statements in the block
    // so fail fast on tiny blocks.
    if stmts.len() < 3 {
        return None;
    }

    let mut tmp_assigns = Vec::new();
    let mut nop_stmts = Vec::new();
    let mut storage_stmts = Vec::new();
    let mut storage_live_stmts = Vec::new();
    let mut storage_dead_stmts = Vec::new();

    type StmtIter<'a, 'tcx> = Peekable<Enumerate<Iter<'a, Statement<'tcx>>>>;

    fn is_storage_stmt(stmt: &Statement<'_>) -> bool {
        matches!(stmt.kind, StatementKind::StorageLive(_) | StatementKind::StorageDead(_))
    }

    /// Eats consecutive Statements which match `test`, performing the specified `action` for each.
    /// The iterator `stmt_iter` is not advanced if none were matched.
    fn try_eat<'a, 'tcx>(
        stmt_iter: &mut StmtIter<'a, 'tcx>,
        test: impl Fn(&'a Statement<'tcx>) -> bool,
        mut action: impl FnMut(usize, &'a Statement<'tcx>),
    ) {
        while stmt_iter.peek().map_or(false, |(_, stmt)| test(stmt)) {
            let (idx, stmt) = stmt_iter.next().unwrap();

            action(idx, stmt);
        }
    }

    /// Eats consecutive `StorageLive` and `StorageDead` Statements.
    /// The iterator `stmt_iter` is not advanced if none were found.
    fn try_eat_storage_stmts(
        stmt_iter: &mut StmtIter<'_, '_>,
        storage_live_stmts: &mut Vec<(usize, Local)>,
        storage_dead_stmts: &mut Vec<(usize, Local)>,
    ) {
        try_eat(stmt_iter, is_storage_stmt, |idx, stmt| {
            if let StatementKind::StorageLive(l) = stmt.kind {
                storage_live_stmts.push((idx, l));
            } else if let StatementKind::StorageDead(l) = stmt.kind {
                storage_dead_stmts.push((idx, l));
            }
        })
    }

    fn is_tmp_storage_stmt(stmt: &Statement<'_>) -> bool {
        use rustc_middle::mir::StatementKind::Assign;
        if let Assign(box (place, Rvalue::Use(Operand::Copy(p) | Operand::Move(p)))) = &stmt.kind {
            place.as_local().is_some() && p.as_local().is_some()
        } else {
            false
        }
    }

    /// Eats consecutive `Assign` Statements.
    // The iterator `stmt_iter` is not advanced if none were found.
    fn try_eat_assign_tmp_stmts(
        stmt_iter: &mut StmtIter<'_, '_>,
        tmp_assigns: &mut Vec<(Local, Local)>,
        nop_stmts: &mut Vec<usize>,
    ) {
        try_eat(stmt_iter, is_tmp_storage_stmt, |idx, stmt| {
            use rustc_middle::mir::StatementKind::Assign;
            if let Assign(box (place, Rvalue::Use(Operand::Copy(p) | Operand::Move(p)))) =
                &stmt.kind
            {
                tmp_assigns.push((place.as_local().unwrap(), p.as_local().unwrap()));
                nop_stmts.push(idx);
            }
        })
    }

    fn find_storage_live_dead_stmts_for_local(
        local: Local,
        stmts: &[Statement<'_>],
    ) -> Option<(usize, usize)> {
        trace!("looking for {:?}", local);
        let mut storage_live_stmt = None;
        let mut storage_dead_stmt = None;
        for (idx, stmt) in stmts.iter().enumerate() {
            if stmt.kind == StatementKind::StorageLive(local) {
                storage_live_stmt = Some(idx);
            } else if stmt.kind == StatementKind::StorageDead(local) {
                storage_dead_stmt = Some(idx);
            }
        }

        Some((storage_live_stmt?, storage_dead_stmt.unwrap_or(usize::MAX)))
    }

    // Try to match the expected MIR structure with the basic block we're processing.
    // We want to see something that looks like:
    // ```
    // (StorageLive(_) | StorageDead(_));*
    // _LOCAL_INTO = ((_LOCAL_FROM as Variant).FIELD: TY);
    // (StorageLive(_) | StorageDead(_));*
    // (tmp_n+1 = tmp_n);*
    // (StorageLive(_) | StorageDead(_));*
    // (tmp_n+1 = tmp_n);*
    // ((LOCAL_FROM as Variant).FIELD: TY) = move tmp;
    // discriminant(LOCAL_FROM) = VariantIdx;
    // (StorageLive(_) | StorageDead(_));*
    // ```
    let mut stmt_iter = stmts.iter().enumerate().peekable();

    try_eat_storage_stmts(&mut stmt_iter, &mut storage_live_stmts, &mut storage_dead_stmts);

    let (get_variant_field_stmt, stmt) = stmt_iter.next()?;
    let (local_tmp_s0, local_1, vf_s0, dbg_projection) = match_get_variant_field(stmt)?;

    try_eat_storage_stmts(&mut stmt_iter, &mut storage_live_stmts, &mut storage_dead_stmts);

    try_eat_assign_tmp_stmts(&mut stmt_iter, &mut tmp_assigns, &mut nop_stmts);

    try_eat_storage_stmts(&mut stmt_iter, &mut storage_live_stmts, &mut storage_dead_stmts);

    try_eat_assign_tmp_stmts(&mut stmt_iter, &mut tmp_assigns, &mut nop_stmts);

    let (idx, stmt) = stmt_iter.next()?;
    let (local_tmp_s1, local_0, vf_s1) = match_set_variant_field(stmt)?;
    nop_stmts.push(idx);

    let (idx, stmt) = stmt_iter.next()?;
    let (set_discr_local, set_discr_var_idx) = match_set_discr(stmt)?;
    let discr_stmt_source_info = stmt.source_info;
    nop_stmts.push(idx);

    try_eat_storage_stmts(&mut stmt_iter, &mut storage_live_stmts, &mut storage_dead_stmts);

    for (live_idx, live_local) in storage_live_stmts {
        if let Some(i) = storage_dead_stmts.iter().rposition(|(_, l)| *l == live_local) {
            let (dead_idx, _) = storage_dead_stmts.swap_remove(i);
            storage_stmts.push((live_idx, dead_idx, live_local));

            if live_local == local_tmp_s0 {
                nop_stmts.push(get_variant_field_stmt);
            }
        }
    }
    // We sort primitive usize here so we can use unstable sort
    nop_stmts.sort_unstable();

    // Use one of the statements we're going to discard between the point
    // where the storage location for the variant field becomes live and
    // is killed.
    let (live_idx, dead_idx) = find_storage_live_dead_stmts_for_local(local_tmp_s0, stmts)?;
    let stmt_to_overwrite =
        nop_stmts.iter().find(|stmt_idx| live_idx < **stmt_idx && **stmt_idx < dead_idx);

    let mut tmp_assigned_vars = BitSet::new_empty(locals_count);
    for (l, r) in &tmp_assigns {
        tmp_assigned_vars.insert(*l);
        tmp_assigned_vars.insert(*r);
    }

    let dbg_info_to_adjust: Vec<_> = debug_info
        .iter()
        .enumerate()
        .filter_map(|(i, var_info)| {
            if let VarDebugInfoContents::Place(p) = var_info.value {
                if tmp_assigned_vars.contains(p.local) {
                    return Some(i);
                }
            }

            None
        })
        .collect();

    Some(ArmIdentityInfo {
        local_temp_0: local_tmp_s0,
        local_1,
        vf_s0,
        get_variant_field_stmt,
        field_tmp_assignments: tmp_assigns,
        local_tmp_s1,
        local_0,
        vf_s1,
        set_discr_local,
        set_discr_var_idx,
        stmt_to_overwrite: *stmt_to_overwrite?,
        source_info: discr_stmt_source_info,
        storage_stmts,
        stmts_to_remove: nop_stmts,
        dbg_info_to_adjust,
        dbg_projection,
    })
}

fn optimization_applies<'tcx>(
    opt_info: &ArmIdentityInfo<'tcx>,
    local_decls: &IndexVec<Local, LocalDecl<'tcx>>,
    local_uses: &IndexVec<Local, usize>,
    var_debug_info: &[VarDebugInfo<'tcx>],
) -> bool {
    trace!("testing if optimization applies...");

    // FIXME(wesleywiser): possibly relax this restriction?
    if opt_info.local_0 == opt_info.local_1 {
        trace!("NO: moving into ourselves");
        return false;
    } else if opt_info.vf_s0 != opt_info.vf_s1 {
        trace!("NO: the field-and-variant information do not match");
        return false;
    } else if local_decls[opt_info.local_0].ty != local_decls[opt_info.local_1].ty {
        // FIXME(Centril,oli-obk): possibly relax to same layout?
        trace!("NO: source and target locals have different types");
        return false;
    } else if (opt_info.local_0, opt_info.vf_s0.var_idx)
        != (opt_info.set_discr_local, opt_info.set_discr_var_idx)
    {
        trace!("NO: the discriminants do not match");
        return false;
    }

    // Verify the assignment chain consists of the form b = a; c = b; d = c; etc...
    if opt_info.field_tmp_assignments.is_empty() {
        trace!("NO: no assignments found");
        return false;
    }
    let mut last_assigned_to = opt_info.field_tmp_assignments[0].1;
    let source_local = last_assigned_to;
    for (l, r) in &opt_info.field_tmp_assignments {
        if *r != last_assigned_to {
            trace!("NO: found unexpected assignment {:?} = {:?}", l, r);
            return false;
        }

        last_assigned_to = *l;
    }

    // Check that the first and last used locals are only used twice
    // since they are of the form:
    //
    // ```
    // _first = ((_x as Variant).n: ty);
    // _n = _first;
    // ...
    // ((_y as Variant).n: ty) = _n;
    // discriminant(_y) = z;
    // ```
    for (l, r) in &opt_info.field_tmp_assignments {
        if local_uses[*l] != 2 {
            warn!("NO: FAILED assignment chain local {:?} was used more than twice", l);
            return false;
        } else if local_uses[*r] != 2 {
            warn!("NO: FAILED assignment chain local {:?} was used more than twice", r);
            return false;
        }
    }

    // Check that debug info only points to full Locals and not projections.
    for dbg_idx in &opt_info.dbg_info_to_adjust {
        let dbg_info = &var_debug_info[*dbg_idx];
        if let VarDebugInfoContents::Place(p) = dbg_info.value {
            if !p.projection.is_empty() {
                trace!("NO: debug info for {:?} had a projection {:?}", dbg_info.name, p);
                return false;
            }
        }
    }

    if source_local != opt_info.local_temp_0 {
        trace!(
            "NO: start of assignment chain does not match enum variant temp: {:?} != {:?}",
            source_local,
            opt_info.local_temp_0
        );
        return false;
    } else if last_assigned_to != opt_info.local_tmp_s1 {
        trace!(
            "NO: end of assignemnt chain does not match written enum temp: {:?} != {:?}",
            last_assigned_to,
            opt_info.local_tmp_s1
        );
        return false;
    }

    trace!("SUCCESS: optimization applies!");
    true
}

impl<'tcx> MirPass<'tcx> for SimplifyArmIdentity {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // FIXME(77359): This optimization can result in unsoundness.
        if !tcx.sess.opts.debugging_opts.unsound_mir_opts {
            return;
        }

        let source = body.source;
        trace!("running SimplifyArmIdentity on {:?}", source);

        let local_uses = LocalUseCounter::get_local_uses(body);
        let (basic_blocks, local_decls, debug_info) =
            body.basic_blocks_local_decls_mut_and_var_debug_info();
        for bb in basic_blocks {
            if let Some(opt_info) =
                get_arm_identity_info(&bb.statements, local_decls.len(), debug_info)
            {
                trace!("got opt_info = {:#?}", opt_info);
                if !optimization_applies(&opt_info, local_decls, &local_uses, &debug_info) {
                    debug!("optimization skipped for {:?}", source);
                    continue;
                }

                // Also remove unused Storage{Live,Dead} statements which correspond
                // to temps used previously.
                for (live_idx, dead_idx, local) in &opt_info.storage_stmts {
                    // The temporary that we've read the variant field into is scoped to this block,
                    // so we can remove the assignment.
                    if *local == opt_info.local_temp_0 {
                        bb.statements[opt_info.get_variant_field_stmt].make_nop();
                    }

                    for (left, right) in &opt_info.field_tmp_assignments {
                        if local == left || local == right {
                            bb.statements[*live_idx].make_nop();
                            bb.statements[*dead_idx].make_nop();
                        }
                    }
                }

                // Right shape; transform
                for stmt_idx in opt_info.stmts_to_remove {
                    bb.statements[stmt_idx].make_nop();
                }

                let stmt = &mut bb.statements[opt_info.stmt_to_overwrite];
                stmt.source_info = opt_info.source_info;
                stmt.kind = StatementKind::Assign(Box::new((
                    opt_info.local_0.into(),
                    Rvalue::Use(Operand::Move(opt_info.local_1.into())),
                )));

                bb.statements.retain(|stmt| stmt.kind != StatementKind::Nop);

                // Fix the debug info to point to the right local
                for dbg_index in opt_info.dbg_info_to_adjust {
                    let dbg_info = &mut debug_info[dbg_index];
                    assert!(
                        matches!(dbg_info.value, VarDebugInfoContents::Place(_)),
                        "value was not a Place"
                    );
                    if let VarDebugInfoContents::Place(p) = &mut dbg_info.value {
                        assert!(p.projection.is_empty());
                        p.local = opt_info.local_0;
                        p.projection = opt_info.dbg_projection;
                    }
                }

                trace!("block is now {:?}", bb.statements);
            }
        }
    }
}

struct LocalUseCounter {
    local_uses: IndexVec<Local, usize>,
}

impl LocalUseCounter {
    fn get_local_uses(body: &Body<'_>) -> IndexVec<Local, usize> {
        let mut counter = LocalUseCounter { local_uses: IndexVec::from_elem(0, &body.local_decls) };
        counter.visit_body(body);
        counter.local_uses
    }
}

impl Visitor<'_> for LocalUseCounter {
    fn visit_local(&mut self, local: &Local, context: PlaceContext, _location: Location) {
        if context.is_storage_marker()
            || context == PlaceContext::NonUse(NonUseContext::VarDebugInfo)
        {
            return;
        }

        self.local_uses[*local] += 1;
    }
}

/// Match on:
/// ```rust
/// _LOCAL_INTO = ((_LOCAL_FROM as Variant).FIELD: TY);
/// ```
fn match_get_variant_field<'tcx>(
    stmt: &Statement<'tcx>,
) -> Option<(Local, Local, VarField<'tcx>, &'tcx List<PlaceElem<'tcx>>)> {
    match &stmt.kind {
        StatementKind::Assign(box (
            place_into,
            Rvalue::Use(Operand::Copy(pf) | Operand::Move(pf)),
        )) => {
            let local_into = place_into.as_local()?;
            let (local_from, vf) = match_variant_field_place(*pf)?;
            Some((local_into, local_from, vf, pf.projection))
        }
        _ => None,
    }
}

/// Match on:
/// ```rust
/// ((_LOCAL_FROM as Variant).FIELD: TY) = move _LOCAL_INTO;
/// ```
fn match_set_variant_field<'tcx>(stmt: &Statement<'tcx>) -> Option<(Local, Local, VarField<'tcx>)> {
    match &stmt.kind {
        StatementKind::Assign(box (place_from, Rvalue::Use(Operand::Move(place_into)))) => {
            let local_into = place_into.as_local()?;
            let (local_from, vf) = match_variant_field_place(*place_from)?;
            Some((local_into, local_from, vf))
        }
        _ => None,
    }
}

/// Match on:
/// ```rust
/// discriminant(_LOCAL_TO_SET) = VAR_IDX;
/// ```
fn match_set_discr(stmt: &Statement<'_>) -> Option<(Local, VariantIdx)> {
    match &stmt.kind {
        StatementKind::SetDiscriminant { place, variant_index } => {
            Some((place.as_local()?, *variant_index))
        }
        _ => None,
    }
}

#[derive(PartialEq, Debug)]
struct VarField<'tcx> {
    field: Field,
    field_ty: Ty<'tcx>,
    var_idx: VariantIdx,
}

/// Match on `((_LOCAL as Variant).FIELD: TY)`.
fn match_variant_field_place<'tcx>(place: Place<'tcx>) -> Option<(Local, VarField<'tcx>)> {
    match place.as_ref() {
        PlaceRef {
            local,
            projection: &[ProjectionElem::Downcast(_, var_idx), ProjectionElem::Field(field, ty)],
        } => Some((local, VarField { field, field_ty: ty, var_idx })),
        _ => None,
    }
}

/// Simplifies `SwitchInt(_) -> [targets]`,
/// where all the `targets` have the same form,
/// into `goto -> target_first`.
pub struct SimplifyBranchSame;

impl<'tcx> MirPass<'tcx> for SimplifyBranchSame {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // This optimization is disabled by default for now due to
        // soundness concerns; see issue #89485 and PR #89489.
        if !tcx.sess.opts.debugging_opts.unsound_mir_opts {
            return;
        }

        trace!("Running SimplifyBranchSame on {:?}", body.source);
        let finder = SimplifyBranchSameOptimizationFinder { body, tcx };
        let opts = finder.find();

        let did_remove_blocks = opts.len() > 0;
        for opt in opts.iter() {
            trace!("SUCCESS: Applying optimization {:?}", opt);
            // Replace `SwitchInt(..) -> [bb_first, ..];` with a `goto -> bb_first;`.
            body.basic_blocks_mut()[opt.bb_to_opt_terminator].terminator_mut().kind =
                TerminatorKind::Goto { target: opt.bb_to_goto };
        }

        if did_remove_blocks {
            // We have dead blocks now, so remove those.
            simplify::remove_dead_blocks(tcx, body);
        }
    }
}

#[derive(Debug)]
struct SimplifyBranchSameOptimization {
    /// All basic blocks are equal so go to this one
    bb_to_goto: BasicBlock,
    /// Basic block where the terminator can be simplified to a goto
    bb_to_opt_terminator: BasicBlock,
}

struct SwitchTargetAndValue {
    target: BasicBlock,
    // None in case of the `otherwise` case
    value: Option<u128>,
}

struct SimplifyBranchSameOptimizationFinder<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> SimplifyBranchSameOptimizationFinder<'_, 'tcx> {
    fn find(&self) -> Vec<SimplifyBranchSameOptimization> {
        self.body
            .basic_blocks()
            .iter_enumerated()
            .filter_map(|(bb_idx, bb)| {
                let (discr_switched_on, targets_and_values) = match &bb.terminator().kind {
                    TerminatorKind::SwitchInt { targets, discr, .. } => {
                        let targets_and_values: Vec<_> = targets.iter()
                            .map(|(val, target)| SwitchTargetAndValue { target, value: Some(val) })
                            .chain(once(SwitchTargetAndValue { target: targets.otherwise(), value: None }))
                            .collect();
                        (discr, targets_and_values)
                    },
                    _ => return None,
                };

                // find the adt that has its discriminant read
                // assuming this must be the last statement of the block
                let adt_matched_on = match &bb.statements.last()?.kind {
                    StatementKind::Assign(box (place, rhs))
                        if Some(*place) == discr_switched_on.place() =>
                    {
                        match rhs {
                            Rvalue::Discriminant(adt_place) if adt_place.ty(self.body, self.tcx).ty.is_enum() => adt_place,
                            _ => {
                                trace!("NO: expected a discriminant read of an enum instead of: {:?}", rhs);
                                return None;
                            }
                        }
                    }
                    other => {
                        trace!("NO: expected an assignment of a discriminant read to a place. Found: {:?}", other);
                        return None
                    },
                };

                let mut iter_bbs_reachable = targets_and_values
                    .iter()
                    .map(|target_and_value| (target_and_value, &self.body.basic_blocks()[target_and_value.target]))
                    .filter(|(_, bb)| {
                        // Reaching `unreachable` is UB so assume it doesn't happen.
                        bb.terminator().kind != TerminatorKind::Unreachable
                    })
                    .peekable();

                let bb_first = iter_bbs_reachable.peek().map_or(&targets_and_values[0], |(idx, _)| *idx);
                let mut all_successors_equivalent = StatementEquality::TrivialEqual;

                // All successor basic blocks must be equal or contain statements that are pairwise considered equal.
                for ((target_and_value_l,bb_l), (target_and_value_r,bb_r)) in iter_bbs_reachable.tuple_windows() {
                    let trivial_checks = bb_l.is_cleanup == bb_r.is_cleanup
                                            && bb_l.terminator().kind == bb_r.terminator().kind
                                            && bb_l.statements.len() == bb_r.statements.len();
                    let statement_check = || {
                        bb_l.statements.iter().zip(&bb_r.statements).try_fold(StatementEquality::TrivialEqual, |acc,(l,r)| {
                            let stmt_equality = self.statement_equality(*adt_matched_on, &l, target_and_value_l, &r, target_and_value_r);
                            if matches!(stmt_equality, StatementEquality::NotEqual) {
                                // short circuit
                                None
                            } else {
                                Some(acc.combine(&stmt_equality))
                            }
                        })
                        .unwrap_or(StatementEquality::NotEqual)
                    };
                    if !trivial_checks {
                        all_successors_equivalent = StatementEquality::NotEqual;
                        break;
                    }
                    all_successors_equivalent = all_successors_equivalent.combine(&statement_check());
                };

                match all_successors_equivalent{
                    StatementEquality::TrivialEqual => {
                        // statements are trivially equal, so just take first
                        trace!("Statements are trivially equal");
                        Some(SimplifyBranchSameOptimization {
                            bb_to_goto: bb_first.target,
                            bb_to_opt_terminator: bb_idx,
                        })
                    }
                    StatementEquality::ConsideredEqual(bb_to_choose) => {
                        trace!("Statements are considered equal");
                        Some(SimplifyBranchSameOptimization {
                            bb_to_goto: bb_to_choose,
                            bb_to_opt_terminator: bb_idx,
                        })
                    }
                    StatementEquality::NotEqual => {
                        trace!("NO: not all successors of basic block {:?} were equivalent", bb_idx);
                        None
                    }
                }
            })
            .collect()
    }

    /// Tests if two statements can be considered equal
    ///
    /// Statements can be trivially equal if the kinds match.
    /// But they can also be considered equal in the following case A:
    /// ```
    /// discriminant(_0) = 0;   // bb1
    /// _0 = move _1;           // bb2
    /// ```
    /// In this case the two statements are equal iff
    /// - `_0` is an enum where the variant index 0 is fieldless, and
    /// -  bb1 was targeted by a switch where the discriminant of `_1` was switched on
    fn statement_equality(
        &self,
        adt_matched_on: Place<'tcx>,
        x: &Statement<'tcx>,
        x_target_and_value: &SwitchTargetAndValue,
        y: &Statement<'tcx>,
        y_target_and_value: &SwitchTargetAndValue,
    ) -> StatementEquality {
        let helper = |rhs: &Rvalue<'tcx>,
                      place: &Place<'tcx>,
                      variant_index: &VariantIdx,
                      switch_value: u128,
                      side_to_choose| {
            let place_type = place.ty(self.body, self.tcx).ty;
            let adt = match *place_type.kind() {
                ty::Adt(adt, _) if adt.is_enum() => adt,
                _ => return StatementEquality::NotEqual,
            };
            // We need to make sure that the switch value that targets the bb with
            // SetDiscriminant is the same as the variant discriminant.
            let variant_discr = adt.discriminant_for_variant(self.tcx, *variant_index).val;
            if variant_discr != switch_value {
                trace!(
                    "NO: variant discriminant {} does not equal switch value {}",
                    variant_discr,
                    switch_value
                );
                return StatementEquality::NotEqual;
            }
            let variant_is_fieldless = adt.variants[*variant_index].fields.is_empty();
            if !variant_is_fieldless {
                trace!("NO: variant {:?} was not fieldless", variant_index);
                return StatementEquality::NotEqual;
            }

            match rhs {
                Rvalue::Use(operand) if operand.place() == Some(adt_matched_on) => {
                    StatementEquality::ConsideredEqual(side_to_choose)
                }
                _ => {
                    trace!(
                        "NO: RHS of assignment was {:?}, but expected it to match the adt being matched on in the switch, which is {:?}",
                        rhs,
                        adt_matched_on
                    );
                    StatementEquality::NotEqual
                }
            }
        };
        match (&x.kind, &y.kind) {
            // trivial case
            (x, y) if x == y => StatementEquality::TrivialEqual,

            // check for case A
            (
                StatementKind::Assign(box (_, rhs)),
                StatementKind::SetDiscriminant { place, variant_index },
            ) if y_target_and_value.value.is_some() => {
                // choose basic block of x, as that has the assign
                helper(
                    rhs,
                    place,
                    variant_index,
                    y_target_and_value.value.unwrap(),
                    x_target_and_value.target,
                )
            }
            (
                StatementKind::SetDiscriminant { place, variant_index },
                StatementKind::Assign(box (_, rhs)),
            ) if x_target_and_value.value.is_some() => {
                // choose basic block of y, as that has the assign
                helper(
                    rhs,
                    place,
                    variant_index,
                    x_target_and_value.value.unwrap(),
                    y_target_and_value.target,
                )
            }
            _ => {
                trace!("NO: statements `{:?}` and `{:?}` not considered equal", x, y);
                StatementEquality::NotEqual
            }
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum StatementEquality {
    /// The two statements are trivially equal; same kind
    TrivialEqual,
    /// The two statements are considered equal, but may be of different kinds. The BasicBlock field is the basic block to jump to when performing the branch-same optimization.
    /// For example, `_0 = _1` and `discriminant(_0) = discriminant(0)` are considered equal if 0 is a fieldless variant of an enum. But we don't want to jump to the basic block with the SetDiscriminant, as that is not legal if _1 is not the 0 variant index
    ConsideredEqual(BasicBlock),
    /// The two statements are not equal
    NotEqual,
}

impl StatementEquality {
    fn combine(&self, other: &StatementEquality) -> StatementEquality {
        use StatementEquality::*;
        match (self, other) {
            (TrivialEqual, TrivialEqual) => TrivialEqual,
            (TrivialEqual, ConsideredEqual(b)) | (ConsideredEqual(b), TrivialEqual) => {
                ConsideredEqual(*b)
            }
            (ConsideredEqual(b1), ConsideredEqual(b2)) => {
                if b1 == b2 {
                    ConsideredEqual(*b1)
                } else {
                    NotEqual
                }
            }
            (_, NotEqual) | (NotEqual, _) => NotEqual,
        }
    }
}
