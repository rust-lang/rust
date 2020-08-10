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

use crate::transform::{simplify, MirPass, MirSource};
use itertools::Itertools as _;
use rustc_index::{bit_set::BitSet, vec::IndexVec};
use rustc_middle::mir::visit::{NonUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{List, Ty, TyCtxt};
use rustc_target::abi::VariantIdx;
use std::iter::{Enumerate, Peekable};
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

    fn is_storage_stmt<'tcx>(stmt: &Statement<'tcx>) -> bool {
        matches!(stmt.kind, StatementKind::StorageLive(_) | StatementKind::StorageDead(_))
    }

    /// Eats consecutive Statements which match `test`, performing the specified `action` for each.
    /// The iterator `stmt_iter` is not advanced if none were matched.
    fn try_eat<'a, 'tcx>(
        stmt_iter: &mut StmtIter<'a, 'tcx>,
        test: impl Fn(&'a Statement<'tcx>) -> bool,
        mut action: impl FnMut(usize, &'a Statement<'tcx>),
    ) {
        while stmt_iter.peek().map(|(_, stmt)| test(stmt)).unwrap_or(false) {
            let (idx, stmt) = stmt_iter.next().unwrap();

            action(idx, stmt);
        }
    }

    /// Eats consecutive `StorageLive` and `StorageDead` Statements.
    /// The iterator `stmt_iter` is not advanced if none were found.
    fn try_eat_storage_stmts<'a, 'tcx>(
        stmt_iter: &mut StmtIter<'a, 'tcx>,
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

    fn is_tmp_storage_stmt<'tcx>(stmt: &Statement<'tcx>) -> bool {
        use rustc_middle::mir::StatementKind::Assign;
        if let Assign(box (place, Rvalue::Use(Operand::Copy(p) | Operand::Move(p)))) = &stmt.kind {
            place.as_local().is_some() && p.as_local().is_some()
        } else {
            false
        }
    }

    /// Eats consecutive `Assign` Statements.
    // The iterator `stmt_iter` is not advanced if none were found.
    fn try_eat_assign_tmp_stmts<'a, 'tcx>(
        stmt_iter: &mut StmtIter<'a, 'tcx>,
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

    fn find_storage_live_dead_stmts_for_local<'tcx>(
        local: Local,
        stmts: &[Statement<'tcx>],
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

    nop_stmts.sort();

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

    let mut dbg_info_to_adjust = Vec::new();
    for (i, var_info) in debug_info.iter().enumerate() {
        if tmp_assigned_vars.contains(var_info.place.local) {
            dbg_info_to_adjust.push(i);
        }
    }

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

    // Verify the assigment chain consists of the form b = a; c = b; d = c; etc...
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
        if !dbg_info.place.projection.is_empty() {
            trace!("NO: debug info for {:?} had a projection {:?}", dbg_info.name, dbg_info.place);
            return false;
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
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.opts.debugging_opts.mir_opt_level < 2 {
            return;
        }

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
                stmt.kind = StatementKind::Assign(box (
                    opt_info.local_0.into(),
                    Rvalue::Use(Operand::Move(opt_info.local_1.into())),
                ));

                bb.statements.retain(|stmt| stmt.kind != StatementKind::Nop);

                // Fix the debug info to point to the right local
                for dbg_index in opt_info.dbg_info_to_adjust {
                    let dbg_info = &mut debug_info[dbg_index];
                    assert!(dbg_info.place.projection.is_empty());
                    dbg_info.place.local = opt_info.local_0;
                    dbg_info.place.projection = opt_info.dbg_projection;
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
    fn get_local_uses<'tcx>(body: &Body<'tcx>) -> IndexVec<Local, usize> {
        let mut counter = LocalUseCounter { local_uses: IndexVec::from_elem(0, &body.local_decls) };
        counter.visit_body(body);
        counter.local_uses
    }
}

impl<'tcx> Visitor<'tcx> for LocalUseCounter {
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
        StatementKind::Assign(box (place_into, rvalue_from)) => match rvalue_from {
            Rvalue::Use(Operand::Copy(pf) | Operand::Move(pf)) => {
                let local_into = place_into.as_local()?;
                let (local_from, vf) = match_variant_field_place(*pf)?;
                Some((local_into, local_from, vf, pf.projection))
            }
            _ => None,
        },
        _ => None,
    }
}

/// Match on:
/// ```rust
/// ((_LOCAL_FROM as Variant).FIELD: TY) = move _LOCAL_INTO;
/// ```
fn match_set_variant_field<'tcx>(stmt: &Statement<'tcx>) -> Option<(Local, Local, VarField<'tcx>)> {
    match &stmt.kind {
        StatementKind::Assign(box (place_from, rvalue_into)) => match rvalue_into {
            Rvalue::Use(Operand::Move(place_into)) => {
                let local_into = place_into.as_local()?;
                let (local_from, vf) = match_variant_field_place(*place_from)?;
                Some((local_into, local_from, vf))
            }
            _ => None,
        },
        _ => None,
    }
}

/// Match on:
/// ```rust
/// discriminant(_LOCAL_TO_SET) = VAR_IDX;
/// ```
fn match_set_discr<'tcx>(stmt: &Statement<'tcx>) -> Option<(Local, VariantIdx)> {
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
    fn run_pass(&self, _: TyCtxt<'tcx>, _: MirSource<'tcx>, body: &mut Body<'tcx>) {
        let mut did_remove_blocks = false;
        let bbs = body.basic_blocks_mut();
        for bb_idx in bbs.indices() {
            let targets = match &bbs[bb_idx].terminator().kind {
                TerminatorKind::SwitchInt { targets, .. } => targets,
                _ => continue,
            };

            let mut iter_bbs_reachable = targets
                .iter()
                .map(|idx| (*idx, &bbs[*idx]))
                .filter(|(_, bb)| {
                    // Reaching `unreachable` is UB so assume it doesn't happen.
                    bb.terminator().kind != TerminatorKind::Unreachable
                    // But `asm!(...)` could abort the program,
                    // so we cannot assume that the `unreachable` terminator itself is reachable.
                    // FIXME(Centril): use a normalization pass instead of a check.
                    || bb.statements.iter().any(|stmt| match stmt.kind {
                        StatementKind::LlvmInlineAsm(..) => true,
                        _ => false,
                    })
                })
                .peekable();

            // We want to `goto -> bb_first`.
            let bb_first = iter_bbs_reachable.peek().map(|(idx, _)| *idx).unwrap_or(targets[0]);

            // All successor basic blocks should have the exact same form.
            let all_successors_equivalent =
                iter_bbs_reachable.map(|(_, bb)| bb).tuple_windows().all(|(bb_l, bb_r)| {
                    bb_l.is_cleanup == bb_r.is_cleanup
                        && bb_l.terminator().kind == bb_r.terminator().kind
                        && bb_l.statements.iter().eq_by(&bb_r.statements, |x, y| x.kind == y.kind)
                });

            if all_successors_equivalent {
                // Replace `SwitchInt(..) -> [bb_first, ..];` with a `goto -> bb_first;`.
                bbs[bb_idx].terminator_mut().kind = TerminatorKind::Goto { target: bb_first };
                did_remove_blocks = true;
            }
        }

        if did_remove_blocks {
            // We have dead blocks now, so remove those.
            simplify::remove_dead_blocks(body);
        }
    }
}
