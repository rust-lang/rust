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
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::*;
use rustc_middle::ty::query::TyCtxtAt;
use rustc_middle::ty::{self, ParamEnv, Ty, TyCtxt};
use rustc_span::{Symbol, DUMMY_SP};
use rustc_target::abi::VariantIdx;
use smallvec::SmallVec;
use std::collections::hash_map::Entry;
use std::iter::once;

/// Simplifies arms of form `Variant(x) => Variant(x)` to just a move.
///
/// We do this by finding statement bundles of the following form:
///
/// ```ignore (syntax-highlighting-only)
/// TMP_1 = move? ((SRC as Variant).FIELD);
/// TMP_2 = move? TMP_1;
/// ((DEST as Variant).FIELD) = move? TMP_2;
/// ```
///
/// where `Variant`, `SRC`, and `DEST` must be consistent across all bundles, while `TMP_1`,
/// `TMP_2`, and `FIELD` must vary across all bundles. The actual number of `TMP_*`s that we require
/// is not fixed; we allow arbitrarily many, including possibly zero.
///
/// The bundles of the kind above may be interlaced with each other, however they must form a
/// contiguous set of statements. Finally, we expect that these bundles are terminated with
///
/// ```ignore (syntax-highlighting-only)
/// discriminant(DEST) = VAR_IDX;
/// ```
///
/// where `VAR_IDX` corresponds to `Variant`.
///
/// After doing some checks, we then transform the statements we matched above as follows:
///
/// 1. We remove the `discriminant(DEST) =` and `((DEST as Variant).FIELD: TY)` statements,
///    replacing the `discriminant(DEST)` with `DEST = move? SRC`. This is correct because our
///    bundles precisely track the flow of data from `SRC` into `DEST`.
/// 2. The assignments to temporaries may or may not be removed. How this works is detailed below.
///
/// Additionally, we allow any amount of `StorageLive`/`StorageDead` statements among the statements
/// we match. If we make the transformation, all the `StorageLive` statements are moved to the
/// beginning of the contiguous section of statements that we operate on, while all the
/// `StorageDead` are moved to the end. This is correct because reordering `StorageLive`
/// backwards/`StorageDead` forwards is always correct.
///
/// ## Dealing with `move`/`copy`
///
/// Whether each of the assignments is `move`/`copy` (the "move mode") may be inconsistent, which
/// causes some complications. Using either move mode comes with requirements: `move` must be the
/// last use, while anything `copy`ed must have satisfy a `Copy` bound. Furthermore, we need to
/// assume that anything `copy`ed may be used again later. Making sure we don't miscompile something
/// here requires care. We impose the following rules:
///
/// 1. All assignments out of the source must have the same move mode. This is the move mode that is
///    used for the single assignment that is found in the new output statement. If that move mode
///    then turns out to be `copy`, we additionally require the type of `src` to satisfy a `Copy`
///    bound.
/// 2. For each bundle, if any assignment to a temporary is via `copy`, then this is sufficient to
///    prove that the field is `Copy`. In this case, we keep all assignments to temporaries
///    associated with this bundle. The move mode of the assignment out of `src` is forced to be
///    `copy`. For all other assignments, the move mode is preserved. If none of the move modes are
///    `copy`, then all of the assignments are dropped.
///
pub struct SimplifyArmIdentity;

#[derive(Debug)]
struct ArmIdentityInfo<'tcx> {
    src: Place<'tcx>,
    dest: Place<'tcx>,
    variant: VariantIdx,
    variant_sym: Option<Symbol>,

    // The move mode of the main assignment
    move_mode: MoveMode,

    /// This maps locals to the bundle they are found in, by mapping them to the field associated
    /// with the bundle, along with the type of the field. We use this to update debug info
    temps: FxHashMap<Local, (Field, Ty<'tcx>)>,

    /// Indicates the role that each statement passed into the info gathering plays. This *does not*
    /// include the `discriminant(DEST) = Variant;` statement, which comes immediately after this,
    /// and terminates the range of statements involved in the optimization.
    stmts: Vec<StatementType>,
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
enum MoveMode {
    Move,
    Copy,
}

#[derive(Debug)]
enum StatementType {
    StorageLive,
    StorageDead,
    /// Drop this assignment from the output.
    ///
    /// This is used for all assignments if the bundle's move mode is `move` and for all assignments
    /// into `dest` if the bundle's move mode is `copy`.
    AssignDrop,
    /// Retain the assignment and preserve its move mode.
    ///
    /// This is used for assignments that are both into and out of temporaries in bundles with
    /// `copy` move mode.
    AssignPreserve,
    /// Retain the assignment and force the move mode to be `copy`.
    ///
    /// This is used for assignments out of `src` and into temporaries in bundles with `copy` move
    /// mode.
    AssignForceCopy,
}

struct BundleData<'tcx> {
    /// The sequence of temporaries that this bundle assigns to. The `usize` is the index of the
    /// associated statement, stored so that we can go update the statement types later.
    temps: SmallVec<[(Local, usize); 2]>,
    // `true` once we have assigned to `dest`
    complete: bool,
    move_mode: MoveMode,
    ty: Ty<'tcx>,
}

fn get_arm_identity_info<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    stmts: &'a [Statement<'tcx>],
    local_decls: &'a LocalDecls<'tcx>,
    param_env: ParamEnv<'tcx>,
) -> Option<ArmIdentityInfo<'tcx>> {
    // This can't possibly match unless there are at least 2 statements in the block
    // so fail fast on tiny blocks.
    if stmts.len() < 2 {
        return None;
    }

    let mut src = None;
    let mut dest = None;
    let mut variant = None;
    let mut variant_sym = None;
    let mut move_mode = None;
    let mut stmt_types = Vec::new();

    // This map is not returned; we create and update it as we process statements.
    let mut bundles: FxHashMap<Field, BundleData<'tcx>> = FxHashMap::default();

    for stmt in stmts {
        match &stmt.kind {
            StatementKind::StorageLive(_) => stmt_types.push(StatementType::StorageLive),
            StatementKind::StorageDead(_) => stmt_types.push(StatementType::StorageDead),
            StatementKind::Assign(assign) => {
                update_with_assign(
                    tcx,
                    assign,
                    &mut src,
                    &mut dest,
                    &mut variant,
                    &mut variant_sym,
                    &mut move_mode,
                    &mut stmt_types,
                    &mut bundles,
                )?;
            }
            StatementKind::SetDiscriminant { place, variant_index: var_idx } => {
                // We've found the end of the potential optimization
                let src = src?;
                let dest = dest?;
                let variant = variant?;
                let move_mode = move_mode?;

                // First, make sure the `SetDiscriminant` has the expected form as well
                if variant != *var_idx || dest != **place {
                    return None;
                }

                // Finally, make sure that all the bundles are complete
                if bundles.values().any(|b| !b.complete) {
                    return None;
                }

                // That the types match
                let src_ty = src.ty(local_decls, tcx).ty;
                if src_ty != dest.ty(local_decls, tcx).ty {
                    return None;
                }

                // That we assigned the correct number of fields
                let num_fields_assigned = bundles.len();
                let num_var_fields = src_ty.ty_adt_def().unwrap().variants[variant].fields.len();
                if num_fields_assigned != num_var_fields {
                    return None;
                }

                // And that if we're going to copy the source, it's really a `Copy` type
                if move_mode == MoveMode::Copy
                    && !src_ty.is_copy_modulo_regions(TyCtxtAt { tcx, span: DUMMY_SP }, param_env)
                {
                    return None;
                }

                let temps = bundles
                    .into_iter()
                    .flat_map(|(f, b)| b.temps.into_iter().map(move |(l, _)| (l, (f, b.ty))))
                    .collect();

                return Some(ArmIdentityInfo {
                    src,
                    dest,
                    variant,
                    variant_sym,
                    move_mode,
                    temps,
                    stmts: stmt_types,
                });
            }
            _ => {
                return None;
            }
        };
    }

    // We ran out of statements before we found the `SetDiscriminant`
    return None;
}

#[must_use]
fn update_with_assign<'tcx>(
    tcx: TyCtxt<'tcx>,
    assign: &Box<(Place<'tcx>, Rvalue<'tcx>)>,
    src: &mut Option<Place<'tcx>>,
    dest: &mut Option<Place<'tcx>>,
    variant: &mut Option<VariantIdx>,
    variant_sym: &mut Option<Symbol>,
    move_mode: &mut Option<MoveMode>,
    stmt_types: &mut Vec<StatementType>,
    bundles: &mut FxHashMap<Field, BundleData<'tcx>>,
) -> Option<()> {
    let lhs = assign.0;
    let Rvalue::Use(rhs_op) = &assign.1 else { return None };

    let (mm, rhs) = match_operand(tcx, rhs_op)?;
    let (src_field, bundle, mut stmt_type) = match rhs {
        TempOrVarField::VarField(place, var_idx, _, field, field_ty) => {
            // We are assigning *from* `field`. First, check that we are using the right
            // `src`, `variant`, and move mode.
            set_or_check(src, place)?;
            set_or_check(variant, var_idx)?;
            set_or_check(move_mode, mm)?;

            let new_bundle =
                BundleData { temps: SmallVec::new(), complete: false, move_mode: mm, ty: field_ty };
            let Entry::Vacant(entry) = bundles.entry(field) else {
                // We had assigned from `field` already
                return None;
            };
            let bundle = entry.insert(new_bundle);
            let stmt_type = if mm == MoveMode::Move {
                StatementType::AssignDrop
            } else {
                StatementType::AssignForceCopy
            };

            (field, bundle, stmt_type)
        }
        TempOrVarField::Temp(local) => {
            // We are assigning out of a `local`. It should be the most recently
            // assigned local of some bundle.
            let (field, bundle) = bundles
                .iter_mut()
                .filter(|(_, b)| !b.complete)
                .find(|(_, b)| b.temps.last().map(|(l, _)| *l) == Some(local))?;
            let statement_type = if bundle.move_mode == MoveMode::Move {
                if mm == MoveMode::Copy {
                    // We need to update this bundle to use `copy` move mode
                    bundle.move_mode = MoveMode::Copy;
                    stmt_types[bundle.temps[0].1] = StatementType::AssignForceCopy;
                    for (_, i) in bundle.temps[1..].iter() {
                        stmt_types[*i] = StatementType::AssignPreserve;
                    }
                    StatementType::AssignPreserve
                } else {
                    StatementType::AssignDrop
                }
            } else {
                StatementType::AssignPreserve
            };
            (*field, bundle, statement_type)
        }
    };

    let lhs = match_place(tcx, lhs)?;
    match lhs {
        TempOrVarField::VarField(place, var_idx, var_sym, dest_field, _) => {
            if *variant_sym == None {
                *variant_sym = var_sym;
            }
            set_or_check(dest, place)?;
            set_or_check(variant, var_idx)?;
            if dest_field != src_field {
                return None;
            }
            bundle.complete = true;
            stmt_type = StatementType::AssignDrop;
        }
        TempOrVarField::Temp(local) => {
            bundle.temps.push((local, stmt_types.len()));
        }
    };
    stmt_types.push(stmt_type);
    Some(())
}

/// If `opt` is `None`, sets it to `val`, then checks if the value in `opt` is the same as
/// `val`, returning `None` if not.
#[must_use]
fn set_or_check<T: Eq>(opt: &mut Option<T>, val: T) -> Option<()> {
    match opt {
        Some(o) => {
            if o == &val {
                Some(())
            } else {
                None
            }
        }
        None => {
            *opt = Some(val);
            Some(())
        }
    }
}

enum TempOrVarField<'tcx> {
    Temp(Local),
    VarField(Place<'tcx>, VariantIdx, Option<Symbol>, Field, Ty<'tcx>),
}

fn match_operand<'tcx>(
    tcx: TyCtxt<'tcx>,
    op: &Operand<'tcx>,
) -> Option<(MoveMode, TempOrVarField<'tcx>)> {
    let (mm, place) = match op {
        Operand::Move(p) => (MoveMode::Move, p),
        Operand::Copy(p) => (MoveMode::Copy, p),
        Operand::Constant(_) => return None,
    };
    match_place(tcx, *place).map(|x| (mm, x))
}

fn match_place<'tcx>(tcx: TyCtxt<'tcx>, place: Place<'tcx>) -> Option<TempOrVarField<'tcx>> {
    let len = place.projection.len();
    if len == 0 {
        return Some(TempOrVarField::Temp(place.local));
    }
    if len == 1 {
        return None;
    }
    let ProjectionElem::Field(f, f_ty) = place.projection[len - 1] else { return None; };
    let ProjectionElem::Downcast(var_sym, var) = place.projection[len - 2] else { return None; };
    let p = Place {
        local: place.local,
        projection: tcx.intern_place_elems(&place.projection[..len - 2]),
    };
    return Some(TempOrVarField::VarField(p, var, var_sym, f, f_ty));
}

impl<'tcx> MirPass<'tcx> for SimplifyArmIdentity {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let param_env = tcx.param_env(body.source.instance.def_id());
        let (bbs, local_decls, debug_info) = body.basic_blocks_local_decls_mut_and_var_debug_info();
        let mut cleanup = false;
        for bb in bbs {
            if let Some(info) = get_arm_identity_info(tcx, &bb.statements, local_decls, param_env) {
                cleanup = true;
                // Apply the optimization as described above. We will collect the statements in the
                // bb into three groups.
                let mut storage_live = Vec::new();
                let mut storage_dead = Vec::new();
                let mut assigns = Vec::new();

                for (mut stmt, stmt_type) in
                    bb.statements.drain(0..info.stmts.len()).zip(info.stmts)
                {
                    match stmt_type {
                        StatementType::StorageLive => storage_live.push(stmt),
                        StatementType::StorageDead => storage_dead.push(stmt),
                        StatementType::AssignPreserve => assigns.push(stmt),
                        StatementType::AssignForceCopy => {
                            let StatementKind::Assign(k) = &mut stmt.kind else { unreachable!() };
                            let Rvalue::Use(op) = &mut k.1 else { unreachable!() };
                            *op = op.to_copy();
                            assigns.push(stmt);
                        }
                        StatementType::AssignDrop => (),
                    }
                }

                // The zeroth statement in `bb.statements` is now the `SetDiscriminant`
                let mut new_statements = storage_live;
                new_statements.extend(assigns);
                // Construct the optimized statement to add in
                let optimized = StatementKind::Assign(Box::new((
                    info.dest,
                    Rvalue::Use(match info.move_mode {
                        MoveMode::Move => Operand::Move(info.src),
                        MoveMode::Copy => Operand::Copy(info.src),
                    }),
                )));
                new_statements
                    .push(Statement { source_info: bb.statements[0].source_info, kind: optimized });
                new_statements.extend(storage_dead);
                new_statements.extend(bb.statements.drain(1..));

                bb.statements = new_statements;

                // Finally, update debug info
                let mut projs = info.dest.projection.to_vec();
                projs.push(ProjectionElem::Downcast(info.variant_sym, info.variant));
                let base_len = projs.len();
                for debug_info in debug_info.iter_mut() {
                    let VarDebugInfoContents::Place(place) = &mut debug_info.value else { continue; };
                    let Some(&(field, field_ty)) = info.temps.get(&place.local) else { continue; };
                    projs.push(ProjectionElem::Field(field, field_ty));
                    projs.extend(place.projection);
                    place.local = info.dest.local;
                    place.projection = tcx.intern_place_elems(&projs);
                    projs.truncate(base_len);
                }
            }
        }
        if cleanup {
            crate::simplify::simplify_locals(body, tcx);
        }
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
