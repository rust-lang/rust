//! MIR borrow checker, which is used in diagnostics like `unused_mut`

// Currently it is an ad-hoc implementation, only useful for mutability analysis. Feel free to remove all of these
// if needed for implementing a proper borrow checker.

use std::iter;

use hir_def::{DefWithBodyId, HasModule};
use la_arena::ArenaMap;
use stdx::never;
use triomphe::Arc;

use crate::{
    db::HirDatabase, mir::Operand, utils::ClosureSubst, ClosureId, Interner, Ty, TyExt, TypeFlags,
};

use super::{
    BasicBlockId, BorrowKind, LocalId, MirBody, MirLowerError, MirSpan, Place, ProjectionElem,
    Rvalue, StatementKind, TerminatorKind,
};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Stores spans which implies that the local should be mutable.
pub enum MutabilityReason {
    Mut { spans: Vec<MirSpan> },
    Not,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MovedOutOfRef {
    pub ty: Ty,
    pub span: MirSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorrowckResult {
    pub mir_body: Arc<MirBody>,
    pub mutability_of_locals: ArenaMap<LocalId, MutabilityReason>,
    pub moved_out_of_ref: Vec<MovedOutOfRef>,
}

fn all_mir_bodies(
    db: &dyn HirDatabase,
    def: DefWithBodyId,
) -> Box<dyn Iterator<Item = Result<Arc<MirBody>, MirLowerError>> + '_> {
    fn for_closure(
        db: &dyn HirDatabase,
        c: ClosureId,
    ) -> Box<dyn Iterator<Item = Result<Arc<MirBody>, MirLowerError>> + '_> {
        match db.mir_body_for_closure(c) {
            Ok(body) => {
                let closures = body.closures.clone();
                Box::new(
                    iter::once(Ok(body))
                        .chain(closures.into_iter().flat_map(|x| for_closure(db, x))),
                )
            }
            Err(e) => Box::new(iter::once(Err(e))),
        }
    }
    match db.mir_body(def) {
        Ok(body) => {
            let closures = body.closures.clone();
            Box::new(
                iter::once(Ok(body)).chain(closures.into_iter().flat_map(|x| for_closure(db, x))),
            )
        }
        Err(e) => Box::new(iter::once(Err(e))),
    }
}

pub fn borrowck_query(
    db: &dyn HirDatabase,
    def: DefWithBodyId,
) -> Result<Arc<[BorrowckResult]>, MirLowerError> {
    let _p = profile::span("borrowck_query");
    let r = all_mir_bodies(db, def)
        .map(|body| {
            let body = body?;
            Ok(BorrowckResult {
                mutability_of_locals: mutability_of_locals(db, &body),
                moved_out_of_ref: moved_out_of_ref(db, &body),
                mir_body: body,
            })
        })
        .collect::<Result<Vec<_>, MirLowerError>>()?;
    Ok(r.into())
}

fn moved_out_of_ref(db: &dyn HirDatabase, body: &MirBody) -> Vec<MovedOutOfRef> {
    let mut result = vec![];
    let mut for_operand = |op: &Operand, span: MirSpan| match op {
        Operand::Copy(p) | Operand::Move(p) => {
            let mut ty: Ty = body.locals[p.local].ty.clone();
            let mut is_dereference_of_ref = false;
            for proj in &*p.projection {
                if *proj == ProjectionElem::Deref && ty.as_reference().is_some() {
                    is_dereference_of_ref = true;
                }
                ty = proj.projected_ty(
                    ty,
                    db,
                    |c, subst, f| {
                        let (def, _) = db.lookup_intern_closure(c.into());
                        let infer = db.infer(def);
                        let (captures, _) = infer.closure_info(&c);
                        let parent_subst = ClosureSubst(subst).parent_subst();
                        captures
                            .get(f)
                            .expect("broken closure field")
                            .ty
                            .clone()
                            .substitute(Interner, parent_subst)
                    },
                    body.owner.module(db.upcast()).krate(),
                );
            }
            if is_dereference_of_ref
                && !ty.clone().is_copy(db, body.owner)
                && !ty.data(Interner).flags.intersects(TypeFlags::HAS_ERROR)
            {
                result.push(MovedOutOfRef { span, ty });
            }
        }
        Operand::Constant(_) | Operand::Static(_) => (),
    };
    for (_, block) in body.basic_blocks.iter() {
        for statement in &block.statements {
            match &statement.kind {
                StatementKind::Assign(_, r) => match r {
                    Rvalue::ShallowInitBoxWithAlloc(_) => (),
                    Rvalue::ShallowInitBox(o, _)
                    | Rvalue::UnaryOp(_, o)
                    | Rvalue::Cast(_, o, _)
                    | Rvalue::Repeat(o, _)
                    | Rvalue::Use(o) => for_operand(o, statement.span),
                    Rvalue::CopyForDeref(_)
                    | Rvalue::Discriminant(_)
                    | Rvalue::Len(_)
                    | Rvalue::Ref(_, _) => (),
                    Rvalue::CheckedBinaryOp(_, o1, o2) => {
                        for_operand(o1, statement.span);
                        for_operand(o2, statement.span);
                    }
                    Rvalue::Aggregate(_, ops) => {
                        for op in ops.iter() {
                            for_operand(op, statement.span);
                        }
                    }
                },
                StatementKind::Deinit(_)
                | StatementKind::StorageLive(_)
                | StatementKind::StorageDead(_)
                | StatementKind::Nop => (),
            }
        }
        match &block.terminator {
            Some(terminator) => match &terminator.kind {
                TerminatorKind::SwitchInt { discr, .. } => for_operand(discr, terminator.span),
                TerminatorKind::FalseEdge { .. }
                | TerminatorKind::FalseUnwind { .. }
                | TerminatorKind::Goto { .. }
                | TerminatorKind::Resume
                | TerminatorKind::GeneratorDrop
                | TerminatorKind::Abort
                | TerminatorKind::Return
                | TerminatorKind::Unreachable
                | TerminatorKind::Drop { .. } => (),
                TerminatorKind::DropAndReplace { value, .. } => {
                    for_operand(value, terminator.span);
                }
                TerminatorKind::Call { func, args, .. } => {
                    for_operand(func, terminator.span);
                    args.iter().for_each(|x| for_operand(x, terminator.span));
                }
                TerminatorKind::Assert { cond, .. } => {
                    for_operand(cond, terminator.span);
                }
                TerminatorKind::Yield { value, .. } => {
                    for_operand(value, terminator.span);
                }
            },
            None => (),
        }
    }
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectionCase {
    /// Projection is a local
    Direct,
    /// Projection is some field or slice of a local
    DirectPart,
    /// Projection is deref of something
    Indirect,
}

fn place_case(db: &dyn HirDatabase, body: &MirBody, lvalue: &Place) -> ProjectionCase {
    let mut is_part_of = false;
    let mut ty = body.locals[lvalue.local].ty.clone();
    for proj in lvalue.projection.iter() {
        match proj {
            ProjectionElem::Deref if ty.as_adt().is_none() => return ProjectionCase::Indirect, // It's indirect in case of reference and raw
            ProjectionElem::Deref // It's direct in case of `Box<T>`
            | ProjectionElem::ConstantIndex { .. }
            | ProjectionElem::Subslice { .. }
            | ProjectionElem::Field(_)
            | ProjectionElem::TupleOrClosureField(_)
            | ProjectionElem::Index(_) => {
                is_part_of = true;
            }
            ProjectionElem::OpaqueCast(_) => (),
        }
        ty = proj.projected_ty(
            ty,
            db,
            |c, subst, f| {
                let (def, _) = db.lookup_intern_closure(c.into());
                let infer = db.infer(def);
                let (captures, _) = infer.closure_info(&c);
                let parent_subst = ClosureSubst(subst).parent_subst();
                captures
                    .get(f)
                    .expect("broken closure field")
                    .ty
                    .clone()
                    .substitute(Interner, parent_subst)
            },
            body.owner.module(db.upcast()).krate(),
        );
    }
    if is_part_of {
        ProjectionCase::DirectPart
    } else {
        ProjectionCase::Direct
    }
}

/// Returns a map from basic blocks to the set of locals that might be ever initialized before
/// the start of the block. Only `StorageDead` can remove something from this map, and we ignore
/// `Uninit` and `drop` and similar after initialization.
fn ever_initialized_map(body: &MirBody) -> ArenaMap<BasicBlockId, ArenaMap<LocalId, bool>> {
    let mut result: ArenaMap<BasicBlockId, ArenaMap<LocalId, bool>> =
        body.basic_blocks.iter().map(|x| (x.0, ArenaMap::default())).collect();
    fn dfs(
        body: &MirBody,
        b: BasicBlockId,
        l: LocalId,
        result: &mut ArenaMap<BasicBlockId, ArenaMap<LocalId, bool>>,
    ) {
        let mut is_ever_initialized = result[b][l]; // It must be filled, as we use it as mark for dfs
        let block = &body.basic_blocks[b];
        for statement in &block.statements {
            match &statement.kind {
                StatementKind::Assign(p, _) => {
                    if p.projection.len() == 0 && p.local == l {
                        is_ever_initialized = true;
                    }
                }
                StatementKind::StorageDead(p) => {
                    if *p == l {
                        is_ever_initialized = false;
                    }
                }
                StatementKind::Deinit(_) | StatementKind::Nop | StatementKind::StorageLive(_) => (),
            }
        }
        let Some(terminator) = &block.terminator else {
            never!("Terminator should be none only in construction");
            return;
        };
        let targets = match &terminator.kind {
            TerminatorKind::Goto { target } => vec![*target],
            TerminatorKind::SwitchInt { targets, .. } => targets.all_targets().to_vec(),
            TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::Unreachable => vec![],
            TerminatorKind::Call { target, cleanup, destination, .. } => {
                if destination.projection.len() == 0 && destination.local == l {
                    is_ever_initialized = true;
                }
                target.into_iter().chain(cleanup.into_iter()).copied().collect()
            }
            TerminatorKind::Drop { target, unwind, place: _ } => {
                Some(target).into_iter().chain(unwind.into_iter()).copied().collect()
            }
            TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Yield { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. } => {
                never!("We don't emit these MIR terminators yet");
                vec![]
            }
        };
        for target in targets {
            if !result[target].contains_idx(l) || !result[target][l] && is_ever_initialized {
                result[target].insert(l, is_ever_initialized);
                dfs(body, target, l, result);
            }
        }
    }
    for &l in &body.param_locals {
        result[body.start_block].insert(l, true);
        dfs(body, body.start_block, l, &mut result);
    }
    for l in body.locals.iter().map(|x| x.0) {
        if !result[body.start_block].contains_idx(l) {
            result[body.start_block].insert(l, false);
            dfs(body, body.start_block, l, &mut result);
        }
    }
    result
}

fn mutability_of_locals(
    db: &dyn HirDatabase,
    body: &MirBody,
) -> ArenaMap<LocalId, MutabilityReason> {
    let mut result: ArenaMap<LocalId, MutabilityReason> =
        body.locals.iter().map(|x| (x.0, MutabilityReason::Not)).collect();
    let mut push_mut_span = |local, span| match &mut result[local] {
        MutabilityReason::Mut { spans } => spans.push(span),
        x @ MutabilityReason::Not => *x = MutabilityReason::Mut { spans: vec![span] },
    };
    let ever_init_maps = ever_initialized_map(body);
    for (block_id, mut ever_init_map) in ever_init_maps.into_iter() {
        let block = &body.basic_blocks[block_id];
        for statement in &block.statements {
            match &statement.kind {
                StatementKind::Assign(place, value) => {
                    match place_case(db, body, place) {
                        ProjectionCase::Direct => {
                            if ever_init_map.get(place.local).copied().unwrap_or_default() {
                                push_mut_span(place.local, statement.span);
                            } else {
                                ever_init_map.insert(place.local, true);
                            }
                        }
                        ProjectionCase::DirectPart => {
                            // Partial initialization is not supported, so it is definitely `mut`
                            push_mut_span(place.local, statement.span);
                        }
                        ProjectionCase::Indirect => (),
                    }
                    if let Rvalue::Ref(BorrowKind::Mut { .. }, p) = value {
                        if place_case(db, body, p) != ProjectionCase::Indirect {
                            push_mut_span(p.local, statement.span);
                        }
                    }
                }
                StatementKind::StorageDead(p) => {
                    ever_init_map.insert(*p, false);
                }
                StatementKind::Deinit(_) | StatementKind::StorageLive(_) | StatementKind::Nop => (),
            }
        }
        let Some(terminator) = &block.terminator else {
            never!("Terminator should be none only in construction");
            continue;
        };
        match &terminator.kind {
            TerminatorKind::Goto { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Drop { .. }
            | TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Yield { .. } => (),
            TerminatorKind::Call { destination, .. } => {
                if destination.projection.len() == 0 {
                    if ever_init_map.get(destination.local).copied().unwrap_or_default() {
                        push_mut_span(destination.local, MirSpan::Unknown);
                    } else {
                        ever_init_map.insert(destination.local, true);
                    }
                }
            }
        }
    }
    result
}
