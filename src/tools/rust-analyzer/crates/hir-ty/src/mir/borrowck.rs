//! MIR borrow checker, which is used in diagnostics like `unused_mut`

// Currently it is an ad-hoc implementation, only useful for mutability analysis. Feel free to remove all of these
// if needed for implementing a proper borrow checker.

use std::iter;

use hir_def::{DefWithBodyId, HasModule};
use la_arena::ArenaMap;
use rustc_hash::FxHashMap;
use stdx::never;
use triomphe::Arc;

use crate::{
    TraitEnvironment,
    db::{HirDatabase, InternedClosure, InternedClosureId},
    display::DisplayTarget,
    mir::OperandKind,
    next_solver::{
        DbInterner, GenericArgs, Ty, TypingMode,
        infer::{DbInternerInferExt, InferCtxt},
    },
};

use super::{
    BasicBlockId, BorrowKind, LocalId, MirBody, MirLowerError, MirSpan, MutBorrowKind, Operand,
    Place, ProjectionElem, Rvalue, StatementKind, TerminatorKind,
};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Stores spans which implies that the local should be mutable.
pub enum MutabilityReason {
    Mut { spans: Vec<MirSpan> },
    Not,
    Unused,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MovedOutOfRef<'db> {
    pub ty: Ty<'db>,
    pub span: MirSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartiallyMoved<'db> {
    pub ty: Ty<'db>,
    pub span: MirSpan,
    pub local: LocalId<'db>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorrowRegion<'db> {
    pub local: LocalId<'db>,
    pub kind: BorrowKind,
    pub places: Vec<MirSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorrowckResult<'db> {
    pub mir_body: Arc<MirBody<'db>>,
    pub mutability_of_locals: ArenaMap<LocalId<'db>, MutabilityReason>,
    pub moved_out_of_ref: Vec<MovedOutOfRef<'db>>,
    pub partially_moved: Vec<PartiallyMoved<'db>>,
    pub borrow_regions: Vec<BorrowRegion<'db>>,
}

fn all_mir_bodies<'db>(
    db: &'db dyn HirDatabase,
    def: DefWithBodyId,
    mut cb: impl FnMut(Arc<MirBody<'db>>),
) -> Result<(), MirLowerError<'db>> {
    fn for_closure<'db>(
        db: &'db dyn HirDatabase,
        c: InternedClosureId,
        cb: &mut impl FnMut(Arc<MirBody<'db>>),
    ) -> Result<(), MirLowerError<'db>> {
        match db.mir_body_for_closure(c) {
            Ok(body) => {
                cb(body.clone());
                body.closures.iter().try_for_each(|&it| for_closure(db, it, cb))
            }
            Err(e) => Err(e),
        }
    }
    match db.mir_body(def) {
        Ok(body) => {
            cb(body.clone());
            body.closures.iter().try_for_each(|&it| for_closure(db, it, &mut cb))
        }
        Err(e) => Err(e),
    }
}

pub fn borrowck_query<'db>(
    db: &'db dyn HirDatabase,
    def: DefWithBodyId,
) -> Result<Arc<[BorrowckResult<'db>]>, MirLowerError<'db>> {
    let _p = tracing::info_span!("borrowck_query").entered();
    let module = def.module(db);
    let interner = DbInterner::new_with(db, Some(module.krate()), module.containing_block());
    let env = db.trait_environment_for_body(def);
    let mut res = vec![];
    // This calculates opaques defining scope which is a bit costly therefore is put outside `all_mir_bodies()`.
    let typing_mode = TypingMode::borrowck(interner, def.into());
    all_mir_bodies(db, def, |body| {
        // FIXME(next-solver): Opaques.
        let infcx = interner.infer_ctxt().build(typing_mode);
        res.push(BorrowckResult {
            mutability_of_locals: mutability_of_locals(&infcx, &body),
            moved_out_of_ref: moved_out_of_ref(&infcx, &env, &body),
            partially_moved: partially_moved(&infcx, &env, &body),
            borrow_regions: borrow_regions(db, &body),
            mir_body: body,
        });
    })?;
    Ok(res.into())
}

fn make_fetch_closure_field<'db>(
    db: &'db dyn HirDatabase,
) -> impl FnOnce(InternedClosureId, GenericArgs<'db>, usize) -> Ty<'db> + use<'db> {
    |c: InternedClosureId, subst: GenericArgs<'db>, f: usize| {
        let InternedClosure(def, _) = db.lookup_intern_closure(c);
        let infer = db.infer(def);
        let (captures, _) = infer.closure_info(c);
        let parent_subst = subst.split_closure_args_untupled().parent_args;
        let interner = DbInterner::new_with(db, None, None);
        captures.get(f).expect("broken closure field").ty.instantiate(interner, parent_subst)
    }
}

fn moved_out_of_ref<'db>(
    infcx: &InferCtxt<'db>,
    env: &TraitEnvironment<'db>,
    body: &MirBody<'db>,
) -> Vec<MovedOutOfRef<'db>> {
    let db = infcx.interner.db;
    let mut result = vec![];
    let mut for_operand = |op: &Operand<'db>, span: MirSpan| match op.kind {
        OperandKind::Copy(p) | OperandKind::Move(p) => {
            let mut ty: Ty<'db> = body.locals[p.local].ty;
            let mut is_dereference_of_ref = false;
            for proj in p.projection.lookup(&body.projection_store) {
                if *proj == ProjectionElem::Deref && ty.as_reference().is_some() {
                    is_dereference_of_ref = true;
                }
                ty = proj.projected_ty(
                    infcx,
                    ty,
                    make_fetch_closure_field(db),
                    body.owner.module(db).krate(),
                );
            }
            if is_dereference_of_ref
                && !infcx.type_is_copy_modulo_regions(env.env, ty)
                && !ty.references_non_lt_error()
            {
                result.push(MovedOutOfRef { span: op.span.unwrap_or(span), ty });
            }
        }
        OperandKind::Constant { .. } | OperandKind::Static(_) => (),
    };
    for (_, block) in body.basic_blocks.iter() {
        db.unwind_if_revision_cancelled();
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
                    Rvalue::ThreadLocalRef(n)
                    | Rvalue::AddressOf(n)
                    | Rvalue::BinaryOp(n)
                    | Rvalue::NullaryOp(n) => match *n {},
                },
                StatementKind::FakeRead(_)
                | StatementKind::Deinit(_)
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
                | TerminatorKind::UnwindResume
                | TerminatorKind::CoroutineDrop
                | TerminatorKind::Abort
                | TerminatorKind::Return
                | TerminatorKind::Unreachable
                | TerminatorKind::Drop { .. } => (),
                TerminatorKind::DropAndReplace { value, .. } => {
                    for_operand(value, terminator.span);
                }
                TerminatorKind::Call { func, args, .. } => {
                    for_operand(func, terminator.span);
                    args.iter().for_each(|it| for_operand(it, terminator.span));
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
    result.shrink_to_fit();
    result
}

fn partially_moved<'db>(
    infcx: &InferCtxt<'db>,
    env: &TraitEnvironment<'db>,
    body: &MirBody<'db>,
) -> Vec<PartiallyMoved<'db>> {
    let db = infcx.interner.db;
    let mut result = vec![];
    let mut for_operand = |op: &Operand<'db>, span: MirSpan| match op.kind {
        OperandKind::Copy(p) | OperandKind::Move(p) => {
            let mut ty: Ty<'db> = body.locals[p.local].ty;
            for proj in p.projection.lookup(&body.projection_store) {
                ty = proj.projected_ty(
                    infcx,
                    ty,
                    make_fetch_closure_field(db),
                    body.owner.module(db).krate(),
                );
            }
            if !infcx.type_is_copy_modulo_regions(env.env, ty) && !ty.references_non_lt_error() {
                result.push(PartiallyMoved { span, ty, local: p.local });
            }
        }
        OperandKind::Constant { .. } | OperandKind::Static(_) => (),
    };
    for (_, block) in body.basic_blocks.iter() {
        db.unwind_if_revision_cancelled();
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
                    Rvalue::ThreadLocalRef(n)
                    | Rvalue::AddressOf(n)
                    | Rvalue::BinaryOp(n)
                    | Rvalue::NullaryOp(n) => match *n {},
                },
                StatementKind::FakeRead(_)
                | StatementKind::Deinit(_)
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
                | TerminatorKind::UnwindResume
                | TerminatorKind::CoroutineDrop
                | TerminatorKind::Abort
                | TerminatorKind::Return
                | TerminatorKind::Unreachable
                | TerminatorKind::Drop { .. } => (),
                TerminatorKind::DropAndReplace { value, .. } => {
                    for_operand(value, terminator.span);
                }
                TerminatorKind::Call { func, args, .. } => {
                    for_operand(func, terminator.span);
                    args.iter().for_each(|it| for_operand(it, terminator.span));
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
    result.shrink_to_fit();
    result
}

fn borrow_regions<'db>(db: &'db dyn HirDatabase, body: &MirBody<'db>) -> Vec<BorrowRegion<'db>> {
    let mut borrows = FxHashMap::default();
    for (_, block) in body.basic_blocks.iter() {
        db.unwind_if_revision_cancelled();
        for statement in &block.statements {
            if let StatementKind::Assign(_, Rvalue::Ref(kind, p)) = &statement.kind {
                borrows
                    .entry(p.local)
                    .and_modify(|it: &mut BorrowRegion<'db>| {
                        it.places.push(statement.span);
                    })
                    .or_insert_with(|| BorrowRegion {
                        local: p.local,
                        kind: *kind,
                        places: vec![statement.span],
                    });
            }
        }
        match &block.terminator {
            Some(terminator) => match &terminator.kind {
                TerminatorKind::FalseEdge { .. }
                | TerminatorKind::FalseUnwind { .. }
                | TerminatorKind::Goto { .. }
                | TerminatorKind::UnwindResume
                | TerminatorKind::CoroutineDrop
                | TerminatorKind::Abort
                | TerminatorKind::Return
                | TerminatorKind::Unreachable
                | TerminatorKind::Drop { .. } => (),
                TerminatorKind::DropAndReplace { .. } => {}
                TerminatorKind::Call { .. } => {}
                _ => (),
            },
            None => (),
        }
    }

    borrows.into_values().collect()
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

fn place_case<'db>(
    infcx: &InferCtxt<'db>,
    body: &MirBody<'db>,
    lvalue: &Place<'db>,
) -> ProjectionCase {
    let db = infcx.interner.db;
    let mut is_part_of = false;
    let mut ty = body.locals[lvalue.local].ty;
    for proj in lvalue.projection.lookup(&body.projection_store).iter() {
        match proj {
            ProjectionElem::Deref if ty.as_adt().is_none() => return ProjectionCase::Indirect, // It's indirect in case of reference and raw
            ProjectionElem::Deref // It's direct in case of `Box<T>`
            | ProjectionElem::ConstantIndex { .. }
            | ProjectionElem::Subslice { .. }
            | ProjectionElem::Field(_)
            | ProjectionElem::ClosureField(_)
            | ProjectionElem::Index(_) => {
                is_part_of = true;
            }
            ProjectionElem::OpaqueCast(_) => (),
        }
        ty = proj.projected_ty(
            infcx,
            ty,
            make_fetch_closure_field(db),
            body.owner.module(db).krate(),
        );
    }
    if is_part_of { ProjectionCase::DirectPart } else { ProjectionCase::Direct }
}

/// Returns a map from basic blocks to the set of locals that might be ever initialized before
/// the start of the block. Only `StorageDead` can remove something from this map, and we ignore
/// `Uninit` and `drop` and similar after initialization.
fn ever_initialized_map<'db>(
    db: &'db dyn HirDatabase,
    body: &MirBody<'db>,
) -> ArenaMap<BasicBlockId<'db>, ArenaMap<LocalId<'db>, bool>> {
    let mut result: ArenaMap<BasicBlockId<'db>, ArenaMap<LocalId<'db>, bool>> =
        body.basic_blocks.iter().map(|it| (it.0, ArenaMap::default())).collect();
    fn dfs<'db>(
        db: &'db dyn HirDatabase,
        body: &MirBody<'db>,
        l: LocalId<'db>,
        stack: &mut Vec<BasicBlockId<'db>>,
        result: &mut ArenaMap<BasicBlockId<'db>, ArenaMap<LocalId<'db>, bool>>,
    ) {
        while let Some(b) = stack.pop() {
            let mut is_ever_initialized = result[b][l]; // It must be filled, as we use it as mark for dfs
            let block = &body.basic_blocks[b];
            for statement in &block.statements {
                match &statement.kind {
                    StatementKind::Assign(p, _) => {
                        if p.projection.lookup(&body.projection_store).is_empty() && p.local == l {
                            is_ever_initialized = true;
                        }
                    }
                    StatementKind::StorageDead(p) => {
                        if *p == l {
                            is_ever_initialized = false;
                        }
                    }
                    StatementKind::Deinit(_)
                    | StatementKind::FakeRead(_)
                    | StatementKind::Nop
                    | StatementKind::StorageLive(_) => (),
                }
            }
            let Some(terminator) = &block.terminator else {
                never!(
                    "Terminator should be none only in construction.\nThe body:\n{}",
                    body.pretty_print(db, DisplayTarget::from_crate(db, body.owner.krate(db)))
                );
                return;
            };
            let mut process = |target, is_ever_initialized| {
                if !result[target].contains_idx(l) || !result[target][l] && is_ever_initialized {
                    result[target].insert(l, is_ever_initialized);
                    stack.push(target);
                }
            };
            match &terminator.kind {
                TerminatorKind::Goto { target } => process(*target, is_ever_initialized),
                TerminatorKind::SwitchInt { targets, .. } => {
                    targets.all_targets().iter().for_each(|&it| process(it, is_ever_initialized));
                }
                TerminatorKind::UnwindResume
                | TerminatorKind::Abort
                | TerminatorKind::Return
                | TerminatorKind::Unreachable => (),
                TerminatorKind::Call { target, cleanup, destination, .. } => {
                    if destination.projection.lookup(&body.projection_store).is_empty()
                        && destination.local == l
                    {
                        is_ever_initialized = true;
                    }
                    target.iter().chain(cleanup).for_each(|&it| process(it, is_ever_initialized));
                }
                TerminatorKind::Drop { target, unwind, place: _ } => {
                    iter::once(target)
                        .chain(unwind)
                        .for_each(|&it| process(it, is_ever_initialized));
                }
                TerminatorKind::DropAndReplace { .. }
                | TerminatorKind::Assert { .. }
                | TerminatorKind::Yield { .. }
                | TerminatorKind::CoroutineDrop
                | TerminatorKind::FalseEdge { .. }
                | TerminatorKind::FalseUnwind { .. } => {
                    never!("We don't emit these MIR terminators yet");
                }
            }
        }
    }
    let mut stack = Vec::new();
    for &l in &body.param_locals {
        result[body.start_block].insert(l, true);
        stack.clear();
        stack.push(body.start_block);
        dfs(db, body, l, &mut stack, &mut result);
    }
    for l in body.locals.iter().map(|it| it.0) {
        db.unwind_if_revision_cancelled();
        if !result[body.start_block].contains_idx(l) {
            result[body.start_block].insert(l, false);
            stack.clear();
            stack.push(body.start_block);
            dfs(db, body, l, &mut stack, &mut result);
        }
    }
    result
}

fn push_mut_span<'db>(
    local: LocalId<'db>,
    span: MirSpan,
    result: &mut ArenaMap<LocalId<'db>, MutabilityReason>,
) {
    match &mut result[local] {
        MutabilityReason::Mut { spans } => spans.push(span),
        it @ (MutabilityReason::Not | MutabilityReason::Unused) => {
            *it = MutabilityReason::Mut { spans: vec![span] }
        }
    };
}

fn record_usage<'db>(local: LocalId<'db>, result: &mut ArenaMap<LocalId<'db>, MutabilityReason>) {
    if let it @ MutabilityReason::Unused = &mut result[local] {
        *it = MutabilityReason::Not;
    };
}

fn record_usage_for_operand<'db>(
    arg: &Operand<'db>,
    result: &mut ArenaMap<LocalId<'db>, MutabilityReason>,
) {
    if let OperandKind::Copy(p) | OperandKind::Move(p) = arg.kind {
        record_usage(p.local, result);
    }
}

fn mutability_of_locals<'db>(
    infcx: &InferCtxt<'db>,
    body: &MirBody<'db>,
) -> ArenaMap<LocalId<'db>, MutabilityReason> {
    let db = infcx.interner.db;
    let mut result: ArenaMap<LocalId<'db>, MutabilityReason> =
        body.locals.iter().map(|it| (it.0, MutabilityReason::Unused)).collect();

    let ever_init_maps = ever_initialized_map(db, body);
    for (block_id, mut ever_init_map) in ever_init_maps.into_iter() {
        let block = &body.basic_blocks[block_id];
        for statement in &block.statements {
            match &statement.kind {
                StatementKind::Assign(place, value) => {
                    match place_case(infcx, body, place) {
                        ProjectionCase::Direct => {
                            if ever_init_map.get(place.local).copied().unwrap_or_default() {
                                push_mut_span(place.local, statement.span, &mut result);
                            } else {
                                ever_init_map.insert(place.local, true);
                            }
                        }
                        ProjectionCase::DirectPart => {
                            // Partial initialization is not supported, so it is definitely `mut`
                            push_mut_span(place.local, statement.span, &mut result);
                        }
                        ProjectionCase::Indirect => {
                            record_usage(place.local, &mut result);
                        }
                    }
                    match value {
                        Rvalue::CopyForDeref(p)
                        | Rvalue::Discriminant(p)
                        | Rvalue::Len(p)
                        | Rvalue::Ref(_, p) => {
                            record_usage(p.local, &mut result);
                        }
                        Rvalue::Use(o)
                        | Rvalue::Repeat(o, _)
                        | Rvalue::Cast(_, o, _)
                        | Rvalue::UnaryOp(_, o) => record_usage_for_operand(o, &mut result),
                        Rvalue::CheckedBinaryOp(_, o1, o2) => {
                            for o in [o1, o2] {
                                record_usage_for_operand(o, &mut result);
                            }
                        }
                        Rvalue::Aggregate(_, args) => {
                            for arg in args.iter() {
                                record_usage_for_operand(arg, &mut result);
                            }
                        }
                        Rvalue::ShallowInitBox(_, _) | Rvalue::ShallowInitBoxWithAlloc(_) => (),
                        Rvalue::ThreadLocalRef(n)
                        | Rvalue::AddressOf(n)
                        | Rvalue::BinaryOp(n)
                        | Rvalue::NullaryOp(n) => match *n {},
                    }
                    if let Rvalue::Ref(
                        BorrowKind::Mut {
                            kind: MutBorrowKind::Default | MutBorrowKind::TwoPhasedBorrow,
                        },
                        p,
                    ) = value
                        && place_case(infcx, body, p) != ProjectionCase::Indirect
                    {
                        push_mut_span(p.local, statement.span, &mut result);
                    }
                }
                StatementKind::FakeRead(p) => {
                    record_usage(p.local, &mut result);
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
            | TerminatorKind::UnwindResume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::Drop { .. }
            | TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Yield { .. } => (),
            TerminatorKind::SwitchInt { discr, targets: _ } => {
                record_usage_for_operand(discr, &mut result);
            }
            TerminatorKind::Call { destination, args, func, .. } => {
                record_usage_for_operand(func, &mut result);
                for arg in args.iter() {
                    record_usage_for_operand(arg, &mut result);
                }
                if destination.projection.lookup(&body.projection_store).is_empty() {
                    if ever_init_map.get(destination.local).copied().unwrap_or_default() {
                        push_mut_span(destination.local, terminator.span, &mut result);
                    } else {
                        ever_init_map.insert(destination.local, true);
                    }
                }
            }
        }
    }
    result
}
