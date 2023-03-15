//! MIR borrow checker, which is used in diagnostics like `unused_mut`

// Currently it is an ad-hoc implementation, only useful for mutability analysis. Feel free to remove all of these
// if needed for implementing a proper borrow checker.

use std::sync::Arc;

use hir_def::DefWithBodyId;
use la_arena::ArenaMap;
use stdx::never;

use crate::db::HirDatabase;

use super::{
    BasicBlockId, BorrowKind, LocalId, MirBody, MirLowerError, MirSpan, Place, ProjectionElem,
    Rvalue, StatementKind, Terminator,
};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Stores spans which implies that the local should be mutable.
pub enum MutabilityReason {
    Mut { spans: Vec<MirSpan> },
    Not,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorrowckResult {
    pub mir_body: Arc<MirBody>,
    pub mutability_of_locals: ArenaMap<LocalId, MutabilityReason>,
}

pub fn borrowck_query(
    db: &dyn HirDatabase,
    def: DefWithBodyId,
) -> Result<Arc<BorrowckResult>, MirLowerError> {
    let _p = profile::span("borrowck_query");
    let body = db.mir_body(def)?;
    let r = BorrowckResult { mutability_of_locals: mutability_of_locals(&body), mir_body: body };
    Ok(Arc::new(r))
}

fn is_place_direct(lvalue: &Place) -> bool {
    !lvalue.projection.iter().any(|x| *x == ProjectionElem::Deref)
}

enum ProjectionCase {
    /// Projection is a local
    Direct,
    /// Projection is some field or slice of a local
    DirectPart,
    /// Projection is deref of something
    Indirect,
}

fn place_case(lvalue: &Place) -> ProjectionCase {
    let mut is_part_of = false;
    for proj in lvalue.projection.iter().rev() {
        match proj {
            ProjectionElem::Deref => return ProjectionCase::Indirect, // It's indirect
            ProjectionElem::ConstantIndex { .. }
            | ProjectionElem::Subslice { .. }
            | ProjectionElem::Field(_)
            | ProjectionElem::TupleField(_)
            | ProjectionElem::Index(_) => {
                is_part_of = true;
            }
            ProjectionElem::OpaqueCast(_) => (),
        }
    }
    if is_part_of {
        ProjectionCase::DirectPart
    } else {
        ProjectionCase::Direct
    }
}

/// Returns a map from basic blocks to the set of locals that might be ever initialized before
/// the start of the block. Only `StorageDead` can remove something from this map, and we ignore
/// `Uninit` and `drop` and similars after initialization.
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
        let targets = match terminator {
            Terminator::Goto { target } => vec![*target],
            Terminator::SwitchInt { targets, .. } => targets.all_targets().to_vec(),
            Terminator::Resume
            | Terminator::Abort
            | Terminator::Return
            | Terminator::Unreachable => vec![],
            Terminator::Call { target, cleanup, destination, .. } => {
                if destination.projection.len() == 0 && destination.local == l {
                    is_ever_initialized = true;
                }
                target.into_iter().chain(cleanup.into_iter()).copied().collect()
            }
            Terminator::Drop { .. }
            | Terminator::DropAndReplace { .. }
            | Terminator::Assert { .. }
            | Terminator::Yield { .. }
            | Terminator::GeneratorDrop
            | Terminator::FalseEdge { .. }
            | Terminator::FalseUnwind { .. } => {
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

fn mutability_of_locals(body: &MirBody) -> ArenaMap<LocalId, MutabilityReason> {
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
                    match place_case(place) {
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
                        if is_place_direct(p) {
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
        match terminator {
            Terminator::Goto { .. }
            | Terminator::Resume
            | Terminator::Abort
            | Terminator::Return
            | Terminator::Unreachable
            | Terminator::FalseEdge { .. }
            | Terminator::FalseUnwind { .. }
            | Terminator::GeneratorDrop
            | Terminator::SwitchInt { .. }
            | Terminator::Drop { .. }
            | Terminator::DropAndReplace { .. }
            | Terminator::Assert { .. }
            | Terminator::Yield { .. } => (),
            Terminator::Call { destination, .. } => {
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
