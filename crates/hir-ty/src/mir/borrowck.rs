//! MIR borrow checker, which is used in diagnostics like `unused_mut`

// Currently it is an ad-hoc implementation, only useful for mutability analysis. Feel free to remove all of these
// and implement a proper borrow checker.

use la_arena::ArenaMap;
use stdx::never;

use super::{
    BasicBlockId, BorrowKind, LocalId, MirBody, MirSpan, Place, ProjectionElem, Rvalue,
    StatementKind, Terminator,
};

#[derive(Debug)]
pub enum Mutability {
    Mut { span: MirSpan },
    Not,
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
    for (b, block) in body.basic_blocks.iter() {
        for statement in &block.statements {
            if let StatementKind::Assign(p, _) = &statement.kind {
                if p.projection.len() == 0 {
                    let l = p.local;
                    if !result[b].contains_idx(l) {
                        result[b].insert(l, false);
                        dfs(body, b, l, &mut result);
                    }
                }
            }
        }
    }
    result
}

pub fn mutability_of_locals(body: &MirBody) -> ArenaMap<LocalId, Mutability> {
    let mut result: ArenaMap<LocalId, Mutability> =
        body.locals.iter().map(|x| (x.0, Mutability::Not)).collect();
    let ever_init_maps = ever_initialized_map(body);
    for (block_id, ever_init_map) in ever_init_maps.iter() {
        let mut ever_init_map = ever_init_map.clone();
        let block = &body.basic_blocks[block_id];
        for statement in &block.statements {
            match &statement.kind {
                StatementKind::Assign(place, value) => {
                    match place_case(place) {
                        ProjectionCase::Direct => {
                            if ever_init_map.get(place.local).copied().unwrap_or_default() {
                                result[place.local] = Mutability::Mut { span: statement.span };
                            } else {
                                ever_init_map.insert(place.local, true);
                            }
                        }
                        ProjectionCase::DirectPart => {
                            // Partial initialization is not supported, so it is definitely `mut`
                            result[place.local] = Mutability::Mut { span: statement.span };
                        }
                        ProjectionCase::Indirect => (),
                    }
                    if let Rvalue::Ref(BorrowKind::Mut { .. }, p) = value {
                        if is_place_direct(p) {
                            result[p.local] = Mutability::Mut { span: statement.span };
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
                        result[destination.local] = Mutability::Mut { span: MirSpan::Unknown };
                    } else {
                        ever_init_map.insert(destination.local, true);
                    }
                }
            }
        }
    }
    result
}
