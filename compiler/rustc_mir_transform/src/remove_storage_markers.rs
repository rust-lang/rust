//! This pass removes storage markers if they won't be emitted during codegen.

use crate::MirPass;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_middle::middle::resolve_bound_vars::Set1;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct RemoveStorageMarkers;

impl<'tcx> MirPass<'tcx> for RemoveStorageMarkers {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.emit_lifetime_markers() {
            remove_redundant_storage_markers(tcx, body);
            return;
        }

        trace!("Running RemoveStorageMarkers on {:?}", body.source);
        for data in body.basic_blocks.as_mut_preserves_cfg() {
            data.statements.retain(|statement| match statement.kind {
                StatementKind::StorageLive(..)
                | StatementKind::StorageDead(..)
                | StatementKind::Nop => false,
                _ => true,
            })
        }
    }
}

fn remove_redundant_storage_markers<'tcx>(_tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    // FIXME: First sift down any StorageDead possible to the return block

    let mut return_block = None;
    for (b, block) in body.basic_blocks.iter_enumerated() {
        if block.terminator().kind == TerminatorKind::Return {
            if return_block.is_some() {
                // Handling multiple return terminators is complicated
                return;
            }
            return_block = Some(b);
        }
    }
    let Some(return_block) = return_block else {
        return;
    };

    let mut visitor = LiveMapVisitor { map: FxHashMap::default() };
    visitor.visit_body(body);
    let liveness_map = visitor.map;

    let mut live = FxHashSet::default();
    for stmt in body.basic_blocks[START_BLOCK].statements.iter() {
        match &stmt.kind {
            StatementKind::StorageLive(local) => {
                live.insert(*local);
            }
            StatementKind::StorageDead(_) => {
                break;
            }
            _ => {}
        }
    }

    let mut dead = FxHashSet::default();
    for stmt in body.basic_blocks[return_block].statements.iter().rev() {
        match &stmt.kind {
            StatementKind::StorageDead(local) => {
                dead.insert(*local);
            }
            StatementKind::StorageLive(_) => {
                break;
            }
            _ => {}
        }
    }

    let mut did_optimization = false;
    for local in live.intersection(&dead) {
        if let (Set1::One(live_at), Set1::One(dead_at)) = liveness_map[local] {
            body.basic_blocks_mut()[live_at.block].statements[live_at.statement_index].make_nop();
            body.basic_blocks_mut()[dead_at.block].statements[dead_at.statement_index].make_nop();
            did_optimization = true;
        }
    }

    if did_optimization {
        body.basic_blocks_mut()[START_BLOCK]
            .statements
            .retain(|stmt| !matches!(stmt.kind, StatementKind::Nop));
        if START_BLOCK != return_block {
            body.basic_blocks_mut()[return_block]
                .statements
                .retain(|stmt| !matches!(stmt.kind, StatementKind::Nop));
        }
    }
}

struct LiveMapVisitor {
    map: FxHashMap<Local, (Set1<Location>, Set1<Location>)>,
}

impl<'tcx> Visitor<'tcx> for LiveMapVisitor {
    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match &statement.kind {
            StatementKind::StorageLive(local) => {
                self.map.entry(*local).or_insert((Set1::Empty, Set1::Empty)).0.insert(location);
            }
            StatementKind::StorageDead(local) => {
                self.map.entry(*local).or_insert((Set1::Empty, Set1::Empty)).1.insert(location);
            }
            _ => {}
        }

        self.super_statement(statement, location);
    }
}
