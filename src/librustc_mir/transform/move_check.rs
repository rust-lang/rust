// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dataflow::move_paths::{HasMoveData, MoveData};
use dataflow::{MaybeObservedLvals, LvalObservers};
use dataflow::state_for_location;
use dataflow;
use rustc::ty::TyCtxt;
use rustc::hir::def_id::DefId;
use rustc::mir::visit::Visitor;
use rustc::mir::{Mir, Lvalue, ProjectionElem};
use rustc_data_structures::indexed_set::IdxSetBuf;
use rustc::ty::maps::Providers;

fn deref_box<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                       mir: &Mir<'tcx>,
                       lvalue: &Lvalue<'tcx>) -> bool {
    match *lvalue {
        Lvalue::Local(..) | Lvalue::Static(..) => false,
        Lvalue::Projection(ref proj) => {
            match proj.elem {
                ProjectionElem::Deref => proj.base.ty(mir, tcx).to_ty(tcx).is_box(),
                _ => deref_box(tcx, mir, &proj.base)
            }
        }
    }
}

// FIXME: This needs to run before optimizations like SimplifyCfg? and SimplifyBranches
fn moveck<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) {
    // NB: this borrow is valid because all the consumers of
    // `mir_const` force this.
    let mir = &*tcx.mir_const(def_id).borrow();
    let id = tcx.hir.as_local_node_id(def_id).unwrap();
    let param_env = tcx.param_env(def_id);
    let move_data = match MoveData::gather_moves(mir, tcx, param_env, true) {
        Ok(move_data) => move_data,
        // Ignore move errors, borrowck will report those
        Err((move_data, _)) => move_data,
    };
    // If there are no moves of immovables types, there is nothing to check.
    if move_data.moves.is_empty() {
        return;
    }

    let dead_unwinds = IdxSetBuf::new_empty(mir.basic_blocks().len());
    let analysis = MaybeObservedLvals::new(tcx, mir, &move_data);
    let observed =
        dataflow::do_dataflow(tcx, mir, id, &[], &dead_unwinds, analysis,
                                |bd, p| &bd.move_data().move_paths[p]);

    for (i, path) in move_data.move_paths.iter_enumerated() {
        debug!("move path: {:?} => {:?}", i, path.lvalue);
    }

    // Enumerated moves where the types do not implement Copy or Move
    for move_out in move_data.moves.iter() {
        let move_lvalue = &move_data.move_paths[move_out.path].lvalue;
        let span = move_out.source.source_info(mir).span;

        // Are we moving an immovable type out of an box?
        if deref_box(tcx, mir, move_lvalue) {
            span_err!(tcx.sess, span, E0802, "cannot move immovable value out of a Box type");
        }

        let state = state_for_location(move_out.source, &analysis, &observed);

        let mut invalid_move = false;

        debug!("move out path: {:?}", move_lvalue);

        // Check if the lvalue moved and all its children are either unobserved or implements Move
        move_data.all_paths_from(move_out.path, |subpath| {
            // Don't look for further errors if this move was already found to be an error
            if invalid_move {
                return false;
            }

            let lvalue = &move_data.move_paths[subpath].lvalue;
            let lvalue_ty = lvalue.ty(mir, tcx).to_ty(tcx);

            debug!("checking subpath: {:?}", move_data.move_paths[subpath].lvalue);
            debug!("is-observed: {}, move: {}", state.contains(&subpath),
                   lvalue_ty.is_move(tcx.global_tcx(), param_env, span));

            if state.contains(&subpath) && !lvalue_ty.is_move(tcx.global_tcx(), param_env, span) {
                // The subpath was observed. Run a dataflow analysis to find out which borrows
                // caused the subpath to get observed and add those locations to the error
                // as notes.

                let mut observers = LvalObservers::new(tcx, mir, &move_data, subpath);
                observers.visit_mir(mir);

                static STR: &'static &'static str = &"<>";

                let observer_result =
                    dataflow::do_dataflow(tcx, mir, id, &[], &dead_unwinds, observers.clone(),
                                        |_, _| STR);

                let state = state_for_location(move_out.source, &observers, &observer_result);

                let mut err = struct_span_err!(tcx.sess, span, E0801,
                    "cannot move value whose address is observed");

                err.note(&format!("required because `{}` does not implement Move", lvalue_ty));

                for (i, loc) in observers.observers.iter().enumerate() {
                    if state.contains(&i) {
                        span_note!(err,
                                    loc.source_info(mir).span,
                                    "the address was observed here");
                    }
                }

                err.emit();

                invalid_move = true;
            }

            true
        });
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        moveck,
        ..*providers
    };
}
