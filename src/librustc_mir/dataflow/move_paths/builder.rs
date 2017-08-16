// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::{self, TyCtxt};
use rustc::mir::*;
use rustc::mir::tcx::RvalueInitializationState;
use rustc::util::nodemap::FxHashMap;
use rustc_data_structures::indexed_vec::{IndexVec};

use syntax::codemap::DUMMY_SP;

use std::collections::hash_map::Entry;
use std::mem;

use super::abs_domain::Lift;

use super::{LocationMap, MoveData, MovePath, MovePathLookup, MovePathIndex, MoveOut, MoveOutIndex};

pub(super) struct MoveDataBuilder<'a, 'tcx: 'a> {
    mir: &'a Mir<'tcx>,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    data: MoveData<'tcx>,
}

pub enum MovePathError {
    IllegalMove,
    UnionMove { path: MovePathIndex },
}

impl<'a, 'tcx> MoveDataBuilder<'a, 'tcx> {
    fn new(mir: &'a Mir<'tcx>,
           tcx: TyCtxt<'a, 'tcx, 'tcx>,
           param_env: ty::ParamEnv<'tcx>)
           -> Self {
        let mut move_paths = IndexVec::new();
        let mut path_map = IndexVec::new();

        MoveDataBuilder {
            mir,
            tcx,
            param_env,
            data: MoveData {
                moves: IndexVec::new(),
                loc_map: LocationMap::new(mir),
                rev_lookup: MovePathLookup {
                    locals: mir.local_decls.indices().map(Lvalue::Local).map(|v| {
                        Self::new_move_path(&mut move_paths, &mut path_map, None, v)
                    }).collect(),
                    projections: FxHashMap(),
                },
                move_paths,
                path_map,
            }
        }
    }

    fn new_move_path(move_paths: &mut IndexVec<MovePathIndex, MovePath<'tcx>>,
                     path_map: &mut IndexVec<MovePathIndex, Vec<MoveOutIndex>>,
                     parent: Option<MovePathIndex>,
                     lvalue: Lvalue<'tcx>)
                     -> MovePathIndex
    {
        let move_path = move_paths.push(MovePath {
            next_sibling: None,
            first_child: None,
            parent,
            lvalue,
        });

        if let Some(parent) = parent {
            let next_sibling =
                mem::replace(&mut move_paths[parent].first_child, Some(move_path));
            move_paths[move_path].next_sibling = next_sibling;
        }

        let path_map_ent = path_map.push(vec![]);
        assert_eq!(path_map_ent, move_path);
        move_path
    }

    /// This creates a MovePath for a given lvalue, returning an `MovePathError`
    /// if that lvalue can't be moved from.
    ///
    /// NOTE: lvalues behind references *do not* get a move path, which is
    /// problematic for borrowck.
    ///
    /// Maybe we should have separate "borrowck" and "moveck" modes.
    fn move_path_for(&mut self, lval: &Lvalue<'tcx>)
                     -> Result<MovePathIndex, MovePathError>
    {
        debug!("lookup({:?})", lval);
        match *lval {
            Lvalue::Local(local) => Ok(self.data.rev_lookup.locals[local]),
            // error: can't move out of a static
            Lvalue::Static(..) => Err(MovePathError::IllegalMove),
            Lvalue::Projection(ref proj) => {
                self.move_path_for_projection(lval, proj)
            }
        }
    }

    fn create_move_path(&mut self, lval: &Lvalue<'tcx>) {
        // This is an assignment, not a move, so this not being a valid
        // move path is OK.
        let _ = self.move_path_for(lval);
    }

    fn move_path_for_projection(&mut self,
                                lval: &Lvalue<'tcx>,
                                proj: &LvalueProjection<'tcx>)
                                -> Result<MovePathIndex, MovePathError>
    {
        let base = try!(self.move_path_for(&proj.base));
        let lv_ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);
        match lv_ty.sty {
            // error: can't move out of borrowed content
            ty::TyRef(..) | ty::TyRawPtr(..) => return Err(MovePathError::IllegalMove),
            // error: can't move out of struct with destructor
            ty::TyAdt(adt, _) if adt.has_dtor(self.tcx) && !adt.is_box() =>
                return Err(MovePathError::IllegalMove),
            // move out of union - always move the entire union
            ty::TyAdt(adt, _) if adt.is_union() =>
                return Err(MovePathError::UnionMove { path: base }),
            // error: can't move out of a slice
            ty::TySlice(..) =>
                return Err(MovePathError::IllegalMove),
            ty::TyArray(..) => match proj.elem {
                // error: can't move out of an array
                ProjectionElem::Index(..) => return Err(MovePathError::IllegalMove),
                _ => {
                    // FIXME: still badly broken
                }
            },
            _ => {}
        };
        match self.data.rev_lookup.projections.entry((base, proj.elem.lift())) {
            Entry::Occupied(ent) => Ok(*ent.get()),
            Entry::Vacant(ent) => {
                let path = Self::new_move_path(
                    &mut self.data.move_paths,
                    &mut self.data.path_map,
                    Some(base),
                    lval.clone()
                );
                ent.insert(path);
                Ok(path)
            }
        }
    }

    fn finalize(self) -> MoveData<'tcx> {
        debug!("{}", {
            debug!("moves for {:?}:", self.mir.span);
            for (j, mo) in self.data.moves.iter_enumerated() {
                debug!("    {:?} = {:?}", j, mo);
            }
            debug!("move paths for {:?}:", self.mir.span);
            for (j, path) in self.data.move_paths.iter_enumerated() {
                debug!("    {:?} = {:?}", j, path);
            }
            "done dumping moves"
        });
        self.data
    }
}

pub(super) fn gather_moves<'a, 'tcx>(mir: &Mir<'tcx>,
                                     tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     param_env: ty::ParamEnv<'tcx>)
                                     -> MoveData<'tcx> {
    let mut builder = MoveDataBuilder::new(mir, tcx, param_env);

    for (bb, block) in mir.basic_blocks().iter_enumerated() {
        for (i, stmt) in block.statements.iter().enumerate() {
            let source = Location { block: bb, statement_index: i };
            builder.gather_statement(source, stmt);
        }

        let terminator_loc = Location {
            block: bb,
            statement_index: block.statements.len()
        };
        builder.gather_terminator(terminator_loc, block.terminator());
    }

    builder.finalize()
}

impl<'a, 'tcx> MoveDataBuilder<'a, 'tcx> {
    fn gather_statement(&mut self, loc: Location, stmt: &Statement<'tcx>) {
        debug!("gather_statement({:?}, {:?})", loc, stmt);
        match stmt.kind {
            StatementKind::Assign(ref lval, ref rval) => {
                self.create_move_path(lval);
                if let RvalueInitializationState::Shallow = rval.initialization_state() {
                    // Box starts out uninitialized - need to create a separate
                    // move-path for the interior so it will be separate from
                    // the exterior.
                    self.create_move_path(&lval.clone().deref());
                }
                self.gather_rvalue(loc, rval);
            }
            StatementKind::StorageLive(_) |
            StatementKind::StorageDead(_) => {}
            StatementKind::SetDiscriminant{ .. } => {
                span_bug!(stmt.source_info.span,
                          "SetDiscriminant should not exist during borrowck");
            }
            StatementKind::InlineAsm { .. } |
            StatementKind::EndRegion(_) |
            StatementKind::Validate(..) |
            StatementKind::Nop => {}
        }
    }

    fn gather_rvalue(&mut self, loc: Location, rvalue: &Rvalue<'tcx>) {
        match *rvalue {
            Rvalue::Use(ref operand) |
            Rvalue::Repeat(ref operand, _) |
            Rvalue::Cast(_, ref operand, _) |
            Rvalue::UnaryOp(_, ref operand) => {
                self.gather_operand(loc, operand)
            }
            Rvalue::BinaryOp(ref _binop, ref lhs, ref rhs) |
            Rvalue::CheckedBinaryOp(ref _binop, ref lhs, ref rhs) => {
                self.gather_operand(loc, lhs);
                self.gather_operand(loc, rhs);
            }
            Rvalue::Aggregate(ref _kind, ref operands) => {
                for operand in operands {
                    self.gather_operand(loc, operand);
                }
            }
            Rvalue::Ref(..) |
            Rvalue::Discriminant(..) |
            Rvalue::Len(..) |
            Rvalue::NullaryOp(NullOp::SizeOf, _) |
            Rvalue::NullaryOp(NullOp::Box, _) => {
                // This returns an rvalue with uninitialized contents. We can't
                // move out of it here because it is an rvalue - assignments always
                // completely initialize their lvalue.
                //
                // However, this does not matter - MIR building is careful to
                // only emit a shallow free for the partially-initialized
                // temporary.
                //
                // In any case, if we want to fix this, we have to register a
                // special move and change the `statement_effect` functions.
            }
        }
    }

    fn gather_terminator(&mut self, loc: Location, term: &Terminator<'tcx>) {
        debug!("gather_terminator({:?}, {:?})", loc, term);
        match term.kind {
            TerminatorKind::Goto { target: _ } |
            TerminatorKind::Resume |
            TerminatorKind::Unreachable => { }

            TerminatorKind::Return => {
                self.gather_move(loc, &Lvalue::Local(RETURN_POINTER));
            }

            TerminatorKind::Assert { .. } |
            TerminatorKind::SwitchInt { .. } => {
                // branching terminators - these don't move anything
            }

            TerminatorKind::Drop { ref location, target: _, unwind: _ } => {
                self.gather_move(loc, location);
            }
            TerminatorKind::DropAndReplace { ref location, ref value, .. } => {
                self.create_move_path(location);
                self.gather_operand(loc, value);
            }
            TerminatorKind::Call { ref func, ref args, ref destination, cleanup: _ } => {
                self.gather_operand(loc, func);
                for arg in args {
                    self.gather_operand(loc, arg);
                }
                if let Some((ref destination, _bb)) = *destination {
                    self.create_move_path(destination);
                }
            }
        }
    }

    fn gather_operand(&mut self, loc: Location, operand: &Operand<'tcx>) {
        match *operand {
            Operand::Constant(..) => {} // not-a-move
            Operand::Consume(ref lval) => { // a move
                self.gather_move(loc, lval);
            }
        }
    }

    fn gather_move(&mut self, loc: Location, lval: &Lvalue<'tcx>) {
        debug!("gather_move({:?}, {:?})", loc, lval);

        let lv_ty = lval.ty(self.mir, self.tcx).to_ty(self.tcx);
        if !lv_ty.moves_by_default(self.tcx, self.param_env, DUMMY_SP) {
            debug!("gather_move({:?}, {:?}) - {:?} is Copy. skipping", loc, lval, lv_ty);
            return
        }

        let path = match self.move_path_for(lval) {
            Ok(path) | Err(MovePathError::UnionMove { path }) => path,
            Err(MovePathError::IllegalMove) => {
                // Moving out of a bad path. Eventually, this should be a MIR
                // borrowck error instead of a bug.
                span_bug!(self.mir.span,
                          "Broken MIR: moving out of lvalue {:?}: {:?} at {:?}",
                          lval, lv_ty, loc);
            }
        };
        let move_out = self.data.moves.push(MoveOut { path: path, source: loc });

        debug!("gather_move({:?}, {:?}): adding move {:?} of {:?}",
               loc, lval, move_out, path);

        self.data.path_map[path].push(move_out);
        self.data.loc_map[loc].push(move_out);
    }
}
