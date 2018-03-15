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
use super::{MoveError};
use super::IllegalMoveOriginKind::*;

struct MoveDataBuilder<'a, 'gcx: 'tcx, 'tcx: 'a> {
    mir: &'a Mir<'tcx>,
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    param_env: ty::ParamEnv<'gcx>,
    data: MoveData<'tcx>,
    errors: Vec<MoveError<'tcx>>,
}

impl<'a, 'gcx, 'tcx> MoveDataBuilder<'a, 'gcx, 'tcx> {
    fn new(mir: &'a Mir<'tcx>,
           tcx: TyCtxt<'a, 'gcx, 'tcx>,
           param_env: ty::ParamEnv<'gcx>)
           -> Self {
        let mut move_paths = IndexVec::new();
        let mut path_map = IndexVec::new();

        MoveDataBuilder {
            mir,
            tcx,
            param_env,
            errors: Vec::new(),
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
}

impl<'b, 'a, 'gcx, 'tcx> Gatherer<'b, 'a, 'gcx, 'tcx> {
    /// This creates a MovePath for a given lvalue, returning an `MovePathError`
    /// if that lvalue can't be moved from.
    ///
    /// NOTE: lvalues behind references *do not* get a move path, which is
    /// problematic for borrowck.
    ///
    /// Maybe we should have separate "borrowck" and "moveck" modes.
    fn move_path_for(&mut self, lval: &Lvalue<'tcx>)
                     -> Result<MovePathIndex, MoveError<'tcx>>
    {
        debug!("lookup({:?})", lval);
        match *lval {
            Lvalue::Local(local) => Ok(self.builder.data.rev_lookup.locals[local]),
            Lvalue::Static(..) => {
                let span = self.builder.mir.source_info(self.loc).span;
                Err(MoveError::cannot_move_out_of(span, Static))
            }
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
                                -> Result<MovePathIndex, MoveError<'tcx>>
    {
        let base = try!(self.move_path_for(&proj.base));
        let mir = self.builder.mir;
        let tcx = self.builder.tcx;
        let lv_ty = proj.base.ty(mir, tcx).to_ty(tcx);
        match lv_ty.sty {
            ty::TyRef(..) | ty::TyRawPtr(..) =>
                return Err(MoveError::cannot_move_out_of(mir.source_info(self.loc).span,
                                                         BorrowedContent)),
            ty::TyAdt(adt, _) if adt.has_dtor(tcx) && !adt.is_box() =>
                return Err(MoveError::cannot_move_out_of(mir.source_info(self.loc).span,
                                                         InteriorOfTypeWithDestructor {
                    container_ty: lv_ty
                })),
            // move out of union - always move the entire union
            ty::TyAdt(adt, _) if adt.is_union() =>
                return Err(MoveError::UnionMove { path: base }),
            ty::TySlice(elem_ty) =>
                return Err(MoveError::cannot_move_out_of(
                    mir.source_info(self.loc).span,
                    InteriorOfSlice {
                        elem_ty, is_index: match proj.elem {
                            ProjectionElem::Index(..) => true,
                            _ => false
                        },
                    })),
            ty::TyArray(elem_ty, _num_elems) => match proj.elem {
                ProjectionElem::Index(..) =>
                    return Err(MoveError::cannot_move_out_of(
                        mir.source_info(self.loc).span,
                        InteriorOfArray {
                            elem_ty, is_index: true
                        })),
                _ => {
                    // FIXME: still badly broken
                }
            },
            _ => {}
        };
        match self.builder.data.rev_lookup.projections.entry((base, proj.elem.lift())) {
            Entry::Occupied(ent) => Ok(*ent.get()),
            Entry::Vacant(ent) => {
                let path = MoveDataBuilder::new_move_path(
                    &mut self.builder.data.move_paths,
                    &mut self.builder.data.path_map,
                    Some(base),
                    lval.clone()
                );
                ent.insert(path);
                Ok(path)
            }
        }
    }
}

impl<'a, 'gcx, 'tcx> MoveDataBuilder<'a, 'gcx, 'tcx> {
    fn finalize(self) -> Result<MoveData<'tcx>, (MoveData<'tcx>, Vec<MoveError<'tcx>>)> {
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

        if self.errors.len() > 0 {
            Err((self.data, self.errors))
        } else {
            Ok(self.data)
        }
    }
}

pub(super) fn gather_moves<'a, 'gcx, 'tcx>(mir: &Mir<'tcx>,
                                           tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                           param_env: ty::ParamEnv<'gcx>)
                                           -> Result<MoveData<'tcx>,
                                                     (MoveData<'tcx>, Vec<MoveError<'tcx>>)> {
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

impl<'a, 'gcx, 'tcx> MoveDataBuilder<'a, 'gcx, 'tcx> {
    fn gather_statement(&mut self, loc: Location, stmt: &Statement<'tcx>) {
        debug!("gather_statement({:?}, {:?})", loc, stmt);
        (Gatherer { builder: self, loc }).gather_statement(stmt);
    }

    fn gather_terminator(&mut self, loc: Location, term: &Terminator<'tcx>) {
        debug!("gather_terminator({:?}, {:?})", loc, term);
        (Gatherer { builder: self, loc }).gather_terminator(term);
    }
}

struct Gatherer<'b, 'a: 'b, 'gcx: 'tcx, 'tcx: 'a> {
    builder: &'b mut MoveDataBuilder<'a, 'gcx, 'tcx>,
    loc: Location,
}

impl<'b, 'a, 'gcx, 'tcx> Gatherer<'b, 'a, 'gcx, 'tcx> {
    fn gather_statement(&mut self, stmt: &Statement<'tcx>) {
        match stmt.kind {
            StatementKind::Assign(ref lval, ref rval) => {
                self.create_move_path(lval);
                if let RvalueInitializationState::Shallow = rval.initialization_state() {
                    // Box starts out uninitialized - need to create a separate
                    // move-path for the interior so it will be separate from
                    // the exterior.
                    self.create_move_path(&lval.clone().deref());
                }
                self.gather_rvalue(rval);
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

    fn gather_rvalue(&mut self, rvalue: &Rvalue<'tcx>) {
        match *rvalue {
            Rvalue::Use(ref operand) |
            Rvalue::Repeat(ref operand, _) |
            Rvalue::Cast(_, ref operand, _) |
            Rvalue::UnaryOp(_, ref operand) => {
                self.gather_operand(operand)
            }
            Rvalue::BinaryOp(ref _binop, ref lhs, ref rhs) |
            Rvalue::CheckedBinaryOp(ref _binop, ref lhs, ref rhs) => {
                self.gather_operand(lhs);
                self.gather_operand(rhs);
            }
            Rvalue::Aggregate(ref _kind, ref operands) => {
                for operand in operands {
                    self.gather_operand(operand);
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

    fn gather_terminator(&mut self, term: &Terminator<'tcx>) {
        match term.kind {
            TerminatorKind::Goto { target: _ } |
            TerminatorKind::Resume |
            TerminatorKind::GeneratorDrop |
            TerminatorKind::FalseEdges { .. } |
            TerminatorKind::Unreachable => { }

            TerminatorKind::Return => {
                self.gather_move(&Lvalue::Local(RETURN_POINTER));
            }

            TerminatorKind::Assert { .. } |
            TerminatorKind::SwitchInt { .. } => {
                // branching terminators - these don't move anything
            }

            TerminatorKind::Yield { ref value, .. } => {
                self.gather_operand(value);
            }

            TerminatorKind::Drop { ref location, target: _, unwind: _ } => {
                self.gather_move(location);
            }
            TerminatorKind::DropAndReplace { ref location, ref value, .. } => {
                self.create_move_path(location);
                self.gather_operand(value);
            }
            TerminatorKind::Call { ref func, ref args, ref destination, cleanup: _ } => {
                self.gather_operand(func);
                for arg in args {
                    self.gather_operand(arg);
                }
                if let Some((ref destination, _bb)) = *destination {
                    self.create_move_path(destination);
                }
            }
        }
    }

    fn gather_operand(&mut self, operand: &Operand<'tcx>) {
        match *operand {
            Operand::Constant(..) => {} // not-a-move
            Operand::Consume(ref lval) => { // a move
                self.gather_move(lval);
            }
        }
    }

    fn gather_move(&mut self, lval: &Lvalue<'tcx>) {
        debug!("gather_move({:?}, {:?})", self.loc, lval);

        let tcx = self.builder.tcx;
        let gcx = tcx.global_tcx();
        let lv_ty = lval.ty(self.builder.mir, tcx).to_ty(tcx);
        let erased_ty = gcx.lift(&tcx.erase_regions(&lv_ty)).unwrap();
        if !erased_ty.moves_by_default(gcx, self.builder.param_env, DUMMY_SP) {
            debug!("gather_move({:?}, {:?}) - {:?} is Copy. skipping", self.loc, lval, lv_ty);
            return
        }

        let path = match self.move_path_for(lval) {
            Ok(path) | Err(MoveError::UnionMove { path }) => path,
            Err(error @ MoveError::IllegalMove { .. }) => {
                self.builder.errors.push(error);
                return;
            }
        };
        let move_out = self.builder.data.moves.push(MoveOut { path: path, source: self.loc });

        debug!("gather_move({:?}, {:?}): adding move {:?} of {:?}",
               self.loc, lval, move_out, path);

        self.builder.data.path_map[path].push(move_out);
        self.builder.data.loc_map[self.loc].push(move_out);
    }
}
