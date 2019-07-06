use rustc::ty::{self, TyCtxt};
use rustc::mir::*;
use rustc::mir::tcx::RvalueInitializationState;
use rustc_data_structures::indexed_vec::{IndexVec};
use smallvec::{SmallVec, smallvec};

use std::collections::hash_map::Entry;
use std::mem;

use super::abs_domain::Lift;
use super::{LocationMap, MoveData, MovePath, MovePathLookup, MovePathIndex, MoveOut, MoveOutIndex};
use super::{MoveError, InitIndex, Init, InitLocation, LookupResult, InitKind};
use super::IllegalMoveOriginKind::*;

struct MoveDataBuilder<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    data: MoveData<'tcx>,
    errors: Vec<(Place<'tcx>, MoveError<'tcx>)>,
}

impl<'a, 'tcx> MoveDataBuilder<'a, 'tcx> {
    fn new(body: &'a Body<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        let mut move_paths = IndexVec::new();
        let mut path_map = IndexVec::new();
        let mut init_path_map = IndexVec::new();

        MoveDataBuilder {
            body,
            tcx,
            errors: Vec::new(),
            data: MoveData {
                moves: IndexVec::new(),
                loc_map: LocationMap::new(body),
                rev_lookup: MovePathLookup {
                    locals: body.local_decls.indices().map(|i| {
                        Self::new_move_path(
                            &mut move_paths,
                            &mut path_map,
                            &mut init_path_map,
                            None,
                            Place::from(i),
                        )
                    }).collect(),
                    projections: Default::default(),
                },
                move_paths,
                path_map,
                inits: IndexVec::new(),
                init_loc_map: LocationMap::new(body),
                init_path_map,
            }
        }
    }

    fn new_move_path(move_paths: &mut IndexVec<MovePathIndex, MovePath<'tcx>>,
                     path_map: &mut IndexVec<MovePathIndex, SmallVec<[MoveOutIndex; 4]>>,
                     init_path_map: &mut IndexVec<MovePathIndex, SmallVec<[InitIndex; 4]>>,
                     parent: Option<MovePathIndex>,
                     place: Place<'tcx>)
                     -> MovePathIndex
    {
        let move_path = move_paths.push(MovePath {
            next_sibling: None,
            first_child: None,
            parent,
            place,
        });

        if let Some(parent) = parent {
            let next_sibling =
                mem::replace(&mut move_paths[parent].first_child, Some(move_path));
            move_paths[move_path].next_sibling = next_sibling;
        }

        let path_map_ent = path_map.push(smallvec![]);
        assert_eq!(path_map_ent, move_path);

        let init_path_map_ent = init_path_map.push(smallvec![]);
        assert_eq!(init_path_map_ent, move_path);

        move_path
    }
}

impl<'b, 'a, 'tcx> Gatherer<'b, 'a, 'tcx> {
    /// This creates a MovePath for a given place, returning an `MovePathError`
    /// if that place can't be moved from.
    ///
    /// NOTE: places behind references *do not* get a move path, which is
    /// problematic for borrowck.
    ///
    /// Maybe we should have separate "borrowck" and "moveck" modes.
    fn move_path_for(&mut self, place: &Place<'tcx>)
                     -> Result<MovePathIndex, MoveError<'tcx>>
    {
        debug!("lookup({:?})", place);
        place.iterate(|place_base, place_projection| {
            let mut base = match place_base {
                PlaceBase::Local(local) => self.builder.data.rev_lookup.locals[*local],
                PlaceBase::Static(..) => {
                    return Err(MoveError::cannot_move_out_of(self.loc, Static));
                }
            };

            for proj in place_projection {
                let body = self.builder.body;
                let tcx = self.builder.tcx;
                let place_ty = proj.base.ty(body, tcx).ty;
                match place_ty.sty {
                    ty::Ref(..) | ty::RawPtr(..) =>
                        return Err(MoveError::cannot_move_out_of(
                            self.loc,
                            BorrowedContent {
                                target_place: Place::Projection(Box::new(proj.clone())),
                            })),
                    ty::Adt(adt, _) if adt.has_dtor(tcx) && !adt.is_box() =>
                        return Err(MoveError::cannot_move_out_of(self.loc,
                                                                 InteriorOfTypeWithDestructor {
                            container_ty: place_ty
                        })),
                    // move out of union - always move the entire union
                    ty::Adt(adt, _) if adt.is_union() =>
                        return Err(MoveError::UnionMove { path: base }),
                    ty::Slice(_) =>
                        return Err(MoveError::cannot_move_out_of(
                            self.loc,
                            InteriorOfSliceOrArray {
                                ty: place_ty, is_index: match proj.elem {
                                    ProjectionElem::Index(..) => true,
                                    _ => false
                                },
                            })),
                    ty::Array(..) => match proj.elem {
                        ProjectionElem::Index(..) =>
                            return Err(MoveError::cannot_move_out_of(
                                self.loc,
                                InteriorOfSliceOrArray {
                                    ty: place_ty, is_index: true
                                })),
                        _ => {
                            // FIXME: still badly broken
                        }
                    },
                    _ => {}
                };

                base = match self
                    .builder
                    .data
                    .rev_lookup
                    .projections
                    .entry((base, proj.elem.lift()))
                {
                    Entry::Occupied(ent) => *ent.get(),
                    Entry::Vacant(ent) => {
                        let path = MoveDataBuilder::new_move_path(
                            &mut self.builder.data.move_paths,
                            &mut self.builder.data.path_map,
                            &mut self.builder.data.init_path_map,
                            Some(base),
                            Place::Projection(Box::new(proj.clone())),
                        );
                        ent.insert(path);
                        path
                    }
                };
            }

            Ok(base)
        })
    }

    fn create_move_path(&mut self, place: &Place<'tcx>) {
        // This is an non-moving access (such as an overwrite or
        // drop), so this not being a valid move path is OK.
        let _ = self.move_path_for(place);
    }
}

impl<'a, 'tcx> MoveDataBuilder<'a, 'tcx> {
    fn finalize(
        self
    ) -> Result<MoveData<'tcx>, (MoveData<'tcx>, Vec<(Place<'tcx>, MoveError<'tcx>)>)> {
        debug!("{}", {
            debug!("moves for {:?}:", self.body.span);
            for (j, mo) in self.data.moves.iter_enumerated() {
                debug!("    {:?} = {:?}", j, mo);
            }
            debug!("move paths for {:?}:", self.body.span);
            for (j, path) in self.data.move_paths.iter_enumerated() {
                debug!("    {:?} = {:?}", j, path);
            }
            "done dumping moves"
        });

        if !self.errors.is_empty() {
            Err((self.data, self.errors))
        } else {
            Ok(self.data)
        }
    }
}

pub(super) fn gather_moves<'tcx>(
    body: &Body<'tcx>,
    tcx: TyCtxt<'tcx>,
) -> Result<MoveData<'tcx>, (MoveData<'tcx>, Vec<(Place<'tcx>, MoveError<'tcx>)>)> {
    let mut builder = MoveDataBuilder::new(body, tcx);

    builder.gather_args();

    for (bb, block) in body.basic_blocks().iter_enumerated() {
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
    fn gather_args(&mut self) {
        for arg in self.body.args_iter() {
            let path = self.data.rev_lookup.locals[arg];

            let init = self.data.inits.push(Init {
                path, kind: InitKind::Deep, location: InitLocation::Argument(arg),
            });

            debug!("gather_args: adding init {:?} of {:?} for argument {:?}",
                init, path, arg);

            self.data.init_path_map[path].push(init);
        }
    }

    fn gather_statement(&mut self, loc: Location, stmt: &Statement<'tcx>) {
        debug!("gather_statement({:?}, {:?})", loc, stmt);
        (Gatherer { builder: self, loc }).gather_statement(stmt);
    }

    fn gather_terminator(&mut self, loc: Location, term: &Terminator<'tcx>) {
        debug!("gather_terminator({:?}, {:?})", loc, term);
        (Gatherer { builder: self, loc }).gather_terminator(term);
    }
}

struct Gatherer<'b, 'a, 'tcx> {
    builder: &'b mut MoveDataBuilder<'a, 'tcx>,
    loc: Location,
}

impl<'b, 'a, 'tcx> Gatherer<'b, 'a, 'tcx> {
    fn gather_statement(&mut self, stmt: &Statement<'tcx>) {
        match stmt.kind {
            StatementKind::Assign(ref place, ref rval) => {
                self.create_move_path(place);
                if let RvalueInitializationState::Shallow = rval.initialization_state() {
                    // Box starts out uninitialized - need to create a separate
                    // move-path for the interior so it will be separate from
                    // the exterior.
                    self.create_move_path(&place.clone().deref());
                    self.gather_init(place, InitKind::Shallow);
                } else {
                    self.gather_init(place, InitKind::Deep);
                }
                self.gather_rvalue(rval);
            }
            StatementKind::FakeRead(_, ref place) => {
                self.create_move_path(place);
            }
            StatementKind::InlineAsm(ref asm) => {
                for (output, kind) in asm.outputs.iter().zip(&asm.asm.outputs) {
                    if !kind.is_indirect {
                        self.gather_init(output, InitKind::Deep);
                    }
                }
                for (_, input) in asm.inputs.iter() {
                    self.gather_operand(input);
                }
            }
            StatementKind::StorageLive(_) => {}
            StatementKind::StorageDead(local) => {
                self.gather_move(&Place::from(local));
            }
            StatementKind::SetDiscriminant{ .. } => {
                span_bug!(stmt.source_info.span,
                          "SetDiscriminant should not exist during borrowck");
            }
            StatementKind::Retag { .. } |
            StatementKind::AscribeUserType(..) |
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
                // completely initialize their place.
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
            TerminatorKind::Abort |
            TerminatorKind::GeneratorDrop |
            TerminatorKind::FalseEdges { .. } |
            TerminatorKind::FalseUnwind { .. } |
            TerminatorKind::Unreachable => { }

            TerminatorKind::Return => {
                self.gather_move(&Place::RETURN_PLACE);
            }

            TerminatorKind::Assert { ref cond, .. } => {
                self.gather_operand(cond);
            }

            TerminatorKind::SwitchInt { ref discr, .. } => {
                self.gather_operand(discr);
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
                self.gather_init(location, InitKind::Deep);
            }
            TerminatorKind::Call {
                ref func,
                ref args,
                ref destination,
                cleanup: _,
                from_hir_call: _,
            } => {
                self.gather_operand(func);
                for arg in args {
                    self.gather_operand(arg);
                }
                if let Some((ref destination, _bb)) = *destination {
                    self.create_move_path(destination);
                    self.gather_init(destination, InitKind::NonPanicPathOnly);
                }
            }
        }
    }

    fn gather_operand(&mut self, operand: &Operand<'tcx>) {
        match *operand {
            Operand::Constant(..) |
            Operand::Copy(..) => {} // not-a-move
            Operand::Move(ref place) => { // a move
                self.gather_move(place);
            }
        }
    }

    fn gather_move(&mut self, place: &Place<'tcx>) {
        debug!("gather_move({:?}, {:?})", self.loc, place);

        let path = match self.move_path_for(place) {
            Ok(path) | Err(MoveError::UnionMove { path }) => path,
            Err(error @ MoveError::IllegalMove { .. }) => {
                self.builder.errors.push((place.clone(), error));
                return;
            }
        };
        let move_out = self.builder.data.moves.push(MoveOut { path: path, source: self.loc });

        debug!("gather_move({:?}, {:?}): adding move {:?} of {:?}",
               self.loc, place, move_out, path);

        self.builder.data.path_map[path].push(move_out);
        self.builder.data.loc_map[self.loc].push(move_out);
    }

    fn gather_init(&mut self, place: &Place<'tcx>, kind: InitKind) {
        debug!("gather_init({:?}, {:?})", self.loc, place);

        let place = match place {
            // Check if we are assigning into a field of a union, if so, lookup the place
            // of the union so it is marked as initialized again.
            Place::Projection(box Projection {
                base,
                elem: ProjectionElem::Field(_, _),
            }) if match base.ty(self.builder.body, self.builder.tcx).ty.sty {
                    ty::Adt(def, _) if def.is_union() => true,
                    _ => false,
            } => base,
            // Otherwise, lookup the place.
            _ => place,
        };

        if let LookupResult::Exact(path) = self.builder.data.rev_lookup.find(place) {
            let init = self.builder.data.inits.push(Init {
                location: InitLocation::Statement(self.loc),
                path,
                kind,
            });

            debug!("gather_init({:?}, {:?}): adding init {:?} of {:?}",
               self.loc, place, init, path);

            self.builder.data.init_path_map[path].push(init);
            self.builder.data.init_loc_map[self.loc].push(init);
        }
    }
}
