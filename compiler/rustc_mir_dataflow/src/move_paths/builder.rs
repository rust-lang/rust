use std::mem;

use rustc_index::IndexVec;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt};
use rustc_middle::{bug, span_bug};
use smallvec::{SmallVec, smallvec};
use tracing::debug;

use super::{
    Init, InitIndex, InitKind, InitLocation, LocationMap, LookupResult, MoveData, MoveOut,
    MoveOutIndex, MovePath, MovePathIndex, MovePathLookup,
};

struct MoveDataBuilder<'a, 'tcx, F> {
    body: &'a Body<'tcx>,
    loc: Location,
    tcx: TyCtxt<'tcx>,
    data: MoveData<'tcx>,
    filter: F,
}

impl<'a, 'tcx, F: Fn(Ty<'tcx>) -> bool> MoveDataBuilder<'a, 'tcx, F> {
    fn new(body: &'a Body<'tcx>, tcx: TyCtxt<'tcx>, filter: F) -> Self {
        let mut move_paths = IndexVec::new();
        let mut path_map = IndexVec::new();
        let mut init_path_map = IndexVec::new();

        let locals = body
            .local_decls
            .iter_enumerated()
            .map(|(i, l)| {
                if l.is_deref_temp() {
                    return None;
                }
                if filter(l.ty) {
                    Some(new_move_path(
                        &mut move_paths,
                        &mut path_map,
                        &mut init_path_map,
                        None,
                        Place::from(i),
                    ))
                } else {
                    None
                }
            })
            .collect();

        MoveDataBuilder {
            body,
            loc: Location::START,
            tcx,
            data: MoveData {
                moves: IndexVec::new(),
                loc_map: LocationMap::new(body),
                rev_lookup: MovePathLookup {
                    locals,
                    projections: Default::default(),
                    un_derefer: Default::default(),
                },
                move_paths,
                path_map,
                inits: IndexVec::new(),
                init_loc_map: LocationMap::new(body),
                init_path_map,
            },
            filter,
        }
    }
}

fn new_move_path<'tcx>(
    move_paths: &mut IndexVec<MovePathIndex, MovePath<'tcx>>,
    path_map: &mut IndexVec<MovePathIndex, SmallVec<[MoveOutIndex; 4]>>,
    init_path_map: &mut IndexVec<MovePathIndex, SmallVec<[InitIndex; 4]>>,
    parent: Option<MovePathIndex>,
    place: Place<'tcx>,
) -> MovePathIndex {
    let move_path =
        move_paths.push(MovePath { next_sibling: None, first_child: None, parent, place });

    if let Some(parent) = parent {
        let next_sibling = mem::replace(&mut move_paths[parent].first_child, Some(move_path));
        move_paths[move_path].next_sibling = next_sibling;
    }

    let path_map_ent = path_map.push(smallvec![]);
    assert_eq!(path_map_ent, move_path);

    let init_path_map_ent = init_path_map.push(smallvec![]);
    assert_eq!(init_path_map_ent, move_path);

    move_path
}

enum MovePathResult {
    Path(MovePathIndex),
    Union(MovePathIndex),
    Error,
}

impl<'a, 'tcx, F: Fn(Ty<'tcx>) -> bool> MoveDataBuilder<'a, 'tcx, F> {
    /// This creates a MovePath for a given place, returning an `MovePathError`
    /// if that place can't be moved from.
    ///
    /// NOTE: places behind references *do not* get a move path, which is
    /// problematic for borrowck.
    ///
    /// Maybe we should have separate "borrowck" and "moveck" modes.
    fn move_path_for(&mut self, place: Place<'tcx>) -> MovePathResult {
        let data = &mut self.data;

        debug!("lookup({:?})", place);
        let Some(mut base) = data.rev_lookup.find_local(place.local) else {
            return MovePathResult::Error;
        };

        // The move path index of the first union that we find. Once this is
        // some we stop creating child move paths, since moves from unions
        // move the whole thing.
        // We continue looking for other move errors though so that moving
        // from `*(u.f: &_)` isn't allowed.
        let mut union_path = None;

        for (place_ref, elem) in data.rev_lookup.un_derefer.iter_projections(place.as_ref()) {
            let body = self.body;
            let tcx = self.tcx;
            let place_ty = place_ref.ty(body, tcx).ty;
            if place_ty.references_error() {
                return MovePathResult::Error;
            }
            match elem {
                ProjectionElem::Deref => match place_ty.kind() {
                    ty::Ref(..) | ty::RawPtr(..) => {
                        return MovePathResult::Error;
                    }
                    ty::Adt(adt, _) => {
                        if !adt.is_box() {
                            bug!("Adt should be a box type when Place is deref");
                        }
                    }
                    ty::Bool
                    | ty::Char
                    | ty::Int(_)
                    | ty::Uint(_)
                    | ty::Float(_)
                    | ty::Foreign(_)
                    | ty::Str
                    | ty::Array(_, _)
                    | ty::Pat(_, _)
                    | ty::Slice(_)
                    | ty::FnDef(_, _)
                    | ty::FnPtr(..)
                    | ty::Dynamic(_, _)
                    | ty::Closure(..)
                    | ty::CoroutineClosure(..)
                    | ty::Coroutine(_, _)
                    | ty::CoroutineWitness(..)
                    | ty::Never
                    | ty::Tuple(_)
                    | ty::UnsafeBinder(_)
                    | ty::Alias(_, _)
                    | ty::Param(_)
                    | ty::Bound(_, _)
                    | ty::Infer(_)
                    | ty::Error(_)
                    | ty::Placeholder(_) => {
                        bug!("When Place is Deref it's type shouldn't be {place_ty:#?}")
                    }
                },
                ProjectionElem::Field(_, _) => match place_ty.kind() {
                    ty::Adt(adt, _) => {
                        if adt.has_dtor(tcx) {
                            return MovePathResult::Error;
                        }
                        if adt.is_union() {
                            union_path.get_or_insert(base);
                        }
                    }
                    ty::Closure(..)
                    | ty::CoroutineClosure(..)
                    | ty::Coroutine(_, _)
                    | ty::Tuple(_) => (),
                    ty::Bool
                    | ty::Char
                    | ty::Int(_)
                    | ty::Uint(_)
                    | ty::Float(_)
                    | ty::Foreign(_)
                    | ty::Str
                    | ty::Array(_, _)
                    | ty::Pat(_, _)
                    | ty::Slice(_)
                    | ty::RawPtr(_, _)
                    | ty::Ref(_, _, _)
                    | ty::FnDef(_, _)
                    | ty::FnPtr(..)
                    | ty::Dynamic(_, _)
                    | ty::CoroutineWitness(..)
                    | ty::Never
                    | ty::UnsafeBinder(_)
                    | ty::Alias(_, _)
                    | ty::Param(_)
                    | ty::Bound(_, _)
                    | ty::Infer(_)
                    | ty::Error(_)
                    | ty::Placeholder(_) => bug!(
                        "When Place contains ProjectionElem::Field its type shouldn't be {place_ty:#?}"
                    ),
                },
                ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. } => {
                    match place_ty.kind() {
                        ty::Slice(_) => {
                            return MovePathResult::Error;
                        }
                        ty::Array(_, _) => (),
                        _ => bug!("Unexpected type {:#?}", place_ty.is_array()),
                    }
                }
                ProjectionElem::Index(_) => match place_ty.kind() {
                    ty::Array(..) | ty::Slice(_) => {
                        return MovePathResult::Error;
                    }
                    _ => bug!("Unexpected type {place_ty:#?}"),
                },
                ProjectionElem::UnwrapUnsafeBinder(_) => {}
                // `OpaqueCast`:Only transmutes the type, so no moves there.
                // `Downcast`  :Only changes information about a `Place` without moving.
                // So it's safe to skip these.
                ProjectionElem::OpaqueCast(_) | ProjectionElem::Downcast(_, _) => (),
            }
            let elem_ty = PlaceTy::from_ty(place_ty).projection_ty(tcx, elem).ty;
            if !(self.filter)(elem_ty) {
                return MovePathResult::Error;
            }
            if union_path.is_none() {
                // inlined from add_move_path because of a borrowck conflict with the iterator
                base =
                    *data.rev_lookup.projections.entry((base, elem.kind())).or_insert_with(|| {
                        new_move_path(
                            &mut data.move_paths,
                            &mut data.path_map,
                            &mut data.init_path_map,
                            Some(base),
                            place_ref.project_deeper(&[elem], tcx),
                        )
                    })
            }
        }

        if let Some(base) = union_path {
            // Move out of union - always move the entire union.
            MovePathResult::Union(base)
        } else {
            MovePathResult::Path(base)
        }
    }

    fn add_move_path(
        &mut self,
        base: MovePathIndex,
        elem: PlaceElem<'tcx>,
        mk_place: impl FnOnce(TyCtxt<'tcx>) -> Place<'tcx>,
    ) -> MovePathIndex {
        let MoveDataBuilder {
            data: MoveData { rev_lookup, move_paths, path_map, init_path_map, .. },
            tcx,
            ..
        } = self;
        *rev_lookup.projections.entry((base, elem.kind())).or_insert_with(move || {
            new_move_path(move_paths, path_map, init_path_map, Some(base), mk_place(*tcx))
        })
    }

    fn create_move_path(&mut self, place: Place<'tcx>) {
        // This is an non-moving access (such as an overwrite or
        // drop), so this not being a valid move path is OK.
        let _ = self.move_path_for(place);
    }

    fn finalize(self) -> MoveData<'tcx> {
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

        self.data
    }
}

pub(super) fn gather_moves<'tcx>(
    body: &Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    filter: impl Fn(Ty<'tcx>) -> bool,
) -> MoveData<'tcx> {
    let mut builder = MoveDataBuilder::new(body, tcx, filter);

    builder.gather_args();

    for (bb, block) in body.basic_blocks.iter_enumerated() {
        for (i, stmt) in block.statements.iter().enumerate() {
            builder.loc = Location { block: bb, statement_index: i };
            builder.gather_statement(stmt);
        }

        builder.loc = Location { block: bb, statement_index: block.statements.len() };
        builder.gather_terminator(block.terminator());
    }

    builder.finalize()
}

impl<'a, 'tcx, F: Fn(Ty<'tcx>) -> bool> MoveDataBuilder<'a, 'tcx, F> {
    fn gather_args(&mut self) {
        for arg in self.body.args_iter() {
            if let Some(path) = self.data.rev_lookup.find_local(arg) {
                let init = self.data.inits.push(Init {
                    path,
                    kind: InitKind::Deep,
                    location: InitLocation::Argument(arg),
                });

                debug!("gather_args: adding init {:?} of {:?} for argument {:?}", init, path, arg);

                self.data.init_path_map[path].push(init);
            }
        }
    }

    fn gather_statement(&mut self, stmt: &Statement<'tcx>) {
        debug!("gather_statement({:?}, {:?})", self.loc, stmt);
        match &stmt.kind {
            StatementKind::Assign(box (place, Rvalue::CopyForDeref(reffed))) => {
                let local = place.as_local().unwrap();
                assert!(self.body.local_decls[local].is_deref_temp());

                let rev_lookup = &mut self.data.rev_lookup;

                rev_lookup.un_derefer.insert(local, reffed.as_ref());
                let base_local = rev_lookup.un_derefer.deref_chain(local).first().unwrap().local;
                rev_lookup.locals[local] = rev_lookup.locals[base_local];
            }
            StatementKind::Assign(box (place, rval)) => {
                self.create_move_path(*place);
                if let RvalueInitializationState::Shallow = rval.initialization_state() {
                    // Box starts out uninitialized - need to create a separate
                    // move-path for the interior so it will be separate from
                    // the exterior.
                    self.create_move_path(self.tcx.mk_place_deref(*place));
                    self.gather_init(place.as_ref(), InitKind::Shallow);
                } else {
                    self.gather_init(place.as_ref(), InitKind::Deep);
                }
                self.gather_rvalue(rval);
            }
            StatementKind::FakeRead(box (_, place)) => {
                self.create_move_path(*place);
            }
            StatementKind::StorageLive(_) => {}
            StatementKind::StorageDead(local) => {
                // DerefTemp locals (results of CopyForDeref) don't actually move anything.
                if !self.body.local_decls[*local].is_deref_temp() {
                    self.gather_move(Place::from(*local));
                }
            }
            StatementKind::SetDiscriminant { .. } | StatementKind::Deinit(..) => {
                span_bug!(
                    stmt.source_info.span,
                    "SetDiscriminant/Deinit should not exist during borrowck"
                );
            }
            StatementKind::Retag { .. }
            | StatementKind::AscribeUserType(..)
            | StatementKind::PlaceMention(..)
            | StatementKind::Coverage(..)
            | StatementKind::Intrinsic(..)
            | StatementKind::ConstEvalCounter
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::Nop => {}
        }
    }

    fn gather_rvalue(&mut self, rvalue: &Rvalue<'tcx>) {
        match *rvalue {
            Rvalue::ThreadLocalRef(_) => {} // not-a-move
            Rvalue::Use(ref operand)
            | Rvalue::Repeat(ref operand, _)
            | Rvalue::Cast(_, ref operand, _)
            | Rvalue::ShallowInitBox(ref operand, _)
            | Rvalue::UnaryOp(_, ref operand)
            | Rvalue::WrapUnsafeBinder(ref operand, _) => self.gather_operand(operand),
            Rvalue::BinaryOp(ref _binop, box (ref lhs, ref rhs)) => {
                self.gather_operand(lhs);
                self.gather_operand(rhs);
            }
            Rvalue::Aggregate(ref _kind, ref operands) => {
                for operand in operands {
                    self.gather_operand(operand);
                }
            }
            Rvalue::CopyForDeref(..) => unreachable!(),
            Rvalue::Ref(..)
            | Rvalue::RawPtr(..)
            | Rvalue::Discriminant(..)
            | Rvalue::NullaryOp(
                NullOp::SizeOf
                | NullOp::AlignOf
                | NullOp::OffsetOf(..)
                | NullOp::UbChecks
                | NullOp::ContractChecks,
                _,
            ) => {}
        }
    }

    fn gather_terminator(&mut self, term: &Terminator<'tcx>) {
        debug!("gather_terminator({:?}, {:?})", self.loc, term);
        match term.kind {
            TerminatorKind::Goto { target: _ }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            // In some sense returning moves the return place into the current
            // call's destination, however, since there are no statements after
            // this that could possibly access the return place, this doesn't
            // need recording.
            | TerminatorKind::Return
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::Unreachable
            | TerminatorKind::Drop { .. } => {}

            TerminatorKind::Assert { ref cond, .. } => {
                self.gather_operand(cond);
            }

            TerminatorKind::SwitchInt { ref discr, .. } => {
                self.gather_operand(discr);
            }

            TerminatorKind::Yield { ref value, resume_arg: place, .. } => {
                self.gather_operand(value);
                self.create_move_path(place);
                self.gather_init(place.as_ref(), InitKind::Deep);
            }
            TerminatorKind::Call {
                ref func,
                ref args,
                destination,
                target,
                unwind: _,
                call_source: _,
                fn_span: _,
            } => {
                self.gather_operand(func);
                for arg in args {
                    self.gather_operand(&arg.node);
                }
                if let Some(_bb) = target {
                    self.create_move_path(destination);
                    self.gather_init(destination.as_ref(), InitKind::NonPanicPathOnly);
                }
            }
            TerminatorKind::TailCall { ref func, ref args, .. } => {
                self.gather_operand(func);
                for arg in args {
                    self.gather_operand(&arg.node);
                }
            }
            TerminatorKind::InlineAsm {
                asm_macro: _,
                template: _,
                ref operands,
                options: _,
                line_spans: _,
                targets: _,
                unwind: _,
            } => {
                for op in operands {
                    match *op {
                        InlineAsmOperand::In { reg: _, ref value }
                         => {
                            self.gather_operand(value);
                        }
                        InlineAsmOperand::Out { reg: _, late: _, place, .. } => {
                            if let Some(place) = place {
                                self.create_move_path(place);
                                self.gather_init(place.as_ref(), InitKind::Deep);
                            }
                        }
                        InlineAsmOperand::InOut { reg: _, late: _, ref in_value, out_place } => {
                            self.gather_operand(in_value);
                            if let Some(out_place) = out_place {
                                self.create_move_path(out_place);
                                self.gather_init(out_place.as_ref(), InitKind::Deep);
                            }
                        }
                        InlineAsmOperand::Const { value: _ }
                        | InlineAsmOperand::SymFn { value: _ }
                        | InlineAsmOperand::SymStatic { def_id: _ }
                        | InlineAsmOperand::Label { target_index: _ } => {}
                    }
                }
            }
        }
    }

    fn gather_operand(&mut self, operand: &Operand<'tcx>) {
        match *operand {
            Operand::Constant(..) | Operand::Copy(..) => {} // not-a-move
            Operand::Move(place) => {
                // a move
                self.gather_move(place);
            }
        }
    }

    fn gather_move(&mut self, place: Place<'tcx>) {
        debug!("gather_move({:?}, {:?})", self.loc, place);
        if let [ref base @ .., ProjectionElem::Subslice { from, to, from_end: false }] =
            **place.projection
        {
            // Split `Subslice` patterns into the corresponding list of
            // `ConstIndex` patterns. This is done to ensure that all move paths
            // are disjoint, which is expected by drop elaboration.
            let base_place =
                Place { local: place.local, projection: self.tcx.mk_place_elems(base) };
            let base_path = match self.move_path_for(base_place) {
                MovePathResult::Path(path) => path,
                MovePathResult::Union(path) => {
                    self.record_move(place, path);
                    return;
                }
                MovePathResult::Error => {
                    return;
                }
            };
            let base_ty = base_place.ty(self.body, self.tcx).ty;
            let len: u64 = match base_ty.kind() {
                ty::Array(_, size) => size
                    .try_to_target_usize(self.tcx)
                    .expect("expected subslice projection on fixed-size array"),
                _ => bug!("from_end: false slice pattern of non-array type"),
            };
            for offset in from..to {
                let elem =
                    ProjectionElem::ConstantIndex { offset, min_length: len, from_end: false };
                let path =
                    self.add_move_path(base_path, elem, |tcx| tcx.mk_place_elem(base_place, elem));
                self.record_move(place, path);
            }
        } else {
            match self.move_path_for(place) {
                MovePathResult::Path(path) | MovePathResult::Union(path) => {
                    self.record_move(place, path)
                }
                MovePathResult::Error => {}
            };
        }
    }

    fn record_move(&mut self, place: Place<'tcx>, path: MovePathIndex) {
        let move_out = self.data.moves.push(MoveOut { path, source: self.loc });
        debug!(
            "gather_move({:?}, {:?}): adding move {:?} of {:?}",
            self.loc, place, move_out, path
        );
        self.data.path_map[path].push(move_out);
        self.data.loc_map[self.loc].push(move_out);
    }

    fn gather_init(&mut self, place: PlaceRef<'tcx>, kind: InitKind) {
        debug!("gather_init({:?}, {:?})", self.loc, place);

        let mut place = place;

        // Check if we are assigning into a field of a union, if so, lookup the place
        // of the union so it is marked as initialized again.
        if let Some((place_base, ProjectionElem::Field(_, _))) = place.last_projection() {
            if place_base.ty(self.body, self.tcx).ty.is_union() {
                place = place_base;
            }
        }

        if let LookupResult::Exact(path) = self.data.rev_lookup.find(place) {
            let init = self.data.inits.push(Init {
                location: InitLocation::Statement(self.loc),
                path,
                kind,
            });

            debug!(
                "gather_init({:?}, {:?}): adding init {:?} of {:?}",
                self.loc, place, init, path
            );

            self.data.init_path_map[path].push(init);
            self.data.init_loc_map[self.loc].push(init);
        }
    }
}
