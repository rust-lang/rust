// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use rustc::ty::{self, TyCtxt, ParameterEnvironment};
use rustc::mir::*;
use rustc::util::nodemap::FxHashMap;
use rustc_data_structures::indexed_vec::{IndexVec};

use syntax::codemap::DUMMY_SP;

use std::collections::hash_map::Entry;
use std::fmt;
use std::mem;
use std::ops::{Index, IndexMut};

use super::abs_domain::{AbstractElem, Lift};

// This submodule holds some newtype'd Index wrappers that are using
// NonZero to ensure that Option<Index> occupies only a single word.
// They are in a submodule to impose privacy restrictions; namely, to
// ensure that other code does not accidentally access `index.0`
// (which is likely to yield a subtle off-by-one error).
mod indexes {
    use std::fmt;
    use core::nonzero::NonZero;
    use rustc_data_structures::indexed_vec::Idx;

    macro_rules! new_index {
        ($Index:ident, $debug_name:expr) => {
            #[derive(Copy, Clone, PartialEq, Eq, Hash)]
            pub struct $Index(NonZero<usize>);

            impl Idx for $Index {
                fn new(idx: usize) -> Self {
                    unsafe { $Index(NonZero::new(idx + 1)) }
                }
                fn index(self) -> usize {
                    *self.0 - 1
                }
            }

            impl fmt::Debug for $Index {
                fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                    write!(fmt, "{}{}", $debug_name, self.index())
                }
            }
        }
    }

    /// Index into MovePathData.move_paths
    new_index!(MovePathIndex, "mp");

    /// Index into MoveData.moves.
    new_index!(MoveOutIndex, "mo");
}

pub use self::indexes::MovePathIndex;
pub use self::indexes::MoveOutIndex;

impl self::indexes::MoveOutIndex {
    pub fn move_path_index(&self, move_data: &MoveData) -> MovePathIndex {
        move_data.moves[*self].path
    }
}

/// `MovePath` is a canonicalized representation of a path that is
/// moved or assigned to.
///
/// It follows a tree structure.
///
/// Given `struct X { m: M, n: N }` and `x: X`, moves like `drop x.m;`
/// move *out* of the l-value `x.m`.
///
/// The MovePaths representing `x.m` and `x.n` are siblings (that is,
/// one of them will link to the other via the `next_sibling` field,
/// and the other will have no entry in its `next_sibling` field), and
/// they both have the MovePath representing `x` as their parent.
#[derive(Clone)]
pub struct MovePath<'tcx> {
    pub next_sibling: Option<MovePathIndex>,
    pub first_child: Option<MovePathIndex>,
    pub parent: Option<MovePathIndex>,
    pub lvalue: Lvalue<'tcx>,
}

impl<'tcx> fmt::Debug for MovePath<'tcx> {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result {
        write!(w, "MovePath {{")?;
        if let Some(parent) = self.parent {
            write!(w, " parent: {:?},", parent)?;
        }
        if let Some(first_child) = self.first_child {
            write!(w, " first_child: {:?},", first_child)?;
        }
        if let Some(next_sibling) = self.next_sibling {
            write!(w, " next_sibling: {:?}", next_sibling)?;
        }
        write!(w, " lvalue: {:?} }}", self.lvalue)
    }
}

#[derive(Debug)]
pub struct MoveData<'tcx> {
    pub move_paths: IndexVec<MovePathIndex, MovePath<'tcx>>,
    pub moves: IndexVec<MoveOutIndex, MoveOut>,
    /// Each Location `l` is mapped to the MoveOut's that are effects
    /// of executing the code at `l`. (There can be multiple MoveOut's
    /// for a given `l` because each MoveOut is associated with one
    /// particular path being moved.)
    pub loc_map: LocationMap<Vec<MoveOutIndex>>,
    pub path_map: IndexVec<MovePathIndex, Vec<MoveOutIndex>>,
    pub rev_lookup: MovePathLookup<'tcx>,
}

pub trait HasMoveData<'tcx> {
    fn move_data(&self) -> &MoveData<'tcx>;
}

#[derive(Debug)]
pub struct LocationMap<T> {
    /// Location-indexed (BasicBlock for outer index, index within BB
    /// for inner index) map.
    map: IndexVec<BasicBlock, Vec<T>>,
}

impl<T> Index<Location> for LocationMap<T> {
    type Output = T;
    fn index(&self, index: Location) -> &Self::Output {
        &self.map[index.block][index.statement_index]
    }
}

impl<T> IndexMut<Location> for LocationMap<T> {
    fn index_mut(&mut self, index: Location) -> &mut Self::Output {
        &mut self.map[index.block][index.statement_index]
    }
}

impl<T> LocationMap<T> where T: Default + Clone {
    fn new(mir: &Mir) -> Self {
        LocationMap {
            map: mir.basic_blocks().iter().map(|block| {
                vec![T::default(); block.statements.len()+1]
            }).collect()
        }
    }
}

/// `MoveOut` represents a point in a program that moves out of some
/// L-value; i.e., "creates" uninitialized memory.
///
/// With respect to dataflow analysis:
/// - Generated by moves and declaration of uninitialized variables.
/// - Killed by assignments to the memory.
#[derive(Copy, Clone)]
pub struct MoveOut {
    /// path being moved
    pub path: MovePathIndex,
    /// location of move
    pub source: Location,
}

impl fmt::Debug for MoveOut {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}@{:?}", self.path, self.source)
    }
}

/// Tables mapping from an l-value to its MovePathIndex.
#[derive(Debug)]
pub struct MovePathLookup<'tcx> {
    locals: IndexVec<Local, MovePathIndex>,

    /// projections are made from a base-lvalue and a projection
    /// elem. The base-lvalue will have a unique MovePathIndex; we use
    /// the latter as the index into the outer vector (narrowing
    /// subsequent search so that it is solely relative to that
    /// base-lvalue). For the remaining lookup, we map the projection
    /// elem to the associated MovePathIndex.
    projections: FxHashMap<(MovePathIndex, AbstractElem<'tcx>), MovePathIndex>
}

struct MoveDataBuilder<'a, 'tcx: 'a> {
    mir: &'a Mir<'tcx>,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: &'a ParameterEnvironment<'tcx>,
    data: MoveData<'tcx>,
}

pub enum MovePathError {
    IllegalMove,
    UnionMove { path: MovePathIndex },
}

impl<'a, 'tcx> MoveDataBuilder<'a, 'tcx> {
    fn new(mir: &'a Mir<'tcx>,
           tcx: TyCtxt<'a, 'tcx, 'tcx>,
           param_env: &'a ParameterEnvironment<'tcx>)
           -> Self {
        let mut move_paths = IndexVec::new();
        let mut path_map = IndexVec::new();

        MoveDataBuilder {
            mir: mir,
            tcx: tcx,
            param_env: param_env,
            data: MoveData {
                moves: IndexVec::new(),
                loc_map: LocationMap::new(mir),
                rev_lookup: MovePathLookup {
                    locals: mir.local_decls.indices().map(Lvalue::Local).map(|v| {
                        Self::new_move_path(&mut move_paths, &mut path_map, None, v)
                    }).collect(),
                    projections: FxHashMap(),
                },
                move_paths: move_paths,
                path_map: path_map,
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
            parent: parent,
            lvalue: lvalue
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
    /// Maybe we should have seperate "borrowck" and "moveck" modes.
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
            ty::TyAdt(adt, _) if adt.has_dtor() =>
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

#[derive(Copy, Clone, Debug)]
pub enum LookupResult {
    Exact(MovePathIndex),
    Parent(Option<MovePathIndex>)
}

impl<'tcx> MovePathLookup<'tcx> {
    // Unlike the builder `fn move_path_for` below, this lookup
    // alternative will *not* create a MovePath on the fly for an
    // unknown l-value, but will rather return the nearest available
    // parent.
    pub fn find(&self, lval: &Lvalue<'tcx>) -> LookupResult {
        match *lval {
            Lvalue::Local(local) => LookupResult::Exact(self.locals[local]),
            Lvalue::Static(..) => LookupResult::Parent(None),
            Lvalue::Projection(ref proj) => {
                match self.find(&proj.base) {
                    LookupResult::Exact(base_path) => {
                        match self.projections.get(&(base_path, proj.elem.lift())) {
                            Some(&subpath) => LookupResult::Exact(subpath),
                            None => LookupResult::Parent(Some(base_path))
                        }
                    }
                    inexact => inexact
                }
            }
        }
    }
}

impl<'a, 'tcx> MoveData<'tcx> {
    pub fn gather_moves(mir: &Mir<'tcx>,
                        tcx: TyCtxt<'a, 'tcx, 'tcx>,
                        param_env: &ParameterEnvironment<'tcx>)
                        -> Self {
        gather_moves(mir, tcx, param_env)
    }
}

fn gather_moves<'a, 'tcx>(mir: &Mir<'tcx>,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          param_env: &ParameterEnvironment<'tcx>)
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
                self.gather_rvalue(loc, rval);
            }
            StatementKind::StorageLive(_) |
            StatementKind::StorageDead(_) => {}
            StatementKind::SetDiscriminant{ .. } => {
                span_bug!(stmt.source_info.span,
                          "SetDiscriminant should not exist during borrowck");
            }
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
            Rvalue::Len(..) |
            Rvalue::InlineAsm { .. } => {}
            Rvalue::Box(..) => {
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

            TerminatorKind::If { .. } |
            TerminatorKind::Assert { .. } |
            TerminatorKind::SwitchInt { .. } |
            TerminatorKind::Switch { .. } => {
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
