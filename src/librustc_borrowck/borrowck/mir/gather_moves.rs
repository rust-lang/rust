// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use rustc::ty::TyCtxt;
use rustc::mir::repr::*;
use rustc::util::nodemap::FnvHashMap;

use std::cell::{Cell};
use std::collections::hash_map::Entry;
use std::fmt;
use std::iter;
use std::ops::Index;

use super::dataflow::BitDenotation;
use super::abs_domain::{AbstractElem, Lift};

// This submodule holds some newtype'd Index wrappers that are using
// NonZero to ensure that Option<Index> occupies only a single word.
// They are in a submodule to impose privacy restrictions; namely, to
// ensure that other code does not accidentally access `index.0`
// (which is likely to yield a subtle off-by-one error).
mod indexes {
    use core::nonzero::NonZero;

    macro_rules! new_index {
        ($Index:ident) => {
            #[derive(Copy, Clone, PartialEq, Eq, Debug)]
            pub struct $Index(NonZero<usize>);

            impl $Index {
                pub fn new(idx: usize) -> Self {
                    unsafe { $Index(NonZero::new(idx + 1)) }
                }
                pub fn idx(&self) -> usize {
                    *self.0 - 1
                }
            }
        }
    }

    /// Index into MovePathData.move_paths
    new_index!(MovePathIndex);

    /// Index into MoveData.moves.
    new_index!(MoveOutIndex);
}

pub use self::indexes::MovePathIndex;
pub use self::indexes::MoveOutIndex;

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
    pub content: MovePathContent<'tcx>,
}

/// MovePaths usually represent a single l-value. The exceptions are
/// forms that arise due to erroneous input code: static data holds
/// l-values that we cannot actually move out of. Therefore we map
/// statics to a special marker value (`MovePathContent::Static`)
/// representing an invalid origin.
#[derive(Clone, Debug)]
pub enum MovePathContent<'tcx> {
    Lvalue(Lvalue<'tcx>),
    Static,
}

/// During construction of the MovePath's, we use PreMovePath to
/// represent accumulated state while we are gathering up all the
/// children of each path.
#[derive(Clone)]
struct PreMovePath<'tcx> {
    pub next_sibling: Option<MovePathIndex>,
    pub first_child: Cell<Option<MovePathIndex>>,
    pub parent: Option<MovePathIndex>,
    pub content: MovePathContent<'tcx>,
}

impl<'tcx> PreMovePath<'tcx> {
    fn into_move_path(self) -> MovePath<'tcx> {
        MovePath {
            next_sibling: self.next_sibling,
            parent: self.parent,
            content: self.content,
            first_child: self.first_child.get(),
        }
    }
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
        write!(w, " content: {:?} }}", self.content)
    }
}

pub struct MoveData<'tcx> {
    pub move_paths: MovePathData<'tcx>,
    pub moves: Vec<MoveOut>,
    pub loc_map: LocMap,
    pub path_map: PathMap,
    pub rev_lookup: MovePathLookup<'tcx>,
}

pub struct LocMap {
    /// Location-indexed (BasicBlock for outer index, index within BB
    /// for inner index) map to list of MoveOutIndex's.
    ///
    /// Each Location `l` is mapped to the MoveOut's that are effects
    /// of executing the code at `l`. (There can be multiple MoveOut's
    /// for a given `l` because each MoveOut is associated with one
    /// particular path being moved.)
    map: Vec<Vec<Vec<MoveOutIndex>>>,
}

impl Index<Location> for LocMap {
    type Output = [MoveOutIndex];
    fn index(&self, index: Location) -> &Self::Output {
        assert!(index.block.index() < self.map.len());
        assert!(index.index < self.map[index.block.index()].len());
        &self.map[index.block.index()][index.index]
    }
}

pub struct PathMap {
    /// Path-indexed map to list of MoveOutIndex's.
    ///
    /// Each Path `p` is mapped to the MoveOut's that move out of `p`.
    map: Vec<Vec<MoveOutIndex>>,
}

impl Index<MovePathIndex> for PathMap {
    type Output = [MoveOutIndex];
    fn index(&self, index: MovePathIndex) -> &Self::Output {
        &self.map[index.idx()]
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
        write!(fmt, "p{}@{:?}", self.path.idx(), self.source)
    }
}

#[derive(Copy, Clone)]
pub struct Location {
    /// block where action is located
    pub block: BasicBlock,
    /// index within above block; statement when < statments.len) or
    /// the terminator (when = statements.len).
    pub index: usize,
}

impl fmt::Debug for Location {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}[{}]", self.block, self.index)
    }
}

pub struct MovePathData<'tcx> {
    move_paths: Vec<MovePath<'tcx>>,
}

impl<'tcx> Index<MovePathIndex> for MovePathData<'tcx> {
    type Output = MovePath<'tcx>;
    fn index(&self, i: MovePathIndex) -> &MovePath<'tcx> {
        &self.move_paths[i.idx()]
    }
}

/// MovePathInverseMap maps from a uint in an lvalue-category to the
/// MovePathIndex for the MovePath for that lvalue.
type MovePathInverseMap = Vec<Option<MovePathIndex>>;

struct MovePathDataBuilder<'a, 'tcx: 'a> {
    mir: &'a Mir<'tcx>,
    pre_move_paths: Vec<PreMovePath<'tcx>>,
    rev_lookup: MovePathLookup<'tcx>,
}

/// Tables mapping from an l-value to its MovePathIndex.
pub struct MovePathLookup<'tcx> {
    vars: MovePathInverseMap,
    temps: MovePathInverseMap,
    args: MovePathInverseMap,

    /// The move path representing the return value is constructed
    /// lazily when we first encounter it in the input MIR.
    return_ptr: Option<MovePathIndex>,

    /// A single move path (representing any static data referenced)
    /// is constructed lazily when we first encounter statics in the
    /// input MIR.
    statics: Option<MovePathIndex>,

    /// projections are made from a base-lvalue and a projection
    /// elem. The base-lvalue will have a unique MovePathIndex; we use
    /// the latter as the index into the outer vector (narrowing
    /// subsequent search so that it is solely relative to that
    /// base-lvalue). For the remaining lookup, we map the projection
    /// elem to the associated MovePathIndex.
    projections: Vec<FnvHashMap<AbstractElem<'tcx>, MovePathIndex>>,

    /// Tracks the next index to allocate during construction of the
    /// MovePathData. Unused after MovePathData is fully constructed.
    next_index: MovePathIndex,
}

trait FillTo {
    type T;
    fn fill_to_with(&mut self, idx: usize, x: Self::T);
    fn fill_to(&mut self, idx: usize) where Self::T: Default {
        self.fill_to_with(idx, Default::default())
    }
}
impl<T:Clone> FillTo for Vec<T> {
    type T = T;
    fn fill_to_with(&mut self, idx: usize, x: T) {
        if idx >= self.len() {
            let delta = idx + 1 - self.len();
            assert_eq!(idx + 1, self.len() + delta);
            self.extend(iter::repeat(x).take(delta))
        }
        debug_assert!(idx < self.len());
    }
}

#[derive(Clone, Debug)]
enum LookupKind { Generate, Reuse }
struct Lookup<T>(LookupKind, T);

impl Lookup<MovePathIndex> {
    fn idx(&self) -> usize { (self.1).idx() }
}

impl<'tcx> MovePathLookup<'tcx> {
    fn new() -> Self {
        MovePathLookup {
            vars: vec![],
            temps: vec![],
            args: vec![],
            statics: None,
            return_ptr: None,
            projections: vec![],
            next_index: MovePathIndex::new(0),
        }
    }

    fn next_index(next: &mut MovePathIndex) -> MovePathIndex {
        let i = *next;
        *next = MovePathIndex::new(i.idx() + 1);
        i
    }

    fn lookup_or_generate(vec: &mut Vec<Option<MovePathIndex>>,
                          idx: u32,
                          next_index: &mut MovePathIndex) -> Lookup<MovePathIndex> {
        let idx = idx as usize;
        vec.fill_to_with(idx, None);
        let entry = &mut vec[idx];
        match *entry {
            None => {
                let i = Self::next_index(next_index);
                *entry = Some(i);
                Lookup(LookupKind::Generate, i)
            }
            Some(entry_idx) => {
                Lookup(LookupKind::Reuse, entry_idx)
            }
        }
    }

    fn lookup_var(&mut self, var_idx: u32) -> Lookup<MovePathIndex> {
        Self::lookup_or_generate(&mut self.vars,
                                 var_idx,
                                 &mut self.next_index)
    }

    fn lookup_temp(&mut self, temp_idx: u32) -> Lookup<MovePathIndex> {
        Self::lookup_or_generate(&mut self.temps,
                                 temp_idx,
                                 &mut self.next_index)
    }

    fn lookup_arg(&mut self, arg_idx: u32) -> Lookup<MovePathIndex> {
        Self::lookup_or_generate(&mut self.args,
                                 arg_idx,
                                 &mut self.next_index)
    }

    fn lookup_static(&mut self) -> Lookup<MovePathIndex> {
        match self.statics {
            Some(mpi) => {
                Lookup(LookupKind::Reuse, mpi)
            }
            ref mut ret @ None => {
                let mpi = Self::next_index(&mut self.next_index);
                *ret = Some(mpi);
                Lookup(LookupKind::Generate, mpi)
            }
        }
    }

    fn lookup_return_pointer(&mut self) -> Lookup<MovePathIndex> {
        match self.return_ptr {
            Some(mpi) => {
                Lookup(LookupKind::Reuse, mpi)
            }
            ref mut ret @ None => {
                let mpi = Self::next_index(&mut self.next_index);
                *ret = Some(mpi);
                Lookup(LookupKind::Generate, mpi)
            }
        }
    }

    fn lookup_proj(&mut self,
                   proj: &LvalueProjection<'tcx>,
                   base: MovePathIndex) -> Lookup<MovePathIndex> {
        let MovePathLookup { ref mut projections,
                             ref mut next_index, .. } = *self;
        projections.fill_to(base.idx());
        match projections[base.idx()].entry(proj.elem.lift()) {
            Entry::Occupied(ent) => {
                Lookup(LookupKind::Reuse, *ent.get())
            }
            Entry::Vacant(ent) => {
                let mpi = Self::next_index(next_index);
                ent.insert(mpi);
                Lookup(LookupKind::Generate, mpi)
            }
        }
    }
}

impl<'tcx> MovePathLookup<'tcx> {
    // Unlike the builder `fn move_path_for` below, this lookup
    // alternative will *not* create a MovePath on the fly for an
    // unknown l-value; it will simply panic.
    pub fn find(&self, lval: &Lvalue<'tcx>) -> MovePathIndex {
        match *lval {
            Lvalue::Var(var_idx) => self.vars[var_idx as usize].unwrap(),
            Lvalue::Temp(temp_idx) => self.temps[temp_idx as usize].unwrap(),
            Lvalue::Arg(arg_idx) => self.args[arg_idx as usize].unwrap(),
            Lvalue::Static(ref _def_id) => self.statics.unwrap(),
            Lvalue::ReturnPointer => self.return_ptr.unwrap(),
            Lvalue::Projection(ref proj) => {
                let base_index = self.find(&proj.base);
                self.projections[base_index.idx()][&proj.elem.lift()]
            }
        }
    }
}

impl<'a, 'tcx> MovePathDataBuilder<'a, 'tcx> {
    fn lookup(&mut self, lval: &Lvalue<'tcx>) -> Lookup<MovePathIndex> {
        let proj = match *lval {
            Lvalue::Var(var_idx) =>
                return self.rev_lookup.lookup_var(var_idx),
            Lvalue::Temp(temp_idx) =>
                return self.rev_lookup.lookup_temp(temp_idx),
            Lvalue::Arg(arg_idx) =>
                return self.rev_lookup.lookup_arg(arg_idx),
            Lvalue::Static(_def_id) =>
                return self.rev_lookup.lookup_static(),
            Lvalue::ReturnPointer =>
                return self.rev_lookup.lookup_return_pointer(),
            Lvalue::Projection(ref proj) => {
                proj
            }
        };

        let base_index = self.move_path_for(&proj.base);
        self.rev_lookup.lookup_proj(proj, base_index)
    }

    fn move_path_for(&mut self, lval: &Lvalue<'tcx>) -> MovePathIndex {
        let lookup: Lookup<MovePathIndex> = self.lookup(lval);

        // `lookup` is either the previously assigned index or a
        // newly-allocated one.
        debug_assert!(lookup.idx() <= self.pre_move_paths.len());

        if let Lookup(LookupKind::Generate, mpi) = lookup {
            let parent;
            let sibling;
            // tracks whether content is Some non-static; statics map to None.
            let content: Option<&Lvalue<'tcx>>;

            match *lval {
                Lvalue::Static(_) => {
                    content = None;
                    sibling = None;
                    parent = None;
                }

                Lvalue::Var(_) | Lvalue::Temp(_) | Lvalue::Arg(_) |
                Lvalue::ReturnPointer => {
                    content = Some(lval);
                    sibling = None;
                    parent = None;
                }
                Lvalue::Projection(ref proj) => {
                    content = Some(lval);

                    // Here, install new MovePath as new first_child.

                    // Note: `parent` previously allocated (Projection
                    // case of match above established this).
                    let idx = self.move_path_for(&proj.base);
                    parent = Some(idx);

                    let parent_move_path = &mut self.pre_move_paths[idx.idx()];

                    // At last: Swap in the new first_child.
                    sibling = parent_move_path.first_child.get();
                    parent_move_path.first_child.set(Some(mpi));
                }
            };

            let content = match content {
                Some(lval) => MovePathContent::Lvalue(lval.clone()),
                None => MovePathContent::Static,
            };

            let move_path = PreMovePath {
                next_sibling: sibling,
                parent: parent,
                content: content,
                first_child: Cell::new(None),
            };

            self.pre_move_paths.push(move_path);
        }

        return lookup.1;
    }
}

impl<'tcx> MoveData<'tcx> {
    pub fn gather_moves(mir: &Mir<'tcx>, tcx: &TyCtxt<'tcx>) -> Self {
        gather_moves(mir, tcx)
    }
}

#[derive(Debug)]
enum StmtKind {
    Use, Repeat, Cast, BinaryOp, UnaryOp, Box,
    Aggregate, Drop, CallFn, CallArg, Return,
}

fn gather_moves<'tcx>(mir: &Mir<'tcx>, tcx: &TyCtxt<'tcx>) -> MoveData<'tcx> {
    use self::StmtKind as SK;

    let bbs = mir.all_basic_blocks();
    let mut moves = Vec::with_capacity(bbs.len());
    let mut loc_map: Vec<_> = iter::repeat(Vec::new()).take(bbs.len()).collect();
    let mut path_map = Vec::new();

    // this is mutable only because we will move it to and fro' the
    // BlockContexts constructed on each iteration. (Moving is more
    // straight-forward than mutable borrows in this instance.)
    let mut builder = MovePathDataBuilder {
        mir: mir,
        pre_move_paths: Vec::new(),
        rev_lookup: MovePathLookup::new(),
    };

    for bb in bbs {
        let loc_map_bb = &mut loc_map[bb.index()];
        let bb_data = mir.basic_block_data(bb);

        debug_assert!(loc_map_bb.len() == 0);
        let len = bb_data.statements.len();
        loc_map_bb.fill_to(len);
        debug_assert!(loc_map_bb.len() == len + 1);

        let mut bb_ctxt = BlockContext {
            tcx: tcx,
            moves: &mut moves,
            builder: builder,
            path_map: &mut path_map,
            loc_map_bb: loc_map_bb,
        };

        for (i, stmt) in bb_data.statements.iter().enumerate() {
            let source = Location { block: bb, index: i };
            match stmt.kind {
                StatementKind::Assign(ref lval, ref rval) => {
                    // ensure MovePath created for `lval`.
                    bb_ctxt.builder.move_path_for(lval);

                    match *rval {
                        Rvalue::Use(ref operand) => {
                            bb_ctxt.on_operand(SK::Use, operand, source)
                        }
                        Rvalue::Repeat(ref operand, ref _const) =>
                            bb_ctxt.on_operand(SK::Repeat, operand, source),
                        Rvalue::Cast(ref _kind, ref operand, ref _ty) =>
                            bb_ctxt.on_operand(SK::Cast, operand, source),
                        Rvalue::BinaryOp(ref _binop, ref operand1, ref operand2) => {
                            bb_ctxt.on_operand(SK::BinaryOp, operand1, source);
                            bb_ctxt.on_operand(SK::BinaryOp, operand2, source);
                        }
                        Rvalue::UnaryOp(ref _unop, ref operand) => {
                            bb_ctxt.on_operand(SK::UnaryOp, operand, source);
                        }
                        Rvalue::Box(ref _ty) => {
                            // this is creating uninitialized
                            // memory that needs to be initialized.
                            let deref_lval = Lvalue::Projection(Box::new(Projection {
                                base: lval.clone(),
                                elem: ProjectionElem::Deref,
                            }));
                            bb_ctxt.on_move_out_lval(SK::Box, &deref_lval, source);
                        }
                        Rvalue::Aggregate(ref _kind, ref operands) => {
                            for operand in operands {
                                bb_ctxt.on_operand(SK::Aggregate, operand, source);
                            }
                        }
                        Rvalue::Ref(..) |
                        Rvalue::Len(..) |
                        Rvalue::InlineAsm { .. } => {}

                        Rvalue::Slice {..} => {
                            bug!("cannot move out of slice");
                        }
                    }
                }
            }
        }

        match bb_data.terminator().kind {
            TerminatorKind::Goto { target: _ } | TerminatorKind::Resume => { }

            TerminatorKind::Return => {
                let source = Location { block: bb,
                                        index: bb_data.statements.len() };
                let lval = &Lvalue::ReturnPointer.deref();
                bb_ctxt.on_move_out_lval(SK::Return, lval, source);
            }

            TerminatorKind::If { ref cond, targets: _ } => {
                // The `cond` is always of (copyable) type `bool`,
                // so there will never be anything to move.
                let _ = cond;
            }

            TerminatorKind::SwitchInt { switch_ty: _, values: _, targets: _, ref discr } |
            TerminatorKind::Switch { adt_def: _, targets: _, ref discr } => {
                // The `discr` is not consumed; that is instead
                // encoded on specific match arms (and for
                // SwitchInt`, it is always a copyable integer
                // type anyway).
                let _ = discr;
            }

            TerminatorKind::Drop { value: ref lval, target: _, unwind: _ } => {
                let source = Location { block: bb,
                                        index: bb_data.statements.len() };
                bb_ctxt.on_move_out_lval(SK::Drop, lval, source);
            }

            TerminatorKind::Call { ref func, ref args, ref destination, cleanup: _ } => {
                let source = Location { block: bb,
                                        index: bb_data.statements.len() };
                bb_ctxt.on_operand(SK::CallFn, func, source);
                for arg in args {
                    bb_ctxt.on_operand(SK::CallArg, arg, source);
                }
                if let Some((ref destination, _bb)) = *destination {
                    // Create MovePath for `destination`, then
                    // discard returned index.
                    bb_ctxt.builder.move_path_for(destination);
                }
            }
        }

        builder = bb_ctxt.builder;
    }

    // At this point, we may have created some MovePaths that do not
    // have corresponding entries in the path map.
    //
    // (For example, creating the path `a.b.c` may, as a side-effect,
    // create a path for the parent path `a.b`.)
    //
    // All such paths were not referenced ...
    //
    // well you know, lets actually try just asserting that the path map *is* complete.
    assert_eq!(path_map.len(), builder.pre_move_paths.len());
    path_map.fill_to(builder.pre_move_paths.len() - 1);

    let pre_move_paths = builder.pre_move_paths;
    let move_paths: Vec<_> = pre_move_paths.into_iter()
        .map(|p| p.into_move_path())
        .collect();

    debug!("{}", {
        let mut seen: Vec<_> = move_paths.iter().map(|_| false).collect();
        for (j, &MoveOut { ref path, ref source }) in moves.iter().enumerate() {
            debug!("MovePathData moves[{}]: MoveOut {{ path: {:?} = {:?}, source: {:?} }}",
                   j, path, move_paths[path.idx()], source);
            seen[path.idx()] = true;
        }
        for (j, path) in move_paths.iter().enumerate() {
            if !seen[j] {
                debug!("MovePathData move_paths[{}]: {:?}", j, path);
            }
        }
        "done dumping MovePathData"
    });

    MoveData {
        move_paths: MovePathData { move_paths: move_paths, },
        moves: moves,
        loc_map: LocMap { map: loc_map },
        path_map: PathMap { map: path_map },
        rev_lookup: builder.rev_lookup,
    }
}

struct BlockContext<'b, 'a: 'b, 'tcx: 'a> {
    tcx: &'b TyCtxt<'tcx>,
    moves: &'b mut Vec<MoveOut>,
    builder: MovePathDataBuilder<'a, 'tcx>,
    path_map: &'b mut Vec<Vec<MoveOutIndex>>,
    loc_map_bb: &'b mut Vec<Vec<MoveOutIndex>>,
}

impl<'b, 'a: 'b, 'tcx: 'a> BlockContext<'b, 'a, 'tcx> {
    fn on_move_out_lval(&mut self,
                        stmt_kind: StmtKind,
                        lval: &Lvalue<'tcx>,
                        source: Location) {
        let tcx = self.tcx;
        let lval_ty = self.builder.mir.lvalue_ty(tcx, lval);

        // FIXME: does lvalue_ty ever return TyError, or is it
        // guaranteed to always return non-Infer/non-Error values?

        // This code is just trying to avoid creating a MoveOut
        // entry for values that do not need move semantics.
        //
        // type_contents is imprecise (may claim needs drop for
        // types that in fact have no destructor). But that is
        // still usable for our purposes here.
        let consumed = lval_ty.to_ty(tcx).type_contents(tcx).needs_drop(tcx);

        if !consumed {
            debug!("ctxt: {:?} no consume of lval: {:?} of type {:?}",
                   stmt_kind, lval, lval_ty);
            return;
        }
        let i = source.index;
        let index = MoveOutIndex::new(self.moves.len());

        let path = self.builder.move_path_for(lval);
        self.moves.push(MoveOut { path: path, source: source.clone() });
        self.path_map.fill_to(path.idx());

        debug!("ctxt: {:?} add consume of lval: {:?} \
                at index: {:?} \
                to path_map for path: {:?} and \
                to loc_map for loc: {:?}",
               stmt_kind, lval, index, path, source);

        debug_assert!(path.idx() < self.path_map.len());
        // this is actually a questionable assert; at the very
        // least, incorrect input code can probably cause it to
        // fire.
        assert!(self.path_map[path.idx()].iter().find(|idx| **idx == index).is_none());
        self.path_map[path.idx()].push(index);

        debug_assert!(i < self.loc_map_bb.len());
        debug_assert!(self.loc_map_bb[i].iter().find(|idx| **idx == index).is_none());
        self.loc_map_bb[i].push(index);
    }

    fn on_operand(&mut self, stmt_kind: StmtKind, operand: &Operand<'tcx>, source: Location) {
        match *operand {
            Operand::Constant(..) => {} // not-a-move
            Operand::Consume(ref lval) => { // a move
                self.on_move_out_lval(stmt_kind, lval, source);
            }
        }
    }
}

impl<'tcx> BitDenotation for MoveData<'tcx>{
    type Bit = MoveOut;
    fn bits_per_block(&self) -> usize {
        self.moves.len()
    }
    fn interpret(&self, idx: usize) -> &Self::Bit {
        &self.moves[idx]
    }
}
