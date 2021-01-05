use core::slice::Iter;
use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::{Enumerated, IndexVec};
use rustc_middle::mir::*;
use rustc_middle::ty::{ParamEnv, Ty, TyCtxt};
use rustc_span::Span;
use smallvec::SmallVec;

use std::fmt;
use std::ops::{Index, IndexMut};

use self::abs_domain::{AbstractElem, Lift};

mod abs_domain;

rustc_index::newtype_index! {
    pub struct MovePathIndex {
        DEBUG_FORMAT = "mp{}"
    }
}

impl polonius_engine::Atom for MovePathIndex {
    fn index(self) -> usize {
        rustc_index::vec::Idx::index(self)
    }
}

rustc_index::newtype_index! {
    pub struct MoveOutIndex {
        DEBUG_FORMAT = "mo{}"
    }
}

rustc_index::newtype_index! {
    pub struct InitIndex {
        DEBUG_FORMAT = "in{}"
    }
}

impl MoveOutIndex {
    pub fn move_path_index(&self, move_data: &MoveData<'_>) -> MovePathIndex {
        move_data.moves[*self].path
    }
}

/// `MovePath` is a canonicalized representation of a path that is
/// moved or assigned to.
///
/// It follows a tree structure.
///
/// Given `struct X { m: M, n: N }` and `x: X`, moves like `drop x.m;`
/// move *out* of the place `x.m`.
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
    pub place: Place<'tcx>,
}

impl<'tcx> MovePath<'tcx> {
    /// Returns an iterator over the parents of `self`.
    pub fn parents<'a>(
        &self,
        move_paths: &'a IndexVec<MovePathIndex, MovePath<'tcx>>,
    ) -> impl 'a + Iterator<Item = (MovePathIndex, &'a MovePath<'tcx>)> {
        let first = self.parent.map(|mpi| (mpi, &move_paths[mpi]));
        MovePathLinearIter {
            next: first,
            fetch_next: move |_, parent: &MovePath<'_>| {
                parent.parent.map(|mpi| (mpi, &move_paths[mpi]))
            },
        }
    }

    /// Returns an iterator over the immediate children of `self`.
    pub fn children<'a>(
        &self,
        move_paths: &'a IndexVec<MovePathIndex, MovePath<'tcx>>,
    ) -> impl 'a + Iterator<Item = (MovePathIndex, &'a MovePath<'tcx>)> {
        let first = self.first_child.map(|mpi| (mpi, &move_paths[mpi]));
        MovePathLinearIter {
            next: first,
            fetch_next: move |_, child: &MovePath<'_>| {
                child.next_sibling.map(|mpi| (mpi, &move_paths[mpi]))
            },
        }
    }

    /// Finds the closest descendant of `self` for which `f` returns `true` using a breadth-first
    /// search.
    ///
    /// `f` will **not** be called on `self`.
    pub fn find_descendant(
        &self,
        move_paths: &IndexVec<MovePathIndex, MovePath<'_>>,
        f: impl Fn(MovePathIndex) -> bool,
    ) -> Option<MovePathIndex> {
        let mut todo = if let Some(child) = self.first_child {
            vec![child]
        } else {
            return None;
        };

        while let Some(mpi) = todo.pop() {
            if f(mpi) {
                return Some(mpi);
            }

            let move_path = &move_paths[mpi];
            if let Some(child) = move_path.first_child {
                todo.push(child);
            }

            // After we've processed the original `mpi`, we should always
            // traverse the siblings of any of its children.
            if let Some(sibling) = move_path.next_sibling {
                todo.push(sibling);
            }
        }

        None
    }
}

impl<'tcx> fmt::Debug for MovePath<'tcx> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        write!(w, " place: {:?} }}", self.place)
    }
}

impl<'tcx> fmt::Display for MovePath<'tcx> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(w, "{:?}", self.place)
    }
}

struct MovePathLinearIter<'a, 'tcx, F> {
    next: Option<(MovePathIndex, &'a MovePath<'tcx>)>,
    fetch_next: F,
}

impl<'a, 'tcx, F> Iterator for MovePathLinearIter<'a, 'tcx, F>
where
    F: FnMut(MovePathIndex, &'a MovePath<'tcx>) -> Option<(MovePathIndex, &'a MovePath<'tcx>)>,
{
    type Item = (MovePathIndex, &'a MovePath<'tcx>);

    fn next(&mut self) -> Option<Self::Item> {
        let ret = self.next.take()?;
        self.next = (self.fetch_next)(ret.0, ret.1);
        Some(ret)
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
    pub loc_map: LocationMap<SmallVec<[MoveOutIndex; 4]>>,
    pub path_map: IndexVec<MovePathIndex, SmallVec<[MoveOutIndex; 4]>>,
    pub rev_lookup: MovePathLookup,
    pub inits: IndexVec<InitIndex, Init>,
    /// Each Location `l` is mapped to the Inits that are effects
    /// of executing the code at `l`.
    pub init_loc_map: LocationMap<SmallVec<[InitIndex; 4]>>,
    pub init_path_map: IndexVec<MovePathIndex, SmallVec<[InitIndex; 4]>>,
}

pub trait HasMoveData<'tcx> {
    fn move_data(&self) -> &MoveData<'tcx>;
}

#[derive(Debug)]
pub struct LocationMap<T> {
    /// Location-indexed (BasicBlock for outer index, index within BB
    /// for inner index) map.
    pub(crate) map: IndexVec<BasicBlock, Vec<T>>,
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

impl<T> LocationMap<T>
where
    T: Default + Clone,
{
    fn new(body: &Body<'_>) -> Self {
        LocationMap {
            map: body
                .basic_blocks()
                .iter()
                .map(|block| vec![T::default(); block.statements.len() + 1])
                .collect(),
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
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{:?}@{:?}", self.path, self.source)
    }
}

/// `Init` represents a point in a program that initializes some L-value;
#[derive(Copy, Clone)]
pub struct Init {
    /// path being initialized
    pub path: MovePathIndex,
    /// location of initialization
    pub location: InitLocation,
    /// Extra information about this initialization
    pub kind: InitKind,
}

/// Initializations can be from an argument or from a statement. Arguments
/// do not have locations, in those cases the `Local` is kept..
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InitLocation {
    Argument(Local),
    Statement(Location),
}

/// Additional information about the initialization.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InitKind {
    /// Deep init, even on panic
    Deep,
    /// Only does a shallow init
    Shallow,
    /// This doesn't initialize the variable on panic (and a panic is possible).
    NonPanicPathOnly,
}

impl fmt::Debug for Init {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{:?}@{:?} ({:?})", self.path, self.location, self.kind)
    }
}

impl Init {
    pub fn span<'tcx>(&self, body: &Body<'tcx>) -> Span {
        match self.location {
            InitLocation::Argument(local) => body.local_decls[local].source_info.span,
            InitLocation::Statement(location) => body.source_info(location).span,
        }
    }
}

/// Tables mapping from a place to its MovePathIndex.
#[derive(Debug)]
pub struct MovePathLookup {
    locals: IndexVec<Local, MovePathIndex>,

    /// projections are made from a base-place and a projection
    /// elem. The base-place will have a unique MovePathIndex; we use
    /// the latter as the index into the outer vector (narrowing
    /// subsequent search so that it is solely relative to that
    /// base-place). For the remaining lookup, we map the projection
    /// elem to the associated MovePathIndex.
    projections: FxHashMap<(MovePathIndex, AbstractElem), MovePathIndex>,
}

mod builder;

#[derive(Copy, Clone, Debug)]
pub enum LookupResult {
    Exact(MovePathIndex),
    Parent(Option<MovePathIndex>),
}

impl MovePathLookup {
    // Unlike the builder `fn move_path_for` below, this lookup
    // alternative will *not* create a MovePath on the fly for an
    // unknown place, but will rather return the nearest available
    // parent.
    pub fn find(&self, place: PlaceRef<'_>) -> LookupResult {
        let mut result = self.locals[place.local];

        for elem in place.projection.iter() {
            if let Some(&subpath) = self.projections.get(&(result, elem.lift())) {
                result = subpath;
            } else {
                return LookupResult::Parent(Some(result));
            }
        }

        LookupResult::Exact(result)
    }

    pub fn find_local(&self, local: Local) -> MovePathIndex {
        self.locals[local]
    }

    /// An enumerated iterator of `local`s and their associated
    /// `MovePathIndex`es.
    pub fn iter_locals_enumerated(&self) -> Enumerated<Local, Iter<'_, MovePathIndex>> {
        self.locals.iter_enumerated()
    }
}

#[derive(Debug)]
pub struct IllegalMoveOrigin<'tcx> {
    pub location: Location,
    pub kind: IllegalMoveOriginKind<'tcx>,
}

#[derive(Debug)]
pub enum IllegalMoveOriginKind<'tcx> {
    /// Illegal move due to attempt to move from behind a reference.
    BorrowedContent {
        /// The place the reference refers to: if erroneous code was trying to
        /// move from `(*x).f` this will be `*x`.
        target_place: Place<'tcx>,
    },

    /// Illegal move due to attempt to move from field of an ADT that
    /// implements `Drop`. Rust maintains invariant that all `Drop`
    /// ADT's remain fully-initialized so that user-defined destructor
    /// can safely read from all of the ADT's fields.
    InteriorOfTypeWithDestructor { container_ty: Ty<'tcx> },

    /// Illegal move due to attempt to move out of a slice or array.
    InteriorOfSliceOrArray { ty: Ty<'tcx>, is_index: bool },
}

#[derive(Debug)]
pub enum MoveError<'tcx> {
    IllegalMove { cannot_move_out_of: IllegalMoveOrigin<'tcx> },
    UnionMove { path: MovePathIndex },
}

impl<'tcx> MoveError<'tcx> {
    fn cannot_move_out_of(location: Location, kind: IllegalMoveOriginKind<'tcx>) -> Self {
        let origin = IllegalMoveOrigin { location, kind };
        MoveError::IllegalMove { cannot_move_out_of: origin }
    }
}

impl<'tcx> MoveData<'tcx> {
    pub fn gather_moves(
        body: &Body<'tcx>,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
    ) -> Result<Self, (Self, Vec<(Place<'tcx>, MoveError<'tcx>)>)> {
        builder::gather_moves(body, tcx, param_env)
    }

    /// For the move path `mpi`, returns the root local variable (if any) that starts the path.
    /// (e.g., for a path like `a.b.c` returns `Some(a)`)
    pub fn base_local(&self, mut mpi: MovePathIndex) -> Option<Local> {
        loop {
            let path = &self.move_paths[mpi];
            if let Some(l) = path.place.as_local() {
                return Some(l);
            }
            if let Some(parent) = path.parent {
                mpi = parent;
                continue;
            } else {
                return None;
            }
        }
    }

    pub fn find_in_move_path_or_its_descendants(
        &self,
        root: MovePathIndex,
        pred: impl Fn(MovePathIndex) -> bool,
    ) -> Option<MovePathIndex> {
        if pred(root) {
            return Some(root);
        }

        self.move_paths[root].find_descendant(&self.move_paths, pred)
    }
}
