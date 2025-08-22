//! The move-analysis portion of borrowck needs to work in an abstract domain of lifted `Place`s.
//! Most of the `Place` variants fall into a one-to-one mapping between the concrete and abstract
//! (e.g., a field projection on a local variable, `x.field`, has the same meaning in both
//! domains). In other words, all field projections for the same field on the same local do not
//! have meaningfully different types if ever. Indexed projections are the exception: `a[x]` needs
//! to be treated as mapping to the same move path as `a[y]` as well as `a[13]`, etc. So we map
//! these `x`/`y` values to `()`.
//!
//! (In theory, the analysis could be extended to work with sets of paths, so that `a[0]` and
//! `a[13]` could be kept distinct, while `a[x]` would still overlap them both. But that is not
//! what this representation does today.)

use std::fmt;
use std::ops::{Index, IndexMut};

use rustc_data_structures::fx::FxHashMap;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::mir::*;
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_span::Span;
use smallvec::SmallVec;

use crate::un_derefer::UnDerefer;

rustc_index::newtype_index! {
    #[orderable]
    #[debug_format = "mp{}"]
    pub struct MovePathIndex {}
}

impl polonius_engine::Atom for MovePathIndex {
    fn index(self) -> usize {
        rustc_index::Idx::index(self)
    }
}

rustc_index::newtype_index! {
    #[orderable]
    #[debug_format = "mo{}"]
    pub struct MoveOutIndex {}
}

rustc_index::newtype_index! {
    #[debug_format = "in{}"]
    pub struct InitIndex {}
}

impl MoveOutIndex {
    pub fn move_path_index(self, move_data: &MoveData<'_>) -> MovePathIndex {
        move_data.moves[self].path
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
        move_paths: &'a IndexSlice<MovePathIndex, MovePath<'tcx>>,
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
        move_paths: &'a IndexSlice<MovePathIndex, MovePath<'tcx>>,
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
        move_paths: &IndexSlice<MovePathIndex, MovePath<'_>>,
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
            write!(w, " parent: {parent:?},")?;
        }
        if let Some(first_child) = self.first_child {
            write!(w, " first_child: {first_child:?},")?;
        }
        if let Some(next_sibling) = self.next_sibling {
            write!(w, " next_sibling: {next_sibling:?}")?;
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
    pub rev_lookup: MovePathLookup<'tcx>,
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
                .basic_blocks
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
pub struct MovePathLookup<'tcx> {
    locals: IndexVec<Local, Option<MovePathIndex>>,

    /// projections are made from a base-place and a projection
    /// elem. The base-place will have a unique MovePathIndex; we use
    /// the latter as the index into the outer vector (narrowing
    /// subsequent search so that it is solely relative to that
    /// base-place). For the remaining lookup, we map the projection
    /// elem to the associated MovePathIndex.
    projections: FxHashMap<(MovePathIndex, ProjectionKind), MovePathIndex>,

    un_derefer: UnDerefer<'tcx>,
}

mod builder;

#[derive(Copy, Clone, Debug)]
pub enum LookupResult {
    Exact(MovePathIndex),
    Parent(Option<MovePathIndex>),
}

impl<'tcx> MovePathLookup<'tcx> {
    // Unlike the builder `fn move_path_for` below, this lookup
    // alternative will *not* create a MovePath on the fly for an
    // unknown place, but will rather return the nearest available
    // parent.
    pub fn find(&self, place: PlaceRef<'tcx>) -> LookupResult {
        let Some(mut result) = self.find_local(place.local) else {
            return LookupResult::Parent(None);
        };

        for (_, elem) in self.un_derefer.iter_projections(place) {
            if let Some(&subpath) = self.projections.get(&(result, elem.kind())) {
                result = subpath;
            } else {
                return LookupResult::Parent(Some(result));
            }
        }

        LookupResult::Exact(result)
    }

    #[inline]
    pub fn find_local(&self, local: Local) -> Option<MovePathIndex> {
        self.locals[local]
    }

    /// An enumerated iterator of `local`s and their associated
    /// `MovePathIndex`es.
    pub fn iter_locals_enumerated(
        &self,
    ) -> impl DoubleEndedIterator<Item = (Local, MovePathIndex)> {
        self.locals.iter_enumerated().filter_map(|(l, &idx)| Some((l, idx?)))
    }
}

impl<'tcx> MoveData<'tcx> {
    pub fn gather_moves(
        body: &Body<'tcx>,
        tcx: TyCtxt<'tcx>,
        filter: impl Fn(Ty<'tcx>) -> bool,
    ) -> MoveData<'tcx> {
        builder::gather_moves(body, tcx, filter)
    }

    /// For the move path `mpi`, returns the root local variable that starts the path.
    /// (e.g., for a path like `a.b.c` returns `a`)
    pub fn base_local(&self, mut mpi: MovePathIndex) -> Local {
        loop {
            let path = &self.move_paths[mpi];
            if let Some(l) = path.place.as_local() {
                return l;
            }
            mpi = path.parent.expect("root move paths should be locals");
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
