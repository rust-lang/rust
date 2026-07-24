//! [`MovePath`]s track the initialization state of places and their sub-paths.

use std::fmt;
use std::ops::{Index, IndexMut};

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_data_structures::fx::FxHashMap;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::mir::*;
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_span::Span;
use smallvec::SmallVec;

use crate::un_derefer::UnDerefer;

rustc_index::newtype_index! {
    /// Index identifying a `MovePath`.
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
    /// Index identifying a `MoveOut`.
    #[orderable]
    #[debug_format = "mo{}"]
    pub struct MoveOutIndex {}
}

rustc_index::newtype_index! {
    /// Index identifying an `Init`.
    #[debug_format = "in{}"]
    pub struct InitIndex {}
}

impl MoveOutIndex {
    pub fn move_path_index(self, move_data: &MoveData<'_>) -> MovePathIndex {
        move_data.move_outs[self].path
    }
}

/// `MovePath` is a canonicalized representation of a place that is of
/// interest to dataflow analysis, as identified by `gather_moves`. This
/// is primarily places that are moved or inited (assigned). Each
/// `MovePath` is assigned a `MovePathIndex` by which it can be referred
/// to.
///
/// `MovePath` follows a tree structure.
///
/// Given `struct X { m: M, n: N }` and `x: X`, moves like `drop x.m;`
/// move *out* of the place `x.m`.
///
/// The MovePaths representing `x.m` and `x.n` are siblings (that is,
/// one of them will link to the other via the `next_sibling` field,
/// and the other will have no entry in its `next_sibling` field), and
/// they both have the MovePath representing `x` as their parent.
/// (All tree roots are locals). This structure allows easy traversal
/// between related paths `x` and `x.m` and `x.n`.
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
        let Some(child) = self.first_child else { return None };
        let mut todo = vec![child];

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
    /// All the gathered `MovePath`s.
    pub move_paths: IndexVec<MovePathIndex, MovePath<'tcx>>,

    /// All the `MoveOut`s.
    pub move_outs: IndexVec<MoveOutIndex, MoveOut>,
    /// Map from locations to `MoveOut`s. `SmallVec` because each location might cause more than
    /// one `MoveOut`. Used during analysis and diagnostics.
    pub move_out_loc_map: LocationMap<SmallVec<[MoveOutIndex; 4]>>,
    /// Map from `MovePath`s (places) to `MoveOuts`. `SmallVec` because each `MovePath` may be
    /// moved-out of more than once. Used mostly for diagnostics.
    pub move_out_path_map: IndexVec<MovePathIndex, SmallVec<[MoveOutIndex; 4]>>,

    /// Map from places/locals to `MovePath`s.
    pub rev_lookup: MovePathLookup<'tcx>,

    /// All the `Init`s.
    pub inits: IndexVec<InitIndex, Init>,
    /// Map from locations to `Init`s. `SmallVec` because each location might cause more than one
    /// `Init`, though more than one is very rare (e.g. inline asm).
    pub init_loc_map: LocationMap<SmallVec<[InitIndex; 1]>>,
    /// Map from `MovePath`s (places) to `Init`s. `SmallVec` because each `MovePath` (place) might
    /// be inited more than once.
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
/// L-value; i.e., "creates" uninitialized memory. The dual of `Init`.
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

/// `Init` represents a point in a program that initializes some L-value. The dual of `MoveOut`.
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
/// do not have locations, in those cases the `Local` is kept.
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

/// Tables mapping from a place to its `MovePathIndex`.
#[derive(Debug)]
pub struct MovePathLookup<'tcx> {
    locals: IndexVec<Local, Option<MovePathIndex>>,

    /// projections are made from a base-place and a projection
    /// elem. The base-place will have a unique MovePathIndex; we use
    /// the latter as the index into the outer vector (narrowing
    /// subsequent search so that it is solely relative to that
    /// base-place). For the remaining lookup, we map the projection
    /// elem to the associated MovePathIndex.
    projections: FxHashMap<(MovePathIndex, MoveSubPath), MovePathIndex>,

    un_derefer: UnDerefer<'tcx>,
}

mod builder;

#[derive(Copy, Clone, Debug)]
pub enum LookupResult {
    /// This exact thing has a move path. E.g. we looked up `x` or `x.m` and it has been moved.
    Exact(MovePathIndex),

    /// - If the field is `None`, neither the exact thing nor any ancestor of it has a move path.
    ///   E.g. we looked up `x.m` and neither it nor `x` have a move path.
    /// - If the field is `Some`, the exact thing has no move path, but an ancestor does. E.g. we
    ///   looked up `x.m` which has no move path but `x` has one. Not possible for locals.
    Parent(Option<MovePathIndex>),
}

impl<'tcx> MovePathLookup<'tcx> {
    // Unlike the builder `fn move_path_for` below, this lookup
    // alternative will *not* create a MovePath on the fly for an
    // unknown place, but will rather return the nearest available
    // parent.
    pub fn find(&self, place: PlaceRef<'tcx>) -> LookupResult {
        // Look first in the locals (roots).
        let Some(mut result) = self.find_local(place.local) else {
            return LookupResult::Parent(None);
        };

        // Look for a projection through the found local.
        for (_, elem) in self.un_derefer.iter_projections(place) {
            let subpath = match MoveSubPath::of(elem.kind()) {
                MoveSubPathResult::One(kind) => self.projections.get(&(result, kind)),
                MoveSubPathResult::Subslice { .. } => None, // just use the parent MovePath
                MoveSubPathResult::Skip => continue,
                MoveSubPathResult::Stop => None,
            };

            let Some(&subpath) = subpath else {
                return LookupResult::Parent(Some(result));
            };
            result = subpath;
        }

        LookupResult::Exact(result)
    }

    /// For locals, which are roots.
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

/// A projection into a move path producing a child path
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum MoveSubPath {
    Deref,
    Field(FieldIdx),
    ConstantIndex(u64),
    Downcast(VariantIdx),
    UnwrapUnsafeBinder,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MoveSubPathResult {
    One(MoveSubPath),
    Subslice { from: u64, to: u64 },
    Skip,
    Stop,
}

impl MoveSubPath {
    fn of(elem: ProjectionKind) -> MoveSubPathResult {
        let subpath = match elem {
            // correspond to a MoveSubPath
            ProjectionKind::Deref => MoveSubPath::Deref,
            ProjectionKind::Field(idx, _) => MoveSubPath::Field(idx),
            ProjectionKind::ConstantIndex { offset, min_length: _, from_end: false } => {
                MoveSubPath::ConstantIndex(offset)
            }
            ProjectionKind::Downcast(_, idx) => MoveSubPath::Downcast(idx),
            ProjectionKind::UnwrapUnsafeBinder(_) => MoveSubPath::UnwrapUnsafeBinder,

            // this should be the same move path as its parent
            // its fine to skip because it cannot have sibling move paths
            // and it is not a user visible path
            ProjectionKind::OpaqueCast(_) => {
                return MoveSubPathResult::Skip;
            }

            // these cannot be moved through
            ProjectionKind::Index(_)
            | ProjectionKind::ConstantIndex { offset: _, min_length: _, from_end: true }
            | ProjectionKind::Subslice { from: _, to: _, from_end: true } => {
                return MoveSubPathResult::Stop;
            }

            // subslice is special.
            // it needs to be split into individual move paths
            ProjectionKind::Subslice { from, to, from_end: false } => {
                return MoveSubPathResult::Subslice { from, to };
            }
        };

        MoveSubPathResult::One(subpath)
    }
}
