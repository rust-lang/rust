use crate::ich::StableHashingContext;
use crate::mir::{self, Local, Location};
use crate::ty::RegionVid;
use polonius_engine::Atom;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};
use rustc_index::bit_set::BitSet;
use rustc_index::vec::Idx;
use std::fmt;
use std::ops::Index;

rustc_index::newtype_index! {
  pub struct BorrowIndex {
    derive [HashStable]
    DEBUG_FORMAT = "bw{}"
  }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for BorrowIndex {
    type KeyType = Self;

    #[inline]
    fn to_stable_hash_key(&self, _: &StableHashingContext<'a>) -> Self {
        *self
    }
}

impl Atom for BorrowIndex {
    fn index(self) -> usize {
        Idx::index(self)
    }
}

#[derive(Debug, Clone, TyEncodable, TyDecodable, HashStable)]
pub struct BorrowSet<'tcx> {
    /// The fundamental map relating bitvector indexes to the borrows
    /// in the MIR. Each borrow is also uniquely identified in the MIR
    /// by the `Location` of the assignment statement in which it
    /// appears on the right hand side. Thus the location is the map
    /// key, and its position in the map corresponds to `BorrowIndex`.
    pub location_map: FxIndexMap<Location, BorrowData<'tcx>>,

    /// Locations which activate borrows.
    /// NOTE: a given location may activate more than one borrow in the future
    /// when more general two-phase borrow support is introduced, but for now we
    /// only need to store one borrow index.
    pub activation_map: FxHashMap<Location, Vec<BorrowIndex>>,

    /// Map from local to all the borrows on that local.
    pub local_map: FxHashMap<Local, FxHashSet<BorrowIndex>>,

    pub locals_state_at_exit: LocalsStateAtExit,
}

impl<'tcx> BorrowSet<'tcx> {
    pub fn activations_at_location(&self, location: Location) -> &[BorrowIndex] {
        self.activation_map.get(&location).map_or(&[], |activations| &activations[..])
    }

    pub fn len(&self) -> usize {
        self.location_map.len()
    }

    pub fn indices(&self) -> impl Iterator<Item = BorrowIndex> {
        BorrowIndex::from_usize(0)..BorrowIndex::from_usize(self.len())
    }

    pub fn iter_enumerated(&self) -> impl Iterator<Item = (BorrowIndex, &BorrowData<'tcx>)> {
        self.indices().zip(self.location_map.values())
    }

    pub fn get_index_of(&self, location: &Location) -> Option<BorrowIndex> {
        self.location_map.get_index_of(location).map(BorrowIndex::from)
    }

    pub fn contains(&self, location: &Location) -> bool {
        self.location_map.contains_key(location)
    }
}

impl<'tcx> Index<BorrowIndex> for BorrowSet<'tcx> {
    type Output = BorrowData<'tcx>;

    fn index(&self, index: BorrowIndex) -> &BorrowData<'tcx> {
        &self.location_map[index.as_usize()]
    }
}

#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub enum LocalsStateAtExit {
    AllAreInvalidated,
    SomeAreInvalidated { has_storage_dead_or_moved: BitSet<Local> },
}

/// Location where a two-phase borrow is activated, if a borrow
/// is in fact a two-phase borrow.
#[derive(Copy, Clone, PartialEq, Eq, Debug, TyEncodable, TyDecodable, HashStable)]
pub enum TwoPhaseActivation {
    NotTwoPhase,
    NotActivated,
    ActivatedAt(Location),
}

#[derive(Debug, Clone, TyEncodable, TyDecodable)]
pub struct BorrowData<'tcx> {
    /// Location where the borrow reservation starts.
    /// In many cases, this will be equal to the activation location but not always.
    pub reserve_location: Location,
    /// Location where the borrow is activated.
    pub activation_location: TwoPhaseActivation,
    /// What kind of borrow this is
    pub kind: mir::BorrowKind,
    /// The region for which this borrow is live
    pub region: RegionVid,
    /// Place from which we are borrowing
    pub borrowed_place: mir::Place<'tcx>,
    /// Place to which the borrow was stored
    pub assigned_place: mir::Place<'tcx>,
}

/*
avoid hashing borrowed_place and assigned_place b/c this error

error: internal compiler error: compiler/rustc_middle/src/ich/impls_ty.rs:94:17: StableHasher: unexpected region '_#10r

thread 'rustc' panicked at 'Box<Any>', /Users/will/Code/rust/library/std/src/panic.rs:59:5
stack backtrace:
   0: std::panicking::begin_panic
   1: std::panic::panic_any
   2: rustc_errors::HandlerInner::bug
   3: rustc_errors::Handler::bug
   4: rustc_middle::util::bug::opt_span_bug_fmt::{{closure}}
   5: rustc_middle::ty::context::tls::with_opt::{{closure}}
   6: rustc_middle::ty::context::tls::with_opt
   7: rustc_middle::util::bug::opt_span_bug_fmt
   8: rustc_middle::util::bug::bug_fmt
   9: rustc_middle::ich::impls_ty::<impl rustc_data_structures::stable_hasher::HashStable<rustc_middle::ich::hcx::StableHashingContext> for rustc_middle::ty::sty::RegionKind>::hash_stable
  10: rustc_middle::ty::sty::_DERIVE_rustc_data_structures_stable_hasher_HashStable_rustc_middle_ich_StableHashingContext_ctx_FOR_TyKind::<impl rustc_data_structures::stable_hasher::HashStable<rustc_middle::ich::hcx::StableHashingContext> for rustc_middle::ty::sty::TyKind>::hash_stable
  11: std::thread::local::LocalKey<T>::with
  12: rustc_middle::ich::impls_ty::<impl rustc_data_struc...
 */
impl<'tcx, 'a> HashStable<StableHashingContext<'a>> for BorrowData<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        self.reserve_location.hash_stable(hcx, hasher);
        self.activation_location.hash_stable(hcx, hasher);
        self.kind.hash_stable(hcx, hasher);
        self.region.hash_stable(hcx, hasher);
        //self.borrowed_place.hash_stable(hcx, hasher);
        //self.assigned_place.hash_stable(hcx, hasher);
    }
}

impl<'tcx> fmt::Display for BorrowData<'tcx> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            mir::BorrowKind::Shared => "",
            mir::BorrowKind::Shallow => "shallow ",
            mir::BorrowKind::Unique => "uniq ",
            mir::BorrowKind::Mut { .. } => "mut ",
        };
        write!(w, "&{:?} {}{:?}", self.region, kind, self.borrowed_place)
    }
}
