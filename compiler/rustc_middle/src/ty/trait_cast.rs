use std::borrow::Borrow;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::stable_hasher::{HashStable, StableCompare, StableHasher, StableOrd};
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_macros::HashStable;

use crate::ich::StableHashingContext;
use crate::mir::interpret::AllocId;
use crate::ty::{self, GenericArg, Instance, Ty, TyCtxt};

/// A `Ty<'tcx>` paired with its pre-computed stable `Fingerprint`,
/// enabling deterministic ordering via `StableCompare`.
///
/// **Identity**: `Eq`/`Hash` delegate to the inner `Ty` (interned
/// pointer identity), so `FingerprintedTy` behaves identically to
/// `Ty` in hash-based collections. Only `StableCompare` uses the
/// fingerprint.
///
/// **Caching**: `Ty`'s `HashStable` impl short-circuits through
/// `WithCachedTypeInfo::stable_hash` when incremental compilation
/// is enabled (the common case), so construction cost is dominated
/// by hashing a `Fingerprint` (two `u64`s), not traversing the
/// full `TyKind`. In non-incremental builds the full hash is
/// computed once at construction and never recomputed.
///
/// **Deref**: `Deref<Target = Ty<'tcx>>` allows transparent use
/// wherever a `&Ty<'tcx>` is expected (pattern matching, method
/// calls, query arguments).
///
/// **Borrow**: `Borrow<Ty<'tcx>>` enables `UnordMap::get(&plain_ty)`
/// lookups without wrapping the key — safe because `Eq`/`Hash` on
/// `FingerprintedTy` agree with `Eq`/`Hash` on `Ty`.
#[derive(Copy, Clone, Debug)]
pub struct FingerprintedTy<'tcx> {
    ty: Ty<'tcx>,
    fingerprint: Fingerprint,
}

impl<'tcx> FingerprintedTy<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Self {
        tcx.with_stable_hashing_context(|mut hcx| {
            let mut hasher = StableHasher::new();
            ty.hash_stable(&mut hcx, &mut hasher);
            FingerprintedTy { ty, fingerprint: hasher.finish() }
        })
    }

    /// Batch-construct from an iterator, sharing a single
    /// `StableHashingContext` across all elements.
    pub fn from_iter(
        tcx: TyCtxt<'tcx>,
        iter: impl IntoIterator<Item = Ty<'tcx>>,
    ) -> UnordSet<Self> {
        tcx.with_stable_hashing_context(|mut hcx| {
            iter.into_iter()
                .map(|ty| {
                    let mut hasher = StableHasher::new();
                    ty.hash_stable(&mut hcx, &mut hasher);
                    FingerprintedTy { ty, fingerprint: hasher.finish() }
                })
                .collect()
        })
    }

    pub fn ty(&self) -> Ty<'tcx> {
        self.ty
    }
}

// --- Identity: delegate to Ty ---

impl PartialEq for FingerprintedTy<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.ty == other.ty
    }
}
impl Eq for FingerprintedTy<'_> {}

impl Hash for FingerprintedTy<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ty.hash(state);
    }
}

// --- Transparent access ---

impl<'tcx> Deref for FingerprintedTy<'tcx> {
    type Target = Ty<'tcx>;
    fn deref(&self) -> &Ty<'tcx> {
        &self.ty
    }
}

impl<'tcx> Borrow<Ty<'tcx>> for FingerprintedTy<'tcx> {
    fn borrow(&self) -> &Ty<'tcx> {
        &self.ty
    }
}

// --- Deterministic ordering ---

impl StableCompare for FingerprintedTy<'_> {
    const CAN_USE_UNSTABLE_SORT: bool = true;

    fn stable_cmp(&self, other: &Self) -> Ordering {
        self.fingerprint.cmp(&other.fingerprint)
    }
}

// --- HashStable: delegate to inner Ty ---

impl<'__ctx> HashStable<StableHashingContext<'__ctx>> for FingerprintedTy<'_> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'__ctx>, hasher: &mut StableHasher) {
        self.ty.hash_stable(hcx, hasher);
    }
}

/// Classifies an augmented intrinsic Instance by kind.
pub enum IntrinsicSiteKind<'tcx> {
    Index {
        super_trait: Ty<'tcx>,
        sub_trait: Ty<'tcx>,
        // OutlivesClass derived from Instance's Outlives tail — not stored here.
    },
    Table {
        super_trait: Ty<'tcx>,
        concrete_type: Ty<'tcx>,
    },
    TableLen {
        super_trait: Ty<'tcx>,
    },
    ErasureSafe {
        super_trait: Ty<'tcx>,
        target_trait: Ty<'tcx>,
    },
}

/// Groups classified intrinsic Instances by kind.
#[derive(Debug, Default, HashStable)]
pub struct TraitCastRequests<'tcx> {
    /// (Super, Sub) → augmented intrinsic Instance.
    /// Multiple Instances for same (Super, Sub) with different Outlives.
    pub index_requests: Vec<IndexRequest<'tcx>>,

    /// (Super, Concrete) → augmented intrinsic Instance.
    pub table_requests: Vec<TableRequest<'tcx>>,

    /// Super → augmented intrinsic Instance.
    pub table_len_requests: Vec<TableLenRequest<'tcx>>,

    /// (Super, Tgt) → augmented intrinsic Instance.
    pub erasure_safe_requests: Vec<ErasureSafeRequest<'tcx>>,
}

#[derive(Debug, HashStable)]
pub struct IndexRequest<'tcx> {
    pub instance: Instance<'tcx>,
    pub super_trait: Ty<'tcx>,
    pub sub_trait: Ty<'tcx>,
}

#[derive(Debug, HashStable)]
pub struct TableRequest<'tcx> {
    pub instance: Instance<'tcx>,
    pub super_trait: Ty<'tcx>,
    pub concrete_type: Ty<'tcx>,
}

#[derive(Debug, HashStable)]
pub struct TableLenRequest<'tcx> {
    pub instance: Instance<'tcx>,
    pub super_trait: Ty<'tcx>,
}

#[derive(Debug, HashStable)]
pub struct ErasureSafeRequest<'tcx> {
    pub instance: Instance<'tcx>,
    pub super_trait: Ty<'tcx>,
    pub target_trait: Ty<'tcx>,
}

impl<'tcx> TraitCastRequests<'tcx> {
    pub fn is_empty(&self) -> bool {
        self.index_requests.is_empty()
            && self.table_requests.is_empty()
            && self.table_len_requests.is_empty()
            && self.erasure_safe_requests.is_empty()
    }

    /// Extract distinct root supertraits from all request kinds.
    pub fn root_traits(&self) -> UnordSet<Ty<'tcx>> {
        let mut roots = UnordSet::default();
        for req in &self.index_requests {
            roots.insert(req.super_trait);
        }
        for req in &self.table_requests {
            roots.insert(req.super_trait);
        }
        for req in &self.table_len_requests {
            roots.insert(req.super_trait);
        }
        roots
    }

    /// Route a pre-classified intrinsic Instance into the appropriate
    /// per-intrinsic list.
    pub fn add(&mut self, site: IntrinsicSiteKind<'tcx>, instance: Instance<'tcx>) {
        match site {
            IntrinsicSiteKind::Index { super_trait, sub_trait } => {
                self.index_requests.push(IndexRequest { instance, super_trait, sub_trait });
            }
            IntrinsicSiteKind::Table { super_trait, concrete_type } => {
                self.table_requests.push(TableRequest { instance, super_trait, concrete_type });
            }
            IntrinsicSiteKind::TableLen { super_trait } => {
                self.table_len_requests.push(TableLenRequest { instance, super_trait });
            }
            IntrinsicSiteKind::ErasureSafe { super_trait, target_trait } => {
                self.erasure_safe_requests.push(ErasureSafeRequest {
                    instance,
                    super_trait,
                    target_trait,
                });
            }
        }
    }
}

/// The trait graph for a single root supertrait.
///
/// Collects all sub-traits that are cast targets and all concrete types
/// that participate via `trait_metadata_table` requests. Built by the
/// `trait_cast_graph` query.
///
/// Keys are [`FingerprintedTy`] wrappers around `Ty<'tcx>`, enabling
/// deterministic materialization via `.into_sorted_stable_ord()`.
/// Lookups via plain `Ty<'tcx>` work through the `Borrow` impl.
#[derive(Debug, HashStable)]
pub struct TraitGraph<'tcx> {
    /// The root supertrait (e.g., `dyn SuperTrait` or `dyn SuperTrait<u8>`).
    pub root: Ty<'tcx>,

    /// All sub-traits that appear as cast targets in `trait_metadata_index`
    /// requests. Each with its set of observed outlives classes.
    /// `UnordMap<FingerprintedTy, _>`: call `.into_sorted_stable_ord()`
    /// to materialize in deterministic order for index assignment.
    /// Lookups via plain `Ty` work through the `Borrow` impl.
    pub sub_traits: UnordMap<FingerprintedTy<'tcx>, SubTraitInfo<'tcx>>,

    /// All concrete types that appear in `trait_metadata_table` requests.
    /// `UnordSet<FingerprintedTy>`: call `.into_sorted_stable_ord()`
    /// to materialize in deterministic order.
    pub concrete_types: UnordSet<FingerprintedTy<'tcx>>,
}

/// Per-sub-trait information within a [`TraitGraph`].
#[derive(Debug, HashStable)]
pub struct SubTraitInfo<'tcx> {
    /// The distinct outlives classes observed from all index requests
    /// targeting this sub-trait.
    /// `UnordSet`: must be materialized via `.into_sorted_stable_ord()`
    /// before use in condensation or index assignment.
    pub outlives_classes: UnordSet<OutlivesClass<'tcx>>,
}

/// An outlives class borrows the interned `&'tcx [GenericArg<'tcx>]`
/// subslice from `Instance::outlives_entries()[1..]` (skipping the
/// sentinel). The slice is already sorted and deduplicated by
/// `augment_callee` / `with_outlives`.
///
/// Because `GenericArg` is interned (pointer-based `Eq`/`Hash`),
/// `PartialEq` on the slice is pointer comparison per element —
/// fast and allocation-free.
#[derive(Clone, Copy, Debug, HashStable)]
pub struct OutlivesClass<'tcx> {
    /// `instance.outlives_entries()[1..]` — the semantic pairs,
    /// skipping the sentinel at position 0.
    pub entries: &'tcx [GenericArg<'tcx>],
}

impl<'tcx> PartialEq for OutlivesClass<'tcx> {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: same subslice pointer and length.
        (std::ptr::eq(self.entries.as_ptr(), other.entries.as_ptr())
            && self.entries.len() == other.entries.len())
        // Fallback: element-wise (still pointer comparisons on interned GenericArg).
        || self.entries == other.entries
    }
}

impl<'tcx> Eq for OutlivesClass<'tcx> {}

impl<'tcx> Hash for OutlivesClass<'tcx> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.entries.hash(state);
    }
}

/// `StableOrd` enables deterministic materialization of `UnordSet<OutlivesClass>`
/// via `into_sorted_stable_ord()`. The ordering is lexicographic on the
/// `(longer, shorter)` pairs — stable across runs because the indices are
/// semantic (binder variable positions), not pointer-derived.
///
/// Safety: the `Ord` is a valid `StableOrd` because `OutlivesArgData`
/// fields `longer` and `shorter` are plain `usize` indices into the dyn
/// type's binder, which are deterministic (TypeVisitor DFS order over
/// interned, canonicalized existential predicates).
impl<'tcx> StableOrd for OutlivesClass<'tcx> {
    const CAN_USE_UNSTABLE_SORT: bool = true;
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl<'tcx> Ord for OutlivesClass<'tcx> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Length first, then lexicographic on (longer, shorter) pairs.
        self.entries.len().cmp(&other.entries.len()).then_with(|| {
            for (a, b) in self.entries.iter().zip(other.entries.iter()) {
                let (al, as_) = match a.kind() {
                    ty::GenericArgKind::Outlives(o) => (o.longer(), o.shorter()),
                    _ => bug!("non-Outlives entry in OutlivesClass"),
                };
                let (bl, bs) = match b.kind() {
                    ty::GenericArgKind::Outlives(o) => (o.longer(), o.shorter()),
                    _ => bug!("non-Outlives entry in OutlivesClass"),
                };
                let ord = al.cmp(&bl).then(as_.cmp(&bs));
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            Ordering::Equal
        })
    }
}

impl<'tcx> PartialOrd for OutlivesClass<'tcx> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'tcx> OutlivesClass<'tcx> {
    /// Build from pre-computed per-call-site outlives entries.
    ///
    /// The caller is responsible for ensuring the slice is already in
    /// the native binder-variable space expected by the consumer. For
    /// MIR-less intrinsic lowerings, that means remapping the transport
    /// slice returned by `augmented_outlives_for_call` before calling
    /// this constructor.
    pub fn from_entries(entries: &'tcx [GenericArg<'tcx>]) -> Self {
        OutlivesClass { entries }
    }

    /// Build directly from an augmented Instance's own Outlives entries.
    /// Callers must ensure the entries are already in the correct index
    /// space (e.g., single-intrinsic bodies with no inlining); in the
    /// general case use `from_entries` with an explicit slice resolved
    /// through `augmented_outlives_for_call`. Panics if the Instance
    /// has no outlives entries.
    pub fn from_instance(instance: Instance<'tcx>) -> Self {
        let all = instance.outlives_entries();
        debug_assert!(!all.is_empty(), "expected augmented Instance with sentinel");
        OutlivesClass { entries: &all[1..] }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Iterate the `(longer, shorter)` index pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> + 'tcx {
        self.entries.iter().map(|a| match a.kind() {
            ty::GenericArgKind::Outlives(o) => (o.longer(), o.shorter()),
            _ => bug!("non-Outlives entry in OutlivesClass"),
        })
    }

    /// Returns the `(longer, shorter)` pair at position `idx`.
    pub fn get(&self, idx: usize) -> (usize, usize) {
        match self.entries[idx].kind() {
            ty::GenericArgKind::Outlives(o) => (o.longer(), o.shorter()),
            _ => bug!("non-Outlives entry in OutlivesClass"),
        }
    }
}

/// Per-root-supertrait table layout: maps (sub_trait, outlives_class) pairs
/// to flat table slot indices. Produced by the `trait_cast_layout` query.
#[derive(Debug, HashStable)]
pub struct TableLayout<'tcx> {
    /// The root supertrait this layout is for.
    pub root: Ty<'tcx>,
    /// Total number of slots in the flat table.
    pub table_length: usize,
    /// (sub_trait, outlives_class) → table slot index.
    pub index_map: UnordMap<(Ty<'tcx>, OutlivesClass<'tcx>), usize>,
    /// Per-slot metadata needed by the population query.
    pub slot_info: Vec<SlotInfo<'tcx>>,
}

impl<'tcx> TableLayout<'tcx> {
    /// Distinct sub-traits that have at least one slot in the table.
    pub fn sub_traits(&self) -> impl Iterator<Item = Ty<'tcx>> + '_ {
        let mut seen = FxHashSet::default();
        self.slot_info
            .iter()
            .filter_map(move |si| seen.insert(si.sub_trait).then_some(si.sub_trait))
    }

    /// All (slot_index, &SlotInfo) pairs for a given sub-trait.
    pub fn slots_for_sub_trait(
        &self,
        sub_trait: Ty<'tcx>,
    ) -> impl Iterator<Item = (usize, &SlotInfo<'tcx>)> {
        self.slot_info.iter().enumerate().filter(move |(_, si)| si.sub_trait == sub_trait)
    }
}

/// Per-slot metadata within a [`TableLayout`].
#[derive(Debug, HashStable)]
pub struct SlotInfo<'tcx> {
    /// The sub-trait this slot corresponds to.
    pub sub_trait: Ty<'tcx>,
    /// The representative outlives class for this slot.
    pub outlives_class: OutlivesClass<'tcx>,
    /// Number of bound variables in the sub-trait's dyn binder.
    pub num_bvs: usize,
}

/// Lookup table mapping each table-dependent intrinsic to its resolved
/// constant value. Built by `build_intrinsic_resolutions` from the query
/// results of `trait_cast_layout` and `trait_cast_table_alloc`.
///
/// Consumed by `cascade_canonicalize` to patch intrinsic calls
/// in MIR bodies. `trait_cast_is_lifetime_erasure_safe` is **not** stored
/// here — it is resolved lazily per call site via the
/// `augmented_outlives_for_call` and `is_lifetime_erasure_safe` queries.
#[derive(Debug)]
pub struct IntrinsicResolutions<'tcx> {
    /// The global crate-id `AllocId`, shared across all index/table resolutions.
    pub global_crate_id: AllocId,
    /// `trait_metadata_index` resolved table indices, keyed by
    /// `(sub_trait_dyn, outlives_class)`. Point-lookup only.
    pub indices: UnordMap<(Ty<'tcx>, OutlivesClass<'tcx>), usize>,
    /// `trait_metadata_table` resolved static allocations, keyed by
    /// `(super_trait_dyn, concrete_ty)`. Point-lookup only.
    pub tables: UnordMap<(Ty<'tcx>, Ty<'tcx>), AllocId>,
    /// `trait_metadata_table_len` resolved values, keyed by
    /// `super_trait_dyn`. Point-lookup only.
    pub table_lens: UnordMap<Ty<'tcx>, usize>,
    /// Deduplicated list of all table static `AllocId`s for vtable
    /// method collection. These are the same values stored in `tables`,
    /// collected into a `Vec` for deterministic iteration.
    pub table_alloc_ids: Vec<AllocId>,
}
