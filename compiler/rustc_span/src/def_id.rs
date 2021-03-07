use crate::crate_disambiguator::CrateDisambiguator;
use crate::HashStableContext;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::AtomicRef;
use rustc_index::vec::Idx;
use rustc_macros::HashStable_Generic;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use std::borrow::Borrow;
use std::fmt;

rustc_index::newtype_index! {
    pub struct CrateId {
        ENCODABLE = custom
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CrateNum {
    /// A special `CrateNum` that we use for the `tcx.rcache` when decoding from
    /// the incr. comp. cache.
    ReservedForIncrCompCache,
    Index(CrateId),
}

/// Item definitions in the currently-compiled crate would have the `CrateNum`
/// `LOCAL_CRATE` in their `DefId`.
pub const LOCAL_CRATE: CrateNum = CrateNum::Index(CrateId::from_u32(0));

impl Idx for CrateNum {
    #[inline]
    fn new(value: usize) -> Self {
        CrateNum::Index(Idx::new(value))
    }

    #[inline]
    fn index(self) -> usize {
        match self {
            CrateNum::Index(idx) => Idx::index(idx),
            _ => panic!("Tried to get crate index of {:?}", self),
        }
    }
}

impl CrateNum {
    pub fn new(x: usize) -> CrateNum {
        CrateNum::from_usize(x)
    }

    pub fn from_usize(x: usize) -> CrateNum {
        CrateNum::Index(CrateId::from_usize(x))
    }

    pub fn from_u32(x: u32) -> CrateNum {
        CrateNum::Index(CrateId::from_u32(x))
    }

    pub fn as_usize(self) -> usize {
        match self {
            CrateNum::Index(id) => id.as_usize(),
            _ => panic!("tried to get index of non-standard crate {:?}", self),
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            CrateNum::Index(id) => id.as_u32(),
            _ => panic!("tried to get index of non-standard crate {:?}", self),
        }
    }

    pub fn as_def_id(&self) -> DefId {
        DefId { krate: *self, index: CRATE_DEF_INDEX }
    }
}

impl fmt::Display for CrateNum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CrateNum::Index(id) => fmt::Display::fmt(&id.private, f),
            CrateNum::ReservedForIncrCompCache => write!(f, "crate for decoding incr comp cache"),
        }
    }
}

/// As a local identifier, a `CrateNum` is only meaningful within its context, e.g. within a tcx.
/// Therefore, make sure to include the context when encode a `CrateNum`.
impl<E: Encoder> Encodable<E> for CrateNum {
    default fn encode(&self, s: &mut E) -> Result<(), E::Error> {
        s.emit_u32(self.as_u32())
    }
}

impl<D: Decoder> Decodable<D> for CrateNum {
    default fn decode(d: &mut D) -> Result<CrateNum, D::Error> {
        Ok(CrateNum::from_u32(d.read_u32()?))
    }
}

impl ::std::fmt::Debug for CrateNum {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        match self {
            CrateNum::Index(id) => write!(fmt, "crate{}", id.private),
            CrateNum::ReservedForIncrCompCache => write!(fmt, "crate for decoding incr comp cache"),
        }
    }
}

/// A `DefPathHash` is a fixed-size representation of a `DefPath` that is
/// stable across crate and compilation session boundaries. It consists of two
/// separate 64-bit hashes. The first uniquely identifies the crate this
/// `DefPathHash` originates from (see [StableCrateId]), and the second
/// uniquely identifies the corresponding `DefPath` within that crate. Together
/// they form a unique identifier within an entire crate graph.
///
/// There is a very small chance of hash collisions, which would mean that two
/// different `DefPath`s map to the same `DefPathHash`. Proceeding compilation
/// with such a hash collision would very probably lead to an ICE, and in the
/// worst case lead to a silent mis-compilation. The compiler therefore actively
/// and exhaustively checks for such hash collisions and aborts compilation if
/// it finds one.
///
/// `DefPathHash` uses 64-bit hashes for both the crate-id part and the
/// crate-internal part, even though it is likely that there are many more
/// `LocalDefId`s in a single crate than there are individual crates in a crate
/// graph. Since we use the same number of bits in both cases, the collision
/// probability for the crate-local part will be quite a bit higher (though
/// still very small).
///
/// This imbalance is not by accident: A hash collision in the
/// crate-local part of a `DefPathHash` will be detected and reported while
/// compiling the crate in question. Such a collision does not depend on
/// outside factors and can be easily fixed by the crate maintainer (e.g. by
/// renaming the item in question or by bumping the crate version in a harmless
/// way).
///
/// A collision between crate-id hashes on the other hand is harder to fix
/// because it depends on the set of crates in the entire crate graph of a
/// compilation session. Again, using the same crate with a different version
/// number would fix the issue with a high probability -- but that might be
/// easier said then done if the crates in questions are dependencies of
/// third-party crates.
///
/// That being said, given a high quality hash function, the collision
/// probabilities in question are very small. For example, for a big crate like
/// `rustc_middle` (with ~50000 `LocalDefId`s as of the time of writing) there
/// is a probability of roughly 1 in 14,750,000,000 of a crate-internal
/// collision occurring. For a big crate graph with 1000 crates in it, there is
/// a probability of 1 in 36,890,000,000,000 of a `StableCrateId` collision.
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[derive(HashStable_Generic, Encodable, Decodable)]
pub struct DefPathHash(pub Fingerprint);

impl DefPathHash {
    /// Returns the [StableCrateId] identifying the crate this [DefPathHash]
    /// originates from.
    #[inline]
    pub fn stable_crate_id(&self) -> StableCrateId {
        StableCrateId(self.0.as_value().0)
    }

    /// Returns the crate-local part of the [DefPathHash].
    #[inline]
    pub fn local_hash(&self) -> u64 {
        self.0.as_value().1
    }

    /// Builds a new [DefPathHash] with the given [StableCrateId] and
    /// `local_hash`, where `local_hash` must be unique within its crate.
    pub fn new(stable_crate_id: StableCrateId, local_hash: u64) -> DefPathHash {
        DefPathHash(Fingerprint::new(stable_crate_id.0, local_hash))
    }
}

impl Borrow<Fingerprint> for DefPathHash {
    #[inline]
    fn borrow(&self) -> &Fingerprint {
        &self.0
    }
}

/// A [StableCrateId] is a 64 bit hash of `(crate-name, crate-disambiguator)`. It
/// is to [CrateNum] what [DefPathHash] is to [DefId]. It is stable across
/// compilation sessions.
///
/// Since the ID is a hash value there is a (very small) chance that two crates
/// end up with the same [StableCrateId]. The compiler will check for such
/// collisions when loading crates and abort compilation in order to avoid
/// further trouble.
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug, Encodable, Decodable)]
pub struct StableCrateId(u64);

impl StableCrateId {
    /// Computes the stable ID for a crate with the given name and
    /// disambiguator.
    pub fn new(crate_name: &str, crate_disambiguator: CrateDisambiguator) -> StableCrateId {
        use std::hash::Hash;

        let mut hasher = StableHasher::new();
        crate_name.hash(&mut hasher);
        crate_disambiguator.hash(&mut hasher);
        StableCrateId(hasher.finish())
    }
}

rustc_index::newtype_index! {
    /// A DefIndex is an index into the hir-map for a crate, identifying a
    /// particular definition. It should really be considered an interned
    /// shorthand for a particular DefPath.
    pub struct DefIndex {
        ENCODABLE = custom // (only encodable in metadata)

        DEBUG_FORMAT = "DefIndex({})",
        /// The crate root is always assigned index 0 by the AST Map code,
        /// thanks to `NodeCollector::new`.
        const CRATE_DEF_INDEX = 0,
    }
}

impl<E: Encoder> Encodable<E> for DefIndex {
    default fn encode(&self, _: &mut E) -> Result<(), E::Error> {
        panic!("cannot encode `DefIndex` with `{}`", std::any::type_name::<E>());
    }
}

impl<D: Decoder> Decodable<D> for DefIndex {
    default fn decode(_: &mut D) -> Result<DefIndex, D::Error> {
        panic!("cannot decode `DefIndex` with `{}`", std::any::type_name::<D>());
    }
}

/// A `DefId` identifies a particular *definition*, by combining a crate
/// index and a def index.
///
/// You can create a `DefId` from a `LocalDefId` using `local_def_id.to_def_id()`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy)]
pub struct DefId {
    pub krate: CrateNum,
    pub index: DefIndex,
}

impl DefId {
    /// Makes a local `DefId` from the given `DefIndex`.
    #[inline]
    pub fn local(index: DefIndex) -> DefId {
        DefId { krate: LOCAL_CRATE, index }
    }

    /// Returns whether the item is defined in the crate currently being compiled.
    #[inline]
    pub fn is_local(self) -> bool {
        self.krate == LOCAL_CRATE
    }

    #[inline]
    pub fn as_local(self) -> Option<LocalDefId> {
        if self.is_local() { Some(LocalDefId { local_def_index: self.index }) } else { None }
    }

    #[inline]
    pub fn expect_local(self) -> LocalDefId {
        self.as_local().unwrap_or_else(|| panic!("DefId::expect_local: `{:?}` isn't local", self))
    }

    pub fn is_top_level_module(self) -> bool {
        self.is_local() && self.index == CRATE_DEF_INDEX
    }
}

impl<E: Encoder> Encodable<E> for DefId {
    default fn encode(&self, s: &mut E) -> Result<(), E::Error> {
        s.emit_struct("DefId", 2, |s| {
            s.emit_struct_field("krate", 0, |s| self.krate.encode(s))?;

            s.emit_struct_field("index", 1, |s| self.index.encode(s))
        })
    }
}

impl<D: Decoder> Decodable<D> for DefId {
    default fn decode(d: &mut D) -> Result<DefId, D::Error> {
        d.read_struct("DefId", 2, |d| {
            Ok(DefId {
                krate: d.read_struct_field("krate", 0, Decodable::decode)?,
                index: d.read_struct_field("index", 1, Decodable::decode)?,
            })
        })
    }
}

pub fn default_def_id_debug(def_id: DefId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("DefId").field("krate", &def_id.krate).field("index", &def_id.index).finish()
}

pub static DEF_ID_DEBUG: AtomicRef<fn(DefId, &mut fmt::Formatter<'_>) -> fmt::Result> =
    AtomicRef::new(&(default_def_id_debug as fn(_, &mut fmt::Formatter<'_>) -> _));

impl fmt::Debug for DefId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (*DEF_ID_DEBUG)(*self, f)
    }
}

rustc_data_structures::define_id_collections!(DefIdMap, DefIdSet, DefId);

/// A LocalDefId is equivalent to a DefId with `krate == LOCAL_CRATE`. Since
/// we encode this information in the type, we can ensure at compile time that
/// no DefIds from upstream crates get thrown into the mix. There are quite a
/// few cases where we know that only DefIds from the local crate are expected
/// and a DefId from a different crate would signify a bug somewhere. This
/// is when LocalDefId comes in handy.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalDefId {
    pub local_def_index: DefIndex,
}

pub const CRATE_DEF_ID: LocalDefId = LocalDefId { local_def_index: CRATE_DEF_INDEX };

impl Idx for LocalDefId {
    #[inline]
    fn new(idx: usize) -> Self {
        LocalDefId { local_def_index: Idx::new(idx) }
    }
    #[inline]
    fn index(self) -> usize {
        self.local_def_index.index()
    }
}

impl LocalDefId {
    #[inline]
    pub fn to_def_id(self) -> DefId {
        DefId { krate: LOCAL_CRATE, index: self.local_def_index }
    }

    #[inline]
    pub fn is_top_level_module(self) -> bool {
        self.local_def_index == CRATE_DEF_INDEX
    }
}

impl fmt::Debug for LocalDefId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_def_id().fmt(f)
    }
}

impl<E: Encoder> Encodable<E> for LocalDefId {
    fn encode(&self, s: &mut E) -> Result<(), E::Error> {
        self.to_def_id().encode(s)
    }
}

impl<D: Decoder> Decodable<D> for LocalDefId {
    fn decode(d: &mut D) -> Result<LocalDefId, D::Error> {
        DefId::decode(d).map(|d| d.expect_local())
    }
}

rustc_data_structures::define_id_collections!(LocalDefIdMap, LocalDefIdSet, LocalDefId);

impl<CTX: HashStableContext> HashStable<CTX> for DefId {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        hcx.hash_def_id(*self, hasher)
    }
}

impl<CTX: HashStableContext> HashStable<CTX> for CrateNum {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        hcx.hash_crate_num(*self, hasher)
    }
}
