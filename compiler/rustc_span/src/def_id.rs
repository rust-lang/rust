use crate::{HashStableContext, Symbol};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};
use rustc_data_structures::AtomicRef;
use rustc_index::vec::Idx;
use rustc_macros::HashStable_Generic;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use std::borrow::Borrow;
use std::fmt;
use std::hash::{Hash, Hasher};

rustc_index::newtype_index! {
    #[custom_encodable]
    #[debug_format = "crate{}"]
    pub struct CrateNum {}
}

/// Item definitions in the currently-compiled crate would have the `CrateNum`
/// `LOCAL_CRATE` in their `DefId`.
pub const LOCAL_CRATE: CrateNum = CrateNum::from_u32(0);

impl CrateNum {
    #[inline]
    pub fn new(x: usize) -> CrateNum {
        CrateNum::from_usize(x)
    }

    #[inline]
    pub fn as_def_id(self) -> DefId {
        DefId { krate: self, index: CRATE_DEF_INDEX }
    }
}

impl fmt::Display for CrateNum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.as_u32(), f)
    }
}

/// As a local identifier, a `CrateNum` is only meaningful within its context, e.g. within a tcx.
/// Therefore, make sure to include the context when encode a `CrateNum`.
impl<E: Encoder> Encodable<E> for CrateNum {
    default fn encode(&self, s: &mut E) {
        s.emit_u32(self.as_u32());
    }
}

impl<D: Decoder> Decodable<D> for CrateNum {
    default fn decode(d: &mut D) -> CrateNum {
        CrateNum::from_u32(d.read_u32())
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
    ///
    /// Used for tests.
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

impl Default for DefPathHash {
    fn default() -> Self {
        DefPathHash(Fingerprint::ZERO)
    }
}

impl Borrow<Fingerprint> for DefPathHash {
    #[inline]
    fn borrow(&self) -> &Fingerprint {
        &self.0
    }
}

/// A [`StableCrateId`] is a 64-bit hash of a crate name, together with all
/// `-Cmetadata` arguments, and some other data. It is to [`CrateNum`] what [`DefPathHash`] is to
/// [`DefId`]. It is stable across compilation sessions.
///
/// Since the ID is a hash value, there is a small chance that two crates
/// end up with the same [`StableCrateId`]. The compiler will check for such
/// collisions when loading crates and abort compilation in order to avoid
/// further trouble.
///
/// For more information on the possibility of hash collisions in rustc,
/// see the discussion in [`DefId`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[derive(HashStable_Generic, Encodable, Decodable)]
pub struct StableCrateId(pub(crate) u64);

impl StableCrateId {
    pub fn to_u64(self) -> u64 {
        self.0
    }

    /// Computes the stable ID for a crate with the given name and
    /// `-Cmetadata` arguments.
    pub fn new(crate_name: Symbol, is_exe: bool, mut metadata: Vec<String>) -> StableCrateId {
        let mut hasher = StableHasher::new();
        // We must hash the string text of the crate name, not the id, as the id is not stable
        // across builds.
        crate_name.as_str().hash(&mut hasher);

        // We don't want the stable crate ID to depend on the order of
        // -C metadata arguments, so sort them:
        metadata.sort();
        // Every distinct -C metadata value is only incorporated once:
        metadata.dedup();

        hasher.write(b"metadata");
        for s in &metadata {
            // Also incorporate the length of a metadata string, so that we generate
            // different values for `-Cmetadata=ab -Cmetadata=c` and
            // `-Cmetadata=a -Cmetadata=bc`
            hasher.write_usize(s.len());
            hasher.write(s.as_bytes());
        }

        // Also incorporate crate type, so that we don't get symbol conflicts when
        // linking against a library of the same name, if this is an executable.
        hasher.write(if is_exe { b"exe" } else { b"lib" });

        // Also incorporate the rustc version. Otherwise, with -Zsymbol-mangling-version=v0
        // and no -Cmetadata, symbols from the same crate compiled with different versions of
        // rustc are named the same.
        //
        // RUSTC_FORCE_RUSTC_VERSION is used to inject rustc version information
        // during testing.
        if let Some(val) = std::env::var_os("RUSTC_FORCE_RUSTC_VERSION") {
            hasher.write(val.to_string_lossy().into_owned().as_bytes())
        } else {
            hasher.write(option_env!("CFG_VERSION").unwrap_or("unknown version").as_bytes());
        }

        StableCrateId(hasher.finish())
    }
}

rustc_index::newtype_index! {
    /// A DefIndex is an index into the hir-map for a crate, identifying a
    /// particular definition. It should really be considered an interned
    /// shorthand for a particular DefPath.
    #[custom_encodable] // (only encodable in metadata)
    #[debug_format = "DefIndex({})"]
    pub struct DefIndex {
        /// The crate root is always assigned index 0 by the AST Map code,
        /// thanks to `NodeCollector::new`.
        const CRATE_DEF_INDEX = 0;
    }
}

impl<E: Encoder> Encodable<E> for DefIndex {
    default fn encode(&self, _: &mut E) {
        panic!("cannot encode `DefIndex` with `{}`", std::any::type_name::<E>());
    }
}

impl<D: Decoder> Decodable<D> for DefIndex {
    default fn decode(_: &mut D) -> DefIndex {
        panic!("cannot decode `DefIndex` with `{}`", std::any::type_name::<D>());
    }
}

/// A `DefId` identifies a particular *definition*, by combining a crate
/// index and a def index.
///
/// You can create a `DefId` from a `LocalDefId` using `local_def_id.to_def_id()`.
#[derive(Clone, PartialEq, Eq, Copy)]
// Don't derive order on 64-bit big-endian, so we can be consistent regardless of field order.
#[cfg_attr(not(all(target_pointer_width = "64", target_endian = "big")), derive(PartialOrd, Ord))]
// On below-64 bit systems we can simply use the derived `Hash` impl
#[cfg_attr(not(target_pointer_width = "64"), derive(Hash))]
#[repr(C)]
#[rustc_pass_by_value]
// We guarantee field order. Note that the order is essential here, see below why.
pub struct DefId {
    // cfg-ing the order of fields so that the `DefIndex` which is high entropy always ends up in
    // the lower bits no matter the endianness. This allows the compiler to turn that `Hash` impl
    // into a direct call to `u64::hash(_)`.
    #[cfg(not(all(target_pointer_width = "64", target_endian = "big")))]
    pub index: DefIndex,
    pub krate: CrateNum,
    #[cfg(all(target_pointer_width = "64", target_endian = "big"))]
    pub index: DefIndex,
}

// On 64-bit systems, we can hash the whole `DefId` as one `u64` instead of two `u32`s. This
// improves performance without impairing `FxHash` quality. So the below code gets compiled to a
// noop on little endian systems because the memory layout of `DefId` is as follows:
//
// ```
//     +-1--------------31-+-32-------------63-+
//     ! index             ! krate             !
//     +-------------------+-------------------+
// ```
//
// The order here has direct impact on `FxHash` quality because we have far more `DefIndex` per
// crate than we have `Crate`s within one compilation. Or in other words, this arrangement puts
// more entropy in the low bits than the high bits. The reason this matters is that `FxHash`, which
// is used throughout rustc, has problems distributing the entropy from the high bits, so reversing
// the order would lead to a large number of collisions and thus far worse performance.
//
// On 64-bit big-endian systems, this compiles to a 64-bit rotation by 32 bits, which is still
// faster than another `FxHash` round.
#[cfg(target_pointer_width = "64")]
impl Hash for DefId {
    fn hash<H: Hasher>(&self, h: &mut H) {
        (((self.krate.as_u32() as u64) << 32) | (self.index.as_u32() as u64)).hash(h)
    }
}

// Implement the same comparison as derived with the other field order.
#[cfg(all(target_pointer_width = "64", target_endian = "big"))]
impl Ord for DefId {
    #[inline]
    fn cmp(&self, other: &DefId) -> std::cmp::Ordering {
        Ord::cmp(&(self.index, self.krate), &(other.index, other.krate))
    }
}
#[cfg(all(target_pointer_width = "64", target_endian = "big"))]
impl PartialOrd for DefId {
    #[inline]
    fn partial_cmp(&self, other: &DefId) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
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
        self.is_local().then(|| LocalDefId { local_def_index: self.index })
    }

    #[inline]
    #[track_caller]
    pub fn expect_local(self) -> LocalDefId {
        // NOTE: `match` below is required to apply `#[track_caller]`,
        // i.e. don't use closures.
        match self.as_local() {
            Some(local_def_id) => local_def_id,
            None => panic!("DefId::expect_local: `{self:?}` isn't local"),
        }
    }

    #[inline]
    pub fn is_crate_root(self) -> bool {
        self.index == CRATE_DEF_INDEX
    }

    #[inline]
    pub fn as_crate_root(self) -> Option<CrateNum> {
        self.is_crate_root().then_some(self.krate)
    }

    #[inline]
    pub fn is_top_level_module(self) -> bool {
        self.is_local() && self.is_crate_root()
    }
}

impl From<LocalDefId> for DefId {
    fn from(local: LocalDefId) -> DefId {
        local.to_def_id()
    }
}

impl<E: Encoder> Encodable<E> for DefId {
    default fn encode(&self, s: &mut E) {
        self.krate.encode(s);
        self.index.encode(s);
    }
}

impl<D: Decoder> Decodable<D> for DefId {
    default fn decode(d: &mut D) -> DefId {
        DefId { krate: Decodable::decode(d), index: Decodable::decode(d) }
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

rustc_data_structures::define_id_collections!(DefIdMap, DefIdSet, DefIdMapEntry, DefId);

/// A `LocalDefId` is equivalent to a `DefId` with `krate == LOCAL_CRATE`. Since
/// we encode this information in the type, we can ensure at compile time that
/// no `DefId`s from upstream crates get thrown into the mix. There are quite a
/// few cases where we know that only `DefId`s from the local crate are expected;
/// a `DefId` from a different crate would signify a bug somewhere. This
/// is when `LocalDefId` comes in handy.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalDefId {
    pub local_def_index: DefIndex,
}

// To ensure correctness of incremental compilation,
// `LocalDefId` must not implement `Ord` or `PartialOrd`.
// See https://github.com/rust-lang/rust/issues/90317.
impl !Ord for LocalDefId {}
impl !PartialOrd for LocalDefId {}

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
        self == CRATE_DEF_ID
    }
}

impl fmt::Debug for LocalDefId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_def_id().fmt(f)
    }
}

impl<E: Encoder> Encodable<E> for LocalDefId {
    fn encode(&self, s: &mut E) {
        self.to_def_id().encode(s);
    }
}

impl<D: Decoder> Decodable<D> for LocalDefId {
    fn decode(d: &mut D) -> LocalDefId {
        DefId::decode(d).expect_local()
    }
}

rustc_data_structures::define_id_collections!(
    LocalDefIdMap,
    LocalDefIdSet,
    LocalDefIdMapEntry,
    LocalDefId
);

impl<CTX: HashStableContext> HashStable<CTX> for DefId {
    #[inline]
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.to_stable_hash_key(hcx).hash_stable(hcx, hasher);
    }
}

impl<CTX: HashStableContext> HashStable<CTX> for LocalDefId {
    #[inline]
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.to_stable_hash_key(hcx).hash_stable(hcx, hasher);
    }
}

impl<CTX: HashStableContext> HashStable<CTX> for CrateNum {
    #[inline]
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.to_stable_hash_key(hcx).hash_stable(hcx, hasher);
    }
}

impl<CTX: HashStableContext> ToStableHashKey<CTX> for DefId {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &CTX) -> DefPathHash {
        hcx.def_path_hash(*self)
    }
}

impl<CTX: HashStableContext> ToStableHashKey<CTX> for LocalDefId {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &CTX) -> DefPathHash {
        hcx.def_path_hash(self.to_def_id())
    }
}

impl<CTX: HashStableContext> ToStableHashKey<CTX> for CrateNum {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &CTX) -> DefPathHash {
        self.as_def_id().to_stable_hash_key(hcx)
    }
}
