//! This module defines the [`DepNode`] type which the compiler uses to represent
//! nodes in the [dependency graph]. A `DepNode` consists of a [`DepKind`] (which
//! specifies the kind of thing it represents, like a piece of HIR, MIR, etc.)
//! and a [`Fingerprint`], a 128-bit hash value, the exact meaning of which
//! depends on the node's `DepKind`. Together, the kind and the fingerprint
//! fully identify a dependency node, even across multiple compilation sessions.
//! In other words, the value of the fingerprint does not depend on anything
//! that is specific to a given compilation session, like an unpredictable
//! interning key (e.g., `NodeId`, `DefId`, `Symbol`) or the numeric value of a
//! pointer. The concept behind this could be compared to how git commit hashes
//! uniquely identify a given commit. The fingerprinting approach has
//! a few advantages:
//!
//! * A `DepNode` can simply be serialized to disk and loaded in another session
//!   without the need to do any "rebasing" (like we have to do for Spans and
//!   NodeIds) or "retracing" (like we had to do for `DefId` in earlier
//!   implementations of the dependency graph).
//! * A `Fingerprint` is just a bunch of bits, which allows `DepNode` to
//!   implement `Copy`, `Sync`, `Send`, `Freeze`, etc.
//! * Since we just have a bit pattern, `DepNode` can be mapped from disk into
//!   memory without any post-processing (e.g., "abomination-style" pointer
//!   reconstruction).
//! * Because a `DepNode` is self-contained, we can instantiate `DepNodes` that
//!   refer to things that do not exist anymore. In previous implementations
//!   `DepNode` contained a `DefId`. A `DepNode` referring to something that
//!   had been removed between the previous and the current compilation session
//!   could not be instantiated because the current compilation session
//!   contained no `DefId` for thing that had been removed.
//!
//! `DepNode` definition happens in `rustc_middle` with the
//! `define_dep_nodes!()` macro. This macro defines the `DepKind` enum. Each
//! `DepKind` has its own parameters that are needed at runtime in order to
//! construct a valid `DepNode` fingerprint. However, only `CompileCodegenUnit`
//! and `CompileMonoItem` are constructed explicitly (with
//! `make_compile_codegen_unit` and `make_compile_mono_item`).
//!
//! Because the macro sees what parameters a given `DepKind` requires, it can
//! "infer" some properties for each kind of `DepNode`:
//!
//! * Whether a `DepNode` of a given kind has any parameters at all. Some
//!   `DepNode`s could represent global concepts with only one value.
//! * Whether it is possible, in principle, to reconstruct a query key from a
//!   given `DepNode`. Many `DepKind`s only require a single `DefId` parameter,
//!   in which case it is possible to map the node's fingerprint back to the
//!   `DefId` it was computed from. In other cases, too much information gets
//!   lost during fingerprint computation.
//!
//! `make_compile_codegen_unit` and `make_compile_mono_items`, together with
//! `DepNode::new()`, ensure that only valid `DepNode` instances can be
//! constructed. For example, the API does not allow for constructing
//! parameterless `DepNode`s with anything other than a zeroed out fingerprint.
//! More generally speaking, it relieves the user of the `DepNode` API of
//! having to know how to compute the expected fingerprint for a given set of
//! node parameters.
//!
//! [dependency graph]: https://rustc-dev-guide.rust-lang.org/query.html

use std::fmt;
use std::hash::Hash;

use rustc_data_structures::AtomicRef;
use rustc_data_structures::fingerprint::{Fingerprint, PackedFingerprint};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, StableOrd, ToStableHashKey};
use rustc_hir::definitions::DefPathHash;
use rustc_macros::{Decodable, Encodable};

use super::{DepContext, FingerprintStyle, SerializedDepNodeIndex};
use crate::ich::StableHashingContext;

/// This serves as an index into arrays built by `make_dep_kind_array`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepKind {
    variant: u16,
}

impl DepKind {
    #[inline]
    pub const fn new(variant: u16) -> Self {
        Self { variant }
    }

    #[inline]
    pub const fn as_inner(&self) -> u16 {
        self.variant
    }

    #[inline]
    pub const fn as_usize(&self) -> usize {
        self.variant as usize
    }
}

pub fn default_dep_kind_debug(kind: DepKind, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("DepKind").field("variant", &kind.variant).finish()
}

pub static DEP_KIND_DEBUG: AtomicRef<fn(DepKind, &mut fmt::Formatter<'_>) -> fmt::Result> =
    AtomicRef::new(&(default_dep_kind_debug as fn(_, &mut fmt::Formatter<'_>) -> _));

impl fmt::Debug for DepKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (*DEP_KIND_DEBUG)(*self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepNode {
    pub kind: DepKind,
    pub hash: PackedFingerprint,
}

impl DepNode {
    /// Creates a new, parameterless DepNode. This method will assert
    /// that the DepNode corresponding to the given DepKind actually
    /// does not require any parameters.
    pub fn new_no_params<Tcx>(tcx: Tcx, kind: DepKind) -> DepNode
    where
        Tcx: super::DepContext,
    {
        debug_assert_eq!(tcx.fingerprint_style(kind), FingerprintStyle::Unit);
        DepNode { kind, hash: Fingerprint::ZERO.into() }
    }

    pub fn construct<Tcx, Key>(tcx: Tcx, kind: DepKind, arg: &Key) -> DepNode
    where
        Tcx: super::DepContext,
        Key: DepNodeParams<Tcx>,
    {
        let hash = arg.to_fingerprint(tcx);
        let dep_node = DepNode { kind, hash: hash.into() };

        #[cfg(debug_assertions)]
        {
            if !tcx.fingerprint_style(kind).reconstructible()
                && (tcx.sess().opts.unstable_opts.incremental_info
                    || tcx.sess().opts.unstable_opts.query_dep_graph)
            {
                tcx.dep_graph().register_dep_node_debug_str(dep_node, || arg.to_debug_str(tcx));
            }
        }

        dep_node
    }

    /// Construct a DepNode from the given DepKind and DefPathHash. This
    /// method will assert that the given DepKind actually requires a
    /// single DefId/DefPathHash parameter.
    pub fn from_def_path_hash<Tcx>(tcx: Tcx, def_path_hash: DefPathHash, kind: DepKind) -> Self
    where
        Tcx: super::DepContext,
    {
        debug_assert!(tcx.fingerprint_style(kind) == FingerprintStyle::DefPathHash);
        DepNode { kind, hash: def_path_hash.0.into() }
    }
}

pub fn default_dep_node_debug(node: DepNode, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("DepNode").field("kind", &node.kind).field("hash", &node.hash).finish()
}

pub static DEP_NODE_DEBUG: AtomicRef<fn(DepNode, &mut fmt::Formatter<'_>) -> fmt::Result> =
    AtomicRef::new(&(default_dep_node_debug as fn(_, &mut fmt::Formatter<'_>) -> _));

impl fmt::Debug for DepNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (*DEP_NODE_DEBUG)(*self, f)
    }
}

pub trait DepNodeParams<Tcx: DepContext>: fmt::Debug + Sized {
    fn fingerprint_style() -> FingerprintStyle;

    /// This method turns the parameters of a DepNodeConstructor into an opaque
    /// Fingerprint to be used in DepNode.
    /// Not all DepNodeParams support being turned into a Fingerprint (they
    /// don't need to if the corresponding DepNode is anonymous).
    fn to_fingerprint(&self, _: Tcx) -> Fingerprint {
        panic!("Not implemented. Accidentally called on anonymous node?")
    }

    fn to_debug_str(&self, tcx: Tcx) -> String;

    /// This method tries to recover the query key from the given `DepNode`,
    /// something which is needed when forcing `DepNode`s during red-green
    /// evaluation. The query system will only call this method if
    /// `fingerprint_style()` is not `FingerprintStyle::Opaque`.
    /// It is always valid to return `None` here, in which case incremental
    /// compilation will treat the query as having changed instead of forcing it.
    fn recover(tcx: Tcx, dep_node: &DepNode) -> Option<Self>;
}

impl<Tcx: DepContext, T> DepNodeParams<Tcx> for T
where
    T: for<'a> HashStable<StableHashingContext<'a>> + fmt::Debug,
{
    #[inline(always)]
    default fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::Opaque
    }

    #[inline(always)]
    default fn to_fingerprint(&self, tcx: Tcx) -> Fingerprint {
        tcx.with_stable_hashing_context(|mut hcx| {
            let mut hasher = StableHasher::new();
            self.hash_stable(&mut hcx, &mut hasher);
            hasher.finish()
        })
    }

    #[inline(always)]
    default fn to_debug_str(&self, tcx: Tcx) -> String {
        // Make sure to print dep node params with reduced queries since printing
        // may themselves call queries, which may lead to (possibly untracked!)
        // query cycles.
        tcx.with_reduced_queries(|| format!("{self:?}"))
    }

    #[inline(always)]
    default fn recover(_: Tcx, _: &DepNode) -> Option<Self> {
        None
    }
}

/// This struct stores metadata about each DepKind.
///
/// Information is retrieved by indexing the `DEP_KINDS` array using the integer value
/// of the `DepKind`. Overall, this allows to implement `DepContext` using this manual
/// jump table instead of large matches.
pub struct DepKindStruct<Tcx: DepContext> {
    /// Anonymous queries cannot be replayed from one compiler invocation to the next.
    /// When their result is needed, it is recomputed. They are useful for fine-grained
    /// dependency tracking, and caching within one compiler invocation.
    pub is_anon: bool,

    /// Eval-always queries do not track their dependencies, and are always recomputed, even if
    /// their inputs have not changed since the last compiler invocation. The result is still
    /// cached within one compiler invocation.
    pub is_eval_always: bool,

    /// Whether the query key can be recovered from the hashed fingerprint.
    /// See [DepNodeParams] trait for the behaviour of each key type.
    pub fingerprint_style: FingerprintStyle,

    /// The red/green evaluation system will try to mark a specific DepNode in the
    /// dependency graph as green by recursively trying to mark the dependencies of
    /// that `DepNode` as green. While doing so, it will sometimes encounter a `DepNode`
    /// where we don't know if it is red or green and we therefore actually have
    /// to recompute its value in order to find out. Since the only piece of
    /// information that we have at that point is the `DepNode` we are trying to
    /// re-evaluate, we need some way to re-run a query from just that. This is what
    /// `force_from_dep_node()` implements.
    ///
    /// In the general case, a `DepNode` consists of a `DepKind` and an opaque
    /// GUID/fingerprint that will uniquely identify the node. This GUID/fingerprint
    /// is usually constructed by computing a stable hash of the query-key that the
    /// `DepNode` corresponds to. Consequently, it is not in general possible to go
    /// back from hash to query-key (since hash functions are not reversible). For
    /// this reason `force_from_dep_node()` is expected to fail from time to time
    /// because we just cannot find out, from the `DepNode` alone, what the
    /// corresponding query-key is and therefore cannot re-run the query.
    ///
    /// The system deals with this case letting `try_mark_green` fail which forces
    /// the root query to be re-evaluated.
    ///
    /// Now, if `force_from_dep_node()` would always fail, it would be pretty useless.
    /// Fortunately, we can use some contextual information that will allow us to
    /// reconstruct query-keys for certain kinds of `DepNode`s. In particular, we
    /// enforce by construction that the GUID/fingerprint of certain `DepNode`s is a
    /// valid `DefPathHash`. Since we also always build a huge table that maps every
    /// `DefPathHash` in the current codebase to the corresponding `DefId`, we have
    /// everything we need to re-run the query.
    ///
    /// Take the `mir_promoted` query as an example. Like many other queries, it
    /// just has a single parameter: the `DefId` of the item it will compute the
    /// validated MIR for. Now, when we call `force_from_dep_node()` on a `DepNode`
    /// with kind `MirValidated`, we know that the GUID/fingerprint of the `DepNode`
    /// is actually a `DefPathHash`, and can therefore just look up the corresponding
    /// `DefId` in `tcx.def_path_hash_to_def_id`.
    pub force_from_dep_node:
        Option<fn(tcx: Tcx, dep_node: DepNode, prev_index: SerializedDepNodeIndex) -> bool>,

    /// Invoke a query to put the on-disk cached value in memory.
    pub try_load_from_on_disk_cache: Option<fn(Tcx, DepNode)>,

    /// The name of this dep kind.
    pub name: &'static &'static str,
}

/// A "work product" corresponds to a `.o` (or other) file that we
/// save in between runs. These IDs do not have a `DefId` but rather
/// some independent path or string that persists between runs without
/// the need to be mapped or unmapped. (This ensures we can serialize
/// them even in the absence of a tcx.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable)]
pub struct WorkProductId {
    hash: Fingerprint,
}

impl WorkProductId {
    pub fn from_cgu_name(cgu_name: &str) -> WorkProductId {
        let mut hasher = StableHasher::new();
        cgu_name.hash(&mut hasher);
        WorkProductId { hash: hasher.finish() }
    }
}

impl<HCX> HashStable<HCX> for WorkProductId {
    #[inline]
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        self.hash.hash_stable(hcx, hasher)
    }
}
impl<HCX> ToStableHashKey<HCX> for WorkProductId {
    type KeyType = Fingerprint;
    #[inline]
    fn to_stable_hash_key(&self, _: &HCX) -> Self::KeyType {
        self.hash
    }
}
impl StableOrd for WorkProductId {
    // Fingerprint can use unstable (just a tuple of `u64`s), so WorkProductId can as well
    const CAN_USE_UNSTABLE_SORT: bool = true;

    // `WorkProductId` sort order is not affected by (de)serialization.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(DepKind, 2);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    static_assert_size!(DepNode, 18);
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    static_assert_size!(DepNode, 24);
    // tidy-alphabetical-end
}
