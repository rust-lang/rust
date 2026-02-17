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
use rustc_hir::def_id::DefId;
use rustc_hir::definitions::DefPathHash;
use rustc_macros::{Decodable, Encodable};
use rustc_query_system::ich::StableHashingContext;
use rustc_span::Symbol;

use super::{FingerprintStyle, SerializedDepNodeIndex};
use crate::mir::mono::MonoItem;
use crate::ty::TyCtxt;

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

    pub(crate) fn name(self) -> &'static str {
        DEP_KIND_NAMES[self.as_usize()]
    }

    /// We use this for most things when incr. comp. is turned off.
    pub(crate) const NULL: DepKind = dep_kinds::Null;

    /// We use this to create a forever-red node.
    pub(crate) const RED: DepKind = dep_kinds::Red;

    /// We use this to create a side effect node.
    pub(crate) const SIDE_EFFECT: DepKind = dep_kinds::SideEffect;

    /// We use this to create the anon node with zero dependencies.
    pub(crate) const ANON_ZERO_DEPS: DepKind = dep_kinds::AnonZeroDeps;

    /// This is the highest value a `DepKind` can have. It's used during encoding to
    /// pack information into the unused bits.
    pub(crate) const MAX: u16 = DEP_KIND_VARIANTS - 1;
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
    pub fn new_no_params<'tcx>(tcx: TyCtxt<'tcx>, kind: DepKind) -> DepNode {
        debug_assert_eq!(tcx.fingerprint_style(kind), FingerprintStyle::Unit);
        DepNode { kind, hash: Fingerprint::ZERO.into() }
    }

    pub fn construct<'tcx, Key>(tcx: TyCtxt<'tcx>, kind: DepKind, arg: &Key) -> DepNode
    where
        Key: DepNodeKey<'tcx>,
    {
        let hash = arg.to_fingerprint(tcx);
        let dep_node = DepNode { kind, hash: hash.into() };

        #[cfg(debug_assertions)]
        {
            if !tcx.fingerprint_style(kind).reconstructible()
                && (tcx.sess.opts.unstable_opts.incremental_info
                    || tcx.sess.opts.unstable_opts.query_dep_graph)
            {
                tcx.dep_graph.register_dep_node_debug_str(dep_node, || arg.to_debug_str(tcx));
            }
        }

        dep_node
    }

    /// Construct a DepNode from the given DepKind and DefPathHash. This
    /// method will assert that the given DepKind actually requires a
    /// single DefId/DefPathHash parameter.
    pub fn from_def_path_hash<'tcx>(
        tcx: TyCtxt<'tcx>,
        def_path_hash: DefPathHash,
        kind: DepKind,
    ) -> Self {
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

/// Trait for query keys as seen by dependency-node tracking.
pub trait DepNodeKey<'tcx>: fmt::Debug + Sized {
    fn fingerprint_style() -> FingerprintStyle;

    /// This method turns a query key into an opaque `Fingerprint` to be used
    /// in `DepNode`.
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint;

    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String;

    /// This method tries to recover the query key from the given `DepNode`,
    /// something which is needed when forcing `DepNode`s during red-green
    /// evaluation. The query system will only call this method if
    /// `fingerprint_style()` is not `FingerprintStyle::Opaque`.
    /// It is always valid to return `None` here, in which case incremental
    /// compilation will treat the query as having changed instead of forcing it.
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self>;
}

// Blanket impl of `DepNodeKey`, which is specialized by other impls elsewhere.
impl<'tcx, T> DepNodeKey<'tcx> for T
where
    T: for<'a> HashStable<StableHashingContext<'a>> + fmt::Debug,
{
    #[inline(always)]
    default fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::Opaque
    }

    #[inline(always)]
    default fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        tcx.with_stable_hashing_context(|mut hcx| {
            let mut hasher = StableHasher::new();
            self.hash_stable(&mut hcx, &mut hasher);
            hasher.finish()
        })
    }

    #[inline(always)]
    default fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        // Make sure to print dep node params with reduced queries since printing
        // may themselves call queries, which may lead to (possibly untracked!)
        // query cycles.
        tcx.with_reduced_queries(|| format!("{self:?}"))
    }

    #[inline(always)]
    default fn recover(_: TyCtxt<'tcx>, _: &DepNode) -> Option<Self> {
        None
    }
}

/// This struct stores function pointers and other metadata for a particular DepKind.
///
/// Information is retrieved by indexing the `DEP_KINDS` array using the integer value
/// of the `DepKind`. Overall, this allows to implement `DepContext` using this manual
/// jump table instead of large matches.
pub struct DepKindVTable<'tcx> {
    /// Anonymous queries cannot be replayed from one compiler invocation to the next.
    /// When their result is needed, it is recomputed. They are useful for fine-grained
    /// dependency tracking, and caching within one compiler invocation.
    pub is_anon: bool,

    /// Eval-always queries do not track their dependencies, and are always recomputed, even if
    /// their inputs have not changed since the last compiler invocation. The result is still
    /// cached within one compiler invocation.
    pub is_eval_always: bool,

    /// Indicates whether and how the query key can be recovered from its hashed fingerprint.
    ///
    /// The [`DepNodeKey`] trait determines the fingerprint style for each key type.
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
    pub force_from_dep_node: Option<
        fn(tcx: TyCtxt<'tcx>, dep_node: DepNode, prev_index: SerializedDepNodeIndex) -> bool,
    >,

    /// Invoke a query to put the on-disk cached value in memory.
    pub try_load_from_on_disk_cache: Option<fn(TyCtxt<'tcx>, DepNode)>,

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

macro_rules! define_dep_nodes {
    (
        $(
            $(#[$attr:meta])*
            [$($modifiers:tt)*] fn $variant:ident($($K:tt)*) -> $V:ty,
        )*
    ) => {

        #[macro_export]
        macro_rules! make_dep_kind_array {
            ($mod:ident) => {[ $($mod::$variant()),* ]};
        }

        /// This enum serves as an index into arrays built by `make_dep_kind_array`.
        // This enum has more than u8::MAX variants so we need some kind of multi-byte
        // encoding. The derived Encodable/Decodable uses leb128 encoding which is
        // dense when only considering this enum. But DepKind is encoded in a larger
        // struct, and there we can take advantage of the unused bits in the u16.
        #[allow(non_camel_case_types)]
        #[repr(u16)] // Must be kept in sync with the inner type of `DepKind`.
        enum DepKindDefs {
            $( $( #[$attr] )* $variant),*
        }

        #[allow(non_upper_case_globals)]
        pub mod dep_kinds {
            use super::*;

            $(
                // The `as u16` cast must be kept in sync with the inner type of `DepKind`.
                pub const $variant: DepKind = DepKind::new(DepKindDefs::$variant as u16);
            )*
        }

        // This checks that the discriminants of the variants have been assigned consecutively
        // from 0 so that they can be used as a dense index.
        pub(crate) const DEP_KIND_VARIANTS: u16 = {
            let deps = &[$(dep_kinds::$variant,)*];
            let mut i = 0;
            while i < deps.len() {
                if i != deps[i].as_usize() {
                    panic!();
                }
                i += 1;
            }
            deps.len() as u16
        };

        /// List containing the name of each dep kind as a static string,
        /// indexable by `DepKind`.
        pub(crate) const DEP_KIND_NAMES: &[&str] = &[
            $( self::label_strs::$variant, )*
        ];

        pub(super) fn dep_kind_from_label_string(label: &str) -> Result<DepKind, ()> {
            match label {
                $( self::label_strs::$variant => Ok(self::dep_kinds::$variant), )*
                _ => Err(()),
            }
        }

        /// Contains variant => str representations for constructing
        /// DepNode groups for tests.
        #[expect(non_upper_case_globals)]
        pub mod label_strs {
            $( pub const $variant: &str = stringify!($variant); )*
        }
    };
}

// Create various data structures for each query, and also for a few things
// that aren't queries.
rustc_with_all_queries!(define_dep_nodes![
    /// We use this for most things when incr. comp. is turned off.
    [] fn Null() -> (),
    /// We use this to create a forever-red node.
    [] fn Red() -> (),
    [] fn SideEffect() -> (),
    [] fn AnonZeroDeps() -> (),
    [] fn TraitSelect() -> (),
    [] fn CompileCodegenUnit() -> (),
    [] fn CompileMonoItem() -> (),
    [] fn Metadata() -> (),
]);

// WARNING: `construct` is generic and does not know that `CompileCodegenUnit` takes `Symbol`s as keys.
// Be very careful changing this type signature!
pub(crate) fn make_compile_codegen_unit(tcx: TyCtxt<'_>, name: Symbol) -> DepNode {
    DepNode::construct(tcx, dep_kinds::CompileCodegenUnit, &name)
}

// WARNING: `construct` is generic and does not know that `CompileMonoItem` takes `MonoItem`s as keys.
// Be very careful changing this type signature!
pub(crate) fn make_compile_mono_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    mono_item: &MonoItem<'tcx>,
) -> DepNode {
    DepNode::construct(tcx, dep_kinds::CompileMonoItem, mono_item)
}

// WARNING: `construct` is generic and does not know that `Metadata` takes `()`s as keys.
// Be very careful changing this type signature!
pub(crate) fn make_metadata(tcx: TyCtxt<'_>) -> DepNode {
    DepNode::construct(tcx, dep_kinds::Metadata, &())
}

impl DepNode {
    /// Extracts the DefId corresponding to this DepNode. This will work
    /// if two conditions are met:
    ///
    /// 1. The Fingerprint of the DepNode actually is a DefPathHash, and
    /// 2. the item that the DefPath refers to exists in the current tcx.
    ///
    /// Condition (1) is determined by the DepKind variant of the
    /// DepNode. Condition (2) might not be fulfilled if a DepNode
    /// refers to something from the previous compilation session that
    /// has been removed.
    pub fn extract_def_id(&self, tcx: TyCtxt<'_>) -> Option<DefId> {
        if tcx.fingerprint_style(self.kind) == FingerprintStyle::DefPathHash {
            tcx.def_path_hash_to_def_id(DefPathHash(self.hash.into()))
        } else {
            None
        }
    }

    pub fn from_label_string(
        tcx: TyCtxt<'_>,
        label: &str,
        def_path_hash: DefPathHash,
    ) -> Result<DepNode, ()> {
        let kind = dep_kind_from_label_string(label)?;

        match tcx.fingerprint_style(kind) {
            FingerprintStyle::Opaque | FingerprintStyle::HirId => Err(()),
            FingerprintStyle::Unit => Ok(DepNode::new_no_params(tcx, kind)),
            FingerprintStyle::DefPathHash => {
                Ok(DepNode::from_def_path_hash(tcx, def_path_hash, kind))
            }
        }
    }

    pub fn has_label_string(label: &str) -> bool {
        dep_kind_from_label_string(label).is_ok()
    }
}

/// Maps a query label to its DepKind. Panics if a query with the given label does not exist.
pub fn dep_kind_from_label(label: &str) -> DepKind {
    dep_kind_from_label_string(label)
        .unwrap_or_else(|_| panic!("Query label {label} does not exist"))
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
