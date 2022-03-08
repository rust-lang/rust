//! Nodes in the dependency graph.
//!
//! A node in the [dependency graph] is represented by a [`DepNode`].
//! A `DepNode` consists of a [`DepKind`] (which
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
//! `DepNode` definition happens in the `define_dep_nodes!()` macro. This macro
//! defines the `DepKind` enum. Each `DepKind` has its own parameters that are
//! needed at runtime in order to construct a valid `DepNode` fingerprint.
//! However, only `CompileCodegenUnit` and `CompileMonoItem` are constructed
//! explicitly (with `make_compile_codegen_unit` cq `make_compile_mono_item`).
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
//! `DepNode::new()`, ensures that only valid `DepNode` instances can be
//! constructed. For example, the API does not allow for constructing
//! parameterless `DepNode`s with anything other than a zeroed out fingerprint.
//! More generally speaking, it relieves the user of the `DepNode` API of
//! having to know how to compute the expected fingerprint for a given set of
//! node parameters.
//!
//! [dependency graph]: https://rustc-dev-guide.rust-lang.org/query.html

use crate::mir::mono::MonoItem;
use crate::ty::TyCtxt;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, CRATE_DEF_INDEX};
use rustc_hir::definitions::DefPathHash;
use rustc_hir::HirId;
use rustc_query_system::dep_graph::FingerprintStyle;
use rustc_span::symbol::Symbol;
use std::hash::Hash;

pub use rustc_query_system::dep_graph::{DepContext, DepNodeParams};

/// This struct stores metadata about each DepKind.
///
/// Information is retrieved by indexing the `DEP_KINDS` array using the integer value
/// of the `DepKind`. Overall, this allows to implement `DepContext` using this manual
/// jump table instead of large matches.
pub struct DepKindStruct {
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
    pub force_from_dep_node: Option<fn(tcx: TyCtxt<'_>, dep_node: DepNode) -> bool>,

    /// Invoke a query to put the on-disk cached value in memory.
    pub try_load_from_on_disk_cache: Option<fn(TyCtxt<'_>, DepNode)>,
}

impl DepKind {
    #[inline(always)]
    pub fn fingerprint_style(self, tcx: TyCtxt<'_>) -> FingerprintStyle {
        // Only fetch the DepKindStruct once.
        let data = tcx.query_kind(self);
        if data.is_anon {
            return FingerprintStyle::Opaque;
        }
        data.fingerprint_style
    }
}

macro_rules! define_dep_nodes {
    (<$tcx:tt>
    $(
        [$($attrs:tt)*]
        $variant:ident $(( $tuple_arg_ty:ty $(,)? ))*
      ,)*
    ) => (
        #[macro_export]
        macro_rules! make_dep_kind_array {
            ($mod:ident) => {[ $($mod::$variant()),* ]};
        }

        /// This enum serves as an index into arrays built by `make_dep_kind_array`.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
        #[allow(non_camel_case_types)]
        pub enum DepKind {
            $($variant),*
        }

        fn dep_kind_from_label_string(label: &str) -> Result<DepKind, ()> {
            match label {
                $(stringify!($variant) => Ok(DepKind::$variant),)*
                _ => Err(()),
            }
        }

        /// Contains variant => str representations for constructing
        /// DepNode groups for tests.
        #[allow(dead_code, non_upper_case_globals)]
        pub mod label_strs {
           $(
                pub const $variant: &str = stringify!($variant);
            )*
        }
    );
}

rustc_dep_node_append!([define_dep_nodes!][ <'tcx>
    // We use this for most things when incr. comp. is turned off.
    [] Null,

    [anon] TraitSelect,

    // WARNING: if `Symbol` is changed, make sure you update `make_compile_codegen_unit` below.
    [] CompileCodegenUnit(Symbol),

    // WARNING: if `MonoItem` is changed, make sure you update `make_compile_mono_item` below.
    // Only used by rustc_codegen_cranelift
    [] CompileMonoItem(MonoItem),
]);

// WARNING: `construct` is generic and does not know that `CompileCodegenUnit` takes `Symbol`s as keys.
// Be very careful changing this type signature!
crate fn make_compile_codegen_unit(tcx: TyCtxt<'_>, name: Symbol) -> DepNode {
    DepNode::construct(tcx, DepKind::CompileCodegenUnit, &name)
}

// WARNING: `construct` is generic and does not know that `CompileMonoItem` takes `MonoItem`s as keys.
// Be very careful changing this type signature!
crate fn make_compile_mono_item<'tcx>(tcx: TyCtxt<'tcx>, mono_item: &MonoItem<'tcx>) -> DepNode {
    DepNode::construct(tcx, DepKind::CompileMonoItem, mono_item)
}

pub type DepNode = rustc_query_system::dep_graph::DepNode<DepKind>;

// We keep a lot of `DepNode`s in memory during compilation. It's not
// required that their size stay the same, but we don't want to change
// it inadvertently. This assert just ensures we're aware of any change.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
static_assert_size!(DepNode, 18);

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
static_assert_size!(DepNode, 24);

pub trait DepNodeExt: Sized {
    /// Construct a DepNode from the given DepKind and DefPathHash. This
    /// method will assert that the given DepKind actually requires a
    /// single DefId/DefPathHash parameter.
    fn from_def_path_hash(tcx: TyCtxt<'_>, def_path_hash: DefPathHash, kind: DepKind) -> Self;

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
    fn extract_def_id(&self, tcx: TyCtxt<'_>) -> Option<DefId>;

    /// Used in testing
    fn from_label_string(
        tcx: TyCtxt<'_>,
        label: &str,
        def_path_hash: DefPathHash,
    ) -> Result<Self, ()>;

    /// Used in testing
    fn has_label_string(label: &str) -> bool;
}

impl DepNodeExt for DepNode {
    /// Construct a DepNode from the given DepKind and DefPathHash. This
    /// method will assert that the given DepKind actually requires a
    /// single DefId/DefPathHash parameter.
    fn from_def_path_hash(tcx: TyCtxt<'_>, def_path_hash: DefPathHash, kind: DepKind) -> DepNode {
        debug_assert!(kind.fingerprint_style(tcx) == FingerprintStyle::DefPathHash);
        DepNode { kind, hash: def_path_hash.0.into() }
    }

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
    fn extract_def_id<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Option<DefId> {
        if self.kind.fingerprint_style(tcx) == FingerprintStyle::DefPathHash {
            Some(tcx.def_path_hash_to_def_id(DefPathHash(self.hash.into()), &mut || {
                panic!("Failed to extract DefId: {:?} {}", self.kind, self.hash)
            }))
        } else {
            None
        }
    }

    /// Used in testing
    fn from_label_string(
        tcx: TyCtxt<'_>,
        label: &str,
        def_path_hash: DefPathHash,
    ) -> Result<DepNode, ()> {
        let kind = dep_kind_from_label_string(label)?;

        match kind.fingerprint_style(tcx) {
            FingerprintStyle::Opaque => Err(()),
            FingerprintStyle::Unit => Ok(DepNode::new_no_params(tcx, kind)),
            FingerprintStyle::DefPathHash => {
                Ok(DepNode::from_def_path_hash(tcx, def_path_hash, kind))
            }
        }
    }

    /// Used in testing
    fn has_label_string(label: &str) -> bool {
        dep_kind_from_label_string(label).is_ok()
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for () {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::Unit
    }

    #[inline(always)]
    fn to_fingerprint(&self, _: TyCtxt<'tcx>) -> Fingerprint {
        Fingerprint::ZERO
    }

    #[inline(always)]
    fn recover(_: TyCtxt<'tcx>, _: &DepNode) -> Option<Self> {
        Some(())
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for DefId {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        tcx.def_path_hash(*self).0
    }

    #[inline(always)]
    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        tcx.def_path_str(*self)
    }

    #[inline(always)]
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx)
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for LocalDefId {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        self.to_def_id().to_fingerprint(tcx)
    }

    #[inline(always)]
    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        self.to_def_id().to_debug_str(tcx)
    }

    #[inline(always)]
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| id.expect_local())
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for CrateNum {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let def_id = DefId { krate: *self, index: CRATE_DEF_INDEX };
        def_id.to_fingerprint(tcx)
    }

    #[inline(always)]
    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        tcx.crate_name(*self).to_string()
    }

    #[inline(always)]
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| id.krate)
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for (DefId, DefId) {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::Opaque
    }

    // We actually would not need to specialize the implementation of this
    // method but it's faster to combine the hashes than to instantiate a full
    // hashing context and stable-hashing state.
    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let (def_id_0, def_id_1) = *self;

        let def_path_hash_0 = tcx.def_path_hash(def_id_0);
        let def_path_hash_1 = tcx.def_path_hash(def_id_1);

        def_path_hash_0.0.combine(def_path_hash_1.0)
    }

    #[inline(always)]
    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        let (def_id_0, def_id_1) = *self;

        format!("({}, {})", tcx.def_path_debug_str(def_id_0), tcx.def_path_debug_str(def_id_1))
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for HirId {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::Opaque
    }

    // We actually would not need to specialize the implementation of this
    // method but it's faster to combine the hashes than to instantiate a full
    // hashing context and stable-hashing state.
    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let HirId { owner, local_id } = *self;

        let def_path_hash = tcx.def_path_hash(owner.to_def_id());
        let local_id = Fingerprint::from_smaller_hash(local_id.as_u32().into());

        def_path_hash.0.combine(local_id)
    }
}
