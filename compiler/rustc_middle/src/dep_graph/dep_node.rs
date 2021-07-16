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
use rustc_span::symbol::Symbol;

use rustc_query_system::dep_graph::{dep_kind_from_label_string, DepKind, DepNode, DepNodeParams};

#[allow(non_upper_case_globals)]
pub mod can_reconstruct_query_key {
    use super::*;
    use crate::ty::query::query_keys;

    // We use this for most things when incr. comp. is turned off.
    pub const Null: fn() -> bool = || true;
    pub const TraitSelect: fn() -> bool = || true;
    pub const CompileCodegenUnit: fn() -> bool = || false;
    pub const CompileMonoItem: fn() -> bool = || false;

    macro_rules! define_query_dep_kinds {
        ($(
            [$($attrs:tt)*]
            $variant:ident $(( $tuple_arg_ty:ty $(,)? ))*
        ,)*) => (
            $(pub const $variant: fn() -> bool =
                <query_keys::$variant<'_> as DepNodeParams<TyCtxt<'_>>>
                    ::can_reconstruct_query_key;)*
        );
    }

    rustc_dep_node_append!([define_query_dep_kinds!][]);
}

// FIXME: Make this a simple boolean once DepNodeParams::can_reconstruct_query_key
// can be made a specialized associated const.
static CAN_RECONSTRUCT: &[fn() -> bool] = &make_dep_kind_array!(can_reconstruct_query_key);

// WARNING: `construct` is generic and does not know that `CompileCodegenUnit` takes `Symbol`s as keys.
// Be very careful changing this type signature!
crate fn make_compile_codegen_unit(tcx: TyCtxt<'_>, name: Symbol) -> DepNode {
    DepNode::construct(tcx, DepKind::CompileCodegenUnit, &name)
}

// WARNING: `construct` is generic and does not know that `CompileMonoItem` takes `MonoItem`s as keys.
// Be very careful changing this type signature!
crate fn make_compile_mono_item(tcx: TyCtxt<'tcx>, mono_item: &MonoItem<'tcx>) -> DepNode {
    DepNode::construct(tcx, DepKind::CompileMonoItem, mono_item)
}

/// Whether the query key can be recovered from the hashed fingerprint.
/// See [DepNodeParams] trait for the behaviour of each key type.
#[inline(always)]
fn can_reconstruct_query_key(kind: DepKind) -> bool {
    if kind.is_anon {
        return false;
    }

    CAN_RECONSTRUCT[kind as usize]()
}

pub trait DepNodeExt: Sized {
    /// Construct a DepNode from the given DepKind and DefPathHash. This
    /// method will assert that the given DepKind actually requires a
    /// single DefId/DefPathHash parameter.
    fn from_def_path_hash(def_path_hash: DefPathHash, kind: DepKind) -> Self;

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
    fn from_label_string(label: &str, def_path_hash: DefPathHash) -> Result<Self, ()>;

    /// Whether the query key can be recovered from the hashed fingerprint.
    /// See [DepNodeParams] trait for the behaviour of each key type.
    fn can_reconstruct_query_key(&self) -> bool;
}

impl DepNodeExt for DepNode {
    /// Construct a DepNode from the given DepKind and DefPathHash. This
    /// method will assert that the given DepKind actually requires a
    /// single DefId/DefPathHash parameter.
    fn from_def_path_hash(def_path_hash: DefPathHash, kind: DepKind) -> DepNode {
        debug_assert!(can_reconstruct_query_key(kind) && kind.has_params);
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
    fn extract_def_id(&self, tcx: TyCtxt<'tcx>) -> Option<DefId> {
        if self.can_reconstruct_query_key() {
            tcx.on_disk_cache.as_ref()?.def_path_hash_to_def_id(tcx, DefPathHash(self.hash.into()))
        } else {
            None
        }
    }

    /// Used in testing
    fn from_label_string(label: &str, def_path_hash: DefPathHash) -> Result<DepNode, ()> {
        let kind = dep_kind_from_label_string(label)?;

        if !can_reconstruct_query_key(kind) {
            return Err(());
        }

        if kind.has_params {
            Ok(DepNode::from_def_path_hash(def_path_hash, kind))
        } else {
            Ok(DepNode::new_no_params(kind))
        }
    }

    /// Whether the query key can be recovered from the hashed fingerprint.
    /// See [DepNodeParams] trait for the behaviour of each key type.
    #[inline(always)]
    fn can_reconstruct_query_key(&self) -> bool {
        can_reconstruct_query_key(self.kind)
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for () {
    #[inline(always)]
    fn can_reconstruct_query_key() -> bool {
        true
    }

    fn to_fingerprint(&self, _: TyCtxt<'tcx>) -> Fingerprint {
        Fingerprint::ZERO
    }

    fn recover(_: TyCtxt<'tcx>, _: &DepNode) -> Option<Self> {
        Some(())
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for DefId {
    #[inline(always)]
    fn can_reconstruct_query_key() -> bool {
        true
    }

    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let hash = tcx.def_path_hash(*self);
        // If this is a foreign `DefId`, store its current value
        // in the incremental cache. When we decode the cache,
        // we will use the old DefIndex as an initial guess for
        // a lookup into the crate metadata.
        if !self.is_local() {
            if let Some(cache) = &tcx.on_disk_cache {
                cache.store_foreign_def_id_hash(*self, hash);
            }
        }
        hash.0
    }

    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        tcx.def_path_str(*self)
    }

    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx)
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for LocalDefId {
    #[inline(always)]
    fn can_reconstruct_query_key() -> bool {
        true
    }

    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        self.to_def_id().to_fingerprint(tcx)
    }

    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        self.to_def_id().to_debug_str(tcx)
    }

    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| id.expect_local())
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for CrateNum {
    #[inline(always)]
    fn can_reconstruct_query_key() -> bool {
        true
    }

    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let def_id = DefId { krate: *self, index: CRATE_DEF_INDEX };
        def_id.to_fingerprint(tcx)
    }

    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        tcx.crate_name(*self).to_string()
    }

    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| id.krate)
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for (DefId, DefId) {
    #[inline(always)]
    fn can_reconstruct_query_key() -> bool {
        false
    }

    // We actually would not need to specialize the implementation of this
    // method but it's faster to combine the hashes than to instantiate a full
    // hashing context and stable-hashing state.
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let (def_id_0, def_id_1) = *self;

        let def_path_hash_0 = tcx.def_path_hash(def_id_0);
        let def_path_hash_1 = tcx.def_path_hash(def_id_1);

        def_path_hash_0.0.combine(def_path_hash_1.0)
    }

    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        let (def_id_0, def_id_1) = *self;

        format!("({}, {})", tcx.def_path_debug_str(def_id_0), tcx.def_path_debug_str(def_id_1))
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for HirId {
    #[inline(always)]
    fn can_reconstruct_query_key() -> bool {
        false
    }

    // We actually would not need to specialize the implementation of this
    // method but it's faster to combine the hashes than to instantiate a full
    // hashing context and stable-hashing state.
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let HirId { owner, local_id } = *self;

        let def_path_hash = tcx.def_path_hash(owner.to_def_id());
        let local_id = Fingerprint::from_smaller_hash(local_id.as_u32().into());

        def_path_hash.0.combine(local_id)
    }
}
