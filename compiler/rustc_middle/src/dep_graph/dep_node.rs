//! This module defines the `DepNode` type which the compiler uses to represent
//! nodes in the dependency graph.
//!
//! A `DepNode` consists of a `DepKind` (which
//! specifies the kind of thing it represents, like a piece of HIR, MIR, etc)
//! and a `Fingerprint`, a 128-bit hash value the exact meaning of which
//! depends on the node's `DepKind`. Together, the kind and the fingerprint
//! fully identify a dependency node, even across multiple compilation sessions.
//! In other words, the value of the fingerprint does not depend on anything
//! that is specific to a given compilation session, like an unpredictable
//! interning key (e.g., NodeId, DefId, Symbol) or the numeric value of a
//! pointer. The concept behind this could be compared to how git commit hashes
//! uniquely identify a given commit and has a few advantages:
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
//! defines the `DepKind` enum and a corresponding `DepConstructor` enum. The
//! `DepConstructor` enum links a `DepKind` to the parameters that are needed at
//! runtime in order to construct a valid `DepNode` fingerprint.
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
//! The `DepConstructor` enum, together with `DepNode::new()`, ensures that only
//! valid `DepNode` instances can be constructed. For example, the API does not
//! allow for constructing parameterless `DepNode`s with anything other
//! than a zeroed out fingerprint. More generally speaking, it relieves the
//! user of the `DepNode` API of having to know how to compute the expected
//! fingerprint for a given set of node parameters.

use crate::mir::interpret::{GlobalId, LitToConstInput};
use crate::traits;
use crate::traits::query::{
    CanonicalPredicateGoal, CanonicalProjectionGoal, CanonicalTyGoal,
    CanonicalTypeOpAscribeUserTypeGoal, CanonicalTypeOpEqGoal, CanonicalTypeOpNormalizeGoal,
    CanonicalTypeOpProvePredicateGoal, CanonicalTypeOpSubtypeGoal,
};
use crate::ty::subst::{GenericArg, SubstsRef};
use crate::ty::{self, ParamEnvAnd, Ty, TyCtxt};
use rustc_middle::ty::query;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, CRATE_DEF_INDEX};
use rustc_hir::definitions::DefPathHash;
use rustc_hir::HirId;
use rustc_query_system::query::QueryAccessors;
use rustc_serialize::{opaque, Decodable, Decoder, Encodable, Encoder};
use rustc_span::symbol::Symbol;
use std::hash::Hash;

pub use rustc_query_system::dep_graph::{DepContext, DepNodeParams};

pub trait DepKindTrait: std::fmt::Debug + Sync {
    fn index(&self) -> DepKindIndex;

    fn can_reconstruct_query_key(&self) -> bool;

    fn is_anon(&self) -> bool;

    fn is_eval_always(&self) -> bool;

    fn has_params(&self) -> bool;

    fn encode_query_results<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        encoder: &mut query::on_disk_cache::CacheEncoder<'a, 'tcx, opaque::Encoder>,
        query_result_index: &mut query::on_disk_cache::EncodedQueryResultIndex,
    );

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
    ///
    /// When you implement a new query, it will likely have a corresponding new
    /// `DepKind`, and you'll have to support it here in `force_from_dep_node()`. As
    /// a rule of thumb, if your query takes a `DefId` or `LocalDefId` as sole parameter,
    /// then `force_from_dep_node()` should not fail for it. Otherwise, you can just
    /// add it to the "We don't have enough information to reconstruct..." group in
    /// the match below.
    fn force_from_dep_node(&self, tcx: TyCtxt<'_>, dep_node: &DepNode) -> bool;

    fn query_stats(&self, tcx: TyCtxt<'_>) -> Option<query::stats::QueryStats>;

    fn try_load_from_on_disk_cache(&self, tcx: TyCtxt<'_>, dep_node: &DepNode);
}

// erase!() just makes tokens go away. It's used to specify which macro argument
// is repeated (i.e., which sub-expression of the macro we are in) but don't need
// to actually use any of the arguments.
macro_rules! erase {
    ($x:tt) => {{}};
}

macro_rules! is_anon_attr {
    (anon) => {
        true
    };
    ($attr:ident) => {
        false
    };
}

macro_rules! is_eval_always_attr {
    (eval_always) => {
        true
    };
    ($attr:ident) => {
        false
    };
}

macro_rules! contains_anon_attr {
    ($($attr:ident $(($($attr_args:tt)*))* ),*) => ({$(is_anon_attr!($attr) | )* false});
}

macro_rules! contains_eval_always_attr {
    ($($attr:ident $(($($attr_args:tt)*))* ),*) => ({$(is_eval_always_attr!($attr) | )* false});
}

macro_rules! encode_query_results {
    ([][$name:ident][$($args:expr),*]) => {{}};
    ([cached $($rest:tt)*][$name:ident][$($args:expr),*]) => {{
        let ret = query::on_disk_cache::encode_query_results::<
            query::queries::$name<'_>
        >($($args),*);
        match ret { Ok(()) => (), Err(_) => () }
    }};
    ([$other:ident $(($($other_args:tt)*))* $(, $($modifiers:tt)*)*][$name:ident][$($args:expr),*]) => {
        encode_query_results!([$($($modifiers)*)*][$name][$($args),*])
    };
}

macro_rules! define_dep_kinds {
    (<$tcx:tt>
    $(
        [$($attrs:tt)*]
        $name:ident $(( $tuple_arg_ty:ty $(,)? ))*
      ,)*
    ) => (
        $(impl DepKindTrait for dep_kind::$name {
            #[inline]
            fn index(&self) -> DepKindIndex {
                DepKindIndex::$name
            }

            #[inline]
            #[allow(unreachable_code)]
            #[allow(unused_lifetimes)] // inside `tuple_arg_ty`
            fn can_reconstruct_query_key<$tcx>(&self) -> bool {
                if contains_anon_attr!($($attrs)*) {
                    return false;
                }

                // tuple args
                $({
                    return <$tuple_arg_ty as DepNodeParams<TyCtxt<'_>>>
                        ::can_reconstruct_query_key();
                })*

                true
            }

            #[inline]
            fn is_anon(&self) -> bool {
                contains_anon_attr!($($attrs)*)
            }

            #[inline]
            fn is_eval_always(&self) -> bool {
                contains_eval_always_attr!($($attrs)*)
            }

            #[inline]
            #[allow(unreachable_code)]
            fn has_params(&self) -> bool {
                // tuple args
                $({
                    erase!($tuple_arg_ty);
                    return true;
                })*

                false
            }

            #[inline]
            fn encode_query_results<'a, 'tcx>(
                &self,
                _tcx: TyCtxt<'tcx>,
                _encoder: &mut query::on_disk_cache::CacheEncoder<'a, 'tcx, opaque::Encoder>,
                _query_result_index: &mut query::on_disk_cache::EncodedQueryResultIndex,
            ) {
                encode_query_results!([$($attrs)*][$name][_tcx, _encoder, _query_result_index]);
            }

            #[inline]
            fn force_from_dep_node(&self, tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> bool {
                use rustc_query_system::query::force_query;
                use rustc_middle::ty::query::queries;
                #[allow(unused_parens)]
                #[allow(unused_lifetimes)]
                type Key<$tcx> = ($($tuple_arg_ty),*);

                // We must avoid ever having to call `force_from_dep_node()` for a
                // `DepNode::codegen_unit`:
                // Since we cannot reconstruct the query key of a `DepNode::codegen_unit`, we
                // would always end up having to evaluate the first caller of the
                // `codegen_unit` query that *is* reconstructible. This might very well be
                // the `compile_codegen_unit` query, thus re-codegenning the whole CGU just
                // to re-trigger calling the `codegen_unit` query with the right key. At
                // that point we would already have re-done all the work we are trying to
                // avoid doing in the first place.
                // The solution is simple: Just explicitly call the `codegen_unit` query for
                // each CGU, right after partitioning. This way `try_mark_green` will always
                // hit the cache instead of having to go through `force_from_dep_node`.
                // This assertion makes sure, we actually keep applying the solution above.
                debug_assert!(
                    dep_node.kind != &dep_kind::codegen_unit,
                    "calling force_from_dep_node() on DepKind::codegen_unit"
                );

                if !self.can_reconstruct_query_key() {
                    return false;
                }

                debug_assert!(<Key<'_> as DepNodeParams<TyCtxt<'_>>>::can_reconstruct_query_key());

                if let Some(key) = <Key<'_> as DepNodeParams<TyCtxt<'_>>>::recover(tcx, dep_node) {
                    force_query::<queries::$name<'_>, _>(
                        tcx,
                        key,
                        rustc_span::DUMMY_SP,
                        *dep_node
                    );
                    return true;
                }

                false
            }

            #[inline]
            fn query_stats(&self, tcx: TyCtxt<'_>) -> Option<query::stats::QueryStats> {
                let ret = query::stats::stats::<
                    query::Query<'_>,
                    <query::queries::$name<'_> as QueryAccessors<TyCtxt<'_>>>::Cache,
                >(
                    stringify!($name),
                    &tcx.queries.$name,
                );
                Some(ret)
            }

            #[inline]
            fn try_load_from_on_disk_cache<'tcx>(&self, tcx: TyCtxt<'tcx>, dep_node: &DepNode) {
                use rustc_query_system::query::QueryDescription;
                use rustc_middle::ty::query::queries;
                #[allow(unused_parens)]
                #[allow(unused_lifetimes)]
                type Key<$tcx> = ($($tuple_arg_ty),*);

                if !<Key<'_> as DepNodeParams<TyCtxt<'_>>>::can_reconstruct_query_key() {
                    return;
                }

                debug_assert!(tcx.dep_graph
                                 .node_color(dep_node)
                                 .map(|c| c.is_green())
                                 .unwrap_or(false));

                let key = <Key<'_> as DepNodeParams<TyCtxt<'_>>>::recover(tcx, dep_node).unwrap();
                if queries::$name::cache_on_disk(tcx, &key, None) {
                    let _ = tcx.$name(key);
                }
            }
        })*
    )
}

rustc_dep_node_append!([define_dep_kinds!][ <'tcx> ]);

macro_rules! define_dep_nodes {
    (<$tcx:tt>
    $(
        [$($attrs:tt)*]
        $name:ident $(( $tuple_arg_ty:ty $(,)? ))*
      ,)*
    ) => (
        pub mod dep_kind {
            $(
                #[allow(non_camel_case_types)]
                #[derive(Debug)]
                pub struct $name;
            )*
        }

        /// This enum serves as an index into the `DEP_KINDS` array.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable)]
        #[allow(non_camel_case_types)]
        pub enum DepKindIndex {
            $($name),*
        }

        crate static DEP_KINDS: &[DepKind] = &[ $(&dep_kind::$name),* ];

        pub struct DepConstructor;

        #[allow(non_camel_case_types)]
        impl DepConstructor {
            $(
                #[inline(always)]
                #[allow(unreachable_code, non_snake_case)]
                pub fn $name(_tcx: TyCtxt<'_>, $(arg: $tuple_arg_ty)*) -> DepNode {
                    // tuple args
                    $({
                        erase!($tuple_arg_ty);
                        return DepNode::construct(_tcx, &dep_kind::$name, &arg)
                    })*

                    return DepNode::construct(_tcx, &dep_kind::$name, &())
                }
            )*
        }

        fn dep_kind_from_label_string(label: &str) -> Result<DepKind, ()> {
            match label {
                $(stringify!($name) => Ok(&dep_kind::$name),)*
                _ => Err(()),
            }
        }

        /// Contains variant => str representations for constructing
        /// DepNode groups for tests.
        #[allow(dead_code, non_upper_case_globals)]
        pub mod label_strs {
           $(
                pub const $name: &str = stringify!($name);
            )*
        }
    );
}

rustc_dep_node_append!([define_dep_nodes!][ <'tcx>
    // We use this for most things when incr. comp. is turned off.
    [] Null,

    // Represents metadata from an extern crate.
    [eval_always] CrateMetadata(CrateNum),

    [anon] TraitSelect,

    [] CompileCodegenUnit(Symbol),
]);

impl DepKindTrait for dep_kind::Null {
    #[inline]
    fn index(&self) -> DepKindIndex {
        DepKindIndex::Null
    }

    #[inline]
    fn can_reconstruct_query_key(&self) -> bool {
        true
    }

    #[inline]
    fn is_anon(&self) -> bool {
        false
    }

    #[inline]
    fn is_eval_always(&self) -> bool {
        false
    }

    #[inline]
    fn has_params(&self) -> bool {
        false
    }

    #[inline]
    fn encode_query_results<'a, 'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _encoder: &mut query::on_disk_cache::CacheEncoder<'a, 'tcx, opaque::Encoder>,
        _query_result_index: &mut query::on_disk_cache::EncodedQueryResultIndex,
    ) {
    }

    #[inline]
    fn force_from_dep_node(&self, _tcx: TyCtxt<'tcx>, _dep_node: &DepNode) -> bool {
        // Forcing this makes no sense.
        bug!("force_from_dep_node: encountered {:?}", _dep_node);
    }

    #[inline]
    fn query_stats(&self, _tcx: TyCtxt<'_>) -> Option<query::stats::QueryStats> {
        None
    }

    #[inline]
    fn try_load_from_on_disk_cache<'tcx>(&self, _tcx: TyCtxt<'tcx>, _dep_node: &DepNode) {}
}

impl DepKindTrait for dep_kind::CrateMetadata {
    #[inline]
    fn index(&self) -> DepKindIndex {
        DepKindIndex::CrateMetadata
    }

    #[inline]
    fn can_reconstruct_query_key(&self) -> bool {
        <CrateNum as DepNodeParams<TyCtxt<'_>>>::can_reconstruct_query_key()
    }

    #[inline]
    fn is_anon(&self) -> bool {
        false
    }

    #[inline]
    fn is_eval_always(&self) -> bool {
        true
    }

    #[inline]
    fn has_params(&self) -> bool {
        true
    }

    #[inline]
    fn encode_query_results<'a, 'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _encoder: &mut query::on_disk_cache::CacheEncoder<'a, 'tcx, opaque::Encoder>,
        _query_result_index: &mut query::on_disk_cache::EncodedQueryResultIndex,
    ) {
    }

    #[inline]
    fn force_from_dep_node(&self, _tcx: TyCtxt<'tcx>, _dep_node: &DepNode) -> bool {
        // These are inputs that are expected to be pre-allocated and that
        // should therefore always be red or green already.
        if !self.can_reconstruct_query_key() {
            return false;
        }

        bug!("force_from_dep_node: encountered {:?}", _dep_node);
    }

    #[inline]
    fn query_stats(&self, _tcx: TyCtxt<'_>) -> Option<query::stats::QueryStats> {
        None
    }

    #[inline]
    fn try_load_from_on_disk_cache<'tcx>(&self, _tcx: TyCtxt<'tcx>, _dep_node: &DepNode) {}
}

impl DepKindTrait for dep_kind::TraitSelect {
    #[inline]
    fn index(&self) -> DepKindIndex {
        DepKindIndex::TraitSelect
    }

    #[inline]
    fn can_reconstruct_query_key(&self) -> bool {
        false
    }

    #[inline]
    fn is_anon(&self) -> bool {
        true
    }

    #[inline]
    fn is_eval_always(&self) -> bool {
        false
    }

    #[inline]
    fn has_params(&self) -> bool {
        false
    }

    #[inline]
    fn encode_query_results<'a, 'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _encoder: &mut query::on_disk_cache::CacheEncoder<'a, 'tcx, opaque::Encoder>,
        _query_result_index: &mut query::on_disk_cache::EncodedQueryResultIndex,
    ) {
    }

    #[inline]
    fn force_from_dep_node(&self, _tcx: TyCtxt<'tcx>, _dep_node: &DepNode) -> bool {
        // These are anonymous nodes.
        if !self.can_reconstruct_query_key() {
            return false;
        }

        bug!("force_from_dep_node: encountered {:?}", _dep_node);
    }

    #[inline]
    fn query_stats(&self, _tcx: TyCtxt<'_>) -> Option<query::stats::QueryStats> {
        None
    }

    #[inline]
    fn try_load_from_on_disk_cache<'tcx>(&self, _tcx: TyCtxt<'tcx>, _dep_node: &DepNode) {}
}

impl DepKindTrait for dep_kind::CompileCodegenUnit {
    #[inline]
    fn index(&self) -> DepKindIndex {
        DepKindIndex::CompileCodegenUnit
    }

    #[inline]
    fn can_reconstruct_query_key(&self) -> bool {
        <Symbol as DepNodeParams<TyCtxt<'_>>>::can_reconstruct_query_key()
    }

    #[inline]
    fn is_anon(&self) -> bool {
        false
    }

    #[inline]
    fn is_eval_always(&self) -> bool {
        false
    }

    #[inline]
    #[allow(unreachable_code)]
    fn has_params(&self) -> bool {
        true
    }

    #[inline]
    fn encode_query_results<'a, 'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _encoder: &mut query::on_disk_cache::CacheEncoder<'a, 'tcx, opaque::Encoder>,
        _query_result_index: &mut query::on_disk_cache::EncodedQueryResultIndex,
    ) {
    }

    #[inline]
    fn force_from_dep_node(&self, _tcx: TyCtxt<'tcx>, _dep_node: &DepNode) -> bool {
        // We don't have enough information to reconstruct the query key of these.
        if !self.can_reconstruct_query_key() {
            return false;
        }

        bug!("force_from_dep_node: encountered {:?}", _dep_node);
    }

    #[inline]
    fn query_stats(&self, _tcx: TyCtxt<'_>) -> Option<query::stats::QueryStats> {
        None
    }

    #[inline]
    fn try_load_from_on_disk_cache<'tcx>(&self, _tcx: TyCtxt<'tcx>, _dep_node: &DepNode) {}
}

pub type DepKind = &'static dyn DepKindTrait;

impl PartialEq for &dyn DepKindTrait {
    fn eq(&self, other: &Self) -> bool {
        self.index() == other.index()
    }
}
impl Eq for &dyn DepKindTrait {}

impl Hash for &dyn DepKindTrait {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        self.index().hash(hasher)
    }
}

impl<E: Encoder> Encodable<E> for &dyn DepKindTrait {
    fn encode(&self, enc: &mut E) -> Result<(), E::Error> {
        self.index().encode(enc)
    }
}

impl<D: Decoder> Decodable<D> for &dyn DepKindTrait {
    fn decode(dec: &mut D) -> Result<Self, D::Error> {
        let idx = DepKindIndex::decode(dec)?;
        let dk = DEP_KINDS[idx as usize];
        debug_assert!(dk.index() == idx);
        Ok(dk)
    }
}

pub type DepNode = rustc_query_system::dep_graph::DepNode<DepKind>;

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

    /// Used in testing
    fn has_label_string(label: &str) -> bool;
}

impl DepNodeExt for DepNode {
    /// Construct a DepNode from the given DepKind and DefPathHash. This
    /// method will assert that the given DepKind actually requires a
    /// single DefId/DefPathHash parameter.
    fn from_def_path_hash(def_path_hash: DefPathHash, kind: DepKind) -> DepNode {
        debug_assert!(kind.can_reconstruct_query_key() && kind.has_params());
        DepNode { kind, hash: def_path_hash.0 }
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
        if self.kind.can_reconstruct_query_key() {
            let def_path_hash = DefPathHash(self.hash);
            tcx.def_path_hash_to_def_id.as_ref()?.get(&def_path_hash).cloned()
        } else {
            None
        }
    }

    /// Used in testing
    fn from_label_string(label: &str, def_path_hash: DefPathHash) -> Result<DepNode, ()> {
        let kind = dep_kind_from_label_string(label)?;

        if !kind.can_reconstruct_query_key() {
            Err(())
        } else if kind.has_params() {
            Ok(DepNode::from_def_path_hash(def_path_hash, kind))
        } else {
            Ok(DepNode::new_no_params(kind))
        }
    }

    /// Used in testing
    fn has_label_string(label: &str) -> bool {
        dep_kind_from_label_string(label).is_ok()
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for DefId {
    #[inline]
    fn can_reconstruct_query_key() -> bool {
        true
    }

    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        tcx.def_path_hash(*self).0
    }

    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        tcx.def_path_str(*self)
    }

    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx)
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for LocalDefId {
    #[inline]
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
    #[inline]
    fn can_reconstruct_query_key() -> bool {
        true
    }

    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let def_id = DefId { krate: *self, index: CRATE_DEF_INDEX };
        tcx.def_path_hash(def_id).0
    }

    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        tcx.crate_name(*self).to_string()
    }

    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| id.krate)
    }
}

impl<'tcx> DepNodeParams<TyCtxt<'tcx>> for (DefId, DefId) {
    #[inline]
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
    #[inline]
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
