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

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, CRATE_DEF_INDEX};
use rustc_hir::definitions::DefPathHash;
use rustc_hir::HirId;
use rustc_span::symbol::Symbol;
use std::hash::Hash;

pub use rustc_query_system::dep_graph::{DepContext, DepNodeParams};

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

macro_rules! define_dep_nodes {
    (<$tcx:tt>
    $(
        [$($attrs:tt)*]
        $variant:ident $(( $tuple_arg_ty:ty $(,)? ))*
      ,)*
    ) => (
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable)]
        #[allow(non_camel_case_types)]
        pub enum DepKind {
            $($variant),*
        }

        impl DepKind {
            #[allow(unreachable_code)]
            pub fn can_reconstruct_query_key<$tcx>(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => {
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
                    )*
                }
            }

            pub fn is_anon(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => { contains_anon_attr!($($attrs)*) }
                    )*
                }
            }

            pub fn is_eval_always(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => { contains_eval_always_attr!($($attrs)*) }
                    )*
                }
            }

            #[allow(unreachable_code)]
            pub fn has_params(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => {
                            // tuple args
                            $({
                                erase!($tuple_arg_ty);
                                return true;
                            })*

                            false
                        }
                    )*
                }
            }
        }

        pub struct DepConstructor;

        #[allow(non_camel_case_types)]
        impl DepConstructor {
            $(
                #[inline(always)]
                #[allow(unreachable_code, non_snake_case)]
                pub fn $variant(_tcx: TyCtxt<'_>, $(arg: $tuple_arg_ty)*) -> DepNode {
                    // tuple args
                    $({
                        erase!($tuple_arg_ty);
                        return DepNode::construct(_tcx, DepKind::$variant, &arg)
                    })*

                    return DepNode::construct(_tcx, DepKind::$variant, &())
                }
            )*
        }

        pub type DepNode = rustc_query_system::dep_graph::DepNode<DepKind>;

        // We keep a lot of `DepNode`s in memory during compilation. It's not
        // required that their size stay the same, but we don't want to change
        // it inadvertently. This assert just ensures we're aware of any change.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        static_assert_size!(DepNode, 17);

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        static_assert_size!(DepNode, 24);

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
            fn from_label_string(label: &str, def_path_hash: DefPathHash)
                -> Result<Self, ()>;

            /// Used in testing
            fn has_label_string(label: &str) -> bool;
        }

        impl DepNodeExt for DepNode {
            /// Construct a DepNode from the given DepKind and DefPathHash. This
            /// method will assert that the given DepKind actually requires a
            /// single DefId/DefPathHash parameter.
            fn from_def_path_hash(def_path_hash: DefPathHash, kind: DepKind) -> DepNode {
                debug_assert!(kind.can_reconstruct_query_key() && kind.has_params());
                DepNode {
                    kind,
                    hash: def_path_hash.0.into(),
                }
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
                    let def_path_hash = DefPathHash(self.hash.into());
                    tcx.def_path_hash_to_def_id.as_ref()?.get(&def_path_hash).cloned()
                } else {
                    None
                }
            }

            /// Used in testing
            fn from_label_string(label: &str, def_path_hash: DefPathHash) -> Result<DepNode, ()> {
                let kind = match label {
                    $(
                        stringify!($variant) => DepKind::$variant,
                    )*
                    _ => return Err(()),
                };

                if !kind.can_reconstruct_query_key() {
                    return Err(());
                }

                if kind.has_params() {
                    Ok(DepNode::from_def_path_hash(def_path_hash, kind))
                } else {
                    Ok(DepNode::new_no_params(kind))
                }
            }

            /// Used in testing
            fn has_label_string(label: &str) -> bool {
                match label {
                    $(
                        stringify!($variant) => true,
                    )*
                    _ => false,
                }
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

    // Represents metadata from an extern crate.
    [eval_always] CrateMetadata(CrateNum),

    [anon] TraitSelect,

    [] CompileCodegenUnit(Symbol),
]);

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
