//! This module defines the `DepNode` type which the compiler uses to represent
//! nodes in the dependency graph. A `DepNode` consists of a `DepKind` (which
//! specifies the kind of thing it represents, like a piece of HIR, MIR, etc)
//! and a `Fingerprint`, a 128 bit hash value the exact meaning of which
//! depends on the node's `DepKind`. Together, the kind and the fingerprint
//! fully identify a dependency node, even across multiple compilation sessions.
//! In other words, the value of the fingerprint does not depend on anything
//! that is specific to a given compilation session, like an unpredictable
//! interning key (e.g., NodeId, DefId, Symbol) or the numeric value of a
//! pointer. The concept behind this could be compared to how git commit hashes
//! uniquely identify a given commit and has a few advantages:
//!
//! * A `DepNode` can simply be serialized to disk and loaded in another session
//!   without the need to do any "rebasing (like we have to do for Spans and
//!   NodeIds) or "retracing" like we had to do for `DefId` in earlier
//!   implementations of the dependency graph.
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
//!   `DepNode`s, like `Krate`, represent global concepts with only one value.
//! * Whether it is possible, in principle, to reconstruct a query key from a
//!   given `DepNode`. Many `DepKind`s only require a single `DefId` parameter,
//!   in which case it is possible to map the node's fingerprint back to the
//!   `DefId` it was computed from. In other cases, too much information gets
//!   lost during fingerprint computation.
//!
//! The `DepConstructor` enum, together with `DepNode::new()` ensures that only
//! valid `DepNode` instances can be constructed. For example, the API does not
//! allow for constructing parameterless `DepNode`s with anything other
//! than a zeroed out fingerprint. More generally speaking, it relieves the
//! user of the `DepNode` API of having to know how to compute the expected
//! fingerprint for a given set of node parameters.

use crate::mir;
use crate::mir::interpret::GlobalId;
use crate::hir::def_id::{CrateNum, DefId, DefIndex, CRATE_DEF_INDEX};
use crate::hir::map::DefPathHash;
use crate::hir::HirId;

use crate::ich::{Fingerprint, StableHashingContext};
use rustc_data_structures::stable_hasher::{StableHasher, HashStable};
use std::fmt;
use std::hash::Hash;
use syntax_pos::symbol::InternedString;
use crate::traits;
use crate::traits::query::{
    CanonicalProjectionGoal, CanonicalTyGoal, CanonicalTypeOpAscribeUserTypeGoal,
    CanonicalTypeOpEqGoal, CanonicalTypeOpSubtypeGoal, CanonicalPredicateGoal,
    CanonicalTypeOpProvePredicateGoal, CanonicalTypeOpNormalizeGoal,
};
use crate::ty::{self, TyCtxt, ParamEnvAnd, Ty};
use crate::ty::subst::SubstsRef;

// erase!() just makes tokens go away. It's used to specify which macro argument
// is repeated (i.e., which sub-expression of the macro we are in) but don't need
// to actually use any of the arguments.
macro_rules! erase {
    ($x:tt) => ({})
}

macro_rules! replace {
    ($x:tt with $($y:tt)*) => ($($y)*)
}

macro_rules! is_anon_attr {
    (anon) => (true);
    ($attr:ident) => (false);
}

macro_rules! is_eval_always_attr {
    (eval_always) => (true);
    ($attr:ident) => (false);
}

macro_rules! contains_anon_attr {
    ($($attr:ident),*) => ({$(is_anon_attr!($attr) | )* false});
}

macro_rules! contains_eval_always_attr {
    ($($attr:ident),*) => ({$(is_eval_always_attr!($attr) | )* false});
}

macro_rules! define_dep_nodes {
    (<$tcx:tt>
    $(
        [$($attr:ident),* ]
        $variant:ident $(( $tuple_arg_ty:ty $(,)? ))*
                       $({ $($struct_arg_name:ident : $struct_arg_ty:ty),* })*
      ,)*
    ) => (
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash,
                 RustcEncodable, RustcDecodable)]
        pub enum DepKind {
            $($variant),*
        }

        impl DepKind {
            #[allow(unreachable_code)]
            #[inline]
            pub fn can_reconstruct_query_key<$tcx>(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => {
                            if contains_anon_attr!($($attr),*) {
                                return false;
                            }

                            // tuple args
                            $({
                                return <$tuple_arg_ty as DepNodeParams>
                                    ::CAN_RECONSTRUCT_QUERY_KEY;
                            })*

                            // struct args
                            $({

                                return <( $($struct_arg_ty,)* ) as DepNodeParams>
                                    ::CAN_RECONSTRUCT_QUERY_KEY;
                            })*

                            true
                        }
                    )*
                }
            }

            pub fn is_anon(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => { contains_anon_attr!($($attr),*) }
                    )*
                }
            }

            #[inline(always)]
            pub fn is_eval_always(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => { contains_eval_always_attr!($($attr), *) }
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

                            // struct args
                            $({
                                $(erase!($struct_arg_name);)*
                                return true;
                            })*

                            false
                        }
                    )*
                }
            }
        }

        pub enum DepConstructor<$tcx> {
            $(
                $variant $(( $tuple_arg_ty ))*
                         $({ $($struct_arg_name : $struct_arg_ty),* })*
            ),*
        }

        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
                 RustcEncodable, RustcDecodable)]
        pub struct DepNode {
            pub kind: DepKind,
            pub hash: Fingerprint,
        }

        impl DepNode {
            #[allow(unreachable_code, non_snake_case)]
            #[inline(always)]
            pub fn new<'tcx>(tcx: TyCtxt<'tcx>,
                                       dep: DepConstructor<'tcx>)
                                       -> DepNode
            {
                match dep {
                    $(
                        DepConstructor :: $variant $(( replace!(($tuple_arg_ty) with arg) ))*
                                                   $({ $($struct_arg_name),* })*
                            =>
                        {
                            // tuple args
                            $({
                                erase!($tuple_arg_ty);
                                let hash = DepNodeParams::to_fingerprint(&arg, tcx);
                                let dep_node = DepNode {
                                    kind: DepKind::$variant,
                                    hash
                                };

                                if cfg!(debug_assertions) &&
                                   !dep_node.kind.can_reconstruct_query_key() &&
                                   (tcx.sess.opts.debugging_opts.incremental_info ||
                                    tcx.sess.opts.debugging_opts.query_dep_graph)
                                {
                                    tcx.dep_graph.register_dep_node_debug_str(dep_node, || {
                                        arg.to_debug_str(tcx)
                                    });
                                }

                                return dep_node;
                            })*

                            // struct args
                            $({
                                let tupled_args = ( $($struct_arg_name,)* );
                                let hash = DepNodeParams::to_fingerprint(&tupled_args,
                                                                         tcx);
                                let dep_node = DepNode {
                                    kind: DepKind::$variant,
                                    hash
                                };

                                if cfg!(debug_assertions) &&
                                   !dep_node.kind.can_reconstruct_query_key() &&
                                   (tcx.sess.opts.debugging_opts.incremental_info ||
                                    tcx.sess.opts.debugging_opts.query_dep_graph)
                                {
                                    tcx.dep_graph.register_dep_node_debug_str(dep_node, || {
                                        tupled_args.to_debug_str(tcx)
                                    });
                                }

                                return dep_node;
                            })*

                            DepNode {
                                kind: DepKind::$variant,
                                hash: Fingerprint::ZERO,
                            }
                        }
                    )*
                }
            }

            /// Construct a DepNode from the given DepKind and DefPathHash. This
            /// method will assert that the given DepKind actually requires a
            /// single DefId/DefPathHash parameter.
            #[inline(always)]
            pub fn from_def_path_hash(kind: DepKind,
                                      def_path_hash: DefPathHash)
                                      -> DepNode {
                debug_assert!(kind.can_reconstruct_query_key() && kind.has_params());
                DepNode {
                    kind,
                    hash: def_path_hash.0,
                }
            }

            /// Creates a new, parameterless DepNode. This method will assert
            /// that the DepNode corresponding to the given DepKind actually
            /// does not require any parameters.
            #[inline(always)]
            pub fn new_no_params(kind: DepKind) -> DepNode {
                debug_assert!(!kind.has_params());
                DepNode {
                    kind,
                    hash: Fingerprint::ZERO,
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
            #[inline]
            pub fn extract_def_id(&self, tcx: TyCtxt<'_>) -> Option<DefId> {
                if self.kind.can_reconstruct_query_key() {
                    let def_path_hash = DefPathHash(self.hash);
                    tcx.def_path_hash_to_def_id.as_ref()?
                        .get(&def_path_hash).cloned()
                } else {
                    None
                }
            }

            /// Used in testing
            pub fn from_label_string(label: &str,
                                     def_path_hash: DefPathHash)
                                     -> Result<DepNode, ()> {
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
                    Ok(def_path_hash.to_dep_node(kind))
                } else {
                    Ok(DepNode::new_no_params(kind))
                }
            }

            /// Used in testing
            pub fn has_label_string(label: &str) -> bool {
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

impl fmt::Debug for DepNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.kind)?;

        if !self.kind.has_params() && !self.kind.is_anon() {
            return Ok(());
        }

        write!(f, "(")?;

        crate::ty::tls::with_opt(|opt_tcx| {
            if let Some(tcx) = opt_tcx {
                if let Some(def_id) = self.extract_def_id(tcx) {
                    write!(f, "{}", tcx.def_path_debug_str(def_id))?;
                } else if let Some(ref s) = tcx.dep_graph.dep_node_debug_str(*self) {
                    write!(f, "{}", s)?;
                } else {
                    write!(f, "{}", self.hash)?;
                }
            } else {
                write!(f, "{}", self.hash)?;
            }
            Ok(())
        })?;

        write!(f, ")")
    }
}


impl DefPathHash {
    #[inline(always)]
    pub fn to_dep_node(self, kind: DepKind) -> DepNode {
        DepNode::from_def_path_hash(kind, self)
    }
}

impl DefId {
    #[inline(always)]
    pub fn to_dep_node(self, tcx: TyCtxt<'_>, kind: DepKind) -> DepNode {
        DepNode::from_def_path_hash(kind, tcx.def_path_hash(self))
    }
}

rustc_dep_node_append!([define_dep_nodes!][ <'tcx>
    // We use this for most things when incr. comp. is turned off.
    [] Null,

    // Represents the `Krate` as a whole (the `hir::Krate` value) (as
    // distinct from the krate module). This is basically a hash of
    // the entire krate, so if you read from `Krate` (e.g., by calling
    // `tcx.hir().krate()`), we will have to assume that any change
    // means that you need to be recompiled. This is because the
    // `Krate` value gives you access to all other items. To avoid
    // this fate, do not call `tcx.hir().krate()`; instead, prefer
    // wrappers like `tcx.visit_all_items_in_krate()`.  If there is no
    // suitable wrapper, you can use `tcx.dep_graph.ignore()` to gain
    // access to the krate, but you must remember to add suitable
    // edges yourself for the individual items that you read.
    [eval_always] Krate,

    // Represents the body of a function or method. The def-id is that of the
    // function/method.
    [eval_always] HirBody(DefId),

    // Represents the HIR node with the given node-id
    [eval_always] Hir(DefId),

    // Represents metadata from an extern crate.
    [eval_always] CrateMetadata(CrateNum),

    [eval_always] AllLocalTraitImpls,

    [anon] TraitSelect,

    [] CompileCodegenUnit(InternedString),

    [eval_always] Analysis(CrateNum),
]);

pub trait RecoverKey<'tcx>: Sized {
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self>;
}

impl RecoverKey<'tcx> for CrateNum {
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| id.krate)
    }
}

impl RecoverKey<'tcx> for DefId {
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx)
    }
}

impl RecoverKey<'tcx> for DefIndex {
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| id.index)
    }
}

trait DepNodeParams<'tcx>: fmt::Debug {
    const CAN_RECONSTRUCT_QUERY_KEY: bool;

    /// This method turns the parameters of a DepNodeConstructor into an opaque
    /// Fingerprint to be used in DepNode.
    /// Not all DepNodeParams support being turned into a Fingerprint (they
    /// don't need to if the corresponding DepNode is anonymous).
    fn to_fingerprint(&self, _: TyCtxt<'tcx>) -> Fingerprint {
        panic!("Not implemented. Accidentally called on anonymous node?")
    }

    fn to_debug_str(&self, _: TyCtxt<'tcx>) -> String {
        format!("{:?}", self)
    }
}

impl<'tcx, T> DepNodeParams<'tcx> for T
where
    T: HashStable<StableHashingContext<'tcx>> + fmt::Debug,
{
    default const CAN_RECONSTRUCT_QUERY_KEY: bool = false;

    default fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let mut hcx = tcx.create_stable_hashing_context();
        let mut hasher = StableHasher::new();

        self.hash_stable(&mut hcx, &mut hasher);

        hasher.finish()
    }

    default fn to_debug_str(&self, _: TyCtxt<'tcx>) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> DepNodeParams<'tcx> for DefId {
    const CAN_RECONSTRUCT_QUERY_KEY: bool = true;

    fn to_fingerprint(&self, tcx: TyCtxt<'_>) -> Fingerprint {
        tcx.def_path_hash(*self).0
    }

    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        tcx.def_path_str(*self)
    }
}

impl<'tcx> DepNodeParams<'tcx> for DefIndex {
    const CAN_RECONSTRUCT_QUERY_KEY: bool = true;

    fn to_fingerprint(&self, tcx: TyCtxt<'_>) -> Fingerprint {
        tcx.hir().definitions().def_path_hash(*self).0
    }

    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        tcx.def_path_str(DefId::local(*self))
    }
}

impl<'tcx> DepNodeParams<'tcx> for CrateNum {
    const CAN_RECONSTRUCT_QUERY_KEY: bool = true;

    fn to_fingerprint(&self, tcx: TyCtxt<'_>) -> Fingerprint {
        let def_id = DefId {
            krate: *self,
            index: CRATE_DEF_INDEX,
        };
        tcx.def_path_hash(def_id).0
    }

    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        tcx.crate_name(*self).as_str().to_string()
    }
}

impl<'tcx> DepNodeParams<'tcx> for (DefId, DefId) {
    const CAN_RECONSTRUCT_QUERY_KEY: bool = false;

    // We actually would not need to specialize the implementation of this
    // method but it's faster to combine the hashes than to instantiate a full
    // hashing context and stable-hashing state.
    fn to_fingerprint(&self, tcx: TyCtxt<'_>) -> Fingerprint {
        let (def_id_0, def_id_1) = *self;

        let def_path_hash_0 = tcx.def_path_hash(def_id_0);
        let def_path_hash_1 = tcx.def_path_hash(def_id_1);

        def_path_hash_0.0.combine(def_path_hash_1.0)
    }

    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        let (def_id_0, def_id_1) = *self;

        format!("({}, {})",
                tcx.def_path_debug_str(def_id_0),
                tcx.def_path_debug_str(def_id_1))
    }
}

impl<'tcx> DepNodeParams<'tcx> for HirId {
    const CAN_RECONSTRUCT_QUERY_KEY: bool = false;

    // We actually would not need to specialize the implementation of this
    // method but it's faster to combine the hashes than to instantiate a full
    // hashing context and stable-hashing state.
    fn to_fingerprint(&self, tcx: TyCtxt<'_>) -> Fingerprint {
        let HirId {
            owner,
            local_id,
        } = *self;

        let def_path_hash = tcx.def_path_hash(DefId::local(owner));
        let local_id = Fingerprint::from_smaller_hash(local_id.as_u32().into());

        def_path_hash.0.combine(local_id)
    }
}

/// A "work product" corresponds to a `.o` (or other) file that we
/// save in between runs. These IDs do not have a `DefId` but rather
/// some independent path or string that persists between runs without
/// the need to be mapped or unmapped. (This ensures we can serialize
/// them even in the absence of a tcx.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash,
         RustcEncodable, RustcDecodable)]
pub struct WorkProductId {
    hash: Fingerprint
}

impl WorkProductId {
    pub fn from_cgu_name(cgu_name: &str) -> WorkProductId {
        let mut hasher = StableHasher::new();
        cgu_name.len().hash(&mut hasher);
        cgu_name.hash(&mut hasher);
        WorkProductId {
            hash: hasher.finish()
        }
    }

    pub fn from_fingerprint(fingerprint: Fingerprint) -> WorkProductId {
        WorkProductId {
            hash: fingerprint
        }
    }
}

impl_stable_hash_for!(struct crate::dep_graph::WorkProductId {
    hash
});
