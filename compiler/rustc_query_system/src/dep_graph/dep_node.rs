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
//! `DepNode` definition happens in `rustc_middle` with the `define_dep_nodes!()` macro.
//! This macro defines the `DepKind` enum and a corresponding `DepConstructor` enum. The
//! `DepConstructor` enum links a `DepKind` to the parameters that are needed at runtime in order
//! to construct a valid `DepNode` fingerprint.
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

use super::{DepContext, DepKind, FingerprintStyle};
use crate::ich::StableHashingContext;

use rustc_data_structures::fingerprint::{Fingerprint, PackedFingerprint};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use std::fmt;
use std::hash::Hash;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable)]
pub struct DepNode<K> {
    pub kind: K,
    pub hash: PackedFingerprint,
}

impl<K: DepKind> DepNode<K> {
    /// Creates a new, parameterless DepNode. This method will assert
    /// that the DepNode corresponding to the given DepKind actually
    /// does not require any parameters.
    pub fn new_no_params(kind: K) -> DepNode<K> {
        debug_assert!(!kind.has_params());
        DepNode { kind, hash: Fingerprint::ZERO.into() }
    }

    pub fn construct<Ctxt, Key>(tcx: Ctxt, kind: K, arg: &Key) -> DepNode<K>
    where
        Ctxt: super::DepContext<DepKind = K>,
        Key: DepNodeParams<Ctxt>,
    {
        let hash = arg.to_fingerprint(tcx);
        let dep_node = DepNode { kind, hash: hash.into() };

        #[cfg(debug_assertions)]
        {
            if !kind.fingerprint_style().reconstructible()
                && (tcx.sess().opts.debugging_opts.incremental_info
                    || tcx.sess().opts.debugging_opts.query_dep_graph)
            {
                tcx.dep_graph().register_dep_node_debug_str(dep_node, || arg.to_debug_str(tcx));
            }
        }

        dep_node
    }
}

impl<K: DepKind> fmt::Debug for DepNode<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        K::debug_node(self, f)
    }
}

pub trait DepNodeParams<Ctxt: DepContext>: fmt::Debug + Sized {
    fn fingerprint_style() -> FingerprintStyle;

    /// This method turns the parameters of a DepNodeConstructor into an opaque
    /// Fingerprint to be used in DepNode.
    /// Not all DepNodeParams support being turned into a Fingerprint (they
    /// don't need to if the corresponding DepNode is anonymous).
    fn to_fingerprint(&self, _: Ctxt) -> Fingerprint {
        panic!("Not implemented. Accidentally called on anonymous node?")
    }

    fn to_debug_str(&self, _: Ctxt) -> String {
        format!("{:?}", self)
    }

    /// This method tries to recover the query key from the given `DepNode`,
    /// something which is needed when forcing `DepNode`s during red-green
    /// evaluation. The query system will only call this method if
    /// `fingerprint_style()` is not `FingerprintStyle::Opaque`.
    /// It is always valid to return `None` here, in which case incremental
    /// compilation will treat the query as having changed instead of forcing it.
    fn recover(tcx: Ctxt, dep_node: &DepNode<Ctxt::DepKind>) -> Option<Self>;
}

impl<Ctxt: DepContext, T> DepNodeParams<Ctxt> for T
where
    T: for<'a> HashStable<StableHashingContext<'a>> + fmt::Debug,
{
    #[inline]
    default fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::Opaque
    }

    default fn to_fingerprint(&self, tcx: Ctxt) -> Fingerprint {
        let mut hcx = tcx.create_stable_hashing_context();
        let mut hasher = StableHasher::new();

        self.hash_stable(&mut hcx, &mut hasher);

        hasher.finish()
    }

    default fn to_debug_str(&self, _: Ctxt) -> String {
        format!("{:?}", *self)
    }

    default fn recover(_: Ctxt, _: &DepNode<Ctxt::DepKind>) -> Option<Self> {
        None
    }
}

/// A "work product" corresponds to a `.o` (or other) file that we
/// save in between runs. These IDs do not have a `DefId` but rather
/// some independent path or string that persists between runs without
/// the need to be mapped or unmapped. (This ensures we can serialize
/// them even in the absence of a tcx.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Encodable, Decodable)]
pub struct WorkProductId {
    hash: Fingerprint,
}

impl WorkProductId {
    pub fn from_cgu_name(cgu_name: &str) -> WorkProductId {
        let mut hasher = StableHasher::new();
        cgu_name.len().hash(&mut hasher);
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
