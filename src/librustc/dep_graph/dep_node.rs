// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def_id::{CrateNum, DefId};
use hir::map::DefPathHash;

use ich::Fingerprint;
use ty::TyCtxt;
use rustc_data_structures::stable_hasher::{StableHasher, HashStable};
use ich::StableHashingContext;
use std::hash::Hash;

// erase!() just makes tokens go away. It's used to specify which macro argument
// is repeated (i.e. which sub-expression of the macro we are in) but don't need
// to actually use any of the arguments.
macro_rules! erase {
    ($x:tt) => ({})
}

macro_rules! define_dep_nodes {
    ($(
        $variant:ident $(( $($tuple_arg:tt),* ))*
                       $({ $($struct_arg_name:ident : $struct_arg_ty:ty),* })*
      ),*
    ) => (
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash,
                 RustcEncodable, RustcDecodable)]
        pub enum DepKind {
            $($variant),*
        }

        impl DepKind {
            #[allow(unreachable_code)]
            #[inline]
            pub fn can_reconstruct_query_key(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => {
                            // tuple args
                            $({
                                return <( $($tuple_arg,)* ) as DepNodeParams>
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

            #[allow(unreachable_code)]
            #[inline]
            pub fn has_params(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => {
                            // tuple args
                            $({
                                $(erase!($tuple_arg);)*
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

        pub enum DepConstructor {
            $(
                $variant $(( $($tuple_arg),* ))*
                         $({ $($struct_arg_name : $struct_arg_ty),* })*
            ),*
        }

        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash,
                 RustcEncodable, RustcDecodable)]
        pub struct DepNode {
            pub kind: DepKind,
            pub hash: Fingerprint,
        }

        impl DepNode {
            #[allow(unreachable_code, non_snake_case)]
            pub fn new(tcx: TyCtxt, dep: DepConstructor) -> DepNode {
                match dep {
                    $(
                        DepConstructor :: $variant $(( $($tuple_arg),* ))*
                                                   $({ $($struct_arg_name),* })*
                            =>
                        {
                            // tuple args
                            $({
                                let tupled_args = ( $($tuple_arg,)* );
                                let hash = DepNodeParams::to_fingerprint(&tupled_args,
                                                                         tcx);
                                return DepNode {
                                    kind: DepKind::$variant,
                                    hash
                                };
                            })*

                            // struct args
                            $({
                                let tupled_args = ( $($struct_arg_name,)* );
                                let hash = DepNodeParams::to_fingerprint(&tupled_args,
                                                                         tcx);
                                return DepNode {
                                    kind: DepKind::$variant,
                                    hash
                                };
                            })*

                            DepNode {
                                kind: DepKind::$variant,
                                hash: Fingerprint::zero(),
                            }
                        }
                    )*
                }
            }

            /// Construct a DepNode from the given DepKind and DefPathHash. This
            /// method will assert that the given DepKind actually requires a
            /// single DefId/DefPathHash parameter.
            #[inline]
            pub fn from_def_path_hash(kind: DepKind,
                                      def_path_hash: DefPathHash)
                                      -> DepNode {
                assert!(kind.can_reconstruct_query_key() && kind.has_params());
                DepNode {
                    kind,
                    hash: def_path_hash.0,
                }
            }

            /// Create a new, parameterless DepNode. This method will assert
            /// that the DepNode corresponding to the given DepKind actually
            /// does not require any parameters.
            #[inline]
            pub fn new_no_params(kind: DepKind) -> DepNode {
                assert!(!kind.has_params());
                DepNode {
                    kind,
                    hash: Fingerprint::zero(),
                }
            }

            /// Extract the DefId corresponding to this DepNode. This will work
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
            pub fn extract_def_id(&self, tcx: TyCtxt) -> Option<DefId> {
                if self.kind.can_reconstruct_query_key() {
                    let def_path_hash = DefPathHash(self.hash);
                    tcx.def_path_hash_to_def_id
                       .as_ref()
                       .unwrap()
                       .get(&def_path_hash)
                       .cloned()
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
        }
    );
}

impl DefPathHash {
    #[inline]
    pub fn to_dep_node(self, kind: DepKind) -> DepNode {
        DepNode::from_def_path_hash(kind, self)
    }
}

impl DefId {
    #[inline]
    pub fn to_dep_node(self, tcx: TyCtxt, kind: DepKind) -> DepNode {
        DepNode::from_def_path_hash(kind, tcx.def_path_hash(self))
    }
}

define_dep_nodes!(
    // Represents the `Krate` as a whole (the `hir::Krate` value) (as
    // distinct from the krate module). This is basically a hash of
    // the entire krate, so if you read from `Krate` (e.g., by calling
    // `tcx.hir.krate()`), we will have to assume that any change
    // means that you need to be recompiled. This is because the
    // `Krate` value gives you access to all other items. To avoid
    // this fate, do not call `tcx.hir.krate()`; instead, prefer
    // wrappers like `tcx.visit_all_items_in_krate()`.  If there is no
    // suitable wrapper, you can use `tcx.dep_graph.ignore()` to gain
    // access to the krate, but you must remember to add suitable
    // edges yourself for the individual items that you read.
    Krate,

    // Represents the HIR node with the given node-id
    Hir(DefId),

    // Represents the body of a function or method. The def-id is that of the
    // function/method.
    HirBody(DefId),

    // Represents the metadata for a given HIR node, typically found
    // in an extern crate.
    MetaData(DefId),

    // Represents some artifact that we save to disk. Note that these
    // do not have a def-id as part of their identifier.
    WorkProduct(WorkProductId),

    // Represents different phases in the compiler.
    RegionMaps(DefId),
    Coherence,
    Resolve,
    CoherenceCheckTrait(DefId),
    CoherenceCheckImpl(DefId),
    CoherenceOverlapCheck(DefId),
    CoherenceOverlapCheckSpecial(DefId),
    Variance,
    PrivacyAccessLevels(CrateNum),

    // Represents the MIR for a fn; also used as the task node for
    // things read/modify that MIR.
    MirKrate,
    Mir(DefId),
    MirShim(DefIdList),

    BorrowCheckKrate,
    BorrowCheck(DefId),
    RvalueCheck(DefId),
    Reachability,
    MirKeys,
    LateLintCheck,
    TransCrateItem(DefId),
    TransWriteMetadata,
    CrateVariances,

    // Nodes representing bits of computed IR in the tcx. Each shared
    // table in the tcx (or elsewhere) maps to one of these
    // nodes. Often we map multiple tables to the same node if there
    // is no point in distinguishing them (e.g., both the type and
    // predicates for an item wind up in `ItemSignature`).
    AssociatedItems(DefId),
    ItemSignature(DefId),
    ItemVarianceConstraints(DefId),
    ItemVariances(DefId),
    IsForeignItem(DefId),
    TypeParamPredicates { item_id: DefId, param_id: DefId },
    SizedConstraint(DefId),
    DtorckConstraint(DefId),
    AdtDestructor(DefId),
    AssociatedItemDefIds(DefId),
    InherentImpls(DefId),
    TypeckBodiesKrate,
    TypeckTables(DefId),
    UsedTraitImports(DefId),
    ConstEval(DefId),
    SymbolName(DefId),
    SpecializationGraph(DefId),
    ObjectSafety(DefId),
    IsCopy(DefId),
    IsSized(DefId),
    IsFreeze(DefId),
    NeedsDrop(DefId),
    Layout(DefId),

    // The set of impls for a given trait. Ultimately, it would be
    // nice to get more fine-grained here (e.g., to include a
    // simplified type), but we can't do that until we restructure the
    // HIR to distinguish the *header* of an impl from its body.  This
    // is because changes to the header may change the self-type of
    // the impl and hence would require us to be more conservative
    // than changes in the impl body.
    TraitImpls(DefId),

    AllLocalTraitImpls,

    // Nodes representing caches. To properly handle a true cache, we
    // don't use a DepTrackingMap, but rather we push a task node.
    // Otherwise the write into the map would be incorrectly
    // attributed to the first task that happened to fill the cache,
    // which would yield an overly conservative dep-graph.
    TraitItems(DefId),
    ReprHints(DefId),

    // Trait selection cache is a little funny. Given a trait
    // reference like `Foo: SomeTrait<Bar>`, there could be
    // arbitrarily many def-ids to map on in there (e.g., `Foo`,
    // `SomeTrait`, `Bar`). We could have a vector of them, but it
    // requires heap-allocation, and trait sel in general can be a
    // surprisingly hot path. So instead we pick two def-ids: the
    // trait def-id, and the first def-id in the input types. If there
    // is no def-id in the input types, then we use the trait def-id
    // again. So for example:
    //
    // - `i32: Clone` -> `TraitSelect { trait_def_id: Clone, self_def_id: Clone }`
    // - `u32: Clone` -> `TraitSelect { trait_def_id: Clone, self_def_id: Clone }`
    // - `Clone: Clone` -> `TraitSelect { trait_def_id: Clone, self_def_id: Clone }`
    // - `Vec<i32>: Clone` -> `TraitSelect { trait_def_id: Clone, self_def_id: Vec }`
    // - `String: Clone` -> `TraitSelect { trait_def_id: Clone, self_def_id: String }`
    // - `Foo: Trait<Bar>` -> `TraitSelect { trait_def_id: Trait, self_def_id: Foo }`
    // - `Foo: Trait<i32>` -> `TraitSelect { trait_def_id: Trait, self_def_id: Foo }`
    // - `(Foo, Bar): Trait` -> `TraitSelect { trait_def_id: Trait, self_def_id: Foo }`
    // - `i32: Trait<Foo>` -> `TraitSelect { trait_def_id: Trait, self_def_id: Foo }`
    //
    // You can see that we map many trait refs to the same
    // trait-select node.  This is not a problem, it just means
    // imprecision in our dep-graph tracking.  The important thing is
    // that for any given trait-ref, we always map to the **same**
    // trait-select node.
    TraitSelect { trait_def_id: DefId, input_def_id: DefId },

    // For proj. cache, we just keep a list of all def-ids, since it is
    // not a hotspot.
    ProjectionCache { def_ids: DefIdList },

    ParamEnv(DefId),
    DescribeDef(DefId),
    DefSpan(DefId),
    Stability(DefId),
    Deprecation(DefId),
    ItemBodyNestedBodies(DefId),
    ConstIsRvaluePromotableToStatic(DefId),
    ImplParent(DefId),
    TraitOfItem(DefId),
    IsExportedSymbol(DefId),
    IsMirAvailable(DefId),
    ItemAttrs(DefId),
    FnArgNames(DefId)
);

trait DepNodeParams<'a, 'gcx: 'tcx + 'a, 'tcx: 'a> {
    const CAN_RECONSTRUCT_QUERY_KEY: bool;
    fn to_fingerprint(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Fingerprint;
}

impl<'a, 'gcx: 'tcx + 'a, 'tcx: 'a, T> DepNodeParams<'a, 'gcx, 'tcx> for T
    where T: HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
{
    default const CAN_RECONSTRUCT_QUERY_KEY: bool = false;

    default fn to_fingerprint(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Fingerprint {
        let mut hcx = StableHashingContext::new(tcx);
        let mut hasher = StableHasher::new();

        self.hash_stable(&mut hcx, &mut hasher);

        hasher.finish()
    }
}

impl<'a, 'gcx: 'tcx + 'a, 'tcx: 'a> DepNodeParams<'a, 'gcx, 'tcx> for (DefId,) {
    const CAN_RECONSTRUCT_QUERY_KEY: bool = true;

    fn to_fingerprint(&self, tcx: TyCtxt) -> Fingerprint {
        tcx.def_path_hash(self.0).0
    }
}

/// A "work product" corresponds to a `.o` (or other) file that we
/// save in between runs. These ids do not have a DefId but rather
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

    pub fn to_dep_node(self) -> DepNode {
        DepNode {
            kind: DepKind::WorkProduct,
            hash: self.hash,
        }
    }
}

impl_stable_hash_for!(struct ::dep_graph::WorkProductId {
    hash
});

type DefIdList = Vec<DefId>;
