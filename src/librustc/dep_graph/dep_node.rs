// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def_id::CrateNum;
use std::fmt::Debug;
use std::sync::Arc;

macro_rules! try_opt {
    ($e:expr) => (
        match $e {
            Some(r) => r,
            None => return None,
        }
    )
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable)]
pub enum DepNode<D: Clone + Debug> {
    // The `D` type is "how definitions are identified".
    // During compilation, it is always `DefId`, but when serializing
    // it is mapped to `DefPath`.

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
    Hir(D),

    // Represents the body of a function or method. The def-id is that of the
    // function/method.
    HirBody(D),

    // Represents the metadata for a given HIR node, typically found
    // in an extern crate.
    MetaData(D),

    // Represents some piece of metadata global to its crate.
    GlobalMetaData(D, GlobalMetaDataKind),

    // Represents some artifact that we save to disk. Note that these
    // do not have a def-id as part of their identifier.
    WorkProduct(Arc<WorkProductId>),

    // Represents different phases in the compiler.
    RegionMaps(D),
    Coherence,
    Resolve,
    CoherenceCheckTrait(D),
    CoherenceCheckImpl(D),
    CoherenceOverlapCheck(D),
    CoherenceOverlapCheckSpecial(D),
    Variance,
    PrivacyAccessLevels(CrateNum),

    // Represents the MIR for a fn; also used as the task node for
    // things read/modify that MIR.
    MirKrate,
    Mir(D),
    MirShim(Vec<D>),

    BorrowCheckKrate,
    BorrowCheck(D),
    RvalueCheck(D),
    Reachability,
    MirKeys,
    LateLintCheck,
    TransCrateItem(D),
    TransWriteMetadata,
    CrateVariances,

    // Nodes representing bits of computed IR in the tcx. Each shared
    // table in the tcx (or elsewhere) maps to one of these
    // nodes. Often we map multiple tables to the same node if there
    // is no point in distinguishing them (e.g., both the type and
    // predicates for an item wind up in `ItemSignature`).
    AssociatedItems(D),
    ItemSignature(D),
    ItemVarianceConstraints(D),
    ItemVariances(D),
    IsForeignItem(D),
    TypeParamPredicates((D, D)),
    SizedConstraint(D),
    DtorckConstraint(D),
    AdtDestructor(D),
    AssociatedItemDefIds(D),
    InherentImpls(D),
    TypeckBodiesKrate,
    TypeckTables(D),
    UsedTraitImports(D),
    ConstEval(D),
    SymbolName(D),
    SpecializationGraph(D),
    ObjectSafety(D),
    IsCopy(D),
    IsSized(D),
    IsFreeze(D),

    // The set of impls for a given trait. Ultimately, it would be
    // nice to get more fine-grained here (e.g., to include a
    // simplified type), but we can't do that until we restructure the
    // HIR to distinguish the *header* of an impl from its body.  This
    // is because changes to the header may change the self-type of
    // the impl and hence would require us to be more conservative
    // than changes in the impl body.
    TraitImpls(D),

    AllLocalTraitImpls,

    // Nodes representing caches. To properly handle a true cache, we
    // don't use a DepTrackingMap, but rather we push a task node.
    // Otherwise the write into the map would be incorrectly
    // attributed to the first task that happened to fill the cache,
    // which would yield an overly conservative dep-graph.
    TraitItems(D),
    ReprHints(D),

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
    TraitSelect { trait_def_id: D, input_def_id: D },

    // For proj. cache, we just keep a list of all def-ids, since it is
    // not a hotspot.
    ProjectionCache { def_ids: Vec<D> },

    DescribeDef(D),
    DefSpan(D),
    Stability(D),
    Deprecation(D),
    ItemBodyNestedBodies(D),
    ConstIsRvaluePromotableToStatic(D),
    ImplParent(D),
    TraitOfItem(D),
    IsExportedSymbol(D),
    IsMirAvailable(D),
    ItemAttrs(D),
    FnArgNames(D),
    FileMap(D, Arc<String>),
}

impl<D: Clone + Debug> DepNode<D> {
    /// Used in testing
    pub fn from_label_string(label: &str, data: D) -> Result<DepNode<D>, ()> {
        macro_rules! check {
            ($($name:ident,)*) => {
                match label {
                    $(stringify!($name) => Ok(DepNode::$name(data)),)*
                    _ => Err(())
                }
            }
        }

        if label == "Krate" {
            // special case
            return Ok(DepNode::Krate);
        }

        check! {
            BorrowCheck,
            Hir,
            HirBody,
            TransCrateItem,
            AssociatedItems,
            ItemSignature,
            ItemVariances,
            IsForeignItem,
            AssociatedItemDefIds,
            InherentImpls,
            TypeckTables,
            UsedTraitImports,
            TraitImpls,
            ReprHints,
        }
    }

    pub fn map_def<E, OP>(&self, mut op: OP) -> Option<DepNode<E>>
        where OP: FnMut(&D) -> Option<E>, E: Clone + Debug
    {
        use self::DepNode::*;

        match *self {
            Krate => Some(Krate),
            BorrowCheckKrate => Some(BorrowCheckKrate),
            MirKrate => Some(MirKrate),
            TypeckBodiesKrate => Some(TypeckBodiesKrate),
            Coherence => Some(Coherence),
            CrateVariances => Some(CrateVariances),
            Resolve => Some(Resolve),
            Variance => Some(Variance),
            PrivacyAccessLevels(k) => Some(PrivacyAccessLevels(k)),
            Reachability => Some(Reachability),
            MirKeys => Some(MirKeys),
            LateLintCheck => Some(LateLintCheck),
            TransWriteMetadata => Some(TransWriteMetadata),

            // work product names do not need to be mapped, because
            // they are always absolute.
            WorkProduct(ref id) => Some(WorkProduct(id.clone())),

            IsCopy(ref d) => op(d).map(IsCopy),
            IsSized(ref d) => op(d).map(IsSized),
            IsFreeze(ref d) => op(d).map(IsFreeze),
            Hir(ref d) => op(d).map(Hir),
            HirBody(ref d) => op(d).map(HirBody),
            MetaData(ref d) => op(d).map(MetaData),
            CoherenceCheckTrait(ref d) => op(d).map(CoherenceCheckTrait),
            CoherenceCheckImpl(ref d) => op(d).map(CoherenceCheckImpl),
            CoherenceOverlapCheck(ref d) => op(d).map(CoherenceOverlapCheck),
            CoherenceOverlapCheckSpecial(ref d) => op(d).map(CoherenceOverlapCheckSpecial),
            Mir(ref d) => op(d).map(Mir),
            MirShim(ref def_ids) => {
                let def_ids: Option<Vec<E>> = def_ids.iter().map(op).collect();
                def_ids.map(MirShim)
            }
            BorrowCheck(ref d) => op(d).map(BorrowCheck),
            RegionMaps(ref d) => op(d).map(RegionMaps),
            RvalueCheck(ref d) => op(d).map(RvalueCheck),
            TransCrateItem(ref d) => op(d).map(TransCrateItem),
            AssociatedItems(ref d) => op(d).map(AssociatedItems),
            ItemSignature(ref d) => op(d).map(ItemSignature),
            ItemVariances(ref d) => op(d).map(ItemVariances),
            ItemVarianceConstraints(ref d) => op(d).map(ItemVarianceConstraints),
            IsForeignItem(ref d) => op(d).map(IsForeignItem),
            TypeParamPredicates((ref item, ref param)) => {
                Some(TypeParamPredicates((try_opt!(op(item)), try_opt!(op(param)))))
            }
            SizedConstraint(ref d) => op(d).map(SizedConstraint),
            DtorckConstraint(ref d) => op(d).map(DtorckConstraint),
            AdtDestructor(ref d) => op(d).map(AdtDestructor),
            AssociatedItemDefIds(ref d) => op(d).map(AssociatedItemDefIds),
            InherentImpls(ref d) => op(d).map(InherentImpls),
            TypeckTables(ref d) => op(d).map(TypeckTables),
            UsedTraitImports(ref d) => op(d).map(UsedTraitImports),
            ConstEval(ref d) => op(d).map(ConstEval),
            SymbolName(ref d) => op(d).map(SymbolName),
            SpecializationGraph(ref d) => op(d).map(SpecializationGraph),
            ObjectSafety(ref d) => op(d).map(ObjectSafety),
            TraitImpls(ref d) => op(d).map(TraitImpls),
            AllLocalTraitImpls => Some(AllLocalTraitImpls),
            TraitItems(ref d) => op(d).map(TraitItems),
            ReprHints(ref d) => op(d).map(ReprHints),
            TraitSelect { ref trait_def_id, ref input_def_id } => {
                op(trait_def_id).and_then(|trait_def_id| {
                    op(input_def_id).and_then(|input_def_id| {
                        Some(TraitSelect { trait_def_id: trait_def_id,
                                           input_def_id: input_def_id })
                    })
                })
            }
            ProjectionCache { ref def_ids } => {
                let def_ids: Option<Vec<E>> = def_ids.iter().map(op).collect();
                def_ids.map(|d| ProjectionCache { def_ids: d })
            }
            DescribeDef(ref d) => op(d).map(DescribeDef),
            DefSpan(ref d) => op(d).map(DefSpan),
            Stability(ref d) => op(d).map(Stability),
            Deprecation(ref d) => op(d).map(Deprecation),
            ItemAttrs(ref d) => op(d).map(ItemAttrs),
            FnArgNames(ref d) => op(d).map(FnArgNames),
            ImplParent(ref d) => op(d).map(ImplParent),
            TraitOfItem(ref d) => op(d).map(TraitOfItem),
            IsExportedSymbol(ref d) => op(d).map(IsExportedSymbol),
            ItemBodyNestedBodies(ref d) => op(d).map(ItemBodyNestedBodies),
            ConstIsRvaluePromotableToStatic(ref d) => op(d).map(ConstIsRvaluePromotableToStatic),
            IsMirAvailable(ref d) => op(d).map(IsMirAvailable),
            GlobalMetaData(ref d, kind) => op(d).map(|d| GlobalMetaData(d, kind)),
            FileMap(ref d, ref file_name) => op(d).map(|d| FileMap(d, file_name.clone())),
        }
    }
}

/// A "work product" corresponds to a `.o` (or other) file that we
/// save in between runs. These ids do not have a DefId but rather
/// some independent path or string that persists between runs without
/// the need to be mapped or unmapped. (This ensures we can serialize
/// them even in the absence of a tcx.)
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable)]
pub struct WorkProductId(pub String);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable)]
pub enum GlobalMetaDataKind {
    Krate,
    CrateDeps,
    DylibDependencyFormats,
    LangItems,
    LangItemsMissing,
    NativeLibraries,
    CodeMap,
    Impls,
    ExportedSymbols,
}
