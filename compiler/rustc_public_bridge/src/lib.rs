//! Crate that implements what will become the rustc side of rustc_public.
//!
//! This crate serves as a proxy for making calls to rustc queries.
//!
//! This crate is not intended to be invoked directly by users.
//! This crate is the public API of rustc that will be invoked by the `rustc_public` crate.
//!
//! For more information see <https://github.com/rust-lang/project-stable-mir>
//!
//! # Note
//!
//! This API is still completely unstable and subject to change.

// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::usage_of_ty_tykind)]
#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
#![doc(rust_logo)]
#![feature(rustdoc_internals)]
#![feature(sized_hierarchy)]
// tidy-alphabetical-end

use std::cell::RefCell;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Index;

use bridge::*;
use context::CompilerCtxt;
use rustc_data_structures::fx::{self, FxIndexMap};
use rustc_middle::mir;
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;
use rustc_span::def_id::{CrateNum, DefId, LOCAL_CRATE};

pub mod alloc;
pub mod bridge;
mod builder;
pub mod context;

#[deprecated(note = "please use `rustc_public::rustc_internal` instead")]
pub mod rustc_internal {}

/// A container which is used for TLS.
pub struct Container<'tcx, B: Bridge> {
    pub tables: RefCell<Tables<'tcx, B>>,
    pub cx: RefCell<CompilerCtxt<'tcx, B>>,
}

pub struct Tables<'tcx, B: Bridge> {
    pub def_ids: IndexMap<DefId, B::DefId>,
    pub alloc_ids: IndexMap<AllocId, B::AllocId>,
    pub spans: IndexMap<rustc_span::Span, B::Span>,
    pub types: IndexMap<Ty<'tcx>, B::Ty>,
    pub instances: IndexMap<ty::Instance<'tcx>, B::InstanceDef>,
    pub ty_consts: IndexMap<ty::Const<'tcx>, B::TyConstId>,
    pub mir_consts: IndexMap<mir::Const<'tcx>, B::MirConstId>,
    pub layouts: IndexMap<rustc_abi::Layout<'tcx>, B::Layout>,
}

impl<'tcx, B: Bridge> Default for Tables<'tcx, B> {
    fn default() -> Self {
        Self {
            def_ids: IndexMap::default(),
            alloc_ids: IndexMap::default(),
            spans: IndexMap::default(),
            types: IndexMap::default(),
            instances: IndexMap::default(),
            ty_consts: IndexMap::default(),
            mir_consts: IndexMap::default(),
            layouts: IndexMap::default(),
        }
    }
}

impl<'tcx, B: Bridge> Index<B::DefId> for Tables<'tcx, B> {
    type Output = DefId;

    #[inline(always)]
    fn index(&self, index: B::DefId) -> &Self::Output {
        &self.def_ids[index]
    }
}

impl<'tcx, B: Bridge> Tables<'tcx, B> {
    pub fn intern_ty(&mut self, ty: Ty<'tcx>) -> B::Ty {
        self.types.create_or_fetch(ty)
    }

    pub fn intern_ty_const(&mut self, ct: ty::Const<'tcx>) -> B::TyConstId {
        self.ty_consts.create_or_fetch(ct)
    }

    pub fn intern_mir_const(&mut self, constant: mir::Const<'tcx>) -> B::MirConstId {
        self.mir_consts.create_or_fetch(constant)
    }

    pub fn create_def_id(&mut self, did: DefId) -> B::DefId {
        self.def_ids.create_or_fetch(did)
    }

    pub fn create_alloc_id(&mut self, aid: AllocId) -> B::AllocId {
        self.alloc_ids.create_or_fetch(aid)
    }

    pub fn create_span(&mut self, span: Span) -> B::Span {
        self.spans.create_or_fetch(span)
    }

    pub fn instance_def(&mut self, instance: ty::Instance<'tcx>) -> B::InstanceDef {
        self.instances.create_or_fetch(instance)
    }

    pub fn layout_id(&mut self, layout: rustc_abi::Layout<'tcx>) -> B::Layout {
        self.layouts.create_or_fetch(layout)
    }

    pub fn crate_item(&mut self, did: rustc_span::def_id::DefId) -> B::CrateItem {
        B::CrateItem::new(self.create_def_id(did))
    }

    pub fn adt_def(&mut self, did: rustc_span::def_id::DefId) -> B::AdtDef {
        B::AdtDef::new(self.create_def_id(did))
    }

    pub fn foreign_module_def(&mut self, did: rustc_span::def_id::DefId) -> B::ForeignModuleDef {
        B::ForeignModuleDef::new(self.create_def_id(did))
    }

    pub fn foreign_def(&mut self, did: rustc_span::def_id::DefId) -> B::ForeignDef {
        B::ForeignDef::new(self.create_def_id(did))
    }

    pub fn fn_def(&mut self, did: rustc_span::def_id::DefId) -> B::FnDef {
        B::FnDef::new(self.create_def_id(did))
    }

    pub fn closure_def(&mut self, did: rustc_span::def_id::DefId) -> B::ClosureDef {
        B::ClosureDef::new(self.create_def_id(did))
    }

    pub fn coroutine_def(&mut self, did: rustc_span::def_id::DefId) -> B::CoroutineDef {
        B::CoroutineDef::new(self.create_def_id(did))
    }

    pub fn coroutine_closure_def(
        &mut self,
        did: rustc_span::def_id::DefId,
    ) -> B::CoroutineClosureDef {
        B::CoroutineClosureDef::new(self.create_def_id(did))
    }

    pub fn alias_def(&mut self, did: rustc_span::def_id::DefId) -> B::AliasDef {
        B::AliasDef::new(self.create_def_id(did))
    }

    pub fn param_def(&mut self, did: rustc_span::def_id::DefId) -> B::ParamDef {
        B::ParamDef::new(self.create_def_id(did))
    }

    pub fn br_named_def(&mut self, did: rustc_span::def_id::DefId) -> B::BrNamedDef {
        B::BrNamedDef::new(self.create_def_id(did))
    }

    pub fn trait_def(&mut self, did: rustc_span::def_id::DefId) -> B::TraitDef {
        B::TraitDef::new(self.create_def_id(did))
    }

    pub fn generic_def(&mut self, did: rustc_span::def_id::DefId) -> B::GenericDef {
        B::GenericDef::new(self.create_def_id(did))
    }

    pub fn const_def(&mut self, did: rustc_span::def_id::DefId) -> B::ConstDef {
        B::ConstDef::new(self.create_def_id(did))
    }

    pub fn impl_def(&mut self, did: rustc_span::def_id::DefId) -> B::ImplDef {
        B::ImplDef::new(self.create_def_id(did))
    }

    pub fn region_def(&mut self, did: rustc_span::def_id::DefId) -> B::RegionDef {
        B::RegionDef::new(self.create_def_id(did))
    }

    pub fn coroutine_witness_def(
        &mut self,
        did: rustc_span::def_id::DefId,
    ) -> B::CoroutineWitnessDef {
        B::CoroutineWitnessDef::new(self.create_def_id(did))
    }

    pub fn assoc_def(&mut self, did: rustc_span::def_id::DefId) -> B::AssocDef {
        B::AssocDef::new(self.create_def_id(did))
    }

    pub fn opaque_def(&mut self, did: rustc_span::def_id::DefId) -> B::OpaqueDef {
        B::OpaqueDef::new(self.create_def_id(did))
    }

    pub fn prov(&mut self, aid: rustc_middle::mir::interpret::AllocId) -> B::Prov {
        B::Prov::new(self.create_alloc_id(aid))
    }

    pub fn static_def(&mut self, did: rustc_span::def_id::DefId) -> B::StaticDef {
        B::StaticDef::new(self.create_def_id(did))
    }
}

/// A trait defining types that are used to emulate rustc_public components, which is really
/// useful when programming in rustc_public-agnostic settings.
pub trait Bridge: Sized {
    type DefId: Copy + Debug + PartialEq + IndexedVal;
    type AllocId: Copy + Debug + PartialEq + IndexedVal;
    type Span: Copy + Debug + PartialEq + IndexedVal;
    type Ty: Copy + Debug + PartialEq + IndexedVal;
    type InstanceDef: Copy + Debug + PartialEq + IndexedVal;
    type TyConstId: Copy + Debug + PartialEq + IndexedVal;
    type MirConstId: Copy + Debug + PartialEq + IndexedVal;
    type Layout: Copy + Debug + PartialEq + IndexedVal;

    type Error: Error;
    type CrateItem: CrateItem<Self>;
    type AdtDef: AdtDef<Self>;
    type ForeignModuleDef: ForeignModuleDef<Self>;
    type ForeignDef: ForeignDef<Self>;
    type FnDef: FnDef<Self>;
    type ClosureDef: ClosureDef<Self>;
    type CoroutineDef: CoroutineDef<Self>;
    type CoroutineClosureDef: CoroutineClosureDef<Self>;
    type AliasDef: AliasDef<Self>;
    type ParamDef: ParamDef<Self>;
    type BrNamedDef: BrNamedDef<Self>;
    type TraitDef: TraitDef<Self>;
    type GenericDef: GenericDef<Self>;
    type ConstDef: ConstDef<Self>;
    type ImplDef: ImplDef<Self>;
    type RegionDef: RegionDef<Self>;
    type CoroutineWitnessDef: CoroutineWitnessDef<Self>;
    type AssocDef: AssocDef<Self>;
    type OpaqueDef: OpaqueDef<Self>;
    type Prov: Prov<Self>;
    type StaticDef: StaticDef<Self>;

    type Allocation: Allocation<Self>;
}

pub trait IndexedVal {
    fn to_val(index: usize) -> Self;

    fn to_index(&self) -> usize;
}

/// Similar to rustc's `FxIndexMap`, `IndexMap` with extra
/// safety features added.
pub struct IndexMap<K, V> {
    index_map: fx::FxIndexMap<K, V>,
}

impl<K, V> Default for IndexMap<K, V> {
    fn default() -> Self {
        Self { index_map: FxIndexMap::default() }
    }
}

impl<K: PartialEq + Hash + Eq, V: Copy + Debug + PartialEq + IndexedVal> IndexMap<K, V> {
    pub fn create_or_fetch(&mut self, key: K) -> V {
        let len = self.index_map.len();
        let v = self.index_map.entry(key).or_insert(V::to_val(len));
        *v
    }
}

impl<K: PartialEq + Hash + Eq, V: Copy + Debug + PartialEq + IndexedVal> Index<V>
    for IndexMap<K, V>
{
    type Output = K;

    fn index(&self, index: V) -> &Self::Output {
        let (k, v) = self.index_map.get_index(index.to_index()).unwrap();
        assert_eq!(*v, index, "Provided value doesn't match with indexed value");
        k
    }
}

/// Iterate over the definitions of the given crate.
pub(crate) fn filter_def_ids<F, T>(tcx: TyCtxt<'_>, krate: CrateNum, mut func: F) -> Vec<T>
where
    F: FnMut(DefId) -> Option<T>,
{
    if krate == LOCAL_CRATE {
        tcx.iter_local_def_id().filter_map(|did| func(did.to_def_id())).collect()
    } else {
        let num_definitions = tcx.num_extern_def_ids(krate);
        (0..num_definitions)
            .filter_map(move |i| {
                let def_id = DefId { krate, index: rustc_span::def_id::DefIndex::from_usize(i) };
                func(def_id)
            })
            .collect()
    }
}
