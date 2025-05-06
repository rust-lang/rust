//! Module that implements what will become the rustc side of Stable MIR.

//! This module is responsible for building Stable MIR components from internal components.
//!
//! This module is not intended to be invoked directly by users. It will eventually
//! become the public API of rustc that will be invoked by the `stable_mir` crate.
//!
//! For now, we are developing everything inside `rustc`, thus, we keep this module private.

use std::cell::RefCell;
use std::fmt::Debug;

use context::SmirCtxt;
use rustc_hir::def::DefKind;
use rustc_middle::mir;
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::{CrateNum, DefId, LOCAL_CRATE};

use crate::rustc_internal::IndexMap;

pub mod alloc;
mod builder;
pub mod context;

/// A container which is used for TLS.
pub struct SmirContainer<'tcx, B: Bridge> {
    pub tables: RefCell<Tables<'tcx, B>>,
    pub cx: RefCell<SmirCtxt<'tcx, B>>,
}

pub struct Tables<'tcx, B: Bridge> {
    pub(crate) def_ids: IndexMap<DefId, B::DefId>,
    pub(crate) alloc_ids: IndexMap<AllocId, B::AllocId>,
    pub(crate) spans: IndexMap<rustc_span::Span, B::Span>,
    pub(crate) types: IndexMap<Ty<'tcx>, B::Ty>,
    pub(crate) instances: IndexMap<ty::Instance<'tcx>, B::InstanceDef>,
    pub(crate) ty_consts: IndexMap<ty::Const<'tcx>, B::TyConstId>,
    pub(crate) mir_consts: IndexMap<mir::Const<'tcx>, B::MirConstId>,
    pub(crate) layouts: IndexMap<rustc_abi::Layout<'tcx>, B::Layout>,
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

impl<'tcx, B: Bridge> Tables<'tcx, B> {
    pub(crate) fn intern_ty(&mut self, ty: Ty<'tcx>) -> B::Ty {
        self.types.create_or_fetch(ty)
    }

    pub(crate) fn intern_ty_const(&mut self, ct: ty::Const<'tcx>) -> B::TyConstId {
        self.ty_consts.create_or_fetch(ct)
    }

    pub(crate) fn intern_mir_const(&mut self, constant: mir::Const<'tcx>) -> B::MirConstId {
        self.mir_consts.create_or_fetch(constant)
    }
}

/// A trait defining types that are used to emulate StableMIR components, which is really
/// useful when programming in stable_mir-agnostic settings.
pub trait Bridge {
    type DefId: Copy + Debug + PartialEq + IndexedVal;
    type AllocId: Copy + Debug + PartialEq + IndexedVal;
    type Span: Copy + Debug + PartialEq + IndexedVal;
    type Ty: Copy + Debug + PartialEq + IndexedVal;
    type InstanceDef: Copy + Debug + PartialEq + IndexedVal;
    type TyConstId: Copy + Debug + PartialEq + IndexedVal;
    type MirConstId: Copy + Debug + PartialEq + IndexedVal;
    type Layout: Copy + Debug + PartialEq + IndexedVal;
    type Error: SmirError;
}

pub trait SmirError {
    fn new(msg: String) -> Self;
    fn from_internal<T: Debug>(err: T) -> Self;
}

pub trait IndexedVal {
    fn to_val(index: usize) -> Self;

    fn to_index(&self) -> usize;
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
