//! Module that implements what will become the rustc side of Stable MIR.

//! This module is responsible for building Stable MIR components from internal components.
//!
//! This module is not intended to be invoked directly by users. It will eventually
//! become the public API of rustc that will be invoked by the `stable_mir` crate.
//!
//! For now, we are developing everything inside `rustc`, thus, we keep this module private.

use std::marker::PointeeSized;
use std::cell::RefCell;
use std::fmt::Debug;

use context::SmirCtxt;
use rustc_hir::def::DefKind;
use rustc_middle::mir;
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_span::def_id::{CrateNum, DefId, LOCAL_CRATE};

use crate::rustc_internal::IndexMap;
use crate::stable_mir;

mod alloc;
mod builder;
pub mod context;
mod convert;

/// A container which is used for TLS.
pub struct SmirContainer<'tcx, B: Bridge> {
    pub tables: RefCell<Tables<'tcx, B>>,
    pub cx: RefCell<SmirCtxt<'tcx, B>>,
}

pub struct Tables<'tcx, B: Bridge> {
    tcx: TyCtxt<'tcx>,
    pub(crate) def_ids: IndexMap<DefId, B::DefId>,
    pub(crate) alloc_ids: IndexMap<AllocId, B::AllocId>,
    pub(crate) spans: IndexMap<rustc_span::Span, B::Span>,
    pub(crate) types: IndexMap<Ty<'tcx>, B::Ty>,
    pub(crate) instances: IndexMap<ty::Instance<'tcx>, B::InstanceDef>,
    pub(crate) ty_consts: IndexMap<ty::Const<'tcx>, B::TyConstId>,
    pub(crate) mir_consts: IndexMap<mir::Const<'tcx>, B::MirConstId>,
    pub(crate) layouts: IndexMap<rustc_abi::Layout<'tcx>, B::Layout>,
}

impl<'tcx, B: Bridge> Tables<'tcx, B> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
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

    /// Return whether the instance as a body available.
    ///
    /// Items and intrinsics may have a body available from its definition.
    /// Shims body may be generated depending on their type.
    pub(crate) fn instance_has_body(&self, instance: Instance<'tcx>) -> bool {
        let def_id = instance.def_id();
        self.item_has_body(def_id)
            || !matches!(
                instance.def,
                ty::InstanceKind::Virtual(..)
                    | ty::InstanceKind::Intrinsic(..)
                    | ty::InstanceKind::Item(..)
            )
    }

    /// Return whether the item has a body defined by the user.
    ///
    /// Note that intrinsics may have a placeholder body that shouldn't be used in practice.
    /// In StableMIR, we handle this case as if the body is not available.
    pub(crate) fn item_has_body(&self, def_id: DefId) -> bool {
        let must_override = if let Some(intrinsic) = self.tcx.intrinsic(def_id) {
            intrinsic.must_be_overridden
        } else {
            false
        };
        !must_override && self.tcx.is_mir_available(def_id)
    }

    fn filter_fn_def(&mut self, def_id: DefId) -> Option<DefId> {
        if matches!(self.tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn) {
            Some(def_id)
        } else {
            None
        }
    }

    fn filter_static_def(&mut self, def_id: DefId) -> Option<DefId> {
        matches!(self.tcx.def_kind(def_id), DefKind::Static { .. }).then(|| def_id)
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

/// Trait used to convert between an internal MIR type to a Stable MIR type.
pub trait Stable<'cx>: PointeeSized {
    /// The stable representation of the type implementing Stable.
    type T;
    /// Converts an object to the equivalent Stable MIR representation.
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T;
}

impl<'tcx, T> Stable<'tcx> for &T
where
    T: Stable<'tcx>,
{
    type T = T::T;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        (*self).stable(tables)
    }
}

impl<'tcx, T> Stable<'tcx> for Option<T>
where
    T: Stable<'tcx>,
{
    type T = Option<T::T>;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        self.as_ref().map(|value| value.stable(tables))
    }
}

impl<'tcx, T, E> Stable<'tcx> for Result<T, E>
where
    T: Stable<'tcx>,
    E: Stable<'tcx>,
{
    type T = Result<T::T, E::T>;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        match self {
            Ok(val) => Ok(val.stable(tables)),
            Err(error) => Err(error.stable(tables)),
        }
    }
}

impl<'tcx, T> Stable<'tcx> for &[T]
where
    T: Stable<'tcx>,
{
    type T = Vec<T::T>;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        self.iter().map(|e| e.stable(tables)).collect()
    }
}

impl<'tcx, T, U> Stable<'tcx> for (T, U)
where
    T: Stable<'tcx>,
    U: Stable<'tcx>,
{
    type T = (T::T, U::T);
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        (self.0.stable(tables), self.1.stable(tables))
    }
}

impl<'tcx, T> Stable<'tcx> for RangeInclusive<T>
where
    T: Stable<'tcx>,
{
    type T = RangeInclusive<T::T>;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        RangeInclusive::new(self.start().stable(tables), self.end().stable(tables))
    }
}
