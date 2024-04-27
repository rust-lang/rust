//! Module that implements what will become the rustc side of Stable MIR.

//! This module is responsible for building Stable MIR components from internal components.
//!
//! This module is not intended to be invoked directly by users. It will eventually
//! become the public API of rustc that will be invoked by the `stable_mir` crate.
//!
//! For now, we are developing everything inside `rustc`, thus, we keep this module private.

use rustc_hir::def::DefKind;
use rustc_middle::mir;
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_span::def_id::{CrateNum, DefId, LOCAL_CRATE};
use stable_mir::abi::Layout;
use stable_mir::mir::mono::InstanceDef;
use stable_mir::ty::{ConstId, Span};
use stable_mir::{CtorKind, ItemKind};
use std::ops::RangeInclusive;
use tracing::debug;

use crate::rustc_internal::IndexMap;

mod alloc;
mod builder;
pub(crate) mod context;
mod convert;

pub struct Tables<'tcx> {
    pub(crate) tcx: TyCtxt<'tcx>,
    pub(crate) def_ids: IndexMap<DefId, stable_mir::DefId>,
    pub(crate) alloc_ids: IndexMap<AllocId, stable_mir::mir::alloc::AllocId>,
    pub(crate) spans: IndexMap<rustc_span::Span, Span>,
    pub(crate) types: IndexMap<Ty<'tcx>, stable_mir::ty::Ty>,
    pub(crate) instances: IndexMap<ty::Instance<'tcx>, InstanceDef>,
    pub(crate) constants: IndexMap<mir::Const<'tcx>, ConstId>,
    pub(crate) layouts: IndexMap<rustc_target::abi::Layout<'tcx>, Layout>,
}

impl<'tcx> Tables<'tcx> {
    pub(crate) fn intern_ty(&mut self, ty: Ty<'tcx>) -> stable_mir::ty::Ty {
        self.types.create_or_fetch(ty)
    }

    pub(crate) fn intern_const(&mut self, constant: mir::Const<'tcx>) -> ConstId {
        self.constants.create_or_fetch(constant)
    }

    pub(crate) fn has_body(&self, instance: Instance<'tcx>) -> bool {
        let def_id = instance.def_id();
        self.tcx.is_mir_available(def_id)
            || !matches!(
                instance.def,
                ty::InstanceDef::Virtual(..)
                    | ty::InstanceDef::Intrinsic(..)
                    | ty::InstanceDef::Item(..)
            )
    }
}

/// Build a stable mir crate from a given crate number.
pub(crate) fn smir_crate(tcx: TyCtxt<'_>, crate_num: CrateNum) -> stable_mir::Crate {
    let crate_name = tcx.crate_name(crate_num).to_string();
    let is_local = crate_num == LOCAL_CRATE;
    debug!(?crate_name, ?crate_num, "smir_crate");
    stable_mir::Crate { id: crate_num.into(), name: crate_name, is_local }
}

pub(crate) fn new_item_kind(kind: DefKind) -> ItemKind {
    match kind {
        DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::AssocTy
        | DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::Macro(_)
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::ForeignMod
        | DefKind::OpaqueTy
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::Impl { .. }
        | DefKind::GlobalAsm => {
            unreachable!("Not a valid item kind: {kind:?}");
        }
        DefKind::Closure | DefKind::AssocFn | DefKind::Fn => ItemKind::Fn,
        DefKind::Const | DefKind::InlineConst | DefKind::AssocConst | DefKind::AnonConst => {
            ItemKind::Const
        }
        DefKind::Static { .. } => ItemKind::Static,
        DefKind::Ctor(_, rustc_hir::def::CtorKind::Const) => ItemKind::Ctor(CtorKind::Const),
        DefKind::Ctor(_, rustc_hir::def::CtorKind::Fn) => ItemKind::Ctor(CtorKind::Fn),
    }
}

/// Trait used to convert between an internal MIR type to a Stable MIR type.
pub trait Stable<'cx> {
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
