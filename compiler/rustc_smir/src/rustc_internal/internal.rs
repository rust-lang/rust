//! Module containing the translation from stable mir constructs to the rustc counterpart.
//!
//! This module will only include a few constructs to allow users to invoke internal rustc APIs
//! due to incomplete stable coverage.

// Prefer importing stable_mir over internal rustc constructs to make this file more readable.
use crate::rustc_smir::Tables;
use rustc_middle::ty::{self as rustc_ty, Ty as InternalTy};
use rustc_span::Symbol;
use stable_mir::mir::mono::{Instance, MonoItem, StaticDef};
use stable_mir::ty::{
    Binder, BoundRegionKind, BoundTyKind, BoundVariableKind, ClosureKind, Const, GenericArgKind,
    GenericArgs, Region, TraitRef, Ty,
};
use stable_mir::{AllocId, CrateItem, DefId};

use super::RustcInternal;

impl<'tcx> RustcInternal<'tcx> for CrateItem {
    type T = rustc_span::def_id::DefId;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        self.0.internal(tables)
    }
}

impl<'tcx> RustcInternal<'tcx> for DefId {
    type T = rustc_span::def_id::DefId;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.def_ids[*self]
    }
}

impl<'tcx> RustcInternal<'tcx> for GenericArgs {
    type T = rustc_ty::GenericArgsRef<'tcx>;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.tcx.mk_args_from_iter(self.0.iter().map(|arg| arg.internal(tables)))
    }
}

impl<'tcx> RustcInternal<'tcx> for GenericArgKind {
    type T = rustc_ty::GenericArg<'tcx>;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            GenericArgKind::Lifetime(reg) => reg.internal(tables).into(),
            GenericArgKind::Type(ty) => ty.internal(tables).into(),
            GenericArgKind::Const(cnst) => ty_const(cnst, tables).into(),
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for Region {
    type T = rustc_ty::Region<'tcx>;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        // Cannot recover region. Use erased instead.
        tables.tcx.lifetimes.re_erased
    }
}

impl<'tcx> RustcInternal<'tcx> for Ty {
    type T = InternalTy<'tcx>;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.types[*self]
    }
}

fn ty_const<'tcx>(constant: &Const, tables: &mut Tables<'tcx>) -> rustc_ty::Const<'tcx> {
    match constant.internal(tables) {
        rustc_middle::mir::Const::Ty(c) => c,
        cnst => {
            panic!("Trying to convert constant `{constant:?}` to type constant, but found {cnst:?}")
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for Const {
    type T = rustc_middle::mir::Const<'tcx>;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.constants[self.id]
    }
}

impl<'tcx> RustcInternal<'tcx> for MonoItem {
    type T = rustc_middle::mir::mono::MonoItem<'tcx>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use rustc_middle::mir::mono as rustc_mono;
        match self {
            MonoItem::Fn(instance) => rustc_mono::MonoItem::Fn(instance.internal(tables)),
            MonoItem::Static(def) => rustc_mono::MonoItem::Static(def.internal(tables)),
            MonoItem::GlobalAsm(_) => {
                unimplemented!()
            }
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for Instance {
    type T = rustc_ty::Instance<'tcx>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.instances[self.def]
    }
}

impl<'tcx> RustcInternal<'tcx> for StaticDef {
    type T = rustc_span::def_id::DefId;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        self.0.internal(tables)
    }
}

#[allow(rustc::usage_of_qualified_ty)]
impl<'tcx, T> RustcInternal<'tcx> for Binder<T>
where
    T: RustcInternal<'tcx>,
    T::T: rustc_ty::TypeVisitable<rustc_ty::TyCtxt<'tcx>>,
{
    type T = rustc_ty::Binder<'tcx, T::T>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        rustc_ty::Binder::bind_with_vars(
            self.value.internal(tables),
            tables.tcx.mk_bound_variable_kinds_from_iter(
                self.bound_vars.iter().map(|bound| bound.internal(tables)),
            ),
        )
    }
}

impl<'tcx> RustcInternal<'tcx> for BoundVariableKind {
    type T = rustc_ty::BoundVariableKind;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            BoundVariableKind::Ty(kind) => rustc_ty::BoundVariableKind::Ty(match kind {
                BoundTyKind::Anon => rustc_ty::BoundTyKind::Anon,
                BoundTyKind::Param(def, symbol) => {
                    rustc_ty::BoundTyKind::Param(def.0.internal(tables), Symbol::intern(&symbol))
                }
            }),
            BoundVariableKind::Region(kind) => rustc_ty::BoundVariableKind::Region(match kind {
                BoundRegionKind::BrAnon => rustc_ty::BoundRegionKind::BrAnon,
                BoundRegionKind::BrNamed(def, symbol) => rustc_ty::BoundRegionKind::BrNamed(
                    def.0.internal(tables),
                    Symbol::intern(&symbol),
                ),
                BoundRegionKind::BrEnv => rustc_ty::BoundRegionKind::BrEnv,
            }),
            BoundVariableKind::Const => rustc_ty::BoundVariableKind::Const,
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for TraitRef {
    type T = rustc_ty::TraitRef<'tcx>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        rustc_ty::TraitRef::new(
            tables.tcx,
            self.def_id.0.internal(tables),
            self.args().internal(tables),
        )
    }
}

impl<'tcx> RustcInternal<'tcx> for AllocId {
    type T = rustc_middle::mir::interpret::AllocId;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.alloc_ids[*self]
    }
}

impl<'tcx> RustcInternal<'tcx> for ClosureKind {
    type T = rustc_ty::ClosureKind;

    fn internal(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            ClosureKind::Fn => rustc_ty::ClosureKind::Fn,
            ClosureKind::FnMut => rustc_ty::ClosureKind::FnMut,
            ClosureKind::FnOnce => rustc_ty::ClosureKind::FnOnce,
        }
    }
}

impl<'tcx, T> RustcInternal<'tcx> for &T
where
    T: RustcInternal<'tcx>,
{
    type T = T::T;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        (*self).internal(tables)
    }
}
