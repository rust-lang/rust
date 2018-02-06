// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def_id::DefId;
use rustc::middle::lang_items::DropInPlaceFnLangItem;
use rustc::traits;
use rustc::ty::adjustment::CustomCoerceUnsized;
use rustc::ty::subst::Kind;
use rustc::ty::{self, Ty, TyCtxt};

pub use rustc::ty::Instance;
pub use self::item::{MonoItem, MonoItemExt};

pub mod collector;
pub mod item;
pub mod partitioning;

fn fn_once_adapter_instance<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    closure_did: DefId,
    substs: ty::ClosureSubsts<'tcx>,
    ) -> Instance<'tcx> {
    debug!("fn_once_adapter_shim({:?}, {:?})",
           closure_did,
           substs);
    let fn_once = tcx.lang_items().fn_once_trait().unwrap();
    let call_once = tcx.associated_items(fn_once)
        .find(|it| it.kind == ty::AssociatedKind::Method)
        .unwrap().def_id;
    let def = ty::InstanceDef::ClosureOnceShim { call_once };

    let self_ty = tcx.mk_closure_from_closure_substs(
        closure_did, substs);

    let sig = substs.closure_sig(closure_did, tcx);
    let sig = tcx.erase_late_bound_regions_and_normalize(&sig);
    assert_eq!(sig.inputs().len(), 1);
    let substs = tcx.mk_substs([
        Kind::from(self_ty),
        Kind::from(sig.inputs()[0]),
    ].iter().cloned());

    debug!("fn_once_adapter_shim: self_ty={:?} sig={:?}", self_ty, sig);
    Instance { def, substs }
}

fn needs_fn_once_adapter_shim(actual_closure_kind: ty::ClosureKind,
                              trait_closure_kind: ty::ClosureKind)
                              -> Result<bool, ()>
{
    match (actual_closure_kind, trait_closure_kind) {
        (ty::ClosureKind::Fn, ty::ClosureKind::Fn) |
        (ty::ClosureKind::FnMut, ty::ClosureKind::FnMut) |
        (ty::ClosureKind::FnOnce, ty::ClosureKind::FnOnce) => {
            // No adapter needed.
           Ok(false)
        }
        (ty::ClosureKind::Fn, ty::ClosureKind::FnMut) => {
            // The closure fn `llfn` is a `fn(&self, ...)`.  We want a
            // `fn(&mut self, ...)`. In fact, at trans time, these are
            // basically the same thing, so we can just return llfn.
            Ok(false)
        }
        (ty::ClosureKind::Fn, ty::ClosureKind::FnOnce) |
        (ty::ClosureKind::FnMut, ty::ClosureKind::FnOnce) => {
            // The closure fn `llfn` is a `fn(&self, ...)` or `fn(&mut
            // self, ...)`.  We want a `fn(self, ...)`. We can produce
            // this by doing something like:
            //
            //     fn call_once(self, ...) { call_mut(&self, ...) }
            //     fn call_once(mut self, ...) { call_mut(&mut self, ...) }
            //
            // These are both the same at trans time.
            Ok(true)
        }
        _ => Err(()),
    }
}

pub fn resolve_closure<'a, 'tcx> (
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    def_id: DefId,
    substs: ty::ClosureSubsts<'tcx>,
    requested_kind: ty::ClosureKind)
    -> Instance<'tcx>
{
    let actual_kind = substs.closure_kind(def_id, tcx);

    match needs_fn_once_adapter_shim(actual_kind, requested_kind) {
        Ok(true) => fn_once_adapter_instance(tcx, def_id, substs),
        _ => Instance::new(def_id, substs.substs)
    }
}

pub fn resolve_drop_in_place<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    ty: Ty<'tcx>)
    -> ty::Instance<'tcx>
{
    let def_id = tcx.require_lang_item(DropInPlaceFnLangItem);
    let substs = tcx.intern_substs(&[Kind::from(ty)]);
    Instance::resolve(tcx, ty::ParamEnv::empty(traits::Reveal::All), def_id, substs).unwrap()
}

pub fn custom_coerce_unsize_info<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                           source_ty: Ty<'tcx>,
                                           target_ty: Ty<'tcx>)
                                           -> CustomCoerceUnsized {
    let def_id = tcx.lang_items().coerce_unsized_trait().unwrap();

    let trait_ref = ty::Binder(ty::TraitRef {
        def_id: def_id,
        substs: tcx.mk_substs_trait(source_ty, &[target_ty])
    });

    match tcx.trans_fulfill_obligation( (ty::ParamEnv::empty(traits::Reveal::All), trait_ref)) {
        traits::VtableImpl(traits::VtableImplData { impl_def_id, .. }) => {
            tcx.coerce_unsized_info(impl_def_id).custom_kind.unwrap()
        }
        vtable => {
            bug!("invalid CoerceUnsized vtable: {:?}", vtable);
        }
    }
}
