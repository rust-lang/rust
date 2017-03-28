// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::Abi;
use common::*;
use glue;

use rustc::hir::def_id::DefId;
use rustc::infer::TransNormalize;
use rustc::middle::lang_items::DropInPlaceFnLangItem;
use rustc::traits::{self, SelectionContext, Reveal};
use rustc::ty::adjustment::CustomCoerceUnsized;
use rustc::ty::fold::{TypeFolder, TypeFoldable};
use rustc::ty::subst::{Kind, Subst, Substs};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::util::common::MemoizationMap;

use syntax::ast;
use syntax::codemap::{Span, DUMMY_SP};

pub use rustc::ty::Instance;

fn fn_once_adapter_instance<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    closure_did: DefId,
    substs: ty::ClosureSubsts<'tcx>,
    ) -> Instance<'tcx> {
    debug!("fn_once_adapter_shim({:?}, {:?})",
           closure_did,
           substs);
    let fn_once = tcx.lang_items.fn_once_trait().unwrap();
    let call_once = tcx.associated_items(fn_once)
        .find(|it| it.kind == ty::AssociatedKind::Method)
        .unwrap().def_id;
    let def = ty::InstanceDef::ClosureOnceShim { call_once };

    let self_ty = tcx.mk_closure_from_closure_substs(
        closure_did, substs);

    let sig = tcx.closure_type(closure_did).subst(tcx, substs.substs);
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
    scx: &SharedCrateContext<'a, 'tcx>,
    def_id: DefId,
    substs: ty::ClosureSubsts<'tcx>,
    requested_kind: ty::ClosureKind)
    -> Instance<'tcx>
{
    let actual_kind = scx.tcx().closure_kind(def_id);

    match needs_fn_once_adapter_shim(actual_kind, requested_kind) {
        Ok(true) => fn_once_adapter_instance(scx.tcx(), def_id, substs),
        _ => Instance::new(def_id, substs.substs)
    }
}

/// Attempts to resolve an obligation. The result is a shallow vtable resolution -- meaning that we
/// do not (necessarily) resolve all nested obligations on the impl. Note that type check should
/// guarantee to us that all nested obligations *could be* resolved if we wanted to.
fn fulfill_obligation<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                span: Span,
                                trait_ref: ty::PolyTraitRef<'tcx>)
                                -> traits::Vtable<'tcx, ()>
{
    let tcx = scx.tcx();

    // Remove any references to regions; this helps improve caching.
    let trait_ref = tcx.erase_regions(&trait_ref);

    scx.trait_cache().memoize(trait_ref, || {
        debug!("trans::fulfill_obligation(trait_ref={:?}, def_id={:?})",
               trait_ref, trait_ref.def_id());

        // Do the initial selection for the obligation. This yields the
        // shallow result we are looking for -- that is, what specific impl.
        tcx.infer_ctxt((), Reveal::All).enter(|infcx| {
            let mut selcx = SelectionContext::new(&infcx);

            let obligation_cause = traits::ObligationCause::misc(span,
                                                             ast::DUMMY_NODE_ID);
            let obligation = traits::Obligation::new(obligation_cause,
                                                     trait_ref.to_poly_trait_predicate());

            let selection = match selcx.select(&obligation) {
                Ok(Some(selection)) => selection,
                Ok(None) => {
                    // Ambiguity can happen when monomorphizing during trans
                    // expands to some humongo type that never occurred
                    // statically -- this humongo type can then overflow,
                    // leading to an ambiguous result. So report this as an
                    // overflow bug, since I believe this is the only case
                    // where ambiguity can result.
                    debug!("Encountered ambiguity selecting `{:?}` during trans, \
                            presuming due to overflow",
                           trait_ref);
                    tcx.sess.span_fatal(span,
                        "reached the recursion limit during monomorphization \
                         (selection ambiguity)");
                }
                Err(e) => {
                    span_bug!(span, "Encountered error `{:?}` selecting `{:?}` during trans",
                              e, trait_ref)
                }
            };

            debug!("fulfill_obligation: selection={:?}", selection);

            // Currently, we use a fulfillment context to completely resolve
            // all nested obligations. This is because they can inform the
            // inference of the impl's type parameters.
            let mut fulfill_cx = traits::FulfillmentContext::new();
            let vtable = selection.map(|predicate| {
                debug!("fulfill_obligation: register_predicate_obligation {:?}", predicate);
                fulfill_cx.register_predicate_obligation(&infcx, predicate);
            });
            let vtable = infcx.drain_fulfillment_cx_or_panic(span, &mut fulfill_cx, &vtable);

            info!("Cache miss: {:?} => {:?}", trait_ref, vtable);
            vtable
        })
    })
}

fn resolve_associated_item<'a, 'tcx>(
    scx: &SharedCrateContext<'a, 'tcx>,
    trait_item: &ty::AssociatedItem,
    trait_id: DefId,
    rcvr_substs: &'tcx Substs<'tcx>
) -> Instance<'tcx> {
    let tcx = scx.tcx();
    let def_id = trait_item.def_id;
    debug!("resolve_associated_item(trait_item={:?}, \
                                    trait_id={:?}, \
                                    rcvr_substs={:?})",
           def_id, trait_id, rcvr_substs);

    let trait_ref = ty::TraitRef::from_method(tcx, trait_id, rcvr_substs);
    let vtbl = fulfill_obligation(scx, DUMMY_SP, ty::Binder(trait_ref));

    // Now that we know which impl is being used, we can dispatch to
    // the actual function:
    match vtbl {
        traits::VtableImpl(impl_data) => {
            let (def_id, substs) = traits::find_associated_item(
                tcx, trait_item, rcvr_substs, &impl_data);
            let substs = tcx.erase_regions(&substs);
            ty::Instance::new(def_id, substs)
        }
        traits::VtableClosure(closure_data) => {
            let trait_closure_kind = tcx.lang_items.fn_trait_kind(trait_id).unwrap();
            resolve_closure(scx, closure_data.closure_def_id, closure_data.substs,
                            trait_closure_kind)
        }
        traits::VtableFnPointer(ref data) => {
            Instance {
                def: ty::InstanceDef::FnPtrShim(trait_item.def_id, data.fn_ty),
                substs: rcvr_substs
            }
        }
        traits::VtableObject(ref data) => {
            let index = tcx.get_vtable_index_of_object_method(data, def_id);
            Instance {
                def: ty::InstanceDef::Virtual(def_id, index),
                substs: rcvr_substs
            }
        }
        _ => {
            bug!("static call to invalid vtable: {:?}", vtbl)
        }
    }
}

/// The point where linking happens. Resolve a (def_id, substs)
/// pair to an instance.
pub fn resolve<'a, 'tcx>(
    scx: &SharedCrateContext<'a, 'tcx>,
    def_id: DefId,
    substs: &'tcx Substs<'tcx>
) -> Instance<'tcx> {
    debug!("resolve(def_id={:?}, substs={:?})",
           def_id, substs);
    let result = if let Some(trait_def_id) = scx.tcx().trait_of_item(def_id) {
        debug!(" => associated item, attempting to find impl");
        let item = scx.tcx().associated_item(def_id);
        resolve_associated_item(scx, &item, trait_def_id, substs)
    } else {
        let item_type = def_ty(scx, def_id, substs);
        let def = match item_type.sty {
            ty::TyFnDef(_, _, f) if
                f.abi() == Abi::RustIntrinsic ||
                f.abi() == Abi::PlatformIntrinsic =>
            {
                debug!(" => intrinsic");
                ty::InstanceDef::Intrinsic(def_id)
            }
            _ => {
                if Some(def_id) == scx.tcx().lang_items.drop_in_place_fn() {
                    let ty = substs.type_at(0);
                    if glue::needs_drop_glue(scx, ty) {
                        debug!(" => nontrivial drop glue");
                        ty::InstanceDef::DropGlue(def_id, Some(ty))
                    } else {
                        debug!(" => trivial drop glue");
                        ty::InstanceDef::DropGlue(def_id, None)
                    }
                } else {
                    debug!(" => free item");
                    ty::InstanceDef::Item(def_id)
                }
            }
        };
        Instance { def, substs }
    };
    debug!("resolve(def_id={:?}, substs={:?}) = {}",
           def_id, substs, result);
    result
}

pub fn resolve_drop_in_place<'a, 'tcx>(
    scx: &SharedCrateContext<'a, 'tcx>,
    ty: Ty<'tcx>)
    -> ty::Instance<'tcx>
{
    let def_id = scx.tcx().require_lang_item(DropInPlaceFnLangItem);
    let substs = scx.tcx().intern_substs(&[Kind::from(ty)]);
    resolve(scx, def_id, substs)
}

pub fn custom_coerce_unsize_info<'scx, 'tcx>(scx: &SharedCrateContext<'scx, 'tcx>,
                                             source_ty: Ty<'tcx>,
                                             target_ty: Ty<'tcx>)
                                             -> CustomCoerceUnsized {
    let trait_ref = ty::Binder(ty::TraitRef {
        def_id: scx.tcx().lang_items.coerce_unsized_trait().unwrap(),
        substs: scx.tcx().mk_substs_trait(source_ty, &[target_ty])
    });

    match fulfill_obligation(scx, DUMMY_SP, trait_ref) {
        traits::VtableImpl(traits::VtableImplData { impl_def_id, .. }) => {
            scx.tcx().coerce_unsized_info(impl_def_id).custom_kind.unwrap()
        }
        vtable => {
            bug!("invalid CoerceUnsized vtable: {:?}", vtable);
        }
    }
}

/// Monomorphizes a type from the AST by first applying the in-scope
/// substitutions and then normalizing any associated types.
pub fn apply_param_substs<'a, 'tcx, T>(scx: &SharedCrateContext<'a, 'tcx>,
                                       param_substs: &Substs<'tcx>,
                                       value: &T)
                                       -> T
    where T: TransNormalize<'tcx>
{
    let tcx = scx.tcx();
    debug!("apply_param_substs(param_substs={:?}, value={:?})", param_substs, value);
    let substituted = value.subst(tcx, param_substs);
    let substituted = scx.tcx().erase_regions(&substituted);
    AssociatedTypeNormalizer::new(scx).fold(&substituted)
}

/// Returns the normalized type of a struct field
pub fn field_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          param_substs: &Substs<'tcx>,
                          f: &'tcx ty::FieldDef)
                          -> Ty<'tcx>
{
    tcx.normalize_associated_type(&f.ty(tcx, param_substs))
}

struct AssociatedTypeNormalizer<'a, 'b: 'a, 'gcx: 'b> {
    shared: &'a SharedCrateContext<'b, 'gcx>,
}

impl<'a, 'b, 'gcx> AssociatedTypeNormalizer<'a, 'b, 'gcx> {
    fn new(shared: &'a SharedCrateContext<'b, 'gcx>) -> Self {
        AssociatedTypeNormalizer {
            shared: shared,
        }
    }

    fn fold<T:TypeFoldable<'gcx>>(&mut self, value: &T) -> T {
        if !value.has_projection_types() {
            value.clone()
        } else {
            value.fold_with(self)
        }
    }
}

impl<'a, 'b, 'gcx> TypeFolder<'gcx, 'gcx> for AssociatedTypeNormalizer<'a, 'b, 'gcx> {
    fn tcx<'c>(&'c self) -> TyCtxt<'c, 'gcx, 'gcx> {
        self.shared.tcx()
    }

    fn fold_ty(&mut self, ty: Ty<'gcx>) -> Ty<'gcx> {
        if !ty.has_projection_types() {
            ty
        } else {
            self.shared.project_cache().memoize(ty, || {
                debug!("AssociatedTypeNormalizer: ty={:?}", ty);
                self.shared.tcx().normalize_associated_type(&ty)
            })
        }
    }
}
