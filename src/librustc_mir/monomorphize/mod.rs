use rustc::hir::def_id::DefId;
use rustc::middle::lang_items::DropInPlaceFnLangItem;
use rustc::traits;
use rustc::ty::adjustment::CustomCoerceUnsized;
use rustc::ty::{self, Ty, TyCtxt};

pub use rustc::ty::Instance;
pub use self::item::{MonoItem, MonoItemExt};

pub mod collector;
pub mod item;
pub mod partitioning;

#[inline(never)] // give this a place in the profiler
pub fn assert_symbols_are_distinct<'a, 'tcx, I>(tcx: TyCtxt<'a, 'tcx, 'tcx>, mono_items: I)
    where I: Iterator<Item=&'a MonoItem<'tcx>>
{
    let mut symbols: Vec<_> = mono_items.map(|mono_item| {
        (mono_item, mono_item.symbol_name(tcx))
    }).collect();

    symbols.sort_by_key(|sym| sym.1);

    for pair in symbols.windows(2) {
        let sym1 = &pair[0].1;
        let sym2 = &pair[1].1;

        if sym1 == sym2 {
            let mono_item1 = pair[0].0;
            let mono_item2 = pair[1].0;

            let span1 = mono_item1.local_span(tcx);
            let span2 = mono_item2.local_span(tcx);

            // Deterministically select one of the spans for error reporting
            let span = match (span1, span2) {
                (Some(span1), Some(span2)) => {
                    Some(if span1.lo().0 > span2.lo().0 {
                        span1
                    } else {
                        span2
                    })
                }
                (span1, span2) => span1.or(span2),
            };

            let error_message = format!("symbol `{}` is already defined", sym1);

            if let Some(span) = span {
                tcx.sess.span_fatal(span, &error_message)
            } else {
                tcx.sess.fatal(&error_message)
            }
        }
    }
}

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
        .find(|it| it.kind == ty::AssocKind::Method)
        .unwrap().def_id;
    let def = ty::InstanceDef::ClosureOnceShim { call_once };

    let self_ty = tcx.mk_closure(closure_did, substs);

    let sig = substs.closure_sig(closure_did, tcx);
    let sig = tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);
    assert_eq!(sig.inputs().len(), 1);
    let substs = tcx.mk_substs_trait(self_ty, &[sig.inputs()[0].into()]);

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
            // `fn(&mut self, ...)`. In fact, at codegen time, these are
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
            // These are both the same at codegen time.
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
    let substs = tcx.intern_substs(&[ty.into()]);
    Instance::resolve(tcx, ty::ParamEnv::reveal_all(), def_id, substs).unwrap()
}

pub fn custom_coerce_unsize_info<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                           source_ty: Ty<'tcx>,
                                           target_ty: Ty<'tcx>)
                                           -> CustomCoerceUnsized {
    let def_id = tcx.lang_items().coerce_unsized_trait().unwrap();

    let trait_ref = ty::Binder::bind(ty::TraitRef {
        def_id: def_id,
        substs: tcx.mk_substs_trait(source_ty, &[target_ty.into()])
    });

    match tcx.codegen_fulfill_obligation( (ty::ParamEnv::reveal_all(), trait_ref)) {
        traits::VtableImpl(traits::VtableImplData { impl_def_id, .. }) => {
            tcx.coerce_unsized_info(impl_def_id).custom_kind.unwrap()
        }
        vtable => {
            bug!("invalid CoerceUnsized vtable: {:?}", vtable);
        }
    }
}
