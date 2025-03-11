use rustc_ast::expand::autodiff_attrs::{AutoDiffItem, DiffActivity};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::bug;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_symbol_mangling::symbol_name_for_instance_in_crate;
use tracing::{debug, trace};

use crate::partitioning::UsageMap;

fn adjust_activity_to_abi<'tcx>(tcx: TyCtxt<'tcx>, fn_ty: Ty<'tcx>, da: &mut Vec<DiffActivity>) {
    if !matches!(fn_ty.kind(), ty::FnDef(..)) {
        bug!("expected fn def for autodiff, got {:?}", fn_ty);
    }

    // We don't actually pass the types back into the type system.
    // All we do is decide how to handle the arguments.
    let sig = fn_ty.fn_sig(tcx).skip_binder();

    let mut new_activities = vec![];
    let mut new_positions = vec![];
    for (i, ty) in sig.inputs().iter().enumerate() {
        if let Some(inner_ty) = ty.builtin_deref(true) {
            if inner_ty.is_slice() {
                // We know that the length will be passed as extra arg.
                if !da.is_empty() {
                    // We are looking at a slice. The length of that slice will become an
                    // extra integer on llvm level. Integers are always const.
                    // However, if the slice get's duplicated, we want to know to later check the
                    // size. So we mark the new size argument as FakeActivitySize.
                    let activity = match da[i] {
                        DiffActivity::DualOnly
                        | DiffActivity::Dual
                        | DiffActivity::DuplicatedOnly
                        | DiffActivity::Duplicated => DiffActivity::FakeActivitySize,
                        DiffActivity::Const => DiffActivity::Const,
                        _ => bug!("unexpected activity for ptr/ref"),
                    };
                    new_activities.push(activity);
                    new_positions.push(i + 1);
                }
                continue;
            }
        }
    }
    // now add the extra activities coming from slices
    // Reverse order to not invalidate the indices
    for _ in 0..new_activities.len() {
        let pos = new_positions.pop().unwrap();
        let activity = new_activities.pop().unwrap();
        da.insert(pos, activity);
    }
}

pub(crate) fn find_autodiff_source_functions<'tcx>(
    tcx: TyCtxt<'tcx>,
    usage_map: &UsageMap<'tcx>,
    autodiff_mono_items: Vec<(&MonoItem<'tcx>, &Instance<'tcx>)>,
) -> Vec<AutoDiffItem> {
    let mut autodiff_items: Vec<AutoDiffItem> = vec![];
    for (item, instance) in autodiff_mono_items {
        let target_id = instance.def_id();
        let cg_fn_attr = &tcx.codegen_fn_attrs(target_id).autodiff_item;
        let Some(target_attrs) = cg_fn_attr else {
            continue;
        };
        let mut input_activities: Vec<DiffActivity> = target_attrs.input_activity.clone();
        if target_attrs.is_source() {
            trace!("source found: {:?}", target_id);
        }
        if !target_attrs.apply_autodiff() {
            continue;
        }

        let target_symbol = symbol_name_for_instance_in_crate(tcx, instance.clone(), LOCAL_CRATE);

        let source =
            usage_map.used_map.get(&item).unwrap().into_iter().find_map(|item| match *item {
                MonoItem::Fn(ref instance_s) => {
                    let source_id = instance_s.def_id();
                    if let Some(ad) = &tcx.codegen_fn_attrs(source_id).autodiff_item
                        && ad.is_active()
                    {
                        return Some(instance_s);
                    }
                    None
                }
                _ => None,
            });
        let inst = match source {
            Some(source) => source,
            None => continue,
        };

        debug!("source_id: {:?}", inst.def_id());
        let fn_ty = inst.ty(tcx, ty::TypingEnv::fully_monomorphized());
        assert!(fn_ty.is_fn());
        adjust_activity_to_abi(tcx, fn_ty, &mut input_activities);
        let symb = symbol_name_for_instance_in_crate(tcx, inst.clone(), LOCAL_CRATE);

        let mut new_target_attrs = target_attrs.clone();
        new_target_attrs.input_activity = input_activities;
        let itm = new_target_attrs.into_item(symb, target_symbol);
        autodiff_items.push(itm);
    }

    if !autodiff_items.is_empty() {
        trace!("AUTODIFF ITEMS EXIST");
        for item in &mut *autodiff_items {
            trace!("{}", &item);
        }
    }

    autodiff_items
}
