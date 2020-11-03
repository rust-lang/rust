use rustc_middle::mir::mono::{Linkage as RLinkage, MonoItem, Visibility};

use crate::prelude::*;

pub(crate) fn get_clif_linkage(
    mono_item: MonoItem<'_>,
    linkage: RLinkage,
    visibility: Visibility,
) -> Linkage {
    match (linkage, visibility) {
        (RLinkage::External, Visibility::Default) => Linkage::Export,
        (RLinkage::Internal, Visibility::Default) => Linkage::Local,
        (RLinkage::External, Visibility::Hidden) => Linkage::Hidden,
        _ => panic!("{:?} = {:?} {:?}", mono_item, linkage, visibility),
    }
}

pub(crate) fn get_static_linkage(tcx: TyCtxt<'_>, def_id: DefId) -> Linkage {
    let fn_attrs = tcx.codegen_fn_attrs(def_id);

    if let Some(linkage) = fn_attrs.linkage {
        match linkage {
            RLinkage::External => Linkage::Export,
            RLinkage::Internal => Linkage::Local,
            RLinkage::ExternalWeak | RLinkage::WeakAny => Linkage::Preemptible,
            _ => panic!("{:?}", linkage),
        }
    } else if tcx.is_reachable_non_generic(def_id) {
        Linkage::Export
    } else {
        Linkage::Hidden
    }
}
