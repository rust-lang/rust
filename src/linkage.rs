use rustc::mir::mono::{Linkage as RLinkage, MonoItem, Visibility};

use crate::prelude::*;

pub fn get_clif_linkage(mono_item: MonoItem, linkage: RLinkage, visibility: Visibility) -> Linkage {
    match (linkage, visibility) {
        (RLinkage::External, Visibility::Default) => Linkage::Export,
        (RLinkage::Internal, Visibility::Default) => Linkage::Local,
        (RLinkage::External, Visibility::Hidden) => Linkage::Hidden,
        _ => panic!("{:?} = {:?} {:?}", mono_item, linkage, visibility),
    }
}

pub fn get_static_ref_linkage(tcx: TyCtxt, def_id: DefId) -> Linkage {
    let fn_attrs = tcx.codegen_fn_attrs(def_id);

    if let Some(linkage) = fn_attrs.linkage {
        match linkage {
            RLinkage::External => Linkage::Export,
            RLinkage::Internal => Linkage::Local,
            RLinkage::ExternalWeak | RLinkage::WeakAny => Linkage::Preemptible,
            _ => panic!("{:?}", linkage),
        }
    } else {
        Linkage::Import
    }
}
