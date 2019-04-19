use rustc::hir;
use rustc::mir::mono::{Linkage, Visibility};
use rustc::ty::layout::HasTyCtxt;
use std::fmt;
use crate::base;
use crate::traits::*;

pub use rustc::mir::mono::MonoItem;

pub use rustc_mir::monomorphize::item::MonoItemExt as BaseMonoItemExt;

pub trait MonoItemExt<'a, 'tcx: 'a>: fmt::Debug + BaseMonoItemExt<'a, 'tcx> {
    fn define<Bx: BuilderMethods<'a, 'tcx>>(&self, cx: &'a Bx::CodegenCx) {
        debug!("BEGIN IMPLEMENTING '{} ({})' in cgu {}",
               self.to_string(cx.tcx(), true),
               self.to_raw_string(),
               cx.codegen_unit().name());

        match *self.as_mono_item() {
            MonoItem::Static(def_id) => {
                cx.codegen_static(def_id, cx.tcx().is_mutable_static(def_id));
            }
            MonoItem::GlobalAsm(hir_id) => {
                let item = cx.tcx().hir().expect_item_by_hir_id(hir_id);
                if let hir::ItemKind::GlobalAsm(ref ga) = item.node {
                    cx.codegen_global_asm(ga);
                } else {
                    span_bug!(item.span, "Mismatch between hir::Item type and MonoItem type")
                }
            }
            MonoItem::Fn(instance) => {
                base::codegen_instance::<Bx>(&cx, instance);
            }
        }

        debug!("END IMPLEMENTING '{} ({})' in cgu {}",
               self.to_string(cx.tcx(), true),
               self.to_raw_string(),
               cx.codegen_unit().name());
    }

    fn predefine<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        cx: &'a Bx::CodegenCx,
        linkage: Linkage,
        visibility: Visibility
    ) {
        debug!("BEGIN PREDEFINING '{} ({})' in cgu {}",
               self.to_string(cx.tcx(), true),
               self.to_raw_string(),
               cx.codegen_unit().name());

        let symbol_name = self.symbol_name(cx.tcx()).as_str();

        debug!("symbol {}", &symbol_name);

        match *self.as_mono_item() {
            MonoItem::Static(def_id) => {
                cx.predefine_static(def_id, linkage, visibility, &symbol_name);
            }
            MonoItem::Fn(instance) => {
                cx.predefine_fn(instance, linkage, visibility, &symbol_name);
            }
            MonoItem::GlobalAsm(..) => {}
        }

        debug!("END PREDEFINING '{} ({})' in cgu {}",
               self.to_string(cx.tcx(), true),
               self.to_raw_string(),
               cx.codegen_unit().name());
    }

    fn to_raw_string(&self) -> String {
        match *self.as_mono_item() {
            MonoItem::Fn(instance) => {
                format!("Fn({:?}, {})",
                        instance.def,
                        instance.substs.as_ptr() as usize)
            }
            MonoItem::Static(id) => {
                format!("Static({:?})", id)
            }
            MonoItem::GlobalAsm(id) => {
                format!("GlobalAsm({:?})", id)
            }
        }
    }
}

impl<'a, 'tcx: 'a> MonoItemExt<'a, 'tcx> for MonoItem<'tcx> {}
