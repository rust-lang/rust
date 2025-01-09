use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::mono::{Linkage, MonoItem, MonoItemData, Visibility};
use rustc_middle::ty::layout::HasTyCtxt;
use tracing::debug;

use crate::base;
use crate::mir::naked_asm;
use crate::traits::*;

pub trait MonoItemExt<'a, 'tcx> {
    fn define<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        cx: &'a mut Bx::CodegenCx,
        cgu_name: &str,
        item_data: MonoItemData,
    );
    fn predefine<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        cx: &'a mut Bx::CodegenCx,
        cgu_name: &str,
        linkage: Linkage,
        visibility: Visibility,
    );
    fn to_raw_string(&self) -> String;
}

impl<'a, 'tcx: 'a> MonoItemExt<'a, 'tcx> for MonoItem<'tcx> {
    fn define<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        cx: &'a mut Bx::CodegenCx,
        cgu_name: &str,
        item_data: MonoItemData,
    ) {
        debug!("BEGIN IMPLEMENTING '{} ({})' in cgu {}", self, self.to_raw_string(), cgu_name);

        match *self {
            MonoItem::Static(def_id) => {
                cx.codegen_static(def_id);
            }
            MonoItem::GlobalAsm(item_id) => {
                base::codegen_global_asm(cx, item_id);
            }
            MonoItem::Fn(instance) => {
                if cx
                    .tcx()
                    .codegen_fn_attrs(instance.def_id())
                    .flags
                    .contains(CodegenFnAttrFlags::NAKED)
                {
                    naked_asm::codegen_naked_asm::<Bx::CodegenCx>(cx, instance, item_data);
                } else {
                    base::codegen_instance::<Bx>(cx, instance);
                }
            }
        }

        debug!("END IMPLEMENTING '{} ({})' in cgu {}", self, self.to_raw_string(), cgu_name);
    }

    fn predefine<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        cx: &'a mut Bx::CodegenCx,
        cgu_name: &str,
        linkage: Linkage,
        visibility: Visibility,
    ) {
        debug!("BEGIN PREDEFINING '{} ({})' in cgu {}", self, self.to_raw_string(), cgu_name);

        let symbol_name = self.symbol_name(cx.tcx()).name;

        debug!("symbol {symbol_name}");

        match *self {
            MonoItem::Static(def_id) => {
                cx.predefine_static(def_id, linkage, visibility, symbol_name);
            }
            MonoItem::Fn(instance) => {
                let attrs = cx.tcx().codegen_fn_attrs(instance.def_id());

                if attrs.flags.contains(CodegenFnAttrFlags::NAKED) {
                    // do not define this function; it will become a global assembly block
                } else {
                    cx.predefine_fn(instance, linkage, visibility, symbol_name);
                };
            }
            MonoItem::GlobalAsm(..) => {}
        }

        debug!("END PREDEFINING '{} ({})' in cgu {}", self, self.to_raw_string(), cgu_name);
    }

    fn to_raw_string(&self) -> String {
        match *self {
            MonoItem::Fn(instance) => {
                format!("Fn({:?}, {})", instance.def, instance.args.as_ptr().addr())
            }
            MonoItem::Static(id) => format!("Static({id:?})"),
            MonoItem::GlobalAsm(id) => format!("GlobalAsm({id:?})"),
        }
    }
}
