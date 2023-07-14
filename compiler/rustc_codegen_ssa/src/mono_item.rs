use crate::base;
use crate::common;
use crate::traits::*;
use rustc_hir as hir;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::mono::{Linkage, Visibility};
use rustc_middle::ty;
use rustc_middle::ty::layout::{HasTyCtxt, LayoutOf};
use rustc_middle::ty::Instance;

pub trait MonoItemExt<'a, 'tcx> {
    fn define<Bx: BuilderMethods<'a, 'tcx>>(&self, cx: &'a Bx::CodegenCx);
    fn predefine<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        cx: &'a Bx::CodegenCx,
        linkage: Linkage,
        visibility: Visibility,
    );
    fn to_raw_string(&self) -> String;
}

impl<'a, 'tcx: 'a> MonoItemExt<'a, 'tcx> for MonoItem<'tcx> {
    fn define<Bx: BuilderMethods<'a, 'tcx>>(&self, cx: &'a Bx::CodegenCx) {
        debug!(
            "BEGIN IMPLEMENTING '{} ({})' in cgu {}",
            self,
            self.to_raw_string(),
            cx.codegen_unit().name()
        );

        match *self {
            MonoItem::Static(def_id) => {
                cx.codegen_static(def_id, cx.tcx().is_mutable_static(def_id));
            }
            MonoItem::GlobalAsm(item_id) => {
                let item = cx.tcx().hir().item(item_id);
                if let hir::ItemKind::GlobalAsm(ref asm) = item.kind {
                    let operands: Vec<_> = asm
                        .operands
                        .iter()
                        .map(|(op, op_sp)| match *op {
                            hir::InlineAsmOperand::Const { ref anon_const } => {
                                let const_value = cx
                                    .tcx()
                                    .const_eval_poly(anon_const.def_id.to_def_id())
                                    .unwrap_or_else(|_| {
                                        span_bug!(*op_sp, "asm const cannot be resolved")
                                    });
                                let ty = cx
                                    .tcx()
                                    .typeck_body(anon_const.body)
                                    .node_type(anon_const.hir_id);
                                let string = common::asm_const_to_str(
                                    cx.tcx(),
                                    *op_sp,
                                    const_value,
                                    cx.layout_of(ty),
                                );
                                GlobalAsmOperandRef::Const { string }
                            }
                            hir::InlineAsmOperand::SymFn { ref anon_const } => {
                                let ty = cx
                                    .tcx()
                                    .typeck_body(anon_const.body)
                                    .node_type(anon_const.hir_id);
                                let instance = match ty.kind() {
                                    &ty::FnDef(def_id, args) => Instance::new(def_id, args),
                                    _ => span_bug!(*op_sp, "asm sym is not a function"),
                                };

                                GlobalAsmOperandRef::SymFn { instance }
                            }
                            hir::InlineAsmOperand::SymStatic { path: _, def_id } => {
                                GlobalAsmOperandRef::SymStatic { def_id }
                            }
                            hir::InlineAsmOperand::In { .. }
                            | hir::InlineAsmOperand::Out { .. }
                            | hir::InlineAsmOperand::InOut { .. }
                            | hir::InlineAsmOperand::SplitInOut { .. } => {
                                span_bug!(*op_sp, "invalid operand type for global_asm!")
                            }
                        })
                        .collect();

                    cx.codegen_global_asm(asm.template, &operands, asm.options, asm.line_spans);
                } else {
                    span_bug!(item.span, "Mismatch between hir::Item type and MonoItem type")
                }
            }
            MonoItem::Fn(instance) => {
                base::codegen_instance::<Bx>(&cx, instance);
            }
        }

        debug!(
            "END IMPLEMENTING '{} ({})' in cgu {}",
            self,
            self.to_raw_string(),
            cx.codegen_unit().name()
        );
    }

    fn predefine<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        cx: &'a Bx::CodegenCx,
        linkage: Linkage,
        visibility: Visibility,
    ) {
        debug!(
            "BEGIN PREDEFINING '{} ({})' in cgu {}",
            self,
            self.to_raw_string(),
            cx.codegen_unit().name()
        );

        let symbol_name = self.symbol_name(cx.tcx()).name;

        debug!("symbol {}", &symbol_name);

        match *self {
            MonoItem::Static(def_id) => {
                cx.predefine_static(def_id, linkage, visibility, &symbol_name);
            }
            MonoItem::Fn(instance) => {
                cx.predefine_fn(instance, linkage, visibility, &symbol_name);
            }
            MonoItem::GlobalAsm(..) => {}
        }

        debug!(
            "END PREDEFINING '{} ({})' in cgu {}",
            self,
            self.to_raw_string(),
            cx.codegen_unit().name()
        );
    }

    fn to_raw_string(&self) -> String {
        match *self {
            MonoItem::Fn(instance) => {
                format!("Fn({:?}, {})", instance.def, instance.args.as_ptr().addr())
            }
            MonoItem::Static(id) => format!("Static({:?})", id),
            MonoItem::GlobalAsm(id) => format!("GlobalAsm({:?})", id),
        }
    }
}
