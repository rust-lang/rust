// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Walks the crate looking for items/impl-items/trait-items that have
//! either a `rustc_symbol_name` or `rustc_item_path` attribute and
//! generates an error giving, respectively, the symbol name or
//! item-path. This is used for unit testing the code that generates
//! paths etc in all kinds of annoying scenarios.

use asm;
use attributes;
use base;
use consts;
use context::CodegenCx;
use declare;
use llvm;
use monomorphize::Instance;
use type_of::LayoutLlvmExt;
use rustc::hir;
use rustc::hir::def::Def;
use rustc::hir::def_id::DefId;
use rustc::mir::mono::{Linkage, Visibility};
use rustc::ty::TypeFoldable;
use rustc::ty::layout::LayoutOf;
use syntax::attr;
use std::fmt;

pub use rustc::mir::mono::MonoItem;

pub use rustc_mir::monomorphize::item::*;
pub use rustc_mir::monomorphize::item::MonoItemExt as BaseMonoItemExt;

pub trait MonoItemExt<'a, 'tcx>: fmt::Debug + BaseMonoItemExt<'a, 'tcx> {
    fn define(&self, cx: &CodegenCx<'a, 'tcx>) {
        debug!("BEGIN IMPLEMENTING '{} ({})' in cgu {}",
               self.to_string(cx.tcx),
               self.to_raw_string(),
               cx.codegen_unit.name());

        match *self.as_mono_item() {
            MonoItem::Static(def_id) => {
                let tcx = cx.tcx;
                let is_mutable = match tcx.describe_def(def_id) {
                    Some(Def::Static(_, is_mutable)) => is_mutable,
                    Some(other) => {
                        bug!("Expected Def::Static, found {:?}", other)
                    }
                    None => {
                        bug!("Expected Def::Static for {:?}, found nothing", def_id)
                    }
                };
                let attrs = tcx.get_attrs(def_id);

                consts::codegen_static(&cx, def_id, is_mutable, &attrs);
            }
            MonoItem::GlobalAsm(node_id) => {
                let item = cx.tcx.hir.expect_item(node_id);
                if let hir::ItemGlobalAsm(ref ga) = item.node {
                    asm::codegen_global_asm(cx, ga);
                } else {
                    span_bug!(item.span, "Mismatch between hir::Item type and MonoItem type")
                }
            }
            MonoItem::Fn(instance) => {
                base::codegen_instance(&cx, instance);
            }
        }

        debug!("END IMPLEMENTING '{} ({})' in cgu {}",
               self.to_string(cx.tcx),
               self.to_raw_string(),
               cx.codegen_unit.name());
    }

    fn predefine(&self,
                 cx: &CodegenCx<'a, 'tcx>,
                 linkage: Linkage,
                 visibility: Visibility) {
        debug!("BEGIN PREDEFINING '{} ({})' in cgu {}",
               self.to_string(cx.tcx),
               self.to_raw_string(),
               cx.codegen_unit.name());

        let symbol_name = self.symbol_name(cx.tcx).as_str();

        debug!("symbol {}", &symbol_name);

        match *self.as_mono_item() {
            MonoItem::Static(def_id) => {
                predefine_static(cx, def_id, linkage, visibility, &symbol_name);
            }
            MonoItem::Fn(instance) => {
                predefine_fn(cx, instance, linkage, visibility, &symbol_name);
            }
            MonoItem::GlobalAsm(..) => {}
        }

        debug!("END PREDEFINING '{} ({})' in cgu {}",
               self.to_string(cx.tcx),
               self.to_raw_string(),
               cx.codegen_unit.name());
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

impl<'a, 'tcx> MonoItemExt<'a, 'tcx> for MonoItem<'tcx> {}

fn predefine_static<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                              def_id: DefId,
                              linkage: Linkage,
                              visibility: Visibility,
                              symbol_name: &str) {
    let instance = Instance::mono(cx.tcx, def_id);
    let ty = instance.ty(cx.tcx);
    let llty = cx.layout_of(ty).llvm_type(cx);

    let g = declare::define_global(cx, symbol_name, llty).unwrap_or_else(|| {
        cx.sess().span_fatal(cx.tcx.def_span(def_id),
            &format!("symbol `{}` is already defined", symbol_name))
    });

    unsafe {
        llvm::LLVMRustSetLinkage(g, base::linkage_to_llvm(linkage));
        llvm::LLVMRustSetVisibility(g, base::visibility_to_llvm(visibility));
    }

    cx.instances.borrow_mut().insert(instance, g);
    cx.statics.borrow_mut().insert(g, def_id);
}

fn predefine_fn<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                          instance: Instance<'tcx>,
                          linkage: Linkage,
                          visibility: Visibility,
                          symbol_name: &str) {
    assert!(!instance.substs.needs_infer() &&
            !instance.substs.has_param_types());

    let mono_ty = instance.ty(cx.tcx);
    let attrs = instance.def.attrs(cx.tcx);
    let lldecl = declare::declare_fn(cx, symbol_name, mono_ty);
    unsafe { llvm::LLVMRustSetLinkage(lldecl, base::linkage_to_llvm(linkage)) };
    base::set_link_section(cx, lldecl, &attrs);
    if linkage == Linkage::LinkOnceODR ||
        linkage == Linkage::WeakODR {
        llvm::SetUniqueComdat(cx.llmod, lldecl);
    }

    // If we're compiling the compiler-builtins crate, e.g. the equivalent of
    // compiler-rt, then we want to implicitly compile everything with hidden
    // visibility as we're going to link this object all over the place but
    // don't want the symbols to get exported.
    if linkage != Linkage::Internal && linkage != Linkage::Private &&
       attr::contains_name(cx.tcx.hir.krate_attrs(), "compiler_builtins") {
        unsafe {
            llvm::LLVMRustSetVisibility(lldecl, llvm::Visibility::Hidden);
        }
    } else {
        unsafe {
            llvm::LLVMRustSetVisibility(lldecl, base::visibility_to_llvm(visibility));
        }
    }

    debug!("predefine_fn: mono_ty = {:?} instance = {:?}", mono_ty, instance);
    if instance.def.is_inline(cx.tcx) {
        attributes::inline(lldecl, attributes::InlineAttr::Hint);
    }
    attributes::from_fn_attrs(cx, lldecl, instance.def.def_id());

    cx.instances.borrow_mut().insert(instance, lldecl);
}
