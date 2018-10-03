// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use attributes;
use base;
use context::CodegenCx;
use llvm;
use monomorphize::Instance;
use type_of::LayoutLlvmExt;
use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::mir::mono::{Linkage, Visibility};
use rustc::ty::TypeFoldable;
use rustc::ty::layout::LayoutOf;
use value::Value;
use rustc_codegen_ssa::interfaces::*;

pub use rustc::mir::mono::MonoItem;

pub use rustc_mir::monomorphize::item::MonoItemExt as BaseMonoItemExt;

impl<'ll, 'tcx: 'll> PreDefineMethods<'ll, 'tcx> for CodegenCx<'ll, 'tcx, &'ll Value> {
    fn predefine_static(&self,
                                  def_id: DefId,
                                  linkage: Linkage,
                                  visibility: Visibility,
                                  symbol_name: &str) {
        let instance = Instance::mono(self.tcx, def_id);
        let ty = instance.ty(self.tcx);
        let llty = self.layout_of(ty).llvm_type(self);

        let g = self.define_global(symbol_name, llty).unwrap_or_else(|| {
            self.sess().span_fatal(self.tcx.def_span(def_id),
                &format!("symbol `{}` is already defined", symbol_name))
        });

        unsafe {
            llvm::LLVMRustSetLinkage(g, base::linkage_to_llvm(linkage));
            llvm::LLVMRustSetVisibility(g, base::visibility_to_llvm(visibility));
        }

        self.instances.borrow_mut().insert(instance, g);
    }

    fn predefine_fn(&self,
                              instance: Instance<'tcx>,
                              linkage: Linkage,
                              visibility: Visibility,
                              symbol_name: &str) {
        assert!(!instance.substs.needs_infer() &&
                !instance.substs.has_param_types());

        let mono_ty = instance.ty(self.tcx);
        let attrs = self.tcx.codegen_fn_attrs(instance.def_id());
        let lldecl = self.declare_fn(symbol_name, mono_ty);
        unsafe { llvm::LLVMRustSetLinkage(lldecl, base::linkage_to_llvm(linkage)) };
        base::set_link_section(lldecl, &attrs);
        if linkage == Linkage::LinkOnceODR ||
            linkage == Linkage::WeakODR {
            llvm::SetUniqueComdat(self.llmod, lldecl);
        }

        // If we're compiling the compiler-builtins crate, e.g. the equivalent of
        // compiler-rt, then we want to implicitly compile everything with hidden
        // visibility as we're going to link this object all over the place but
        // don't want the symbols to get exported.
        if linkage != Linkage::Internal && linkage != Linkage::Private &&
           self.tcx.is_compiler_builtins(LOCAL_CRATE) {
            unsafe {
                llvm::LLVMRustSetVisibility(lldecl, llvm::Visibility::Hidden);
            }
        } else {
            unsafe {
                llvm::LLVMRustSetVisibility(lldecl, base::visibility_to_llvm(visibility));
            }
        }

        debug!("predefine_fn: mono_ty = {:?} instance = {:?}", mono_ty, instance);
        if instance.def.is_inline(self.tcx) {
            attributes::inline(self, lldecl, attributes::InlineAttr::Hint);
        }
        attributes::from_fn_attrs(self, lldecl, Some(instance.def.def_id()));

        self.instances.borrow_mut().insert(instance, lldecl);
    }
}
