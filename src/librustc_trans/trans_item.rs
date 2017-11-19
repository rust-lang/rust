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
use context::CrateContext;
use common;
use declare;
use llvm;
use monomorphize::Instance;
use type_of::LayoutLlvmExt;
use rustc::hir;
use rustc::middle::trans::{Linkage, Visibility};
use rustc::ty::{self, TyCtxt, TypeFoldable};
use rustc::ty::layout::LayoutOf;
use syntax::ast;
use syntax::attr;
use syntax_pos::Span;
use syntax_pos::symbol::Symbol;
use std::fmt;

pub use rustc::middle::trans::TransItem;

pub use rustc_trans_utils::trans_item::*;
pub use rustc_trans_utils::trans_item::TransItemExt as BaseTransItemExt;

pub trait TransItemExt<'a, 'tcx>: fmt::Debug + BaseTransItemExt<'a, 'tcx> {
    fn define(&self, ccx: &CrateContext<'a, 'tcx>) {
        debug!("BEGIN IMPLEMENTING '{} ({})' in cgu {}",
               self.to_string(ccx.tcx()),
               self.to_raw_string(),
               ccx.codegen_unit().name());

        match *self.as_trans_item() {
            TransItem::Static(node_id) => {
                let tcx = ccx.tcx();
                let item = tcx.hir.expect_item(node_id);
                if let hir::ItemStatic(_, m, _) = item.node {
                    match consts::trans_static(&ccx, m, item.id, &item.attrs) {
                        Ok(_) => { /* Cool, everything's alright. */ },
                        Err(err) => {
                            err.report(tcx, item.span, "static");
                        }
                    };
                } else {
                    span_bug!(item.span, "Mismatch between hir::Item type and TransItem type")
                }
            }
            TransItem::GlobalAsm(node_id) => {
                let item = ccx.tcx().hir.expect_item(node_id);
                if let hir::ItemGlobalAsm(ref ga) = item.node {
                    asm::trans_global_asm(ccx, ga);
                } else {
                    span_bug!(item.span, "Mismatch between hir::Item type and TransItem type")
                }
            }
            TransItem::Fn(instance) => {
                base::trans_instance(&ccx, instance);
            }
        }

        debug!("END IMPLEMENTING '{} ({})' in cgu {}",
               self.to_string(ccx.tcx()),
               self.to_raw_string(),
               ccx.codegen_unit().name());
    }

    fn predefine(&self,
                 ccx: &CrateContext<'a, 'tcx>,
                 linkage: Linkage,
                 visibility: Visibility) {
        debug!("BEGIN PREDEFINING '{} ({})' in cgu {}",
               self.to_string(ccx.tcx()),
               self.to_raw_string(),
               ccx.codegen_unit().name());

        let symbol_name = self.symbol_name(ccx.tcx());

        debug!("symbol {}", &symbol_name);

        match *self.as_trans_item() {
            TransItem::Static(node_id) => {
                predefine_static(ccx, node_id, linkage, visibility, &symbol_name);
            }
            TransItem::Fn(instance) => {
                predefine_fn(ccx, instance, linkage, visibility, &symbol_name);
            }
            TransItem::GlobalAsm(..) => {}
        }

        debug!("END PREDEFINING '{} ({})' in cgu {}",
               self.to_string(ccx.tcx()),
               self.to_raw_string(),
               ccx.codegen_unit().name());
    }

    fn symbol_name(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> ty::SymbolName {
        match *self.as_trans_item() {
            TransItem::Fn(instance) => tcx.symbol_name(instance),
            TransItem::Static(node_id) => {
                let def_id = tcx.hir.local_def_id(node_id);
                tcx.symbol_name(Instance::mono(tcx, def_id))
            }
            TransItem::GlobalAsm(node_id) => {
                let def_id = tcx.hir.local_def_id(node_id);
                ty::SymbolName {
                    name: Symbol::intern(&format!("global_asm_{:?}", def_id)).as_str()
                }
            }
        }
    }

    fn local_span(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Option<Span> {
        match *self.as_trans_item() {
            TransItem::Fn(Instance { def, .. }) => {
                tcx.hir.as_local_node_id(def.def_id())
            }
            TransItem::Static(node_id) |
            TransItem::GlobalAsm(node_id) => {
                Some(node_id)
            }
        }.map(|node_id| tcx.hir.span(node_id))
    }

    fn is_generic_fn(&self) -> bool {
        match *self.as_trans_item() {
            TransItem::Fn(ref instance) => {
                instance.substs.types().next().is_some()
            }
            TransItem::Static(..) |
            TransItem::GlobalAsm(..) => false,
        }
    }

    fn to_raw_string(&self) -> String {
        match *self.as_trans_item() {
            TransItem::Fn(instance) => {
                format!("Fn({:?}, {})",
                         instance.def,
                         instance.substs.as_ptr() as usize)
            }
            TransItem::Static(id) => {
                format!("Static({:?})", id)
            }
            TransItem::GlobalAsm(id) => {
                format!("GlobalAsm({:?})", id)
            }
        }
    }
}

impl<'a, 'tcx> TransItemExt<'a, 'tcx> for TransItem<'tcx> {}

fn predefine_static<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                              node_id: ast::NodeId,
                              linkage: Linkage,
                              visibility: Visibility,
                              symbol_name: &str) {
    let def_id = ccx.tcx().hir.local_def_id(node_id);
    let instance = Instance::mono(ccx.tcx(), def_id);
    let ty = common::instance_ty(ccx.tcx(), &instance);
    let llty = ccx.layout_of(ty).llvm_type(ccx);

    let g = declare::define_global(ccx, symbol_name, llty).unwrap_or_else(|| {
        ccx.sess().span_fatal(ccx.tcx().hir.span(node_id),
            &format!("symbol `{}` is already defined", symbol_name))
    });

    unsafe {
        llvm::LLVMRustSetLinkage(g, base::linkage_to_llvm(linkage));
        llvm::LLVMRustSetVisibility(g, base::visibility_to_llvm(visibility));
    }

    ccx.instances().borrow_mut().insert(instance, g);
    ccx.statics().borrow_mut().insert(g, def_id);
}

fn predefine_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                          instance: Instance<'tcx>,
                          linkage: Linkage,
                          visibility: Visibility,
                          symbol_name: &str) {
    assert!(!instance.substs.needs_infer() &&
            !instance.substs.has_param_types());

    let mono_ty = common::instance_ty(ccx.tcx(), &instance);
    let attrs = instance.def.attrs(ccx.tcx());
    let lldecl = declare::declare_fn(ccx, symbol_name, mono_ty);
    unsafe { llvm::LLVMRustSetLinkage(lldecl, base::linkage_to_llvm(linkage)) };
    base::set_link_section(ccx, lldecl, &attrs);
    if linkage == Linkage::LinkOnceODR ||
        linkage == Linkage::WeakODR {
        llvm::SetUniqueComdat(ccx.llmod(), lldecl);
    }

    // If we're compiling the compiler-builtins crate, e.g. the equivalent of
    // compiler-rt, then we want to implicitly compile everything with hidden
    // visibility as we're going to link this object all over the place but
    // don't want the symbols to get exported.
    if linkage != Linkage::Internal && linkage != Linkage::Private &&
       attr::contains_name(ccx.tcx().hir.krate_attrs(), "compiler_builtins") {
        unsafe {
            llvm::LLVMRustSetVisibility(lldecl, llvm::Visibility::Hidden);
        }
    } else {
        unsafe {
            llvm::LLVMRustSetVisibility(lldecl, base::visibility_to_llvm(visibility));
        }
    }

    debug!("predefine_fn: mono_ty = {:?} instance = {:?}", mono_ty, instance);
    if common::is_inline_instance(ccx.tcx(), &instance) {
        attributes::inline(lldecl, attributes::InlineAttr::Hint);
    }
    attributes::from_fn_attrs(ccx, &attrs, lldecl);

    ccx.instances().borrow_mut().insert(instance, lldecl);
}
