// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use llvm;
use llvm::{SetUnnamedAddr};
use llvm::{ValueRef, True};
use rustc_const_eval::ConstEvalErr;
use rustc::hir::def_id::DefId;
use rustc::hir::map as hir_map;
use {debuginfo, machine};
use base;
use trans_item::TransItem;
use common::{CrateContext, val_ty};
use declare;
use monomorphize::{Instance};
use type_::Type;
use type_of;
use rustc::ty;

use rustc::hir;

use std::ffi::{CStr, CString};
use syntax::ast;
use syntax::attr;

pub fn ptrcast(val: ValueRef, ty: Type) -> ValueRef {
    unsafe {
        llvm::LLVMConstPointerCast(val, ty.to_ref())
    }
}

pub fn addr_of_mut(ccx: &CrateContext,
                   cv: ValueRef,
                   align: machine::llalign,
                   kind: &str)
                    -> ValueRef {
    unsafe {
        let name = ccx.generate_local_symbol_name(kind);
        let gv = declare::define_global(ccx, &name[..], val_ty(cv)).unwrap_or_else(||{
            bug!("symbol `{}` is already defined", name);
        });
        llvm::LLVMSetInitializer(gv, cv);
        llvm::LLVMSetAlignment(gv, align);
        llvm::LLVMRustSetLinkage(gv, llvm::Linkage::InternalLinkage);
        SetUnnamedAddr(gv, true);
        gv
    }
}

pub fn addr_of(ccx: &CrateContext,
               cv: ValueRef,
               align: machine::llalign,
               kind: &str)
               -> ValueRef {
    if let Some(&gv) = ccx.const_globals().borrow().get(&cv) {
        unsafe {
            // Upgrade the alignment in cases where the same constant is used with different
            // alignment requirements
            if align > llvm::LLVMGetAlignment(gv) {
                llvm::LLVMSetAlignment(gv, align);
            }
        }
        return gv;
    }
    let gv = addr_of_mut(ccx, cv, align, kind);
    unsafe {
        llvm::LLVMSetGlobalConstant(gv, True);
    }
    ccx.const_globals().borrow_mut().insert(cv, gv);
    gv
}

pub fn get_static(ccx: &CrateContext, def_id: DefId) -> ValueRef {
    let instance = Instance::mono(ccx.shared(), def_id);
    if let Some(&g) = ccx.instances().borrow().get(&instance) {
        return g;
    }

    let ty = ccx.tcx().item_type(def_id);
    let g = if let Some(id) = ccx.tcx().map.as_local_node_id(def_id) {

        let llty = type_of::type_of(ccx, ty);
        let (g, attrs) = match ccx.tcx().map.get(id) {
            hir_map::NodeItem(&hir::Item {
                ref attrs, span, node: hir::ItemStatic(..), ..
            }) => {
                let sym = ccx.symbol_map()
                             .get(TransItem::Static(id))
                             .expect("Local statics should always be in the SymbolMap");

                let defined_in_current_codegen_unit = ccx.codegen_unit()
                                                         .items()
                                                         .contains_key(&TransItem::Static(id));
                assert!(!defined_in_current_codegen_unit);

                if declare::get_declared_value(ccx, sym).is_some() {
                    span_bug!(span, "trans: Conflicting symbol names for static?");
                }

                let g = declare::define_global(ccx, sym, llty).unwrap();

                (g, attrs)
            }

            hir_map::NodeForeignItem(&hir::ForeignItem {
                ref attrs, span, node: hir::ForeignItemStatic(..), ..
            }) => {
                let sym = instance.symbol_name(ccx.shared());
                let g = if let Some(name) =
                        attr::first_attr_value_str_by_name(&attrs, "linkage") {
                    // If this is a static with a linkage specified, then we need to handle
                    // it a little specially. The typesystem prevents things like &T and
                    // extern "C" fn() from being non-null, so we can't just declare a
                    // static and call it a day. Some linkages (like weak) will make it such
                    // that the static actually has a null value.
                    let linkage = match base::llvm_linkage_by_name(&name.as_str()) {
                        Some(linkage) => linkage,
                        None => {
                            ccx.sess().span_fatal(span, "invalid linkage specified");
                        }
                    };
                    let llty2 = match ty.sty {
                        ty::TyRawPtr(ref mt) => type_of::type_of(ccx, mt.ty),
                        _ => {
                            ccx.sess().span_fatal(span, "must have type `*const T` or `*mut T`");
                        }
                    };
                    unsafe {
                        // Declare a symbol `foo` with the desired linkage.
                        let g1 = declare::declare_global(ccx, &sym, llty2);
                        llvm::LLVMRustSetLinkage(g1, linkage);

                        // Declare an internal global `extern_with_linkage_foo` which
                        // is initialized with the address of `foo`.  If `foo` is
                        // discarded during linking (for example, if `foo` has weak
                        // linkage and there are no definitions), then
                        // `extern_with_linkage_foo` will instead be initialized to
                        // zero.
                        let mut real_name = "_rust_extern_with_linkage_".to_string();
                        real_name.push_str(&sym);
                        let g2 = declare::define_global(ccx, &real_name, llty).unwrap_or_else(||{
                            ccx.sess().span_fatal(span,
                                &format!("symbol `{}` is already defined", &sym))
                        });
                        llvm::LLVMRustSetLinkage(g2, llvm::Linkage::InternalLinkage);
                        llvm::LLVMSetInitializer(g2, g1);
                        g2
                    }
                } else {
                    // Generate an external declaration.
                    declare::declare_global(ccx, &sym, llty)
                };

                (g, attrs)
            }

            item => bug!("get_static: expected static, found {:?}", item)
        };

        for attr in attrs {
            if attr.check_name("thread_local") {
                llvm::set_thread_local(g, true);
            }
        }

        g
    } else {
        let sym = instance.symbol_name(ccx.shared());

        // FIXME(nagisa): perhaps the map of externs could be offloaded to llvm somehow?
        // FIXME(nagisa): investigate whether it can be changed into define_global
        let g = declare::declare_global(ccx, &sym, type_of::type_of(ccx, ty));
        // Thread-local statics in some other crate need to *always* be linked
        // against in a thread-local fashion, so we need to be sure to apply the
        // thread-local attribute locally if it was present remotely. If we
        // don't do this then linker errors can be generated where the linker
        // complains that one object files has a thread local version of the
        // symbol and another one doesn't.
        for attr in ccx.tcx().get_attrs(def_id).iter() {
            if attr.check_name("thread_local") {
                llvm::set_thread_local(g, true);
            }
        }
        if ccx.use_dll_storage_attrs() && !ccx.sess().cstore.is_foreign_item(def_id) {
            // This item is external but not foreign, i.e. it originates from an external Rust
            // crate. Since we don't know whether this crate will be linked dynamically or
            // statically in the final application, we always mark such symbols as 'dllimport'.
            // If final linkage happens to be static, we rely on compiler-emitted __imp_ stubs to
            // make things work.
            unsafe {
                llvm::LLVMSetDLLStorageClass(g, llvm::DLLStorageClass::DllImport);
            }
        }
        g
    };

    if ccx.use_dll_storage_attrs() && ccx.sess().cstore.is_dllimport_foreign_item(def_id) {
        // For foreign (native) libs we know the exact storage type to use.
        unsafe {
            llvm::LLVMSetDLLStorageClass(g, llvm::DLLStorageClass::DllImport);
        }
    }
    ccx.instances().borrow_mut().insert(instance, g);
    ccx.statics().borrow_mut().insert(g, def_id);
    g
}

pub fn trans_static(ccx: &CrateContext,
                    m: hir::Mutability,
                    id: ast::NodeId,
                    attrs: &[ast::Attribute])
                    -> Result<ValueRef, ConstEvalErr> {
    unsafe {
        let def_id = ccx.tcx().map.local_def_id(id);
        let g = get_static(ccx, def_id);

        let v = ::mir::trans_static_initializer(ccx, def_id)?;

        // boolean SSA values are i1, but they have to be stored in i8 slots,
        // otherwise some LLVM optimization passes don't work as expected
        let mut val_llty = val_ty(v);
        let v = if val_llty == Type::i1(ccx) {
            val_llty = Type::i8(ccx);
            llvm::LLVMConstZExt(v, val_llty.to_ref())
        } else {
            v
        };

        let ty = ccx.tcx().item_type(def_id);
        let llty = type_of::type_of(ccx, ty);
        let g = if val_llty == llty {
            g
        } else {
            // If we created the global with the wrong type,
            // correct the type.
            let empty_string = CString::new("").unwrap();
            let name_str_ref = CStr::from_ptr(llvm::LLVMGetValueName(g));
            let name_string = CString::new(name_str_ref.to_bytes()).unwrap();
            llvm::LLVMSetValueName(g, empty_string.as_ptr());
            let new_g = llvm::LLVMRustGetOrInsertGlobal(
                ccx.llmod(), name_string.as_ptr(), val_llty.to_ref());
            // To avoid breaking any invariants, we leave around the old
            // global for the moment; we'll replace all references to it
            // with the new global later. (See base::trans_crate.)
            ccx.statics_to_rauw().borrow_mut().push((g, new_g));
            new_g
        };
        llvm::LLVMSetAlignment(g, type_of::align_of(ccx, ty));
        llvm::LLVMSetInitializer(g, v);

        // As an optimization, all shared statics which do not have interior
        // mutability are placed into read-only memory.
        if m != hir::MutMutable {
            let tcontents = ty.type_contents(ccx.tcx());
            if !tcontents.interior_unsafe() {
                llvm::LLVMSetGlobalConstant(g, llvm::True);
            }
        }

        debuginfo::create_global_var_metadata(ccx, id, g);

        if attr::contains_name(attrs,
                               "thread_local") {
            llvm::set_thread_local(g, true);
        }

        base::set_link_section(ccx, g, attrs);

        Ok(g)
    }
}
