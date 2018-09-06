// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::c_uint;
use llvm::{self, SetUnnamedAddr, True};
use rustc::hir::def_id::DefId;
use rustc::hir::Node;
use debuginfo;
use base;
use monomorphize::MonoItem;
use common::CodegenCx;
use declare;
use monomorphize::Instance;
use syntax_pos::Span;
use syntax_pos::symbol::LocalInternedString;
use type_::Type;
use type_of::LayoutLlvmExt;
use value::Value;
use rustc::ty::{self, Ty};
use interfaces::{CommonWriteMethods, TypeMethods};

use rustc::ty::layout::{Align, LayoutOf};

use rustc::hir::{self, CodegenFnAttrs, CodegenFnAttrFlags};

use std::ffi::{CStr, CString};

pub fn ptrcast(val: &'ll Value, ty: &'ll Type) -> &'ll Value {
    unsafe {
        llvm::LLVMConstPointerCast(val, ty)
    }
}

pub fn bitcast(val: &'ll Value, ty: &'ll Type) -> &'ll Value {
    unsafe {
        llvm::LLVMConstBitCast(val, ty)
    }
}

fn set_global_alignment(cx: &CodegenCx<'ll, '_>,
                        gv: &'ll Value,
                        mut align: Align) {
    // The target may require greater alignment for globals than the type does.
    // Note: GCC and Clang also allow `__attribute__((aligned))` on variables,
    // which can force it to be smaller.  Rust doesn't support this yet.
    if let Some(min) = cx.sess().target.target.options.min_global_align {
        match ty::layout::Align::from_bits(min, min) {
            Ok(min) => align = align.max(min),
            Err(err) => {
                cx.sess().err(&format!("invalid minimum global alignment: {}", err));
            }
        }
    }
    unsafe {
        llvm::LLVMSetAlignment(gv, align.abi() as u32);
    }
}

pub fn addr_of_mut(
    cx: &CodegenCx<'ll, '_>,
    cv: &'ll Value,
    align: Align,
    kind: Option<&str>,
) -> &'ll Value {
    unsafe {
        let gv = match kind {
            Some(kind) if !cx.tcx.sess.fewer_names() => {
                let name = cx.generate_local_symbol_name(kind);
                let gv = declare::define_global(cx, &name[..],
                    cx.val_ty(cv)).unwrap_or_else(||{
                        bug!("symbol `{}` is already defined", name);
                });
                llvm::LLVMRustSetLinkage(gv, llvm::Linkage::PrivateLinkage);
                gv
            },
            _ => declare::define_private_global(cx, cx.val_ty(cv)),
        };
        llvm::LLVMSetInitializer(gv, cv);
        set_global_alignment(cx, gv, align);
        SetUnnamedAddr(gv, true);
        gv
    }
}

pub fn addr_of(
    cx: &CodegenCx<'ll, '_>,
    cv: &'ll Value,
    align: Align,
    kind: Option<&str>,
) -> &'ll Value {
    if let Some(&gv) = cx.const_globals.borrow().get(&cv) {
        unsafe {
            // Upgrade the alignment in cases where the same constant is used with different
            // alignment requirements
            let llalign = align.abi() as u32;
            if llalign > llvm::LLVMGetAlignment(gv) {
                llvm::LLVMSetAlignment(gv, llalign);
            }
        }
        return gv;
    }
    let gv = addr_of_mut(cx, cv, align, kind);
    unsafe {
        llvm::LLVMSetGlobalConstant(gv, True);
    }
    cx.const_globals.borrow_mut().insert(cv, gv);
    gv
}

pub fn get_static(cx: &CodegenCx<'ll, '_>, def_id: DefId) -> &'ll Value {
    let instance = Instance::mono(cx.tcx, def_id);
    if let Some(&g) = cx.instances.borrow().get(&instance) {
        return g;
    }

    let defined_in_current_codegen_unit = cx.codegen_unit
                                            .items()
                                            .contains_key(&MonoItem::Static(def_id));
    assert!(!defined_in_current_codegen_unit,
            "consts::get_static() should always hit the cache for \
             statics defined in the same CGU, but did not for `{:?}`",
            def_id);

    let ty = instance.ty(cx.tcx);
    let sym = cx.tcx.symbol_name(instance).as_str();

    debug!("get_static: sym={} instance={:?}", sym, instance);

    let g = if let Some(id) = cx.tcx.hir.as_local_node_id(def_id) {

        let llty = cx.layout_of(ty).llvm_type(cx);
        let (g, attrs) = match cx.tcx.hir.get(id) {
            Node::Item(&hir::Item {
                ref attrs, span, node: hir::ItemKind::Static(..), ..
            }) => {
                if declare::get_declared_value(cx, &sym[..]).is_some() {
                    span_bug!(span, "Conflicting symbol names for static?");
                }

                let g = declare::define_global(cx, &sym[..], llty).unwrap();

                if !cx.tcx.is_reachable_non_generic(def_id) {
                    unsafe {
                        llvm::LLVMRustSetVisibility(g, llvm::Visibility::Hidden);
                    }
                }

                (g, attrs)
            }

            Node::ForeignItem(&hir::ForeignItem {
                ref attrs, span, node: hir::ForeignItemKind::Static(..), ..
            }) => {
                let fn_attrs = cx.tcx.codegen_fn_attrs(def_id);
                (check_and_apply_linkage(cx, &fn_attrs, ty, sym, Some(span)), attrs)
            }

            item => bug!("get_static: expected static, found {:?}", item)
        };

        debug!("get_static: sym={} attrs={:?}", sym, attrs);

        for attr in attrs {
            if attr.check_name("thread_local") {
                llvm::set_thread_local_mode(g, cx.tls_model);
            }
        }

        g
    } else {
        // FIXME(nagisa): perhaps the map of externs could be offloaded to llvm somehow?
        debug!("get_static: sym={} item_attr={:?}", sym, cx.tcx.item_attrs(def_id));

        let attrs = cx.tcx.codegen_fn_attrs(def_id);
        let g = check_and_apply_linkage(cx, &attrs, ty, sym, None);

        // Thread-local statics in some other crate need to *always* be linked
        // against in a thread-local fashion, so we need to be sure to apply the
        // thread-local attribute locally if it was present remotely. If we
        // don't do this then linker errors can be generated where the linker
        // complains that one object files has a thread local version of the
        // symbol and another one doesn't.
        if attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL) {
            llvm::set_thread_local_mode(g, cx.tls_model);
        }

        let needs_dll_storage_attr =
            cx.use_dll_storage_attrs && !cx.tcx.is_foreign_item(def_id) &&
            // ThinLTO can't handle this workaround in all cases, so we don't
            // emit the attrs. Instead we make them unnecessary by disallowing
            // dynamic linking when cross-language LTO is enabled.
            !cx.tcx.sess.opts.debugging_opts.cross_lang_lto.enabled();

        // If this assertion triggers, there's something wrong with commandline
        // argument validation.
        debug_assert!(!(cx.tcx.sess.opts.debugging_opts.cross_lang_lto.enabled() &&
                        cx.tcx.sess.target.target.options.is_like_msvc &&
                        cx.tcx.sess.opts.cg.prefer_dynamic));

        if needs_dll_storage_attr {
            // This item is external but not foreign, i.e. it originates from an external Rust
            // crate. Since we don't know whether this crate will be linked dynamically or
            // statically in the final application, we always mark such symbols as 'dllimport'.
            // If final linkage happens to be static, we rely on compiler-emitted __imp_ stubs to
            // make things work.
            //
            // However, in some scenarios we defer emission of statics to downstream
            // crates, so there are cases where a static with an upstream DefId
            // is actually present in the current crate. We can find out via the
            // is_codegened_item query.
            if !cx.tcx.is_codegened_item(def_id) {
                unsafe {
                    llvm::LLVMSetDLLStorageClass(g, llvm::DLLStorageClass::DllImport);
                }
            }
        }
        g
    };

    if cx.use_dll_storage_attrs && cx.tcx.is_dllimport_foreign_item(def_id) {
        // For foreign (native) libs we know the exact storage type to use.
        unsafe {
            llvm::LLVMSetDLLStorageClass(g, llvm::DLLStorageClass::DllImport);
        }
    }

    cx.instances.borrow_mut().insert(instance, g);
    g
}

fn check_and_apply_linkage(
    cx: &CodegenCx<'ll, 'tcx>,
    attrs: &CodegenFnAttrs,
    ty: Ty<'tcx>,
    sym: LocalInternedString,
    span: Option<Span>
) -> &'ll Value {
    let llty = cx.layout_of(ty).llvm_type(cx);
    if let Some(linkage) = attrs.linkage {
        debug!("get_static: sym={} linkage={:?}", sym, linkage);

        // If this is a static with a linkage specified, then we need to handle
        // it a little specially. The typesystem prevents things like &T and
        // extern "C" fn() from being non-null, so we can't just declare a
        // static and call it a day. Some linkages (like weak) will make it such
        // that the static actually has a null value.
        let llty2 = if let ty::RawPtr(ref mt) = ty.sty {
            cx.layout_of(mt.ty).llvm_type(cx)
        } else {
            if let Some(span) = span {
                cx.sess().span_fatal(span, "must have type `*const T` or `*mut T`")
            } else {
                bug!("must have type `*const T` or `*mut T`")
            }
        };
        unsafe {
            // Declare a symbol `foo` with the desired linkage.
            let g1 = declare::declare_global(cx, &sym, llty2);
            llvm::LLVMRustSetLinkage(g1, base::linkage_to_llvm(linkage));

            // Declare an internal global `extern_with_linkage_foo` which
            // is initialized with the address of `foo`.  If `foo` is
            // discarded during linking (for example, if `foo` has weak
            // linkage and there are no definitions), then
            // `extern_with_linkage_foo` will instead be initialized to
            // zero.
            let mut real_name = "_rust_extern_with_linkage_".to_string();
            real_name.push_str(&sym);
            let g2 = declare::define_global(cx, &real_name, llty).unwrap_or_else(||{
                if let Some(span) = span {
                    cx.sess().span_fatal(
                        span,
                        &format!("symbol `{}` is already defined", &sym)
                    )
                } else {
                    bug!("symbol `{}` is already defined", &sym)
                }
            });
            llvm::LLVMRustSetLinkage(g2, llvm::Linkage::InternalLinkage);
            llvm::LLVMSetInitializer(g2, g1);
            g2
        }
    } else {
        // Generate an external declaration.
        // FIXME(nagisa): investigate whether it can be changed into define_global
        declare::declare_global(cx, &sym, llty)
    }
}

pub fn codegen_static<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    def_id: DefId,
    is_mutable: bool,
) {
    unsafe {
        let attrs = cx.tcx.codegen_fn_attrs(def_id);

        let (v, alloc) = match ::mir::codegen_static_initializer(cx, def_id) {
            Ok(v) => v,
            // Error has already been reported
            Err(_) => return,
        };

        let g = get_static(cx, def_id);

        // boolean SSA values are i1, but they have to be stored in i8 slots,
        // otherwise some LLVM optimization passes don't work as expected
        let mut val_llty = cx.val_ty(v);
        let v = if val_llty == cx.type_i1() {
            val_llty = cx.type_i8();
            llvm::LLVMConstZExt(v, val_llty)
        } else {
            v
        };

        let instance = Instance::mono(cx.tcx, def_id);
        let ty = instance.ty(cx.tcx);
        let llty = cx.layout_of(ty).llvm_type(cx);
        let g = if val_llty == llty {
            g
        } else {
            // If we created the global with the wrong type,
            // correct the type.
            let empty_string = const_cstr!("");
            let name_str_ref = CStr::from_ptr(llvm::LLVMGetValueName(g));
            let name_string = CString::new(name_str_ref.to_bytes()).unwrap();
            llvm::LLVMSetValueName(g, empty_string.as_ptr());

            let linkage = llvm::LLVMRustGetLinkage(g);
            let visibility = llvm::LLVMRustGetVisibility(g);

            let new_g = llvm::LLVMRustGetOrInsertGlobal(
                cx.llmod, name_string.as_ptr(), val_llty);

            llvm::LLVMRustSetLinkage(new_g, linkage);
            llvm::LLVMRustSetVisibility(new_g, visibility);

            // To avoid breaking any invariants, we leave around the old
            // global for the moment; we'll replace all references to it
            // with the new global later. (See base::codegen_backend.)
            cx.statics_to_rauw.borrow_mut().push((g, new_g));
            new_g
        };
        set_global_alignment(cx, g, cx.align_of(ty));
        llvm::LLVMSetInitializer(g, v);

        // As an optimization, all shared statics which do not have interior
        // mutability are placed into read-only memory.
        if !is_mutable {
            if cx.type_is_freeze(ty) {
                llvm::LLVMSetGlobalConstant(g, llvm::True);
            }
        }

        debuginfo::create_global_var_metadata(cx, def_id, g);

        if attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL) {
            llvm::set_thread_local_mode(g, cx.tls_model);

            // Do not allow LLVM to change the alignment of a TLS on macOS.
            //
            // By default a global's alignment can be freely increased.
            // This allows LLVM to generate more performant instructions
            // e.g. using load-aligned into a SIMD register.
            //
            // However, on macOS 10.10 or below, the dynamic linker does not
            // respect any alignment given on the TLS (radar 24221680).
            // This will violate the alignment assumption, and causing segfault at runtime.
            //
            // This bug is very easy to trigger. In `println!` and `panic!`,
            // the `LOCAL_STDOUT`/`LOCAL_STDERR` handles are stored in a TLS,
            // which the values would be `mem::replace`d on initialization.
            // The implementation of `mem::replace` will use SIMD
            // whenever the size is 32 bytes or higher. LLVM notices SIMD is used
            // and tries to align `LOCAL_STDOUT`/`LOCAL_STDERR` to a 32-byte boundary,
            // which macOS's dyld disregarded and causing crashes
            // (see issues #51794, #51758, #50867, #48866 and #44056).
            //
            // To workaround the bug, we trick LLVM into not increasing
            // the global's alignment by explicitly assigning a section to it
            // (equivalent to automatically generating a `#[link_section]` attribute).
            // See the comment in the `GlobalValue::canIncreaseAlignment()` function
            // of `lib/IR/Globals.cpp` for why this works.
            //
            // When the alignment is not increased, the optimized `mem::replace`
            // will use load-unaligned instructions instead, and thus avoiding the crash.
            //
            // We could remove this hack whenever we decide to drop macOS 10.10 support.
            if cx.tcx.sess.target.target.options.is_like_osx {
                let sect_name = if alloc.bytes.iter().all(|b| *b == 0) {
                    CStr::from_bytes_with_nul_unchecked(b"__DATA,__thread_bss\0")
                } else {
                    CStr::from_bytes_with_nul_unchecked(b"__DATA,__thread_data\0")
                };
                llvm::LLVMSetSection(g, sect_name.as_ptr());
            }
        }


        // Wasm statics with custom link sections get special treatment as they
        // go into custom sections of the wasm executable.
        if cx.tcx.sess.opts.target_triple.triple().starts_with("wasm32") {
            if let Some(section) = attrs.link_section {
                let section = llvm::LLVMMDStringInContext(
                    cx.llcx,
                    section.as_str().as_ptr() as *const _,
                    section.as_str().len() as c_uint,
                );
                let alloc = llvm::LLVMMDStringInContext(
                    cx.llcx,
                    alloc.bytes.as_ptr() as *const _,
                    alloc.bytes.len() as c_uint,
                );
                let data = [section, alloc];
                let meta = llvm::LLVMMDNodeInContext(cx.llcx, data.as_ptr(), 2);
                llvm::LLVMAddNamedMetadataOperand(
                    cx.llmod,
                    "wasm.custom_sections\0".as_ptr() as *const _,
                    meta,
                );
            }
        } else {
            base::set_link_section(g, &attrs);
        }

        if attrs.flags.contains(CodegenFnAttrFlags::USED) {
            // This static will be stored in the llvm.used variable which is an array of i8*
            let cast = llvm::LLVMConstPointerCast(g, cx.type_i8p());
            cx.used_statics.borrow_mut().push(cast);
        }
    }
}
