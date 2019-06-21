use crate::llvm::{self, SetUnnamedAddr, True};
use crate::debuginfo;
use crate::common::CodegenCx;
use crate::base;
use crate::type_::Type;
use crate::type_of::LayoutLlvmExt;
use crate::value::Value;
use libc::c_uint;
use rustc::hir::def_id::DefId;
use rustc::mir::interpret::{ConstValue, Allocation, read_target_uint,
    Pointer, ErrorHandled, GlobalId};
use rustc::mir::mono::MonoItem;
use rustc::hir::Node;
use syntax_pos::Span;
use rustc_target::abi::HasDataLayout;
use syntax::symbol::sym;
use syntax_pos::symbol::LocalInternedString;
use rustc::ty::{self, Ty, Instance};
use rustc_codegen_ssa::traits::*;

use rustc::ty::layout::{self, Size, Align, LayoutOf};

use rustc::hir::{self, CodegenFnAttrs, CodegenFnAttrFlags};

use std::ffi::{CStr, CString};

pub fn const_alloc_to_llvm(cx: &CodegenCx<'ll, '_>, alloc: &Allocation) -> &'ll Value {
    let mut llvals = Vec::with_capacity(alloc.relocations.len() + 1);
    let dl = cx.data_layout();
    let pointer_size = dl.pointer_size.bytes() as usize;

    let mut next_offset = 0;
    for &(offset, ((), alloc_id)) in alloc.relocations.iter() {
        let offset = offset.bytes();
        assert_eq!(offset as usize as u64, offset);
        let offset = offset as usize;
        if offset > next_offset {
            llvals.push(cx.const_bytes(&alloc.bytes[next_offset..offset]));
        }
        let ptr_offset = read_target_uint(
            dl.endian,
            &alloc.bytes[offset..(offset + pointer_size)],
        ).expect("const_alloc_to_llvm: could not read relocation pointer") as u64;
        llvals.push(cx.scalar_to_backend(
            Pointer::new(alloc_id, Size::from_bytes(ptr_offset)).into(),
            &layout::Scalar {
                value: layout::Primitive::Pointer,
                valid_range: 0..=!0
            },
            cx.type_i8p()
        ));
        next_offset = offset + pointer_size;
    }
    if alloc.bytes.len() >= next_offset {
        llvals.push(cx.const_bytes(&alloc.bytes[next_offset ..]));
    }

    cx.const_struct(&llvals, true)
}

pub fn codegen_static_initializer(
    cx: &CodegenCx<'ll, 'tcx>,
    def_id: DefId,
) -> Result<(&'ll Value, &'tcx Allocation), ErrorHandled> {
    let instance = ty::Instance::mono(cx.tcx, def_id);
    let cid = GlobalId {
        instance,
        promoted: None,
    };
    let param_env = ty::ParamEnv::reveal_all();
    let static_ = cx.tcx.const_eval(param_env.and(cid))?;

    let alloc = match static_.val {
        ConstValue::ByRef {
            offset, align, alloc,
        } if offset.bytes() == 0 && align == alloc.align => {
            alloc
        },
        _ => bug!("static const eval returned {:#?}", static_),
    };
    Ok((const_alloc_to_llvm(cx, alloc), alloc))
}

fn set_global_alignment(cx: &CodegenCx<'ll, '_>,
                        gv: &'ll Value,
                        mut align: Align) {
    // The target may require greater alignment for globals than the type does.
    // Note: GCC and Clang also allow `__attribute__((aligned))` on variables,
    // which can force it to be smaller.  Rust doesn't support this yet.
    if let Some(min) = cx.sess().target.target.options.min_global_align {
        match Align::from_bits(min) {
            Ok(min) => align = align.max(min),
            Err(err) => {
                cx.sess().err(&format!("invalid minimum global alignment: {}", err));
            }
        }
    }
    unsafe {
        llvm::LLVMSetAlignment(gv, align.bytes() as u32);
    }
}

fn check_and_apply_linkage(
    cx: &CodegenCx<'ll, 'tcx>,
    attrs: &CodegenFnAttrs,
    ty: Ty<'tcx>,
    sym: LocalInternedString,
    span: Span
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
            cx.sess().span_fatal(
                span, "must have type `*const T` or `*mut T` due to `#[linkage]` attribute")
        };
        unsafe {
            // Declare a symbol `foo` with the desired linkage.
            let g1 = cx.declare_global(&sym, llty2);
            llvm::LLVMRustSetLinkage(g1, base::linkage_to_llvm(linkage));

            // Declare an internal global `extern_with_linkage_foo` which
            // is initialized with the address of `foo`.  If `foo` is
            // discarded during linking (for example, if `foo` has weak
            // linkage and there are no definitions), then
            // `extern_with_linkage_foo` will instead be initialized to
            // zero.
            let mut real_name = "_rust_extern_with_linkage_".to_string();
            real_name.push_str(&sym);
            let g2 = cx.define_global(&real_name, llty).unwrap_or_else(||{
                cx.sess().span_fatal(span, &format!("symbol `{}` is already defined", &sym))
            });
            llvm::LLVMRustSetLinkage(g2, llvm::Linkage::InternalLinkage);
            llvm::LLVMSetInitializer(g2, g1);
            g2
        }
    } else {
        // Generate an external declaration.
        // FIXME(nagisa): investigate whether it can be changed into define_global
        cx.declare_global(&sym, llty)
    }
}

pub fn ptrcast(val: &'ll Value, ty: &'ll Type) -> &'ll Value {
    unsafe {
        llvm::LLVMConstPointerCast(val, ty)
    }
}

impl CodegenCx<'ll, 'tcx> {
    crate fn const_bitcast(&self, val: &'ll Value, ty: &'ll Type) -> &'ll Value {
        unsafe {
            llvm::LLVMConstBitCast(val, ty)
        }
    }

    crate fn static_addr_of_mut(
        &self,
        cv: &'ll Value,
        align: Align,
        kind: Option<&str>,
    ) -> &'ll Value {
        unsafe {
            let gv = match kind {
                Some(kind) if !self.tcx.sess.fewer_names() => {
                    let name = self.generate_local_symbol_name(kind);
                    let gv = self.define_global(&name[..],
                        self.val_ty(cv)).unwrap_or_else(||{
                            bug!("symbol `{}` is already defined", name);
                    });
                    llvm::LLVMRustSetLinkage(gv, llvm::Linkage::PrivateLinkage);
                    gv
                },
                _ => self.define_private_global(self.val_ty(cv)),
            };
            llvm::LLVMSetInitializer(gv, cv);
            set_global_alignment(&self, gv, align);
            SetUnnamedAddr(gv, true);
            gv
        }
    }

    crate fn get_static(&self, def_id: DefId) -> &'ll Value {
        let instance = Instance::mono(self.tcx, def_id);
        if let Some(&g) = self.instances.borrow().get(&instance) {
            return g;
        }

        let defined_in_current_codegen_unit = self.codegen_unit
                                                .items()
                                                .contains_key(&MonoItem::Static(def_id));
        assert!(!defined_in_current_codegen_unit,
                "consts::get_static() should always hit the cache for \
                 statics defined in the same CGU, but did not for `{:?}`",
                 def_id);

        let ty = instance.ty(self.tcx);
        let sym = self.tcx.symbol_name(instance).as_str();

        debug!("get_static: sym={} instance={:?}", sym, instance);

        let g = if let Some(id) = self.tcx.hir().as_local_hir_id(def_id) {

            let llty = self.layout_of(ty).llvm_type(self);
            let (g, attrs) = match self.tcx.hir().get(id) {
                Node::Item(&hir::Item {
                    ref attrs, span, node: hir::ItemKind::Static(..), ..
                }) => {
                    if self.get_declared_value(&sym[..]).is_some() {
                        span_bug!(span, "Conflicting symbol names for static?");
                    }

                    let g = self.define_global(&sym[..], llty).unwrap();

                    if !self.tcx.is_reachable_non_generic(def_id) {
                        unsafe {
                            llvm::LLVMRustSetVisibility(g, llvm::Visibility::Hidden);
                        }
                    }

                    (g, attrs)
                }

                Node::ForeignItem(&hir::ForeignItem {
                    ref attrs, span, node: hir::ForeignItemKind::Static(..), ..
                }) => {
                    let fn_attrs = self.tcx.codegen_fn_attrs(def_id);
                    (check_and_apply_linkage(&self, &fn_attrs, ty, sym, span), attrs)
                }

                item => bug!("get_static: expected static, found {:?}", item)
            };

            debug!("get_static: sym={} attrs={:?}", sym, attrs);

            for attr in attrs {
                if attr.check_name(sym::thread_local) {
                    llvm::set_thread_local_mode(g, self.tls_model);
                }
            }

            g
        } else {
            // FIXME(nagisa): perhaps the map of externs could be offloaded to llvm somehow?
            debug!("get_static: sym={} item_attr={:?}", sym, self.tcx.item_attrs(def_id));

            let attrs = self.tcx.codegen_fn_attrs(def_id);
            let span = self.tcx.def_span(def_id);
            let g = check_and_apply_linkage(&self, &attrs, ty, sym, span);

            // Thread-local statics in some other crate need to *always* be linked
            // against in a thread-local fashion, so we need to be sure to apply the
            // thread-local attribute locally if it was present remotely. If we
            // don't do this then linker errors can be generated where the linker
            // complains that one object files has a thread local version of the
            // symbol and another one doesn't.
            if attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL) {
                llvm::set_thread_local_mode(g, self.tls_model);
            }

            let needs_dll_storage_attr =
                self.use_dll_storage_attrs && !self.tcx.is_foreign_item(def_id) &&
                // ThinLTO can't handle this workaround in all cases, so we don't
                // emit the attrs. Instead we make them unnecessary by disallowing
                // dynamic linking when linker plugin based LTO is enabled.
                !self.tcx.sess.opts.cg.linker_plugin_lto.enabled();

            // If this assertion triggers, there's something wrong with commandline
            // argument validation.
            debug_assert!(!(self.tcx.sess.opts.cg.linker_plugin_lto.enabled() &&
                            self.tcx.sess.target.target.options.is_like_msvc &&
                            self.tcx.sess.opts.cg.prefer_dynamic));

            if needs_dll_storage_attr {
                // This item is external but not foreign, i.e., it originates from an external Rust
                // crate. Since we don't know whether this crate will be linked dynamically or
                // statically in the final application, we always mark such symbols as 'dllimport'.
                // If final linkage happens to be static, we rely on compiler-emitted __imp_ stubs
                // to make things work.
                //
                // However, in some scenarios we defer emission of statics to downstream
                // crates, so there are cases where a static with an upstream DefId
                // is actually present in the current crate. We can find out via the
                // is_codegened_item query.
                if !self.tcx.is_codegened_item(def_id) {
                    unsafe {
                        llvm::LLVMSetDLLStorageClass(g, llvm::DLLStorageClass::DllImport);
                    }
                }
            }
            g
        };

        if self.use_dll_storage_attrs && self.tcx.is_dllimport_foreign_item(def_id) {
            // For foreign (native) libs we know the exact storage type to use.
            unsafe {
                llvm::LLVMSetDLLStorageClass(g, llvm::DLLStorageClass::DllImport);
            }
        }

        self.instances.borrow_mut().insert(instance, g);
        g
    }
}

impl StaticMethods for CodegenCx<'ll, 'tcx> {
    fn static_addr_of(
        &self,
        cv: &'ll Value,
        align: Align,
        kind: Option<&str>,
    ) -> &'ll Value {
        if let Some(&gv) = self.const_globals.borrow().get(&cv) {
            unsafe {
                // Upgrade the alignment in cases where the same constant is used with different
                // alignment requirements
                let llalign = align.bytes() as u32;
                if llalign > llvm::LLVMGetAlignment(gv) {
                    llvm::LLVMSetAlignment(gv, llalign);
                }
            }
            return gv;
        }
        let gv = self.static_addr_of_mut(cv, align, kind);
        unsafe {
            llvm::LLVMSetGlobalConstant(gv, True);
        }
        self.const_globals.borrow_mut().insert(cv, gv);
        gv
    }

    fn codegen_static(
        &self,
        def_id: DefId,
        is_mutable: bool,
    ) {
        unsafe {
            let attrs = self.tcx.codegen_fn_attrs(def_id);

            let (v, alloc) = match codegen_static_initializer(&self, def_id) {
                Ok(v) => v,
                // Error has already been reported
                Err(_) => return,
            };

            let g = self.get_static(def_id);

            // boolean SSA values are i1, but they have to be stored in i8 slots,
            // otherwise some LLVM optimization passes don't work as expected
            let mut val_llty = self.val_ty(v);
            let v = if val_llty == self.type_i1() {
                val_llty = self.type_i8();
                llvm::LLVMConstZExt(v, val_llty)
            } else {
                v
            };

            let instance = Instance::mono(self.tcx, def_id);
            let ty = instance.ty(self.tcx);
            let llty = self.layout_of(ty).llvm_type(self);
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
                    self.llmod, name_string.as_ptr(), val_llty);

                llvm::LLVMRustSetLinkage(new_g, linkage);
                llvm::LLVMRustSetVisibility(new_g, visibility);

                // To avoid breaking any invariants, we leave around the old
                // global for the moment; we'll replace all references to it
                // with the new global later. (See base::codegen_backend.)
                self.statics_to_rauw.borrow_mut().push((g, new_g));
                new_g
            };
            set_global_alignment(&self, g, self.align_of(ty));
            llvm::LLVMSetInitializer(g, v);

            // As an optimization, all shared statics which do not have interior
            // mutability are placed into read-only memory.
            if !is_mutable {
                if self.type_is_freeze(ty) {
                    llvm::LLVMSetGlobalConstant(g, llvm::True);
                }
            }

            debuginfo::create_global_var_metadata(&self, def_id, g);

            if attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL) {
                llvm::set_thread_local_mode(g, self.tls_model);

                // Do not allow LLVM to change the alignment of a TLS on macOS.
                //
                // By default a global's alignment can be freely increased.
                // This allows LLVM to generate more performant instructions
                // e.g., using load-aligned into a SIMD register.
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
                if self.tcx.sess.target.target.options.is_like_osx {
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
            if self.tcx.sess.opts.target_triple.triple().starts_with("wasm32") {
                if let Some(section) = attrs.link_section {
                    let section = llvm::LLVMMDStringInContext(
                        self.llcx,
                        section.as_str().as_ptr() as *const _,
                        section.as_str().len() as c_uint,
                    );
                    let alloc = llvm::LLVMMDStringInContext(
                        self.llcx,
                        alloc.bytes.as_ptr() as *const _,
                        alloc.bytes.len() as c_uint,
                    );
                    let data = [section, alloc];
                    let meta = llvm::LLVMMDNodeInContext(self.llcx, data.as_ptr(), 2);
                    llvm::LLVMAddNamedMetadataOperand(
                        self.llmod,
                        "wasm.custom_sections\0".as_ptr() as *const _,
                        meta,
                    );
                }
            } else {
                base::set_link_section(g, &attrs);
            }

            if attrs.flags.contains(CodegenFnAttrFlags::USED) {
                // This static will be stored in the llvm.used variable which is an array of i8*
                let cast = llvm::LLVMConstPointerCast(g, self.type_i8p());
                self.used_statics.borrow_mut().push(cast);
            }
        }
    }
}
