use crate::base;
use crate::common::CodegenCx;
use crate::debuginfo;
use crate::llvm::{self, True};
use crate::type_::Type;
use crate::type_of::LayoutLlvmExt;
use crate::value::Value;
use cstr::cstr;
use libc::c_uint;
use rustc_codegen_ssa::traits::*;
use rustc_hir::def_id::DefId;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::mir::interpret::{
    read_target_uint, Allocation, ErrorHandled, GlobalAlloc, InitChunk, Pointer,
    Scalar as InterpScalar,
};
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, Instance, Ty};
use rustc_middle::{bug, span_bug};
use rustc_target::abi::{
    AddressSpace, Align, HasDataLayout, Primitive, Scalar, Size, WrappingRange,
};
use std::ops::Range;
use tracing::debug;

pub fn const_alloc_to_llvm(cx: &CodegenCx<'ll, '_>, alloc: &Allocation) -> &'ll Value {
    let mut llvals = Vec::with_capacity(alloc.relocations().len() + 1);
    let dl = cx.data_layout();
    let pointer_size = dl.pointer_size.bytes() as usize;

    // Note: this function may call `inspect_with_uninit_and_ptr_outside_interpreter`,
    // so `range` must be within the bounds of `alloc` and not contain or overlap a relocation.
    fn append_chunks_of_init_and_uninit_bytes<'ll, 'a, 'b>(
        llvals: &mut Vec<&'ll Value>,
        cx: &'a CodegenCx<'ll, 'b>,
        alloc: &'a Allocation,
        range: Range<usize>,
    ) {
        let mut chunks = alloc
            .init_mask()
            .range_as_init_chunks(Size::from_bytes(range.start), Size::from_bytes(range.end));

        let chunk_to_llval = move |chunk| match chunk {
            InitChunk::Init(range) => {
                let range = (range.start.bytes() as usize)..(range.end.bytes() as usize);
                let bytes = alloc.inspect_with_uninit_and_ptr_outside_interpreter(range);
                cx.const_bytes(bytes)
            }
            InitChunk::Uninit(range) => {
                let len = range.end.bytes() - range.start.bytes();
                cx.const_undef(cx.type_array(cx.type_i8(), len))
            }
        };

        // Generating partially-uninit consts inhibits optimizations, so it is disabled by default.
        // See https://github.com/rust-lang/rust/issues/84565.
        let allow_partially_uninit =
            match cx.sess().opts.debugging_opts.partially_uninit_const_threshold {
                Some(max) => range.len() <= max,
                None => false,
            };

        if allow_partially_uninit {
            llvals.extend(chunks.map(chunk_to_llval));
        } else {
            let llval = match (chunks.next(), chunks.next()) {
                (Some(chunk), None) => {
                    // exactly one chunk, either fully init or fully uninit
                    chunk_to_llval(chunk)
                }
                _ => {
                    // partially uninit, codegen as if it was initialized
                    // (using some arbitrary value for uninit bytes)
                    let bytes = alloc.inspect_with_uninit_and_ptr_outside_interpreter(range);
                    cx.const_bytes(bytes)
                }
            };
            llvals.push(llval);
        }
    }

    let mut next_offset = 0;
    for &(offset, alloc_id) in alloc.relocations().iter() {
        let offset = offset.bytes();
        assert_eq!(offset as usize as u64, offset);
        let offset = offset as usize;
        if offset > next_offset {
            // This `inspect` is okay since we have checked that it is not within a relocation, it
            // is within the bounds of the allocation, and it doesn't affect interpreter execution
            // (we inspect the result after interpreter execution).
            append_chunks_of_init_and_uninit_bytes(&mut llvals, cx, alloc, next_offset..offset);
        }
        let ptr_offset = read_target_uint(
            dl.endian,
            // This `inspect` is okay since it is within the bounds of the allocation, it doesn't
            // affect interpreter execution (we inspect the result after interpreter execution),
            // and we properly interpret the relocation as a relocation pointer offset.
            alloc.inspect_with_uninit_and_ptr_outside_interpreter(offset..(offset + pointer_size)),
        )
        .expect("const_alloc_to_llvm: could not read relocation pointer")
            as u64;

        let address_space = match cx.tcx.global_alloc(alloc_id) {
            GlobalAlloc::Function(..) => cx.data_layout().instruction_address_space,
            GlobalAlloc::Static(..) | GlobalAlloc::Memory(..) => AddressSpace::DATA,
        };

        llvals.push(cx.scalar_to_backend(
            InterpScalar::from_pointer(
                Pointer::new(alloc_id, Size::from_bytes(ptr_offset)),
                &cx.tcx,
            ),
            &Scalar { value: Primitive::Pointer, valid_range: WrappingRange { start: 0, end: !0 } },
            cx.type_i8p_ext(address_space),
        ));
        next_offset = offset + pointer_size;
    }
    if alloc.len() >= next_offset {
        let range = next_offset..alloc.len();
        // This `inspect` is okay since we have check that it is after all relocations, it is
        // within the bounds of the allocation, and it doesn't affect interpreter execution (we
        // inspect the result after interpreter execution).
        append_chunks_of_init_and_uninit_bytes(&mut llvals, cx, alloc, range);
    }

    cx.const_struct(&llvals, true)
}

pub fn codegen_static_initializer(
    cx: &CodegenCx<'ll, 'tcx>,
    def_id: DefId,
) -> Result<(&'ll Value, &'tcx Allocation), ErrorHandled> {
    let alloc = cx.tcx.eval_static_initializer(def_id)?;
    Ok((const_alloc_to_llvm(cx, alloc), alloc))
}

fn set_global_alignment(cx: &CodegenCx<'ll, '_>, gv: &'ll Value, mut align: Align) {
    // The target may require greater alignment for globals than the type does.
    // Note: GCC and Clang also allow `__attribute__((aligned))` on variables,
    // which can force it to be smaller.  Rust doesn't support this yet.
    if let Some(min) = cx.sess().target.min_global_align {
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
    sym: &str,
    span_def_id: DefId,
) -> &'ll Value {
    let llty = cx.layout_of(ty).llvm_type(cx);
    if let Some(linkage) = attrs.linkage {
        debug!("get_static: sym={} linkage={:?}", sym, linkage);

        // If this is a static with a linkage specified, then we need to handle
        // it a little specially. The typesystem prevents things like &T and
        // extern "C" fn() from being non-null, so we can't just declare a
        // static and call it a day. Some linkages (like weak) will make it such
        // that the static actually has a null value.
        let llty2 = if let ty::RawPtr(ref mt) = ty.kind() {
            cx.layout_of(mt.ty).llvm_type(cx)
        } else {
            cx.sess().span_fatal(
                cx.tcx.def_span(span_def_id),
                "must have type `*const T` or `*mut T` due to `#[linkage]` attribute",
            )
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
            let g2 = cx.define_global(&real_name, llty).unwrap_or_else(|| {
                cx.sess().span_fatal(
                    cx.tcx.def_span(span_def_id),
                    &format!("symbol `{}` is already defined", &sym),
                )
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
    unsafe { llvm::LLVMConstPointerCast(val, ty) }
}

impl CodegenCx<'ll, 'tcx> {
    crate fn const_bitcast(&self, val: &'ll Value, ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMConstBitCast(val, ty) }
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
                    let gv = self.define_global(&name[..], self.val_ty(cv)).unwrap_or_else(|| {
                        bug!("symbol `{}` is already defined", name);
                    });
                    llvm::LLVMRustSetLinkage(gv, llvm::Linkage::PrivateLinkage);
                    gv
                }
                _ => self.define_private_global(self.val_ty(cv)),
            };
            llvm::LLVMSetInitializer(gv, cv);
            set_global_alignment(&self, gv, align);
            llvm::SetUnnamedAddress(gv, llvm::UnnamedAddr::Global);
            gv
        }
    }

    crate fn get_static(&self, def_id: DefId) -> &'ll Value {
        let instance = Instance::mono(self.tcx, def_id);
        if let Some(&g) = self.instances.borrow().get(&instance) {
            return g;
        }

        let defined_in_current_codegen_unit =
            self.codegen_unit.items().contains_key(&MonoItem::Static(def_id));
        assert!(
            !defined_in_current_codegen_unit,
            "consts::get_static() should always hit the cache for \
                 statics defined in the same CGU, but did not for `{:?}`",
            def_id
        );

        let ty = instance.ty(self.tcx, ty::ParamEnv::reveal_all());
        let sym = self.tcx.symbol_name(instance).name;
        let fn_attrs = self.tcx.codegen_fn_attrs(def_id);

        debug!("get_static: sym={} instance={:?} fn_attrs={:?}", sym, instance, fn_attrs);

        let g = if def_id.is_local() && !self.tcx.is_foreign_item(def_id) {
            let llty = self.layout_of(ty).llvm_type(self);
            if let Some(g) = self.get_declared_value(sym) {
                if self.val_ty(g) != self.type_ptr_to(llty) {
                    span_bug!(self.tcx.def_span(def_id), "Conflicting types for static");
                }
            }

            let g = self.declare_global(sym, llty);

            if !self.tcx.is_reachable_non_generic(def_id) {
                unsafe {
                    llvm::LLVMRustSetVisibility(g, llvm::Visibility::Hidden);
                }
            }

            g
        } else {
            check_and_apply_linkage(&self, &fn_attrs, ty, sym, def_id)
        };

        // Thread-local statics in some other crate need to *always* be linked
        // against in a thread-local fashion, so we need to be sure to apply the
        // thread-local attribute locally if it was present remotely. If we
        // don't do this then linker errors can be generated where the linker
        // complains that one object files has a thread local version of the
        // symbol and another one doesn't.
        if fn_attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL) {
            llvm::set_thread_local_mode(g, self.tls_model);
        }

        if !def_id.is_local() {
            let needs_dll_storage_attr = self.use_dll_storage_attrs && !self.tcx.is_foreign_item(def_id) &&
                // ThinLTO can't handle this workaround in all cases, so we don't
                // emit the attrs. Instead we make them unnecessary by disallowing
                // dynamic linking when linker plugin based LTO is enabled.
                !self.tcx.sess.opts.cg.linker_plugin_lto.enabled();

            // If this assertion triggers, there's something wrong with commandline
            // argument validation.
            debug_assert!(
                !(self.tcx.sess.opts.cg.linker_plugin_lto.enabled()
                    && self.tcx.sess.target.is_like_windows
                    && self.tcx.sess.opts.cg.prefer_dynamic)
            );

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
        }

        if self.use_dll_storage_attrs && self.tcx.is_dllimport_foreign_item(def_id) {
            // For foreign (native) libs we know the exact storage type to use.
            unsafe {
                llvm::LLVMSetDLLStorageClass(g, llvm::DLLStorageClass::DllImport);
            }
        }

        unsafe {
            if self.should_assume_dso_local(g, true) {
                llvm::LLVMRustSetDSOLocal(g, true);
            }
        }

        self.instances.borrow_mut().insert(instance, g);
        g
    }
}

impl StaticMethods for CodegenCx<'ll, 'tcx> {
    fn static_addr_of(&self, cv: &'ll Value, align: Align, kind: Option<&str>) -> &'ll Value {
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

    fn codegen_static(&self, def_id: DefId, is_mutable: bool) {
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
            let ty = instance.ty(self.tcx, ty::ParamEnv::reveal_all());
            let llty = self.layout_of(ty).llvm_type(self);
            let g = if val_llty == llty {
                g
            } else {
                // If we created the global with the wrong type,
                // correct the type.
                let name = llvm::get_value_name(g).to_vec();
                llvm::set_value_name(g, b"");

                let linkage = llvm::LLVMRustGetLinkage(g);
                let visibility = llvm::LLVMRustGetVisibility(g);

                let new_g = llvm::LLVMRustGetOrInsertGlobal(
                    self.llmod,
                    name.as_ptr().cast(),
                    name.len(),
                    val_llty,
                );

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

            if self.should_assume_dso_local(g, true) {
                llvm::LLVMRustSetDSOLocal(g, true);
            }

            // As an optimization, all shared statics which do not have interior
            // mutability are placed into read-only memory.
            if !is_mutable && self.type_is_freeze(ty) {
                llvm::LLVMSetGlobalConstant(g, llvm::True);
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
                if self.tcx.sess.target.is_like_osx {
                    // The `inspect` method is okay here because we checked relocations, and
                    // because we are doing this access to inspect the final interpreter state
                    // (not as part of the interpreter execution).
                    //
                    // FIXME: This check requires that the (arbitrary) value of undefined bytes
                    // happens to be zero. Instead, we should only check the value of defined bytes
                    // and set all undefined bytes to zero if this allocation is headed for the
                    // BSS.
                    let all_bytes_are_zero = alloc.relocations().is_empty()
                        && alloc
                            .inspect_with_uninit_and_ptr_outside_interpreter(0..alloc.len())
                            .iter()
                            .all(|&byte| byte == 0);

                    let sect_name = if all_bytes_are_zero {
                        cstr!("__DATA,__thread_bss")
                    } else {
                        cstr!("__DATA,__thread_data")
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
                        section.as_str().as_ptr().cast(),
                        section.as_str().len() as c_uint,
                    );
                    assert!(alloc.relocations().is_empty());

                    // The `inspect` method is okay here because we checked relocations, and
                    // because we are doing this access to inspect the final interpreter state (not
                    // as part of the interpreter execution).
                    let bytes =
                        alloc.inspect_with_uninit_and_ptr_outside_interpreter(0..alloc.len());
                    let alloc = llvm::LLVMMDStringInContext(
                        self.llcx,
                        bytes.as_ptr().cast(),
                        bytes.len() as c_uint,
                    );
                    let data = [section, alloc];
                    let meta = llvm::LLVMMDNodeInContext(self.llcx, data.as_ptr(), 2);
                    llvm::LLVMAddNamedMetadataOperand(
                        self.llmod,
                        "wasm.custom_sections\0".as_ptr().cast(),
                        meta,
                    );
                }
            } else {
                base::set_link_section(g, &attrs);
            }

            if attrs.flags.contains(CodegenFnAttrFlags::USED) {
                // The semantics of #[used] in Rust only require the symbol to make it into the
                // object file. It is explicitly allowed for the linker to strip the symbol if it
                // is dead. As such, use llvm.compiler.used instead of llvm.used.
                // Additionally, https://reviews.llvm.org/D97448 in LLVM 13 started emitting unique
                // sections with SHF_GNU_RETAIN flag for llvm.used symbols, which may trigger bugs
                // in some versions of the gold linker.
                self.add_compiler_used_global(g);
            }
        }
    }

    /// Add a global value to a list to be stored in the `llvm.used` variable, an array of i8*.
    fn add_used_global(&self, global: &'ll Value) {
        let cast = unsafe { llvm::LLVMConstPointerCast(global, self.type_i8p()) };
        self.used_statics.borrow_mut().push(cast);
    }

    /// Add a global value to a list to be stored in the `llvm.compiler.used` variable,
    /// an array of i8*.
    fn add_compiler_used_global(&self, global: &'ll Value) {
        let cast = unsafe { llvm::LLVMConstPointerCast(global, self.type_i8p()) };
        self.compiler_used_statics.borrow_mut().push(cast);
    }
}
