use std::ops::Range;

use rustc_abi::{
    Align, AlignFromBytesError, HasDataLayout, Primitive, Scalar, Size, WrappingRange,
};
use rustc_codegen_ssa::common;
use rustc_codegen_ssa::traits::*;
use rustc_hir::LangItem;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::mir::interpret::{
    Allocation, ConstAllocation, ErrorHandled, InitChunk, Pointer, Scalar as InterpScalar,
    read_target_uint,
};
use rustc_middle::mir::mono::{Linkage, MonoItem};
use rustc_middle::ty::layout::{HasTypingEnv, LayoutOf};
use rustc_middle::ty::{self, Instance};
use rustc_middle::{bug, span_bug};
use tracing::{debug, instrument, trace};

use crate::common::{AsCCharPtr, CodegenCx};
use crate::errors::{
    InvalidMinimumAlignmentNotPowerOfTwo, InvalidMinimumAlignmentTooLarge, SymbolAlreadyDefined,
};
use crate::llvm::{self, True};
use crate::type_::Type;
use crate::type_of::LayoutLlvmExt;
use crate::value::Value;
use crate::{base, debuginfo};

pub(crate) fn const_alloc_to_llvm<'ll>(
    cx: &CodegenCx<'ll, '_>,
    alloc: ConstAllocation<'_>,
    is_static: bool,
) -> &'ll Value {
    let alloc = alloc.inner();
    // We expect that callers of const_alloc_to_llvm will instead directly codegen a pointer or
    // integer for any &ZST where the ZST is a constant (i.e. not a static). We should never be
    // producing empty LLVM allocations as they're just adding noise to binaries and forcing less
    // optimal codegen.
    //
    // Statics have a guaranteed meaningful address so it's less clear that we want to do
    // something like this; it's also harder.
    if !is_static {
        assert!(alloc.len() != 0);
    }
    let mut llvals = Vec::with_capacity(alloc.provenance().ptrs().len() + 1);
    let dl = cx.data_layout();
    let pointer_size = dl.pointer_size.bytes() as usize;

    // Note: this function may call `inspect_with_uninit_and_ptr_outside_interpreter`, so `range`
    // must be within the bounds of `alloc` and not contain or overlap a pointer provenance.
    fn append_chunks_of_init_and_uninit_bytes<'ll, 'a, 'b>(
        llvals: &mut Vec<&'ll Value>,
        cx: &'a CodegenCx<'ll, 'b>,
        alloc: &'a Allocation,
        range: Range<usize>,
    ) {
        let chunks = alloc.init_mask().range_as_init_chunks(range.clone().into());

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

        // Generating partially-uninit consts is limited to small numbers of chunks,
        // to avoid the cost of generating large complex const expressions.
        // For example, `[(u32, u8); 1024 * 1024]` contains uninit padding in each element, and
        // would result in `{ [5 x i8] zeroinitializer, [3 x i8] undef, ...repeat 1M times... }`.
        let max = cx.sess().opts.unstable_opts.uninit_const_chunk_threshold;
        let allow_uninit_chunks = chunks.clone().take(max.saturating_add(1)).count() <= max;

        if allow_uninit_chunks {
            llvals.extend(chunks.map(chunk_to_llval));
        } else {
            // If this allocation contains any uninit bytes, codegen as if it was initialized
            // (using some arbitrary value for uninit bytes).
            let bytes = alloc.inspect_with_uninit_and_ptr_outside_interpreter(range);
            llvals.push(cx.const_bytes(bytes));
        }
    }

    let mut next_offset = 0;
    for &(offset, prov) in alloc.provenance().ptrs().iter() {
        let offset = offset.bytes();
        assert_eq!(offset as usize as u64, offset);
        let offset = offset as usize;
        if offset > next_offset {
            // This `inspect` is okay since we have checked that there is no provenance, it
            // is within the bounds of the allocation, and it doesn't affect interpreter execution
            // (we inspect the result after interpreter execution).
            append_chunks_of_init_and_uninit_bytes(&mut llvals, cx, alloc, next_offset..offset);
        }
        let ptr_offset = read_target_uint(
            dl.endian,
            // This `inspect` is okay since it is within the bounds of the allocation, it doesn't
            // affect interpreter execution (we inspect the result after interpreter execution),
            // and we properly interpret the provenance as a relocation pointer offset.
            alloc.inspect_with_uninit_and_ptr_outside_interpreter(offset..(offset + pointer_size)),
        )
        .expect("const_alloc_to_llvm: could not read relocation pointer")
            as u64;

        let address_space = cx.tcx.global_alloc(prov.alloc_id()).address_space(cx);

        llvals.push(cx.scalar_to_backend(
            InterpScalar::from_pointer(Pointer::new(prov, Size::from_bytes(ptr_offset)), &cx.tcx),
            Scalar::Initialized {
                value: Primitive::Pointer(address_space),
                valid_range: WrappingRange::full(dl.pointer_size),
            },
            cx.type_ptr_ext(address_space),
        ));
        next_offset = offset + pointer_size;
    }
    if alloc.len() >= next_offset {
        let range = next_offset..alloc.len();
        // This `inspect` is okay since we have check that it is after all provenance, it is
        // within the bounds of the allocation, and it doesn't affect interpreter execution (we
        // inspect the result after interpreter execution).
        append_chunks_of_init_and_uninit_bytes(&mut llvals, cx, alloc, range);
    }

    // Avoid wrapping in a struct if there is only a single value. This ensures
    // that LLVM is able to perform the string merging optimization if the constant
    // is a valid C string. LLVM only considers bare arrays for this optimization,
    // not arrays wrapped in a struct. LLVM handles this at:
    // https://github.com/rust-lang/llvm-project/blob/acaea3d2bb8f351b740db7ebce7d7a40b9e21488/llvm/lib/Target/TargetLoweringObjectFile.cpp#L249-L280
    if let &[data] = &*llvals { data } else { cx.const_struct(&llvals, true) }
}

fn codegen_static_initializer<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    def_id: DefId,
) -> Result<(&'ll Value, ConstAllocation<'tcx>), ErrorHandled> {
    let alloc = cx.tcx.eval_static_initializer(def_id)?;
    Ok((const_alloc_to_llvm(cx, alloc, /*static*/ true), alloc))
}

fn set_global_alignment<'ll>(cx: &CodegenCx<'ll, '_>, gv: &'ll Value, mut align: Align) {
    // The target may require greater alignment for globals than the type does.
    // Note: GCC and Clang also allow `__attribute__((aligned))` on variables,
    // which can force it to be smaller. Rust doesn't support this yet.
    if let Some(min) = cx.sess().target.min_global_align {
        match Align::from_bits(min) {
            Ok(min) => align = align.max(min),
            Err(err) => match err {
                AlignFromBytesError::NotPowerOfTwo(align) => {
                    cx.sess().dcx().emit_err(InvalidMinimumAlignmentNotPowerOfTwo { align });
                }
                AlignFromBytesError::TooLarge(align) => {
                    cx.sess().dcx().emit_err(InvalidMinimumAlignmentTooLarge { align });
                }
            },
        }
    }
    unsafe {
        llvm::LLVMSetAlignment(gv, align.bytes() as u32);
    }
}

fn check_and_apply_linkage<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    attrs: &CodegenFnAttrs,
    llty: &'ll Type,
    sym: &str,
    def_id: DefId,
) -> &'ll Value {
    if let Some(linkage) = attrs.import_linkage {
        debug!("get_static: sym={} linkage={:?}", sym, linkage);

        // Declare a symbol `foo`. If `foo` is an extern_weak symbol, we declare
        // an extern_weak function, otherwise a global with the desired linkage.
        let g1 = if matches!(attrs.import_linkage, Some(Linkage::ExternalWeak)) {
            // An `extern_weak` function is represented as an `Option<unsafe extern ...>`,
            // we extract the function signature and declare it as an extern_weak function
            // instead of an extern_weak i8.
            let instance = Instance::mono(cx.tcx, def_id);
            if let ty::Adt(struct_def, args) = instance.ty(cx.tcx, cx.typing_env()).kind()
                && cx.tcx.is_lang_item(struct_def.did(), LangItem::Option)
                && let ty::FnPtr(sig, header) = args.type_at(0).kind()
            {
                let fn_sig = sig.with(*header);

                let fn_abi = cx.fn_abi_of_fn_ptr(fn_sig, ty::List::empty());
                cx.declare_fn(sym, &fn_abi, None)
            } else {
                cx.declare_global(sym, cx.type_i8())
            }
        } else {
            cx.declare_global(sym, cx.type_i8())
        };
        llvm::set_linkage(g1, base::linkage_to_llvm(linkage));

        // Declare an internal global `extern_with_linkage_foo` which
        // is initialized with the address of `foo`. If `foo` is
        // discarded during linking (for example, if `foo` has weak
        // linkage and there are no definitions), then
        // `extern_with_linkage_foo` will instead be initialized to
        // zero.
        let mut real_name = "_rust_extern_with_linkage_".to_string();
        real_name.push_str(sym);
        let g2 = cx.define_global(&real_name, llty).unwrap_or_else(|| {
            cx.sess().dcx().emit_fatal(SymbolAlreadyDefined {
                span: cx.tcx.def_span(def_id),
                symbol_name: sym,
            })
        });
        llvm::set_linkage(g2, llvm::Linkage::InternalLinkage);
        llvm::set_initializer(g2, g1);
        g2
    } else if cx.tcx.sess.target.arch == "x86"
        && common::is_mingw_gnu_toolchain(&cx.tcx.sess.target)
        && let Some(dllimport) = crate::common::get_dllimport(cx.tcx, def_id, sym)
    {
        cx.declare_global(&common::i686_decorated_name(dllimport, true, true, false), llty)
    } else {
        // Generate an external declaration.
        // FIXME(nagisa): investigate whether it can be changed into define_global
        cx.declare_global(sym, llty)
    }
}

impl<'ll> CodegenCx<'ll, '_> {
    pub(crate) fn const_bitcast(&self, val: &'ll Value, ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMConstBitCast(val, ty) }
    }

    pub(crate) fn const_pointercast(&self, val: &'ll Value, ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMConstPointerCast(val, ty) }
    }

    /// Create a global variable.
    ///
    /// The returned global variable is a pointer in the default address space for globals.
    /// Fails if a symbol with the given name already exists.
    pub(crate) fn static_addr_of_mut(
        &self,
        cv: &'ll Value,
        align: Align,
        kind: Option<&str>,
    ) -> &'ll Value {
        let gv = match kind {
            Some(kind) if !self.tcx.sess.fewer_names() => {
                let name = self.generate_local_symbol_name(kind);
                let gv = self.define_global(&name, self.val_ty(cv)).unwrap_or_else(|| {
                    bug!("symbol `{}` is already defined", name);
                });
                llvm::set_linkage(gv, llvm::Linkage::PrivateLinkage);
                gv
            }
            _ => self.define_private_global(self.val_ty(cv)),
        };
        llvm::set_initializer(gv, cv);
        set_global_alignment(self, gv, align);
        llvm::SetUnnamedAddress(gv, llvm::UnnamedAddr::Global);
        gv
    }

    /// Create a global constant.
    ///
    /// The returned global variable is a pointer in the default address space for globals.
    pub(crate) fn static_addr_of_impl(
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

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn get_static(&self, def_id: DefId) -> &'ll Value {
        let instance = Instance::mono(self.tcx, def_id);
        trace!(?instance);

        let DefKind::Static { nested, .. } = self.tcx.def_kind(def_id) else { bug!() };
        // Nested statics do not have a type, so pick a dummy type and let `codegen_static` figure
        // out the llvm type from the actual evaluated initializer.
        let llty = if nested {
            self.type_i8()
        } else {
            let ty = instance.ty(self.tcx, self.typing_env());
            trace!(?ty);
            self.layout_of(ty).llvm_type(self)
        };
        self.get_static_inner(def_id, llty)
    }

    #[instrument(level = "debug", skip(self, llty))]
    fn get_static_inner(&self, def_id: DefId, llty: &'ll Type) -> &'ll Value {
        let instance = Instance::mono(self.tcx, def_id);
        if let Some(&g) = self.instances.borrow().get(&instance) {
            trace!("used cached value");
            return g;
        }

        let defined_in_current_codegen_unit =
            self.codegen_unit.items().contains_key(&MonoItem::Static(def_id));
        assert!(
            !defined_in_current_codegen_unit,
            "consts::get_static() should always hit the cache for \
                 statics defined in the same CGU, but did not for `{def_id:?}`"
        );

        let sym = self.tcx.symbol_name(instance).name;
        let fn_attrs = self.tcx.codegen_fn_attrs(def_id);

        debug!(?sym, ?fn_attrs);

        let g = if def_id.is_local() && !self.tcx.is_foreign_item(def_id) {
            if let Some(g) = self.get_declared_value(sym) {
                if self.val_ty(g) != self.type_ptr() {
                    span_bug!(self.tcx.def_span(def_id), "Conflicting types for static");
                }
            }

            let g = self.declare_global(sym, llty);

            if !self.tcx.is_reachable_non_generic(def_id) {
                llvm::set_visibility(g, llvm::Visibility::Hidden);
            }

            g
        } else {
            check_and_apply_linkage(self, fn_attrs, llty, sym, def_id)
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

        let dso_local = self.assume_dso_local(g, true);

        if !def_id.is_local() {
            let needs_dll_storage_attr = self.use_dll_storage_attrs
                && !self.tcx.is_foreign_item(def_id)
                // Local definitions can never be imported, so we must not apply
                // the DLLImport annotation.
                && !dso_local
                // Linker plugin ThinLTO doesn't create the self-dllimport Rust uses for rlibs
                // as the code generation happens out of process. Instead we assume static linkage
                // and disallow dynamic linking when linker plugin based LTO is enabled.
                // Regular in-process ThinLTO doesn't need this workaround.
                && !self.tcx.sess.opts.cg.linker_plugin_lto.enabled();

            // If this assertion triggers, there's something wrong with commandline
            // argument validation.
            assert!(
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
                    llvm::set_dllimport_storage_class(g);
                }
            }
        }

        if self.use_dll_storage_attrs
            && let Some(library) = self.tcx.native_library(def_id)
            && library.kind.is_dllimport()
        {
            // For foreign (native) libs we know the exact storage type to use.
            llvm::set_dllimport_storage_class(g);
        }

        self.instances.borrow_mut().insert(instance, g);
        g
    }

    fn codegen_static_item(&mut self, def_id: DefId) {
        unsafe {
            assert!(
                llvm::LLVMGetInitializer(
                    self.instances.borrow().get(&Instance::mono(self.tcx, def_id)).unwrap()
                )
                .is_none()
            );
            let attrs = self.tcx.codegen_fn_attrs(def_id);

            let Ok((v, alloc)) = codegen_static_initializer(self, def_id) else {
                // Error has already been reported
                return;
            };
            let alloc = alloc.inner();

            let val_llty = self.val_ty(v);

            let g = self.get_static_inner(def_id, val_llty);
            let llty = self.get_type_of_global(g);

            let g = if val_llty == llty {
                g
            } else {
                // codegen_static_initializer creates the global value just from the
                // `Allocation` data by generating one big struct value that is just
                // all the bytes and pointers after each other. This will almost never
                // match the type that the static was declared with. Unfortunately
                // we can't just LLVMConstBitCast our way out of it because that has very
                // specific rules on what can be cast. So instead of adding a new way to
                // generate static initializers that match the static's type, we picked
                // the easier option and retroactively change the type of the static item itself.
                let name = llvm::get_value_name(g).to_vec();
                llvm::set_value_name(g, b"");

                let linkage = llvm::get_linkage(g);
                let visibility = llvm::get_visibility(g);

                let new_g = llvm::LLVMRustGetOrInsertGlobal(
                    self.llmod,
                    name.as_c_char_ptr(),
                    name.len(),
                    val_llty,
                );

                llvm::set_linkage(new_g, linkage);
                llvm::set_visibility(new_g, visibility);

                // The old global has had its name removed but is returned by
                // get_static since it is in the instance cache. Provide an
                // alternative lookup that points to the new global so that
                // global_asm! can compute the correct mangled symbol name
                // for the global.
                self.renamed_statics.borrow_mut().insert(def_id, new_g);

                // To avoid breaking any invariants, we leave around the old
                // global for the moment; we'll replace all references to it
                // with the new global later. (See base::codegen_backend.)
                self.statics_to_rauw.borrow_mut().push((g, new_g));
                new_g
            };
            set_global_alignment(self, g, alloc.align);
            llvm::set_initializer(g, v);

            self.assume_dso_local(g, true);

            // Forward the allocation's mutability (picked by the const interner) to LLVM.
            if alloc.mutability.is_not() {
                llvm::LLVMSetGlobalConstant(g, llvm::True);
            }

            debuginfo::build_global_var_di_node(self, def_id, g);

            if attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL) {
                llvm::set_thread_local_mode(g, self.tls_model);
            }

            // Wasm statics with custom link sections get special treatment as they
            // go into custom sections of the wasm executable. The exception to this
            // is the `.init_array` section which are treated specially by the wasm linker.
            if self.tcx.sess.target.is_like_wasm
                && attrs
                    .link_section
                    .map(|link_section| !link_section.as_str().starts_with(".init_array"))
                    .unwrap_or(true)
            {
                if let Some(section) = attrs.link_section {
                    let section = llvm::LLVMMDStringInContext2(
                        self.llcx,
                        section.as_str().as_c_char_ptr(),
                        section.as_str().len(),
                    );
                    assert!(alloc.provenance().ptrs().is_empty());

                    // The `inspect` method is okay here because we checked for provenance, and
                    // because we are doing this access to inspect the final interpreter state (not
                    // as part of the interpreter execution).
                    let bytes =
                        alloc.inspect_with_uninit_and_ptr_outside_interpreter(0..alloc.len());
                    let alloc =
                        llvm::LLVMMDStringInContext2(self.llcx, bytes.as_c_char_ptr(), bytes.len());
                    let data = [section, alloc];
                    let meta = llvm::LLVMMDNodeInContext2(self.llcx, data.as_ptr(), data.len());
                    let val = self.get_metadata_value(meta);
                    llvm::LLVMAddNamedMetadataOperand(
                        self.llmod,
                        c"wasm.custom_sections".as_ptr(),
                        val,
                    );
                }
            } else {
                base::set_link_section(g, attrs);
            }

            base::set_variable_sanitizer_attrs(g, attrs);

            if attrs.flags.contains(CodegenFnAttrFlags::USED_COMPILER) {
                // `USED` and `USED_LINKER` can't be used together.
                assert!(!attrs.flags.contains(CodegenFnAttrFlags::USED_LINKER));

                // The semantics of #[used] in Rust only require the symbol to make it into the
                // object file. It is explicitly allowed for the linker to strip the symbol if it
                // is dead, which means we are allowed to use `llvm.compiler.used` instead of
                // `llvm.used` here.
                //
                // Additionally, https://reviews.llvm.org/D97448 in LLVM 13 started emitting unique
                // sections with SHF_GNU_RETAIN flag for llvm.used symbols, which may trigger bugs
                // in the handling of `.init_array` (the static constructor list) in versions of
                // the gold linker (prior to the one released with binutils 2.36).
                //
                // That said, we only ever emit these when compiling for ELF targets, unless
                // `#[used(compiler)]` is explicitly requested. This is to avoid similar breakage
                // on other targets, in particular MachO targets have *their* static constructor
                // lists broken if `llvm.compiler.used` is emitted rather than `llvm.used`. However,
                // that check happens when assigning the `CodegenFnAttrFlags` in
                // `rustc_hir_analysis`, so we don't need to take care of it here.
                self.add_compiler_used_global(g);
            }
            if attrs.flags.contains(CodegenFnAttrFlags::USED_LINKER) {
                // `USED` and `USED_LINKER` can't be used together.
                assert!(!attrs.flags.contains(CodegenFnAttrFlags::USED_COMPILER));

                self.add_used_global(g);
            }
        }
    }

    /// Add a global value to a list to be stored in the `llvm.used` variable, an array of ptr.
    pub(crate) fn add_used_global(&mut self, global: &'ll Value) {
        self.used_statics.push(global);
    }

    /// Add a global value to a list to be stored in the `llvm.compiler.used` variable,
    /// an array of ptr.
    pub(crate) fn add_compiler_used_global(&mut self, global: &'ll Value) {
        self.compiler_used_statics.push(global);
    }
}

impl<'ll> StaticCodegenMethods for CodegenCx<'ll, '_> {
    /// Get a pointer to a global variable.
    ///
    /// The pointer will always be in the default address space. If global variables default to a
    /// different address space, an addrspacecast is inserted.
    fn static_addr_of(&self, cv: &'ll Value, align: Align, kind: Option<&str>) -> &'ll Value {
        let gv = self.static_addr_of_impl(cv, align, kind);
        // static_addr_of_impl returns the bare global variable, which might not be in the default
        // address space. Cast to the default address space if necessary.
        self.const_pointercast(gv, self.type_ptr())
    }

    fn codegen_static(&mut self, def_id: DefId) {
        self.codegen_static_item(def_id)
    }
}
