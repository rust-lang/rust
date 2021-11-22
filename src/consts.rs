use gccjit::{LValue, RValue, ToRValue, Type};
use rustc_codegen_ssa::traits::{BaseTypeMethods, ConstMethods, DerivedTypeMethods, StaticMethods};
use rustc_hir as hir;
use rustc_hir::Node;
use rustc_middle::{bug, span_bug};
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::ty::{self, Instance, Ty};
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::mir::interpret::{self, Allocation, ErrorHandled, Scalar as InterpScalar, read_target_uint};
use rustc_span::Span;
use rustc_span::def_id::DefId;
use rustc_target::abi::{self, Align, HasDataLayout, Primitive, Size, WrappingRange};

use crate::base;
use crate::context::CodegenCx;
use crate::type_of::LayoutGccExt;

impl<'gcc, 'tcx> CodegenCx<'gcc, 'tcx> {
    pub fn const_bitcast(&self, value: RValue<'gcc>, typ: Type<'gcc>) -> RValue<'gcc> {
        if value.get_type() == self.bool_type.make_pointer() {
            if let Some(pointee) = typ.get_pointee() {
                if pointee.dyncast_vector().is_some() {
                    panic!()
                }
            }
        }
        self.context.new_bitcast(None, value, typ)
    }
}

impl<'gcc, 'tcx> StaticMethods for CodegenCx<'gcc, 'tcx> {
    fn static_addr_of(&self, cv: RValue<'gcc>, align: Align, kind: Option<&str>) -> RValue<'gcc> {
        // TODO(antoyo): implement a proper rvalue comparison in libgccjit instead of doing the
        // following:
        for (value, variable) in &*self.const_globals.borrow() {
            if format!("{:?}", value) == format!("{:?}", cv) {
                // TODO(antoyo): upgrade alignment.
                return *variable;
            }
        }
        let global_value = self.static_addr_of_mut(cv, align, kind);
        // TODO(antoyo): set global constant.
        self.const_globals.borrow_mut().insert(cv, global_value);
        global_value
    }

    fn codegen_static(&self, def_id: DefId, is_mutable: bool) {
        let attrs = self.tcx.codegen_fn_attrs(def_id);

        let value =
            match codegen_static_initializer(&self, def_id) {
                Ok((value, _)) => value,
                // Error has already been reported
                Err(_) => return,
            };

        let global = self.get_static(def_id);

        // boolean SSA values are i1, but they have to be stored in i8 slots,
        // otherwise some LLVM optimization passes don't work as expected
        let val_llty = self.val_ty(value);
        let value =
            if val_llty == self.type_i1() {
                unimplemented!();
            }
            else {
                value
            };

        let instance = Instance::mono(self.tcx, def_id);
        let ty = instance.ty(self.tcx, ty::ParamEnv::reveal_all());
        let gcc_type = self.layout_of(ty).gcc_type(self, true);

        // TODO(antoyo): set alignment.

        let value =
            if value.get_type() != gcc_type {
                self.context.new_bitcast(None, value, gcc_type)
            }
            else {
                value
            };
        global.global_set_initializer_rvalue(value);

        // As an optimization, all shared statics which do not have interior
        // mutability are placed into read-only memory.
        if !is_mutable {
            if self.type_is_freeze(ty) {
                // TODO(antoyo): set global constant.
            }
        }

        if attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL) {
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
            if self.tcx.sess.target.options.is_like_osx {
                // The `inspect` method is okay here because we checked relocations, and
                // because we are doing this access to inspect the final interpreter state
                // (not as part of the interpreter execution).
                //
                // FIXME: This check requires that the (arbitrary) value of undefined bytes
                // happens to be zero. Instead, we should only check the value of defined bytes
                // and set all undefined bytes to zero if this allocation is headed for the
                // BSS.
                unimplemented!();
            }
        }

        // Wasm statics with custom link sections get special treatment as they
        // go into custom sections of the wasm executable.
        if self.tcx.sess.opts.target_triple.triple().starts_with("wasm32") {
            if let Some(_section) = attrs.link_section {
                unimplemented!();
            }
        } else {
            // TODO(antoyo): set link section.
        }

        if attrs.flags.contains(CodegenFnAttrFlags::USED) || attrs.flags.contains(CodegenFnAttrFlags::USED_LINKER) {
            self.add_used_global(global.to_rvalue());
        }
    }

    /// Add a global value to a list to be stored in the `llvm.used` variable, an array of i8*.
    fn add_used_global(&self, _global: RValue<'gcc>) {
        // TODO(antoyo)
    }

    fn add_compiler_used_global(&self, _global: RValue<'gcc>) {
        // TODO(antoyo)
    }
}

impl<'gcc, 'tcx> CodegenCx<'gcc, 'tcx> {
    pub fn static_addr_of_mut(&self, cv: RValue<'gcc>, align: Align, kind: Option<&str>) -> RValue<'gcc> {
        let global =
            match kind {
                Some(kind) if !self.tcx.sess.fewer_names() => {
                    let name = self.generate_local_symbol_name(kind);
                    // TODO(antoyo): check if it's okay that TLS is off here.
                    // TODO(antoyo): check if it's okay that link_section is None here.
                    // TODO(antoyo): set alignment here as well.
                    let global = self.define_global(&name[..], self.val_ty(cv), false, None);
                    // TODO(antoyo): set linkage.
                    global
                }
                _ => {
                    let typ = self.val_ty(cv).get_aligned(align.bytes());
                    let global = self.declare_unnamed_global(typ);
                    global
                },
            };
        // FIXME(antoyo): I think the name coming from generate_local_symbol_name() above cannot be used
        // globally.
        global.global_set_initializer_rvalue(cv);
        // TODO(antoyo): set unnamed address.
        global.get_address(None)
    }

    pub fn get_static(&self, def_id: DefId) -> LValue<'gcc> {
        let instance = Instance::mono(self.tcx, def_id);
        let fn_attrs = self.tcx.codegen_fn_attrs(def_id);
        if let Some(&global) = self.instances.borrow().get(&instance) {
            return global;
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

        let global =
            if let Some(def_id) = def_id.as_local() {
                let id = self.tcx.hir().local_def_id_to_hir_id(def_id);
                let llty = self.layout_of(ty).gcc_type(self, true);
                // FIXME: refactor this to work without accessing the HIR
                let global = match self.tcx.hir().get(id) {
                    Node::Item(&hir::Item { span, kind: hir::ItemKind::Static(..), .. }) => {
                        if let Some(global) = self.get_declared_value(&sym) {
                            if self.val_ty(global) != self.type_ptr_to(llty) {
                                span_bug!(span, "Conflicting types for static");
                            }
                        }

                        let is_tls = fn_attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL);
                        let global = self.declare_global(&sym, llty, is_tls, fn_attrs.link_section);

                        if !self.tcx.is_reachable_non_generic(def_id) {
                            // TODO(antoyo): set visibility.
                        }

                        global
                    }

                    Node::ForeignItem(&hir::ForeignItem {
                        span,
                        kind: hir::ForeignItemKind::Static(..),
                        ..
                    }) => {
                        let fn_attrs = self.tcx.codegen_fn_attrs(def_id);
                        check_and_apply_linkage(&self, &fn_attrs, ty, sym, span)
                    }

                    item => bug!("get_static: expected static, found {:?}", item),
                };

                global
            }
            else {
                // FIXME(nagisa): perhaps the map of externs could be offloaded to llvm somehow?
                //debug!("get_static: sym={} item_attr={:?}", sym, self.tcx.item_attrs(def_id));

                let attrs = self.tcx.codegen_fn_attrs(def_id);
                let span = self.tcx.def_span(def_id);
                let global = check_and_apply_linkage(&self, &attrs, ty, sym, span);

                let needs_dll_storage_attr = false; // TODO(antoyo)

                // If this assertion triggers, there's something wrong with commandline
                // argument validation.
                debug_assert!(
                    !(self.tcx.sess.opts.cg.linker_plugin_lto.enabled()
                        && self.tcx.sess.target.options.is_like_msvc
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
                        unimplemented!();
                    }
                }
                global
            };

        // TODO(antoyo): set dll storage class.

        self.instances.borrow_mut().insert(instance, global);
        global
    }
}

pub fn const_alloc_to_gcc<'gcc, 'tcx>(cx: &CodegenCx<'gcc, 'tcx>, alloc: &Allocation) -> RValue<'gcc> {
    let mut llvals = Vec::with_capacity(alloc.relocations().len() + 1);
    let dl = cx.data_layout();
    let pointer_size = dl.pointer_size.bytes() as usize;

    let mut next_offset = 0;
    for &(offset, alloc_id) in alloc.relocations().iter() {
        let offset = offset.bytes();
        assert_eq!(offset as usize as u64, offset);
        let offset = offset as usize;
        if offset > next_offset {
            // This `inspect` is okay since we have checked that it is not within a relocation, it
            // is within the bounds of the allocation, and it doesn't affect interpreter execution
            // (we inspect the result after interpreter execution). Any undef byte is replaced with
            // some arbitrary byte value.
            //
            // FIXME: relay undef bytes to codegen as undef const bytes
            let bytes = alloc.inspect_with_uninit_and_ptr_outside_interpreter(next_offset..offset);
            llvals.push(cx.const_bytes(bytes));
        }
        let ptr_offset =
            read_target_uint( dl.endian,
                // This `inspect` is okay since it is within the bounds of the allocation, it doesn't
                // affect interpreter execution (we inspect the result after interpreter execution),
                // and we properly interpret the relocation as a relocation pointer offset.
                alloc.inspect_with_uninit_and_ptr_outside_interpreter(offset..(offset + pointer_size)),
            )
            .expect("const_alloc_to_llvm: could not read relocation pointer")
            as u64;
        llvals.push(cx.scalar_to_backend(
            InterpScalar::from_pointer(
                interpret::Pointer::new(alloc_id, Size::from_bytes(ptr_offset)),
                &cx.tcx,
            ),
            abi::Scalar { value: Primitive::Pointer, valid_range: WrappingRange { start: 0, end: !0 } },
            cx.type_i8p(),
        ));
        next_offset = offset + pointer_size;
    }
    if alloc.len() >= next_offset {
        let range = next_offset..alloc.len();
        // This `inspect` is okay since we have check that it is after all relocations, it is
        // within the bounds of the allocation, and it doesn't affect interpreter execution (we
        // inspect the result after interpreter execution). Any undef byte is replaced with some
        // arbitrary byte value.
        //
        // FIXME: relay undef bytes to codegen as undef const bytes
        let bytes = alloc.inspect_with_uninit_and_ptr_outside_interpreter(range);
        llvals.push(cx.const_bytes(bytes));
    }

    cx.const_struct(&llvals, true)
}

pub fn codegen_static_initializer<'gcc, 'tcx>(cx: &CodegenCx<'gcc, 'tcx>, def_id: DefId) -> Result<(RValue<'gcc>, &'tcx Allocation), ErrorHandled> {
    let alloc = cx.tcx.eval_static_initializer(def_id)?;
    Ok((const_alloc_to_gcc(cx, alloc), alloc))
}

fn check_and_apply_linkage<'gcc, 'tcx>(cx: &CodegenCx<'gcc, 'tcx>, attrs: &CodegenFnAttrs, ty: Ty<'tcx>, sym: &str, span: Span) -> LValue<'gcc> {
    let is_tls = attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL);
    let llty = cx.layout_of(ty).gcc_type(cx, true);
    if let Some(linkage) = attrs.linkage {
        // If this is a static with a linkage specified, then we need to handle
        // it a little specially. The typesystem prevents things like &T and
        // extern "C" fn() from being non-null, so we can't just declare a
        // static and call it a day. Some linkages (like weak) will make it such
        // that the static actually has a null value.
        let llty2 =
            if let ty::RawPtr(ref mt) = ty.kind() {
                cx.layout_of(mt.ty).gcc_type(cx, true)
            }
            else {
                cx.sess().span_fatal(
                    span,
                    "must have type `*const T` or `*mut T` due to `#[linkage]` attribute",
                )
            };
        // Declare a symbol `foo` with the desired linkage.
        let global1 = cx.declare_global_with_linkage(&sym, llty2, base::global_linkage_to_gcc(linkage));

        // Declare an internal global `extern_with_linkage_foo` which
        // is initialized with the address of `foo`.  If `foo` is
        // discarded during linking (for example, if `foo` has weak
        // linkage and there are no definitions), then
        // `extern_with_linkage_foo` will instead be initialized to
        // zero.
        let mut real_name = "_rust_extern_with_linkage_".to_string();
        real_name.push_str(&sym);
        let global2 = cx.define_global(&real_name, llty, is_tls, attrs.link_section);
        // TODO(antoyo): set linkage.
        global2.global_set_initializer_rvalue(global1.get_address(None));
        // TODO(antoyo): use global_set_initializer() when it will work.
        global2
    }
    else {
        // Generate an external declaration.
        // FIXME(nagisa): investigate whether it can be changed into define_global

        // Thread-local statics in some other crate need to *always* be linked
        // against in a thread-local fashion, so we need to be sure to apply the
        // thread-local attribute locally if it was present remotely. If we
        // don't do this then linker errors can be generated where the linker
        // complains that one object files has a thread local version of the
        // symbol and another one doesn't.
        cx.declare_global(&sym, llty, is_tls, attrs.link_section)
    }
}
