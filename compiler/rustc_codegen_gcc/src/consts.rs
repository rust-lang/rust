#[cfg(feature = "master")]
use gccjit::{FnAttribute, VarAttribute, Visibility};
use gccjit::{Function, GlobalKind, LValue, RValue, ToRValue, Type};
use rustc_abi::{self as abi, Align, HasDataLayout, Primitive, Size, WrappingRange};
use rustc_codegen_ssa::traits::{
    BaseTypeCodegenMethods, ConstCodegenMethods, StaticCodegenMethods,
};
use rustc_hir::def::DefKind;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::mir::interpret::{
    self, ConstAllocation, ErrorHandled, Scalar as InterpScalar, read_target_uint,
};
use rustc_middle::mir::mono::Linkage;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, Instance};
use rustc_middle::{bug, span_bug};
use rustc_span::def_id::DefId;

use crate::base;
use crate::context::CodegenCx;
use crate::type_of::LayoutGccExt;

fn set_global_alignment<'gcc, 'tcx>(
    cx: &CodegenCx<'gcc, 'tcx>,
    gv: LValue<'gcc>,
    mut align: Align,
) {
    // The target may require greater alignment for globals than the type does.
    // Note: GCC and Clang also allow `__attribute__((aligned))` on variables,
    // which can force it to be smaller. Rust doesn't support this yet.
    if let Some(min_global) = cx.sess().target.min_global_align {
        align = Ord::max(align, min_global);
    }
    gv.set_alignment(align.bytes() as i32);
}

impl<'gcc, 'tcx> StaticCodegenMethods for CodegenCx<'gcc, 'tcx> {
    fn static_addr_of(&self, cv: RValue<'gcc>, align: Align, kind: Option<&str>) -> RValue<'gcc> {
        // TODO(antoyo): implement a proper rvalue comparison in libgccjit instead of doing the
        // following:
        for (value, variable) in &*self.const_globals.borrow() {
            if format!("{:?}", value) == format!("{:?}", cv) {
                if let Some(global_variable) = self.global_lvalues.borrow().get(variable) {
                    let alignment = align.bits() as i32;
                    if alignment > global_variable.get_alignment() {
                        global_variable.set_alignment(alignment);
                    }
                }
                return *variable;
            }
        }
        let global_value = self.static_addr_of_mut(cv, align, kind);
        #[cfg(feature = "master")]
        self.global_lvalues
            .borrow()
            .get(&global_value)
            .expect("`static_addr_of_mut` did not add the global to `self.global_lvalues`")
            .global_set_readonly();
        self.const_globals.borrow_mut().insert(cv, global_value);
        global_value
    }

    #[cfg_attr(not(feature = "master"), allow(unused_mut))]
    fn codegen_static(&mut self, def_id: DefId) {
        let attrs = self.tcx.codegen_fn_attrs(def_id);

        let Ok((value, alloc)) = codegen_static_initializer(self, def_id) else {
            // Error has already been reported
            return;
        };
        let alloc = alloc.inner();

        // boolean SSA values are i1, but they have to be stored in i8 slots,
        // otherwise some LLVM optimization passes don't work as expected
        let val_llty = self.val_ty(value);
        if val_llty == self.type_i1() {
            unimplemented!();
        };

        let is_thread_local = attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL);
        let global = self.get_static_inner(def_id, val_llty);

        #[cfg(feature = "master")]
        if global.to_rvalue().get_type() != val_llty {
            global.to_rvalue().set_type(val_llty);
        }
        set_global_alignment(self, global, alloc.align);

        global.global_set_initializer_rvalue(value);

        // As an optimization, all shared statics which do not have interior
        // mutability are placed into read-only memory.
        if alloc.mutability.is_not() {
            #[cfg(feature = "master")]
            global.global_set_readonly();
        }

        if is_thread_local {
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
            if self.tcx.sess.target.options.is_like_darwin {
                // The `inspect` method is okay here because we checked for provenance, and
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
        if self.tcx.sess.target.is_like_wasm {
            if let Some(_section) = attrs.link_section {
                unimplemented!();
            }
        } else {
            // TODO(antoyo): set link section.
        }

        if attrs.flags.contains(CodegenFnAttrFlags::USED_COMPILER)
            || attrs.flags.contains(CodegenFnAttrFlags::USED_LINKER)
        {
            self.add_used_global(global.to_rvalue());
        }
    }
}

impl<'gcc, 'tcx> CodegenCx<'gcc, 'tcx> {
    /// Add a global value to a list to be stored in the `llvm.used` variable, an array of i8*.
    pub fn add_used_global(&mut self, _global: RValue<'gcc>) {
        // TODO(antoyo)
    }

    #[cfg_attr(not(feature = "master"), allow(unused_variables))]
    pub fn add_used_function(&self, function: Function<'gcc>) {
        #[cfg(feature = "master")]
        function.add_attribute(FnAttribute::Used);
    }

    pub fn static_addr_of_mut(
        &self,
        cv: RValue<'gcc>,
        align: Align,
        kind: Option<&str>,
    ) -> RValue<'gcc> {
        let global = match kind {
            Some(kind) if !self.tcx.sess.fewer_names() => {
                let name = self.generate_local_symbol_name(kind);
                // TODO(antoyo): check if it's okay that no link_section is set.

                let typ = self.val_ty(cv).get_aligned(align.bytes());
                self.declare_private_global(&name[..], typ)
            }
            _ => {
                let typ = self.val_ty(cv).get_aligned(align.bytes());
                self.declare_unnamed_global(typ)
            }
        };
        global.global_set_initializer_rvalue(cv);
        // TODO(antoyo): set unnamed address.
        let rvalue = global.get_address(None);
        self.global_lvalues.borrow_mut().insert(rvalue, global);
        rvalue
    }

    pub fn get_static(&self, def_id: DefId) -> LValue<'gcc> {
        let instance = Instance::mono(self.tcx, def_id);
        let DefKind::Static { nested, .. } = self.tcx.def_kind(def_id) else { bug!() };
        // Nested statics do not have a type, so pick a random type and let `define_static` figure out
        // the gcc type from the actual evaluated initializer.
        let gcc_type = if nested {
            self.type_i8()
        } else {
            let ty = instance.ty(self.tcx, ty::TypingEnv::fully_monomorphized());
            self.layout_of(ty).gcc_type(self)
        };

        self.get_static_inner(def_id, gcc_type)
    }

    pub(crate) fn get_static_inner(&self, def_id: DefId, gcc_type: Type<'gcc>) -> LValue<'gcc> {
        let instance = Instance::mono(self.tcx, def_id);
        if let Some(&global) = self.instances.borrow().get(&instance) {
            trace!("used cached value");
            return global;
        }

        // FIXME: Once we stop removing globals in `codegen_static`, we can uncomment this code.
        // let defined_in_current_codegen_unit =
        //     self.codegen_unit.items().contains_key(&MonoItem::Static(def_id));
        // assert!(
        //     !defined_in_current_codegen_unit,
        //     "consts::get_static() should always hit the cache for \
        //          statics defined in the same CGU, but did not for `{:?}`",
        //     def_id
        // );
        let sym = self.tcx.symbol_name(instance).name;
        let fn_attrs = self.tcx.codegen_fn_attrs(def_id);

        let global = if def_id.is_local() && !self.tcx.is_foreign_item(def_id) {
            if let Some(global) = self.get_declared_value(sym)
                && self.val_ty(global) != self.type_ptr_to(gcc_type)
            {
                span_bug!(self.tcx.def_span(def_id), "Conflicting types for static");
            }

            let is_tls = fn_attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL);
            let global = self.declare_global(
                sym,
                gcc_type,
                GlobalKind::Imported,
                is_tls,
                fn_attrs.link_section,
            );

            if !self.tcx.is_reachable_non_generic(def_id) {
                #[cfg(feature = "master")]
                global.add_attribute(VarAttribute::Visibility(Visibility::Hidden));
            }

            global
        } else {
            check_and_apply_linkage(self, fn_attrs, gcc_type, sym)
        };

        if !def_id.is_local() {
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
        }

        // TODO(antoyo): set dll storage class.

        self.instances.borrow_mut().insert(instance, global);
        global
    }
}

pub fn const_alloc_to_gcc<'gcc>(
    cx: &CodegenCx<'gcc, '_>,
    alloc: ConstAllocation<'_>,
) -> RValue<'gcc> {
    let alloc = alloc.inner();
    let mut llvals = Vec::with_capacity(alloc.provenance().ptrs().len() + 1);
    let dl = cx.data_layout();
    let pointer_size = dl.pointer_size.bytes() as usize;

    let mut next_offset = 0;
    for &(offset, prov) in alloc.provenance().ptrs().iter() {
        let alloc_id = prov.alloc_id();
        let offset = offset.bytes();
        assert_eq!(offset as usize as u64, offset);
        let offset = offset as usize;
        if offset > next_offset {
            // This `inspect` is okay since we have checked that it is not within a pointer with provenance, it
            // is within the bounds of the allocation, and it doesn't affect interpreter execution
            // (we inspect the result after interpreter execution). Any undef byte is replaced with
            // some arbitrary byte value.
            //
            // FIXME: relay undef bytes to codegen as undef const bytes
            let bytes = alloc.inspect_with_uninit_and_ptr_outside_interpreter(next_offset..offset);
            llvals.push(cx.const_bytes(bytes));
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

        let address_space = cx.tcx.global_alloc(alloc_id).address_space(cx);

        llvals.push(cx.scalar_to_backend(
            InterpScalar::from_pointer(
                interpret::Pointer::new(prov, Size::from_bytes(ptr_offset)),
                &cx.tcx,
            ),
            abi::Scalar::Initialized {
                value: Primitive::Pointer(address_space),
                valid_range: WrappingRange::full(dl.pointer_size),
            },
            cx.type_i8p_ext(address_space),
        ));
        next_offset = offset + pointer_size;
    }
    if alloc.len() >= next_offset {
        let range = next_offset..alloc.len();
        // This `inspect` is okay since we have check that it is after all provenance, it is
        // within the bounds of the allocation, and it doesn't affect interpreter execution (we
        // inspect the result after interpreter execution). Any undef byte is replaced with some
        // arbitrary byte value.
        //
        // FIXME: relay undef bytes to codegen as undef const bytes
        let bytes = alloc.inspect_with_uninit_and_ptr_outside_interpreter(range);
        llvals.push(cx.const_bytes(bytes));
    }

    // FIXME(bjorn3) avoid wrapping in a struct when there is only a single element.
    cx.const_struct(&llvals, true)
}

fn codegen_static_initializer<'gcc, 'tcx>(
    cx: &CodegenCx<'gcc, 'tcx>,
    def_id: DefId,
) -> Result<(RValue<'gcc>, ConstAllocation<'tcx>), ErrorHandled> {
    let alloc = cx.tcx.eval_static_initializer(def_id)?;
    Ok((const_alloc_to_gcc(cx, alloc), alloc))
}

fn check_and_apply_linkage<'gcc, 'tcx>(
    cx: &CodegenCx<'gcc, 'tcx>,
    attrs: &CodegenFnAttrs,
    gcc_type: Type<'gcc>,
    sym: &str,
) -> LValue<'gcc> {
    let is_tls = attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL);
    if let Some(linkage) = attrs.import_linkage {
        // Declare a symbol `foo` with the desired linkage.
        let global1 =
            cx.declare_global_with_linkage(sym, cx.type_i8(), base::global_linkage_to_gcc(linkage));

        if linkage == Linkage::ExternalWeak {
            #[cfg(feature = "master")]
            global1.add_attribute(VarAttribute::Weak);
        }

        // Declare an internal global `extern_with_linkage_foo` which
        // is initialized with the address of `foo`.  If `foo` is
        // discarded during linking (for example, if `foo` has weak
        // linkage and there are no definitions), then
        // `extern_with_linkage_foo` will instead be initialized to
        // zero.
        let mut real_name = "_rust_extern_with_linkage_".to_string();
        real_name.push_str(sym);
        let global2 = cx.define_global(&real_name, gcc_type, is_tls, attrs.link_section);
        // TODO(antoyo): set linkage.
        let value = cx.const_ptrcast(global1.get_address(None), gcc_type);
        global2.global_set_initializer_rvalue(value);
        global2
    } else {
        // Generate an external declaration.
        // FIXME(nagisa): investigate whether it can be changed into define_global

        // Thread-local statics in some other crate need to *always* be linked
        // against in a thread-local fashion, so we need to be sure to apply the
        // thread-local attribute locally if it was present remotely. If we
        // don't do this then linker errors can be generated where the linker
        // complains that one object files has a thread local version of the
        // symbol and another one doesn't.
        cx.declare_global(sym, gcc_type, GlobalKind::Imported, is_tls, attrs.link_section)
    }
}
