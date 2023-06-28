use std::{collections::hash_map::Entry, io::Write, iter, path::Path};

use log::trace;

use rustc_apfloat::Float;
use rustc_ast::expand::allocator::AllocatorKind;
use rustc_hir::{
    def::DefKind,
    def_id::{CrateNum, DefId, LOCAL_CRATE},
};
use rustc_middle::middle::{
    codegen_fn_attrs::CodegenFnAttrFlags, dependency_format::Linkage,
    exported_symbols::ExportedSymbol,
};
use rustc_middle::mir;
use rustc_middle::ty;
use rustc_session::config::CrateType;
use rustc_span::Symbol;
use rustc_target::{
    abi::{Align, Size},
    spec::abi::Abi,
};

use super::backtrace::EvalContextExt as _;
use crate::helpers::{convert::Truncate, target_os_is_unix};
use crate::*;

/// Returned by `emulate_foreign_item_by_name`.
pub enum EmulateByNameResult<'mir, 'tcx> {
    /// The caller is expected to jump to the return block.
    NeedsJumping,
    /// Jumping has already been taken care of.
    AlreadyJumped,
    /// A MIR body has been found for the function.
    MirBody(&'mir mir::Body<'tcx>, ty::Instance<'tcx>),
    /// The item is not supported.
    NotSupported,
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    /// Returns the minimum alignment for the target architecture for allocations of the given size.
    fn min_align(&self, size: u64, kind: MiriMemoryKind) -> Align {
        let this = self.eval_context_ref();
        // List taken from `library/std/src/sys/common/alloc.rs`.
        // This list should be kept in sync with the one from libstd.
        let min_align = match this.tcx.sess.target.arch.as_ref() {
            "x86" | "arm" | "mips" | "mips32r6" | "powerpc" | "powerpc64" | "asmjs" | "wasm32" => 8,
            "x86_64" | "aarch64" | "mips64" | "mips64r6" | "s390x" | "sparc64" | "loongarch64" =>
                16,
            arch => bug!("unsupported target architecture for malloc: `{}`", arch),
        };
        // Windows always aligns, even small allocations.
        // Source: <https://support.microsoft.com/en-us/help/286470/how-to-use-pageheap-exe-in-windows-xp-windows-2000-and-windows-server>
        // But jemalloc does not, so for the C heap we only align if the allocation is sufficiently big.
        if kind == MiriMemoryKind::WinHeap || size >= min_align {
            return Align::from_bytes(min_align).unwrap();
        }
        // We have `size < min_align`. Round `size` *down* to the next power of two and use that.
        fn prev_power_of_two(x: u64) -> u64 {
            let next_pow2 = x.next_power_of_two();
            if next_pow2 == x {
                // x *is* a power of two, just use that.
                x
            } else {
                // x is between two powers, so next = 2*prev.
                next_pow2 / 2
            }
        }
        Align::from_bytes(prev_power_of_two(size)).unwrap()
    }

    fn malloc(
        &mut self,
        size: u64,
        zero_init: bool,
        kind: MiriMemoryKind,
    ) -> InterpResult<'tcx, Pointer<Option<Provenance>>> {
        let this = self.eval_context_mut();
        if size == 0 {
            Ok(Pointer::null())
        } else {
            let align = this.min_align(size, kind);
            let ptr = this.allocate_ptr(Size::from_bytes(size), align, kind.into())?;
            if zero_init {
                // We just allocated this, the access is definitely in-bounds and fits into our address space.
                this.write_bytes_ptr(
                    ptr.into(),
                    iter::repeat(0u8).take(usize::try_from(size).unwrap()),
                )
                .unwrap();
            }
            Ok(ptr.into())
        }
    }

    fn free(
        &mut self,
        ptr: Pointer<Option<Provenance>>,
        kind: MiriMemoryKind,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        if !this.ptr_is_null(ptr)? {
            this.deallocate_ptr(ptr, None, kind.into())?;
        }
        Ok(())
    }

    fn realloc(
        &mut self,
        old_ptr: Pointer<Option<Provenance>>,
        new_size: u64,
        kind: MiriMemoryKind,
    ) -> InterpResult<'tcx, Pointer<Option<Provenance>>> {
        let this = self.eval_context_mut();
        let new_align = this.min_align(new_size, kind);
        if this.ptr_is_null(old_ptr)? {
            if new_size == 0 {
                Ok(Pointer::null())
            } else {
                let new_ptr =
                    this.allocate_ptr(Size::from_bytes(new_size), new_align, kind.into())?;
                Ok(new_ptr.into())
            }
        } else {
            if new_size == 0 {
                this.deallocate_ptr(old_ptr, None, kind.into())?;
                Ok(Pointer::null())
            } else {
                let new_ptr = this.reallocate_ptr(
                    old_ptr,
                    None,
                    Size::from_bytes(new_size),
                    new_align,
                    kind.into(),
                )?;
                Ok(new_ptr.into())
            }
        }
    }

    /// Lookup the body of a function that has `link_name` as the symbol name.
    fn lookup_exported_symbol(
        &mut self,
        link_name: Symbol,
    ) -> InterpResult<'tcx, Option<(&'mir mir::Body<'tcx>, ty::Instance<'tcx>)>> {
        let this = self.eval_context_mut();
        let tcx = this.tcx.tcx;

        // If the result was cached, just return it.
        // (Cannot use `or_insert` since the code below might have to throw an error.)
        let entry = this.machine.exported_symbols_cache.entry(link_name);
        let instance = *match entry {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                // Find it if it was not cached.
                let mut instance_and_crate: Option<(ty::Instance<'_>, CrateNum)> = None;
                // `dependency_formats` includes all the transitive informations needed to link a crate,
                // which is what we need here since we need to dig out `exported_symbols` from all transitive
                // dependencies.
                let dependency_formats = tcx.dependency_formats(());
                let dependency_format = dependency_formats
                    .iter()
                    .find(|(crate_type, _)| *crate_type == CrateType::Executable)
                    .expect("interpreting a non-executable crate");
                for cnum in iter::once(LOCAL_CRATE).chain(
                    dependency_format.1.iter().enumerate().filter_map(|(num, &linkage)| {
                        // We add 1 to the number because that's what rustc also does everywhere it
                        // calls `CrateNum::new`...
                        #[allow(clippy::arithmetic_side_effects)]
                        (linkage != Linkage::NotLinked).then_some(CrateNum::new(num + 1))
                    }),
                ) {
                    // We can ignore `_export_info` here: we are a Rust crate, and everything is exported
                    // from a Rust crate.
                    for &(symbol, _export_info) in tcx.exported_symbols(cnum) {
                        if let ExportedSymbol::NonGeneric(def_id) = symbol {
                            let attrs = tcx.codegen_fn_attrs(def_id);
                            let symbol_name = if let Some(export_name) = attrs.export_name {
                                export_name
                            } else if attrs.flags.contains(CodegenFnAttrFlags::NO_MANGLE) {
                                tcx.item_name(def_id)
                            } else {
                                // Skip over items without an explicitly defined symbol name.
                                continue;
                            };
                            if symbol_name == link_name {
                                if let Some((original_instance, original_cnum)) = instance_and_crate
                                {
                                    // Make sure we are consistent wrt what is 'first' and 'second'.
                                    let original_span =
                                        tcx.def_span(original_instance.def_id()).data();
                                    let span = tcx.def_span(def_id).data();
                                    if original_span < span {
                                        throw_machine_stop!(
                                            TerminationInfo::MultipleSymbolDefinitions {
                                                link_name,
                                                first: original_span,
                                                first_crate: tcx.crate_name(original_cnum),
                                                second: span,
                                                second_crate: tcx.crate_name(cnum),
                                            }
                                        );
                                    } else {
                                        throw_machine_stop!(
                                            TerminationInfo::MultipleSymbolDefinitions {
                                                link_name,
                                                first: span,
                                                first_crate: tcx.crate_name(cnum),
                                                second: original_span,
                                                second_crate: tcx.crate_name(original_cnum),
                                            }
                                        );
                                    }
                                }
                                if !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn) {
                                    throw_ub_format!(
                                        "attempt to call an exported symbol that is not defined as a function"
                                    );
                                }
                                instance_and_crate = Some((ty::Instance::mono(tcx, def_id), cnum));
                            }
                        }
                    }
                }

                e.insert(instance_and_crate.map(|ic| ic.0))
            }
        };
        match instance {
            None => Ok(None), // no symbol with this name
            Some(instance) => Ok(Some((this.load_mir(instance.def, None)?, instance))),
        }
    }

    /// Read bytes from a `(ptr, len)` argument
    fn read_byte_slice<'i>(&'i self, bytes: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx, &'i [u8]>
    where
        'mir: 'i,
    {
        let this = self.eval_context_ref();
        let (ptr, len) = this.read_immediate(bytes)?.to_scalar_pair();
        let ptr = ptr.to_pointer(this)?;
        let len = len.to_target_usize(this)?;
        let bytes = this.read_bytes_ptr_strip_provenance(ptr, Size::from_bytes(len))?;
        Ok(bytes)
    }

    /// Emulates calling a foreign item, failing if the item is not supported.
    /// This function will handle `goto_block` if needed.
    /// Returns Ok(None) if the foreign item was completely handled
    /// by this function.
    /// Returns Ok(Some(body)) if processing the foreign item
    /// is delegated to another function.
    fn emulate_foreign_item(
        &mut self,
        def_id: DefId,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
        ret: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx, Option<(&'mir mir::Body<'tcx>, ty::Instance<'tcx>)>> {
        let this = self.eval_context_mut();
        let link_name = this.item_link_name(def_id);
        let tcx = this.tcx.tcx;

        // First: functions that diverge.
        let ret = match ret {
            None =>
                match link_name.as_str() {
                    "miri_start_panic" => {
                        // `check_shim` happens inside `handle_miri_start_panic`.
                        this.handle_miri_start_panic(abi, link_name, args, unwind)?;
                        return Ok(None);
                    }
                    // This matches calls to the foreign item `panic_impl`.
                    // The implementation is provided by the function with the `#[panic_handler]` attribute.
                    "panic_impl" => {
                        // We don't use `check_shim` here because we are just forwarding to the lang
                        // item. Argument count checking will be performed when the returned `Body` is
                        // called.
                        this.check_abi_and_shim_symbol_clash(abi, Abi::Rust, link_name)?;
                        let panic_impl_id = tcx.lang_items().panic_impl().unwrap();
                        let panic_impl_instance = ty::Instance::mono(tcx, panic_impl_id);
                        return Ok(Some((
                            this.load_mir(panic_impl_instance.def, None)?,
                            panic_impl_instance,
                        )));
                    }
                    #[rustfmt::skip]
                    | "exit"
                    | "ExitProcess"
                    => {
                        let exp_abi = if link_name.as_str() == "exit" {
                            Abi::C { unwind: false }
                        } else {
                            Abi::System { unwind: false }
                        };
                        let [code] = this.check_shim(abi, exp_abi, link_name, args)?;
                        // it's really u32 for ExitProcess, but we have to put it into the `Exit` variant anyway
                        let code = this.read_scalar(code)?.to_i32()?;
                        throw_machine_stop!(TerminationInfo::Exit { code: code.into(), leak_check: false });
                    }
                    "abort" => {
                        let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                        throw_machine_stop!(TerminationInfo::Abort(
                            "the program aborted execution".to_owned()
                        ))
                    }
                    _ => {
                        if let Some(body) = this.lookup_exported_symbol(link_name)? {
                            return Ok(Some(body));
                        }
                        this.handle_unsupported(format!(
                            "can't call (diverging) foreign function: {link_name}"
                        ))?;
                        return Ok(None);
                    }
                },
            Some(p) => p,
        };

        // Second: functions that return immediately.
        match this.emulate_foreign_item_by_name(link_name, abi, args, dest)? {
            EmulateByNameResult::NeedsJumping => {
                trace!("{:?}", this.dump_place(**dest));
                this.go_to_block(ret);
            }
            EmulateByNameResult::AlreadyJumped => (),
            EmulateByNameResult::MirBody(mir, instance) => return Ok(Some((mir, instance))),
            EmulateByNameResult::NotSupported => {
                if let Some(body) = this.lookup_exported_symbol(link_name)? {
                    return Ok(Some(body));
                }

                this.handle_unsupported(format!(
                    "can't call foreign function `{link_name}` on OS `{os}`",
                    os = this.tcx.sess.target.os,
                ))?;
                return Ok(None);
            }
        }

        Ok(None)
    }

    /// Emulates calling the internal __rust_* allocator functions
    fn emulate_allocator(
        &mut self,
        default: impl FnOnce(&mut MiriInterpCx<'mir, 'tcx>) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx, EmulateByNameResult<'mir, 'tcx>> {
        let this = self.eval_context_mut();

        let Some(allocator_kind) = this.tcx.allocator_kind(()) else {
            // in real code, this symbol does not exist without an allocator
            return Ok(EmulateByNameResult::NotSupported);
        };

        match allocator_kind {
            AllocatorKind::Global => {
                // When `#[global_allocator]` is used, `__rust_*` is defined by the macro expansion
                // of this attribute. As such we have to call an exported Rust function,
                // and not execute any Miri shim. Somewhat unintuitively doing so is done
                // by returning `NotSupported`, which triggers the `lookup_exported_symbol`
                // fallback case in `emulate_foreign_item`.
                return Ok(EmulateByNameResult::NotSupported);
            }
            AllocatorKind::Default => {
                default(this)?;
                Ok(EmulateByNameResult::NeedsJumping)
            }
        }
    }

    /// Emulates calling a foreign item using its name.
    fn emulate_foreign_item_by_name(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, EmulateByNameResult<'mir, 'tcx>> {
        let this = self.eval_context_mut();

        // First deal with any external C functions in linked .so file.
        #[cfg(target_os = "linux")]
        if this.machine.external_so_lib.as_ref().is_some() {
            use crate::shims::ffi_support::EvalContextExt as _;
            // An Ok(false) here means that the function being called was not exported
            // by the specified `.so` file; we should continue and check if it corresponds to
            // a provided shim.
            if this.call_external_c_fct(link_name, dest, args)? {
                return Ok(EmulateByNameResult::NeedsJumping);
            }
        }

        // When adding a new shim, you should follow the following pattern:
        // ```
        // "shim_name" => {
        //     let [arg1, arg2, arg3] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
        //     let result = this.shim_name(arg1, arg2, arg3)?;
        //     this.write_scalar(result, dest)?;
        // }
        // ```
        // and then define `shim_name` as a helper function in an extension trait in a suitable file
        // (see e.g. `unix/fs.rs`):
        // ```
        // fn shim_name(
        //     &mut self,
        //     arg1: &OpTy<'tcx, Provenance>,
        //     arg2: &OpTy<'tcx, Provenance>,
        //     arg3: &OpTy<'tcx, Provenance>,
        //     arg4: &OpTy<'tcx, Provenance>)
        // -> InterpResult<'tcx, Scalar<Provenance>> {
        //     let this = self.eval_context_mut();
        //
        //     // First thing: load all the arguments. Details depend on the shim.
        //     let arg1 = this.read_scalar(arg1)?.to_u32()?;
        //     let arg2 = this.read_pointer(arg2)?; // when you need to work with the pointer directly
        //     let arg3 = this.deref_operand_as(arg3, this.libc_ty_layout("some_libc_struct"))?; // when you want to load/store
        //         // through the pointer and supply the type information yourself
        //     let arg4 = this.deref_operand(arg4)?; // when you want to load/store through the pointer and trust
        //         // the user-given type (which you shouldn't usually do)
        //
        //     // ...
        //
        //     Ok(Scalar::from_u32(42))
        // }
        // ```
        // You might find existing shims not following this pattern, most
        // likely because they predate it or because for some reason they cannot be made to fit.

        // Here we dispatch all the shims for foreign functions. If you have a platform specific
        // shim, add it to the corresponding submodule.
        match link_name.as_str() {
            // Miri-specific extern functions
            "miri_get_alloc_id" => {
                let [ptr] = this.check_shim(abi, Abi::Rust, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let (alloc_id, _, _) = this.ptr_get_alloc_id(ptr).map_err(|_e| {
                    err_machine_stop!(TerminationInfo::Abort(format!(
                        "pointer passed to miri_get_alloc_id must not be dangling, got {ptr:?}"
                    )))
                })?;
                this.write_scalar(Scalar::from_u64(alloc_id.0.get()), dest)?;
            }
            "miri_print_borrow_state" => {
                let [id, show_unnamed] = this.check_shim(abi, Abi::Rust, link_name, args)?;
                let id = this.read_scalar(id)?.to_u64()?;
                let show_unnamed = this.read_scalar(show_unnamed)?.to_bool()?;
                if let Some(id) = std::num::NonZeroU64::new(id) {
                    this.print_borrow_state(AllocId(id), show_unnamed)?;
                }
            }
            "miri_pointer_name" => {
                // This associates a name to a tag. Very useful for debugging, and also makes
                // tests more strict.
                let [ptr, nth_parent, name] = this.check_shim(abi, Abi::Rust, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let nth_parent = this.read_scalar(nth_parent)?.to_u8()?;
                let name = this.read_byte_slice(name)?;
                // We must make `name` owned because we need to
                // end the shared borrow from `read_byte_slice` before we can
                // start the mutable borrow for `give_pointer_debug_name`.
                let name = String::from_utf8_lossy(name).into_owned();
                this.give_pointer_debug_name(ptr, nth_parent, &name)?;
            }
            "miri_static_root" => {
                let [ptr] = this.check_shim(abi, Abi::Rust, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let (alloc_id, offset, _) = this.ptr_get_alloc_id(ptr)?;
                if offset != Size::ZERO {
                    throw_unsup_format!(
                        "pointer passed to miri_static_root must point to beginning of an allocated block"
                    );
                }
                this.machine.static_roots.push(alloc_id);
            }
            "miri_host_to_target_path" => {
                let [ptr, out, out_size] = this.check_shim(abi, Abi::Rust, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let out = this.read_pointer(out)?;
                let out_size = this.read_scalar(out_size)?.to_target_usize(this)?;

                // The host affects program behavior here, so this requires isolation to be disabled.
                this.check_no_isolation("`miri_host_to_target_path`")?;

                // We read this as a plain OsStr and write it as a path, which will convert it to the target.
                let path = this.read_os_str_from_c_str(ptr)?.to_owned();
                let (success, needed_size) =
                    this.write_path_to_c_str(Path::new(&path), out, out_size)?;
                // Return value: 0 on success, otherwise the size it would have needed.
                this.write_int(if success { 0 } else { needed_size }, dest)?;
            }

            // Obtains the size of a Miri backtrace. See the README for details.
            "miri_backtrace_size" => {
                this.handle_miri_backtrace_size(abi, link_name, args, dest)?;
            }

            // Obtains a Miri backtrace. See the README for details.
            "miri_get_backtrace" => {
                // `check_shim` happens inside `handle_miri_get_backtrace`.
                this.handle_miri_get_backtrace(abi, link_name, args, dest)?;
            }

            // Resolves a Miri backtrace frame. See the README for details.
            "miri_resolve_frame" => {
                // `check_shim` happens inside `handle_miri_resolve_frame`.
                this.handle_miri_resolve_frame(abi, link_name, args, dest)?;
            }

            // Writes the function and file names of a Miri backtrace frame into a user provided buffer. See the README for details.
            "miri_resolve_frame_names" => {
                this.handle_miri_resolve_frame_names(abi, link_name, args)?;
            }

            // Writes some bytes to the interpreter's stdout/stderr. See the
            // README for details.
            "miri_write_to_stdout" | "miri_write_to_stderr" => {
                let [msg] = this.check_shim(abi, Abi::Rust, link_name, args)?;
                let msg = this.read_byte_slice(msg)?;
                // Note: we're ignoring errors writing to host stdout/stderr.
                let _ignore = match link_name.as_str() {
                    "miri_write_to_stdout" => std::io::stdout().write_all(msg),
                    "miri_write_to_stderr" => std::io::stderr().write_all(msg),
                    _ => unreachable!(),
                };
            }

            // Standard C allocation
            "malloc" => {
                let [size] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let size = this.read_target_usize(size)?;
                let res = this.malloc(size, /*zero_init:*/ false, MiriMemoryKind::C)?;
                this.write_pointer(res, dest)?;
            }
            "calloc" => {
                let [items, len] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let items = this.read_target_usize(items)?;
                let len = this.read_target_usize(len)?;
                let size = items
                    .checked_mul(len)
                    .ok_or_else(|| err_ub_format!("overflow during calloc size computation"))?;
                let res = this.malloc(size, /*zero_init:*/ true, MiriMemoryKind::C)?;
                this.write_pointer(res, dest)?;
            }
            "free" => {
                let [ptr] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                this.free(ptr, MiriMemoryKind::C)?;
            }
            "realloc" => {
                let [old_ptr, new_size] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let old_ptr = this.read_pointer(old_ptr)?;
                let new_size = this.read_target_usize(new_size)?;
                let res = this.realloc(old_ptr, new_size, MiriMemoryKind::C)?;
                this.write_pointer(res, dest)?;
            }

            // Rust allocation
            "__rust_alloc" | "miri_alloc" => {
                let default = |this: &mut MiriInterpCx<'mir, 'tcx>| {
                    // Only call `check_shim` when `#[global_allocator]` isn't used. When that
                    // macro is used, we act like no shim exists, so that the exported function can run.
                    let [size, align] = this.check_shim(abi, Abi::Rust, link_name, args)?;
                    let size = this.read_target_usize(size)?;
                    let align = this.read_target_usize(align)?;

                    Self::check_alloc_request(size, align)?;

                    let memory_kind = match link_name.as_str() {
                        "__rust_alloc" => MiriMemoryKind::Rust,
                        "miri_alloc" => MiriMemoryKind::Miri,
                        _ => unreachable!(),
                    };

                    let ptr = this.allocate_ptr(
                        Size::from_bytes(size),
                        Align::from_bytes(align).unwrap(),
                        memory_kind.into(),
                    )?;

                    this.write_pointer(ptr, dest)
                };

                match link_name.as_str() {
                    "__rust_alloc" => return this.emulate_allocator(default),
                    "miri_alloc" => {
                        default(this)?;
                        return Ok(EmulateByNameResult::NeedsJumping);
                    }
                    _ => unreachable!(),
                }
            }
            "__rust_alloc_zeroed" => {
                return this.emulate_allocator(|this| {
                    // See the comment for `__rust_alloc` why `check_shim` is only called in the
                    // default case.
                    let [size, align] = this.check_shim(abi, Abi::Rust, link_name, args)?;
                    let size = this.read_target_usize(size)?;
                    let align = this.read_target_usize(align)?;

                    Self::check_alloc_request(size, align)?;

                    let ptr = this.allocate_ptr(
                        Size::from_bytes(size),
                        Align::from_bytes(align).unwrap(),
                        MiriMemoryKind::Rust.into(),
                    )?;

                    // We just allocated this, the access is definitely in-bounds.
                    this.write_bytes_ptr(
                        ptr.into(),
                        iter::repeat(0u8).take(usize::try_from(size).unwrap()),
                    )
                    .unwrap();
                    this.write_pointer(ptr, dest)
                });
            }
            "__rust_dealloc" | "miri_dealloc" => {
                let default = |this: &mut MiriInterpCx<'mir, 'tcx>| {
                    // See the comment for `__rust_alloc` why `check_shim` is only called in the
                    // default case.
                    let [ptr, old_size, align] =
                        this.check_shim(abi, Abi::Rust, link_name, args)?;
                    let ptr = this.read_pointer(ptr)?;
                    let old_size = this.read_target_usize(old_size)?;
                    let align = this.read_target_usize(align)?;

                    let memory_kind = match link_name.as_str() {
                        "__rust_dealloc" => MiriMemoryKind::Rust,
                        "miri_dealloc" => MiriMemoryKind::Miri,
                        _ => unreachable!(),
                    };

                    // No need to check old_size/align; we anyway check that they match the allocation.
                    this.deallocate_ptr(
                        ptr,
                        Some((Size::from_bytes(old_size), Align::from_bytes(align).unwrap())),
                        memory_kind.into(),
                    )
                };

                match link_name.as_str() {
                    "__rust_dealloc" => {
                        return this.emulate_allocator(default);
                    }
                    "miri_dealloc" => {
                        default(this)?;
                        return Ok(EmulateByNameResult::NeedsJumping);
                    }
                    _ => unreachable!(),
                }
            }
            "__rust_realloc" => {
                return this.emulate_allocator(|this| {
                    // See the comment for `__rust_alloc` why `check_shim` is only called in the
                    // default case.
                    let [ptr, old_size, align, new_size] =
                        this.check_shim(abi, Abi::Rust, link_name, args)?;
                    let ptr = this.read_pointer(ptr)?;
                    let old_size = this.read_target_usize(old_size)?;
                    let align = this.read_target_usize(align)?;
                    let new_size = this.read_target_usize(new_size)?;
                    // No need to check old_size; we anyway check that they match the allocation.

                    Self::check_alloc_request(new_size, align)?;

                    let align = Align::from_bytes(align).unwrap();
                    let new_ptr = this.reallocate_ptr(
                        ptr,
                        Some((Size::from_bytes(old_size), align)),
                        Size::from_bytes(new_size),
                        align,
                        MiriMemoryKind::Rust.into(),
                    )?;
                    this.write_pointer(new_ptr, dest)
                });
            }

            // C memory handling functions
            "memcmp" => {
                let [left, right, n] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let left = this.read_pointer(left)?;
                let right = this.read_pointer(right)?;
                let n = Size::from_bytes(this.read_target_usize(n)?);

                let result = {
                    let left_bytes = this.read_bytes_ptr_strip_provenance(left, n)?;
                    let right_bytes = this.read_bytes_ptr_strip_provenance(right, n)?;

                    use std::cmp::Ordering::*;
                    match left_bytes.cmp(right_bytes) {
                        Less => -1i32,
                        Equal => 0,
                        Greater => 1,
                    }
                };

                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "memrchr" => {
                let [ptr, val, num] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let val = this.read_scalar(val)?.to_i32()?;
                let num = this.read_target_usize(num)?;
                // The docs say val is "interpreted as unsigned char".
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let val = val as u8;

                if let Some(idx) = this
                    .read_bytes_ptr_strip_provenance(ptr, Size::from_bytes(num))?
                    .iter()
                    .rev()
                    .position(|&c| c == val)
                {
                    let idx = u64::try_from(idx).unwrap();
                    #[allow(clippy::arithmetic_side_effects)] // idx < num, so this never wraps
                    let new_ptr = ptr.offset(Size::from_bytes(num - idx - 1), this)?;
                    this.write_pointer(new_ptr, dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }
            "memchr" => {
                let [ptr, val, num] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let val = this.read_scalar(val)?.to_i32()?;
                let num = this.read_target_usize(num)?;
                // The docs say val is "interpreted as unsigned char".
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let val = val as u8;

                let idx = this
                    .read_bytes_ptr_strip_provenance(ptr, Size::from_bytes(num))?
                    .iter()
                    .position(|&c| c == val);
                if let Some(idx) = idx {
                    let new_ptr = ptr.offset(Size::from_bytes(idx as u64), this)?;
                    this.write_pointer(new_ptr, dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }
            "strlen" => {
                let [ptr] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let n = this.read_c_str(ptr)?.len();
                this.write_scalar(
                    Scalar::from_target_usize(u64::try_from(n).unwrap(), this),
                    dest,
                )?;
            }
            "memcpy" => {
                let [ptr_dest, ptr_src, n] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let ptr_dest = this.read_pointer(ptr_dest)?;
                let ptr_src = this.read_pointer(ptr_src)?;
                let n = this.read_target_usize(n)?;

                // C requires that this must always be a valid pointer, even if `n` is zero, so we better check that.
                // (This is more than Rust requires, so `mem_copy` is not sufficient.)
                this.ptr_get_alloc_id(ptr_dest)?;
                this.ptr_get_alloc_id(ptr_src)?;

                this.mem_copy(
                    ptr_src,
                    Align::ONE,
                    ptr_dest,
                    Align::ONE,
                    Size::from_bytes(n),
                    true,
                )?;
                this.write_pointer(ptr_dest, dest)?;
            }
            "strcpy" => {
                let [ptr_dest, ptr_src] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let ptr_dest = this.read_pointer(ptr_dest)?;
                let ptr_src = this.read_pointer(ptr_src)?;

                // We use `read_c_str` to determine the amount of data to copy,
                // and then use `mem_copy` for the actual copy. This means
                // pointer provenance is preserved by this implementation of `strcpy`.
                // That is probably overly cautious, but there also is no fundamental
                // reason to have `strcpy` destroy pointer provenance.
                let n = this.read_c_str(ptr_src)?.len().checked_add(1).unwrap();
                this.mem_copy(
                    ptr_src,
                    Align::ONE,
                    ptr_dest,
                    Align::ONE,
                    Size::from_bytes(n),
                    true,
                )?;
                this.write_pointer(ptr_dest, dest)?;
            }

            // math functions (note that there are also intrinsics for some other functions)
            #[rustfmt::skip]
            | "cbrtf"
            | "coshf"
            | "sinhf"
            | "tanf"
            | "tanhf"
            | "acosf"
            | "asinf"
            | "atanf"
            | "log1pf"
            | "expm1f"
            => {
                let [f] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(f)?.to_u32()?);
                let res = match link_name.as_str() {
                    "cbrtf" => f.cbrt(),
                    "coshf" => f.cosh(),
                    "sinhf" => f.sinh(),
                    "tanf" => f.tan(),
                    "tanhf" => f.tanh(),
                    "acosf" => f.acos(),
                    "asinf" => f.asin(),
                    "atanf" => f.atan(),
                    "log1pf" => f.ln_1p(),
                    "expm1f" => f.exp_m1(),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u32(res.to_bits()), dest)?;
            }
            #[rustfmt::skip]
            | "_hypotf"
            | "hypotf"
            | "atan2f"
            | "fdimf"
            => {
                let [f1, f2] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // underscore case for windows, here and below
                // (see https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/floating-point-primitives?view=vs-2019)
                // FIXME: Using host floats.
                let f1 = f32::from_bits(this.read_scalar(f1)?.to_u32()?);
                let f2 = f32::from_bits(this.read_scalar(f2)?.to_u32()?);
                let res = match link_name.as_str() {
                    "_hypotf" | "hypotf" => f1.hypot(f2),
                    "atan2f" => f1.atan2(f2),
                    #[allow(deprecated)]
                    "fdimf" => f1.abs_sub(f2),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u32(res.to_bits()), dest)?;
            }
            #[rustfmt::skip]
            | "cbrt"
            | "cosh"
            | "sinh"
            | "tan"
            | "tanh"
            | "acos"
            | "asin"
            | "atan"
            | "log1p"
            | "expm1"
            => {
                let [f] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(f)?.to_u64()?);
                let res = match link_name.as_str() {
                    "cbrt" => f.cbrt(),
                    "cosh" => f.cosh(),
                    "sinh" => f.sinh(),
                    "tan" => f.tan(),
                    "tanh" => f.tanh(),
                    "acos" => f.acos(),
                    "asin" => f.asin(),
                    "atan" => f.atan(),
                    "log1p" => f.ln_1p(),
                    "expm1" => f.exp_m1(),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u64(res.to_bits()), dest)?;
            }
            #[rustfmt::skip]
            | "_hypot"
            | "hypot"
            | "atan2"
            | "fdim"
            => {
                let [f1, f2] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // FIXME: Using host floats.
                let f1 = f64::from_bits(this.read_scalar(f1)?.to_u64()?);
                let f2 = f64::from_bits(this.read_scalar(f2)?.to_u64()?);
                let res = match link_name.as_str() {
                    "_hypot" | "hypot" => f1.hypot(f2),
                    "atan2" => f1.atan2(f2),
                    #[allow(deprecated)]
                    "fdim" => f1.abs_sub(f2),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u64(res.to_bits()), dest)?;
            }
            #[rustfmt::skip]
            | "_ldexp"
            | "ldexp"
            | "scalbn"
            => {
                let [x, exp] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // For radix-2 (binary) systems, `ldexp` and `scalbn` are the same.
                let x = this.read_scalar(x)?.to_f64()?;
                let exp = this.read_scalar(exp)?.to_i32()?;

                // Saturating cast to i16. Even those are outside the valid exponent range so
                // `scalbn` below will do its over/underflow handling.
                let exp = if exp > i32::from(i16::MAX) {
                    i16::MAX
                } else if exp < i32::from(i16::MIN) {
                    i16::MIN
                } else {
                    exp.try_into().unwrap()
                };

                let res = x.scalbn(exp);
                this.write_scalar(Scalar::from_f64(res), dest)?;
            }

            // Architecture-specific shims
            "llvm.x86.addcarry.64" if this.tcx.sess.target.arch == "x86_64" => {
                // Computes u8+u64+u64, returning tuple (u8,u64) comprising the output carry and truncated sum.
                let [c_in, a, b] = this.check_shim(abi, Abi::Unadjusted, link_name, args)?;
                let c_in = this.read_scalar(c_in)?.to_u8()?;
                let a = this.read_scalar(a)?.to_u64()?;
                let b = this.read_scalar(b)?.to_u64()?;

                #[allow(clippy::arithmetic_side_effects)]
                // adding two u64 and a u8 cannot wrap in a u128
                let wide_sum = u128::from(c_in) + u128::from(a) + u128::from(b);
                #[allow(clippy::arithmetic_side_effects)] // it's a u128, we can shift by 64
                let (c_out, sum) = ((wide_sum >> 64).truncate::<u8>(), wide_sum.truncate::<u64>());

                let c_out_field = this.place_field(dest, 0)?;
                this.write_scalar(Scalar::from_u8(c_out), &c_out_field)?;
                let sum_field = this.place_field(dest, 1)?;
                this.write_scalar(Scalar::from_u64(sum), &sum_field)?;
            }
            "llvm.x86.sse2.pause"
                if this.tcx.sess.target.arch == "x86" || this.tcx.sess.target.arch == "x86_64" =>
            {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.yield_active_thread();
            }
            "llvm.aarch64.isb" if this.tcx.sess.target.arch == "aarch64" => {
                let [arg] = this.check_shim(abi, Abi::Unadjusted, link_name, args)?;
                let arg = this.read_scalar(arg)?.to_i32()?;
                match arg {
                    // SY ("full system scope")
                    15 => {
                        this.yield_active_thread();
                    }
                    _ => {
                        throw_unsup_format!("unsupported llvm.aarch64.isb argument {}", arg);
                    }
                }
            }
            "llvm.arm.hint" if this.tcx.sess.target.arch == "arm" => {
                let [arg] = this.check_shim(abi, Abi::Unadjusted, link_name, args)?;
                let arg = this.read_scalar(arg)?.to_i32()?;
                match arg {
                    // YIELD
                    1 => {
                        this.yield_active_thread();
                    }
                    _ => {
                        throw_unsup_format!("unsupported llvm.arm.hint argument {}", arg);
                    }
                }
            }

            // Platform-specific shims
            _ =>
                return match this.tcx.sess.target.os.as_ref() {
                    target_os if target_os_is_unix(target_os) =>
                        shims::unix::foreign_items::EvalContextExt::emulate_foreign_item_by_name(
                            this, link_name, abi, args, dest,
                        ),
                    "windows" =>
                        shims::windows::foreign_items::EvalContextExt::emulate_foreign_item_by_name(
                            this, link_name, abi, args, dest,
                        ),
                    _ => Ok(EmulateByNameResult::NotSupported),
                },
        };
        // We only fall through to here if we did *not* hit the `_` arm above,
        // i.e., if we actually emulated the function with one of the shims.
        Ok(EmulateByNameResult::NeedsJumping)
    }

    /// Check some basic requirements for this allocation request:
    /// non-zero size, power-of-two alignment.
    fn check_alloc_request(size: u64, align: u64) -> InterpResult<'tcx> {
        if size == 0 {
            throw_ub_format!("creating allocation with size 0");
        }
        if !align.is_power_of_two() {
            throw_ub_format!("creating allocation with non-power-of-two alignment {}", align);
        }
        Ok(())
    }
}
