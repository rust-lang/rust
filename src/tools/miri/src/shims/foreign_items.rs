use std::collections::hash_map::Entry;
use std::io::Write;
use std::path::Path;

use rustc_abi::{Align, AlignFromBytesError, CanonAbi, Size};
use rustc_apfloat::Float;
use rustc_ast::expand::allocator::alloc_error_handler_name;
use rustc_hir::attrs::Linkage;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::CrateNum;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::interpret::AllocInit;
use rustc_middle::ty::{Instance, Ty};
use rustc_middle::{mir, ty};
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use self::helpers::{ToHost, ToSoft};
use super::alloc::EvalContextExt as _;
use super::backtrace::EvalContextExt as _;
use crate::helpers::EvalContextExt as _;
use crate::*;

/// Type of dynamic symbols (for `dlsym` et al)
#[derive(Debug, Copy, Clone)]
pub struct DynSym(Symbol);

#[expect(clippy::should_implement_trait)]
impl DynSym {
    pub fn from_str(name: &str) -> Self {
        DynSym(Symbol::intern(name))
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Emulates calling a foreign item, failing if the item is not supported.
    /// This function will handle `goto_block` if needed.
    /// Returns Ok(None) if the foreign item was completely handled
    /// by this function.
    /// Returns Ok(Some(body)) if processing the foreign item
    /// is delegated to another function.
    fn emulate_foreign_item(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx>,
        ret: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx, Option<(&'tcx mir::Body<'tcx>, ty::Instance<'tcx>)>> {
        let this = self.eval_context_mut();

        // Some shims forward to other MIR bodies.
        match link_name.as_str() {
            name if name == this.mangle_internal_symbol("__rust_alloc_error_handler") => {
                // Forward to the right symbol that implements this function.
                let Some(handler_kind) = this.tcx.alloc_error_handler_kind(()) else {
                    // in real code, this symbol does not exist without an allocator
                    throw_unsup_format!(
                        "`__rust_alloc_error_handler` cannot be called when no alloc error handler is set"
                    );
                };
                let name = Symbol::intern(
                    this.mangle_internal_symbol(alloc_error_handler_name(handler_kind)),
                );
                let handler =
                    this.lookup_exported_symbol(name)?.expect("missing alloc error handler symbol");
                return interp_ok(Some(handler));
            }
            _ => {}
        }

        // FIXME: avoid allocating memory
        let dest = this.force_allocation(dest)?;

        // The rest either implements the logic, or falls back to `lookup_exported_symbol`.
        match this.emulate_foreign_item_inner(link_name, abi, args, &dest)? {
            EmulateItemResult::NeedsReturn => {
                trace!("{:?}", this.dump_place(&dest.clone().into()));
                this.return_to_block(ret)?;
            }
            EmulateItemResult::NeedsUnwind => {
                // Jump to the unwind block to begin unwinding.
                this.unwind_to_block(unwind)?;
            }
            EmulateItemResult::AlreadyJumped => (),
            EmulateItemResult::NotSupported => {
                if let Some(body) = this.lookup_exported_symbol(link_name)? {
                    return interp_ok(Some(body));
                }

                throw_machine_stop!(TerminationInfo::UnsupportedForeignItem(format!(
                    "can't call foreign function `{link_name}` on OS `{os}`",
                    os = this.tcx.sess.target.os,
                )));
            }
        }

        interp_ok(None)
    }

    fn is_dyn_sym(&self, name: &str) -> bool {
        let this = self.eval_context_ref();
        match this.tcx.sess.target.os.as_ref() {
            os if this.target_os_is_unix() => shims::unix::foreign_items::is_dyn_sym(name, os),
            "wasi" => shims::wasi::foreign_items::is_dyn_sym(name),
            "windows" => shims::windows::foreign_items::is_dyn_sym(name),
            _ => false,
        }
    }

    /// Emulates a call to a `DynSym`.
    fn emulate_dyn_sym(
        &mut self,
        sym: DynSym,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx>,
        ret: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx> {
        let res = self.emulate_foreign_item(sym.0, abi, args, dest, ret, unwind)?;
        assert!(res.is_none(), "DynSyms that delegate are not supported");
        interp_ok(())
    }

    /// Lookup the body of a function that has `link_name` as the symbol name.
    fn lookup_exported_symbol(
        &mut self,
        link_name: Symbol,
    ) -> InterpResult<'tcx, Option<(&'tcx mir::Body<'tcx>, ty::Instance<'tcx>)>> {
        let this = self.eval_context_mut();
        let tcx = this.tcx.tcx;

        // If the result was cached, just return it.
        // (Cannot use `or_insert` since the code below might have to throw an error.)
        let entry = this.machine.exported_symbols_cache.entry(link_name);
        let instance = *match entry {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                // Find it if it was not cached.

                struct SymbolTarget<'tcx> {
                    instance: ty::Instance<'tcx>,
                    cnum: CrateNum,
                    is_weak: bool,
                }
                let mut symbol_target: Option<SymbolTarget<'tcx>> = None;
                helpers::iter_exported_symbols(tcx, |cnum, def_id| {
                    let attrs = tcx.codegen_fn_attrs(def_id);
                    // Skip over imports of items.
                    if tcx.is_foreign_item(def_id) {
                        return interp_ok(());
                    }
                    // Skip over items without an explicitly defined symbol name.
                    if !(attrs.symbol_name.is_some()
                        || attrs.flags.contains(CodegenFnAttrFlags::NO_MANGLE)
                        || attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL))
                    {
                        return interp_ok(());
                    }

                    let instance = Instance::mono(tcx, def_id);
                    let symbol_name = tcx.symbol_name(instance).name;
                    let is_weak = attrs.linkage == Some(Linkage::WeakAny);
                    if symbol_name == link_name.as_str() {
                        if let Some(original) = &symbol_target {
                            // There is more than one definition with this name. What we do now
                            // depends on whether one or both definitions are weak.
                            match (is_weak, original.is_weak) {
                                (false, true) => {
                                    // Original definition is a weak definition. Override it.

                                    symbol_target = Some(SymbolTarget {
                                        instance: ty::Instance::mono(tcx, def_id),
                                        cnum,
                                        is_weak,
                                    });
                                }
                                (true, false) => {
                                    // Current definition is a weak definition. Keep the original one.
                                }
                                (true, true) | (false, false) => {
                                    // Either both definitions are non-weak or both are weak. In
                                    // either case return an error. For weak definitions we error
                                    // because it is unspecified which definition would have been
                                    // picked by the linker.

                                    // Make sure we are consistent wrt what is 'first' and 'second'.
                                    let original_span =
                                        tcx.def_span(original.instance.def_id()).data();
                                    let span = tcx.def_span(def_id).data();
                                    if original_span < span {
                                        throw_machine_stop!(
                                            TerminationInfo::MultipleSymbolDefinitions {
                                                link_name,
                                                first: original_span,
                                                first_crate: tcx.crate_name(original.cnum),
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
                                                second_crate: tcx.crate_name(original.cnum),
                                            }
                                        );
                                    }
                                }
                            }
                        } else {
                            symbol_target = Some(SymbolTarget {
                                instance: ty::Instance::mono(tcx, def_id),
                                cnum,
                                is_weak,
                            });
                        }
                    }
                    interp_ok(())
                })?;

                // Once we identified the instance corresponding to the symbol, ensure
                // it is a function. It is okay to encounter non-functions in the search above
                // as long as the final instance we arrive at is a function.
                if let Some(SymbolTarget { instance, .. }) = symbol_target {
                    if !matches!(tcx.def_kind(instance.def_id()), DefKind::Fn | DefKind::AssocFn) {
                        throw_ub_format!(
                            "attempt to call an exported symbol that is not defined as a function"
                        );
                    }
                }

                e.insert(symbol_target.map(|SymbolTarget { instance, .. }| instance))
            }
        };
        match instance {
            None => interp_ok(None), // no symbol with this name
            Some(instance) => interp_ok(Some((this.load_mir(instance.def, None)?, instance))),
        }
    }
}

impl<'tcx> EvalContextExtPriv<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextExtPriv<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Check some basic requirements for this allocation request:
    /// non-zero size, power-of-two alignment.
    fn check_rustc_alloc_request(&self, size: u64, align: u64) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();
        if size == 0 {
            throw_ub_format!("creating allocation with size 0");
        }
        if size > this.max_size_of_val().bytes() {
            throw_ub_format!("creating an allocation larger than half the address space");
        }
        if let Err(e) = Align::from_bytes(align) {
            match e {
                AlignFromBytesError::TooLarge(_) => {
                    throw_unsup_format!(
                        "creating allocation with alignment {align} exceeding rustc's maximum \
                         supported value"
                    );
                }
                AlignFromBytesError::NotPowerOfTwo(_) => {
                    throw_ub_format!("creating allocation with non-power-of-two alignment {align}");
                }
            }
        }

        interp_ok(())
    }

    fn emulate_foreign_item_inner(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();

        // First deal with any external C functions in linked .so file.
        #[cfg(all(unix, feature = "native-lib"))]
        if !this.machine.native_lib.is_empty() {
            use crate::shims::native_lib::EvalContextExt as _;
            // An Ok(false) here means that the function being called was not exported
            // by the specified `.so` file; we should continue and check if it corresponds to
            // a provided shim.
            if this.call_native_fn(link_name, dest, args)? {
                return interp_ok(EmulateItemResult::NeedsReturn);
            }
        }
        // When adding a new shim, you should follow the following pattern:
        // ```
        // "shim_name" => {
        //     let [arg1, arg2, arg3] = this.check_shim(abi, CanonAbi::C , link_name, args)?;
        //     let result = this.shim_name(arg1, arg2, arg3)?;
        //     this.write_scalar(result, dest)?;
        // }
        // ```
        // and then define `shim_name` as a helper function in an extension trait in a suitable file
        // (see e.g. `unix/fs.rs`):
        // ```
        // fn shim_name(
        //     &mut self,
        //     arg1: &OpTy<'tcx>,
        //     arg2: &OpTy<'tcx>,
        //     arg3: &OpTy<'tcx>,
        //     arg4: &OpTy<'tcx>)
        // -> InterpResult<'tcx, Scalar> {
        //     let this = self.eval_context_mut();
        //
        //     // First thing: load all the arguments. Details depend on the shim.
        //     let arg1 = this.read_scalar(arg1)?.to_u32()?;
        //     let arg2 = this.read_pointer(arg2)?; // when you need to work with the pointer directly
        //     let arg3 = this.deref_pointer_as(arg3, this.libc_ty_layout("some_libc_struct"))?; // when you want to load/store
        //         // through the pointer and supply the type information yourself
        //     let arg4 = this.deref_pointer(arg4)?; // when you want to load/store through the pointer and trust
        //         // the user-given type (which you shouldn't usually do)
        //
        //     // ...
        //
        //     interp_ok(Scalar::from_u32(42))
        // }
        // ```
        // You might find existing shims not following this pattern, most
        // likely because they predate it or because for some reason they cannot be made to fit.

        // Here we dispatch all the shims for foreign functions. If you have a platform specific
        // shim, add it to the corresponding submodule.
        match link_name.as_str() {
            // Miri-specific extern functions
            "miri_start_unwind" => {
                let [payload] =
                    this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                this.handle_miri_start_unwind(payload)?;
                return interp_ok(EmulateItemResult::NeedsUnwind);
            }
            "miri_run_provenance_gc" => {
                let [] = this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                this.run_provenance_gc();
            }
            "miri_get_alloc_id" => {
                let [ptr] = this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let (alloc_id, _, _) = this.ptr_get_alloc_id(ptr, 0).map_err_kind(|_e| {
                    err_machine_stop!(TerminationInfo::Abort(format!(
                        "pointer passed to `miri_get_alloc_id` must not be dangling, got {ptr:?}"
                    )))
                })?;
                this.write_scalar(Scalar::from_u64(alloc_id.0.get()), dest)?;
            }
            "miri_print_borrow_state" => {
                let [id, show_unnamed] =
                    this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                let id = this.read_scalar(id)?.to_u64()?;
                let show_unnamed = this.read_scalar(show_unnamed)?.to_bool()?;
                if let Some(id) = std::num::NonZero::new(id).map(AllocId)
                    && this.get_alloc_info(id).kind == AllocKind::LiveData
                {
                    this.print_borrow_state(id, show_unnamed)?;
                } else {
                    eprintln!("{id} is not the ID of a live data allocation");
                }
            }
            "miri_pointer_name" => {
                // This associates a name to a tag. Very useful for debugging, and also makes
                // tests more strict.
                let [ptr, nth_parent, name] =
                    this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let nth_parent = this.read_scalar(nth_parent)?.to_u8()?;
                let name = this.read_immediate(name)?;

                let name = this.read_byte_slice(&name)?;
                // We must make `name` owned because we need to
                // end the shared borrow from `read_byte_slice` before we can
                // start the mutable borrow for `give_pointer_debug_name`.
                let name = String::from_utf8_lossy(name).into_owned();
                this.give_pointer_debug_name(ptr, nth_parent, &name)?;
            }
            "miri_static_root" => {
                let [ptr] = this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let (alloc_id, offset, _) = this.ptr_get_alloc_id(ptr, 0)?;
                if offset != Size::ZERO {
                    throw_unsup_format!(
                        "pointer passed to `miri_static_root` must point to beginning of an allocated block"
                    );
                }
                this.machine.static_roots.push(alloc_id);
            }
            "miri_host_to_target_path" => {
                let [ptr, out, out_size] =
                    this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
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
                this.handle_miri_get_backtrace(abi, link_name, args)?;
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
                let [msg] = this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                let msg = this.read_immediate(msg)?;
                let msg = this.read_byte_slice(&msg)?;
                // Note: we're ignoring errors writing to host stdout/stderr.
                let _ignore = match link_name.as_str() {
                    "miri_write_to_stdout" => std::io::stdout().write_all(msg),
                    "miri_write_to_stderr" => std::io::stderr().write_all(msg),
                    _ => unreachable!(),
                };
            }
            // Promises that a pointer has a given symbolic alignment.
            "miri_promise_symbolic_alignment" => {
                use rustc_abi::AlignFromBytesError;

                let [ptr, align] =
                    this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let align = this.read_target_usize(align)?;
                if !align.is_power_of_two() {
                    throw_unsup_format!(
                        "`miri_promise_symbolic_alignment`: alignment must be a power of 2, got {align}"
                    );
                }
                let align = Align::from_bytes(align).unwrap_or_else(|err| {
                    match err {
                        AlignFromBytesError::NotPowerOfTwo(_) => unreachable!(),
                        // When the alignment is a power of 2 but too big, clamp it to MAX.
                        AlignFromBytesError::TooLarge(_) => Align::MAX,
                    }
                });
                let addr = ptr.addr();
                // Cannot panic since `align` is a power of 2 and hence non-zero.
                if addr.bytes().strict_rem(align.bytes()) != 0 {
                    throw_unsup_format!(
                        "`miri_promise_symbolic_alignment`: pointer is not actually aligned"
                    );
                }
                if let Ok((alloc_id, offset, ..)) = this.ptr_try_get_alloc_id(ptr, 0) {
                    let alloc_align = this.get_alloc_info(alloc_id).align;
                    // If the newly promised alignment is bigger than the native alignment of this
                    // allocation, and bigger than the previously promised alignment, then set it.
                    if align > alloc_align
                        && this
                            .machine
                            .symbolic_alignment
                            .get_mut()
                            .get(&alloc_id)
                            .is_none_or(|&(_, old_align)| align > old_align)
                    {
                        this.machine.symbolic_alignment.get_mut().insert(alloc_id, (offset, align));
                    }
                }
            }

            // Aborting the process.
            "exit" => {
                let [code] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let code = this.read_scalar(code)?.to_i32()?;
                throw_machine_stop!(TerminationInfo::Exit { code, leak_check: false });
            }
            "abort" => {
                let [] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                throw_machine_stop!(TerminationInfo::Abort(
                    "the program aborted execution".to_owned()
                ))
            }

            // Standard C allocation
            "malloc" => {
                let [size] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let size = this.read_target_usize(size)?;
                if size <= this.max_size_of_val().bytes() {
                    let res = this.malloc(size, AllocInit::Uninit)?;
                    this.write_pointer(res, dest)?;
                } else {
                    // If this does not fit in an isize, return null and, on Unix, set errno.
                    if this.target_os_is_unix() {
                        this.set_last_error(LibcError("ENOMEM"))?;
                    }
                    this.write_null(dest)?;
                }
            }
            "calloc" => {
                let [items, elem_size] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let items = this.read_target_usize(items)?;
                let elem_size = this.read_target_usize(elem_size)?;
                if let Some(size) = this.compute_size_in_bytes(Size::from_bytes(elem_size), items) {
                    let res = this.malloc(size.bytes(), AllocInit::Zero)?;
                    this.write_pointer(res, dest)?;
                } else {
                    // On size overflow, return null and, on Unix, set errno.
                    if this.target_os_is_unix() {
                        this.set_last_error(LibcError("ENOMEM"))?;
                    }
                    this.write_null(dest)?;
                }
            }
            "free" => {
                let [ptr] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                this.free(ptr)?;
            }
            "realloc" => {
                let [old_ptr, new_size] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let old_ptr = this.read_pointer(old_ptr)?;
                let new_size = this.read_target_usize(new_size)?;
                if new_size <= this.max_size_of_val().bytes() {
                    let res = this.realloc(old_ptr, new_size)?;
                    this.write_pointer(res, dest)?;
                } else {
                    // If this does not fit in an isize, return null and, on Unix, set errno.
                    if this.target_os_is_unix() {
                        this.set_last_error(LibcError("ENOMEM"))?;
                    }
                    this.write_null(dest)?;
                }
            }

            // Rust allocation
            name if name == this.mangle_internal_symbol("__rust_alloc") || name == "miri_alloc" => {
                let default = |ecx: &mut MiriInterpCx<'tcx>| {
                    // Only call `check_shim` when `#[global_allocator]` isn't used. When that
                    // macro is used, we act like no shim exists, so that the exported function can run.
                    let [size, align] =
                        ecx.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                    let size = ecx.read_target_usize(size)?;
                    let align = ecx.read_target_usize(align)?;

                    ecx.check_rustc_alloc_request(size, align)?;

                    let memory_kind = match link_name.as_str() {
                        "miri_alloc" => MiriMemoryKind::Miri,
                        _ => MiriMemoryKind::Rust,
                    };

                    let ptr = ecx.allocate_ptr(
                        Size::from_bytes(size),
                        Align::from_bytes(align).unwrap(),
                        memory_kind.into(),
                        AllocInit::Uninit,
                    )?;

                    ecx.write_pointer(ptr, dest)
                };

                match link_name.as_str() {
                    "miri_alloc" => {
                        default(this)?;
                        return interp_ok(EmulateItemResult::NeedsReturn);
                    }
                    _ => return this.emulate_allocator(default),
                }
            }
            name if name == this.mangle_internal_symbol("__rust_alloc_zeroed") => {
                return this.emulate_allocator(|this| {
                    // See the comment for `__rust_alloc` why `check_shim` is only called in the
                    // default case.
                    let [size, align] =
                        this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                    let size = this.read_target_usize(size)?;
                    let align = this.read_target_usize(align)?;

                    this.check_rustc_alloc_request(size, align)?;

                    let ptr = this.allocate_ptr(
                        Size::from_bytes(size),
                        Align::from_bytes(align).unwrap(),
                        MiriMemoryKind::Rust.into(),
                        AllocInit::Zero,
                    )?;
                    this.write_pointer(ptr, dest)
                });
            }
            name if name == this.mangle_internal_symbol("__rust_dealloc")
                || name == "miri_dealloc" =>
            {
                let default = |ecx: &mut MiriInterpCx<'tcx>| {
                    // See the comment for `__rust_alloc` why `check_shim` is only called in the
                    // default case.
                    let [ptr, old_size, align] =
                        ecx.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                    let ptr = ecx.read_pointer(ptr)?;
                    let old_size = ecx.read_target_usize(old_size)?;
                    let align = ecx.read_target_usize(align)?;

                    let memory_kind = match link_name.as_str() {
                        "miri_dealloc" => MiriMemoryKind::Miri,
                        _ => MiriMemoryKind::Rust,
                    };

                    // No need to check old_size/align; we anyway check that they match the allocation.
                    ecx.deallocate_ptr(
                        ptr,
                        Some((Size::from_bytes(old_size), Align::from_bytes(align).unwrap())),
                        memory_kind.into(),
                    )
                };

                match link_name.as_str() {
                    "miri_dealloc" => {
                        default(this)?;
                        return interp_ok(EmulateItemResult::NeedsReturn);
                    }
                    _ => return this.emulate_allocator(default),
                }
            }
            name if name == this.mangle_internal_symbol("__rust_realloc") => {
                return this.emulate_allocator(|this| {
                    // See the comment for `__rust_alloc` why `check_shim` is only called in the
                    // default case.
                    let [ptr, old_size, align, new_size] =
                        this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                    let ptr = this.read_pointer(ptr)?;
                    let old_size = this.read_target_usize(old_size)?;
                    let align = this.read_target_usize(align)?;
                    let new_size = this.read_target_usize(new_size)?;
                    // No need to check old_size; we anyway check that they match the allocation.

                    this.check_rustc_alloc_request(new_size, align)?;

                    let align = Align::from_bytes(align).unwrap();
                    let new_ptr = this.reallocate_ptr(
                        ptr,
                        Some((Size::from_bytes(old_size), align)),
                        Size::from_bytes(new_size),
                        align,
                        MiriMemoryKind::Rust.into(),
                        AllocInit::Uninit,
                    )?;
                    this.write_pointer(new_ptr, dest)
                });
            }
            name if name == this.mangle_internal_symbol("__rust_no_alloc_shim_is_unstable_v2") => {
                // This is a no-op shim that only exists to prevent making the allocator shims instantly stable.
                let [] = this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
            }
            name if name
                == this.mangle_internal_symbol("__rust_alloc_error_handler_should_panic_v2") =>
            {
                // Gets the value of the `oom` option.
                let [] = this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                let val = this.tcx.sess.opts.unstable_opts.oom.should_panic();
                this.write_int(val, dest)?;
            }

            // C memory handling functions
            "memcmp" => {
                let [left, right, n] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let left = this.read_pointer(left)?;
                let right = this.read_pointer(right)?;
                let n = Size::from_bytes(this.read_target_usize(n)?);

                // C requires that this must always be a valid pointer (C18 §7.1.4).
                this.ptr_get_alloc_id(left, 0)?;
                this.ptr_get_alloc_id(right, 0)?;

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
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let val = this.read_scalar(val)?.to_i32()?;
                let num = this.read_target_usize(num)?;
                // The docs say val is "interpreted as unsigned char".
                #[expect(clippy::as_conversions)]
                let val = val as u8;

                // C requires that this must always be a valid pointer (C18 §7.1.4).
                this.ptr_get_alloc_id(ptr, 0)?;

                if let Some(idx) = this
                    .read_bytes_ptr_strip_provenance(ptr, Size::from_bytes(num))?
                    .iter()
                    .rev()
                    .position(|&c| c == val)
                {
                    let idx = u64::try_from(idx).unwrap();
                    #[expect(clippy::arithmetic_side_effects)] // idx < num, so this never wraps
                    let new_ptr = ptr.wrapping_offset(Size::from_bytes(num - idx - 1), this);
                    this.write_pointer(new_ptr, dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }
            "memchr" => {
                let [ptr, val, num] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let val = this.read_scalar(val)?.to_i32()?;
                let num = this.read_target_usize(num)?;
                // The docs say val is "interpreted as unsigned char".
                #[expect(clippy::as_conversions)]
                let val = val as u8;

                // C requires that this must always be a valid pointer (C18 §7.1.4).
                this.ptr_get_alloc_id(ptr, 0)?;

                let idx = this
                    .read_bytes_ptr_strip_provenance(ptr, Size::from_bytes(num))?
                    .iter()
                    .position(|&c| c == val);
                if let Some(idx) = idx {
                    let new_ptr = ptr.wrapping_offset(Size::from_bytes(idx), this);
                    this.write_pointer(new_ptr, dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }
            "strlen" => {
                let [ptr] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                // This reads at least 1 byte, so we are already enforcing that this is a valid pointer.
                let n = this.read_c_str(ptr)?.len();
                this.write_scalar(
                    Scalar::from_target_usize(u64::try_from(n).unwrap(), this),
                    dest,
                )?;
            }
            "wcslen" => {
                let [ptr] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                // This reads at least 1 byte, so we are already enforcing that this is a valid pointer.
                let n = this.read_wchar_t_str(ptr)?.len();
                this.write_scalar(
                    Scalar::from_target_usize(u64::try_from(n).unwrap(), this),
                    dest,
                )?;
            }
            "memcpy" => {
                let [ptr_dest, ptr_src, n] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let ptr_dest = this.read_pointer(ptr_dest)?;
                let ptr_src = this.read_pointer(ptr_src)?;
                let n = this.read_target_usize(n)?;

                // C requires that this must always be a valid pointer, even if `n` is zero, so we better check that.
                // (This is more than Rust requires, so `mem_copy` is not sufficient.)
                this.ptr_get_alloc_id(ptr_dest, 0)?;
                this.ptr_get_alloc_id(ptr_src, 0)?;

                this.mem_copy(ptr_src, ptr_dest, Size::from_bytes(n), true)?;
                this.write_pointer(ptr_dest, dest)?;
            }
            "strcpy" => {
                let [ptr_dest, ptr_src] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let ptr_dest = this.read_pointer(ptr_dest)?;
                let ptr_src = this.read_pointer(ptr_src)?;

                // We use `read_c_str` to determine the amount of data to copy,
                // and then use `mem_copy` for the actual copy. This means
                // pointer provenance is preserved by this implementation of `strcpy`.
                // That is probably overly cautious, but there also is no fundamental
                // reason to have `strcpy` destroy pointer provenance.
                // This reads at least 1 byte, so we are already enforcing that this is a valid pointer.
                let n = this.read_c_str(ptr_src)?.len().strict_add(1);
                this.mem_copy(ptr_src, ptr_dest, Size::from_bytes(n), true)?;
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
            | "tgammaf"
            | "erff"
            | "erfcf"
            => {
                let [f] = this.check_shim_sig_lenient(abi, CanonAbi::C , link_name, args)?;
                let f = this.read_scalar(f)?.to_f32()?;

                let res = math::fixed_float_value(this, link_name.as_str(), &[f]).unwrap_or_else(|| {
                    // Using host floats (but it's fine, these operations do not have
                    // guaranteed precision).
                    let f_host = f.to_host();
                    let res = match link_name.as_str() {
                        "cbrtf" => f_host.cbrt(),
                        "coshf" => f_host.cosh(),
                        "sinhf" => f_host.sinh(),
                        "tanf" => f_host.tan(),
                        "tanhf" => f_host.tanh(),
                        "acosf" => f_host.acos(),
                        "asinf" => f_host.asin(),
                        "atanf" => f_host.atan(),
                        "log1pf" => f_host.ln_1p(),
                        "expm1f" => f_host.exp_m1(),
                        "tgammaf" => f_host.gamma(),
                        "erff" => f_host.erf(),
                        "erfcf" => f_host.erfc(),
                        _ => bug!(),
                    };
                    let res = res.to_soft();
                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    let res = math::apply_random_float_error_ulp(this, res, 4);

                    // Clamp the result to the guaranteed range of this function according to the C standard,
                    // if any.
                    math::clamp_float_value(link_name.as_str(), res)
                });
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            #[rustfmt::skip]
            | "_hypotf"
            | "hypotf"
            | "atan2f"
            | "fdimf"
            => {
                let [f1, f2] = this.check_shim_sig_lenient(abi, CanonAbi::C , link_name, args)?;
                let f1 = this.read_scalar(f1)?.to_f32()?;
                let f2 = this.read_scalar(f2)?.to_f32()?;

                let res = math::fixed_float_value(this, link_name.as_str(), &[f1, f2])
                    .unwrap_or_else(|| {
                        let res = match link_name.as_str() {
                            // underscore case for windows, here and below
                            // (see https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/floating-point-primitives?view=vs-2019)
                            // Using host floats (but it's fine, these operations do not have guaranteed precision).
                            "_hypotf" | "hypotf" => f1.to_host().hypot(f2.to_host()).to_soft(),
                            "atan2f" => f1.to_host().atan2(f2.to_host()).to_soft(),
                            #[allow(deprecated)]
                            "fdimf" => f1.to_host().abs_sub(f2.to_host()).to_soft(),
                            _ => bug!(),
                        };
                        // Apply a relative error of 4ULP to introduce some non-determinism
                        // simulating imprecise implementations and optimizations.
                        let res = math::apply_random_float_error_ulp(this, res, 4);

                        // Clamp the result to the guaranteed range of this function according to the C standard,
                        // if any.
                        math::clamp_float_value(link_name.as_str(), res)
                    });
                let res = this.adjust_nan(res, &[f1, f2]);
                this.write_scalar(res, dest)?;
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
            | "tgamma"
            | "erf"
            | "erfc"
            => {
                let [f] = this.check_shim_sig_lenient(abi, CanonAbi::C , link_name, args)?;
                let f = this.read_scalar(f)?.to_f64()?;

                let res = math::fixed_float_value(this, link_name.as_str(), &[f]).unwrap_or_else(|| {
                    // Using host floats (but it's fine, these operations do not have
                    // guaranteed precision).
                    let f_host = f.to_host();
                    let res = match link_name.as_str() {
                        "cbrt" => f_host.cbrt(),
                        "cosh" => f_host.cosh(),
                        "sinh" => f_host.sinh(),
                        "tan" => f_host.tan(),
                        "tanh" => f_host.tanh(),
                        "acos" => f_host.acos(),
                        "asin" => f_host.asin(),
                        "atan" => f_host.atan(),
                        "log1p" => f_host.ln_1p(),
                        "expm1" => f_host.exp_m1(),
                        "tgamma" => f_host.gamma(),
                        "erf" => f_host.erf(),
                        "erfc" => f_host.erfc(),
                        _ => bug!(),
                    };
                    let res = res.to_soft();
                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    let res = math::apply_random_float_error_ulp(this, res, 4);

                    // Clamp the result to the guaranteed range of this function according to the C standard,
                    // if any.
                    math::clamp_float_value(link_name.as_str(), res)
                });
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            #[rustfmt::skip]
            | "_hypot"
            | "hypot"
            | "atan2"
            | "fdim"
            => {
                let [f1, f2] = this.check_shim_sig_lenient(abi, CanonAbi::C , link_name, args)?;
                let f1 = this.read_scalar(f1)?.to_f64()?;
                let f2 = this.read_scalar(f2)?.to_f64()?;

                let res = math::fixed_float_value(this, link_name.as_str(), &[f1, f2]).unwrap_or_else(|| {
                    let res = match link_name.as_str() {
                        // underscore case for windows, here and below
                        // (see https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/floating-point-primitives?view=vs-2019)
                        // Using host floats (but it's fine, these operations do not have guaranteed precision).
                        "_hypot" | "hypot" => f1.to_host().hypot(f2.to_host()).to_soft(),
                        "atan2" => f1.to_host().atan2(f2.to_host()).to_soft(),
                        #[allow(deprecated)]
                        "fdim" => f1.to_host().abs_sub(f2.to_host()).to_soft(),
                        _ => bug!(),
                    };
                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    let res = math::apply_random_float_error_ulp(this, res, 4);

                    // Clamp the result to the guaranteed range of this function according to the C standard,
                    // if any.
                    math::clamp_float_value(link_name.as_str(), res)
                });
                let res = this.adjust_nan(res, &[f1, f2]);
                this.write_scalar(res, dest)?;
            }
            #[rustfmt::skip]
            | "_ldexp"
            | "ldexp"
            | "scalbn"
            => {
                let [x, exp] = this.check_shim_sig_lenient(abi, CanonAbi::C , link_name, args)?;
                // For radix-2 (binary) systems, `ldexp` and `scalbn` are the same.
                let x = this.read_scalar(x)?.to_f64()?;
                let exp = this.read_scalar(exp)?.to_i32()?;

                let res = x.scalbn(exp);
                let res = this.adjust_nan(res, &[x]);
                this.write_scalar(res, dest)?;
            }
            "lgammaf_r" => {
                let [x, signp] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let x = this.read_scalar(x)?.to_f32()?;
                let signp = this.deref_pointer_as(signp, this.machine.layouts.i32)?;

                // Using host floats (but it's fine, these operations do not have guaranteed precision).
                let (res, sign) = x.to_host().ln_gamma();
                this.write_int(sign, &signp)?;

                let res = res.to_soft();
                // Apply a relative error of 4ULP to introduce some non-determinism
                // simulating imprecise implementations and optimizations.
                let res = math::apply_random_float_error_ulp(this, res, 4);
                // Clamp the result to the guaranteed range of this function according to the C standard,
                // if any.
                let res = math::clamp_float_value(link_name.as_str(), res);
                let res = this.adjust_nan(res, &[x]);
                this.write_scalar(res, dest)?;
            }
            "lgamma_r" => {
                let [x, signp] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let x = this.read_scalar(x)?.to_f64()?;
                let signp = this.deref_pointer_as(signp, this.machine.layouts.i32)?;

                // Using host floats (but it's fine, these operations do not have guaranteed precision).
                let (res, sign) = x.to_host().ln_gamma();
                this.write_int(sign, &signp)?;

                let res = res.to_soft();
                // Apply a relative error of 4ULP to introduce some non-determinism
                // simulating imprecise implementations and optimizations.
                let res = math::apply_random_float_error_ulp(this, res, 4);
                // Clamp the result to the guaranteed range of this function according to the C standard,
                // if any.
                let res = math::clamp_float_value(link_name.as_str(), res);
                let res = this.adjust_nan(res, &[x]);
                this.write_scalar(res, dest)?;
            }

            // LLVM intrinsics
            "llvm.prefetch" => {
                let [p, rw, loc, ty] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let _ = this.read_pointer(p)?;
                let rw = this.read_scalar(rw)?.to_i32()?;
                let loc = this.read_scalar(loc)?.to_i32()?;
                let ty = this.read_scalar(ty)?.to_i32()?;

                if ty == 1 {
                    // Data cache prefetch.
                    // Notably, we do not have to check the pointer, this operation is never UB!

                    if !matches!(rw, 0 | 1) {
                        throw_unsup_format!("invalid `rw` value passed to `llvm.prefetch`: {}", rw);
                    }
                    if !matches!(loc, 0..=3) {
                        throw_unsup_format!(
                            "invalid `loc` value passed to `llvm.prefetch`: {}",
                            loc
                        );
                    }
                } else {
                    throw_unsup_format!("unsupported `llvm.prefetch` type argument: {}", ty);
                }
            }
            // Used to implement the x86 `_mm{,256,512}_popcnt_epi{8,16,32,64}` and wasm
            // `{i,u}8x16_popcnt` functions.
            name if name.starts_with("llvm.ctpop.v") => {
                let [op] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let (op, op_len) = this.project_to_simd(op)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, op_len);

                for i in 0..dest_len {
                    let op = this.read_immediate(&this.project_index(&op, i)?)?;
                    // Use `to_uint` to get a zero-extended `u128`. Those
                    // extra zeros will not affect `count_ones`.
                    let res = op.to_scalar().to_uint(op.layout.size)?.count_ones();

                    this.write_scalar(
                        Scalar::from_uint(res, op.layout.size),
                        &this.project_index(&dest, i)?,
                    )?;
                }
            }

            // Target-specific shims
            name if name.starts_with("llvm.x86.")
                && (this.tcx.sess.target.arch == "x86"
                    || this.tcx.sess.target.arch == "x86_64") =>
            {
                return shims::x86::EvalContextExt::emulate_x86_intrinsic(
                    this, link_name, abi, args, dest,
                );
            }
            name if name.starts_with("llvm.aarch64.") && this.tcx.sess.target.arch == "aarch64" => {
                return shims::aarch64::EvalContextExt::emulate_aarch64_intrinsic(
                    this, link_name, abi, args, dest,
                );
            }
            // FIXME: Move this to an `arm` submodule.
            "llvm.arm.hint" if this.tcx.sess.target.arch == "arm" => {
                let [arg] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let arg = this.read_scalar(arg)?.to_i32()?;
                // Note that different arguments might have different target feature requirements.
                match arg {
                    // YIELD
                    1 => {
                        this.expect_target_feature_for_intrinsic(link_name, "v6")?;
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
                    _ if this.target_os_is_unix() =>
                        shims::unix::foreign_items::EvalContextExt::emulate_foreign_item_inner(
                            this, link_name, abi, args, dest,
                        ),
                    "wasi" =>
                        shims::wasi::foreign_items::EvalContextExt::emulate_foreign_item_inner(
                            this, link_name, abi, args, dest,
                        ),
                    "windows" =>
                        shims::windows::foreign_items::EvalContextExt::emulate_foreign_item_inner(
                            this, link_name, abi, args, dest,
                        ),
                    _ => interp_ok(EmulateItemResult::NotSupported),
                },
        };
        // We only fall through to here if we did *not* hit the `_` arm above,
        // i.e., if we actually emulated the function with one of the shims.
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
