use std::{
    collections::hash_map::Entry,
    convert::{TryFrom, TryInto},
    iter,
};

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
use rustc_span::{symbol::sym, Symbol};
use rustc_target::{
    abi::{Align, Size},
    spec::abi::Abi,
};

use super::backtrace::EvalContextExt as _;
use crate::*;

/// Returned by `emulate_foreign_item_by_name`.
pub enum EmulateByNameResult<'mir, 'tcx> {
    /// The caller is expected to jump to the return block.
    NeedsJumping,
    /// Jumping has already been taken care of.
    AlreadyJumped,
    /// A MIR body has been found for the function
    MirBody(&'mir mir::Body<'tcx>, ty::Instance<'tcx>),
    /// The item is not supported.
    NotSupported,
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    /// Returns the minimum alignment for the target architecture for allocations of the given size.
    fn min_align(&self, size: u64, kind: MiriMemoryKind) -> Align {
        let this = self.eval_context_ref();
        // List taken from `libstd/sys_common/alloc.rs`.
        let min_align = match this.tcx.sess.target.arch.as_str() {
            "x86" | "arm" | "mips" | "powerpc" | "powerpc64" | "asmjs" | "wasm32" => 8,
            "x86_64" | "aarch64" | "mips64" | "s390x" | "sparc64" => 16,
            arch => bug!("Unsupported target architecture: {}", arch),
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
    ) -> InterpResult<'tcx, Pointer<Option<Tag>>> {
        let this = self.eval_context_mut();
        if size == 0 {
            Ok(Pointer::null())
        } else {
            let align = this.min_align(size, kind);
            let ptr = this.memory.allocate(Size::from_bytes(size), align, kind.into())?;
            if zero_init {
                // We just allocated this, the access is definitely in-bounds.
                this.memory.write_bytes(ptr.into(), iter::repeat(0u8).take(size as usize)).unwrap();
            }
            Ok(ptr.into())
        }
    }

    fn free(&mut self, ptr: Pointer<Option<Tag>>, kind: MiriMemoryKind) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        if !this.ptr_is_null(ptr)? {
            this.memory.deallocate(ptr, None, kind.into())?;
        }
        Ok(())
    }

    fn realloc(
        &mut self,
        old_ptr: Pointer<Option<Tag>>,
        new_size: u64,
        kind: MiriMemoryKind,
    ) -> InterpResult<'tcx, Pointer<Option<Tag>>> {
        let this = self.eval_context_mut();
        let new_align = this.min_align(new_size, kind);
        if this.ptr_is_null(old_ptr)? {
            if new_size == 0 {
                Ok(Pointer::null())
            } else {
                let new_ptr =
                    this.memory.allocate(Size::from_bytes(new_size), new_align, kind.into())?;
                Ok(new_ptr.into())
            }
        } else {
            if new_size == 0 {
                this.memory.deallocate(old_ptr, None, kind.into())?;
                Ok(Pointer::null())
            } else {
                let new_ptr = this.memory.reallocate(
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
                        (linkage != Linkage::NotLinked).then_some(CrateNum::new(num + 1))
                    }),
                ) {
                    // We can ignore `_export_level` here: we are a Rust crate, and everything is exported
                    // from a Rust crate.
                    for &(symbol, _export_level) in tcx.exported_symbols(cnum) {
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
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(&PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
        unwind: StackPopUnwind,
    ) -> InterpResult<'tcx, Option<(&'mir mir::Body<'tcx>, ty::Instance<'tcx>)>> {
        let this = self.eval_context_mut();
        let attrs = this.tcx.get_attrs(def_id);
        let link_name = this
            .tcx
            .sess
            .first_attr_value_str_by_name(&attrs, sym::link_name)
            .unwrap_or_else(|| this.tcx.item_name(def_id));
        let tcx = this.tcx.tcx;

        // First: functions that diverge.
        let (dest, ret) = match ret {
            None =>
                match &*link_name.as_str() {
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
                            &*this.load_mir(panic_impl_instance.def, None)?,
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
                        let &[ref code] = this.check_shim(abi, exp_abi, link_name, args)?;
                        // it's really u32 for ExitProcess, but we have to put it into the `Exit` variant anyway
                        let code = this.read_scalar(code)?.to_i32()?;
                        throw_machine_stop!(TerminationInfo::Exit(code.into()));
                    }
                    "abort" => {
                        let &[] =
                            this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                        throw_machine_stop!(TerminationInfo::Abort(
                            "the program aborted execution".to_owned()
                        ))
                    }
                    _ => {
                        if let Some(body) = this.lookup_exported_symbol(link_name)? {
                            return Ok(Some(body));
                        }
                        this.handle_unsupported(format!(
                            "can't call (diverging) foreign function: {}",
                            link_name
                        ))?;
                        return Ok(None);
                    }
                },
            Some(p) => p,
        };

        // Second: functions that return.
        match this.emulate_foreign_item_by_name(link_name, abi, args, dest, ret)? {
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

                this.handle_unsupported(format!("can't call foreign function: {}", link_name))?;
                return Ok(None);
            }
        }

        Ok(None)
    }

    /// Emulates calling the internal __rust_* allocator functions
    fn emulate_allocator(
        &mut self,
        symbol: Symbol,
        default: impl FnOnce(&mut MiriEvalContext<'mir, 'tcx>) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx, EmulateByNameResult<'mir, 'tcx>> {
        let this = self.eval_context_mut();

        let allocator_kind = if let Some(allocator_kind) = this.tcx.allocator_kind(()) {
            allocator_kind
        } else {
            // in real code, this symbol does not exist without an allocator
            return Ok(EmulateByNameResult::NotSupported);
        };

        match allocator_kind {
            AllocatorKind::Global => {
                let (body, instance) = this
                    .lookup_exported_symbol(symbol)?
                    .expect("symbol should be present if there is a global allocator");

                Ok(EmulateByNameResult::MirBody(body, instance))
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
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        ret: mir::BasicBlock,
    ) -> InterpResult<'tcx, EmulateByNameResult<'mir, 'tcx>> {
        let this = self.eval_context_mut();

        // Here we dispatch all the shims for foreign functions. If you have a platform specific
        // shim, add it to the corresponding submodule.
        match &*link_name.as_str() {
            // Miri-specific extern functions
            "miri_static_root" => {
                let &[ref ptr] = this.check_shim(abi, Abi::Rust, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let (alloc_id, offset, _) = this.memory.ptr_get_alloc(ptr)?;
                if offset != Size::ZERO {
                    throw_unsup_format!("pointer passed to miri_static_root must point to beginning of an allocated block");
                }
                this.machine.static_roots.push(alloc_id);
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


            // Standard C allocation
            "malloc" => {
                let &[ref size] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let size = this.read_scalar(size)?.to_machine_usize(this)?;
                let res = this.malloc(size, /*zero_init:*/ false, MiriMemoryKind::C)?;
                this.write_pointer(res, dest)?;
            }
            "calloc" => {
                let &[ref items, ref len] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let items = this.read_scalar(items)?.to_machine_usize(this)?;
                let len = this.read_scalar(len)?.to_machine_usize(this)?;
                let size =
                    items.checked_mul(len).ok_or_else(|| err_ub_format!("overflow during calloc size computation"))?;
                let res = this.malloc(size, /*zero_init:*/ true, MiriMemoryKind::C)?;
                this.write_pointer(res, dest)?;
            }
            "free" => {
                let &[ref ptr] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                this.free(ptr, MiriMemoryKind::C)?;
            }
            "realloc" => {
                let &[ref old_ptr, ref new_size] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let old_ptr = this.read_pointer(old_ptr)?;
                let new_size = this.read_scalar(new_size)?.to_machine_usize(this)?;
                let res = this.realloc(old_ptr, new_size, MiriMemoryKind::C)?;
                this.write_pointer(res, dest)?;
            }

            // Rust allocation
            "__rust_alloc" => {
                let &[ref size, ref align] = this.check_shim(abi, Abi::Rust, link_name, args)?;
                let size = this.read_scalar(size)?.to_machine_usize(this)?;
                let align = this.read_scalar(align)?.to_machine_usize(this)?;

                return this.emulate_allocator(Symbol::intern("__rg_alloc"), |this| {
                    Self::check_alloc_request(size, align)?;

                    let ptr = this.memory.allocate(
                        Size::from_bytes(size),
                        Align::from_bytes(align).unwrap(),
                        MiriMemoryKind::Rust.into(),
                    )?;

                    this.write_pointer(ptr, dest)
                });
            }
            "__rust_alloc_zeroed" => {
                let &[ref size, ref align] = this.check_shim(abi, Abi::Rust, link_name, args)?;
                let size = this.read_scalar(size)?.to_machine_usize(this)?;
                let align = this.read_scalar(align)?.to_machine_usize(this)?;

                return this.emulate_allocator(Symbol::intern("__rg_alloc_zeroed"), |this| {
                    Self::check_alloc_request(size, align)?;

                    let ptr = this.memory.allocate(
                        Size::from_bytes(size),
                        Align::from_bytes(align).unwrap(),
                        MiriMemoryKind::Rust.into(),
                    )?;

                    // We just allocated this, the access is definitely in-bounds.
                    this.memory.write_bytes(ptr.into(), iter::repeat(0u8).take(usize::try_from(size).unwrap())).unwrap();
                    this.write_pointer(ptr, dest)
                });
            }
            "__rust_dealloc" => {
                let &[ref ptr, ref old_size, ref align] = this.check_shim(abi, Abi::Rust, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let old_size = this.read_scalar(old_size)?.to_machine_usize(this)?;
                let align = this.read_scalar(align)?.to_machine_usize(this)?;

                return this.emulate_allocator(Symbol::intern("__rg_dealloc"), |this| {
                    // No need to check old_size/align; we anyway check that they match the allocation.
                    this.memory.deallocate(
                        ptr,
                        Some((Size::from_bytes(old_size), Align::from_bytes(align).unwrap())),
                        MiriMemoryKind::Rust.into(),
                    )
                });
            }
            "__rust_realloc" => {
                let &[ref ptr, ref old_size, ref align, ref new_size] = this.check_shim(abi, Abi::Rust, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let old_size = this.read_scalar(old_size)?.to_machine_usize(this)?;
                let align = this.read_scalar(align)?.to_machine_usize(this)?;
                let new_size = this.read_scalar(new_size)?.to_machine_usize(this)?;
                // No need to check old_size; we anyway check that they match the allocation.

                return this.emulate_allocator(Symbol::intern("__rg_realloc"), |this| {
                    Self::check_alloc_request(new_size, align)?;

                    let align = Align::from_bytes(align).unwrap();
                    let new_ptr = this.memory.reallocate(
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
                let &[ref left, ref right, ref n] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let left = this.read_pointer(left)?;
                let right = this.read_pointer(right)?;
                let n = Size::from_bytes(this.read_scalar(n)?.to_machine_usize(this)?);

                let result = {
                    let left_bytes = this.memory.read_bytes(left, n)?;
                    let right_bytes = this.memory.read_bytes(right, n)?;

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
                let &[ref ptr, ref val, ref num] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let val = this.read_scalar(val)?.to_i32()? as u8;
                let num = this.read_scalar(num)?.to_machine_usize(this)?;
                if let Some(idx) = this
                    .memory
                    .read_bytes(ptr, Size::from_bytes(num))?
                    .iter()
                    .rev()
                    .position(|&c| c == val)
                {
                    let new_ptr = ptr.offset(Size::from_bytes(num - idx as u64 - 1), this)?;
                    this.write_pointer(new_ptr, dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }
            "memchr" => {
                let &[ref ptr, ref val, ref num] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let val = this.read_scalar(val)?.to_i32()? as u8;
                let num = this.read_scalar(num)?.to_machine_usize(this)?;
                let idx = this
                    .memory
                    .read_bytes(ptr, Size::from_bytes(num))?
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
                let &[ref ptr] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let n = this.read_c_str(ptr)?.len();
                this.write_scalar(Scalar::from_machine_usize(u64::try_from(n).unwrap(), this), dest)?;
            }

            // math functions
            #[rustfmt::skip]
            | "cbrtf"
            | "coshf"
            | "sinhf"
            | "tanf"
            | "acosf"
            | "asinf"
            | "atanf"
            => {
                let &[ref f] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(f)?.to_u32()?);
                let f = match &*link_name.as_str() {
                    "cbrtf" => f.cbrt(),
                    "coshf" => f.cosh(),
                    "sinhf" => f.sinh(),
                    "tanf" => f.tan(),
                    "acosf" => f.acos(),
                    "asinf" => f.asin(),
                    "atanf" => f.atan(),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u32(f.to_bits()), dest)?;
            }
            #[rustfmt::skip]
            | "_hypotf"
            | "hypotf"
            | "atan2f"
            => {
                let &[ref f1, ref f2] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // underscore case for windows, here and below
                // (see https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/floating-point-primitives?view=vs-2019)
                // FIXME: Using host floats.
                let f1 = f32::from_bits(this.read_scalar(f1)?.to_u32()?);
                let f2 = f32::from_bits(this.read_scalar(f2)?.to_u32()?);
                let n = match &*link_name.as_str() {
                    "_hypotf" | "hypotf" => f1.hypot(f2),
                    "atan2f" => f1.atan2(f2),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u32(n.to_bits()), dest)?;
            }
            #[rustfmt::skip]
            | "cbrt"
            | "cosh"
            | "sinh"
            | "tan"
            | "acos"
            | "asin"
            | "atan"
            => {
                let &[ref f] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(f)?.to_u64()?);
                let f = match &*link_name.as_str() {
                    "cbrt" => f.cbrt(),
                    "cosh" => f.cosh(),
                    "sinh" => f.sinh(),
                    "tan" => f.tan(),
                    "acos" => f.acos(),
                    "asin" => f.asin(),
                    "atan" => f.atan(),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u64(f.to_bits()), dest)?;
            }
            #[rustfmt::skip]
            | "_hypot"
            | "hypot"
            | "atan2"
            => {
                let &[ref f1, ref f2] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // FIXME: Using host floats.
                let f1 = f64::from_bits(this.read_scalar(f1)?.to_u64()?);
                let f2 = f64::from_bits(this.read_scalar(f2)?.to_u64()?);
                let n = match &*link_name.as_str() {
                    "_hypot" | "hypot" => f1.hypot(f2),
                    "atan2" => f1.atan2(f2),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u64(n.to_bits()), dest)?;
            }
            #[rustfmt::skip]
            | "_ldexp"
            | "ldexp"
            | "scalbn"
            => {
                let &[ref x, ref exp] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // For radix-2 (binary) systems, `ldexp` and `scalbn` are the same.
                let x = this.read_scalar(x)?.to_f64()?;
                let exp = this.read_scalar(exp)?.to_i32()?;

                // Saturating cast to i16. Even those are outside the valid exponent range to
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
            "llvm.x86.sse2.pause" if this.tcx.sess.target.arch == "x86" || this.tcx.sess.target.arch == "x86_64" => {
                let &[] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.yield_active_thread();
            }
            "llvm.aarch64.isb" if this.tcx.sess.target.arch == "aarch64" => {
                let &[ref arg] = this.check_shim(abi, Abi::Unadjusted, link_name, args)?;
                let arg = this.read_scalar(arg)?.to_i32()?;
                match arg {
                    15 => { // SY ("full system scope")
                        this.yield_active_thread();
                    }
                    _ => {
                        throw_unsup_format!("unsupported llvm.aarch64.isb argument {}", arg);
                    }
                }
            }

            // Platform-specific shims
            _ => match this.tcx.sess.target.os.as_str() {
                "linux" | "macos" => return shims::posix::foreign_items::EvalContextExt::emulate_foreign_item_by_name(this, link_name, abi, args, dest, ret),
                "windows" => return shims::windows::foreign_items::EvalContextExt::emulate_foreign_item_by_name(this, link_name, abi, args, dest, ret),
                target => throw_unsup_format!("the target `{}` is not supported", target),
            }
        };

        // We only fall through to here if we did *not* hit the `_` arm above,
        // i.e., if we actually emulated the function.
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
