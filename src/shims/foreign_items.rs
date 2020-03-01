mod windows;
mod posix;

use std::{convert::TryInto, iter};

use rustc_hir::def_id::DefId;
use rustc::mir;
use rustc::ty;
use rustc::ty::layout::{Align, Size};
use rustc_apfloat::Float;
use rustc_span::symbol::sym;
use rustc_ast::attr;

use crate::*;

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    /// Returns the minimum alignment for the target architecture for allocations of the given size.
    fn min_align(&self, size: u64, kind: MiriMemoryKind) -> Align {
        let this = self.eval_context_ref();
        // List taken from `libstd/sys_common/alloc.rs`.
        let min_align = match this.tcx.tcx.sess.target.target.arch.as_str() {
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

    fn malloc(&mut self, size: u64, zero_init: bool, kind: MiriMemoryKind) -> Scalar<Tag> {
        let this = self.eval_context_mut();
        if size == 0 {
            Scalar::from_int(0, this.pointer_size())
        } else {
            let align = this.min_align(size, kind);
            let ptr = this.memory.allocate(Size::from_bytes(size), align, kind.into());
            if zero_init {
                // We just allocated this, the access is definitely in-bounds.
                this.memory.write_bytes(ptr.into(), iter::repeat(0u8).take(size as usize)).unwrap();
            }
            Scalar::Ptr(ptr)
        }
    }

    fn free(&mut self, ptr: Scalar<Tag>, kind: MiriMemoryKind) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        if !this.is_null(ptr)? {
            let ptr = this.force_ptr(ptr)?;
            this.memory.deallocate(ptr, None, kind.into())?;
        }
        Ok(())
    }

    fn realloc(
        &mut self,
        old_ptr: Scalar<Tag>,
        new_size: u64,
        kind: MiriMemoryKind,
    ) -> InterpResult<'tcx, Scalar<Tag>> {
        let this = self.eval_context_mut();
        let new_align = this.min_align(new_size, kind);
        if this.is_null(old_ptr)? {
            if new_size == 0 {
                Ok(Scalar::from_int(0, this.pointer_size()))
            } else {
                let new_ptr =
                    this.memory.allocate(Size::from_bytes(new_size), new_align, kind.into());
                Ok(Scalar::Ptr(new_ptr))
            }
        } else {
            let old_ptr = this.force_ptr(old_ptr)?;
            if new_size == 0 {
                this.memory.deallocate(old_ptr, None, kind.into())?;
                Ok(Scalar::from_int(0, this.pointer_size()))
            } else {
                let new_ptr = this.memory.reallocate(
                    old_ptr,
                    None,
                    Size::from_bytes(new_size),
                    new_align,
                    kind.into(),
                )?;
                Ok(Scalar::Ptr(new_ptr))
            }
        }
    }

    /// Emulates calling a foreign item, failing if the item is not supported.
    /// This function will handle `goto_block` if needed.
    /// Returns Ok(None) if the foreign item was completely handled
    /// by this function.
    /// Returns Ok(Some(body)) if processing the foreign item
    /// is delegated to another function.
    #[rustfmt::skip]
    fn emulate_foreign_item(
        &mut self,
        def_id: DefId,
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
        _unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, Option<&'mir mir::Body<'tcx>>> {
        let this = self.eval_context_mut();
        let attrs = this.tcx.get_attrs(def_id);
        let link_name = match attr::first_attr_value_str_by_name(&attrs, sym::link_name) {
            Some(name) => name.as_str(),
            None => this.tcx.item_name(def_id).as_str(),
        };
        // Strip linker suffixes (seen on 32-bit macOS).
        let link_name = link_name.trim_end_matches("$UNIX2003");
        let tcx = &{ this.tcx.tcx };

        // First: functions that diverge.
        let (dest, ret) = match link_name {
            // Note that this matches calls to the *foreign* item `__rust_start_panic* -
            // that is, calls to `extern "Rust" { fn __rust_start_panic(...) }`.
            // We forward this to the underlying *implementation* in the panic runtime crate.
            // Normally, this will be either `libpanic_unwind` or `libpanic_abort`, but it could
            // also be a custom user-provided implementation via `#![feature(panic_runtime)]`
            "__rust_start_panic" => {
                // FIXME we might want to cache this... but it's not really performance-critical.
                let panic_runtime = tcx
                    .crates()
                    .iter()
                    .find(|cnum| tcx.is_panic_runtime(**cnum))
                    .expect("No panic runtime found!");
                let panic_runtime = tcx.crate_name(*panic_runtime);
                let start_panic_instance =
                    this.resolve_path(&[&*panic_runtime.as_str(), "__rust_start_panic"])?;
                return Ok(Some(&*this.load_mir(start_panic_instance.def, None)?));
            }
            // Similarly, we forward calls to the `panic_impl` foreign item to its implementation.
            // The implementation is provided by the function with the `#[panic_handler]` attribute.
            "panic_impl" => {
                let panic_impl_id = this.tcx.lang_items().panic_impl().unwrap();
                let panic_impl_instance = ty::Instance::mono(*this.tcx, panic_impl_id);
                return Ok(Some(&*this.load_mir(panic_impl_instance.def, None)?));
            }

            | "exit"
            | "ExitProcess"
            => {
                // it's really u32 for ExitProcess, but we have to put it into the `Exit` variant anyway
                let code = this.read_scalar(args[0])?.to_i32()?;
                throw_machine_stop!(TerminationInfo::Exit(code.into()));
            }
            _ => {
                if let Some(p) = ret {
                    p
                } else {
                    throw_unsup_format!("can't call (diverging) foreign function: {}", link_name);
                }
            }
        };

        // Next: functions that return.
        if this.emulate_foreign_item_by_name(link_name, args, dest, ret)? {
            this.dump_place(*dest);
            this.go_to_block(ret);
        }

        Ok(None)
    }

    /// Emulates calling a foreign item using its name, failing if the item is not supported.
    /// Returns `true` if the caller is expected to jump to the return block, and `false` if
    /// jumping has already been taken care of.
    fn emulate_foreign_item_by_name(
        &mut self,
        link_name: &str,
        args: &[OpTy<'tcx, Tag>],
        dest: PlaceTy<'tcx, Tag>,
        ret: mir::BasicBlock,
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();

        // Here we dispatch all the shims for foreign functions. If you have a platform specific
        // shim, add it to the corresponding submodule.
        match link_name {
            "malloc" => {
                let size = this.read_scalar(args[0])?.to_machine_usize(this)?;
                let res = this.malloc(size, /*zero_init:*/ false, MiriMemoryKind::C);
                this.write_scalar(res, dest)?;
            }
            "calloc" => {
                let items = this.read_scalar(args[0])?.to_machine_usize(this)?;
                let len = this.read_scalar(args[1])?.to_machine_usize(this)?;
                let size =
                    items.checked_mul(len).ok_or_else(|| err_ub_format!("overflow during calloc size computation"))?;
                let res = this.malloc(size, /*zero_init:*/ true, MiriMemoryKind::C);
                this.write_scalar(res, dest)?;
            }
            "free" => {
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                this.free(ptr, MiriMemoryKind::C)?;
            }
            "realloc" => {
                let old_ptr = this.read_scalar(args[0])?.not_undef()?;
                let new_size = this.read_scalar(args[1])?.to_machine_usize(this)?;
                let res = this.realloc(old_ptr, new_size, MiriMemoryKind::C)?;
                this.write_scalar(res, dest)?;
            }

            "__rust_alloc" => {
                let size = this.read_scalar(args[0])?.to_machine_usize(this)?;
                let align = this.read_scalar(args[1])?.to_machine_usize(this)?;
                if size == 0 {
                    throw_unsup!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    throw_unsup!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = this.memory.allocate(
                    Size::from_bytes(size),
                    Align::from_bytes(align).unwrap(),
                    MiriMemoryKind::Rust.into(),
                );
                this.write_scalar(ptr, dest)?;
            }
            "__rust_alloc_zeroed" => {
                let size = this.read_scalar(args[0])?.to_machine_usize(this)?;
                let align = this.read_scalar(args[1])?.to_machine_usize(this)?;
                if size == 0 {
                    throw_unsup!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    throw_unsup!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = this.memory.allocate(
                    Size::from_bytes(size),
                    Align::from_bytes(align).unwrap(),
                    MiriMemoryKind::Rust.into(),
                );
                // We just allocated this, the access is definitely in-bounds.
                this.memory.write_bytes(ptr.into(), iter::repeat(0u8).take(size as usize)).unwrap();
                this.write_scalar(ptr, dest)?;
            }
            "__rust_dealloc" => {
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let old_size = this.read_scalar(args[1])?.to_machine_usize(this)?;
                let align = this.read_scalar(args[2])?.to_machine_usize(this)?;
                if old_size == 0 {
                    throw_unsup!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    throw_unsup!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = this.force_ptr(ptr)?;
                this.memory.deallocate(
                    ptr,
                    Some((Size::from_bytes(old_size), Align::from_bytes(align).unwrap())),
                    MiriMemoryKind::Rust.into(),
                )?;
            }
            "__rust_realloc" => {
                let old_size = this.read_scalar(args[1])?.to_machine_usize(this)?;
                let align = this.read_scalar(args[2])?.to_machine_usize(this)?;
                let new_size = this.read_scalar(args[3])?.to_machine_usize(this)?;
                if old_size == 0 || new_size == 0 {
                    throw_unsup!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    throw_unsup!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = this.force_ptr(this.read_scalar(args[0])?.not_undef()?)?;
                let align = Align::from_bytes(align).unwrap();
                let new_ptr = this.memory.reallocate(
                    ptr,
                    Some((Size::from_bytes(old_size), align)),
                    Size::from_bytes(new_size),
                    align,
                    MiriMemoryKind::Rust.into(),
                )?;
                this.write_scalar(new_ptr, dest)?;
            }

            "__rust_maybe_catch_panic" => {
                this.handle_catch_panic(args, dest, ret)?;
                return Ok(false);
            }

            "memcmp" => {
                let left = this.read_scalar(args[0])?.not_undef()?;
                let right = this.read_scalar(args[1])?.not_undef()?;
                let n = Size::from_bytes(this.read_scalar(args[2])?.to_machine_usize(this)?);

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

                this.write_scalar(Scalar::from_int(result, Size::from_bits(32)), dest)?;
            }

            "memrchr" => {
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let val = this.read_scalar(args[1])?.to_i32()? as u8;
                let num = this.read_scalar(args[2])?.to_machine_usize(this)?;
                if let Some(idx) = this
                    .memory
                    .read_bytes(ptr, Size::from_bytes(num))?
                    .iter()
                    .rev()
                    .position(|&c| c == val)
                {
                    let new_ptr = ptr.ptr_offset(Size::from_bytes(num - idx as u64 - 1), this)?;
                    this.write_scalar(new_ptr, dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }

            "memchr" => {
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let val = this.read_scalar(args[1])?.to_i32()? as u8;
                let num = this.read_scalar(args[2])?.to_machine_usize(this)?;
                let idx = this
                    .memory
                    .read_bytes(ptr, Size::from_bytes(num))?
                    .iter()
                    .position(|&c| c == val);
                if let Some(idx) = idx {
                    let new_ptr = ptr.ptr_offset(Size::from_bytes(idx as u64), this)?;
                    this.write_scalar(new_ptr, dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }

            "strlen" => {
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let n = this.memory.read_c_str(ptr)?.len();
                this.write_scalar(Scalar::from_uint(n as u64, dest.layout.size), dest)?;
            }

            // math functions
            | "cbrtf"
            | "coshf"
            | "sinhf"
            | "tanf"
            | "acosf"
            | "asinf"
            | "atanf"
            => {
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(args[0])?.to_u32()?);
                let f = match link_name {
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
            // underscore case for windows
            | "_hypotf"
            | "hypotf"
            | "atan2f"
            => {
                // FIXME: Using host floats.
                let f1 = f32::from_bits(this.read_scalar(args[0])?.to_u32()?);
                let f2 = f32::from_bits(this.read_scalar(args[1])?.to_u32()?);
                let n = match link_name {
                    "_hypotf" | "hypotf" => f1.hypot(f2),
                    "atan2f" => f1.atan2(f2),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u32(n.to_bits()), dest)?;
            }

            | "cbrt"
            | "cosh"
            | "sinh"
            | "tan"
            | "acos"
            | "asin"
            | "atan"
            => {
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(args[0])?.to_u64()?);
                let f = match link_name {
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
            // underscore case for windows, here and below
            // (see https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/floating-point-primitives?view=vs-2019)
            | "_hypot"
            | "hypot"
            | "atan2"
            => {
                // FIXME: Using host floats.
                let f1 = f64::from_bits(this.read_scalar(args[0])?.to_u64()?);
                let f2 = f64::from_bits(this.read_scalar(args[1])?.to_u64()?);
                let n = match link_name {
                    "_hypot" | "hypot" => f1.hypot(f2),
                    "atan2" => f1.atan2(f2),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u64(n.to_bits()), dest)?;
            }
            // For radix-2 (binary) systems, `ldexp` and `scalbn` are the same.
            | "_ldexp"
            | "ldexp"
            | "scalbn"
            => {
                let x = this.read_scalar(args[0])?.to_f64()?;
                let exp = this.read_scalar(args[1])?.to_i32()?;

                // Saturating cast to i16. Even those are outside the valid exponent range to
                // `scalbn` below will do its over/underflow handling.
                let exp = if exp > i16::max_value() as i32 {
                    i16::max_value()
                } else if exp < i16::min_value() as i32 {
                    i16::min_value()
                } else {
                    exp.try_into().unwrap()
                };

                let res = x.scalbn(exp);
                this.write_scalar(Scalar::from_f64(res), dest)?;
            }

            _ => match this.tcx.sess.target.target.target_os.as_str() {
                "linux" | "macos" => return posix::EvalContextExt::emulate_foreign_item_by_name(this, link_name, args, dest, ret),
                "windows" => return windows::EvalContextExt::emulate_foreign_item_by_name(this, link_name, args, dest, ret),
                target => throw_unsup_format!("The {} target platform is not supported", target),
            }
        };

        Ok(true)
    }

    /// Evaluates the scalar at the specified path. Returns Some(val)
    /// if the path could be resolved, and None otherwise
    fn eval_path_scalar(
        &mut self,
        path: &[&str],
    ) -> InterpResult<'tcx, Option<ScalarMaybeUndef<Tag>>> {
        let this = self.eval_context_mut();
        if let Ok(instance) = this.resolve_path(path) {
            let cid = GlobalId { instance, promoted: None };
            let const_val = this.const_eval_raw(cid)?;
            let const_val = this.read_scalar(const_val.into())?;
            return Ok(Some(const_val));
        }
        return Ok(None);
    }
}
