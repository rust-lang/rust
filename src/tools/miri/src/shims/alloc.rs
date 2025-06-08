use rustc_abi::{Align, Size};
use rustc_ast::expand::allocator::AllocatorKind;

use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Returns the alignment that `malloc` would guarantee for requests of the given size.
    fn malloc_align(&self, size: u64) -> Align {
        let this = self.eval_context_ref();
        // The C standard says: "The pointer returned if the allocation succeeds is suitably aligned
        // so that it may be assigned to a pointer to any type of object with a fundamental
        // alignment requirement and size less than or equal to the size requested."
        // So first we need to figure out what the limits are for "fundamental alignment".
        // This is given by `alignof(max_align_t)`. The following list is taken from
        // `library/std/src/sys/alloc/mod.rs` (where this is called `MIN_ALIGN`) and should
        // be kept in sync.
        let max_fundamental_align = match this.tcx.sess.target.arch.as_ref() {
            "x86" | "arm" | "loongarch32" | "mips" | "mips32r6" | "powerpc" | "powerpc64"
            | "wasm32" => 8,
            "x86_64" | "aarch64" | "mips64" | "mips64r6" | "s390x" | "sparc64" | "loongarch64" =>
                16,
            arch => bug!("unsupported target architecture for malloc: `{}`", arch),
        };
        // The C standard only requires sufficient alignment for any *type* with size less than or
        // equal to the size requested. Types one can define in standard C seem to never have an alignment
        // bigger than their size. So if the size is 2, then only alignment 2 is guaranteed, even if
        // `max_fundamental_align` is bigger.
        // This matches what some real-world implementations do, see e.g.
        // - https://github.com/jemalloc/jemalloc/issues/1533
        // - https://github.com/llvm/llvm-project/issues/53540
        // - https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2293.htm
        if size >= max_fundamental_align {
            return Align::from_bytes(max_fundamental_align).unwrap();
        }
        // C doesn't have zero-sized types, so presumably nothing is guaranteed here.
        if size == 0 {
            return Align::ONE;
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

    /// Emulates calling the internal __rust_* allocator functions
    fn emulate_allocator(
        &mut self,
        default: impl FnOnce(&mut MiriInterpCx<'tcx>) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();

        let Some(allocator_kind) = this.tcx.allocator_kind(()) else {
            // in real code, this symbol does not exist without an allocator
            return interp_ok(EmulateItemResult::NotSupported);
        };

        match allocator_kind {
            AllocatorKind::Global => {
                // When `#[global_allocator]` is used, `__rust_*` is defined by the macro expansion
                // of this attribute. As such we have to call an exported Rust function,
                // and not execute any Miri shim. Somewhat unintuitively doing so is done
                // by returning `NotSupported`, which triggers the `lookup_exported_symbol`
                // fallback case in `emulate_foreign_item`.
                interp_ok(EmulateItemResult::NotSupported)
            }
            AllocatorKind::Default => {
                default(this)?;
                interp_ok(EmulateItemResult::NeedsReturn)
            }
        }
    }

    fn malloc(&mut self, size: u64, init: AllocInit) -> InterpResult<'tcx, Pointer> {
        let this = self.eval_context_mut();
        let align = this.malloc_align(size);
        let ptr =
            this.allocate_ptr(Size::from_bytes(size), align, MiriMemoryKind::C.into(), init)?;
        interp_ok(ptr.into())
    }

    fn posix_memalign(
        &mut self,
        memptr: &OpTy<'tcx>,
        align: &OpTy<'tcx>,
        size: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        let memptr = this.deref_pointer_as(memptr, this.machine.layouts.mut_raw_ptr)?;
        let align = this.read_target_usize(align)?;
        let size = this.read_target_usize(size)?;

        // Align must be power of 2, and also at least ptr-sized (POSIX rules).
        // But failure to adhere to this is not UB, it's an error condition.
        if !align.is_power_of_two() || align < this.pointer_size().bytes() {
            interp_ok(this.eval_libc("EINVAL"))
        } else {
            let ptr = this.allocate_ptr(
                Size::from_bytes(size),
                Align::from_bytes(align).unwrap(),
                MiriMemoryKind::C.into(),
                AllocInit::Uninit,
            )?;
            this.write_pointer(ptr, &memptr)?;
            interp_ok(Scalar::from_i32(0))
        }
    }

    fn free(&mut self, ptr: Pointer) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        if !this.ptr_is_null(ptr)? {
            this.deallocate_ptr(ptr, None, MiriMemoryKind::C.into())?;
        }
        interp_ok(())
    }

    fn realloc(&mut self, old_ptr: Pointer, new_size: u64) -> InterpResult<'tcx, Pointer> {
        let this = self.eval_context_mut();
        let new_align = this.malloc_align(new_size);
        if this.ptr_is_null(old_ptr)? {
            // Here we must behave like `malloc`.
            self.malloc(new_size, AllocInit::Uninit)
        } else {
            if new_size == 0 {
                // C, in their infinite wisdom, made this UB.
                // <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2464.pdf>
                throw_ub_format!("`realloc` with a size of zero");
            } else {
                let new_ptr = this.reallocate_ptr(
                    old_ptr,
                    None,
                    Size::from_bytes(new_size),
                    new_align,
                    MiriMemoryKind::C.into(),
                    AllocInit::Uninit,
                )?;
                interp_ok(new_ptr.into())
            }
        }
    }

    fn aligned_alloc(
        &mut self,
        align: &OpTy<'tcx>,
        size: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Pointer> {
        let this = self.eval_context_mut();
        let align = this.read_target_usize(align)?;
        let size = this.read_target_usize(size)?;

        // Alignment must be a power of 2, and "supported by the implementation".
        // We decide that "supported by the implementation" means that the
        // size must be a multiple of the alignment. (This restriction seems common
        // enough that it is stated on <https://en.cppreference.com/w/c/memory/aligned_alloc>
        // as a general rule, but the actual standard has no such rule.)
        // If any of these are violated, we have to return NULL.
        // All fundamental alignments must be supported.
        //
        // macOS and Illumos are buggy in that they require the alignment
        // to be at least the size of a pointer, so they do not support all fundamental
        // alignments. We do not emulate those platform bugs.
        //
        // Linux also sets errno to EINVAL, but that's non-standard behavior that we do not
        // emulate.
        // FreeBSD says some of these cases are UB but that's violating the C standard.
        // http://en.cppreference.com/w/cpp/memory/c/aligned_alloc
        // Linux: https://linux.die.net/man/3/aligned_alloc
        // FreeBSD: https://man.freebsd.org/cgi/man.cgi?query=aligned_alloc&apropos=0&sektion=3&manpath=FreeBSD+9-current&format=html
        match size.checked_rem(align) {
            Some(0) if align.is_power_of_two() => {
                let align = align.max(this.malloc_align(size).bytes());
                let ptr = this.allocate_ptr(
                    Size::from_bytes(size),
                    Align::from_bytes(align).unwrap(),
                    MiriMemoryKind::C.into(),
                    AllocInit::Uninit,
                )?;
                interp_ok(ptr.into())
            }
            _ => interp_ok(Pointer::null()),
        }
    }
}
