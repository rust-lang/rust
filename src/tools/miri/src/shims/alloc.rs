use std::iter;

use rustc_ast::expand::allocator::AllocatorKind;
use rustc_target::abi::{Align, Size};

use crate::*;
use shims::foreign_items::EmulateForeignItemResult;

/// Check some basic requirements for this allocation request:
/// non-zero size, power-of-two alignment.
pub(super) fn check_alloc_request<'tcx>(size: u64, align: u64) -> InterpResult<'tcx> {
    if size == 0 {
        throw_ub_format!("creating allocation with size 0");
    }
    if !align.is_power_of_two() {
        throw_ub_format!("creating allocation with non-power-of-two alignment {}", align);
    }
    Ok(())
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    /// Returns the minimum alignment for the target architecture for allocations of the given size.
    fn min_align(&self, size: u64, kind: MiriMemoryKind) -> Align {
        let this = self.eval_context_ref();
        // List taken from `library/std/src/sys/pal/common/alloc.rs`.
        // This list should be kept in sync with the one from libstd.
        let min_align = match this.tcx.sess.target.arch.as_ref() {
            "x86" | "arm" | "mips" | "mips32r6" | "powerpc" | "powerpc64" | "wasm32" => 8,
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

    /// Emulates calling the internal __rust_* allocator functions
    fn emulate_allocator(
        &mut self,
        default: impl FnOnce(&mut MiriInterpCx<'mir, 'tcx>) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx, EmulateForeignItemResult> {
        let this = self.eval_context_mut();

        let Some(allocator_kind) = this.tcx.allocator_kind(()) else {
            // in real code, this symbol does not exist without an allocator
            return Ok(EmulateForeignItemResult::NotSupported);
        };

        match allocator_kind {
            AllocatorKind::Global => {
                // When `#[global_allocator]` is used, `__rust_*` is defined by the macro expansion
                // of this attribute. As such we have to call an exported Rust function,
                // and not execute any Miri shim. Somewhat unintuitively doing so is done
                // by returning `NotSupported`, which triggers the `lookup_exported_symbol`
                // fallback case in `emulate_foreign_item`.
                return Ok(EmulateForeignItemResult::NotSupported);
            }
            AllocatorKind::Default => {
                default(this)?;
                Ok(EmulateForeignItemResult::NeedsJumping)
            }
        }
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
            // Here we must behave like `malloc`.
            if new_size == 0 {
                Ok(Pointer::null())
            } else {
                let new_ptr =
                    this.allocate_ptr(Size::from_bytes(new_size), new_align, kind.into())?;
                Ok(new_ptr.into())
            }
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
                    kind.into(),
                )?;
                Ok(new_ptr.into())
            }
        }
    }
}
