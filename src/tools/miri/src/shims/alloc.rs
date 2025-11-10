use rustc_abi::{Align, AlignFromBytesError, CanonAbi, Size};
use rustc_ast::expand::allocator::SpecialAllocatorMethod;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;
use rustc_target::spec::Arch;

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
        let os = this.tcx.sess.target.os.as_ref();
        let max_fundamental_align = match &this.tcx.sess.target.arch {
            Arch::RiscV32 if matches!(os, "espidf" | "zkvm") => 4,
            Arch::Xtensa if matches!(os, "espidf") => 4,
            Arch::X86
            | Arch::Arm
            | Arch::M68k
            | Arch::CSky
            | Arch::LoongArch32
            | Arch::Mips
            | Arch::Mips32r6
            | Arch::PowerPC
            | Arch::PowerPC64
            | Arch::Sparc
            | Arch::Wasm32
            | Arch::Hexagon
            | Arch::RiscV32
            | Arch::Xtensa => 8,
            Arch::X86_64
            | Arch::AArch64
            | Arch::Arm64EC
            | Arch::LoongArch64
            | Arch::Mips64
            | Arch::Mips64r6
            | Arch::S390x
            | Arch::Sparc64
            | Arch::RiscV64
            | Arch::Wasm64 => 16,
            arch @ (Arch::AmdGpu
            | Arch::Avr
            | Arch::Bpf
            | Arch::Msp430
            | Arch::Nvptx64
            | Arch::PowerPC64LE
            | Arch::SpirV
            | Arch::Unknown(_)) => bug!("unsupported target architecture for malloc: `{arch}`"),
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

    /// Check some basic requirements for this allocation request:
    /// non-zero size, power-of-two alignment.
    fn check_rust_alloc_request(&self, size: u64, align: u64) -> InterpResult<'tcx> {
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

    fn rust_special_allocator_method(
        &mut self,
        method: SpecialAllocatorMethod,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        match method {
            SpecialAllocatorMethod::Alloc | SpecialAllocatorMethod::AllocZeroed => {
                let [size, align] =
                    this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                let size = this.read_target_usize(size)?;
                let align = this.read_target_usize(align)?;

                this.check_rust_alloc_request(size, align)?;

                let ptr = this.allocate_ptr(
                    Size::from_bytes(size),
                    Align::from_bytes(align).unwrap(),
                    MiriMemoryKind::Rust.into(),
                    if matches!(method, SpecialAllocatorMethod::AllocZeroed) {
                        AllocInit::Zero
                    } else {
                        AllocInit::Uninit
                    },
                )?;

                this.write_pointer(ptr, dest)
            }
            SpecialAllocatorMethod::Dealloc => {
                let [ptr, old_size, align] =
                    this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let old_size = this.read_target_usize(old_size)?;
                let align = this.read_target_usize(align)?;

                // No need to check old_size/align; we anyway check that they match the allocation.
                this.deallocate_ptr(
                    ptr,
                    Some((Size::from_bytes(old_size), Align::from_bytes(align).unwrap())),
                    MiriMemoryKind::Rust.into(),
                )
            }
            SpecialAllocatorMethod::Realloc => {
                let [ptr, old_size, align, new_size] =
                    this.check_shim_sig_lenient(abi, CanonAbi::Rust, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let old_size = this.read_target_usize(old_size)?;
                let align = this.read_target_usize(align)?;
                let new_size = this.read_target_usize(new_size)?;
                // No need to check old_size; we anyway check that they match the allocation.

                this.check_rust_alloc_request(new_size, align)?;

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
