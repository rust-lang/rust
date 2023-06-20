//! This is an incomplete implementation of mmap/munmap which is restricted in order to be
//! implementable on top of the existing memory system. The point of these function as-written is
//! to allow memory allocators written entirely in Rust to be executed by Miri. This implementation
//! does not support other uses of mmap such as file mappings.
//!
//! mmap/munmap behave a lot like alloc/dealloc, and for simple use they are exactly
//! equivalent. That is the only part we support: no MAP_FIXED or MAP_SHARED or anything
//! else that goes beyond a basic allocation API.

use crate::*;
use rustc_target::abi::Size;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn mmap(
        &mut self,
        addr: &OpTy<'tcx, Provenance>,
        length: &OpTy<'tcx, Provenance>,
        prot: &OpTy<'tcx, Provenance>,
        flags: &OpTy<'tcx, Provenance>,
        fd: &OpTy<'tcx, Provenance>,
        offset: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        // We do not support MAP_FIXED, so the addr argument is always ignored (except for the MacOS hack)
        let addr = this.read_target_usize(addr)?;
        let length = this.read_target_usize(length)?;
        let prot = this.read_scalar(prot)?.to_i32()?;
        let flags = this.read_scalar(flags)?.to_i32()?;
        let fd = this.read_scalar(fd)?.to_i32()?;
        let offset = this.read_target_usize(offset)?;

        let map_private = this.eval_libc_i32("MAP_PRIVATE");
        let map_anonymous = this.eval_libc_i32("MAP_ANONYMOUS");
        let map_shared = this.eval_libc_i32("MAP_SHARED");
        let map_fixed = this.eval_libc_i32("MAP_FIXED");

        // This is a horrible hack, but on MacOS the guard page mechanism uses mmap
        // in a way we do not support. We just give it the return value it expects.
        if this.frame_in_std() && this.tcx.sess.target.os == "macos" && (flags & map_fixed) != 0 {
            return Ok(Scalar::from_maybe_pointer(Pointer::from_addr_invalid(addr), this));
        }

        let prot_read = this.eval_libc_i32("PROT_READ");
        let prot_write = this.eval_libc_i32("PROT_WRITE");

        // First, we do some basic argument validation as required by mmap
        if (flags & (map_private | map_shared)).count_ones() != 1 {
            this.set_last_error(Scalar::from_i32(this.eval_libc_i32("EINVAL")))?;
            return Ok(Scalar::from_maybe_pointer(Pointer::null(), this));
        }
        if length == 0 {
            this.set_last_error(Scalar::from_i32(this.eval_libc_i32("EINVAL")))?;
            return Ok(Scalar::from_maybe_pointer(Pointer::null(), this));
        }

        // If a user tries to map a file, we want to loudly inform them that this is not going
        // to work. It is possible that POSIX gives us enough leeway to return an error, but the
        // outcome for the user (I need to add cfg(miri)) is the same, just more frustrating.
        if fd != -1 {
            throw_unsup_format!("Miri does not support file-backed memory mappings");
        }

        // POSIX says:
        // [ENOTSUP]
        // * MAP_FIXED or MAP_PRIVATE was specified in the flags argument and the implementation
        // does not support this functionality.
        // * The implementation does not support the combination of accesses requested in the
        // prot argument.
        //
        // Miri doesn't support MAP_FIXED or any any protections other than PROT_READ|PROT_WRITE.
        if flags & map_fixed != 0 || prot != prot_read | prot_write {
            this.set_last_error(Scalar::from_i32(this.eval_libc_i32("ENOTSUP")))?;
            return Ok(Scalar::from_maybe_pointer(Pointer::null(), this));
        }

        // Miri does not support shared mappings, or any of the other extensions that for example
        // Linux has added to the flags arguments.
        if flags != map_private | map_anonymous {
            throw_unsup_format!(
                "Miri only supports calls to mmap which set the flags argument to MAP_PRIVATE|MAP_ANONYMOUS"
            );
        }

        // This is only used for file mappings, which we don't support anyway.
        if offset != 0 {
            throw_unsup_format!("Miri does not support non-zero offsets to mmap");
        }

        let align = this.machine.page_align();
        let map_length = this.machine.round_up_to_multiple_of_page_size(length).unwrap_or(u64::MAX);

        let ptr =
            this.allocate_ptr(Size::from_bytes(map_length), align, MiriMemoryKind::Mmap.into())?;
        // We just allocated this, the access is definitely in-bounds and fits into our address space.
        // mmap guarantees new mappings are zero-init.
        this.write_bytes_ptr(
            ptr.into(),
            std::iter::repeat(0u8).take(usize::try_from(map_length).unwrap()),
        )
        .unwrap();
        // Memory mappings don't use provenance, and are always exposed.
        Machine::expose_ptr(this, ptr)?;

        Ok(Scalar::from_pointer(ptr, this))
    }

    fn munmap(
        &mut self,
        addr: &OpTy<'tcx, Provenance>,
        length: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let addr = this.read_target_usize(addr)?;
        let length = this.read_target_usize(length)?;

        // addr must be a multiple of the page size
        #[allow(clippy::arithmetic_side_effects)] // PAGE_SIZE is nonzero
        if addr % this.machine.page_size != 0 {
            this.set_last_error(Scalar::from_i32(this.eval_libc_i32("EINVAL")))?;
            return Ok(Scalar::from_i32(-1));
        }

        let length = this.machine.round_up_to_multiple_of_page_size(length).unwrap_or(u64::MAX);

        let ptr = Machine::ptr_from_addr_cast(this, addr)?;

        let Ok(ptr) = ptr.into_pointer_or_addr() else {
            throw_unsup_format!("Miri only supports munmap on memory allocated directly by mmap");
        };
        let Some((alloc_id, offset, _prov)) = Machine::ptr_get_alloc(this, ptr) else {
            throw_unsup_format!("Miri only supports munmap on memory allocated directly by mmap");
        };

        // Elsewhere in this function we are careful to check what we can and throw an unsupported
        // error instead of Undefined Behavior when use of this function falls outside of the
        // narrow scope we support. We deliberately do not check the MemoryKind of this allocation,
        // because we want to report UB on attempting to unmap memory that Rust "understands", such
        // the stack, heap, or statics.
        let (_kind, alloc) = this.memory.alloc_map().get(alloc_id).unwrap();
        if offset != Size::ZERO || alloc.len() as u64 != length {
            throw_unsup_format!(
                "Miri only supports munmap calls that exactly unmap a region previously returned by mmap"
            );
        }

        let len = Size::from_bytes(alloc.len() as u64);
        this.deallocate_ptr(
            ptr.into(),
            Some((len, this.machine.page_align())),
            MemoryKind::Machine(MiriMemoryKind::Mmap),
        )?;

        Ok(Scalar::from_i32(0))
    }
}
