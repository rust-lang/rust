//! This is an incomplete implementation of mmap/munmap which is restricted in order to be
//! implementable on top of the existing memory system. The point of these function as-written is
//! to allow memory allocators written entirely in Rust to be executed by Miri. This implementation
//! does not support other uses of mmap such as file mappings.
//!
//! mmap/munmap behave a lot like alloc/dealloc, and for simple use they are exactly
//! equivalent. That is the only part we support: no MAP_FIXED or MAP_SHARED or anything
//! else that goes beyond a basic allocation API.
//!
//! Note that in addition to only supporting malloc-like calls to mmap, we only support free-like
//! calls to munmap, but for a very different reason. In principle, according to the man pages, it
//! is possible to unmap arbitrary regions of address space. But in a high-level language like Rust
//! this amounts to partial deallocation, which LLVM does not support. So any attempt to call our
//! munmap shim which would partially unmap a region of address space previously mapped by mmap will
//! report UB.

use rustc_abi::Size;
use rustc_target::spec::Os;

use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn mmap(
        &mut self,
        addr: &OpTy<'tcx>,
        length: &OpTy<'tcx>,
        prot: &OpTy<'tcx>,
        flags: &OpTy<'tcx>,
        fd: &OpTy<'tcx>,
        offset: i128,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        // We do not support MAP_FIXED, so the addr argument is always ignored (except for the MacOS hack)
        let addr = this.read_target_usize(addr)?;
        let length = this.read_target_usize(length)?;
        let prot = this.read_scalar(prot)?.to_i32()?;
        let flags = this.read_scalar(flags)?.to_i32()?;
        let fd = this.read_scalar(fd)?.to_i32()?;

        let map_private = this.eval_libc_i32("MAP_PRIVATE");
        let map_anonymous = this.eval_libc_i32("MAP_ANONYMOUS");
        let map_shared = this.eval_libc_i32("MAP_SHARED");
        let map_fixed = this.eval_libc_i32("MAP_FIXED");

        // This is a horrible hack, but on MacOS and Solarish the guard page mechanism uses mmap
        // in a way we do not support. We just give it the return value it expects.
        if this.frame_in_std()
            && matches!(&this.tcx.sess.target.os, Os::MacOs | Os::Solaris | Os::Illumos)
            && (flags & map_fixed) != 0
        {
            return interp_ok(Scalar::from_maybe_pointer(Pointer::without_provenance(addr), this));
        }

        // First, we do some basic argument validation as required by mmap
        if (flags & (map_private | map_shared)).count_ones() != 1 {
            this.set_last_error(LibcError("EINVAL"))?;
            return interp_ok(this.eval_libc("MAP_FAILED"));
        }
        if length == 0 {
            this.set_last_error(LibcError("EINVAL"))?;
            return interp_ok(this.eval_libc("MAP_FAILED"));
        }

        // If a user tries to map a file, we want to loudly inform them that this is not going
        // to work. It is possible that POSIX gives us enough leeway to return an error, but the
        // outcome for the user (I need to add cfg(miri)) is the same, just more frustrating.
        if fd != -1 {
            throw_unsup_format!("Miri does not support file-backed memory mappings");
        }

        // Miri doesn't support MAP_FIXED.
        if flags & map_fixed != 0 {
            throw_unsup_format!(
                "Miri does not support calls to mmap with MAP_FIXED as part of the flags argument",
            );
        }

        verify_prot(this, prot)?;

        // Miri does not support shared mappings, or any of the other extensions that for example
        // Linux has added to the flags arguments.
        if flags != map_private | map_anonymous {
            throw_unsup_format!(
                "Miri only supports calls to mmap which set the flags argument to \
                 MAP_PRIVATE|MAP_ANONYMOUS",
            );
        }

        // This is only used for file mappings, which we don't support anyway.
        if offset != 0 {
            throw_unsup_format!("Miri does not support non-zero offsets to mmap");
        }

        let align = this.machine.page_align();
        let Some(map_length) = round_up_to_page_size(this, length) else {
            this.set_last_error(LibcError("EINVAL"))?;
            return interp_ok(this.eval_libc("MAP_FAILED"));
        };

        let ptr = this.allocate_ptr(
            Size::from_bytes(map_length),
            align,
            MiriMemoryKind::Mmap.into(),
            // mmap guarantees new mappings are zero-init.
            AllocInit::Zero,
        )?;

        interp_ok(Scalar::from_pointer(ptr, this))
    }

    fn munmap(&mut self, addr: &OpTy<'tcx>, length: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let addr = this.read_pointer(addr)?;
        let length = this.read_target_usize(length)?;

        // addr must be a multiple of the page size, but apart from that munmap is just implemented
        // as a dealloc.
        if !addr.addr().bytes().is_multiple_of(this.machine.page_size) {
            return this.set_errno_and_return_neg1_i32(LibcError("EINVAL"));
        }

        let Some(length) = round_up_to_page_size(this, length) else {
            return this.set_errno_and_return_neg1_i32(LibcError("EINVAL"));
        };

        let length = Size::from_bytes(length);
        this.deallocate_ptr(
            addr,
            Some((length, this.machine.page_align())),
            MemoryKind::Machine(MiriMemoryKind::Mmap),
        )?;

        interp_ok(Scalar::from_i32(0))
    }

    fn mprotect(
        &mut self,
        addr: &OpTy<'tcx>,
        length: &OpTy<'tcx>,
        prot: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let addr = this.read_pointer(addr)?;
        let length = this.read_target_usize(length)?;
        let prot = this.read_scalar(prot)?.to_i32()?;

        // addr must be a multiple of the page size.
        if !addr.addr().bytes().is_multiple_of(this.machine.page_size) {
            return this.set_errno_and_return_neg1_i32(LibcError("EINVAL"));
        }

        verify_prot(this, prot)?;

        // The pages from `[addr, addr + length)` must be mapped, so length definitely must not overflow.
        let Some(length) = round_up_to_page_size(this, length) else {
            return this.set_errno_and_return_neg1_i32(LibcError("ENOMEM"));
        };
        // Ensure this is actually allocated memory we can access.
        this.check_ptr_access(addr, Size::from_bytes(length), CheckInAllocMsg::MemoryAccess)
            .map_err_kind(|_| err_ub_format!("`mprotect` called on out-of-bounds memory"))?;

        // If the memory comes from memory the Rust program has allocated with mmap, we only support
        // `PROT_READ|PROT_WRITE`, so this `mprotect` is a no-op. If the memory was mmaped by the
        // runtime (e.g. if it's the stack, executable memory, or static memory), POSIX also allows
        // us to remap it. In those cases, such a call to `PROT_READ|PROT_WRITE` might actually change the permissions,
        // but treating them as the new permissions is still UB. Therefore, we just pretend that we
        // did the permission change by returning success, and will still reject if you try to use
        // it with the "new" permissions.
        interp_ok(Scalar::from_i32(0))
    }

    fn madvise(
        &mut self,
        addr: &OpTy<'tcx>,
        length: &OpTy<'tcx>,
        advice: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let addr = this.read_pointer(addr)?;
        let length = this.read_target_usize(length)?;
        let advise = this.read_scalar(advice)?.to_i32()?;

        // addr must be a multiple of the page size.
        if !addr.addr().bytes().is_multiple_of(this.machine.page_size) {
            return this.set_errno_and_return_neg1_i32(LibcError("EINVAL"));
        }

        // advise must be supported.
        let madv_normal = this.eval_libc_i32("MADV_NORMAL");
        let madv_random = this.eval_libc_i32("MADV_RANDOM");
        let madv_sequential = this.eval_libc_i32("MADV_SEQUENTIAL");
        let madv_willneed = this.eval_libc_i32("MADV_WILLNEED");
        if advise != madv_normal
            && advise != madv_random
            && advise != madv_sequential
            && advise != madv_willneed
        {
            throw_unsup_format!(
                "Miri does not support calls to madvise with advice other than MADV_NORMAL, MADV_RANDOM, MADV_SEQUENTIAL, or MADV_WILLNEED",
            );
        }

        // The pages from `[addr, addr + length)` must be mapped, so length definitely must not overflow.
        let Some(length) = round_up_to_page_size(this, length) else {
            return this.set_errno_and_return_neg1_i32(LibcError("ENOMEM"));
        };
        // Ensure this is actually allocated memory we can access.
        this.check_ptr_access(addr, Size::from_bytes(length), CheckInAllocMsg::MemoryAccess)
            .map_err_kind(|_| err_ub_format!("`madvise` called on out-of-bounds memory"))?;

        // All advises we support are no-ops.
        interp_ok(Scalar::from_i32(0))
    }
}

fn round_up_to_page_size(this: &MiriInterpCx<'_>, length: u64) -> Option<u64> {
    length
        .checked_next_multiple_of(this.machine.page_size)
        .filter(|length| *length <= this.target_isize_max().try_into().unwrap())
}

fn verify_prot<'tcx>(this: &mut MiriInterpCx<'tcx>, prot: i32) -> InterpResult<'tcx> {
    let prot_read = this.eval_libc_i32("PROT_READ");
    let prot_write = this.eval_libc_i32("PROT_WRITE");

    // Miri doesn't support protections other than PROT_READ|PROT_WRITE.
    if prot != prot_read | prot_write {
        throw_unsup_format!(
            "Miri does not support calls to mmap/mprotect with protections other than \
             PROT_READ|PROT_WRITE",
        );
    }

    interp_ok(())
}
