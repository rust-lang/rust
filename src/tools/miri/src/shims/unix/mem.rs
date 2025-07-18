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
            && matches!(&*this.tcx.sess.target.os, "macos" | "solaris" | "illumos")
            && (flags & map_fixed) != 0
        {
            return interp_ok(Scalar::from_maybe_pointer(Pointer::without_provenance(addr), this));
        }

        let prot_read = this.eval_libc_i32("PROT_READ");
        let prot_write = this.eval_libc_i32("PROT_WRITE");

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

        // Miri doesn't support protections other than PROT_READ|PROT_WRITE.
        if prot != prot_read | prot_write {
            throw_unsup_format!(
                "Miri does not support calls to mmap with protections other than \
                 PROT_READ|PROT_WRITE",
            );
        }

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
        let Some(map_length) = length.checked_next_multiple_of(this.machine.page_size) else {
            this.set_last_error(LibcError("EINVAL"))?;
            return interp_ok(this.eval_libc("MAP_FAILED"));
        };
        if map_length > this.target_usize_max() {
            this.set_last_error(LibcError("EINVAL"))?;
            return interp_ok(this.eval_libc("MAP_FAILED"));
        }

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
            return this.set_last_error_and_return_i32(LibcError("EINVAL"));
        }

        let Some(length) = length.checked_next_multiple_of(this.machine.page_size) else {
            return this.set_last_error_and_return_i32(LibcError("EINVAL"));
        };
        if length > this.target_usize_max() {
            this.set_last_error(LibcError("EINVAL"))?;
            return interp_ok(this.eval_libc("MAP_FAILED"));
        }

        let length = Size::from_bytes(length);
        this.deallocate_ptr(
            addr,
            Some((length, this.machine.page_align())),
            MemoryKind::Machine(MiriMemoryKind::Mmap),
        )?;

        interp_ok(Scalar::from_i32(0))
    }
}
