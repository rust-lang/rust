//! This is an incomplete implementation of mmap/mremap/munmap which is restricted in order to be
//! implementable on top of the existing memory system. The point of these function as-written is
//! to allow memory allocators written entirely in Rust to be executed by Miri. This implementation
//! does not support other uses of mmap such as file mappings.
//!
//! mmap/mremap/munmap behave a lot like alloc/realloc/dealloc, and for simple use they are exactly
//! equivalent. That is the only part we support: no MAP_FIXED or MAP_SHARED or anything
//! else that goes beyond a basic allocation API.

use crate::*;
use rustc_target::abi::{Align, Size};

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

        // We do not support MAP_FIXED, so the addr argument is always ignored
        let addr = this.read_pointer(addr)?;
        let length = this.read_target_usize(length)?;
        let prot = this.read_scalar(prot)?.to_i32()?;
        let flags = this.read_scalar(flags)?.to_i32()?;
        let fd = this.read_scalar(fd)?.to_i32()?;
        let offset = this.read_scalar(offset)?.to_target_usize(this)?;

        let map_private = this.eval_libc_i32("MAP_PRIVATE");
        let map_anonymous = this.eval_libc_i32("MAP_ANONYMOUS");
        let map_shared = this.eval_libc_i32("MAP_SHARED");
        let map_fixed = this.eval_libc_i32("MAP_FIXED");

        // This is a horrible hack, but on macos  the guard page mechanism uses mmap
        // in a way we do not support. We just give it the return value it expects.
        if this.frame_in_std() && this.tcx.sess.target.os == "macos" && (flags & map_fixed) != 0 {
            return Ok(Scalar::from_maybe_pointer(addr, this));
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

        let align = Align::from_bytes(this.machine.page_size).unwrap();
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

    fn mremap(
        &mut self,
        old_address: &OpTy<'tcx, Provenance>,
        old_size: &OpTy<'tcx, Provenance>,
        new_size: &OpTy<'tcx, Provenance>,
        flags: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let old_address = this.read_pointer(old_address)?;
        let old_size = this.read_scalar(old_size)?.to_target_usize(this)?;
        let new_size = this.read_scalar(new_size)?.to_target_usize(this)?;
        let flags = this.read_scalar(flags)?.to_i32()?;

        // old_address must be a multiple of the page size
        #[allow(clippy::arithmetic_side_effects)] // PAGE_SIZE is nonzero
        if old_address.addr().bytes() % this.machine.page_size != 0 || new_size == 0 {
            this.set_last_error(Scalar::from_i32(this.eval_libc_i32("EINVAL")))?;
            return Ok(this.eval_libc("MAP_FAILED"));
        }

        if flags & this.eval_libc_i32("MREMAP_FIXED") != 0 {
            throw_unsup_format!("Miri does not support mremap wth MREMAP_FIXED");
        }

        if flags & this.eval_libc_i32("MREMAP_DONTUNMAP") != 0 {
            throw_unsup_format!("Miri does not support mremap wth MREMAP_DONTUNMAP");
        }

        if flags & this.eval_libc_i32("MREMAP_MAYMOVE") == 0 {
            // We only support MREMAP_MAYMOVE, so not passing the flag is just a failure
            this.set_last_error(Scalar::from_i32(this.eval_libc_i32("EINVAL")))?;
            return Ok(Scalar::from_maybe_pointer(Pointer::null(), this));
        }

        let align = this.machine.page_align();
        let ptr = this.reallocate_ptr(
            old_address,
            Some((Size::from_bytes(old_size), align)),
            Size::from_bytes(new_size),
            align,
            MiriMemoryKind::Mmap.into(),
        )?;
        if let Some(increase) = new_size.checked_sub(old_size) {
            // We just allocated this, the access is definitely in-bounds and fits into our address space.
            // mmap guarantees new mappings are zero-init.
            this.write_bytes_ptr(
                ptr.offset(Size::from_bytes(old_size), this).unwrap().into(),
                std::iter::repeat(0u8).take(usize::try_from(increase).unwrap()),
            )
            .unwrap();
        }
        // Memory mappings are always exposed
        Machine::expose_ptr(this, ptr)?;

        Ok(Scalar::from_pointer(ptr, this))
    }

    fn munmap(
        &mut self,
        addr: &OpTy<'tcx, Provenance>,
        length: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let addr = this.read_pointer(addr)?;
        let length = this.read_scalar(length)?.to_target_usize(this)?;

        // addr must be a multiple of the page size
        #[allow(clippy::arithmetic_side_effects)] // PAGE_SIZE is nonzero
        if addr.addr().bytes() % this.machine.page_size != 0 {
            this.set_last_error(Scalar::from_i32(this.eval_libc_i32("EINVAL")))?;
            return Ok(Scalar::from_i32(-1));
        }

        let length = this.machine.round_up_to_multiple_of_page_size(length).unwrap_or(u64::MAX);

        let mut addr = addr.addr().bytes();
        let mut bytes_unmapped = 0;
        while bytes_unmapped < length {
            // munmap specifies:
            // It is not an error if the indicated range does not contain any mapped pages.
            // So we make sure that if our address is not that of an exposed allocation, we just
            // step forward to the next page.
            let ptr = Machine::ptr_from_addr_cast(this, addr)?;
            let Ok(ptr) = ptr.into_pointer_or_addr() else {
                bytes_unmapped = bytes_unmapped.checked_add(this.machine.page_size).unwrap();
                addr = addr.wrapping_add(this.machine.page_size);
                continue;
            };
            // FIXME: This should fail if the pointer is to an unexposed allocation. But it
            // doesn't.
            let Some((alloc_id, offset, _prov)) = Machine::ptr_get_alloc(this, ptr) else {
                bytes_unmapped = bytes_unmapped.checked_add(this.machine.page_size).unwrap();
                addr = addr.wrapping_add(this.machine.page_size);
                continue;
            };

            if offset != Size::ZERO {
                throw_unsup_format!("Miri does not support partial munmap");
            }
            let (_kind, alloc) = this.memory.alloc_map().get(alloc_id).unwrap();
            let this_alloc_len = alloc.len() as u64;
            bytes_unmapped = bytes_unmapped.checked_add(this_alloc_len).unwrap();
            if bytes_unmapped > length {
                throw_unsup_format!("Miri does not support partial munmap");
            }

            this.deallocate_ptr(
                Pointer::new(Some(Provenance::Wildcard), Size::from_bytes(addr)),
                Some((Size::from_bytes(this_alloc_len), this.machine.page_align())),
                MemoryKind::Machine(MiriMemoryKind::Mmap),
            )?;
            addr = addr.wrapping_add(this_alloc_len);
        }

        Ok(Scalar::from_i32(0))
    }
}

trait RangeExt {
    fn overlaps(&self, other: &Self) -> bool;
}
impl RangeExt for std::ops::Range<Size> {
    fn overlaps(&self, other: &Self) -> bool {
        self.start.max(other.start) <= self.end.min(other.end)
    }
}
