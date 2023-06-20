//! This follows the pattern in src/shims/unix/mem.rs: We only support uses of mremap that would
//! correspond to valid uses of realloc.

use crate::*;
use rustc_target::abi::Size;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn mremap(
        &mut self,
        old_address: &OpTy<'tcx, Provenance>,
        old_size: &OpTy<'tcx, Provenance>,
        new_size: &OpTy<'tcx, Provenance>,
        flags: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let old_address = this.read_target_usize(old_address)?;
        let old_size = this.read_target_usize(old_size)?;
        let new_size = this.read_target_usize(new_size)?;
        let flags = this.read_scalar(flags)?.to_i32()?;

        // old_address must be a multiple of the page size
        #[allow(clippy::arithmetic_side_effects)] // PAGE_SIZE is nonzero
        if old_address % this.machine.page_size != 0 || new_size == 0 {
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

        let old_address = Machine::ptr_from_addr_cast(this, old_address)?;
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
}
