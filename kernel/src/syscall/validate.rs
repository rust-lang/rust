use abi::errors::{Errno, SysResult};

/// Validates that a user range is within the user address space.
///
/// # Arguments
/// * `base` - The base address of the range.
/// * `len` - The length of the range.
/// * `writable` - Whether the range must be writable.
pub fn validate_user_range(base: usize, len: usize, writable: bool) -> SysResult<()> {
    // Basic checks
    if base == 0 {
        return Err(Errno::EFAULT);
    }
    if base.checked_add(len).is_none() {
        return Err(Errno::EFAULT);
    }
    // Simplistic kernel boundary check (assuming higher half kernel)
    if base >= 0xffffffff80000000 {
        return Err(Errno::EFAULT);
    }

    // Check against actual mappings if the hook is available
    if let Some(valid) = unsafe { crate::sched::check_user_mapping_current(base, len, writable) } {
        if !valid {
            // crate::kinfo!("validate_user_range: check failed base={:#x} len={} w={}", base, len, writable);
            return Err(Errno::EFAULT);
        }
    }

    Ok(())
}

/// Copies data from user memory to kernel memory.
///
/// # Safety
/// Caller must ensure `src_user` is a valid user pointer (validated via `validate_user_range`).
/// Usage of this function implies we are effectively trusting the user range is mapped.
/// In a real implementation this would use `copy_from_user` assembly or similar to handle page faults safely.
pub unsafe fn copyin(dst_kernel: &mut [u8], src_user: usize) -> SysResult<()> {
    if let Err(e) = validate_user_range(src_user, dst_kernel.len(), false) {
        crate::kinfo!(
            "copyin: EFAULT src={:#x} len={}",
            src_user,
            dst_kernel.len()
        );
        return Err(e);
    }
    let src = src_user as *const u8;
    unsafe {
        // This is still dangerous if not mapped, will cause PF in kernel mode.
        // We rely on page fault handler to kill the task if this faults.
        // Or we use specific copy primitives that catch faults.
        // For v0.5, we assume if validate passes, we try access.
        core::ptr::copy_nonoverlapping(src, dst_kernel.as_mut_ptr(), dst_kernel.len());
    }
    Ok(())
}

/// Copies data from kernel memory to user memory.
pub unsafe fn copyout(dst_user: usize, src_kernel: &[u8]) -> SysResult<()> {
    if let Err(e) = validate_user_range(dst_user, src_kernel.len(), true) {
        crate::kinfo!(
            "copyout: EFAULT dst={:#x} len={}",
            dst_user,
            src_kernel.len()
        );
        return Err(e);
    }
    let dst = dst_user as *mut u8;
    unsafe {
        core::ptr::copy_nonoverlapping(src_kernel.as_ptr(), dst, src_kernel.len());
    }
    Ok(())
}
