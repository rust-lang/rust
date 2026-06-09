use core::arch::asm;

// Do not remove inline: will result in relocation failure
#[inline(always)]
pub(crate) unsafe fn rel_ptr<T>(offset: u64) -> *const T {
    (image_base() + offset) as *const T
}

// Do not remove inline: will result in relocation failure
#[inline(always)]
pub(crate) unsafe fn rel_ptr_mut<T>(offset: u64) -> *mut T {
    (image_base() + offset) as *mut T
}

unsafe extern "C" {
    static ENCLAVE_SIZE: usize;
    static HEAP_BASE: u64;
    static HEAP_SIZE: usize;
}

/// Returns the base memory address of the heap
pub(crate) fn heap_base() -> *const u8 {
    unsafe { rel_ptr_mut(HEAP_BASE) }
}

/// Returns the size of the heap
pub(crate) fn heap_size() -> usize {
    unsafe { HEAP_SIZE }
}

// Do not remove inline: will result in relocation failure
// For the same reason we use inline ASM here instead of an extern static to
// locate the base
/// Returns address at which current enclave is loaded.
#[inline(always)]
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn image_base() -> u64 {
    let base: u64;
    unsafe {
        asm!(
            "lea IMAGE_BASE(%rip), {}",
            lateout(reg) base,
            options(att_syntax, nostack, preserves_flags, nomem, pure),
        )
    };
    base
}

/// Returns `true` if the specified memory range is in the enclave.
///
/// For safety, this function also checks whether the range given overflows,
/// returning `false` if so.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn is_enclave_range(p: *const u8, len: usize) -> bool {
    let start = p as usize;

    // Subtract one from `len` when calculating `end` in case `p + len` is
    // exactly at the end of addressable memory (`p + len` would overflow, but
    // the range is still valid).
    let end = if len == 0 {
        start
    } else if let Some(end) = start.checked_add(len - 1) {
        end
    } else {
        return false;
    };

    let base = image_base() as usize;
    start >= base && end <= base + (unsafe { ENCLAVE_SIZE } - 1) // unsafe ok: link-time constant
}

/// Returns `true` if the specified memory range is in userspace.
///
/// For safety, this function also checks whether the range given overflows,
/// returning `false` if so.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn is_user_range(p: *const u8, len: usize) -> bool {
    let start = p as usize;

    // Subtract one from `len` when calculating `end` in case `p + len` is
    // exactly at the end of addressable memory (`p + len` would overflow, but
    // the range is still valid).
    let end = if len == 0 {
        start
    } else if let Some(end) = start.checked_add(len - 1) {
        end
    } else {
        return false;
    };

    let base = image_base() as usize;
    end < base || start > base + (unsafe { ENCLAVE_SIZE } - 1) // unsafe ok: link-time constant
}
