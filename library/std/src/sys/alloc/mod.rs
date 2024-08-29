#![forbid(unsafe_op_in_unsafe_fn)]

use crate::alloc::{GlobalAlloc, Layout, System};
use crate::ptr;

// The minimum alignment guaranteed by the architecture. This value is used to
// add fast paths for low alignment values.
#[allow(dead_code)]
const MIN_ALIGN: usize = if cfg!(any(
    all(target_arch = "riscv32", any(target_os = "espidf", target_os = "zkvm")),
    all(target_arch = "xtensa", target_os = "espidf"),
)) {
    // The allocator on the esp-idf and zkvm platforms guarantees 4 byte alignment.
    4
} else if cfg!(any(
    target_arch = "x86",
    target_arch = "arm",
    target_arch = "m68k",
    target_arch = "csky",
    target_arch = "mips",
    target_arch = "mips32r6",
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "sparc",
    target_arch = "wasm32",
    target_arch = "hexagon",
    target_arch = "riscv32",
    target_arch = "xtensa",
)) {
    8
} else if cfg!(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm64ec",
    target_arch = "loongarch64",
    target_arch = "mips64",
    target_arch = "mips64r6",
    target_arch = "s390x",
    target_arch = "sparc64",
    target_arch = "riscv64",
    target_arch = "wasm64",
)) {
    16
} else {
    panic!("add a value for MIN_ALIGN")
};

#[allow(dead_code)]
unsafe fn realloc_fallback(
    alloc: &System,
    ptr: *mut u8,
    old_layout: Layout,
    new_size: usize,
) -> *mut u8 {
    // SAFETY: Docs for GlobalAlloc::realloc require this to be valid
    unsafe {
        let new_layout = Layout::from_size_align_unchecked(new_size, old_layout.align());

        let new_ptr = GlobalAlloc::alloc(alloc, new_layout);
        if !new_ptr.is_null() {
            let size = usize::min(old_layout.size(), new_size);
            ptr::copy_nonoverlapping(ptr, new_ptr, size);
            GlobalAlloc::dealloc(alloc, ptr, old_layout);
        }

        new_ptr
    }
}

cfg_if::cfg_if! {
    if #[cfg(any(
        target_family = "unix",
        target_os = "wasi",
        target_os = "teeos",
    ))] {
        mod unix;
    } else if #[cfg(target_os = "windows")] {
        mod windows;
    } else if #[cfg(target_os = "hermit")] {
        mod hermit;
    } else if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        mod sgx;
    } else if #[cfg(target_os = "solid_asp3")] {
        mod solid;
    } else if #[cfg(target_os = "uefi")] {
        mod uefi;
    } else if #[cfg(target_family = "wasm")] {
        mod wasm;
    } else if #[cfg(target_os = "xous")] {
        mod xous;
    } else if #[cfg(target_os = "zkvm")] {
        mod zkvm;
    }
}
