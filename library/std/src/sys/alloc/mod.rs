#![forbid(unsafe_op_in_unsafe_fn)]

use crate::alloc::Layout;
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
    target_arch = "loongarch32",
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
unsafe fn realloc_fallback(ptr: *mut u8, old_layout: Layout, new_size: usize) -> *mut u8 {
    // SAFETY: Docs for GlobalAlloc::realloc require this to be valid
    unsafe {
        let new_layout = Layout::from_size_align_unchecked(new_size, old_layout.align());

        let new_ptr = alloc(new_layout);
        if !new_ptr.is_null() {
            let size = usize::min(old_layout.size(), new_size);
            ptr::copy_nonoverlapping(ptr, new_ptr, size);
            dealloc(ptr, old_layout);
        }

        new_ptr
    }
}

cfg_select! {
    any(
        target_family = "unix",
        target_os = "wasi",
        target_os = "teeos",
        target_os = "trusty",
    ) => {
        mod unix;
        use unix as imp;
    }
    target_os = "windows" => {
        mod windows;
        use windows as imp;
    }
    target_os = "hermit" => {
        mod hermit;
        use hermit as imp;
    }
    target_os = "motor" => {
        mod motor;
        use motor as imp;
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        use sgx as imp;
    }
    target_os = "solid_asp3" => {
        mod solid;
        use solid as imp;
    }
    target_os = "uefi" => {
        mod uefi;
        use uefi as imp;
    }
    target_os = "vexos" => {
        mod vexos;
        use vexos as imp;
    }
    target_family = "wasm" => {
        mod wasm;
        use wasm as imp;
    }
    target_os = "xous" => {
        mod xous;
        use xous as imp;
    }
    target_os = "zkvm" => {
        mod zkvm;
        use zkvm as imp;
    }
}

pub use imp::{alloc, dealloc, realloc};

cfg_select! {
    any(
        target_os = "hermit",
        target_os = "solid_asp3",
        target_os = "uefi",
        target_os = "zkvm",
    ) => {
        #[inline]
        pub unsafe fn alloc_zeroed(layout: Layout) -> *mut u8 {
            let ptr = unsafe { alloc(layout) };
            if !ptr.is_null() {
                unsafe { ptr.write_bytes(0, layout.size()) };
            }
            ptr
        }
    }
    _ => {
        pub use imp::alloc_zeroed;
    }
}
