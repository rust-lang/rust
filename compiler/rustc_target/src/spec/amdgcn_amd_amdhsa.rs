//!
//! Compile with: `rustc --target spirv-unknown-unknown -Z build-std=core,compiler_builtins --crate-type=cdylib`

use core::arch::asm;
use core::panic::PanicInfo;
use core::sync::atomic::{AtomicU32, Ordering};

/// Kernel launch configuration encoded in the binary for the runtime loader.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct KernelLaunchConfig {
    /// Workgroup size (X, Y, Z)
    pub workgroup_size: [u32; 3],
    /// Grid size (X, Y, Z)
    pub grid_size: [u32; 3],
    /// Dynamic shared memory per workgroup (bytes)
    pub dynamic_shared_mem_bytes: u32,
    /// Reserved for future use / alignment
    pub _reserved: u32,
}

impl Default for KernelLaunchConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            workgroup_size: [256, 1, 1],
            grid_size: [1, 1, 1],
            dynamic_shared_mem_bytes: 0,
            _reserved: 0,
        }
    }
}

/// Global kernel metadata — placed in a dedicated ELF section for tooling extraction.
#[used]
#[link_section = ".kernel.metadata"]
static KERNEL_METADATA: KernelLaunchConfig = KernelLaunchConfig::default();

/// Panic counter for diagnostics (visible in debugger / crash dumps).
static PANIC_COUNT: AtomicU32 = AtomicU32::new(0);

/// **Production-grade panic handler** — *must* diverge (`!`).
///
/// Uses inline PTX/AMDGCN `s_endpgm` / `trap` to terminate the wavefront
/// without relying on `core::intrinsics::abort()` (which is unstable/absent
/// on GPU targets). The infinite `loop {}` fallback satisfies the type checker
/// but is never reached.
#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    // Increment atomic counter for post-mortem analysis.
    PANIC_COUNT.fetch_add(1, Ordering::Relaxed);

    // Optional: emit panic message via device-side printf (if supported by runtime).
    // SAFETY: `asm!` is target-specific; guarded by `#[cfg(target_arch = "amdgcn")]`.
    #[cfg(target_arch = "amdgcn")]
    unsafe {
        // `s_endpgm` terminates the current wavefront cleanly on AMD GCN/RDNA.
        asm!("s_endpgm", options(noreturn, nostack));
    }

    #[cfg(not(target_arch = "amdgcn"))]
    unsafe {
        // Generic trap for SPIR-V / other targets.
        asm!("trap", options(noreturn, nostack));
    }

    // Unreachable — type system requires `!`, but asm above never returns.
    #[allow(clippy::empty_loop)]
    loop {
        core::hint::spin_loop();
    }
}

/// **Kernel entry point** — `extern "gpu-kernel"` ABI mandated by `abi_gpu_kernel` feature.
///
/// Arguments are passed via kernel argument buffer (vendor-specific layout).
/// This signature is a placeholder; replace with your actual kernel parameters.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn kernel(
    // Example arguments — adjust to your kernel's interface:
    // global_id: u32,
    // input: *const f32,
    // output: *mut f32,
    // count: u32,
) {
    // Kernel body goes here.
    // Example:
    // let idx = global_id as usize;
    // if idx < count as usize {
    //     *output.add(idx) = *input.add(idx) * 2.0;
    // }
}

/// Optional: explicit `#[no_mangle]` symbol for runtime loader to locate entry point.
/// Some toolchains expect `_start` or a specific name; adjust per your loader spec.
#[no_mangle]
#[link_section = ".text.kernel_entry"]
pub unsafe extern "C" fn _amdgpu_kernel_entry() -> ! {
    // Delegate to the Rust kernel entry point.
    kernel();
    // Should never return; terminate wavefront.
    #[cfg(target_arch = "amdgcn")]
    asm!("s_endpgm", options(noreturn, nostack));
    #[cfg(not(target_arch = "amdgcn"))]
    asm!("trap", options(noreturn, nostack));
    #[allow(clippy::empty_loop)]
    loop {
        core::hint::spin_loop();
    }
}

/// **Language items** — required for `no_std` + `panic = "abort"` with LTO.
///
/// These satisfy the compiler's expectations for `eh_personality` and
/// `panic_fmt` when building with `-Z build-std=core,compiler_builtins`.
#[cfg(all(target_arch = "amdgcn", not(test)))]
mod lang_items {
    use core::panic::PanicInfo;

    #[lang = "eh_personality"]
    #[no_mangle]
    pub extern "C" fn eh_personality() {}

    #[lang = "panic_fmt"]
    #[no_mangle]
    pub extern "C" fn panic_fmt(_info: &PanicInfo<'_>) -> ! {
        crate::panic(_info)
    }
}

/// **Compiler-builtins shim** — provides minimal `memcpy`/`memset`/`memcmp`
/// to avoid pulling in `compiler_builtins` crate (which may not compile for GPU).
#[cfg(all(target_arch = "amdgcn", not(test)))]
mod builtins {
    use core::ptr;

    #[no_mangle]
    pub unsafe extern "C" fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
        ptr::copy_nonoverlapping(src, dest, n);
        dest
    }

    #[no_mangle]
    pub unsafe extern "C" fn memset(s: *mut u8, c: i32, n: usize) -> *mut u8 {
        ptr::write_bytes(s, c as u8, n);
        s
    }

    #[no_mangle]
    pub unsafe extern "C" fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32 {
        for i in 0..n {
            let a = *s1.add(i);
            let b = *s2.add(i);
            if a != b {
                return (a as i32) - (b as i32);
            }
        }
        0
    }
}