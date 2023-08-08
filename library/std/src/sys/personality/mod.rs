//! This module contains the implementation of the `eh_personality` lang item.
//!
//! The actual implementation is heavily dependent on the target since Rust
//! tries to use the native stack unwinding mechanism whenever possible.
//!
//! This personality function is still required with `-C panic=abort` because
//! it is used to catch foreign exceptions from `extern "C-unwind"` and turn
//! them into aborts.
//!
//! Additionally, ARM EHABI uses the personality function when generating
//! backtraces.

mod dwarf;

#[cfg(not(test))]
cfg_if::cfg_if! {
    if #[cfg(target_os = "emscripten")] {
        mod emcc;
    } else if #[cfg(target_env = "msvc")] {
        // This is required by the compiler to exist (e.g., it's a lang item),
        // but it's never actually called by the compiler because
        // _CxxFrameHandler3 is the personality function that is always used.
        // Hence this is just an aborting stub.
        #[lang = "eh_personality"]
        fn rust_eh_personality() {
            core::intrinsics::abort()
        }
    } else if #[cfg(any(
        all(target_family = "windows", target_env = "gnu"),
        target_os = "psp",
        target_os = "solid_asp3",
        all(target_family = "unix", not(target_os = "espidf"), not(target_os = "l4re")),
        all(target_vendor = "fortanix", target_env = "sgx"),
    ))] {
        mod gcc;
    } else {
        // Targets that don't support unwinding.
        // - family=wasm
        // - os=none ("bare metal" targets)
        // - os=uefi
        // - os=espidf
        // - os=hermit
        // - nvptx64-nvidia-cuda
        // - arch=avr
    }
}
