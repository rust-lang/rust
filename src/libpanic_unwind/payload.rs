// Type definition for the payload argument of the try intrinsic.
//
// This must be kept in sync with the implementations of the try intrinsic.
//
// This file is included by both panic runtimes and libstd. It is part of the
// panic runtime ABI.
cfg_if::cfg_if! {
    if #[cfg(target_os = "emscripten")] {
        type TryPayload = *mut u8;
    } else if #[cfg(target_arch = "wasm32")] {
        type TryPayload = *mut u8;
    } else if #[cfg(target_os = "hermit")] {
        type TryPayload = *mut u8;
    } else if #[cfg(all(target_env = "msvc", target_arch = "aarch64"))] {
        type TryPayload = *mut u8;
    } else if #[cfg(target_env = "msvc")] {
        type TryPayload = [u64; 2];
    } else {
        type TryPayload = *mut u8;
    }
}
