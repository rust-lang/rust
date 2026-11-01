//! Platform-specific extensions to `std` for Preview 2 of the WebAssembly System Interface (WASI).
//!
//! This module is currently empty, but will be filled over time as wasi-libc support for WASI Preview 2 is stabilized.

#![forbid(unsafe_op_in_unsafe_fn)]
#![unstable(feature = "wasip2", issue = "none")]
#![doc(cfg(all(target_os = "wasi", target_env = "p2")))]
