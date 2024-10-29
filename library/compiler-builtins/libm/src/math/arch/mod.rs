//! Architecture-specific routines and operations.
//!
//! LLVM will already optimize calls to some of these in cases that there are hardware
//! instructions. Providing an implementation here just ensures that the faster implementation
//! is used when calling the function directly. This helps anyone who uses `libm` directly, as
//! well as improving things when these routines are called as part of other implementations.

#[cfg(intrinsics_enabled)]
pub mod intrinsics;
