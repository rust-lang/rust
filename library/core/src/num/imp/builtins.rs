//! Bindings to functions provided by `compiler-builtins`.
//!
//! These are always available as `compiler-builtins` provides implementations, although other
//! implementations from libraries like libgcc/libgcc_s and compiler-rt may end up getting chosen by
//! the linker.

// SAFETY: These symbols are defined by `compiler-builtins`, which is linked into every Rust
// program.
#[allow(dead_code)] // This list reflects what is available rather than what is consumed.
unsafe extern "C" {
    pub(crate) safe fn __powisf2(a: f32, b: i32) -> f32;
    pub(crate) safe fn __powidf2(a: f64, b: i32) -> f64;
    // PowerPC uses `kf` rather than `tf` for `f128`.
    #[cfg_attr(any(target_arch = "powerpc", target_arch = "powerpc64"), link_name = "__powikf2")]
    pub(crate) safe fn __powitf2(a: f128, b: i32) -> f128;
}
