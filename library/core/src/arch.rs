#![doc = include_str!("../../stdarch/crates/core_arch/src/core_arch_docs.md")]

#[allow(unused_imports)]
#[stable(feature = "simd_arch", since = "1.27.0")]
pub use crate::core_arch::arch::*;

#[cfg(bootstrap)]
#[allow(dead_code)]
#[unstable(feature = "sha512_sm_x86", issue = "126624")]
fn dummy() {
    // AArch64 also has a target feature named `sm4`, so we need `#![feature(sha512_sm_x86)]` in lib.rs
    // But as the bootstrap compiler doesn't know about this feature yet, we need to convert it to a
    // library feature until bootstrap gets bumped
}

/// Inline assembly.
///
/// Refer to [Rust By Example] for a usage guide and the [reference] for
/// detailed information about the syntax and available options.
///
/// [Rust By Example]: https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html
/// [reference]: https://doc.rust-lang.org/nightly/reference/inline-assembly.html
#[stable(feature = "asm", since = "1.59.0")]
#[rustc_builtin_macro]
pub macro asm("assembly template", $(operands,)* $(options($(option),*))?) {
    /* compiler built-in */
}

/// Module-level inline assembly.
///
/// Refer to [Rust By Example] for a usage guide and the [reference] for
/// detailed information about the syntax and available options.
///
/// [Rust By Example]: https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html
/// [reference]: https://doc.rust-lang.org/nightly/reference/inline-assembly.html
#[stable(feature = "global_asm", since = "1.59.0")]
#[rustc_builtin_macro]
pub macro global_asm("assembly template", $(operands,)* $(options($(option),*))?) {
    /* compiler built-in */
}
