#![doc = include_str!("../../stdarch/crates/core_arch/src/core_arch_docs.md")]

#[allow(unused_imports)]
#[stable(feature = "simd_arch", since = "1.27.0")]
pub use crate::core_arch::arch::*;

/// Inline assembly.
///
/// Refer to [rust by example] for a usage guide and the [reference] for
/// detailed information about the syntax and available options.
///
/// [rust by example]: https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html
/// [reference]: https://doc.rust-lang.org/nightly/reference/inline-assembly.html
#[stable(feature = "asm", since = "1.59.0")]
#[rustc_builtin_macro]
pub macro asm("assembly template", $(operands,)* $(options($(option),*))?) {
    /* compiler built-in */
}

/// Module-level inline assembly.
///
/// Refer to [rust by example] for a usage guide and the [reference] for
/// detailed information about the syntax and available options.
///
/// [rust by example]: https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html
/// [reference]: https://doc.rust-lang.org/nightly/reference/inline-assembly.html
#[stable(feature = "global_asm", since = "1.59.0")]
#[rustc_builtin_macro]
pub macro global_asm("assembly template", $(operands,)* $(options($(option),*))?) {
    /* compiler built-in */
}
