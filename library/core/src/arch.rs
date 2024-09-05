#![doc = include_str!("../../stdarch/crates/core_arch/src/core_arch_docs.md")]

#[allow(unused_imports)]
#[stable(feature = "simd_arch", since = "1.27.0")]
pub use crate::core_arch::arch::*;
#[unstable(feature = "naked_functions", issue = "90957")]
#[cfg(bootstrap)]
pub use crate::naked_asm;

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

/// Inline assembly used in combination with `#[naked]` functions.
///
/// Refer to [Rust By Example] for a usage guide and the [reference] for
/// detailed information about the syntax and available options.
///
/// [Rust By Example]: https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html
/// [reference]: https://doc.rust-lang.org/nightly/reference/inline-assembly.html
#[unstable(feature = "naked_functions", issue = "90957")]
#[macro_export]
#[cfg(bootstrap)]
macro_rules! naked_asm {
    ([$last:expr], [$($pushed:expr),*]) => {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            core::arch::asm!($($pushed),*, options(att_syntax, noreturn))
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
        {
            core::arch::asm!($($pushed),* , $last, options(noreturn))
        }
    };

    ([$first:expr $(, $rest:expr)*], [$($pushed:expr),*]) => {
        naked_asm!([$($rest),*], [$($pushed,)* $first]);
    };

    ($($expr:expr),* $(,)?) => {
        naked_asm!([$($expr),*], []);
    };
}

/// Inline assembly used in combination with `#[naked]` functions.
///
/// Refer to [Rust By Example] for a usage guide and the [reference] for
/// detailed information about the syntax and available options.
///
/// [Rust By Example]: https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html
/// [reference]: https://doc.rust-lang.org/nightly/reference/inline-assembly.html
#[unstable(feature = "naked_functions", issue = "90957")]
#[rustc_builtin_macro]
#[cfg(not(bootstrap))]
pub macro naked_asm("assembly template", $(operands,)* $(options($(option),*))?) {
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
