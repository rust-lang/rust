#![doc = include_str!("../../stdarch/crates/core_arch/src/core_arch_docs.md")]

#[allow(
    // some targets don't have anything to reexport, which
    // makes the `pub use` unused and unreachable, allow
    // both lints as to not have `#[cfg]`s
    //
    // cf. https://github.com/rust-lang/rust/pull/116033#issuecomment-1760085575
    unused_imports,
    unreachable_pub
)]
#[stable(feature = "simd_arch", since = "1.27.0")]
pub use crate::core_arch::arch::*;

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
#[stable(feature = "naked_functions", since = "1.88.0")]
#[rustc_builtin_macro]
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

/// Compiles to a target-specific software breakpoint instruction or equivalent.
///
/// This will typically abort the program. It may result in a core dump, and/or the system logging
/// debug information. Additional target-specific capabilities may be possible depending on
/// debuggers or other tooling; in particular, a debugger may be able to resume execution.
///
/// If possible, this will produce an instruction sequence that allows a debugger to resume *after*
/// the breakpoint, rather than resuming *at* the breakpoint; however, the exact behavior is
/// target-specific and debugger-specific, and not guaranteed.
///
/// If the target platform does not have any kind of debug breakpoint instruction, this may compile
/// to a trapping instruction (e.g. an undefined instruction) instead, or to some other form of
/// target-specific abort that may or may not support convenient resumption.
///
/// The precise behavior and the precise instruction generated are not guaranteed, except that in
/// normal execution with no debug tooling involved this will not continue executing.
///
/// - On x86 targets, this produces an `int3` instruction.
/// - On aarch64 targets, this produces a `brk #0xf000` instruction.
// When stabilizing this, update the comment on `core::intrinsics::breakpoint`.
#[unstable(feature = "breakpoint", issue = "133724")]
#[inline(always)]
pub fn breakpoint() {
    core::intrinsics::breakpoint();
}
