//! Before #78122, writing any `fmt::Arguments` would trigger the inclusion of `usize` formatting
//! and padding code in the resulting binary, because indexing used in `fmt::write` would generate
//! code using `panic_bounds_check`, which prints the index and length.
//!
//! These bounds checks are not necessary, as `fmt::Arguments` never contains any out-of-bounds
//! indexes. The test is a `run-make` test, because it needs to check the result after linking. A
//! codegen or assembly test doesn't check the parts that will be pulled in from `core` by the
//! linker.
//!
//! In this test, we try to check that the `usize` formatting and padding code are not present in
//! the final binary by checking that panic symbols such as `panic_bounds_check` are **not**
//! present.
//!
//! Some CI jobs try to run faster by disabling debug assertions (through setting
//! `NO_DEBUG_ASSERTIONS=1`). If debug assertions are disabled, then we can check for the absence of
//! additional `usize` formatting and padding related symbols.

//@ ignore-windows
// Reason:
// - MSVC targets really need to parse the .pdb file (aka the debug information).
//   On Windows there's an API for that (dbghelp) which maybe we can use
// - MinGW targets have a lot of symbols included in their runtime which we can't avoid.
//   We would need to make the symbols we're looking for more specific for this test to work.
//@ ignore-cross-compile

use run_make_support::env::no_debug_assertions;
use run_make_support::rustc;
use run_make_support::symbols::any_symbol_contains;

fn main() {
    rustc().input("main.rs").opt().run();
    // panic machinery identifiers, these should not appear in the final binary
    let mut panic_syms = vec!["panic_bounds_check", "Debug"];
    if no_debug_assertions() {
        // if debug assertions are allowed, we need to allow these,
        // otherwise, add them to the list of symbols to deny.
        panic_syms.extend_from_slice(&["panicking", "panic_fmt", "pad_integral", "Display"]);
    }
    assert!(!any_symbol_contains("main", &panic_syms));
}
