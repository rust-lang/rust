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

//@ ignore-cross-compile

use run_make_support::artifact_names::bin_name;
use run_make_support::env::std_debug_assertions_enabled;
use run_make_support::rustc;
use run_make_support::symbols::object_contains_any_symbol_substring;

fn main() {
    rustc().input("main.rs").opt().run();
    // panic machinery identifiers, these should not appear in the final binary
    let mut panic_syms = vec!["panic_bounds_check", "Debug"];
    if std_debug_assertions_enabled() {
        // if debug assertions are allowed, we need to allow these,
        // otherwise, add them to the list of symbols to deny.
        panic_syms.extend_from_slice(&["panicking", "panic_fmt", "pad_integral", "Display"]);
    }
    assert!(!object_contains_any_symbol_substring(bin_name("main"), &panic_syms));
}
