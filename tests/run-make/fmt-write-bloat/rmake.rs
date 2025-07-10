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

// ignore-tidy-linelength

//@ ignore-cross-compile

use run_make_support::env::no_debug_assertions;
use run_make_support::symbols::any_symbol_contains;
use run_make_support::{bin_name, is_darwin, is_windows_msvc, pdb, rustc, target};

// Not applicable for `extern "C"` symbol decoration handling.
fn sym(sym_name: &str) -> String {
    if is_darwin() {
        // Symbols are decorated with an underscore prefix on darwin platforms.
        format!("_{sym_name}")
    } else {
        sym_name.to_string()
    }
}

fn main() {
    rustc().input("main.rs").opt().run();
    // panic machinery identifiers, these should not appear in the final binary
    let mut panic_syms = vec![sym("panic_bounds_check"), sym("Debug")];
    if no_debug_assertions() {
        // if debug assertions are allowed, we need to allow these,
        // otherwise, add them to the list of symbols to deny.
        panic_syms.extend_from_slice(&[
            sym("panicking"),
            sym("panic_fmt"),
            sym("pad_integral"),
            sym("Display"),
        ]);
    }

    if is_windows_msvc() {
        use pdb::FallibleIterator;

        let file = std::fs::File::open("main.pdb").expect("failed to open `main.pdb`");
        let mut pdb = pdb::PDB::open(file).expect("failed to parse `main.pdb`");

        let symbol_table = pdb.global_symbols().expect("failed to parse PDB global symbols");
        let mut symbols = symbol_table.iter();

        let mut found_symbols = vec![];

        while let Some(symbol) = symbols.next().expect("failed to parse symbol") {
            match symbol.parse() {
                Ok(pdb::SymbolData::Public(data)) => {
                    found_symbols.push(data.name.to_string());
                }
                _ => {}
            }
        }

        // Make sure we at least have the `main` symbol itself, otherwise even no symbols can
        // trivially satisfy the "no panic symbol" assertion.
        let main_sym = if is_darwin() {
            // Symbols are decorated with an underscore prefix on darwin platforms.
            "_main"
        } else if target().contains("i686") && is_windows_msvc() {
            // `extern "C"` i.e. `__cdecl` on `i686` windows-msvc means that the symbol will be
            // decorated with an underscore, but not on `x86_64` windows-msvc.
            // See <https://learn.microsoft.com/en-us/cpp/build/reference/decorated-names?view=msvc-170#FormatC>.
            "_main"
        } else {
            "main"
        };

        assert!(found_symbols.iter().any(|sym| sym == main_sym), "expected `main` symbol");

        for found_symbol in found_symbols {
            for panic_symbol in &panic_syms {
                assert_ne!(
                    found_symbol,
                    panic_symbol.as_str(),
                    "found unexpected panic machinery symbol"
                );
            }
        }
    } else {
        let panic_syms = panic_syms.iter().map(String::as_str).collect::<Vec<_>>();
        // Make sure we at least have the `main` symbol itself, otherwise even no symbols can
        // trivially satisfy the "no panic symbol" assertion.
        assert!(any_symbol_contains(bin_name("main"), &["main"]));
        assert!(!any_symbol_contains(bin_name("main"), &panic_syms));
    }
}
