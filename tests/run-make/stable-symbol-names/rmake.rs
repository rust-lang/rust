//@ needs-target-std
//
// A typo in rustc caused generic symbol names to be non-deterministic -
// that is, it was possible to compile the same file twice with no changes
// and get outputs with different symbol names.
// This test compiles each of the two crates twice, and checks that each output
// contains exactly the same symbol names.
// Additionally, both crates should agree on the same symbol names for monomorphic
// functions.
// See https://github.com/rust-lang/rust/issues/32554

use std::collections::HashSet;

use run_make_support::{llvm_readobj, regex, rfs, rust_lib_name, rustc};

static LEGACY_PATTERN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
static V0_PATTERN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();

fn main() {
    LEGACY_PATTERN.set(regex::Regex::new(r"_ZN.*E").unwrap()).unwrap();
    V0_PATTERN.set(regex::Regex::new(r"_R[a-zA-Z0-9_]*").unwrap()).unwrap();
    // test 1: first file
    rustc().input("stable-symbol-names1.rs").run();
    let sym1 = process_symbols("stable_symbol_names1", "generic_|mono_");
    rfs::remove_file(rust_lib_name("stable_symbol_names1"));
    rustc().input("stable-symbol-names1.rs").run();
    let sym2 = process_symbols("stable_symbol_names1", "generic_|mono_");
    assert_eq!(sym1, sym2);

    // test 2: second file
    rustc().input("stable-symbol-names2.rs").run();
    let sym1 = process_symbols("stable_symbol_names2", "generic_|mono_");
    rfs::remove_file(rust_lib_name("stable_symbol_names2"));
    rustc().input("stable-symbol-names2.rs").run();
    let sym2 = process_symbols("stable_symbol_names2", "generic_|mono_");
    assert_eq!(sym1, sym2);

    // test 3: crossed files
    let sym1 = process_symbols("stable_symbol_names1", "mono_");
    let sym2 = process_symbols("stable_symbol_names2", "mono_");
    assert_eq!(sym1, sym2);
}

#[track_caller]
fn process_symbols(path: &str, symbol: &str) -> Vec<String> {
    // Dump all symbols.
    let out = llvm_readobj().input(rust_lib_name(path)).symbols().run().stdout_utf8();
    // Extract only lines containing `symbol`.
    let symbol_regex = regex::Regex::new(symbol).unwrap();
    let out = out.lines().filter(|&line| symbol_regex.find(line).is_some());

    // HashSet - duplicates should be excluded!
    let mut symbols: HashSet<String> = HashSet::new();
    // From those lines, extract just the symbol name via `regex`, which:
    //   * always starts with "_ZN" and ends with "E" (`legacy` mangling)
    //   * always starts with "_R" (`v0` mangling)
    for line in out {
        if let Some(mat) = LEGACY_PATTERN.get().unwrap().find(line) {
            symbols.insert(mat.as_str().to_string());
        }
        if let Some(mat) = V0_PATTERN.get().unwrap().find(line) {
            symbols.insert(mat.as_str().to_string());
        }
    }

    let mut symbols: Vec<String> = symbols.into_iter().collect();
    // Sort those symbol names for deterministic comparison.
    symbols.sort();
    symbols
}
