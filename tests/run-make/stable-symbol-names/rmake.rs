// A typo in rustc caused generic symbol names to be non-deterministic -
// that is, it was possible to compile the same file twice with no changes
// and get outputs with different symbol names.
// This test compiles each of the two crates twice, and checks that each output
// contains exactly the same symbol names.
// Additionally, both crates should agree on the same symbol names for monomorphic
// functions.
// See https://github.com/rust-lang/rust/issues/32554

use run_make_support::{fs_wrapper, llvm_readobj, regex, rust_lib_name, rustc};
use std::collections::HashSet;

fn main() {
    // test 1: first file
    rustc().input("stable-symbol-names1.rs").run();
    let sym1 = process_symbols("stable_symbol_names1", "generic_|mono_");
    fs_wrapper::remove_file(rust_lib_name("stable_symbol_names1"));
    rustc().input("stable-symbol-names1.rs").run();
    let sym2 = process_symbols("stable_symbol_names1", "generic_|mono_");
    assert_eq!(sym1, sym2);

    // test 2: second file
    rustc().input("stable-symbol-names2.rs").run();
    let sym1 = process_symbols("stable_symbol_names2", "generic_|mono_");
    fs_wrapper::remove_file(rust_lib_name("stable_symbol_names2"));
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
    let out = llvm_readobj().input(rust_lib_name(path)).arg("--symbols").run().stdout_utf8();
    // Extract only lines containing `symbol`.
    let symbol_regex = regex::Regex::new(symbol).unwrap();
    let out = out.lines().filter(|&line| symbol_regex.find(line).is_some());
    // From those lines, extract just the symbol name via `regex`, which:
    //   * always starts with "_ZN" and ends with "E" (`legacy` mangling)
    //   * always starts with "_R" (`v0` mangling)
    let legacy_pattern = regex::Regex::new(r"_ZN.*E").unwrap();
    let v0_pattern = regex::Regex::new(r"_R[a-zA-Z0-9_]*").unwrap();

    // HashSet - duplicates should be excluded!
    let mut symbols: HashSet<String> = HashSet::new();
    for line in out {
        if let Some(mat) = legacy_pattern.find(line) {
            symbols.insert(mat.as_str().to_string());
        }
        if let Some(mat) = v0_pattern.find(line) {
            symbols.insert(mat.as_str().to_string());
        }
    }

    let mut symbols: Vec<String> = symbols.into_iter().collect();
    // Sort those symbol names for deterministic comparison.
    symbols.sort();
    symbols
}
