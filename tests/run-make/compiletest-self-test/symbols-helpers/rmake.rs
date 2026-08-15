//! `run_make_support::symbols` helpers self test.

// Only intended as a basic smoke test, does not try to account for platform or calling convention
// specific symbol decorations.
//@ only-x86_64-unknown-linux-gnu
//@ ignore-cross-compile

use std::collections::BTreeSet;

use object::{Object, ObjectSymbol};
use run_make_support::symbols::{
    ContainsAllSymbolSubstringsOutcome, ContainsAllSymbolsOutcome,
    object_contains_all_symbol_substring, object_contains_all_symbols, object_contains_any_symbol,
    object_contains_any_symbol_substring,
};
use run_make_support::{object, rfs, rust_lib_name, rustc};

fn main() {
    rustc().input("sample.rs").emit("obj").edition("2024").run();

    // `sample.rs` has two `no_mangle` functions, `eszett` and `beta`, in addition to `main`.
    //
    // These two symbol names and the test substrings used below are carefully picked to make sure
    // they do not overlap with `sample` and contain non-hex characters, to avoid accidentally
    // matching against CGU names like `sample.dad0f15d00c84e70-cgu.0`.

    let obj_filename = "sample.o";
    let blob = rfs::read(obj_filename);
    let obj = object::File::parse(&*blob).unwrap();
    eprintln!("found symbols:");
    for sym in obj.symbols() {
        eprintln!("symbol = {}", sym.name().unwrap());
    }

    // `hello` contains `hel`
    assert!(object_contains_any_symbol_substring(obj_filename, &["zett"]));
    assert!(object_contains_any_symbol_substring(obj_filename, &["zett", "does_not_exist"]));
    assert!(!object_contains_any_symbol_substring(obj_filename, &["does_not_exist"]));

    assert!(object_contains_any_symbol(obj_filename, &["eszett"]));
    assert!(object_contains_any_symbol(obj_filename, &["eszett", "beta"]));
    assert!(!object_contains_any_symbol(obj_filename, &["zett"]));
    assert!(!object_contains_any_symbol(obj_filename, &["does_not_exist"]));

    assert_eq!(
        object_contains_all_symbol_substring(obj_filename, &["zett"]),
        ContainsAllSymbolSubstringsOutcome::Ok
    );
    assert_eq!(
        object_contains_all_symbol_substring(obj_filename, &["zett", "bet"]),
        ContainsAllSymbolSubstringsOutcome::Ok
    );
    assert_eq!(
        object_contains_all_symbol_substring(obj_filename, &["does_not_exist"]),
        ContainsAllSymbolSubstringsOutcome::MissingSymbolSubstrings(BTreeSet::from([
            "does_not_exist"
        ]))
    );
    assert_eq!(
        object_contains_all_symbol_substring(obj_filename, &["zett", "does_not_exist"]),
        ContainsAllSymbolSubstringsOutcome::MissingSymbolSubstrings(BTreeSet::from([
            "does_not_exist"
        ]))
    );
    assert_eq!(
        object_contains_all_symbol_substring(obj_filename, &["zett", "bet", "does_not_exist"]),
        ContainsAllSymbolSubstringsOutcome::MissingSymbolSubstrings(BTreeSet::from([
            "does_not_exist"
        ]))
    );

    assert_eq!(
        object_contains_all_symbols(obj_filename, &["eszett"]),
        ContainsAllSymbolsOutcome::Ok
    );
    assert_eq!(
        object_contains_all_symbols(obj_filename, &["eszett", "beta"]),
        ContainsAllSymbolsOutcome::Ok
    );
    assert_eq!(
        object_contains_all_symbols(obj_filename, &["zett"]),
        ContainsAllSymbolsOutcome::MissingSymbols(BTreeSet::from(["zett"]))
    );
    assert_eq!(
        object_contains_all_symbols(obj_filename, &["zett", "beta"]),
        ContainsAllSymbolsOutcome::MissingSymbols(BTreeSet::from(["zett"]))
    );
    assert_eq!(
        object_contains_all_symbols(obj_filename, &["does_not_exist"]),
        ContainsAllSymbolsOutcome::MissingSymbols(BTreeSet::from(["does_not_exist"]))
    );
}
