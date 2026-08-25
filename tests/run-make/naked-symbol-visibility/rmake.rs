//@ ignore-windows
//@ only-x86_64
//@ needs-target-std
//@ needs-crate-type: dylib

use run_make_support::object::ObjectSymbol;
use run_make_support::object::read::{File, Object, Symbol};
use run_make_support::targets::is_windows;
use run_make_support::{dynamic_lib_name, rfs, rustc};

fn main() {
    let rdylib_name = dynamic_lib_name("a_rust_dylib");
    rustc().arg("-Zshare-generics=no").input("a_rust_dylib.rs").run();

    let binary_data = rfs::read(&rdylib_name);
    let rdylib = File::parse(&*binary_data).unwrap();

    // naked should mirror vanilla
    not_exported(&rdylib, "private_vanilla");
    not_exported(&rdylib, "private_naked");

    global_function(&rdylib, "public_vanilla");
    global_function(&rdylib, "public_naked_nongeneric");

    not_exported(&rdylib, "public_vanilla_generic");
    // #[naked] functions are implicitly #[inline(never)], so they get shared regardless of
    // -Zshare-generics.
    global_function(&rdylib, "public_naked_generic");

    global_function(&rdylib, "vanilla_external_linkage");
    global_function(&rdylib, "naked_external_linkage");

    // FIXME: make this work on windows (gnu and msvc). See the PR
    // https://github.com/rust-lang/rust/pull/128362 for some approaches
    // that don't work
    //
    // #[linkage = "weak"] does not work well on windows, we get
    //
    // lib.def : error LNK2001: unresolved external symbol naked_weak_linkage␍
    // lib.def : error LNK2001: unresolved external symbol vanilla_weak_linkage
    //
    // so just skip weak symbols on windows (for now)
    if !is_windows() {
        weak_function(&rdylib, "vanilla_weak_linkage");
        weak_function(&rdylib, "naked_weak_linkage");
    }

    // functions that are declared in an `extern "C"` block are currently not exported
    // this maybe should change in the future, this is just tracking the current behavior
    // reported in https://github.com/rust-lang/rust/issues/128071
    not_exported(&rdylib, "function_defined_in_global_asm");

    // share generics should expose the generic functions
    rustc().arg("-Zshare-generics=yes").input("a_rust_dylib.rs").run();
    let binary_data = rfs::read(&rdylib_name);
    let rdylib = File::parse(&*binary_data).unwrap();

    global_function(&rdylib, "public_vanilla_generic");
    global_function(&rdylib, "public_naked_generic");
}

#[track_caller]
fn global_function(file: &File, symbol_name: &str) {
    let symbols = find_dynamic_symbol(file, symbol_name);
    let [symbol] = symbols.as_slice() else {
        panic!("symbol {symbol_name} occurs {} times", symbols.len())
    };

    assert!(symbol.is_definition(), "`{symbol_name}` is not a function");
    assert!(symbol.is_global(), "`{symbol_name}` is not marked as global");
}

#[track_caller]
fn weak_function(file: &File, symbol_name: &str) {
    let symbols = find_dynamic_symbol(file, symbol_name);
    let [symbol] = symbols.as_slice() else {
        panic!("symbol {symbol_name} occurs {} times", symbols.len())
    };

    assert!(symbol.is_definition(), "`{symbol_name}` is not a function");
    assert!(symbol.is_weak(), "`{symbol_name}` is not marked as weak");
}

#[track_caller]
fn not_exported(file: &File, symbol_name: &str) {
    assert_eq!(find_dynamic_symbol(file, symbol_name).len(), 0)
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|window| window == needle)
}

fn find_dynamic_symbol<'file, 'data>(
    file: &'file File<'data>,
    expected: &str,
) -> Vec<Symbol<'data, 'file>> {
    file.exports()
        .unwrap()
        .into_iter()
        .filter(|e| find_subsequence(e.name(), expected.as_bytes()))
        .filter_map(|e| file.symbol_by_name_bytes(e.name()))
        .collect()
}
