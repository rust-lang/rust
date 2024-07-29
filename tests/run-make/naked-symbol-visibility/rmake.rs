// @only-x86_64
use run_make_support::{dynamic_lib_name, llvm_readobj, regex, rustc};

fn main() {
    let rdylib_name = dynamic_lib_name("a_rust_dylib");
    rustc().arg("-Zshare-generics=no").input("a_rust_dylib.rs").run();

    // check vanilla symbols
    not_exported(&rdylib_name, "private_vanilla_rust_function_from_rust_dylib");
    global_function(&rdylib_name, "public_vanilla_rust_function_from_rust_dylib");
    not_exported(&rdylib_name, "public_vanilla_generic_function_from_rust_dylib");

    weak_function(&rdylib_name, "vanilla_weak_linkage");
    global_function(&rdylib_name, "vanilla_external_linkage");

    // naked should mirror vanilla
    not_exported(&rdylib_name, "private_naked_rust_function_from_rust_dylib");
    global_function(&rdylib_name, "public_naked_rust_function_from_rust_dylib");
    not_exported(&rdylib_name, "public_naked_generic_function_from_rust_dylib");

    weak_function(&rdylib_name, "naked_weak_linkage");
    global_function(&rdylib_name, "naked_external_linkage");

    // share generics should expose the generic functions
    rustc().arg("-Zshare-generics=yes").input("a_rust_dylib.rs").run();
    global_function(&rdylib_name, "public_vanilla_generic_function_from_rust_dylib");
    global_function(&rdylib_name, "public_naked_generic_function_from_rust_dylib");
}

#[track_caller]
fn global_function(path: &str, symbol_name: &str) {
    let lines = find_dynamic_symbol(path, symbol_name);
    let [line] = lines.as_slice() else {
        panic!("symbol {symbol_name} occurs {} times", lines.len())
    };

    assert!(line.contains("FUNC"), "`{symbol_name}` is not a function");
    assert!(line.contains("GLOBAL"), "`{symbol_name}` is not marked as global");
}

#[track_caller]
fn weak_function(path: &str, symbol_name: &str) {
    let lines = find_dynamic_symbol(path, symbol_name);
    let [line] = lines.as_slice() else {
        panic!("symbol {symbol_name} occurs {} times", lines.len())
    };

    assert!(line.contains("FUNC"), "`{symbol_name}` is not a function");
    assert!(line.contains("WEAK"), "`{symbol_name}` is not marked as weak");
}

#[track_caller]
fn not_exported(path: &str, symbol_name: &str) {
    assert_eq!(find_dynamic_symbol(path, symbol_name).len(), 0)
}

fn find_dynamic_symbol<'a>(path: &str, symbol_name: &str) -> Vec<String> {
    let out = llvm_readobj().arg("--dyn-symbols").input(path).run().stdout_utf8();
    out.lines()
        .filter(|&line| !line.contains("__imp_") && line.contains(symbol_name))
        .map(|line| line.to_string())
        .collect()
}
