// Dynamic libraries on Rust used to export a very high amount of symbols,
// going as far as filling the output with mangled names and generic function
// names. After the rework of #38117, this test checks that no mangled Rust symbols
// are exported, and that generics are only shown if explicitely requested.
// See https://github.com/rust-lang/rust/issues/37530

//@ ignore-windows-msvc

use run_make_support::{bin_name, dynamic_lib_name, is_windows, llvm_readobj, regex, rustc};

fn main() {
    let mut cdylib_name = dynamic_lib_name("a_cdylib");
    let mut rdylib_name = dynamic_lib_name("a_rust_dylib");
    let exe_name = bin_name("an_executable");
    let mut combined_cdylib_name = dynamic_lib_name("combined_rlib_dylib");
    if is_windows() {
        cdylib_name.push_str(".a");
        rdylib_name.push_str(".a");
        combined_cdylib_name.push_str(".a");
    }
    rustc().arg("-Zshare-generics=no").input("an_rlib.rs").run();
    rustc().arg("-Zshare-generics=no").input("a_cdylib.rs").run();
    rustc().arg("-Zshare-generics=no").input("a_rust_dylib.rs").run();
    rustc().arg("-Zshare-generics=no").input("a_proc_macro.rs").run();
    rustc().arg("-Zshare-generics=no").input("an_executable.rs").run();
    rustc()
        .arg("-Zshare-generics=no")
        .input("a_cdylib.rs")
        .crate_name("combined_rlib_dylib")
        .crate_type("rlib,cdylib")
        .run();

    // Check that a cdylib exports its public #[no_mangle] functions
    symbols_check(&cdylib_name, SymbolCheckType::StrSymbol("public_c_function_from_cdylib"), 2);
    // Check that a cdylib exports the public #[no_mangle] functions of dependencies
    symbols_check(&cdylib_name, SymbolCheckType::StrSymbol("public_c_function_from_rlib"), 2);
    // Check that a cdylib DOES NOT export any public Rust functions
    symbols_check(&cdylib_name, SymbolCheckType::AnyRustSymbol, 0);

    // Check that a Rust dylib exports its monomorphic functions
    symbols_check(&rdylib_name, SymbolCheckType::StrSymbol("public_c_function_from_rust_dylib"), 2);
    symbols_check(
        &rdylib_name,
        SymbolCheckType::StrSymbol("public_rust_function_from_rust_dylib"),
        2,
    );
    // Check that a Rust dylib does not export generics if -Zshare-generics=no
    symbols_check(
        &rdylib_name,
        SymbolCheckType::StrSymbol("public_generic_function_from_rust_dylib"),
        1,
    );

    // Check that a Rust dylib exports the monomorphic functions from its dependencies
    symbols_check(&rdylib_name, SymbolCheckType::StrSymbol("public_c_function_from_rlib"), 2);
    symbols_check(&rdylib_name, SymbolCheckType::StrSymbol("public_rust_function_from_rlib"), 2);
    // Check that a Rust dylib does not export generics if -Zshare-generics=no
    symbols_check(&rdylib_name, SymbolCheckType::StrSymbol("public_generic_function_from_rlib"), 1);

    if is_windows() {
        // Check that an executable does not export any dynamic symbols
        symbols_check(&exe_name, SymbolCheckType::StrSymbol("public_c_function_from_rlib"), 0);
        symbols_check(&exe_name, SymbolCheckType::StrSymbol("public_rust_function_from_exe"), 0);
    }

    // Check the combined case, where we generate a cdylib and an rlib in the same
    // compilation session:
    // Check that a cdylib exports its public //[no_mangle] functions
    symbols_check(
        &combined_cdylib_name,
        SymbolCheckType::StrSymbol("public_c_function_from_cdylib"),
        2,
    );
    // Check that a cdylib exports the public //[no_mangle] functions of dependencies
    symbols_check(
        &combined_cdylib_name,
        SymbolCheckType::StrSymbol("public_c_function_from_rlib"),
        2,
    );
    // Check that a cdylib DOES NOT export any public Rust functions
    symbols_check(&combined_cdylib_name, SymbolCheckType::AnyRustSymbol, 0);

    rustc().arg("-Zshare-generics=yes").input("an_rlib.rs").run();
    rustc().arg("-Zshare-generics=yes").input("a_cdylib.rs").run();
    rustc().arg("-Zshare-generics=yes").input("a_rust_dylib.rs").run();
    rustc().arg("-Zshare-generics=yes").input("an_executable.rs").run();

    // Check that a cdylib exports its public //[no_mangle] functions
    symbols_check(&cdylib_name, SymbolCheckType::StrSymbol("public_c_function_from_cdylib"), 2);
    // Check that a cdylib exports the public //[no_mangle] functions of dependencies
    symbols_check(&cdylib_name, SymbolCheckType::StrSymbol("public_c_function_from_rlib"), 2);
    // Check that a cdylib DOES NOT export any public Rust functions
    symbols_check(&cdylib_name, SymbolCheckType::AnyRustSymbol, 0);

    // Check that a Rust dylib exports its monomorphic functions, including generics this time
    symbols_check(&rdylib_name, SymbolCheckType::StrSymbol("public_c_function_from_rust_dylib"), 2);
    symbols_check(
        &rdylib_name,
        SymbolCheckType::StrSymbol("public_rust_function_from_rust_dylib"),
        2,
    );
    symbols_check(
        &rdylib_name,
        SymbolCheckType::StrSymbol("public_generic_function_from_rust_dylib"),
        2,
    );

    // Check that a Rust dylib exports the monomorphic functions from its dependencies
    symbols_check(&rdylib_name, SymbolCheckType::StrSymbol("public_c_function_from_rlib"), 2);
    symbols_check(&rdylib_name, SymbolCheckType::StrSymbol("public_rust_function_from_rlib"), 2);
    symbols_check(&rdylib_name, SymbolCheckType::StrSymbol("public_generic_function_from_rlib"), 2);

    if is_windows() {
        // Check that an executable does not export any dynamic symbols
        symbols_check(&exe_name, SymbolCheckType::StrSymbol("public_c_function_from_rlib"), 0);
        symbols_check(&exe_name, SymbolCheckType::StrSymbol("public_rust_function_from_exe"), 0);
    }
}

#[track_caller]
fn symbols_check(path: &str, symbol_check_type: SymbolCheckType, count: usize) {
    let out = llvm_readobj().arg("--symbols").input(path).run().stdout_utf8();
    assert_eq!(
        out.lines()
            .filter(|&line| !line.contains("__imp_") && has_symbol(line, symbol_check_type))
            .count(),
        count
    );
}

fn has_symbol(line: &str, symbol_check_type: SymbolCheckType) -> bool {
    if let SymbolCheckType::StrSymbol(expected) = symbol_check_type {
        line.contains(expected)
    } else {
        let regex = regex::Regex::new(r#"_ZN.*h.*E\|_R[a-zA-Z0-9_]+"#).unwrap();
        regex.is_match(line)
    }
}

#[derive(Clone, Copy)]
enum SymbolCheckType {
    StrSymbol(&'static str),
    AnyRustSymbol,
}
