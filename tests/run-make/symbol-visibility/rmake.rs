//@ ignore-cross-compile
//@ needs-crate-type: dylib, cdylib, proc-macro
//@ needs-dynamic-linking

// Dynamic libraries on Rust used to export a very high amount of symbols,
// going as far as filling the output with mangled names and generic function
// names. After the rework of #38117, this test checks that no mangled Rust symbols
// are exported, and that generics are only shown if explicitly requested.
// See https://github.com/rust-lang/rust/issues/37530

use run_make_support::object::read::Object;
use run_make_support::{bin_name, dynamic_lib_name, is_msvc, object, regex, rfs, rustc};

fn main() {
    let cdylib_name = dynamic_lib_name("a_cdylib");
    let rdylib_name = dynamic_lib_name("a_rust_dylib");
    let exe_name = bin_name("an_executable");
    let combined_cdylib_name = dynamic_lib_name("combined_rlib_dylib");
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
    symbols_check(&cdylib_name, SymbolCheckType::StrSymbol("public_c_function_from_cdylib"), true);
    // Check that a cdylib exports the public #[no_mangle] functions of dependencies
    symbols_check(&cdylib_name, SymbolCheckType::StrSymbol("public_c_function_from_rlib"), true);
    // Check that a cdylib DOES NOT export any public Rust functions
    symbols_check(&cdylib_name, SymbolCheckType::AnyRustSymbol, false);

    // Check that a Rust dylib exports its monomorphic functions
    symbols_check(
        &rdylib_name,
        SymbolCheckType::StrSymbol("public_c_function_from_rust_dylib"),
        true,
    );
    symbols_check(
        &rdylib_name,
        SymbolCheckType::StrSymbol("public_rust_function_from_rust_dylib"),
        true,
    );
    // Check that a Rust dylib does not export generics if -Zshare-generics=no
    symbols_check(
        &rdylib_name,
        SymbolCheckType::StrSymbol("public_generic_function_from_rust_dylib"),
        false,
    );

    // Check that a Rust dylib exports the monomorphic functions from its dependencies
    symbols_check(&rdylib_name, SymbolCheckType::StrSymbol("public_c_function_from_rlib"), true);
    symbols_check(&rdylib_name, SymbolCheckType::StrSymbol("public_rust_function_from_rlib"), true);
    // Check that a Rust dylib does not export generics if -Zshare-generics=no
    symbols_check(
        &rdylib_name,
        SymbolCheckType::StrSymbol("public_generic_function_from_rlib"),
        false,
    );

    // FIXME(nbdd0121): This is broken in MinGW, see https://github.com/rust-lang/rust/pull/95604#issuecomment-1101564032
    if is_msvc() {
        // Check that an executable does not export any dynamic symbols
        symbols_check(&exe_name, SymbolCheckType::StrSymbol("public_c_function_from_rlib"), false);
        symbols_check(
            &exe_name,
            SymbolCheckType::StrSymbol("public_rust_function_from_exe"),
            false,
        );
    }

    // Check the combined case, where we generate a cdylib and an rlib in the same
    // compilation session:
    // Check that a cdylib exports its public #[no_mangle] functions
    symbols_check(
        &combined_cdylib_name,
        SymbolCheckType::StrSymbol("public_c_function_from_cdylib"),
        true,
    );
    // Check that a cdylib exports the public #[no_mangle] functions of dependencies
    symbols_check(
        &combined_cdylib_name,
        SymbolCheckType::StrSymbol("public_c_function_from_rlib"),
        true,
    );
    // Check that a cdylib DOES NOT export any public Rust functions
    symbols_check(&combined_cdylib_name, SymbolCheckType::AnyRustSymbol, false);

    rustc().arg("-Zshare-generics=yes").input("an_rlib.rs").run();
    rustc().arg("-Zshare-generics=yes").input("a_cdylib.rs").run();
    rustc().arg("-Zshare-generics=yes").input("a_rust_dylib.rs").run();
    rustc().arg("-Zshare-generics=yes").input("an_executable.rs").run();

    // Check that a cdylib exports its public #[no_mangle] functions
    symbols_check(&cdylib_name, SymbolCheckType::StrSymbol("public_c_function_from_cdylib"), true);
    // Check that a cdylib exports the public #[no_mangle] functions of dependencies
    symbols_check(&cdylib_name, SymbolCheckType::StrSymbol("public_c_function_from_rlib"), true);
    // Check that a cdylib DOES NOT export any public Rust functions
    symbols_check(&cdylib_name, SymbolCheckType::AnyRustSymbol, false);

    // Check that a Rust dylib exports its monomorphic functions, including generics this time
    symbols_check(
        &rdylib_name,
        SymbolCheckType::StrSymbol("public_c_function_from_rust_dylib"),
        true,
    );
    symbols_check(
        &rdylib_name,
        SymbolCheckType::StrSymbol("public_rust_function_from_rust_dylib"),
        true,
    );
    symbols_check(
        &rdylib_name,
        SymbolCheckType::StrSymbol("public_generic_function_from_rust_dylib"),
        true,
    );

    // Check that a Rust dylib exports the monomorphic functions from its dependencies
    symbols_check(&rdylib_name, SymbolCheckType::StrSymbol("public_c_function_from_rlib"), true);
    symbols_check(&rdylib_name, SymbolCheckType::StrSymbol("public_rust_function_from_rlib"), true);
    symbols_check(
        &rdylib_name,
        SymbolCheckType::StrSymbol("public_generic_function_from_rlib"),
        true,
    );

    // FIXME(nbdd0121): This is broken in MinGW, see https://github.com/rust-lang/rust/pull/95604#issuecomment-1101564032
    if is_msvc() {
        // Check that an executable does not export any dynamic symbols
        symbols_check(&exe_name, SymbolCheckType::StrSymbol("public_c_function_from_rlib"), false);
        symbols_check(
            &exe_name,
            SymbolCheckType::StrSymbol("public_rust_function_from_exe"),
            false,
        );
    }
}

#[track_caller]
fn symbols_check(path: &str, symbol_check_type: SymbolCheckType, exists_once: bool) {
    let binary_data = rfs::read(path);
    let file = object::File::parse(&*binary_data).unwrap();
    let mut found: u64 = 0;
    for export in file.exports().unwrap() {
        let name = std::str::from_utf8(export.name()).unwrap();
        if has_symbol(name, symbol_check_type) {
            found += 1;
        }
    }
    assert_eq!(found, exists_once as u64);
}

fn has_symbol(name: &str, symbol_check_type: SymbolCheckType) -> bool {
    if let SymbolCheckType::StrSymbol(expected) = symbol_check_type {
        name.contains(expected)
    } else {
        let regex = regex::Regex::new(r#"_ZN.*h.*E\|_R[a-zA-Z0-9_]+"#).unwrap();
        regex.is_match(name)
    }
}

#[derive(Clone, Copy)]
enum SymbolCheckType {
    StrSymbol(&'static str),
    AnyRustSymbol,
}
