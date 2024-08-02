//@ ignore-cross-compile
//@ only-linux
//@ only-x86_64

use run_make_support::object::ObjectSymbol;
use run_make_support::symbols::*;
use run_make_support::{bin_name, cwd, dynamic_lib_name, run, rust_lib_name, rustc};

fn main() {
    rustc()
        .prefer_dynamic()
        .symbol_mangling_version("hashed")
        .metadata("foo")
        .input("a_dylib.rs")
        .run();
    rustc()
        .prefer_dynamic()
        .symbol_mangling_version("hashed")
        .metadata("bar")
        .input("a_rlib.rs")
        .run();
    /*println!("{}", cwd().display());
    dbg!(cwd().display());
    use std::io::Write;
    std::io::stdout().flush().unwrap();
    std::fs::write("/tmp/the_dir", cwd().display().to_string()).unwrap();
    loop {}*/
    //run("sh");
    //run("pwd");
    rustc().prefer_dynamic().library_search_path(cwd()).input("b_dylib.rs").run();
    rustc().prefer_dynamic().library_search_path(cwd()).input("b_bin.rs").run();
    let a_dylib = dynamic_lib_name("a_dylib");
    assert!(!any_symbol_contains(&a_dylib, &["hello"]));
    assert!(contains_exact_symbols(
        &a_dylib,
        &["_RNxC7a_dylib12H98WkzJ7B2nk", "_RNxC7a_dylib12HjermeVgSqiY",]
    ));
    let b_dylib = dynamic_lib_name("b_dylib");
    // b_dylib was compiled with regular symbol mangling.
    assert!(any_symbol_contains(&b_dylib, &["hello"]));
    // it depends on a_rlib, which was compiled with
    // hashed symbol mangling.
    assert!(contains_exact_symbols(
        &b_dylib,
        &[
            "_RNxC6a_rlib12H85r05hDVgWS",
            "_RNxC6a_rlib12HeiQWRC1rtuF",
            "_RNxC7a_dylib12HjermeVgSqiY",
        ]
    ));
    let b_bin = bin_name("b_bin");
    assert!(contains_exact_symbols(
        &b_bin,
        &[
            "_RNxC6a_rlib12HeiQWRC1rtuF",
            "_RNxC7a_dylib12HjermeVgSqiY",
            "_ZN7b_dylib5hello17h3a39df941aa66c40E",
        ]
    ));
    print_symbols(b_bin);
}
