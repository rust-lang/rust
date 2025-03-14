// ignore-tidy-linelength
//! Basic smoke test for the unstable option `-C symbol_mangling_version=hashed` which aims to
//! replace full symbol mangling names based on hash digests to shorten symbol name lengths in
//! dylibs for space savings.
//!
//! # References
//!
//! - MCP #705: Provide option to shorten symbol names by replacing them with a digest:
//!   <https://github.com/rust-lang/compiler-team/issues/705>.
//! - Implementation PR: <https://github.com/rust-lang/rust/pull/118636>.
//! - PE format: <https://learn.microsoft.com/en-us/windows/win32/debug/pe-format>.

//@ ignore-cross-compile

#![deny(warnings)]

use run_make_support::symbols::exported_dynamic_symbol_names;
use run_make_support::{bin_name, cwd, dynamic_lib_name, is_darwin, object, rfs, run, rustc};

macro_rules! adjust_symbol_prefix {
    ($name:literal) => {
        if is_darwin() { concat!("_", $name) } else { $name }
    };
}

fn main() {
    rustc()
        .input("hashed_dylib.rs")
        .prefer_dynamic()
        .arg("-Zunstable-options")
        .symbol_mangling_version("hashed")
        .metadata("foo")
        .run();

    rustc()
        .input("hashed_rlib.rs")
        .prefer_dynamic()
        .arg("-Zunstable-options")
        .symbol_mangling_version("hashed")
        .metadata("bar")
        .run();

    rustc().input("default_dylib.rs").library_search_path(cwd()).prefer_dynamic().run();
    rustc().input("default_bin.rs").library_search_path(cwd()).prefer_dynamic().run();

    {
        // Check hashed symbol name

        let dylib_filename = dynamic_lib_name("hashed_dylib");
        println!("checking dylib `{dylib_filename}`");

        let dylib_blob = rfs::read(&dylib_filename);
        let dylib_file = object::File::parse(&*dylib_blob)
            .unwrap_or_else(|e| panic!("failed to parse `{dylib_filename}`: {e}"));

        let dynamic_symbols = exported_dynamic_symbol_names(&dylib_file);

        if dynamic_symbols.iter().filter(|sym| sym.contains("hdhello")).count() != 0 {
            eprintln!("exported dynamic symbols: {:#?}", dynamic_symbols);
            panic!("expected no occurrence of `hdhello`");
        }

        let expected_prefix = adjust_symbol_prefix!("_RNxC12hashed_dylib");
        if dynamic_symbols.iter().filter(|sym| sym.starts_with(expected_prefix)).count() != 2 {
            eprintln!("exported dynamic symbols: {:#?}", dynamic_symbols);
            panic!("expected two dynamic symbols starting with `{expected_prefix}`");
        }
    }

    {
        let dylib_filename = dynamic_lib_name("default_dylib");
        println!("checking so `{dylib_filename}`");

        let dylib_blob = rfs::read(&dylib_filename);
        let dylib_file = object::File::parse(&*dylib_blob)
            .unwrap_or_else(|e| panic!("failed to parse `{dylib_filename}`: {e}"));

        let dynamic_symbols = exported_dynamic_symbol_names(&dylib_file);

        if dynamic_symbols
            .iter()
            .filter(|sym| sym.contains("default_dylib") && sym.contains("ddhello"))
            .count()
            != 1
        {
            eprintln!("exported dynamic symbols: {:#?}", dynamic_symbols);
            panic!("expected one occurrence of mangled `ddhello`");
        }

        let expected_rlib_prefix = adjust_symbol_prefix!("_RNxC11hashed_rlib");
        if dynamic_symbols.iter().filter(|sym| sym.starts_with(expected_rlib_prefix)).count() != 2 {
            eprintln!("exported dynamic symbols: {:#?}", dynamic_symbols);
            panic!("expected two exported symbols starting with `{expected_rlib_prefix}`");
        }

        let expected_dylib_prefix = adjust_symbol_prefix!("_RNxC12hashed_dylib");
        if dynamic_symbols.iter().any(|sym| sym.starts_with("_RNxC12hashed_dylib")) {
            eprintln!("exported dynamic symbols: {:#?}", dynamic_symbols);
            panic!("did not expect any symbols starting with `{expected_dylib_prefix}`");
        }
    }

    // Check that the final binary can be run.
    {
        let bin_filename = bin_name("default_bin");
        run(&bin_filename);
    }
}
