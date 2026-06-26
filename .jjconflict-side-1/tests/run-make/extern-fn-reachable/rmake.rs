//! Smoke test to check that that symbols of `extern "C"` functions and `#[no_mangle]` rust
//! functions:
//!
//! 1. Are externally visible in the dylib produced.
//! 2. That the symbol visibility is orthogonal to the Rust nameres visibility of the functions
//!    involved.

//@ ignore-cross-compile

use std::collections::BTreeSet;

use run_make_support::object::{self, Object};
use run_make_support::{dynamic_lib_name, is_darwin, path, rfs, rustc};

fn main() {
    let dylib = dynamic_lib_name("dylib");
    rustc().input("dylib.rs").output(&dylib).arg("-Cprefer-dynamic").run();

    let expected_symbols = if is_darwin() {
        // Mach-O states that all exported symbols should have an underscore as prefix. At the
        // same time dlsym will implicitly add it, so outside of compilers, linkers and people
        // writing assembly, nobody needs to be aware of this.
        BTreeSet::from(["_fun1", "_fun2", "_fun3", "_fun4", "_fun5", "_fun6"])
    } else {
        BTreeSet::from(["fun1", "fun2", "fun3", "fun4", "fun5", "fun6"])
    };

    let mut found_symbols = BTreeSet::new();

    let blob = rfs::read(path(dylib));
    let file = object::File::parse(&*blob).unwrap();
    for export in file.exports().unwrap() {
        let sym_name = export.name();
        let sym_name = std::str::from_utf8(sym_name).unwrap();
        found_symbols.insert(sym_name);
    }

    println!("expected_symbols = {:?}", expected_symbols);
    println!("found_symbols = {:?}", found_symbols);
    if !found_symbols.is_superset(&expected_symbols) {
        for diff in expected_symbols.difference(&found_symbols) {
            eprintln!("missing symbol: {}", diff);
        }
        panic!("missing expected symbols");
    }
}
