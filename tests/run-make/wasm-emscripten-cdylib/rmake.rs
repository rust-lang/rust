//! Check that cdylib crate type is supported for the wasm32-unknown-emscripten
//! target and produces a valid Emscripten dynamic library.
//!
//! A cdylib has no `main`, so rustc must link it as entryless (`--no-entry`),
//! otherwise emcc pulls in the standalone-wasm entry shim and `wasm-ld` errors
//! with `undefined symbol: main`. Executables must keep their entry, so this
//! also checks that a `bin` crate with `main` still links.

//@ only-wasm32-unknown-emscripten

use run_make_support::{bare_rustc, rfs, wasmparser};

fn main() {
    // A cdylib must link without `--no-entry` being supplied by the user.
    bare_rustc().input("foo.rs").target("wasm32-unknown-emscripten").crate_type("cdylib").run();

    // Verify the output is a valid wasm file with a dylink.0 section
    let file = rfs::read("foo.wasm");
    let mut has_dylink = false;

    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        let payload = payload.unwrap();
        if let wasmparser::Payload::CustomSection(s) = payload {
            if s.name() == "dylink.0" {
                has_dylink = true;
            }
        }
    }

    assert!(has_dylink, "expected dylink.0 section in emscripten cdylib output");

    // An executable has a `main`, so `--no-entry` must NOT be applied: it must
    // still link with its entry preserved.
    bare_rustc().input("main.rs").target("wasm32-unknown-emscripten").crate_type("bin").run();
}
