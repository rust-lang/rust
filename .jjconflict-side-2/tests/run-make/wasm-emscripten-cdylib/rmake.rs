//! Check that cdylib crate type is supported for the wasm32-unknown-emscripten
//! target and produces a valid Emscripten dynamic library.

//@ only-wasm32-unknown-emscripten

use run_make_support::{bare_rustc, rfs, wasmparser};

fn main() {
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
}
