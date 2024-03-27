extern crate run_make_support;

use run_make_support::{rustc, tmp_dir, wasmparser};
use std::path::Path;

fn main() {
    if std::env::var("TARGET").unwrap() != "wasm32-wasip1" {
        return;
    }

    rustc().input("foo.rs").target("wasm32-wasip1").run();
    verify_symbols(&tmp_dir().join("foo.wasm"));
    rustc().input("foo.rs").target("wasm32-wasip1").opt().run();
    verify_symbols(&tmp_dir().join("foo.wasm"));

    rustc().input("bar.rs").target("wasm32-wasip1").run();
    verify_symbols(&tmp_dir().join("bar.wasm"));
    rustc().input("bar.rs").target("wasm32-wasip1").opt().run();
    verify_symbols(&tmp_dir().join("bar.wasm"));
}

fn verify_symbols(path: &Path) {
    eprintln!("verify {path:?}");
    let file = std::fs::read(&path).unwrap();

    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        let payload = payload.unwrap();
        if let wasmparser::Payload::ExportSection(s) = payload {
            for e in s {
                let e = e.unwrap();
                if e.kind != wasmparser::ExternalKind::Func {
                    continue;
                }
                if e.name == "foo" {
                    continue;
                }
                panic!("unexpected export {e:?}");
            }
        }
    }
}
