extern crate run_make_support;

use run_make_support::{out_dir, rustc, wasmparser};
use std::path::Path;

fn main() {
    if std::env::var("TARGET").unwrap() != "wasm32-wasip1" {
        return;
    }

    rustc().arg("foo.rs").arg("--target=wasm32-wasip1").run();
    verify_symbols(&out_dir().join("foo.wasm"));
    rustc().arg("foo.rs").arg("--target=wasm32-wasip1").arg("-O").run();
    verify_symbols(&out_dir().join("foo.wasm"));

    rustc().arg("bar.rs").arg("--target=wasm32-wasip1").run();
    verify_symbols(&out_dir().join("bar.wasm"));
    rustc().arg("bar.rs").arg("--target=wasm32-wasip1").arg("-O").run();
    verify_symbols(&out_dir().join("bar.wasm"));
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
