//@ only-wasm32-wasip1

use std::collections::HashMap;
use std::path::Path;

use run_make_support::{rfs, rustc, wasmparser};
use wasmparser::ExternalKind::*;

fn main() {
    test(&[]);
    test(&["-O"]);
    test(&["-Clto"]);
}

fn test(args: &[&str]) {
    eprintln!("running with {args:?}");

    rustc().input("bar.rs").target("wasm32-wasip1").args(args).run();
    rustc().input("foo.rs").target("wasm32-wasip1").args(args).run();
    rustc().input("main.rs").target("wasm32-wasip1").args(args).run();

    verify_exports(Path::new("foo.wasm"), &[("foo", Func), ("FOO", Global), ("memory", Memory)]);
    verify_exports(
        Path::new("main.wasm"),
        &[
            ("foo", Func),
            ("FOO", Global),
            ("_start", Func),
            ("__main_void", Func),
            ("memory", Memory),
        ],
    );
}

fn verify_exports(path: &Path, exports: &[(&str, wasmparser::ExternalKind)]) {
    println!("verify {path:?}");
    let file = rfs::read(path);
    let mut wasm_exports = HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        let payload = payload.unwrap();
        if let wasmparser::Payload::ExportSection(s) = payload {
            for export in s {
                let export = export.unwrap();
                wasm_exports.insert(export.name, export.kind);
            }
        }
    }

    eprintln!("found exports {wasm_exports:?}");

    assert_eq!(exports.len(), wasm_exports.len());
    for (export, expected_kind) in exports {
        assert_eq!(wasm_exports[export], *expected_kind);
    }
}
