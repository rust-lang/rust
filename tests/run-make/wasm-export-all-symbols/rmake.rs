//@ only-wasm32

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

    rustc().input("bar.rs").args(args).run();
    rustc().input("foo.rs").args(args).emit_wasm_core_module().run();
    rustc().input("main.rs").args(args).emit_wasm_core_module().run();

    verify_exports(
        Path::new("foo.wasm"),
        &[("foo", Func), ("FOO", Global), ("memory", Memory), ("_initialize", Func)],
    );
    verify_exports(
        Path::new("main.wasm"),
        &[("foo", Func), ("FOO", Global), ("__main_void", Func), ("memory", Memory)],
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
                match export.name {
                    "__wasm_task_hook"
                    | "cabi_realloc"
                    | "__indirect_function_table"
                    | "_start" => {}
                    name if name.starts_with("wasi:") => {}
                    other => {
                        wasm_exports.insert(other, export.kind);
                    }
                }
            }
        }
    }

    eprintln!("found exports {wasm_exports:?}");

    assert_eq!(exports.len(), wasm_exports.len());
    for (export, expected_kind) in exports {
        assert_eq!(wasm_exports[export], *expected_kind);
    }
}
