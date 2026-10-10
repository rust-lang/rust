//@ only-wasm32

use std::collections::HashMap;

use run_make_support::{rfs, rustc, wasmparser};

fn main() {
    rustc()
        .input("main.rs")
        .arg("-Coverflow-checks")
        .arg("-Cpanic=abort")
        .arg("-Clto")
        .arg("-Copt-level=z")
        .emit_wasm_core_module()
        .run();

    let file = rfs::read("main.wasm");

    let mut imports = HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        let payload = payload.unwrap();
        if let wasmparser::Payload::ImportSection(s) = payload {
            for i in s.into_imports() {
                let i = i.unwrap();
                // ignore intrinsics like `__wasm_{get,set}_stack_pointer`
                if i.name.starts_with("__wasm_") {
                    continue;
                }
                imports.entry(i.module).or_insert(Vec::new()).push((i.name, i.ty));
            }
        }
    }

    assert!(imports.is_empty(), "imports are not empty {:?}", imports);
}
