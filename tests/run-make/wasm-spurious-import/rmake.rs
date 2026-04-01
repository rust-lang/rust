//@ only-wasm32-wasip1

use std::collections::HashMap;

use run_make_support::{rfs, rustc, wasmparser};

fn main() {
    rustc()
        .input("main.rs")
        .target("wasm32-wasip1")
        .arg("-Coverflow-checks")
        .arg("-Cpanic=abort")
        .arg("-Clto")
        .arg("-Copt-level=z")
        .run();

    let file = rfs::read("main.wasm");

    let mut imports = HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        let payload = payload.unwrap();
        if let wasmparser::Payload::ImportSection(s) = payload {
            for i in s {
                let i = i.unwrap();
                imports.entry(i.module).or_insert(Vec::new()).push((i.name, i.ty));
            }
        }
    }

    assert!(imports.is_empty(), "imports are not empty {:?}", imports);
}
