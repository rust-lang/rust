extern crate run_make_support;

use run_make_support::{out_dir, rustc, wasmparser};
use std::collections::HashMap;
use wasmparser::TypeRef::Func;

fn main() {
    if std::env::var("TARGET").unwrap() != "wasm32-wasip1" {
        return;
    }

    rustc().arg("foo.rs").arg("--target=wasm32-wasip1").run();
    rustc().arg("bar.rs").arg("--target=wasm32-wasip1").arg("-Clto").arg("-O").run();

    let file = std::fs::read(&out_dir().join("bar.wasm")).unwrap();

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

    let import = imports.remove("./dep");
    assert!(matches!(import.as_deref(), Some([("dep", Func(_))])), "bad import {:?}", import);
    let import = imports.remove("./me");
    assert!(matches!(import.as_deref(), Some([("me_in_dep", Func(_))])), "bad import {:?}", import);
}
