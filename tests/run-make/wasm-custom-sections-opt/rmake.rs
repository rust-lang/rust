extern crate run_make_support;

use run_make_support::{out_dir, rustc, wasmparser};
use std::collections::HashMap;
use std::path::Path;

fn main() {
    if std::env::var("TARGET").unwrap() != "wasm32-wasip1" {
        return;
    }

    rustc().arg("foo.rs").arg("--target=wasm32-wasip1").arg("-O").run();
    verify(&out_dir().join("foo.wasm"));
}

fn verify(path: &Path) {
    eprintln!("verify {path:?}");
    let file = std::fs::read(&path).unwrap();

    let mut custom = HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        let payload = payload.unwrap();
        if let wasmparser::Payload::CustomSection(s) = payload {
            let prev = custom.insert(s.name(), s.data());
            assert!(prev.is_none());
        }
    }

    assert_eq!(custom.remove("foo"), Some(&[1, 2, 3, 4][..]));
}
