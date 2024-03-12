extern crate run_make_support;

use run_make_support::{out_dir, rustc, wasmparser};
use std::collections::HashMap;

fn main() {
    if std::env::var("TARGET").unwrap() != "wasm32-wasip1" {
        return;
    }

    rustc().arg("foo.rs").arg("--target=wasm32-wasip1").run();
    rustc().arg("bar.rs").arg("--target=wasm32-wasip1").arg("-Clto").arg("-O").run();

    let file = std::fs::read(&out_dir().join("bar.wasm")).unwrap();

    let mut custom = HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        let payload = payload.unwrap();
        if let wasmparser::Payload::CustomSection(s) = payload {
            let prev = custom.insert(s.name(), s.data());
            assert!(prev.is_none());
        }
    }

    assert_eq!(custom.remove("foo"), Some(&[5, 6, 1, 2][..]));
    assert_eq!(custom.remove("bar"), Some(&[3, 4][..]));
    assert_eq!(custom.remove("baz"), Some(&[7, 8][..]));
}
