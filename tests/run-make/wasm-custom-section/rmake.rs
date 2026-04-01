//@ only-wasm32-wasip1

use std::collections::HashMap;

use run_make_support::{rfs, rustc, wasmparser};

fn main() {
    rustc().input("foo.rs").target("wasm32-wasip1").run();
    rustc().input("bar.rs").target("wasm32-wasip1").arg("-Clto").opt().run();

    let file = rfs::read("bar.wasm");

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
