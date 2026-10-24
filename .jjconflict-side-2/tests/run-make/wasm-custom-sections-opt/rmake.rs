//@ only-wasm32-wasip1

use std::collections::HashMap;
use std::path::Path;

use run_make_support::{rfs, rustc, wasmparser};

fn main() {
    rustc().input("foo.rs").target("wasm32-wasip1").opt().run();

    verify(Path::new("foo.wasm"));
}

fn verify(path: &Path) {
    eprintln!("verify {path:?}");
    let file = rfs::read(&path);

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
