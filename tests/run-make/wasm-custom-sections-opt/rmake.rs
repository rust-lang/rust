//@ only-wasm32

use std::collections::HashMap;
use std::path::Path;

use run_make_support::{rfs, rustc, wasmparser};

fn main() {
    rustc().input("foo.rs").opt().run();

    verify(Path::new("foo.wasm"));
}

fn verify(path: &Path) {
    eprintln!("verify {path:?}");
    let file = rfs::read(&path);

    let mut custom = HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        let payload = payload.unwrap();
        if let wasmparser::Payload::CustomSection(s) = payload {
            custom.insert(s.name(), s.data());
        }
    }

    assert_eq!(custom.remove("foo"), Some(&[1, 2, 3, 4][..]));
}
