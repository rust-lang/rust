//@ only-wasm32

use std::collections::HashMap;

use run_make_support::{rfs, rustc, wasmparser};

fn main() {
    rustc().input("foo.rs").run();
    rustc().input("bar.rs").arg("-Clto").opt().run();

    let file = rfs::read("bar.wasm");

    let mut custom = HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        let payload = payload.unwrap();
        if let wasmparser::Payload::CustomSection(s) = payload {
            custom.insert(s.name(), s.data());
        }
    }

    assert_eq!(custom.remove("foo"), Some(&[5, 6, 1, 2][..]));
    assert_eq!(custom.remove("bar"), Some(&[3, 4][..]));
    assert_eq!(custom.remove("baz"), Some(&[7, 8][..]));
}
