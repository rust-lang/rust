//@ only-wasm32-wasip1

use run_make_support::{rustc, wasmparser};
use std::path::Path;

fn main() {
    rustc().input("foo.rs").target("wasm32-wasip1").run();
    verify_symbols(Path::new("foo.wasm"));
    rustc().input("foo.rs").target("wasm32-wasip1").arg("-Clto").run();
    verify_symbols(Path::new("foo.wasm"));
    rustc().input("foo.rs").target("wasm32-wasip1").opt().run();
    verify_symbols(Path::new("foo.wasm"));
    rustc().input("foo.rs").target("wasm32-wasip1").arg("-Clto").opt().run();
    verify_symbols(Path::new("foo.wasm"));
}

fn verify_symbols(path: &Path) {
    eprintln!("verify {path:?}");
    let file = std::fs::read(&path).unwrap();

    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        let payload = payload.unwrap();
        if let wasmparser::Payload::ImportSection(_) = payload {
            panic!("import section found");
        }
    }
}
