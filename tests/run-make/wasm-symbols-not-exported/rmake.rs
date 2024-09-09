//@ only-wasm32-wasip1

use std::path::Path;

use run_make_support::{rfs, rustc, wasmparser};

fn main() {
    rustc().input("foo.rs").target("wasm32-wasip1").run();
    verify_symbols(Path::new("foo.wasm"));
    rustc().input("foo.rs").target("wasm32-wasip1").opt().run();
    verify_symbols(Path::new("foo.wasm"));

    rustc().input("bar.rs").target("wasm32-wasip1").run();
    verify_symbols(Path::new("bar.wasm"));
    rustc().input("bar.rs").target("wasm32-wasip1").opt().run();
    verify_symbols(Path::new("bar.wasm"));
}

fn verify_symbols(path: &Path) {
    eprintln!("verify {path:?}");
    let file = rfs::read(&path);

    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        let payload = payload.unwrap();
        if let wasmparser::Payload::ExportSection(s) = payload {
            for e in s {
                let e = e.unwrap();
                if e.kind != wasmparser::ExternalKind::Func {
                    continue;
                }
                if e.name == "foo" {
                    continue;
                }
                panic!("unexpected export {e:?}");
            }
        }
    }
}
