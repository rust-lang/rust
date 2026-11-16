//@ only-wasm32

use std::path::Path;

use run_make_support::{rfs, rustc, wasmparser};

fn main() {
    rustc().input("foo.rs").emit_wasm_core_module().arg("-Cpanic=abort").run();
    verify_symbols(Path::new("foo.wasm"));
    rustc().input("foo.rs").emit_wasm_core_module().arg("-Cpanic=abort").arg("-Clto").run();
    verify_symbols(Path::new("foo.wasm"));
    rustc().input("foo.rs").emit_wasm_core_module().arg("-Cpanic=abort").opt().run();
    verify_symbols(Path::new("foo.wasm"));
    rustc().input("foo.rs").emit_wasm_core_module().arg("-Cpanic=abort").arg("-Clto").opt().run();
    verify_symbols(Path::new("foo.wasm"));
}

fn verify_symbols(path: &Path) {
    eprintln!("verify {path:?}");
    let file = rfs::read(&path);

    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        let payload = payload.unwrap();
        if let wasmparser::Payload::ImportSection(i) = payload {
            for import in i.into_imports() {
                let import = import.unwrap();
                if !import.name.starts_with("__wasm_") {
                    panic!("unexpected import: {import:?}");
                }
            }
        }
    }
}
