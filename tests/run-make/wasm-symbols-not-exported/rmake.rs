//@ only-wasm32

use std::path::Path;

use run_make_support::{rfs, rustc, wasmparser};

fn main() {
    rustc().input("foo.rs").emit_wasm_core_module().run();
    verify_symbols(Path::new("foo.wasm"));
    rustc().input("foo.rs").emit_wasm_core_module().opt().run();
    verify_symbols(Path::new("foo.wasm"));

    rustc().input("bar.rs").emit_wasm_core_module().arg("-Cpanic=abort").run();
    verify_symbols(Path::new("bar.wasm"));
    rustc().input("bar.rs").emit_wasm_core_module().arg("-Cpanic=abort").opt().run();
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
                if e.name == "foo"
                    || e.name == "_initialize"
                    || e.name == "__wasm_task_hook"
                    || e.name == "cabi_realloc"
                {
                    continue;
                }
                panic!("unexpected export {e:?}");
            }
        }
    }
}
