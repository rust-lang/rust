//@ only-wasm32-unknown-emscripten

// On wasm the age-old C trick of a `main` that is either `int main(void)` or
// `int main(int, char**)` doesn't work, because wasm requires caller and callee
// signatures to match. The platform ABI (as used by Clang) therefore mangles
// the entry point by signature; the argc/argv form is emitted as
// `__main_argc_argv`. Rust's emscripten target uses the argc/argv form, so its
// entry point must be emitted under that name rather than as a raw `main`.
//
// This matters beyond cosmetics: emscripten's own crt references
// `__main_argc_argv` directly on some entry paths (notably `crt1_proxy_main.o`,
// used by `-sPROXY_TO_PTHREAD`), which fail to link against a raw `main`.
//
// The JS-visible name of the entry point stays `_main`; emscripten bridges that
// to the wasm `__main_argc_argv` export internally.

use run_make_support::{rfs, rustc, wasmparser};
use wasmparser::ExternalKind;

fn main() {
    rustc().input("main.rs").target("wasm32-unknown-emscripten").output("main.js").run();

    let file = rfs::read("main.wasm");
    let mut entry = None;
    let mut has_plain_main = false;
    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        if let wasmparser::Payload::ExportSection(s) = payload.unwrap() {
            for export in s {
                let export = export.unwrap();
                match export.name {
                    "__main_argc_argv" => entry = Some(export.kind),
                    "main" => has_plain_main = true,
                    _ => {}
                }
            }
        }
    }

    assert_eq!(
        entry,
        Some(ExternalKind::Func),
        "the emscripten entry point must be exported as the wasm C-ABI symbol `__main_argc_argv`",
    );
    assert!(!has_plain_main, "a raw `main` symbol must not be exported on wasm");
}
