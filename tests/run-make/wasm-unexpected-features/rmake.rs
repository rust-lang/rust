//@ only-wasm32-wasip1

use std::path::Path;

use run_make_support::{rfs, rustc, wasmparser};

fn main() {
    rustc()
        .input("foo.rs")
        .target("wasm32-wasip1")
        .target_cpu("mvp")
        .opt_level("z")
        .lto("fat")
        .linker_plugin_lto("on")
        .link_arg("--import-memory")
        .run();
    verify_features(Path::new("foo.wasm"));
}

fn verify_features(path: &Path) {
    eprintln!("verify {path:?}");
    let file = rfs::read(&path);

    let mut validator = wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::MVP);
    validator.validate_all(&file).unwrap();
}
