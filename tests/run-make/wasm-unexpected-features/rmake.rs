//@ only-wasm32-wasip1

use std::path::Path;

use run_make_support::{cargo, path, rfs, target, wasmparser};

fn main() {
    let target_dir = path("target");

    cargo()
        .args([
            "rustc",
            "--manifest-path",
            "wasm32_test/Cargo.toml",
            "--profile",
            "release",
            "--target",
            "wasm32-wasip1",
            "-Zbuild-std=core,alloc,panic_abort",
            "--",
            "-Clink-arg=--import-memory",
            "-Clinker-plugin-lto=on",
        ])
        .env("RUSTFLAGS", "-Ctarget-cpu=mvp")
        .env("CARGO_TARGET_DIR", &target_dir)
        .run();

    let wasm32_program_path = target_dir.join(target()).join("release").join("wasm32_program.wasm");
    verify_features(&wasm32_program_path);
}

fn verify_features(path: &Path) {
    eprintln!("verify {path:?}");
    let file = rfs::read(&path);

    let mut validator = wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::WASM1);
    validator.validate_all(&file).unwrap();
}
