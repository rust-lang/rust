// This is a codegen test which checks that when code is compiled with panic=immediate-abort,
// we get a `tail call void @llvm.trap()` in user code instead of a call into the standard
// library's panic formatting code (such as panic_fmt) or one of the numerous panic outlining shims
// (such as slice_index_fail).

#![deny(warnings)]

use run_make_support::{cargo, llvm_filecheck, path, rfs, target};

fn main() {
    let target_dir = path("target");

    cargo()
        .args(&[
            "build",
            "--release",
            "--lib",
            "--manifest-path",
            "Cargo.toml",
            "-Zbuild-std=core",
            "--target",
            &target(),
        ])
        .env("RUSTFLAGS", "-Zunstable-options -Cpanic=immediate-abort")
        .env("CARGO_TARGET_DIR", &target_dir)
        .env("RUSTC_BOOTSTRAP", "1")
        // Visual Studio 2022 requires that the LIB env var be set so it can
        // find the Windows SDK.
        .env("LIB", std::env::var("LIB").unwrap_or_default())
        .run();

    let out_dir = target_dir.join(target()).join("release").join("deps");
    let ir_file = rfs::read_dir(out_dir)
        .find_map(|e| {
            let path = e.unwrap().path();
            let file_name = path.file_name().unwrap().to_str().unwrap();
            if file_name.starts_with("panic_scenarios") && file_name.ends_with(".ll") {
                Some(path)
            } else {
                None
            }
        })
        .unwrap();

    llvm_filecheck().patterns("lib.rs").input_file(ir_file).run();
}
