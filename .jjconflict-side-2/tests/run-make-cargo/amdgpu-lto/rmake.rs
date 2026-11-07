// Check that compiling for the amdgpu target which needs LTO works with a default
// cargo configuration.

//@ needs-llvm-components: amdgpu
//@ needs-rust-lld

#![deny(warnings)]

use run_make_support::{cargo, path};

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
            "amdgcn-amd-amdhsa",
        ])
        .env("RUSTFLAGS", "-Ctarget-cpu=gfx900")
        .env("CARGO_TARGET_DIR", &target_dir)
        .run();
}
