//! Verifies that the examples in <https://github.com/rcvalle/rust-cfi-examples> build and run with
//!`-Zbuild-std` to prevent regressions such as [rust-lang/rust#142284].

//@ needs-sanitizer-cfi
//@ needs-force-clang-based-tests
//@ needs-rust-lld
//@ needs-target-std
//@ ignore-cross-compile
//@ only-x86_64-unknown-linux-gnu

#![deny(warnings)]

use std::path::Path;

use run_make_support::external_deps::rustc::sysroot as rustc_sysroot;
use run_make_support::run::cmd;
use run_make_support::{bin_name, cargo, path, target};

fn clang_path() -> String {
    if let Ok(d) = std::env::var("LLVM_BIN_DIR") {
        let clang = Path::new(d.trim_end_matches('/')).join("clang");
        if clang.exists() {
            return clang.display().to_string();
        }
    }
    if let Ok(clang) = std::env::var("CLANG") {
        let clang = Path::new(clang.trim_end_matches('/'));
        if clang.exists() {
            return clang.display().to_string();
        }
    }
    "clang".to_string()
}

fn fuse_ld_path() -> String {
    if let Ok(d) = std::env::var("LLVM_BIN_DIR") {
        let llvm_bin_dir = Path::new(d.trim_end_matches('/'));
        let gcc_ld_lld = llvm_bin_dir.join("gcc-ld").join("ld.lld");
        if gcc_ld_lld.exists() {
            return gcc_ld_lld.display().to_string();
        }
        let ld_lld = llvm_bin_dir.join("ld.lld");
        if ld_lld.exists() {
            return ld_lld.display().to_string();
        }
    }
    if let Ok(clang) = std::env::var("CLANG") {
        let clang = Path::new(clang.trim_end_matches('/'));
        if let Some(clang_dir) = clang.parent() {
            let gcc_ld_lld = clang_dir.join("gcc-ld").join("ld.lld");
            if gcc_ld_lld.exists() {
                return gcc_ld_lld.display().to_string();
            }
            let ld_lld = clang_dir.join("ld.lld");
            if ld_lld.exists() {
                return ld_lld.display().to_string();
            }
        }
    }
    let target_bin_dir = rustc_sysroot().join("lib").join("rustlib").join(target()).join("bin");
    let gcc_ld_lld = target_bin_dir.join("gcc-ld").join("ld.lld");
    if gcc_ld_lld.exists() {
        return gcc_ld_lld.display().to_string();
    }
    "ld.lld".to_string()
}

fn run_and_expect_cfi_abort(target_dir: &Path, target: &str, binary: &str) {
    let exe = target_dir.join(target).join("release").join(bin_name(binary));
    let output = cmd(&exe).run_fail();
    output
        .assert_stdout_contains("With CFI enabled, you should not see the next answer")
        .assert_stdout_not_contains("The next answer is:");
}

fn run_and_expect_cfi_not_abort(target_dir: &Path, target: &str, binary: &str) {
    let exe = target_dir.join(target).join("release").join(bin_name(binary));
    let output = cmd(&exe).run();
    output.assert_stdout_contains("Hello from C!");
}

fn main() {
    let clang = clang_path();
    let fuse_ld = fuse_ld_path();
    let tgt = target();
    let target_dir = path("target");
    let lib = std::env::var("LIB").unwrap_or_default();

    let prior_rustflags = std::env::var("RUSTFLAGS").unwrap_or_default();

    let rustflags = format!(
        "{prior_rustflags} -Clinker-plugin-lto -Clinker={clang} \
         -Clink-arg=-fuse-ld={fuse_ld} -Zsanitizer=cfi \
         -Ctarget-feature=-crt-static"
    )
    .trim()
    .to_owned();

    let rustflags_with_integer_normalization =
        format!("{rustflags} -Zsanitizer-cfi-normalize-integers").trim().to_owned();

    let run = |manifest: &Path, rustflags: &str, workspace: bool| {
        let mut c = cargo();
        c.arg("build")
            .arg("--manifest-path")
            .arg(manifest)
            .arg("--release")
            .arg("-Zbuild-std")
            .arg("--target")
            .arg(&tgt);
        if workspace {
            c.arg("--workspace");
        }
        c.env("RUSTFLAGS", rustflags)
            .env("CC", &clang)
            .env("CARGO_TARGET_DIR", &target_dir)
            .env("RUSTC_BOOTSTRAP", "1")
            .env("LIB", &lib)
            .run();
    };

    run(Path::new("Cargo.toml"), &rustflags, true);
    for bin in [
        "invalid-branch-target-abort",
        "indirect-arity-mismatch-abort",
        "indirect-pointee-type-mismatch-abort",
        "indirect-return-type-mismatch-abort",
        "indirect-type-qualifier-mismatch-abort",
        "indirect-type-mismatch-abort",
        "cross-lang-cfi-types-crate-abort",
    ] {
        run_and_expect_cfi_abort(&target_dir, &tgt, bin);
    }
    for bin in ["cross-lang-cfi-types-crate-not-abort"] {
        run_and_expect_cfi_not_abort(&target_dir, &tgt, bin);
    }

    run(
        Path::new("cross-lang-integer-normalization-abort/Cargo.toml"),
        &rustflags_with_integer_normalization,
        false,
    );
    run_and_expect_cfi_abort(&target_dir, &tgt, "cross-lang-integer-normalization-abort");

    run(
        Path::new("cross-lang-integer-normalization-not-abort/Cargo.toml"),
        &rustflags_with_integer_normalization,
        false,
    );
    run_and_expect_cfi_not_abort(&target_dir, &tgt, "cross-lang-integer-normalization-not-abort");
}
