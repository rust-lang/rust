use std::path::Path;

use anyhow::Context;
use camino::{Utf8Path, Utf8PathBuf};

use crate::environment::{Environment, executable_extension};
use crate::exec::cmd;
use crate::utils::io::{copy_directory, find_file_in_dir, unpack_archive};

/// Run tests on optimized dist artifacts.
pub fn run_tests(env: &Environment) -> anyhow::Result<()> {
    // After `dist` is executed, we extract its archived components into a sysroot directory,
    // and then use that extracted rustc as a stage0 compiler.
    // Then we run a subset of tests using that compiler, to have a basic smoke test which checks
    // whether the optimization pipeline hasn't broken something.
    let build_dir = env.build_root().join("build");
    let dist_dir = build_dir.join("dist");
    let unpacked_dist_dir = build_dir.join("unpacked-dist");
    std::fs::create_dir_all(&unpacked_dist_dir)?;

    let extract_dist_dir = |name: &str| -> anyhow::Result<Utf8PathBuf> {
        unpack_archive(&dist_dir.join(format!("{name}.tar.xz")), &unpacked_dist_dir)?;
        let extracted_path = unpacked_dist_dir.join(name);
        assert!(extracted_path.is_dir());
        Ok(extracted_path)
    };
    let host_triple = env.host_tuple();
    let version = find_dist_version(&dist_dir)?;

    let channel = version_to_channel(&version);

    // Extract rustc, libstd, cargo and src archives to create the optimized sysroot
    let rustc_dir = extract_dist_dir(&format!("rustc-{version}-{host_triple}"))?.join("rustc");
    let libstd_dir = extract_dist_dir(&format!("rust-std-{version}-{host_triple}"))?
        .join(format!("rust-std-{host_triple}"));
    let cargo_dir = extract_dist_dir(&format!("cargo-{version}-{host_triple}"))?.join("cargo");
    let extracted_src_dir = extract_dist_dir(&format!("rust-src-{version}"))?.join("rust-src");

    // We need to manually copy libstd to the extracted rustc sysroot
    copy_directory(
        &libstd_dir.join("lib").join("rustlib").join(host_triple).join("lib"),
        &rustc_dir.join("lib").join("rustlib").join(host_triple).join("lib"),
    )?;

    // Extract sources - they aren't in the `rustc-nightly-{host}` tarball, so we need to manually copy libstd
    // sources to the extracted sysroot. We need sources available so that `-Zsimulate-remapped-rust-src-base`
    // works correctly.
    copy_directory(
        &extracted_src_dir.join("lib").join("rustlib").join("src"),
        &rustc_dir.join("lib").join("rustlib").join("src"),
    )?;

    let rustc_path = rustc_dir.join("bin").join(format!("rustc{}", executable_extension()));
    assert!(rustc_path.is_file());
    let cargo_path = cargo_dir.join("bin").join(format!("cargo{}", executable_extension()));
    assert!(cargo_path.is_file());

    // Specify path to a LLVM config so that LLVM is not rebuilt.
    // It doesn't really matter which LLVM config we choose, because no sysroot will be compiled.
    let llvm_config = env
        .build_artifacts()
        .join("llvm")
        .join("bin")
        .join(format!("llvm-config{}", executable_extension()));
    assert!(llvm_config.is_file());

    let config_content = format!(
        r#"
profile = "user"
change-id = 115898

[rust]
channel = "{channel}"
verbose-tests = true
# rust-lld cannot be combined with an external LLVM
lld = false

[build]
rustc = "{rustc}"
cargo = "{cargo}"
local-rebuild = true

[target.{host_triple}]
llvm-config = "{llvm_config}"
"#,
        rustc = rustc_path.to_string().replace('\\', "/"),
        cargo = cargo_path.to_string().replace('\\', "/"),
        llvm_config = llvm_config.to_string().replace('\\', "/")
    );
    log::info!("Using following `bootstrap.toml` for running tests:\n{config_content}");

    // Simulate a stage 0 compiler with the extracted optimized dist artifacts.
    with_backed_up_file(Path::new("bootstrap.toml"), &config_content, || {
        let x_py = env.checkout_path().join("x.py");
        let mut args = vec![
            env.python_binary(),
            x_py.as_str(),
            "test",
            "--build",
            env.host_tuple(),
            "--stage",
            "0",
            "tests/assembly",
            "tests/codegen",
            "tests/codegen-units",
            "tests/incremental",
            "tests/mir-opt",
            "tests/pretty",
            "tests/run-make/glibc-symbols-x86_64-unknown-linux-gnu",
            "tests/ui",
            "tests/crashes",
        ];
        for test_path in env.skipped_tests() {
            args.extend(["--skip", test_path]);
        }
        cmd(&args)
            .env("COMPILETEST_FORCE_STAGE0", "1")
            // Also run dist-only tests
            .env("COMPILETEST_ENABLE_DIST_TESTS", "1")
            .run()
            .context("Cannot execute tests")
    })
}

/// Backup `path` (if it exists), then write `contents` into it, and then restore the original
/// contents of the file.
fn with_backed_up_file<F>(path: &Path, contents: &str, func: F) -> anyhow::Result<()>
where
    F: FnOnce() -> anyhow::Result<()>,
{
    let original_contents =
        if path.is_file() { Some(std::fs::read_to_string(path)?) } else { None };

    // Overwrite it with new contents
    std::fs::write(path, contents)?;

    let ret = func();

    if let Some(original_contents) = original_contents {
        std::fs::write(path, original_contents)?;
    }

    ret
}

/// Tries to find the version of the dist artifacts (either nightly, beta, or 1.XY.Z).
fn find_dist_version(directory: &Utf8Path) -> anyhow::Result<String> {
    // Lookup a known file with a unique prefix and extract the version from its filename
    let archive = find_file_in_dir(directory, "reproducible-artifacts-", ".tar.xz")?
        .file_name()
        .unwrap()
        .to_string();
    let (version, _) =
        archive.strip_prefix("reproducible-artifacts-").unwrap().split_once('-').unwrap();
    Ok(version.to_string())
}

/// Roughly convert a version string (`nightly`, `beta`, or `1.XY.Z`) to channel string (`nightly`,
/// `beta` or `stable`).
fn version_to_channel(version_str: &str) -> &'static str {
    match version_str {
        "nightly" => "nightly",
        "beta" => "beta",
        _ => "stable",
    }
}
