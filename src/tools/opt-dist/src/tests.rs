use crate::environment::Environment;
use crate::exec::cmd;
use crate::utils::io::{copy_directory, unpack_archive};
use anyhow::Context;
use camino::Utf8PathBuf;

/// Run tests on optimized dist artifacts.
pub fn run_tests(env: &dyn Environment) -> anyhow::Result<()> {
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
    let host_triple = env.host_triple();

    // Extract rustc, libstd, cargo and src archives to create the optimized sysroot
    let rustc_dir = extract_dist_dir(&format!("rustc-nightly-{host_triple}"))?.join("rustc");
    let libstd_dir = extract_dist_dir(&format!("rust-std-nightly-{host_triple}"))?
        .join(format!("rust-std-{host_triple}"));
    let cargo_dir = extract_dist_dir(&format!("cargo-nightly-{host_triple}"))?.join("cargo");
    let extracted_src_dir = extract_dist_dir("rust-src-nightly")?.join("rust-src");

    // We need to manually copy libstd to the extracted rustc sysroot
    copy_directory(
        &libstd_dir.join("lib").join("rustlib").join(&host_triple).join("lib"),
        &rustc_dir.join("lib").join("rustlib").join(&host_triple).join("lib"),
    )?;

    // Extract sources - they aren't in the `rustc-nightly-{host}` tarball, so we need to manually copy libstd
    // sources to the extracted sysroot. We need sources available so that `-Zsimulate-remapped-rust-src-base`
    // works correctly.
    copy_directory(
        &extracted_src_dir.join("lib").join("rustlib").join("src"),
        &rustc_dir.join("lib").join("rustlib").join("src"),
    )?;

    let rustc_path = rustc_dir.join("bin").join(format!("rustc{}", env.executable_extension()));
    assert!(rustc_path.is_file());
    let cargo_path = cargo_dir.join("bin").join(format!("cargo{}", env.executable_extension()));
    assert!(cargo_path.is_file());

    // Specify path to a LLVM config so that LLVM is not rebuilt.
    // It doesn't really matter which LLVM config we choose, because no sysroot will be compiled.
    let llvm_config = env
        .build_artifacts()
        .join("llvm")
        .join("bin")
        .join(format!("llvm-config{}", env.executable_extension()));
    assert!(llvm_config.is_file());

    let config_content = format!(
        r#"profile = "user"
changelog-seen = 2

[build]
rustc = "{rustc}"
cargo = "{cargo}"

[target.{host_triple}]
llvm-config = "{llvm_config}"
"#,
        rustc = rustc_path.to_string().replace('\\', "/"),
        cargo = cargo_path.to_string().replace('\\', "/"),
        llvm_config = llvm_config.to_string().replace('\\', "/")
    );
    log::info!("Using following `config.toml` for running tests:\n{config_content}");

    // Simulate a stage 0 compiler with the extracted optimized dist artifacts.
    std::fs::write("config.toml", config_content)?;

    let x_py = env.checkout_path().join("x.py");
    let mut args = vec![
        env.python_binary(),
        x_py.as_str(),
        "test",
        "--stage",
        "0",
        "tests/assembly",
        "tests/codegen",
        "tests/codegen-units",
        "tests/incremental",
        "tests/mir-opt",
        "tests/pretty",
        "tests/run-pass-valgrind",
        "tests/ui",
    ];
    for test_path in env.skipped_tests() {
        args.extend(["--exclude", test_path]);
    }
    cmd(&args).env("COMPILETEST_FORCE_STAGE0", "1").run().context("Cannot execute tests")
}
