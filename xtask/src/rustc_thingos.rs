//! Build and package a stage-1 `rustc` cross-compiler for `x86_64-unknown-thingos`.
//!
//! The primary artifact is produced from `x.py build --stage 1`:
//!
//! 1. Linux-hosted cross-compiler (`target/rustc-thingos/rustc`): runs on
//!    `x86_64-unknown-linux-gnu`, targets `x86_64-unknown-thingos`.
//! 2. ThingOS-native compiler (`target/rustc-thingos/thingos-rustc`): an
//!    optional stage-1 `rustc` binary cross-compiled for
//!    `x86_64-unknown-thingos`.
//!
//! Unlike the old out-of-tree setup, this repository is the Rust source tree.
//! There is no `vendor/rust` indirection here.

use anyhow::{Context, Result, bail};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use xshell::{Shell, cmd};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

const RUSTC_BINARY: &str = "target/rustc-thingos/rustc";
const RUSTC_WRAPPER: &str = "target/rustc-thingos/rustc-wrapper";
const CACHE_KEY_FILE: &str = "target/rustc-thingos/.cache-key";
const OUTPUT_DIR: &str = "target/rustc-thingos";
const RUSTLIB_CACHE_DIR: &str = "target/rustc-thingos/lib/rustlib";
const LIB_CACHE_DIR: &str = "target/rustc-thingos/lib";
const THINGOS_RUSTC_BINARY: &str = "target/rustc-thingos/thingos-rustc";
const THINGOS_RUSTLIB_CACHE_DIR: &str = "target/rustc-thingos/thingos-rustlib";
const ROOT_CONFIG_TOML: &str = "config.toml";
const ROOT_CONFIG_BACKUP: &str = "target/rustc-thingos/config.toml.backup";

fn compute_cache_key() -> String {
    let mut hasher = DefaultHasher::new();

    for path in &["targets/x86_64-unknown-thingos.json", "rust-toolchain.toml"] {
        if let Ok(content) = std::fs::read_to_string(path) {
            content.hash(&mut hasher);
        }
    }

    if let Ok(output) = Command::new("git").args(["rev-parse", "HEAD"]).output() {
        if output.status.success() {
            output.stdout.hash(&mut hasher);
        }
    }

    format!("{:016x}", hasher.finish())
}

fn is_cache_valid() -> bool {
    if !Path::new(RUSTC_BINARY).exists() {
        return false;
    }

    match std::fs::read_to_string(CACHE_KEY_FILE) {
        Ok(stored) => stored.trim() == compute_cache_key(),
        Err(_) => false,
    }
}

fn write_cache_key() -> std::io::Result<()> {
    std::fs::write(CACHE_KEY_FILE, compute_cache_key())
}

fn should_attempt_native_recovery() -> bool {
    std::env::var("BUILD_THINGOS_NATIVE_RUSTC").as_deref() == Ok("1")
}

fn write_rustc_wrapper(cwd: &Path) -> Result<()> {
    let wrapper_path = cwd.join(RUSTC_WRAPPER);
    let compiler_path = cwd.join(RUSTC_BINARY);
    let sysroot = cwd.join("target/rustc-thingos");
    let libdir = cwd.join(LIB_CACHE_DIR);

    let script = format!(
        "#!/usr/bin/env bash\nset -euo pipefail\nexport LD_LIBRARY_PATH=\"{libdir}:${{LD_LIBRARY_PATH:-}}\"\nexec \"{compiler}\" --sysroot \"{sysroot}\" \"$@\"\n",
        libdir = libdir.display(),
        compiler = compiler_path.display(),
        sysroot = sysroot.display(),
    );

    std::fs::write(&wrapper_path, script)?;
    #[cfg(unix)]
    {
        let mut perms = std::fs::metadata(&wrapper_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&wrapper_path, perms)?;
    }

    Ok(())
}

fn xpy_build_root(cwd: &Path) -> PathBuf {
    cwd.join("build/x86_64-unknown-linux-gnu")
}

fn locate_stage1_rustc(cwd: &Path) -> Option<PathBuf> {
    let root_build = xpy_build_root(cwd);
    let candidates = [
        root_build.join("stage1/bin/rustc"),
        root_build.join("stage1-rustc/x86_64-unknown-linux-gnu/release/rustc-main"),
    ];

    candidates.into_iter().find(|path| path.exists())
}

fn locate_thingos_rustc(cwd: &Path) -> Option<PathBuf> {
    let root_build = xpy_build_root(cwd);
    let candidates = [
        root_build.join("stage1-rustc/x86_64-unknown-thingos/release/rustc-main"),
        cwd.join("build/x86_64-unknown-thingos/stage1/bin/rustc"),
    ];

    candidates.into_iter().find(|path| path.exists())
}

fn cache_rustlib_tree(sh: &Shell, cwd: &Path) -> Result<()> {
    let build_root = xpy_build_root(cwd);
    let cached_rustlib = cwd.join(RUSTLIB_CACHE_DIR);
    let cached_lib = cwd.join(LIB_CACHE_DIR);

    sh.remove_path(&cached_rustlib)?;
    sh.create_dir(&cached_rustlib)?;
    sh.create_dir(&cached_lib)?;

    let host_rustlib = build_root.join("stage1/lib/rustlib");
    if host_rustlib.exists() {
        let host_src = host_rustlib.to_str().context("non-utf8 rustlib path")?;
        let host_dst = cached_rustlib.to_str().context("non-utf8 rustlib cache path")?;
        cmd!(sh, "cp -r {host_src}/. {host_dst}").run()?;
    }

    let host_lib = build_root.join("stage1/lib");
    if host_lib.exists() {
        for entry in std::fs::read_dir(&host_lib)? {
            let entry = entry?;
            let path = entry.path();
            let keep = matches!(path.extension().and_then(|ext| ext.to_str()), Some("so"))
                || path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.contains(".so."));
            if keep {
                let file_name = path.file_name().context("missing shared library file name")?;
                std::fs::copy(&path, cached_lib.join(file_name))?;
            }
        }
    }

    let thingos_lib = build_root.join("stage1-std/x86_64-unknown-thingos/release/deps");
    if thingos_lib.exists() {
        let thingos_dst = cached_rustlib.join("x86_64-unknown-thingos/lib");
        sh.create_dir(thingos_dst.parent().context("missing ThingOS rustlib parent")?)?;
        sh.create_dir(&thingos_dst)?;
        for entry in std::fs::read_dir(&thingos_lib)? {
            let entry = entry?;
            let path = entry.path();
            let keep = matches!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("rlib") | Some("rmeta")
            );
            if keep {
                let file_name = path.file_name().context("missing ThingOS rustlib file name")?;
                std::fs::copy(&path, thingos_dst.join(file_name))?;
            }
        }
    }

    Ok(())
}

fn cache_thingos_rustc_tree(sh: &Shell, cwd: &Path, thingos_rustc: &Path) -> Result<()> {
    let cached_dir = cwd.join(THINGOS_RUSTLIB_CACHE_DIR);

    sh.remove_path(&cached_dir)?;
    sh.create_dir(&cached_dir)?;
    sh.copy_file(thingos_rustc, cwd.join(THINGOS_RUSTC_BINARY))?;

    #[cfg(unix)]
    {
        let binary_path = cwd.join(THINGOS_RUSTC_BINARY);
        let mut perms = std::fs::metadata(&binary_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&binary_path, perms)?;
    }

    let thingos_rlibs_dst = cached_dir.join("x86_64-unknown-thingos/lib");
    sh.create_dir(thingos_rlibs_dst.parent().context("missing native rustlib parent")?)?;
    sh.create_dir(&thingos_rlibs_dst)?;

    let build_root = xpy_build_root(cwd);
    let thingos_std = build_root.join("stage1-std/x86_64-unknown-thingos/release/deps");
    if thingos_std.exists() {
        for entry in std::fs::read_dir(&thingos_std)? {
            let entry = entry?;
            let path = entry.path();
            let keep = matches!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("rlib") | Some("rmeta")
            );
            if keep {
                let file_name = path.file_name().context("missing native rustlib file name")?;
                std::fs::copy(&path, thingos_rlibs_dst.join(file_name))?;
            }
        }
    } else {
        println!(
            "rustc-thingos: warning: ThingOS std rlibs not found at {:?}; the ThingOS-native rustc will ship without prebuilt libraries",
            thingos_std
        );
    }

    Ok(())
}

fn bootstrap_config(include_thingos_host: bool, local_rebuild: bool) -> String {
    let host_list = if include_thingos_host {
        "[\"x86_64-unknown-linux-gnu\", \"x86_64-unknown-thingos\"]"
    } else {
        "[\"x86_64-unknown-linux-gnu\"]"
    };

    let local_rebuild = if local_rebuild { "true" } else { "false" };

    format!(
        r#"change-id = "ignore"
# Auto-generated by `cargo run -p xtask -- rustc-thingos` - do not edit by hand.
[build]
build = "x86_64-unknown-linux-gnu"
host = {host_list}
target = ["x86_64-unknown-linux-gnu", "x86_64-unknown-thingos"]
local-rebuild = {local_rebuild}
docs = false
compiler-docs = false

[rust]
optimize = true
debug-assertions = false
codegen-units = 1
lto = "off"
deny-warnings = false
rpath = false

[target.x86_64-unknown-thingos]
sanitizers = false
profiler = false

[llvm]
download-ci-llvm = true
"#
    )
}

fn seed_stage0_thingos_sysroot(cwd: &Path) -> Result<()> {
    let build_root = xpy_build_root(cwd);
    let src = build_root.join("stage0-std/x86_64-unknown-thingos/release/deps");
    let dst = build_root.join("stage0-sysroot/lib/rustlib/x86_64-unknown-thingos/lib");

    if !src.exists() {
        return Ok(());
    }

    std::fs::create_dir_all(&dst)?;
    for entry in std::fs::read_dir(&src)? {
        let entry = entry?;
        let path = entry.path();
        let keep = matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("rlib") | Some("rmeta")
        );
        if keep {
            let file_name = path.file_name().context("missing stage0 rustlib file name")?;
            std::fs::copy(&path, dst.join(file_name))?;
        }
    }

    Ok(())
}

fn backup_root_config(sh: &Shell, cwd: &Path) -> Result<bool> {
    let config = cwd.join(ROOT_CONFIG_TOML);
    let backup = cwd.join(ROOT_CONFIG_BACKUP);
    sh.create_dir(backup.parent().context("missing backup parent")?)?;

    if config.exists() {
        std::fs::copy(&config, &backup)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

fn restore_root_config(cwd: &Path, had_backup: bool) -> Result<()> {
    let config = cwd.join(ROOT_CONFIG_TOML);
    let backup = cwd.join(ROOT_CONFIG_BACKUP);

    if had_backup {
        std::fs::copy(&backup, &config)?;
        std::fs::remove_file(&backup)?;
    } else if config.exists() {
        std::fs::remove_file(&config)?;
    }

    Ok(())
}

fn try_build_thingos_native_rustc(sh: &Shell, cwd: &Path) -> Result<bool> {
    let target_dir = cwd.join("targets");
    let target_dir_str = target_dir.to_str().context("non-utf8 target dir")?;

    std::fs::write(cwd.join(ROOT_CONFIG_TOML), bootstrap_config(true, true))?;
    seed_stage0_thingos_sysroot(cwd)?;

    let native_build = cmd!(sh, "python3 x.py build --stage 1 compiler/rustc")
        .env("RUST_TARGET_PATH", target_dir_str)
        .run();

    if let Err(err) = native_build {
        println!(
            "rustc-thingos: warning: native rustc build failed; continuing without ISO rustc: {err}"
        );
        std::fs::write(cwd.join(ROOT_CONFIG_TOML), bootstrap_config(false, true))?;
        return Ok(false);
    }

    match locate_thingos_rustc(cwd) {
        Some(thingos_binary) => {
            cache_thingos_rustc_tree(sh, cwd, &thingos_binary)?;
            println!(
                "rustc-thingos: ThingOS-native binary cached at {}",
                THINGOS_RUSTC_BINARY
            );
            std::fs::write(cwd.join(ROOT_CONFIG_TOML), bootstrap_config(false, true))?;
            Ok(true)
        }
        None => {
            println!(
                "rustc-thingos: warning: native rustc build finished but artifact was not found; ISO will not include a native compiler."
            );
            std::fs::write(cwd.join(ROOT_CONFIG_TOML), bootstrap_config(false, true))?;
            Ok(false)
        }
    }
}

pub fn build_rustc_thingos(sh: &Shell, arch: &str) -> Result<Option<PathBuf>> {
    if arch != "x86_64" {
        return Ok(None);
    }

    if std::env::var("SKIP_RUSTC_THINGOS").as_deref() == Ok("1") {
        return Ok(None);
    }

    let cwd = std::env::current_dir()?;

    if is_cache_valid() {
        println!("rustc-thingos: cache hit, reusing {}", RUSTC_BINARY);
        if !Path::new(THINGOS_RUSTC_BINARY).exists() && should_attempt_native_recovery() {
            println!(
                "rustc-thingos: ThingOS-native rustc missing from cache; attempting a recovery build..."
            );
            let had_backup = backup_root_config(sh, &cwd)?;
            let recovery = try_build_thingos_native_rustc(sh, &cwd);
            restore_root_config(&cwd, had_backup)?;
            recovery?;
        }
        return Ok(Some(PathBuf::from(RUSTC_BINARY)));
    }

    if !cwd.join(".git").exists() {
        bail!("repository root .git directory not found");
    }

    println!("rustc-thingos: building stage-1 rustc (Linux-hosted cross-compiler) ...");
    std::fs::create_dir_all(OUTPUT_DIR)?;

    let had_backup = backup_root_config(sh, &cwd)?;
    let build_result = (|| -> Result<Option<PathBuf>> {
        std::fs::write(cwd.join(ROOT_CONFIG_TOML), bootstrap_config(false, true))?;

        let target_dir = cwd.join("targets");
        let target_dir_str = target_dir.to_str().context("non-utf8 target dir")?;

        cmd!(sh, "python3 x.py build --stage 1 library compiler/rustc")
            .env("RUST_TARGET_PATH", target_dir_str)
            .run()
            .context("x.py stage1 build failed")?;

        let stage1_rustc = locate_stage1_rustc(&cwd).ok_or_else(|| {
            anyhow::anyhow!(
                "stage-1 rustc binary not found after bootstrap under {:?}",
                xpy_build_root(&cwd)
            )
        })?;

        sh.copy_file(&stage1_rustc, RUSTC_BINARY)?;
        cache_rustlib_tree(sh, &cwd)?;
        write_rustc_wrapper(&cwd)?;

        if should_attempt_native_recovery() {
            let _ = try_build_thingos_native_rustc(sh, &cwd)?;
        } else {
            println!(
                "rustc-thingos: ThingOS-native rustc recovery disabled; set BUILD_THINGOS_NATIVE_RUSTC=1 to attempt building {}",
                THINGOS_RUSTC_BINARY
            );
        }

        write_cache_key()?;
        println!(
            "rustc-thingos: Linux-hosted binary cached at {}",
            RUSTC_BINARY
        );
        Ok(Some(PathBuf::from(RUSTC_BINARY)))
    })();

    restore_root_config(&cwd, had_backup)?;
    build_result
}

#[allow(dead_code)]
pub fn stage_rustc_for_iso(sh: &Shell, iso_root: &Path) -> Result<()> {
    if std::env::var("SKIP_RUSTC_THINGOS").as_deref() == Ok("1") {
        return Ok(());
    }

    let thingos_rustc = Path::new(THINGOS_RUSTC_BINARY);
    if !thingos_rustc.exists() {
        println!(
            "rustc-thingos: ThingOS-native rustc not in cache ({}); skipping ISO staging.",
            THINGOS_RUSTC_BINARY
        );
        return Ok(());
    }

    let iso_bin = iso_root.join("bin");
    sh.create_dir(&iso_bin)?;
    let iso_rustc = iso_bin.join("rustc");
    sh.copy_file(thingos_rustc, &iso_rustc)?;

    #[cfg(unix)]
    {
        let mut perms = std::fs::metadata(&iso_rustc)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&iso_rustc, perms)?;
    }

    let thingos_rustlib_src = Path::new(THINGOS_RUSTLIB_CACHE_DIR);
    if thingos_rustlib_src.exists() {
        let iso_rustlib = iso_root.join("lib/rustlib");
        sh.create_dir(iso_rustlib.parent().context("missing ISO lib parent")?)?;
        sh.create_dir(&iso_rustlib)?;
        let src = thingos_rustlib_src.to_str().context("non-utf8 ThingOS rustlib cache path")?;
        let dst = iso_rustlib.to_str().context("non-utf8 ISO rustlib path")?;
        cmd!(sh, "cp -r {src}/. {dst}").run()?;
    } else {
        println!(
            "rustc-thingos: warning: ThingOS sysroot cache not found at {}; staged rustc will lack prebuilt rlibs",
            THINGOS_RUSTLIB_CACHE_DIR
        );
    }

    Ok(())
}
