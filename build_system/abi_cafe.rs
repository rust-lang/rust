use std::env;
use std::path::Path;

use super::build_sysroot;
use super::config;
use super::prepare;
use super::utils::{cargo_command, spawn_and_wait};
use super::SysrootKind;

pub(crate) fn run(
    channel: &str,
    sysroot_kind: SysrootKind,
    target_dir: &Path,
    cg_clif_dylib: &Path,
    host_triple: &str,
    target_triple: &str,
) {
    if !config::get_bool("testsuite.abi-cafe") {
        eprintln!("[SKIP] abi-cafe");
        return;
    }

    if host_triple != target_triple {
        eprintln!("[SKIP] abi-cafe (cross-compilation not supported)");
        return;
    }

    eprintln!("Building sysroot for abi-cafe");
    build_sysroot::build_sysroot(
        channel,
        sysroot_kind,
        target_dir,
        cg_clif_dylib,
        host_triple,
        target_triple,
    );

    eprintln!("Running abi-cafe");
    let abi_cafe_path = prepare::ABI_CAFE.source_dir();
    env::set_current_dir(abi_cafe_path.clone()).unwrap();

    let pairs = ["rustc_calls_cgclif", "cgclif_calls_rustc", "cgclif_calls_cc", "cc_calls_cgclif"];

    let mut cmd = cargo_command("cargo", "run", Some(target_triple), &abi_cafe_path);
    cmd.arg("--");
    cmd.arg("--pairs");
    cmd.args(pairs);
    cmd.arg("--add-rustc-codegen-backend");
    cmd.arg(format!("cgclif:{}", cg_clif_dylib.display()));

    spawn_and_wait(cmd);
}
