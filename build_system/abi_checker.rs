use super::build_sysroot;
use super::utils::spawn_and_wait_with_input;
use build_system::SysrootKind;
use std::env;
use std::path::Path;
use std::process::Command;

pub(crate) fn run(
    channel: &str,
    sysroot_kind: SysrootKind,
    target_dir: &Path,
    cg_clif_build_dir: &Path,
    host_triple: &str,
    target_triple: &str,
) {
    assert_eq!(
        host_triple, target_triple,
        "abi-checker not supported on cross-compilation scenarios"
    );

    eprintln!("Building sysroot for abi-checker");
    build_sysroot::build_sysroot(
        channel,
        sysroot_kind,
        target_dir,
        cg_clif_build_dir,
        host_triple,
        target_triple,
    );

    eprintln!("Running abi-checker");
    let mut abi_checker_path = env::current_dir().unwrap();
    abi_checker_path.push("abi-checker");
    env::set_current_dir(abi_checker_path.clone()).unwrap();

    let build_dir = abi_checker_path.parent().unwrap().join("build");
    let cg_clif_dylib_path = build_dir.join(if cfg!(windows) { "bin" } else { "lib" }).join(
        env::consts::DLL_PREFIX.to_string() + "rustc_codegen_cranelift" + env::consts::DLL_SUFFIX,
    );

    let pairs = ["rustc_calls_cgclif", "cgclif_calls_rustc", "cgclif_calls_cc", "cc_calls_cgclif"];

    let mut cmd = Command::new("cargo");
    cmd.arg("run");
    cmd.arg("--target");
    cmd.arg(target_triple);
    cmd.arg("--");
    cmd.arg("--pairs");
    cmd.args(pairs);
    cmd.arg("--add-rustc-codegen-backend");
    cmd.arg(format!("cgclif:{}", cg_clif_dylib_path.display()));

    let output = spawn_and_wait_with_input(cmd, "".to_string());

    // TODO: The correct thing to do here is to check the exit code, but abi-checker
    // currently doesn't return 0 on success, so check for the test fail count.
    // See: https://github.com/Gankra/abi-checker/issues/10
    let failed = !(output.contains("0 failed") && output.contains("0 completely failed"));
    if failed {
        panic!("abi-checker failed!");
    }
}
