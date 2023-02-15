use std::path::Path;

use super::build_sysroot;
use super::path::Dirs;
use super::prepare::GitRepo;
use super::utils::{spawn_and_wait, CargoProject, Compiler};
use super::SysrootKind;

static ABI_CAFE_REPO: GitRepo =
    GitRepo::github("Gankra", "abi-cafe", "4c6dc8c9c687e2b3a760ff2176ce236872b37212", "abi-cafe");

static ABI_CAFE: CargoProject = CargoProject::new(&ABI_CAFE_REPO.source_dir(), "abi_cafe");

pub(crate) fn run(
    channel: &str,
    sysroot_kind: SysrootKind,
    dirs: &Dirs,
    cg_clif_dylib: &Path,
    bootstrap_host_compiler: &Compiler,
) {
    ABI_CAFE_REPO.fetch(dirs);
    spawn_and_wait(ABI_CAFE.fetch("cargo", &bootstrap_host_compiler.rustc, dirs));

    eprintln!("Building sysroot for abi-cafe");
    build_sysroot::build_sysroot(
        dirs,
        channel,
        sysroot_kind,
        cg_clif_dylib,
        bootstrap_host_compiler,
        bootstrap_host_compiler.triple.clone(),
    );

    eprintln!("Running abi-cafe");

    let pairs = ["rustc_calls_cgclif", "cgclif_calls_rustc", "cgclif_calls_cc", "cc_calls_cgclif"];

    let mut cmd = ABI_CAFE.run(bootstrap_host_compiler, dirs);
    cmd.arg("--");
    cmd.arg("--pairs");
    cmd.args(pairs);
    cmd.arg("--add-rustc-codegen-backend");
    cmd.arg(format!("cgclif:{}", cg_clif_dylib.display()));
    cmd.current_dir(ABI_CAFE.source_dir(dirs));

    spawn_and_wait(cmd);
}
