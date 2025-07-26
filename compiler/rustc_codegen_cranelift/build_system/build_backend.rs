use std::env;
use std::path::PathBuf;

use crate::path::{Dirs, RelPath};
use crate::rustc_info::get_file_name;
use crate::shared_utils::{rustflags_from_env, rustflags_to_cmd_env};
use crate::utils::{CargoProject, Compiler, LogGroup};

static CG_CLIF: CargoProject = CargoProject::new(&RelPath::source("."), "cg_clif");

pub(crate) fn build_backend(
    dirs: &Dirs,
    bootstrap_host_compiler: &Compiler,
    use_unstable_features: bool,
) -> PathBuf {
    let _group = LogGroup::guard("Build backend");

    let mut cmd = CG_CLIF.build(&bootstrap_host_compiler, dirs);

    let mut rustflags = rustflags_from_env("RUSTFLAGS");
    rustflags.push("-Zallow-features=rustc_private,f16,f128".to_owned());
    rustflags_to_cmd_env(&mut cmd, "RUSTFLAGS", &rustflags);

    if env::var("CG_CLIF_EXPENSIVE_CHECKS").is_ok() {
        // Enabling debug assertions implicitly enables the clif ir verifier
        cmd.env("CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS", "true");
        cmd.env("CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS", "true");
    }

    if use_unstable_features {
        cmd.arg("--features").arg("unstable-features");
    }

    cmd.arg("--release");

    eprintln!("[BUILD] rustc_codegen_cranelift");
    crate::utils::spawn_and_wait(cmd);

    CG_CLIF
        .target_dir(dirs)
        .join(&bootstrap_host_compiler.triple)
        .join("release")
        .join(get_file_name(&bootstrap_host_compiler.rustc, "rustc_codegen_cranelift", "dylib"))
}
