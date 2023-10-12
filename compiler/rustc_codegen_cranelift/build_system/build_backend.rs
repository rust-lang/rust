use std::path::PathBuf;

use crate::path::{Dirs, RelPath};
use crate::rustc_info::get_file_name;
use crate::shared_utils::{rustflags_from_env, rustflags_to_cmd_env};
use crate::utils::{is_ci, is_ci_opt, maybe_incremental, CargoProject, Compiler, LogGroup};

pub(crate) static CG_CLIF: CargoProject = CargoProject::new(&RelPath::SOURCE, "cg_clif");

pub(crate) fn build_backend(
    dirs: &Dirs,
    channel: &str,
    bootstrap_host_compiler: &Compiler,
    use_unstable_features: bool,
) -> PathBuf {
    let _group = LogGroup::guard("Build backend");

    let mut cmd = CG_CLIF.build(&bootstrap_host_compiler, dirs);
    maybe_incremental(&mut cmd);

    let mut rustflags = rustflags_from_env("RUSTFLAGS");

    rustflags.push("-Zallow-features=rustc_private".to_owned());

    if is_ci() {
        // Deny warnings on CI
        rustflags.push("-Dwarnings".to_owned());

        if !is_ci_opt() {
            cmd.env("CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS", "true");
            cmd.env("CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS", "true");
        }
    }

    if use_unstable_features {
        cmd.arg("--features").arg("unstable-features");
    }

    match channel {
        "debug" => {}
        "release" => {
            cmd.arg("--release");
        }
        _ => unreachable!(),
    }

    rustflags_to_cmd_env(&mut cmd, "RUSTFLAGS", &rustflags);

    eprintln!("[BUILD] rustc_codegen_cranelift");
    crate::utils::spawn_and_wait(cmd);

    CG_CLIF
        .target_dir(dirs)
        .join(&bootstrap_host_compiler.triple)
        .join(channel)
        .join(get_file_name(&bootstrap_host_compiler.rustc, "rustc_codegen_cranelift", "dylib"))
}
