use std::env::consts::EXE_SUFFIX;
use std::fs;
use std::fs::File;
use std::path::{MAIN_SEPARATOR_STR, Path, PathBuf};

use crate::core::builder::{Builder, CommandLineStep, RunConfig, ShouldRun};
use crate::utils::exec::command;

/// Creates a toolchain link for stage1 using `rustup`
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Link;

impl CommandLineStep for Link {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("link")
    }

    fn is_default_step(_builder: &Builder<'_>) -> bool {
        true
    }

    fn make_run(run: RunConfig<'_>) {
        if run.builder.config.dry_run() {
            return;
        }
        if let [cmd] = &run.paths[..]
            && cmd.assert_single_path().path.as_path().as_os_str() == "link"
        {
            run.builder.ensure(Link);
        }
    }
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let config = &builder.config;

        if config.dry_run() {
            return;
        }

        if !rustup_installed(builder) {
            println!("WARNING: `rustup` is not installed; Skipping `stage1` toolchain linking.");
            return;
        }

        let stage_path =
            ["build", config.host_target.rustc_target_arg(), "stage1"].join(MAIN_SEPARATOR_STR);

        if stage_dir_exists(&stage_path[..]) && !config.dry_run() {
            attempt_toolchain_link(builder, &stage_path[..]);
        }
    }
}

fn rustup_installed(builder: &Builder<'_>) -> bool {
    let mut rustup = command("rustup");
    rustup.arg("--version");

    rustup.allow_failure().run_in_dry_run().run_capture_stdout(builder).is_success()
}

fn stage_dir_exists(stage_path: &str) -> bool {
    match fs::create_dir(stage_path) {
        Ok(_) => true,
        Err(_) => Path::new(&stage_path).exists(),
    }
}

fn attempt_toolchain_link(builder: &Builder<'_>, stage_path: &str) {
    if toolchain_is_linked(builder) {
        return;
    }

    if !ensure_stage1_toolchain_placeholder_exists(stage_path) {
        eprintln!(
            "Failed to create a template for stage 1 toolchain or confirm that it already exists"
        );
        return;
    }

    if try_link_toolchain(builder, stage_path) {
        println!(
            "Added `stage1` rustup toolchain; try `cargo +stage1 build` on a separate rust project to run a newly-built toolchain"
        );
    } else {
        eprintln!("`rustup` failed to link stage 1 build to `stage1` toolchain");
        eprintln!(
            "To manually link stage 1 build to `stage1` toolchain, run:\n
            `rustup toolchain link stage1 {}`",
            stage_path
        );
    }
}

fn toolchain_is_linked(builder: &Builder<'_>) -> bool {
    match command("rustup")
        .allow_failure()
        .args(["toolchain", "list"])
        .run_capture_stdout(builder)
        .stdout_if_ok()
    {
        Some(toolchain_list) => {
            if !toolchain_list.contains("stage1") {
                return false;
            }
            // The toolchain has already been linked.
            println!(
                "`stage1` toolchain already linked; not attempting to link `stage1` toolchain"
            );
        }
        None => {
            // In this case, we don't know if the `stage1` toolchain has been linked;
            // but `rustup` failed, so let's not go any further.
            println!(
                "`rustup` failed to list current toolchains; not attempting to link `stage1` toolchain"
            );
        }
    }
    true
}

fn try_link_toolchain(builder: &Builder<'_>, stage_path: &str) -> bool {
    command("rustup")
        .args(["toolchain", "link", "stage1", stage_path])
        .run_capture_stdout(builder)
        .is_success()
}

fn ensure_stage1_toolchain_placeholder_exists(stage_path: &str) -> bool {
    let pathbuf = PathBuf::from(stage_path);

    if fs::create_dir_all(pathbuf.join("lib")).is_err() {
        return false;
    };

    let pathbuf = pathbuf.join("bin");
    if fs::create_dir_all(&pathbuf).is_err() {
        return false;
    };

    let pathbuf = pathbuf.join(format!("rustc{EXE_SUFFIX}"));

    if pathbuf.exists() {
        return true;
    }

    // Take care not to overwrite the file
    let result = File::options().append(true).create(true).open(&pathbuf);
    result.is_ok()
}
