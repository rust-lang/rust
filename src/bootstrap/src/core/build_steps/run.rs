//! Build-and-run steps for in-repo tools
//!
//! A bit of a hodge-podge as e.g. if a tool's a test fixture it should be in `build_steps::test`.
//! If it can be reached from `./x.py run` it can go here.

use std::path::PathBuf;

use crate::Mode;
use crate::core::build_steps::dist::distdir;
use crate::core::build_steps::test;
use crate::core::build_steps::tool::{self, SourceType, Tool};
use crate::core::builder::{Builder, Kind, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::core::config::flags::get_completion;
use crate::utils::exec::command;

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct BuildManifest;

impl Step for BuildManifest {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/build-manifest")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(BuildManifest);
    }

    fn run(self, builder: &Builder<'_>) {
        // This gets called by `promote-release`
        // (https://github.com/rust-lang/promote-release).
        let mut cmd = builder.tool_cmd(Tool::BuildManifest);
        let sign = builder.config.dist_sign_folder.as_ref().unwrap_or_else(|| {
            panic!("\n\nfailed to specify `dist.sign-folder` in `config.toml`\n\n")
        });
        let addr = builder.config.dist_upload_addr.as_ref().unwrap_or_else(|| {
            panic!("\n\nfailed to specify `dist.upload-addr` in `config.toml`\n\n")
        });

        let today = command("date").arg("+%Y-%m-%d").run_capture_stdout(builder).stdout();

        cmd.arg(sign);
        cmd.arg(distdir(builder));
        cmd.arg(today.trim());
        cmd.arg(addr);
        cmd.arg(&builder.config.channel);

        builder.create_dir(&distdir(builder));
        cmd.run(builder);
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct BumpStage0;

impl Step for BumpStage0 {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/bump-stage0")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(BumpStage0);
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let mut cmd = builder.tool_cmd(Tool::BumpStage0);
        cmd.args(builder.config.args());
        cmd.run(builder);
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct ReplaceVersionPlaceholder;

impl Step for ReplaceVersionPlaceholder {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/replace-version-placeholder")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(ReplaceVersionPlaceholder);
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let mut cmd = builder.tool_cmd(Tool::ReplaceVersionPlaceholder);
        cmd.arg(&builder.src);
        cmd.run(builder);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Miri {
    target: TargetSelection,
}

impl Step for Miri {
    type Output = ();
    const ONLY_HOSTS: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/miri")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Miri { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let host = builder.build.build;
        let target = self.target;
        let stage = builder.top_stage;
        if stage == 0 {
            eprintln!("miri cannot be run at stage 0");
            std::process::exit(1);
        }

        // This compiler runs on the host, we'll just use it for the target.
        let target_compiler = builder.compiler(stage, host);
        // Similar to `compile::Assemble`, build with the previous stage's compiler. Otherwise
        // we'd have stageN/bin/rustc and stageN/bin/rustdoc be effectively different stage
        // compilers, which isn't what we want. Rustdoc should be linked in the same way as the
        // rustc compiler it's paired with, so it must be built with the previous stage compiler.
        let host_compiler = builder.compiler(stage - 1, host);

        // Get a target sysroot for Miri.
        let miri_sysroot = test::Miri::build_miri_sysroot(builder, target_compiler, target);

        // # Run miri.
        // Running it via `cargo run` as that figures out the right dylib path.
        // add_rustc_lib_path does not add the path that contains librustc_driver-<...>.so.
        let mut miri = tool::prepare_tool_cargo(
            builder,
            host_compiler,
            Mode::ToolRustc,
            host,
            Kind::Run,
            "src/tools/miri",
            SourceType::InTree,
            &[],
        );
        miri.add_rustc_lib_path(builder);
        miri.arg("--").arg("--target").arg(target.rustc_target_arg());

        // miri tests need to know about the stage sysroot
        miri.arg("--sysroot").arg(miri_sysroot);

        // Forward arguments. This may contain further arguments to the program
        // after another --, so this must be at the end.
        miri.args(builder.config.args());

        miri.into_cmd().run(builder);
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct CollectLicenseMetadata;

impl Step for CollectLicenseMetadata {
    type Output = PathBuf;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/collect-license-metadata")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(CollectLicenseMetadata);
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let Some(reuse) = &builder.config.reuse else {
            panic!("REUSE is required to collect the license metadata");
        };

        let dest = builder.src.join("license-metadata.json");

        let mut cmd = builder.tool_cmd(Tool::CollectLicenseMetadata);
        cmd.env("REUSE_EXE", reuse);
        cmd.env("DEST", &dest);
        cmd.run(builder);

        dest
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct GenerateCopyright;

impl Step for GenerateCopyright {
    type Output = PathBuf;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/generate-copyright")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(GenerateCopyright);
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let license_metadata = builder.src.join("license-metadata.json");
        let dest = builder.out.join("COPYRIGHT.html");
        let dest_libstd = builder.out.join("COPYRIGHT-library.html");

        let mut cmd = builder.tool_cmd(Tool::GenerateCopyright);
        cmd.env("LICENSE_METADATA", &license_metadata);
        cmd.env("DEST", &dest);
        cmd.env("DEST_LIBSTD", &dest_libstd);
        cmd.env("OUT_DIR", &builder.out);
        cmd.env("CARGO", &builder.initial_cargo);
        cmd.run(builder);

        dest
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct GenerateWindowsSys;

impl Step for GenerateWindowsSys {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/generate-windows-sys")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(GenerateWindowsSys);
    }

    fn run(self, builder: &Builder<'_>) {
        let mut cmd = builder.tool_cmd(Tool::GenerateWindowsSys);
        cmd.arg(&builder.src);
        cmd.run(builder);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenerateCompletions;

macro_rules! generate_completions {
    ( $( ( $shell:ident, $filename:expr ) ),* ) => {
        $(
            if let Some(comp) = get_completion($shell, &$filename) {
                std::fs::write(&$filename, comp).expect(&format!("writing {} completion", stringify!($shell)));
            }
        )*
    };
}

impl Step for GenerateCompletions {
    type Output = ();

    /// Uses `clap_complete` to generate shell completions.
    fn run(self, builder: &Builder<'_>) {
        use clap_complete::shells::{Bash, Fish, PowerShell, Zsh};

        generate_completions!(
            (Bash, builder.src.join("src/etc/completions/x.py.sh")),
            (Zsh, builder.src.join("src/etc/completions/x.py.zsh")),
            (Fish, builder.src.join("src/etc/completions/x.py.fish")),
            (PowerShell, builder.src.join("src/etc/completions/x.py.ps1"))
        );
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("generate-completions")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(GenerateCompletions);
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct UnicodeTableGenerator;

impl Step for UnicodeTableGenerator {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/unicode-table-generator")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(UnicodeTableGenerator);
    }

    fn run(self, builder: &Builder<'_>) {
        let mut cmd = builder.tool_cmd(Tool::UnicodeTableGenerator);
        cmd.arg(builder.src.join("library/core/src/unicode/unicode_data.rs"));
        cmd.run(builder);
    }
}
