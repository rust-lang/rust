use std::path::PathBuf;
use std::process::Command;

use crate::core::build_steps::dist::distdir;
use crate::core::build_steps::test;
use crate::core::build_steps::tool::{self, SourceType, Tool};
use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::core::config::flags::get_completion;
use crate::core::config::TargetSelection;
use crate::utils::helpers::output;
use crate::Mode;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExpandYamlAnchors;

impl Step for ExpandYamlAnchors {
    type Output = ();

    /// Runs the `expand-yaml_anchors` tool.
    ///
    /// This tool in `src/tools` reads the CI configuration files written in YAML and expands the
    /// anchors in them, since GitHub Actions doesn't support them.
    fn run(self, builder: &Builder<'_>) {
        builder.info("Expanding YAML anchors in the GitHub Actions configuration");
        builder.run_delaying_failure(
            builder.tool_cmd(Tool::ExpandYamlAnchors).arg("generate").arg(&builder.src),
        );
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/expand-yaml-anchors")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(ExpandYamlAnchors);
    }
}

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

        let today = output(Command::new("date").arg("+%Y-%m-%d"));

        cmd.arg(sign);
        cmd.arg(distdir(builder));
        cmd.arg(today.trim());
        cmd.arg(addr);
        cmd.arg(&builder.config.channel);

        builder.create_dir(&distdir(builder));
        builder.run(&mut cmd);
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
        builder.run(&mut cmd);
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
        builder.run(&mut cmd);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Miri {
    stage: u32,
    host: TargetSelection,
    target: TargetSelection,
}

impl Step for Miri {
    type Output = ();
    const ONLY_HOSTS: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/miri")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Miri {
            stage: run.builder.top_stage,
            host: run.build_triple(),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let host = self.host;
        let target = self.target;
        let compiler = builder.compiler(stage, host);

        let miri =
            builder.ensure(tool::Miri { compiler, target: self.host, extra_features: Vec::new() });
        let miri_sysroot = test::Miri::build_miri_sysroot(builder, compiler, &miri, target);

        // # Run miri.
        // Running it via `cargo run` as that figures out the right dylib path.
        // add_rustc_lib_path does not add the path that contains librustc_driver-<...>.so.
        let mut miri = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            host,
            "run",
            "src/tools/miri",
            SourceType::InTree,
            &[],
        );
        miri.add_rustc_lib_path(builder);
        // Forward arguments.
        miri.arg("--").arg("--target").arg(target.rustc_target_arg());
        miri.args(builder.config.args());

        // miri tests need to know about the stage sysroot
        miri.env("MIRI_SYSROOT", &miri_sysroot);

        let mut miri = Command::from(miri);
        builder.run(&mut miri);
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

        // Temporary location, it will be moved to src/etc once it's accurate.
        let dest = builder.out.join("license-metadata.json");

        let mut cmd = builder.tool_cmd(Tool::CollectLicenseMetadata);
        cmd.env("REUSE_EXE", reuse);
        cmd.env("DEST", &dest);
        builder.run(&mut cmd);

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
        let license_metadata = builder.ensure(CollectLicenseMetadata);

        // Temporary location, it will be moved to the proper one once it's accurate.
        let dest = builder.out.join("COPYRIGHT.md");

        let mut cmd = builder.tool_cmd(Tool::GenerateCopyright);
        cmd.env("LICENSE_METADATA", &license_metadata);
        cmd.env("DEST", &dest);
        builder.run(&mut cmd);

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
        builder.run(&mut cmd);
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
