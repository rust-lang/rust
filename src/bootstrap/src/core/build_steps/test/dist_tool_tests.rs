//! Tool-based tests for dist workflows.

#![warn(unused_imports)]
use std::fs;
use std::path::PathBuf;

use super::test_helpers::run_cargo_test;
use crate::Mode;
use crate::core::build_steps::dist;
use crate::core::build_steps::tool::{self, SourceType, Tool};
use crate::core::builder::{Builder, Kind, RunConfig, ShouldRun, Step};
use crate::utils::exec::command;
use crate::utils::helpers::{self, t};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Distcheck;

impl Step for Distcheck {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("distcheck")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Distcheck);
    }

    /// Runs "distcheck", a 'make check' from a tarball
    fn run(self, builder: &Builder<'_>) {
        builder.info("Distcheck");
        let dir = builder.tempdir().join("distcheck");
        let _ = fs::remove_dir_all(&dir);
        t!(fs::create_dir_all(&dir));

        // Guarantee that these are built before we begin running.
        builder.ensure(dist::PlainSourceTarball);
        builder.ensure(dist::Src);

        command("tar")
            .arg("-xf")
            .arg(builder.ensure(dist::PlainSourceTarball).tarball())
            .arg("--strip-components=1")
            .current_dir(&dir)
            .run(builder);
        command("./configure")
            .args(&builder.config.configure_args)
            .arg("--enable-vendor")
            .current_dir(&dir)
            .run(builder);
        command(helpers::make(&builder.config.build.triple))
            .arg("check")
            .current_dir(&dir)
            .run(builder);

        // Now make sure that rust-src has all of libstd's dependencies
        builder.info("Distcheck rust-src");
        let dir = builder.tempdir().join("distcheck-src");
        let _ = fs::remove_dir_all(&dir);
        t!(fs::create_dir_all(&dir));

        command("tar")
            .arg("-xf")
            .arg(builder.ensure(dist::Src).tarball())
            .arg("--strip-components=1")
            .current_dir(&dir)
            .run(builder);

        let toml = dir.join("rust-src/lib/rustlib/src/rust/library/std/Cargo.toml");
        command(&builder.initial_cargo)
            // Will read the libstd Cargo.toml
            // which uses the unstable `public-dependency` feature.
            .env("RUSTC_BOOTSTRAP", "1")
            .arg("generate-lockfile")
            .arg("--manifest-path")
            .arg(&toml)
            .current_dir(&dir)
            .run(builder);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RustInstaller;

impl Step for RustInstaller {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    /// Ensure the version placeholder replacement tool builds
    fn run(self, builder: &Builder<'_>) {
        let bootstrap_host = builder.config.build;
        let compiler = builder.compiler(0, bootstrap_host);
        let cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolBootstrap,
            bootstrap_host,
            Kind::Test,
            "src/tools/rust-installer",
            SourceType::InTree,
            &[],
        );

        let _guard = builder.msg(
            Kind::Test,
            compiler.stage,
            "rust-installer",
            bootstrap_host,
            bootstrap_host,
        );
        run_cargo_test(cargo, &[], &[], "installer", None, bootstrap_host, builder);

        // We currently don't support running the test.sh script outside linux(?) environments.
        // Eventually this should likely migrate to #[test]s in rust-installer proper rather than a
        // set of scripts, which will likely allow dropping this if.
        if bootstrap_host != "x86_64-unknown-linux-gnu" {
            return;
        }

        let mut cmd = command(builder.src.join("src/tools/rust-installer/test.sh"));
        let tmpdir = builder.test_out(compiler.host).join("rust-installer");
        let _ = std::fs::remove_dir_all(&tmpdir);
        let _ = std::fs::create_dir_all(&tmpdir);
        cmd.current_dir(&tmpdir);
        cmd.env("CARGO_TARGET_DIR", tmpdir.join("cargo-target"));
        cmd.env("CARGO", &builder.initial_cargo);
        cmd.env("RUSTC", &builder.initial_rustc);
        cmd.env("TMP_DIR", &tmpdir);
        cmd.delay_failure().run(builder);
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/rust-installer")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Self);
    }
}

/// Runs the tool `src/tools/collect-license-metadata` in `ONLY_CHECK=1` mode, which verifies that
/// `license-metadata.json` is up-to-date and therefore running the tool normally would not update
/// anything.
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
        cmd.env("ONLY_CHECK", "1");
        cmd.run(builder);

        dest
    }
}
