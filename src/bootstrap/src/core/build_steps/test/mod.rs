//! Build-and-run steps for `./x.py test` test fixtures
//!
//! `./x.py test` (aka [`Kind::Test`]) is currently allowed to reach build steps in other modules.
//! However, this contains ~all test parts we expect people to be able to build and run locally.

// FIXME(jieyouxu): keeping unused imports here before all test steps are properly split out.
#![expect(unused_imports)]

use std::collections::HashSet;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::{env, fs, iter};

use clap_complete::shells;

use self::test_helpers::{RemoteCopyLibs, prepare_cargo_test, run_cargo_test};
use crate::core::build_steps::compile::run_cargo;
use crate::core::build_steps::doc::DocumentationFormat;
use crate::core::build_steps::llvm::get_llvm_version;
use crate::core::build_steps::synthetic_targets::MirOptPanicAbortSyntheticTarget;
use crate::core::build_steps::tool::{self, SourceType, Tool};
use crate::core::build_steps::toolstate::ToolState;
use crate::core::build_steps::{compile, dist, llvm};
use crate::core::builder::{
    self, Alias, Builder, Compiler, Kind, RunConfig, ShouldRun, Step, crate_description,
};
use crate::core::config::TargetSelection;
use crate::core::config::flags::{Subcommand, get_completion};
use crate::utils::build_stamp::{self, BuildStamp};
use crate::utils::exec::{BootstrapCommand, command};
use crate::utils::helpers::{
    self, LldThreads, add_rustdoc_cargo_linker_args, dylib_path, dylib_path_var, linker_args,
    linker_flags, t, target_supports_cranelift_backend, up_to_date,
};
use crate::utils::render_tests::{add_flags_and_try_run_tests, try_run_tests};
use crate::{CLang, DocTests, GitRepo, Mode, PathSet, envify};

mod book_tests;
mod bootstrap_self_tests;
mod cargotest;
mod compiler_crate_tests;
mod compiletest_self_tests;
mod compiletest_suites;
mod devtool_tests;
mod miri_tests;
mod rustdoc_tests;
mod std_tests;
mod test_helpers;
mod tidy;

pub(crate) use book_tests::{
    EditionGuide, EmbeddedBook, ErrorIndex, Nomicon, Reference, RustByExample, RustcBook,
    RustdocBook, TheBook, UnstableBook,
};
pub(crate) use bootstrap_self_tests::{Bootstrap, CrateBootstrap, CrateBuildHelper};
pub(crate) use cargotest::Cargotest;
pub(crate) use compiler_crate_tests::{CodegenCranelift, CodegenGCC, CrateLibrustc};
pub(crate) use compiletest_self_tests::CompiletestTest;
pub(crate) use compiletest_suites::{
    Assembly, Codegen, CodegenUnits, Coverage, CoverageRunRustdoc, Crashes, CrateRunMakeSupport,
    Debuginfo, Incremental, MirOpt, Pretty, RunMake, Rustdoc, RustdocGUI, RustdocJSNotStd,
    RustdocJSStd, RustdocJson, RustdocUi, Ui, UiFullDeps,
};
pub(crate) use devtool_tests::{Cargo, Clippy, RustAnalyzer, Rustfmt};
pub(crate) use doc_tool_tests::{HtmlCheck, Linkcheck};
pub(crate) use miri_tests::{CargoMiri, Miri};
pub(crate) use rustdoc_tests::{CrateRustdoc, CrateRustdocJsonTypes, RustdocTheme};
pub(crate) use std_tests::TestFloatParse;
pub(crate) use test_helpers::Crate;
pub(crate) use tidy::Tidy;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Linkcheck {
    host: TargetSelection,
}

impl Step for Linkcheck {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    /// Runs the `linkchecker` tool as compiled in `stage` by the `host` compiler.
    ///
    /// This tool in `src/tools` will verify the validity of all our links in the
    /// documentation to ensure we don't have a bunch of dead ones.
    fn run(self, builder: &Builder<'_>) {
        let host = self.host;
        let hosts = &builder.hosts;
        let targets = &builder.targets;

        // if we have different hosts and targets, some things may be built for
        // the host (e.g. rustc) and others for the target (e.g. std). The
        // documentation built for each will contain broken links to
        // docs built for the other platform (e.g. rustc linking to cargo)
        if (hosts != targets) && !hosts.is_empty() && !targets.is_empty() {
            panic!(
                "Linkcheck currently does not support builds with different hosts and targets.
You can skip linkcheck with --skip src/tools/linkchecker"
            );
        }

        builder.info(&format!("Linkcheck ({host})"));

        // Test the linkchecker itself.
        let bootstrap_host = builder.config.build;
        let compiler = builder.compiler(0, bootstrap_host);

        let cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolBootstrap,
            bootstrap_host,
            Kind::Test,
            "src/tools/linkchecker",
            SourceType::InTree,
            &[],
        );
        run_cargo_test(
            cargo,
            &[],
            &[],
            "linkchecker",
            "linkchecker self tests",
            bootstrap_host,
            builder,
        );

        if builder.doc_tests == DocTests::No {
            return;
        }

        // Build all the default documentation.
        builder.default_doc(&[]);

        // Build the linkchecker before calling `msg`, since GHA doesn't support nested groups.
        let linkchecker = builder.tool_cmd(Tool::Linkchecker);

        // Run the linkchecker.
        let _guard =
            builder.msg(Kind::Test, compiler.stage, "Linkcheck", bootstrap_host, bootstrap_host);
        let _time = helpers::timeit(builder);
        linkchecker.delay_failure().arg(builder.out.join(host).join("doc")).run(builder);
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        let run = run.path("src/tools/linkchecker");
        run.default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Linkcheck { host: run.target });
    }
}

fn check_if_tidy_is_installed(builder: &Builder<'_>) -> bool {
    command("tidy").allow_failure().arg("--version").run_capture_stdout(builder).is_success()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HtmlCheck {
    target: TargetSelection,
}

impl Step for HtmlCheck {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        let run = run.path("src/tools/html-checker");
        run.lazy_default_condition(Box::new(|| check_if_tidy_is_installed(builder)))
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(HtmlCheck { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        if !check_if_tidy_is_installed(builder) {
            eprintln!("not running HTML-check tool because `tidy` is missing");
            eprintln!(
                "You need the HTML tidy tool https://www.html-tidy.org/, this tool is *not* part of the rust project and needs to be installed separately, for example via your package manager."
            );
            panic!("Cannot run html-check tests");
        }
        // Ensure that a few different kinds of documentation are available.
        builder.default_doc(&[]);
        builder.ensure(crate::core::build_steps::doc::Rustc::new(
            builder.top_stage,
            self.target,
            builder,
        ));

        builder
            .tool_cmd(Tool::HtmlChecker)
            .delay_failure()
            .arg(builder.doc_out(self.target))
            .run(builder);
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RustdocTheme {
    pub compiler: Compiler,
}

impl Step for RustdocTheme {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/rustdoc-themes")
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler = run.builder.compiler(run.builder.top_stage, run.target);

        run.builder.ensure(RustdocTheme { compiler });
    }

    fn run(self, builder: &Builder<'_>) {
        let rustdoc = builder.bootstrap_out.join("rustdoc");
        let mut cmd = builder.tool_cmd(Tool::RustdocTheme);
        cmd.arg(rustdoc.to_str().unwrap())
            .arg(builder.src.join("src/librustdoc/html/static/css/rustdoc.css").to_str().unwrap())
            .env("RUSTC_STAGE", self.compiler.stage.to_string())
            .env("RUSTC_SYSROOT", builder.sysroot(self.compiler))
            .env("RUSTDOC_LIBDIR", builder.sysroot_target_libdir(self.compiler, self.compiler.host))
            .env("CFG_RELEASE_CHANNEL", &builder.config.channel)
            .env("RUSTDOC_REAL", builder.rustdoc(self.compiler))
            .env("RUSTC_BOOTSTRAP", "1");
        cmd.args(linker_args(builder, self.compiler.host, LldThreads::No));

        cmd.delay_failure().run(builder);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateBuildHelper {
    host: TargetSelection,
}

impl Step for CrateBuildHelper {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/build_helper")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(CrateBuildHelper { host: run.target });
    }

    /// Runs `cargo test` for build_helper.
    fn run(self, builder: &Builder<'_>) {
        let host = self.host;
        let compiler = builder.compiler(0, host);

        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolBootstrap,
            host,
            Kind::Test,
            "src/build_helper",
            SourceType::InTree,
            &[],
        );
        cargo.allow_features("test");
        run_cargo_test(cargo, &[], &[], "build_helper", "build_helper self test", host, builder);
    }
}

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
pub struct TierCheck {
    pub compiler: Compiler,
}

impl Step for TierCheck {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/tier-check")
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler =
            run.builder.compiler_for(run.builder.top_stage, run.builder.build.build, run.target);
        run.builder.ensure(TierCheck { compiler });
    }

    /// Tests the Platform Support page in the rustc book.
    fn run(self, builder: &Builder<'_>) {
        builder.ensure(compile::Std::new(self.compiler, self.compiler.host));
        let mut cargo = tool::prepare_tool_cargo(
            builder,
            self.compiler,
            Mode::ToolStd,
            self.compiler.host,
            Kind::Run,
            "src/tools/tier-check",
            SourceType::InTree,
            &[],
        );
        cargo.arg(builder.src.join("src/doc/rustc/src/platform-support.md"));
        cargo.arg(builder.rustc(self.compiler));
        if builder.is_verbose() {
            cargo.arg("--verbose");
        }

        let _guard = builder.msg(
            Kind::Test,
            self.compiler.stage,
            "platform support check",
            self.compiler.host,
            self.compiler.host,
        );
        BootstrapCommand::from(cargo).delay_failure().run(builder);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LintDocs {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for LintDocs {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/lint-docs")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(LintDocs {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
            target: run.target,
        });
    }

    /// Tests that the lint examples in the rustc book generate the correct
    /// lints and have the expected format.
    fn run(self, builder: &Builder<'_>) {
        builder.ensure(crate::core::build_steps::doc::RustcBook {
            compiler: self.compiler,
            target: self.target,
            validate: true,
        });
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

/// Runs the tool `src/tools/collect-license-metadata` in `ONLY_CHECK=1` mode,
/// which verifies that `license-metadata.json` is up-to-date and therefore
/// running the tool normally would not update anything.
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
