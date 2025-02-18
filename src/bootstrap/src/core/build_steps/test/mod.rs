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
mod compiletest_self_tests;
mod compiletest_suites;
mod devtool_tests;
mod miri_tests;
mod rustdoc_tests;
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

/// Builds cargo and then runs the `src/tools/cargotest` tool, which checks out
/// some representative crate repositories and runs `cargo test` on them, in
/// order to test cargo.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cargotest {
    stage: u32,
    host: TargetSelection,
}

impl Step for Cargotest {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/cargotest")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Cargotest { stage: run.builder.top_stage, host: run.target });
    }

    /// Runs the `cargotest` tool as compiled in `stage` by the `host` compiler.
    ///
    /// This tool in `src/tools` will check out a few Rust projects and run `cargo
    /// test` to ensure that we don't regress the test suites there.
    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(self.stage, self.host);
        builder.ensure(compile::Rustc::new(compiler, compiler.host));
        let cargo = builder.ensure(tool::Cargo { compiler, target: compiler.host });

        // Note that this is a short, cryptic, and not scoped directory name. This
        // is currently to minimize the length of path on Windows where we otherwise
        // quickly run into path name limit constraints.
        let out_dir = builder.out.join("ct");
        t!(fs::create_dir_all(&out_dir));

        let _time = helpers::timeit(builder);
        let mut cmd = builder.tool_cmd(Tool::CargoTest);
        cmd.arg(&cargo)
            .arg(&out_dir)
            .args(builder.config.test_args())
            .env("RUSTC", builder.rustc(compiler))
            .env("RUSTDOC", builder.rustdoc(compiler));
        add_rustdoc_cargo_linker_args(&mut cmd, builder, compiler.host, LldThreads::No);
        cmd.delay_failure().run(builder);
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

/// Runs `cargo test` for the compiler crates in `compiler/`.
///
/// (This step does not test `rustc_codegen_cranelift` or `rustc_codegen_gcc`,
/// which have their own separate test steps.)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateLibrustc {
    compiler: Compiler,
    target: TargetSelection,
    crates: Vec<String>,
}

impl Step for CrateLibrustc {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.crate_or_deps("rustc-main").path("compiler")
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let host = run.build_triple();
        let compiler = builder.compiler_for(builder.top_stage, host, host);
        let crates = run.make_run_crates(Alias::Compiler);

        builder.ensure(CrateLibrustc { compiler, target: run.target, crates });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.ensure(compile::Std::new(self.compiler, self.target));

        // To actually run the tests, delegate to a copy of the `Crate` step.
        builder.ensure(Crate {
            compiler: self.compiler,
            target: self.target,
            mode: Mode::Rustc,
            crates: self.crates,
        });
    }
}

/// Runs `cargo test` for standard library crates.
///
/// (Also used internally to run `cargo test` for compiler crates.)
///
/// FIXME(Zalathar): Try to split this into two separate steps: a user-visible
/// step for testing standard library crates, and an internal step used for both
/// library crates and compiler crates.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Crate {
    pub compiler: Compiler,
    pub target: TargetSelection,
    pub mode: Mode,
    pub crates: Vec<String>,
}

impl Step for Crate {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.crate_or_deps("sysroot").crate_or_deps("coretests")
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let host = run.build_triple();
        let compiler = builder.compiler_for(builder.top_stage, host, host);
        let crates = run
            .paths
            .iter()
            .map(|p| builder.crate_paths[&p.assert_single_path().path].clone())
            .collect();

        builder.ensure(Crate { compiler, target: run.target, mode: Mode::Std, crates });
    }

    /// Runs all unit tests plus documentation tests for a given crate defined
    /// by a `Cargo.toml` (single manifest)
    ///
    /// This is what runs tests for crates like the standard library, compiler, etc.
    /// It essentially is the driver for running `cargo test`.
    ///
    /// Currently this runs all tests for a DAG by passing a bunch of `-p foo`
    /// arguments, and those arguments are discovered from `cargo metadata`.
    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target = self.target;
        let mode = self.mode;

        // Prepare sysroot
        // See [field@compile::Std::force_recompile].
        builder.ensure(compile::Std::new(compiler, compiler.host).force_recompile(true));

        // If we're not doing a full bootstrap but we're testing a stage2
        // version of libstd, then what we're actually testing is the libstd
        // produced in stage1. Reflect that here by updating the compiler that
        // we're working with automatically.
        let compiler = builder.compiler_for(compiler.stage, compiler.host, target);

        let mut cargo = if builder.kind == Kind::Miri {
            if builder.top_stage == 0 {
                eprintln!("ERROR: `x.py miri` requires stage 1 or higher");
                std::process::exit(1);
            }

            // Build `cargo miri test` command
            // (Implicitly prepares target sysroot)
            let mut cargo = builder::Cargo::new(
                builder,
                compiler,
                mode,
                SourceType::InTree,
                target,
                Kind::MiriTest,
            );
            // This hack helps bootstrap run standard library tests in Miri. The issue is as
            // follows: when running `cargo miri test` on libcore, cargo builds a local copy of core
            // and makes it a dependency of the integration test crate. This copy duplicates all the
            // lang items, so the build fails. (Regular testing avoids this because the sysroot is a
            // literal copy of what `cargo build` produces, but since Miri builds its own sysroot
            // this does not work for us.) So we need to make it so that the locally built libcore
            // contains all the items from `core`, but does not re-define them -- we want to replace
            // the entire crate but a re-export of the sysroot crate. We do this by swapping out the
            // source file: if `MIRI_REPLACE_LIBRS_IF_NOT_TEST` is set and we are building a
            // `lib.rs` file, and a `lib.miri.rs` file exists in the same folder, we build that
            // instead. But crucially we only do that for the library, not the test builds.
            cargo.env("MIRI_REPLACE_LIBRS_IF_NOT_TEST", "1");
            // std needs to be built with `-Zforce-unstable-if-unmarked`. For some reason the builder
            // does not set this directly, but relies on the rustc wrapper to set it, and we are not using
            // the wrapper -- hence we have to set it ourselves.
            cargo.rustflag("-Zforce-unstable-if-unmarked");
            cargo
        } else {
            // Also prepare a sysroot for the target.
            if !builder.is_builder_target(target) {
                builder.ensure(compile::Std::new(compiler, target).force_recompile(true));
                builder.ensure(RemoteCopyLibs { compiler, target });
            }

            // Build `cargo test` command
            builder::Cargo::new(builder, compiler, mode, SourceType::InTree, target, builder.kind)
        };

        match mode {
            Mode::Std => {
                if builder.kind == Kind::Miri {
                    // We can't use `std_cargo` as that uses `optimized-compiler-builtins` which
                    // needs host tools for the given target. This is similar to what `compile::Std`
                    // does when `is_for_mir_opt_tests` is true. There's probably a chance for
                    // de-duplication here... `std_cargo` should support a mode that avoids needing
                    // host tools.
                    cargo
                        .arg("--manifest-path")
                        .arg(builder.src.join("library/sysroot/Cargo.toml"));
                } else {
                    compile::std_cargo(builder, target, compiler.stage, &mut cargo);
                    // `std_cargo` actually does the wrong thing: it passes `--sysroot build/host/stage2`,
                    // but we want to use the force-recompile std we just built in `build/host/stage2-test-sysroot`.
                    // Override it.
                    if builder.download_rustc() && compiler.stage > 0 {
                        let sysroot = builder
                            .out
                            .join(compiler.host)
                            .join(format!("stage{}-test-sysroot", compiler.stage));
                        cargo.env("RUSTC_SYSROOT", sysroot);
                    }
                }
            }
            Mode::Rustc => {
                compile::rustc_cargo(builder, &mut cargo, target, &compiler, &self.crates);
            }
            _ => panic!("can only test libraries"),
        };

        run_cargo_test(
            cargo,
            &[],
            &self.crates,
            &self.crates[0],
            &*crate_description(&self.crates),
            target,
            builder,
        );
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CodegenCranelift {
    compiler: Compiler,
    target: TargetSelection,
}

impl Step for CodegenCranelift {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.paths(&["compiler/rustc_codegen_cranelift"])
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let host = run.build_triple();
        let compiler = run.builder.compiler_for(run.builder.top_stage, host, host);

        if builder.doc_tests == DocTests::Only {
            return;
        }

        if builder.download_rustc() {
            builder.info("CI rustc uses the default codegen backend. skipping");
            return;
        }

        if !target_supports_cranelift_backend(run.target) {
            builder.info("target not supported by rustc_codegen_cranelift. skipping");
            return;
        }

        if builder.remote_tested(run.target) {
            builder.info("remote testing is not supported by rustc_codegen_cranelift. skipping");
            return;
        }

        if !builder.config.codegen_backends(run.target).contains(&"cranelift".to_owned()) {
            builder.info("cranelift not in rust.codegen-backends. skipping");
            return;
        }

        builder.ensure(CodegenCranelift { compiler, target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target = self.target;

        builder.ensure(compile::Std::new(compiler, target));

        // If we're not doing a full bootstrap but we're testing a stage2
        // version of libstd, then what we're actually testing is the libstd
        // produced in stage1. Reflect that here by updating the compiler that
        // we're working with automatically.
        let compiler = builder.compiler_for(compiler.stage, compiler.host, target);

        let build_cargo = || {
            let mut cargo = builder::Cargo::new(
                builder,
                compiler,
                Mode::Codegen, // Must be codegen to ensure dlopen on compiled dylibs works
                SourceType::InTree,
                target,
                Kind::Run,
            );

            cargo.current_dir(&builder.src.join("compiler/rustc_codegen_cranelift"));
            cargo
                .arg("--manifest-path")
                .arg(builder.src.join("compiler/rustc_codegen_cranelift/build_system/Cargo.toml"));
            compile::rustc_cargo_env(builder, &mut cargo, target, compiler.stage);

            // Avoid incremental cache issues when changing rustc
            cargo.env("CARGO_BUILD_INCREMENTAL", "false");

            cargo
        };

        builder.info(&format!(
            "{} cranelift stage{} ({} -> {})",
            Kind::Test.description(),
            compiler.stage,
            &compiler.host,
            target
        ));
        let _time = helpers::timeit(builder);

        // FIXME handle vendoring for source tarballs before removing the --skip-test below
        let download_dir = builder.out.join("cg_clif_download");

        // FIXME: Uncomment the `prepare` command below once vendoring is implemented.
        /*
        let mut prepare_cargo = build_cargo();
        prepare_cargo.arg("--").arg("prepare").arg("--download-dir").arg(&download_dir);
        #[allow(deprecated)]
        builder.config.try_run(&mut prepare_cargo.into()).unwrap();
        */

        let mut cargo = build_cargo();
        cargo
            .arg("--")
            .arg("test")
            .arg("--download-dir")
            .arg(&download_dir)
            .arg("--out-dir")
            .arg(builder.stage_out(compiler, Mode::ToolRustc).join("cg_clif"))
            .arg("--no-unstable-features")
            .arg("--use-backend")
            .arg("cranelift")
            // Avoid having to vendor the standard library dependencies
            .arg("--sysroot")
            .arg("llvm")
            // These tests depend on crates that are not yet vendored
            // FIXME remove once vendoring is handled
            .arg("--skip-test")
            .arg("testsuite.extended_sysroot");

        cargo.into_cmd().run(builder);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CodegenGCC {
    compiler: Compiler,
    target: TargetSelection,
}

impl Step for CodegenGCC {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.paths(&["compiler/rustc_codegen_gcc"])
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let host = run.build_triple();
        let compiler = run.builder.compiler_for(run.builder.top_stage, host, host);

        if builder.doc_tests == DocTests::Only {
            return;
        }

        if builder.download_rustc() {
            builder.info("CI rustc uses the default codegen backend. skipping");
            return;
        }

        let triple = run.target.triple;
        let target_supported =
            if triple.contains("linux") { triple.contains("x86_64") } else { false };
        if !target_supported {
            builder.info("target not supported by rustc_codegen_gcc. skipping");
            return;
        }

        if builder.remote_tested(run.target) {
            builder.info("remote testing is not supported by rustc_codegen_gcc. skipping");
            return;
        }

        if !builder.config.codegen_backends(run.target).contains(&"gcc".to_owned()) {
            builder.info("gcc not in rust.codegen-backends. skipping");
            return;
        }

        builder.ensure(CodegenGCC { compiler, target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target = self.target;

        builder.ensure(
            compile::Std::new(compiler, target)
                .extra_rust_args(&["-Csymbol-mangling-version=v0", "-Cpanic=abort"]),
        );

        // If we're not doing a full bootstrap but we're testing a stage2
        // version of libstd, then what we're actually testing is the libstd
        // produced in stage1. Reflect that here by updating the compiler that
        // we're working with automatically.
        let compiler = builder.compiler_for(compiler.stage, compiler.host, target);

        let build_cargo = || {
            let mut cargo = builder::Cargo::new(
                builder,
                compiler,
                Mode::Codegen, // Must be codegen to ensure dlopen on compiled dylibs works
                SourceType::InTree,
                target,
                Kind::Run,
            );

            cargo.current_dir(&builder.src.join("compiler/rustc_codegen_gcc"));
            cargo
                .arg("--manifest-path")
                .arg(builder.src.join("compiler/rustc_codegen_gcc/build_system/Cargo.toml"));
            compile::rustc_cargo_env(builder, &mut cargo, target, compiler.stage);

            // Avoid incremental cache issues when changing rustc
            cargo.env("CARGO_BUILD_INCREMENTAL", "false");
            cargo.rustflag("-Cpanic=abort");

            cargo
        };

        builder.info(&format!(
            "{} GCC stage{} ({} -> {})",
            Kind::Test.description(),
            compiler.stage,
            &compiler.host,
            target
        ));
        let _time = helpers::timeit(builder);

        // FIXME: Uncomment the `prepare` command below once vendoring is implemented.
        /*
        let mut prepare_cargo = build_cargo();
        prepare_cargo.arg("--").arg("prepare");
        #[allow(deprecated)]
        builder.config.try_run(&mut prepare_cargo.into()).unwrap();
        */

        let mut cargo = build_cargo();

        cargo
            // cg_gcc's build system ignores RUSTFLAGS. pass some flags through CG_RUSTFLAGS instead.
            .env("CG_RUSTFLAGS", "-Alinker-messages")
            .arg("--")
            .arg("test")
            .arg("--use-system-gcc")
            .arg("--use-backend")
            .arg("gcc")
            .arg("--out-dir")
            .arg(builder.stage_out(compiler, Mode::ToolRustc).join("cg_gcc"))
            .arg("--release")
            .arg("--mini-tests")
            .arg("--std-tests");
        cargo.args(builder.config.test_args());

        cargo.into_cmd().run(builder);
    }
}

/// Test step that does two things:
/// - Runs `cargo test` for the `src/etc/test-float-parse` tool.
/// - Invokes the `test-float-parse` tool to test the standard library's
///   float parsing routines.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TestFloatParse {
    path: PathBuf,
    host: TargetSelection,
}

impl Step for TestFloatParse {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/etc/test-float-parse")
    }

    fn make_run(run: RunConfig<'_>) {
        for path in run.paths {
            let path = path.assert_single_path().path.clone();
            run.builder.ensure(Self { path, host: run.target });
        }
    }

    fn run(self, builder: &Builder<'_>) {
        let bootstrap_host = builder.config.build;
        let compiler = builder.compiler(builder.top_stage, bootstrap_host);
        let path = self.path.to_str().unwrap();
        let crate_name = self.path.components().last().unwrap().as_os_str().to_str().unwrap();

        builder.ensure(tool::TestFloatParse { host: self.host });

        // Run any unit tests in the crate
        let cargo_test = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolStd,
            bootstrap_host,
            Kind::Test,
            path,
            SourceType::InTree,
            &[],
        );

        run_cargo_test(cargo_test, &[], &[], crate_name, crate_name, bootstrap_host, builder);

        // Run the actual parse tests.
        let mut cargo_run = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolStd,
            bootstrap_host,
            Kind::Run,
            path,
            SourceType::InTree,
            &[],
        );

        if !matches!(env::var("FLOAT_PARSE_TESTS_NO_SKIP_HUGE").as_deref(), Ok("1") | Ok("true")) {
            cargo_run.args(["--", "--skip-huge"]);
        }

        cargo_run.into_cmd().run(builder);
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
