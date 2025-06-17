//! Build-and-run steps for `./x.py test` test fixtures
//!
//! `./x.py test` (aka [`Kind::Test`]) is currently allowed to reach build steps in other modules.
//! However, this contains ~all test parts we expect people to be able to build and run locally.

use std::collections::HashSet;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::{env, fs, iter};

use clap_complete::shells;

use crate::core::build_steps::compile::run_cargo;
use crate::core::build_steps::doc::DocumentationFormat;
use crate::core::build_steps::gcc::{Gcc, add_cg_gcc_cargo_flags};
use crate::core::build_steps::llvm::get_llvm_version;
use crate::core::build_steps::synthetic_targets::MirOptPanicAbortSyntheticTarget;
use crate::core::build_steps::tool::{self, COMPILETEST_ALLOW_FEATURES, SourceType, Tool};
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

const ADB_TEST_DIR: &str = "/data/local/tmp/work";

/// Runs `cargo test` on various internal tools used by bootstrap.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateBootstrap {
    path: PathBuf,
    host: TargetSelection,
}

impl Step for CrateBootstrap {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        // This step is responsible for several different tool paths. By default
        // it will test all of them, but requesting specific tools on the
        // command-line (e.g. `./x test suggest-tests`) will test only the
        // specified tools.
        run.path("src/tools/jsondoclint")
            .path("src/tools/suggest-tests")
            .path("src/tools/replace-version-placeholder")
            .path("src/tools/coverage-dump")
            // We want `./x test tidy` to _run_ the tidy tool, not its tests.
            // So we need a separate alias to test the tidy tool itself.
            .alias("tidyselftest")
    }

    fn make_run(run: RunConfig<'_>) {
        // Create and ensure a separate instance of this step for each path
        // that was selected on the command-line (or selected by default).
        for path in run.paths {
            let path = path.assert_single_path().path.clone();
            run.builder.ensure(CrateBootstrap { host: run.target, path });
        }
    }

    fn run(self, builder: &Builder<'_>) {
        let bootstrap_host = builder.config.host_target;
        let compiler = builder.compiler(0, bootstrap_host);
        let mut path = self.path.to_str().unwrap();

        // Map alias `tidyselftest` back to the actual crate path of tidy.
        if path == "tidyselftest" {
            path = "src/tools/tidy";
        }

        let cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolBootstrap,
            bootstrap_host,
            Kind::Test,
            path,
            SourceType::InTree,
            &[],
        );

        let crate_name = path.rsplit_once('/').unwrap().1;
        run_cargo_test(cargo, &[], &[], crate_name, bootstrap_host, builder);
    }
}

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
        let bootstrap_host = builder.config.host_target;
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
        run_cargo_test(cargo, &[], &[], "linkchecker self tests", bootstrap_host, builder);

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
        cmd.arg(&cargo.tool_path)
            .arg(&out_dir)
            .args(builder.config.test_args())
            .env("RUSTC", builder.rustc(compiler))
            .env("RUSTDOC", builder.rustdoc(compiler));
        add_rustdoc_cargo_linker_args(&mut cmd, builder, compiler.host, LldThreads::No);
        cmd.delay_failure().run(builder);
    }
}

/// Runs `cargo test` for cargo itself.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cargo {
    stage: u32,
    host: TargetSelection,
}

impl Cargo {
    const CRATE_PATH: &str = "src/tools/cargo";
}

impl Step for Cargo {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path(Self::CRATE_PATH)
    }

    fn make_run(run: RunConfig<'_>) {
        // If stage is explicitly set or not lower than 2, keep it. Otherwise, make sure it's at least 2
        // as tests for this step don't work with a lower stage.
        let stage = if run.builder.config.is_explicit_stage() || run.builder.top_stage >= 2 {
            run.builder.top_stage
        } else {
            2
        };

        run.builder.ensure(Cargo { stage, host: run.target });
    }

    /// Runs `cargo test` for `cargo` packaged with Rust.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;

        if stage < 2 {
            eprintln!("WARNING: cargo tests on stage {stage} may not behave well.");
            eprintln!("HELP: consider using stage 2");
        }

        let compiler = builder.compiler(stage, self.host);

        let cargo = builder.ensure(tool::Cargo { compiler, target: self.host });
        let compiler = cargo.build_compiler;

        let cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            self.host,
            Kind::Test,
            Self::CRATE_PATH,
            SourceType::Submodule,
            &[],
        );

        // NOTE: can't use `run_cargo_test` because we need to overwrite `PATH`
        let mut cargo = prepare_cargo_test(cargo, &[], &[], self.host, builder);

        // Don't run cross-compile tests, we may not have cross-compiled libstd libs
        // available.
        cargo.env("CFG_DISABLE_CROSS_TESTS", "1");
        // Forcibly disable tests using nightly features since any changes to
        // those features won't be able to land.
        cargo.env("CARGO_TEST_DISABLE_NIGHTLY", "1");
        cargo.env("PATH", path_for_cargo(builder, compiler));
        // Cargo's test suite uses `CARGO_RUSTC_CURRENT_DIR` to determine the path that `file!` is
        // relative to. Cargo no longer sets this env var, so we have to do that. This has to be the
        // same value as `-Zroot-dir`.
        cargo.env("CARGO_RUSTC_CURRENT_DIR", builder.src.display().to_string());

        #[cfg(feature = "build-metrics")]
        builder.metrics.begin_test_suite(
            build_helper::metrics::TestSuiteMetadata::CargoPackage {
                crates: vec!["cargo".into()],
                target: self.host.triple.to_string(),
                host: self.host.triple.to_string(),
                stage,
            },
            builder,
        );

        let _time = helpers::timeit(builder);
        add_flags_and_try_run_tests(builder, &mut cargo);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RustAnalyzer {
    stage: u32,
    host: TargetSelection,
}

impl Step for RustAnalyzer {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/rust-analyzer")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Self { stage: run.builder.top_stage, host: run.target });
    }

    /// Runs `cargo test` for rust-analyzer
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let host = self.host;
        let compiler = builder.compiler(stage, host);
        let compiler = tool::get_tool_rustc_compiler(builder, compiler);

        // We don't need to build the whole Rust Analyzer for the proc-macro-srv test suite,
        // but we do need the standard library to be present.
        builder.ensure(compile::Rustc::new(compiler, host));

        let workspace_path = "src/tools/rust-analyzer";
        // until the whole RA test suite runs on `i686`, we only run
        // `proc-macro-srv` tests
        let crate_path = "src/tools/rust-analyzer/crates/proc-macro-srv";
        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            host,
            Kind::Test,
            crate_path,
            SourceType::InTree,
            &["in-rust-tree".to_owned()],
        );
        cargo.allow_features(tool::RustAnalyzer::ALLOW_FEATURES);

        let dir = builder.src.join(workspace_path);
        // needed by rust-analyzer to find its own text fixtures, cf.
        // https://github.com/rust-analyzer/expect-test/issues/33
        cargo.env("CARGO_WORKSPACE_DIR", &dir);

        // RA's test suite tries to write to the source directory, that can't
        // work in Rust CI
        cargo.env("SKIP_SLOW_TESTS", "1");

        cargo.add_rustc_lib_path(builder);
        run_cargo_test(cargo, &[], &[], "rust-analyzer", host, builder);
    }
}

/// Runs `cargo test` for rustfmt.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rustfmt {
    stage: u32,
    host: TargetSelection,
}

impl Step for Rustfmt {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/rustfmt")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rustfmt { stage: run.builder.top_stage, host: run.target });
    }

    /// Runs `cargo test` for rustfmt.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let host = self.host;
        let compiler = builder.compiler(stage, host);

        let tool_result = builder.ensure(tool::Rustfmt { compiler, target: self.host });
        let compiler = tool_result.build_compiler;

        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            host,
            Kind::Test,
            "src/tools/rustfmt",
            SourceType::InTree,
            &[],
        );

        let dir = testdir(builder, compiler.host);
        t!(fs::create_dir_all(&dir));
        cargo.env("RUSTFMT_TEST_DIR", dir);

        cargo.add_rustc_lib_path(builder);

        run_cargo_test(cargo, &[], &[], "rustfmt", host, builder);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Miri {
    target: TargetSelection,
}

impl Miri {
    /// Run `cargo miri setup` for the given target, return where the Miri sysroot was put.
    pub fn build_miri_sysroot(
        builder: &Builder<'_>,
        compiler: Compiler,
        target: TargetSelection,
    ) -> PathBuf {
        let miri_sysroot = builder.out.join(compiler.host).join("miri-sysroot");
        let mut cargo = builder::Cargo::new(
            builder,
            compiler,
            Mode::Std,
            SourceType::Submodule,
            target,
            Kind::MiriSetup,
        );

        // Tell `cargo miri setup` where to find the sources.
        cargo.env("MIRI_LIB_SRC", builder.src.join("library"));
        // Tell it where to put the sysroot.
        cargo.env("MIRI_SYSROOT", &miri_sysroot);

        let mut cargo = BootstrapCommand::from(cargo);
        let _guard =
            builder.msg(Kind::Build, compiler.stage, "miri sysroot", compiler.host, target);
        cargo.run(builder);

        // # Determine where Miri put its sysroot.
        // To this end, we run `cargo miri setup --print-sysroot` and capture the output.
        // (We do this separately from the above so that when the setup actually
        // happens we get some output.)
        // We re-use the `cargo` from above.
        cargo.arg("--print-sysroot");

        builder.verbose(|| println!("running: {cargo:?}"));
        let stdout = cargo.run_capture_stdout(builder).stdout();
        // Output is "<sysroot>\n".
        let sysroot = stdout.trim_end();
        builder.verbose(|| println!("`cargo miri setup --print-sysroot` said: {sysroot:?}"));
        PathBuf::from(sysroot)
    }
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

    /// Runs `cargo test` for miri.
    fn run(self, builder: &Builder<'_>) {
        let host = builder.build.host_target;
        let target = self.target;
        let stage = builder.top_stage;
        if stage == 0 {
            eprintln!("miri cannot be tested at stage 0");
            std::process::exit(1);
        }

        // This compiler runs on the host, we'll just use it for the target.
        let target_compiler = builder.compiler(stage, host);

        // Build our tools.
        let miri = builder.ensure(tool::Miri { compiler: target_compiler, target: host });
        // the ui tests also assume cargo-miri has been built
        builder.ensure(tool::CargoMiri { compiler: target_compiler, target: host });

        // We also need sysroots, for Miri and for the host (the latter for build scripts).
        // This is for the tests so everything is done with the target compiler.
        let miri_sysroot = Miri::build_miri_sysroot(builder, target_compiler, target);
        builder.ensure(compile::Std::new(target_compiler, host));
        let host_sysroot = builder.sysroot(target_compiler);

        // Miri has its own "target dir" for ui test dependencies. Make sure it gets cleared when
        // the sysroot gets rebuilt, to avoid "found possibly newer version of crate `std`" errors.
        if !builder.config.dry_run() {
            let ui_test_dep_dir =
                builder.stage_out(miri.build_compiler, Mode::ToolStd).join("miri_ui");
            // The mtime of `miri_sysroot` changes when the sysroot gets rebuilt (also see
            // <https://github.com/RalfJung/rustc-build-sysroot/commit/10ebcf60b80fe2c3dc765af0ff19fdc0da4b7466>).
            // We can hence use that directly as a signal to clear the ui test dir.
            build_stamp::clear_if_dirty(builder, &ui_test_dep_dir, &miri_sysroot);
        }

        // Run `cargo test`.
        // This is with the Miri crate, so it uses the host compiler.
        let mut cargo = tool::prepare_tool_cargo(
            builder,
            miri.build_compiler,
            Mode::ToolRustc,
            host,
            Kind::Test,
            "src/tools/miri",
            SourceType::InTree,
            &[],
        );

        cargo.add_rustc_lib_path(builder);

        // We can NOT use `run_cargo_test` since Miri's integration tests do not use the usual test
        // harness and therefore do not understand the flags added by `add_flags_and_try_run_test`.
        let mut cargo = prepare_cargo_test(cargo, &[], &[], host, builder);

        // miri tests need to know about the stage sysroot
        cargo.env("MIRI_SYSROOT", &miri_sysroot);
        cargo.env("MIRI_HOST_SYSROOT", &host_sysroot);
        cargo.env("MIRI", &miri.tool_path);

        // Set the target.
        cargo.env("MIRI_TEST_TARGET", target.rustc_target_arg());

        {
            let _guard = builder.msg_sysroot_tool(Kind::Test, stage, "miri", host, target);
            let _time = helpers::timeit(builder);
            cargo.run(builder);
        }

        // Run it again for mir-opt-level 4 to catch some miscompilations.
        if builder.config.test_args().is_empty() {
            cargo.env("MIRIFLAGS", "-O -Zmir-opt-level=4 -Cdebug-assertions=yes");
            // Optimizations can change backtraces
            cargo.env("MIRI_SKIP_UI_CHECKS", "1");
            // `MIRI_SKIP_UI_CHECKS` and `RUSTC_BLESS` are incompatible
            cargo.env_remove("RUSTC_BLESS");
            // Optimizations can change error locations and remove UB so don't run `fail` tests.
            cargo.args(["tests/pass", "tests/panic"]);

            {
                let _guard = builder.msg_sysroot_tool(
                    Kind::Test,
                    stage,
                    "miri (mir-opt-level 4)",
                    host,
                    target,
                );
                let _time = helpers::timeit(builder);
                cargo.run(builder);
            }
        }
    }
}

/// Runs `cargo miri test` to demonstrate that `src/tools/miri/cargo-miri`
/// works and that libtest works under miri.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CargoMiri {
    target: TargetSelection,
}

impl Step for CargoMiri {
    type Output = ();
    const ONLY_HOSTS: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/miri/cargo-miri")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(CargoMiri { target: run.target });
    }

    /// Tests `cargo miri test`.
    fn run(self, builder: &Builder<'_>) {
        let host = builder.build.host_target;
        let target = self.target;
        let stage = builder.top_stage;
        if stage == 0 {
            eprintln!("cargo-miri cannot be tested at stage 0");
            std::process::exit(1);
        }

        // This compiler runs on the host, we'll just use it for the target.
        let compiler = builder.compiler(stage, host);

        // Run `cargo miri test`.
        // This is just a smoke test (Miri's own CI invokes this in a bunch of different ways and ensures
        // that we get the desired output), but that is sufficient to make sure that the libtest harness
        // itself executes properly under Miri, and that all the logic in `cargo-miri` does not explode.
        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolStd, // it's unclear what to use here, we're not building anything just doing a smoke test!
            target,
            Kind::MiriTest,
            "src/tools/miri/test-cargo-miri",
            SourceType::Submodule,
            &[],
        );

        // We're not using `prepare_cargo_test` so we have to do this ourselves.
        // (We're not using that as the test-cargo-miri crate is not known to bootstrap.)
        match builder.doc_tests {
            DocTests::Yes => {}
            DocTests::No => {
                cargo.args(["--lib", "--bins", "--examples", "--tests", "--benches"]);
            }
            DocTests::Only => {
                cargo.arg("--doc");
            }
        }

        // Finally, pass test-args and run everything.
        cargo.arg("--").args(builder.config.test_args());
        let mut cargo = BootstrapCommand::from(cargo);
        {
            let _guard = builder.msg_sysroot_tool(Kind::Test, stage, "cargo-miri", host, target);
            let _time = helpers::timeit(builder);
            cargo.run(builder);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompiletestTest {
    host: TargetSelection,
}

impl Step for CompiletestTest {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/compiletest")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(CompiletestTest { host: run.target });
    }

    /// Runs `cargo test` for compiletest.
    fn run(self, builder: &Builder<'_>) {
        let host = self.host;
        let compiler = builder.compiler(builder.top_stage, host);

        // We need `ToolStd` for the locally-built sysroot because
        // compiletest uses unstable features of the `test` crate.
        builder.ensure(compile::Std::new(compiler, host));
        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            // compiletest uses libtest internals; make it use the in-tree std to make sure it never breaks
            // when std sources change.
            Mode::ToolStd,
            host,
            Kind::Test,
            "src/tools/compiletest",
            SourceType::InTree,
            &[],
        );
        cargo.allow_features(COMPILETEST_ALLOW_FEATURES);
        run_cargo_test(cargo, &[], &[], "compiletest self test", host, builder);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Clippy {
    stage: u32,
    host: TargetSelection,
}

impl Step for Clippy {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.suite_path("src/tools/clippy/tests").path("src/tools/clippy")
    }

    fn make_run(run: RunConfig<'_>) {
        // If stage is explicitly set or not lower than 2, keep it. Otherwise, make sure it's at least 2
        // as tests for this step don't work with a lower stage.
        let stage = if run.builder.config.is_explicit_stage() || run.builder.top_stage >= 2 {
            run.builder.top_stage
        } else {
            2
        };

        run.builder.ensure(Clippy { stage, host: run.target });
    }

    /// Runs `cargo test` for clippy.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let host = self.host;
        let compiler = builder.compiler(stage, host);

        if stage < 2 {
            eprintln!("WARNING: clippy tests on stage {stage} may not behave well.");
            eprintln!("HELP: consider using stage 2");
        }

        let tool_result = builder.ensure(tool::Clippy { compiler, target: self.host });
        let compiler = tool_result.build_compiler;
        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            host,
            Kind::Test,
            "src/tools/clippy",
            SourceType::InTree,
            &[],
        );

        cargo.env("RUSTC_TEST_SUITE", builder.rustc(compiler));
        cargo.env("RUSTC_LIB_PATH", builder.rustc_libdir(compiler));
        let host_libs = builder.stage_out(compiler, Mode::ToolRustc).join(builder.cargo_dir());
        cargo.env("HOST_LIBS", host_libs);

        // Collect paths of tests to run
        'partially_test: {
            let paths = &builder.config.paths[..];
            let mut test_names = Vec::new();
            for path in paths {
                if let Some(path) =
                    helpers::is_valid_test_suite_arg(path, "src/tools/clippy/tests", builder)
                {
                    test_names.push(path);
                } else if path.ends_with("src/tools/clippy") {
                    // When src/tools/clippy is called directly, all tests should be run.
                    break 'partially_test;
                }
            }
            cargo.env("TESTNAME", test_names.join(","));
        }

        cargo.add_rustc_lib_path(builder);
        let cargo = prepare_cargo_test(cargo, &[], &[], host, builder);

        let _guard = builder.msg_sysroot_tool(Kind::Test, compiler.stage, "clippy", host, host);

        // Clippy reports errors if it blessed the outputs
        if cargo.allow_failure().run(builder) {
            // The tests succeeded; nothing to do.
            return;
        }

        if !builder.config.cmd.bless() {
            crate::exit!(1);
        }
    }
}

fn path_for_cargo(builder: &Builder<'_>, compiler: Compiler) -> OsString {
    // Configure PATH to find the right rustc. NB. we have to use PATH
    // and not RUSTC because the Cargo test suite has tests that will
    // fail if rustc is not spelled `rustc`.
    let path = builder.sysroot(compiler).join("bin");
    let old_path = env::var_os("PATH").unwrap_or_default();
    env::join_paths(iter::once(path).chain(env::split_paths(&old_path))).expect("")
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

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RustdocJSStd {
    pub target: TargetSelection,
}

impl Step for RustdocJSStd {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = run.builder.config.nodejs.is_some();
        run.suite_path("tests/rustdoc-js-std").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustdocJSStd { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let nodejs =
            builder.config.nodejs.as_ref().expect("need nodejs to run rustdoc-js-std tests");
        let mut command = command(nodejs);
        command
            .arg(builder.src.join("src/tools/rustdoc-js/tester.js"))
            .arg("--crate-name")
            .arg("std")
            .arg("--resource-suffix")
            .arg(&builder.version)
            .arg("--doc-folder")
            .arg(builder.doc_out(self.target))
            .arg("--test-folder")
            .arg(builder.src.join("tests/rustdoc-js-std"));
        for path in &builder.paths {
            if let Some(p) = helpers::is_valid_test_suite_arg(path, "tests/rustdoc-js-std", builder)
            {
                if !p.ends_with(".js") {
                    eprintln!("A non-js file was given: `{}`", path.display());
                    panic!("Cannot run rustdoc-js-std tests");
                }
                command.arg("--test-file").arg(path);
            }
        }
        builder.ensure(crate::core::build_steps::doc::Std::new(
            builder.top_stage,
            self.target,
            DocumentationFormat::Html,
        ));
        let _guard = builder.msg(
            Kind::Test,
            builder.top_stage,
            "rustdoc-js-std",
            builder.config.host_target,
            self.target,
        );
        command.run(builder);
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RustdocJSNotStd {
    pub target: TargetSelection,
    pub compiler: Compiler,
}

impl Step for RustdocJSNotStd {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = run.builder.config.nodejs.is_some();
        run.suite_path("tests/rustdoc-js").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler = run.builder.compiler(run.builder.top_stage, run.build_triple());
        run.builder.ensure(RustdocJSNotStd { target: run.target, compiler });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.ensure(Compiletest {
            compiler: self.compiler,
            target: self.target,
            mode: "rustdoc-js",
            suite: "rustdoc-js",
            path: "tests/rustdoc-js",
            compare_mode: None,
        });
    }
}

fn get_browser_ui_test_version_inner(
    builder: &Builder<'_>,
    npm: &Path,
    global: bool,
) -> Option<String> {
    let mut command = command(npm);
    command.arg("list").arg("--parseable").arg("--long").arg("--depth=0");
    if global {
        command.arg("--global");
    }
    let lines = command.allow_failure().run_capture(builder).stdout();
    lines
        .lines()
        .find_map(|l| l.split(':').nth(1)?.strip_prefix("browser-ui-test@"))
        .map(|v| v.to_owned())
}

fn get_browser_ui_test_version(builder: &Builder<'_>, npm: &Path) -> Option<String> {
    get_browser_ui_test_version_inner(builder, npm, false)
        .or_else(|| get_browser_ui_test_version_inner(builder, npm, true))
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RustdocGUI {
    pub target: TargetSelection,
    pub compiler: Compiler,
}

impl Step for RustdocGUI {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        let run = run.suite_path("tests/rustdoc-gui");
        run.lazy_default_condition(Box::new(move || {
            builder.config.nodejs.is_some()
                && builder.doc_tests != DocTests::Only
                && builder
                    .config
                    .npm
                    .as_ref()
                    .map(|p| get_browser_ui_test_version(builder, p).is_some())
                    .unwrap_or(false)
        }))
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler = run.builder.compiler(run.builder.top_stage, run.build_triple());
        run.builder.ensure(RustdocGUI { target: run.target, compiler });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.ensure(compile::Std::new(self.compiler, self.target));

        let mut cmd = builder.tool_cmd(Tool::RustdocGUITest);

        let out_dir = builder.test_out(self.target).join("rustdoc-gui");
        build_stamp::clear_if_dirty(builder, &out_dir, &builder.rustdoc(self.compiler));

        if let Some(src) = builder.config.src.to_str() {
            cmd.arg("--rust-src").arg(src);
        }

        if let Some(out_dir) = out_dir.to_str() {
            cmd.arg("--out-dir").arg(out_dir);
        }

        if let Some(initial_cargo) = builder.config.initial_cargo.to_str() {
            cmd.arg("--initial-cargo").arg(initial_cargo);
        }

        cmd.arg("--jobs").arg(builder.jobs().to_string());

        cmd.env("RUSTDOC", builder.rustdoc(self.compiler))
            .env("RUSTC", builder.rustc(self.compiler));

        add_rustdoc_cargo_linker_args(&mut cmd, builder, self.compiler.host, LldThreads::No);

        for path in &builder.paths {
            if let Some(p) = helpers::is_valid_test_suite_arg(path, "tests/rustdoc-gui", builder) {
                if !p.ends_with(".goml") {
                    eprintln!("A non-goml file was given: `{}`", path.display());
                    panic!("Cannot run rustdoc-gui tests");
                }
                if let Some(name) = path.file_name().and_then(|f| f.to_str()) {
                    cmd.arg("--goml-file").arg(name);
                }
            }
        }

        for test_arg in builder.config.test_args() {
            cmd.arg("--test-arg").arg(test_arg);
        }

        if let Some(ref nodejs) = builder.config.nodejs {
            cmd.arg("--nodejs").arg(nodejs);
        }

        if let Some(ref npm) = builder.config.npm {
            cmd.arg("--npm").arg(npm);
        }

        let _time = helpers::timeit(builder);
        let _guard = builder.msg_sysroot_tool(
            Kind::Test,
            self.compiler.stage,
            "rustdoc-gui",
            self.compiler.host,
            self.target,
        );
        try_run_tests(builder, &mut cmd, true);
    }
}

/// Runs `src/tools/tidy` and `cargo fmt --check` to detect various style
/// problems in the repository.
///
/// (To run the tidy tool's internal tests, use the alias "tidyselftest" instead.)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tidy;

impl Step for Tidy {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    /// Runs the `tidy` tool.
    ///
    /// This tool in `src/tools` checks up on various bits and pieces of style and
    /// otherwise just implements a few lint-like checks that are specific to the
    /// compiler itself.
    ///
    /// Once tidy passes, this step also runs `fmt --check` if tests are being run
    /// for the `dev` or `nightly` channels.
    fn run(self, builder: &Builder<'_>) {
        let mut cmd = builder.tool_cmd(Tool::Tidy);
        cmd.arg(&builder.src);
        cmd.arg(&builder.initial_cargo);
        cmd.arg(&builder.out);
        // Tidy is heavily IO constrained. Still respect `-j`, but use a higher limit if `jobs` hasn't been configured.
        let jobs = builder.config.jobs.unwrap_or_else(|| {
            8 * std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get) as u32
        });
        cmd.arg(jobs.to_string());
        if builder.is_verbose() {
            cmd.arg("--verbose");
        }
        if builder.config.cmd.bless() {
            cmd.arg("--bless");
        }
        if let Some(s) = builder.config.cmd.extra_checks() {
            cmd.arg(format!("--extra-checks={s}"));
        }
        let mut args = std::env::args_os();
        if args.any(|arg| arg == OsStr::new("--")) {
            cmd.arg("--");
            cmd.args(args);
        }

        if builder.config.channel == "dev" || builder.config.channel == "nightly" {
            if !builder.config.json_output {
                builder.info("fmt check");
                if builder.config.initial_rustfmt.is_none() {
                    let inferred_rustfmt_dir = builder.initial_sysroot.join("bin");
                    eprintln!(
                        "\
ERROR: no `rustfmt` binary found in {PATH}
INFO: `rust.channel` is currently set to \"{CHAN}\"
HELP: if you are testing a beta branch, set `rust.channel` to \"beta\" in the `bootstrap.toml` file
HELP: to skip test's attempt to check tidiness, pass `--skip src/tools/tidy` to `x.py test`",
                        PATH = inferred_rustfmt_dir.display(),
                        CHAN = builder.config.channel,
                    );
                    crate::exit!(1);
                }
                let all = false;
                crate::core::build_steps::format::format(
                    builder,
                    !builder.config.cmd.bless(),
                    all,
                    &[],
                );
            } else {
                eprintln!(
                    "WARNING: `--json-output` is not supported on rustfmt, formatting will be skipped"
                );
            }
        }

        builder.info("tidy check");
        cmd.delay_failure().run(builder);

        builder.info("x.py completions check");
        let [bash, zsh, fish, powershell] = ["x.py.sh", "x.py.zsh", "x.py.fish", "x.py.ps1"]
            .map(|filename| builder.src.join("src/etc/completions").join(filename));
        if builder.config.cmd.bless() {
            builder.ensure(crate::core::build_steps::run::GenerateCompletions);
        } else if get_completion(shells::Bash, &bash).is_some()
            || get_completion(shells::Fish, &fish).is_some()
            || get_completion(shells::PowerShell, &powershell).is_some()
            || crate::flags::get_completion(shells::Zsh, &zsh).is_some()
        {
            eprintln!(
                "x.py completions were changed; run `x.py run generate-completions` to update them"
            );
            crate::exit!(1);
        }
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = run.builder.doc_tests != DocTests::Only;
        run.path("src/tools/tidy").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Tidy);
    }
}

fn testdir(builder: &Builder<'_>, host: TargetSelection) -> PathBuf {
    builder.out.join(host).join("test")
}

/// Declares a test step that invokes compiletest on a particular test suite.
macro_rules! test {
    (
        $( #[$attr:meta] )* // allow docstrings and attributes
        $name:ident {
            path: $path:expr,
            mode: $mode:expr,
            suite: $suite:expr,
            default: $default:expr
            $( , only_hosts: $only_hosts:expr )? // default: false
            $( , compare_mode: $compare_mode:expr )? // default: None
            $( , )? // optional trailing comma
        }
    ) => {
        $( #[$attr] )*
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name {
            pub compiler: Compiler,
            pub target: TargetSelection,
        }

        impl Step for $name {
            type Output = ();
            const DEFAULT: bool = $default;
            const ONLY_HOSTS: bool = (const {
                #[allow(unused_assignments, unused_mut)]
                let mut value = false;
                $( value = $only_hosts; )?
                value
            });

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                run.suite_path($path)
            }

            fn make_run(run: RunConfig<'_>) {
                let compiler = run.builder.compiler(run.builder.top_stage, run.build_triple());

                run.builder.ensure($name { compiler, target: run.target });
            }

            fn run(self, builder: &Builder<'_>) {
                builder.ensure(Compiletest {
                    compiler: self.compiler,
                    target: self.target,
                    mode: $mode,
                    suite: $suite,
                    path: $path,
                    compare_mode: (const {
                        #[allow(unused_assignments, unused_mut)]
                        let mut value = None;
                        $( value = $compare_mode; )?
                        value
                    }),
                })
            }
        }
    };
}

/// Runs `cargo test` on the `src/tools/run-make-support` crate.
/// That crate is used by run-make tests.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateRunMakeSupport {
    host: TargetSelection,
}

impl Step for CrateRunMakeSupport {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/run-make-support")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(CrateRunMakeSupport { host: run.target });
    }

    /// Runs `cargo test` for run-make-support.
    fn run(self, builder: &Builder<'_>) {
        let host = self.host;
        let compiler = builder.compiler(0, host);

        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolBootstrap,
            host,
            Kind::Test,
            "src/tools/run-make-support",
            SourceType::InTree,
            &[],
        );
        cargo.allow_features("test");
        run_cargo_test(cargo, &[], &[], "run-make-support self test", host, builder);
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
        run_cargo_test(cargo, &[], &[], "build_helper self test", host, builder);
    }
}

test!(Ui { path: "tests/ui", mode: "ui", suite: "ui", default: true });

test!(Crashes { path: "tests/crashes", mode: "crashes", suite: "crashes", default: true });

test!(Codegen { path: "tests/codegen", mode: "codegen", suite: "codegen", default: true });

test!(CodegenUnits {
    path: "tests/codegen-units",
    mode: "codegen-units",
    suite: "codegen-units",
    default: true,
});

test!(Incremental {
    path: "tests/incremental",
    mode: "incremental",
    suite: "incremental",
    default: true,
});

test!(Debuginfo {
    path: "tests/debuginfo",
    mode: "debuginfo",
    suite: "debuginfo",
    default: true,
    compare_mode: Some("split-dwarf"),
});

test!(UiFullDeps {
    path: "tests/ui-fulldeps",
    mode: "ui",
    suite: "ui-fulldeps",
    default: true,
    only_hosts: true,
});

test!(Rustdoc {
    path: "tests/rustdoc",
    mode: "rustdoc",
    suite: "rustdoc",
    default: true,
    only_hosts: true,
});
test!(RustdocUi {
    path: "tests/rustdoc-ui",
    mode: "ui",
    suite: "rustdoc-ui",
    default: true,
    only_hosts: true,
});

test!(RustdocJson {
    path: "tests/rustdoc-json",
    mode: "rustdoc-json",
    suite: "rustdoc-json",
    default: true,
    only_hosts: true,
});

test!(Pretty {
    path: "tests/pretty",
    mode: "pretty",
    suite: "pretty",
    default: true,
    only_hosts: true,
});

test!(RunMake { path: "tests/run-make", mode: "run-make", suite: "run-make", default: true });

test!(Assembly { path: "tests/assembly", mode: "assembly", suite: "assembly", default: true });

/// Runs the coverage test suite at `tests/coverage` in some or all of the
/// coverage test modes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Coverage {
    pub compiler: Compiler,
    pub target: TargetSelection,
    pub mode: &'static str,
}

impl Coverage {
    const PATH: &'static str = "tests/coverage";
    const SUITE: &'static str = "coverage";
    const ALL_MODES: &[&str] = &["coverage-map", "coverage-run"];
}

impl Step for Coverage {
    type Output = ();
    const DEFAULT: bool = true;
    /// Compiletest will automatically skip the "coverage-run" tests if necessary.
    const ONLY_HOSTS: bool = false;

    fn should_run(mut run: ShouldRun<'_>) -> ShouldRun<'_> {
        // Support various invocation styles, including:
        // - `./x test coverage`
        // - `./x test tests/coverage/trivial.rs`
        // - `./x test coverage-map`
        // - `./x test coverage-run -- tests/coverage/trivial.rs`
        run = run.suite_path(Self::PATH);
        for mode in Self::ALL_MODES {
            run = run.alias(mode);
        }
        run
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler = run.builder.compiler(run.builder.top_stage, run.build_triple());
        let target = run.target;

        // List of (coverage) test modes that the coverage test suite will be
        // run in. It's OK for this to contain duplicates, because the call to
        // `Builder::ensure` below will take care of deduplication.
        let mut modes = vec![];

        // From the pathsets that were selected on the command-line (or by default),
        // determine which modes to run in.
        for path in &run.paths {
            match path {
                PathSet::Set(_) => {
                    for mode in Self::ALL_MODES {
                        if path.assert_single_path().path == Path::new(mode) {
                            modes.push(mode);
                            break;
                        }
                    }
                }
                PathSet::Suite(_) => {
                    modes.extend(Self::ALL_MODES);
                    break;
                }
            }
        }

        // Skip any modes that were explicitly skipped/excluded on the command-line.
        // FIXME(Zalathar): Integrate this into central skip handling somehow?
        modes.retain(|mode| !run.builder.config.skip.iter().any(|skip| skip == Path::new(mode)));

        // FIXME(Zalathar): Make these commands skip all coverage tests, as expected:
        // - `./x test --skip=tests`
        // - `./x test --skip=tests/coverage`
        // - `./x test --skip=coverage`
        // Skip handling currently doesn't have a way to know that skipping the coverage
        // suite should also skip the `coverage-map` and `coverage-run` aliases.

        for mode in modes {
            run.builder.ensure(Coverage { compiler, target, mode });
        }
    }

    fn run(self, builder: &Builder<'_>) {
        let Self { compiler, target, mode } = self;
        // Like other compiletest suite test steps, delegate to an internal
        // compiletest task to actually run the tests.
        builder.ensure(Compiletest {
            compiler,
            target,
            mode,
            suite: Self::SUITE,
            path: Self::PATH,
            compare_mode: None,
        });
    }
}

test!(CoverageRunRustdoc {
    path: "tests/coverage-run-rustdoc",
    mode: "coverage-run",
    suite: "coverage-run-rustdoc",
    default: true,
    only_hosts: true,
});

// For the mir-opt suite we do not use macros, as we need custom behavior when blessing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MirOpt {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for MirOpt {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.suite_path("tests/mir-opt")
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler = run.builder.compiler(run.builder.top_stage, run.build_triple());
        run.builder.ensure(MirOpt { compiler, target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let run = |target| {
            builder.ensure(Compiletest {
                compiler: self.compiler,
                target,
                mode: "mir-opt",
                suite: "mir-opt",
                path: "tests/mir-opt",
                compare_mode: None,
            })
        };

        run(self.target);

        // Run more targets with `--bless`. But we always run the host target first, since some
        // tests use very specific `only` clauses that are not covered by the target set below.
        if builder.config.cmd.bless() {
            // All that we really need to do is cover all combinations of 32/64-bit and unwind/abort,
            // but while we're at it we might as well flex our cross-compilation support. This
            // selection covers all our tier 1 operating systems and architectures using only tier
            // 1 targets.

            for target in ["aarch64-unknown-linux-gnu", "i686-pc-windows-msvc"] {
                run(TargetSelection::from_user(target));
            }

            for target in ["x86_64-apple-darwin", "i686-unknown-linux-musl"] {
                let target = TargetSelection::from_user(target);
                let panic_abort_target = builder.ensure(MirOptPanicAbortSyntheticTarget {
                    compiler: self.compiler,
                    base: target,
                });
                run(panic_abort_target);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Compiletest {
    compiler: Compiler,
    target: TargetSelection,
    mode: &'static str,
    suite: &'static str,
    path: &'static str,
    compare_mode: Option<&'static str>,
}

impl Step for Compiletest {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Executes the `compiletest` tool to run a suite of tests.
    ///
    /// Compiles all tests with `compiler` for `target` with the specified
    /// compiletest `mode` and `suite` arguments. For example `mode` can be
    /// "run-pass" or `suite` can be something like `debuginfo`.
    fn run(self, builder: &Builder<'_>) {
        if builder.doc_tests == DocTests::Only {
            return;
        }

        if builder.top_stage == 0 && env::var("COMPILETEST_FORCE_STAGE0").is_err() {
            eprintln!("\
ERROR: `--stage 0` runs compiletest on the stage0 (precompiled) compiler, not your local changes, and will almost always cause tests to fail
HELP: to test the compiler, use `--stage 1` instead
HELP: to test the standard library, use `--stage 0 library/std` instead
NOTE: if you're sure you want to do this, please open an issue as to why. In the meantime, you can override this with `COMPILETEST_FORCE_STAGE0=1`."
            );
            crate::exit!(1);
        }

        let mut compiler = self.compiler;
        let target = self.target;
        let mode = self.mode;
        let suite = self.suite;

        // Path for test suite
        let suite_path = self.path;

        // Skip codegen tests if they aren't enabled in configuration.
        if !builder.config.codegen_tests && suite == "codegen" {
            return;
        }

        // Support stage 1 ui-fulldeps. This is somewhat complicated: ui-fulldeps tests for the most
        // part test the *API* of the compiler, not how it compiles a given file. As a result, we
        // can run them against the stage 1 sources as long as we build them with the stage 0
        // bootstrap compiler.
        // NOTE: Only stage 1 is special cased because we need the rustc_private artifacts to match the
        // running compiler in stage 2 when plugins run.
        let (stage, stage_id) = if suite == "ui-fulldeps" && compiler.stage == 1 {
            // At stage 0 (stage - 1) we are using the stage0 compiler. Using `self.target` can lead
            // finding an incorrect compiler path on cross-targets, as the stage 0 is always equal to
            // `build.build` in the configuration.
            let build = builder.build.host_target;
            compiler = builder.compiler(compiler.stage - 1, build);
            let test_stage = compiler.stage + 1;
            (test_stage, format!("stage{test_stage}-{build}"))
        } else {
            let stage = compiler.stage;
            (stage, format!("stage{stage}-{target}"))
        };

        if suite.ends_with("fulldeps") {
            builder.ensure(compile::Rustc::new(compiler, target));
        }

        if suite == "debuginfo" {
            builder.ensure(dist::DebuggerScripts {
                sysroot: builder.sysroot(compiler).to_path_buf(),
                host: target,
            });
        }
        if suite == "run-make" {
            builder.tool_exe(Tool::RunMakeSupport);
        }

        // ensure that `libproc_macro` is available on the host.
        if suite == "mir-opt" {
            builder.ensure(compile::Std::new(compiler, compiler.host).is_for_mir_opt_tests(true));
        } else {
            builder.ensure(compile::Std::new(compiler, compiler.host));
        }

        let mut cmd = builder.tool_cmd(Tool::Compiletest);

        if suite == "mir-opt" {
            builder.ensure(compile::Std::new(compiler, target).is_for_mir_opt_tests(true));
        } else {
            builder.ensure(compile::Std::new(compiler, target));
        }

        builder.ensure(RemoteCopyLibs { compiler, target });

        // compiletest currently has... a lot of arguments, so let's just pass all
        // of them!

        cmd.arg("--stage").arg(stage.to_string());
        cmd.arg("--stage-id").arg(stage_id);

        cmd.arg("--compile-lib-path").arg(builder.rustc_libdir(compiler));
        cmd.arg("--run-lib-path").arg(builder.sysroot_target_libdir(compiler, target));
        cmd.arg("--rustc-path").arg(builder.rustc(compiler));

        // Minicore auxiliary lib for `no_core` tests that need `core` stubs in cross-compilation
        // scenarios.
        cmd.arg("--minicore-path")
            .arg(builder.src.join("tests").join("auxiliary").join("minicore.rs"));

        let is_rustdoc = suite == "rustdoc-ui" || suite == "rustdoc-js";

        if mode == "run-make" {
            let cargo_path = if builder.top_stage == 0 {
                // If we're using `--stage 0`, we should provide the bootstrap cargo.
                builder.initial_cargo.clone()
            } else {
                builder.ensure(tool::Cargo { compiler, target: compiler.host }).tool_path
            };

            cmd.arg("--cargo-path").arg(cargo_path);

            // We need to pass the compiler that was used to compile run-make-support,
            // because we have to use the same compiler to compile rmake.rs recipes.
            let stage0_rustc_path = builder.compiler(0, compiler.host);
            cmd.arg("--stage0-rustc-path").arg(builder.rustc(stage0_rustc_path));
        }

        // Avoid depending on rustdoc when we don't need it.
        if mode == "rustdoc"
            || mode == "run-make"
            || (mode == "ui" && is_rustdoc)
            || mode == "rustdoc-js"
            || mode == "rustdoc-json"
            || suite == "coverage-run-rustdoc"
        {
            cmd.arg("--rustdoc-path").arg(builder.rustdoc(compiler));
        }

        if mode == "rustdoc-json" {
            // Use the stage0 compiler for jsondocck
            let json_compiler = compiler.with_stage(0);
            cmd.arg("--jsondocck-path")
                .arg(builder.ensure(tool::JsonDocCk { compiler: json_compiler, target }).tool_path);
            cmd.arg("--jsondoclint-path").arg(
                builder.ensure(tool::JsonDocLint { compiler: json_compiler, target }).tool_path,
            );
        }

        if matches!(mode, "coverage-map" | "coverage-run") {
            let coverage_dump = builder.tool_exe(Tool::CoverageDump);
            cmd.arg("--coverage-dump-path").arg(coverage_dump);
        }

        cmd.arg("--src-root").arg(&builder.src);
        cmd.arg("--src-test-suite-root").arg(builder.src.join("tests").join(suite));

        // N.B. it's important to distinguish between the *root* build directory, the *host* build
        // directory immediately under the root build directory, and the test-suite-specific build
        // directory.
        cmd.arg("--build-root").arg(&builder.out);
        cmd.arg("--build-test-suite-root").arg(testdir(builder, compiler.host).join(suite));

        // When top stage is 0, that means that we're testing an externally provided compiler.
        // In that case we need to use its specific sysroot for tests to pass.
        let sysroot = if builder.top_stage == 0 {
            builder.initial_sysroot.clone()
        } else {
            builder.sysroot(compiler)
        };

        cmd.arg("--sysroot-base").arg(sysroot);

        cmd.arg("--suite").arg(suite);
        cmd.arg("--mode").arg(mode);
        cmd.arg("--target").arg(target.rustc_target_arg());
        cmd.arg("--host").arg(&*compiler.host.triple);
        cmd.arg("--llvm-filecheck").arg(builder.llvm_filecheck(builder.config.host_target));

        if builder.build.config.llvm_enzyme {
            cmd.arg("--has-enzyme");
        }

        if builder.config.cmd.bless() {
            cmd.arg("--bless");
        }

        if builder.config.cmd.force_rerun() {
            cmd.arg("--force-rerun");
        }

        if builder.config.cmd.no_capture() {
            cmd.arg("--no-capture");
        }

        let compare_mode =
            builder.config.cmd.compare_mode().or_else(|| {
                if builder.config.test_compare_mode { self.compare_mode } else { None }
            });

        if let Some(ref pass) = builder.config.cmd.pass() {
            cmd.arg("--pass");
            cmd.arg(pass);
        }

        if let Some(ref run) = builder.config.cmd.run() {
            cmd.arg("--run");
            cmd.arg(run);
        }

        if let Some(ref nodejs) = builder.config.nodejs {
            cmd.arg("--nodejs").arg(nodejs);
        } else if mode == "rustdoc-js" {
            panic!("need nodejs to run rustdoc-js suite");
        }
        if let Some(ref npm) = builder.config.npm {
            cmd.arg("--npm").arg(npm);
        }
        if builder.config.rust_optimize_tests {
            cmd.arg("--optimize-tests");
        }
        if builder.config.rust_randomize_layout {
            cmd.arg("--rust-randomized-layout");
        }
        if builder.config.cmd.only_modified() {
            cmd.arg("--only-modified");
        }
        if let Some(compiletest_diff_tool) = &builder.config.compiletest_diff_tool {
            cmd.arg("--compiletest-diff-tool").arg(compiletest_diff_tool);
        }

        let mut flags = if is_rustdoc { Vec::new() } else { vec!["-Crpath".to_string()] };
        flags.push(format!("-Cdebuginfo={}", builder.config.rust_debuginfo_level_tests));
        flags.extend(builder.config.cmd.compiletest_rustc_args().iter().map(|s| s.to_string()));

        if suite != "mir-opt" {
            if let Some(linker) = builder.linker(target) {
                cmd.arg("--target-linker").arg(linker);
            }
            if let Some(linker) = builder.linker(compiler.host) {
                cmd.arg("--host-linker").arg(linker);
            }
        }

        // FIXME(136096): on macOS, we get linker warnings about duplicate `-lm` flags.
        if suite == "ui-fulldeps" && target.ends_with("darwin") {
            flags.push("-Alinker_messages".into());
        }

        let mut hostflags = flags.clone();
        hostflags.extend(linker_flags(builder, compiler.host, LldThreads::No));

        let mut targetflags = flags;

        // Provide `rust_test_helpers` for both host and target.
        if suite == "ui" || suite == "incremental" {
            builder.ensure(TestHelpers { target: compiler.host });
            builder.ensure(TestHelpers { target });
            hostflags
                .push(format!("-Lnative={}", builder.test_helpers_out(compiler.host).display()));
            targetflags.push(format!("-Lnative={}", builder.test_helpers_out(target).display()));
        }

        for flag in hostflags {
            cmd.arg("--host-rustcflags").arg(flag);
        }
        for flag in targetflags {
            cmd.arg("--target-rustcflags").arg(flag);
        }

        cmd.arg("--python").arg(builder.python());

        if let Some(ref gdb) = builder.config.gdb {
            cmd.arg("--gdb").arg(gdb);
        }

        let lldb_exe = builder.config.lldb.clone().unwrap_or_else(|| PathBuf::from("lldb"));
        let lldb_version = command(&lldb_exe)
            .allow_failure()
            .arg("--version")
            .run_capture(builder)
            .stdout_if_ok()
            .and_then(|v| if v.trim().is_empty() { None } else { Some(v) });
        if let Some(ref vers) = lldb_version {
            cmd.arg("--lldb-version").arg(vers);
            let lldb_python_dir = command(&lldb_exe)
                .allow_failure()
                .arg("-P")
                .run_capture_stdout(builder)
                .stdout_if_ok()
                .map(|p| p.lines().next().expect("lldb Python dir not found").to_string());
            if let Some(ref dir) = lldb_python_dir {
                cmd.arg("--lldb-python-dir").arg(dir);
            }
        }

        if helpers::forcing_clang_based_tests() {
            let clang_exe = builder.llvm_out(target).join("bin").join("clang");
            cmd.arg("--run-clang-based-tests-with").arg(clang_exe);
        }

        for exclude in &builder.config.skip {
            cmd.arg("--skip");
            cmd.arg(exclude);
        }

        // Get paths from cmd args
        let paths = match &builder.config.cmd {
            Subcommand::Test { .. } => &builder.config.paths[..],
            _ => &[],
        };

        // Get test-args by striping suite path
        let mut test_args: Vec<&str> = paths
            .iter()
            .filter_map(|p| helpers::is_valid_test_suite_arg(p, suite_path, builder))
            .collect();

        test_args.append(&mut builder.config.test_args());

        // On Windows, replace forward slashes in test-args by backslashes
        // so the correct filters are passed to libtest
        if cfg!(windows) {
            let test_args_win: Vec<String> =
                test_args.iter().map(|s| s.replace('/', "\\")).collect();
            cmd.args(&test_args_win);
        } else {
            cmd.args(&test_args);
        }

        if builder.is_verbose() {
            cmd.arg("--verbose");
        }

        cmd.arg("--json");

        if builder.config.rustc_debug_assertions {
            cmd.arg("--with-rustc-debug-assertions");
        }

        if builder.config.std_debug_assertions {
            cmd.arg("--with-std-debug-assertions");
        }

        let mut llvm_components_passed = false;
        let mut copts_passed = false;
        if builder.config.llvm_enabled(compiler.host) {
            let llvm::LlvmResult { llvm_config, .. } =
                builder.ensure(llvm::Llvm { target: builder.config.host_target });
            if !builder.config.dry_run() {
                let llvm_version = get_llvm_version(builder, &llvm_config);
                let llvm_components =
                    command(&llvm_config).arg("--components").run_capture_stdout(builder).stdout();
                // Remove trailing newline from llvm-config output.
                cmd.arg("--llvm-version")
                    .arg(llvm_version.trim())
                    .arg("--llvm-components")
                    .arg(llvm_components.trim());
                llvm_components_passed = true;
            }
            if !builder.config.is_rust_llvm(target) {
                cmd.arg("--system-llvm");
            }

            // Tests that use compiler libraries may inherit the `-lLLVM` link
            // requirement, but the `-L` library path is not propagated across
            // separate compilations. We can add LLVM's library path to the
            // rustc args as a workaround.
            if !builder.config.dry_run() && suite.ends_with("fulldeps") {
                let llvm_libdir =
                    command(&llvm_config).arg("--libdir").run_capture_stdout(builder).stdout();
                let link_llvm = if target.is_msvc() {
                    format!("-Clink-arg=-LIBPATH:{llvm_libdir}")
                } else {
                    format!("-Clink-arg=-L{llvm_libdir}")
                };
                cmd.arg("--host-rustcflags").arg(link_llvm);
            }

            if !builder.config.dry_run() && matches!(mode, "run-make" | "coverage-run") {
                // The llvm/bin directory contains many useful cross-platform
                // tools. Pass the path to run-make tests so they can use them.
                // (The coverage-run tests also need these tools to process
                // coverage reports.)
                let llvm_bin_path = llvm_config
                    .parent()
                    .expect("Expected llvm-config to be contained in directory");
                assert!(llvm_bin_path.is_dir());
                cmd.arg("--llvm-bin-dir").arg(llvm_bin_path);
            }

            if !builder.config.dry_run() && mode == "run-make" {
                // If LLD is available, add it to the PATH
                if builder.config.lld_enabled {
                    let lld_install_root =
                        builder.ensure(llvm::Lld { target: builder.config.host_target });

                    let lld_bin_path = lld_install_root.join("bin");

                    let old_path = env::var_os("PATH").unwrap_or_default();
                    let new_path = env::join_paths(
                        std::iter::once(lld_bin_path).chain(env::split_paths(&old_path)),
                    )
                    .expect("Could not add LLD bin path to PATH");
                    cmd.env("PATH", new_path);
                }
            }
        }

        // Only pass correct values for these flags for the `run-make` suite as it
        // requires that a C++ compiler was configured which isn't always the case.
        if !builder.config.dry_run() && mode == "run-make" {
            let mut cflags = builder.cc_handled_clags(target, CLang::C);
            cflags.extend(builder.cc_unhandled_cflags(target, GitRepo::Rustc, CLang::C));
            let mut cxxflags = builder.cc_handled_clags(target, CLang::Cxx);
            cxxflags.extend(builder.cc_unhandled_cflags(target, GitRepo::Rustc, CLang::Cxx));
            cmd.arg("--cc")
                .arg(builder.cc(target))
                .arg("--cxx")
                .arg(builder.cxx(target).unwrap())
                .arg("--cflags")
                .arg(cflags.join(" "))
                .arg("--cxxflags")
                .arg(cxxflags.join(" "));
            copts_passed = true;
            if let Some(ar) = builder.ar(target) {
                cmd.arg("--ar").arg(ar);
            }
        }

        if !llvm_components_passed {
            cmd.arg("--llvm-components").arg("");
        }
        if !copts_passed {
            cmd.arg("--cc")
                .arg("")
                .arg("--cxx")
                .arg("")
                .arg("--cflags")
                .arg("")
                .arg("--cxxflags")
                .arg("");
        }

        if builder.remote_tested(target) {
            cmd.arg("--remote-test-client").arg(builder.tool_exe(Tool::RemoteTestClient));
        } else if let Some(tool) = builder.runner(target) {
            cmd.arg("--runner").arg(tool);
        }

        if suite != "mir-opt" {
            // Running a C compiler on MSVC requires a few env vars to be set, to be
            // sure to set them here.
            //
            // Note that if we encounter `PATH` we make sure to append to our own `PATH`
            // rather than stomp over it.
            if !builder.config.dry_run() && target.is_msvc() {
                for (k, v) in builder.cc[&target].env() {
                    if k != "PATH" {
                        cmd.env(k, v);
                    }
                }
            }
        }

        // Special setup to enable running with sanitizers on MSVC.
        if !builder.config.dry_run()
            && target.contains("msvc")
            && builder.config.sanitizers_enabled(target)
        {
            // Ignore interception failures: not all dlls in the process will have been built with
            // address sanitizer enabled (e.g., ntdll.dll).
            cmd.env("ASAN_WIN_CONTINUE_ON_INTERCEPTION_FAILURE", "1");
            // Add the address sanitizer runtime to the PATH - it is located next to cl.exe.
            let asan_runtime_path = builder.cc[&target].path().parent().unwrap().to_path_buf();
            let old_path = cmd
                .get_envs()
                .find_map(|(k, v)| (k == "PATH").then_some(v))
                .flatten()
                .map_or_else(|| env::var_os("PATH").unwrap_or_default(), |v| v.to_owned());
            let new_path = env::join_paths(
                env::split_paths(&old_path).chain(std::iter::once(asan_runtime_path)),
            )
            .expect("Could not add ASAN runtime path to PATH");
            cmd.env("PATH", new_path);
        }

        // Some UI tests trigger behavior in rustc where it reads $CARGO and changes behavior if it exists.
        // To make the tests work that rely on it not being set, make sure it is not set.
        cmd.env_remove("CARGO");

        cmd.env("RUSTC_BOOTSTRAP", "1");
        // Override the rustc version used in symbol hashes to reduce the amount of normalization
        // needed when diffing test output.
        cmd.env("RUSTC_FORCE_RUSTC_VERSION", "compiletest");
        cmd.env("DOC_RUST_LANG_ORG_CHANNEL", builder.doc_rust_lang_org_channel());
        builder.add_rust_test_threads(&mut cmd);

        if builder.config.sanitizers_enabled(target) {
            cmd.env("RUSTC_SANITIZER_SUPPORT", "1");
        }

        if builder.config.profiler_enabled(target) {
            cmd.arg("--profiler-runtime");
        }

        cmd.env("RUST_TEST_TMPDIR", builder.tempdir());

        cmd.arg("--adb-path").arg("adb");
        cmd.arg("--adb-test-dir").arg(ADB_TEST_DIR);
        if target.contains("android") && !builder.config.dry_run() {
            // Assume that cc for this target comes from the android sysroot
            cmd.arg("--android-cross-path")
                .arg(builder.cc(target).parent().unwrap().parent().unwrap());
        } else {
            cmd.arg("--android-cross-path").arg("");
        }

        if builder.config.cmd.rustfix_coverage() {
            cmd.arg("--rustfix-coverage");
        }

        cmd.arg("--channel").arg(&builder.config.channel);

        if !builder.config.omit_git_hash {
            cmd.arg("--git-hash");
        }

        let git_config = builder.config.git_config();
        cmd.arg("--nightly-branch").arg(git_config.nightly_branch);
        cmd.arg("--git-merge-commit-email").arg(git_config.git_merge_commit_email);
        cmd.force_coloring_in_ci();

        #[cfg(feature = "build-metrics")]
        builder.metrics.begin_test_suite(
            build_helper::metrics::TestSuiteMetadata::Compiletest {
                suite: suite.into(),
                mode: mode.into(),
                compare_mode: None,
                target: self.target.triple.to_string(),
                host: self.compiler.host.triple.to_string(),
                stage: self.compiler.stage,
            },
            builder,
        );

        let _group = builder.msg(
            Kind::Test,
            compiler.stage,
            format!("compiletest suite={suite} mode={mode}"),
            compiler.host,
            target,
        );
        try_run_tests(builder, &mut cmd, false);

        if let Some(compare_mode) = compare_mode {
            cmd.arg("--compare-mode").arg(compare_mode);

            #[cfg(feature = "build-metrics")]
            builder.metrics.begin_test_suite(
                build_helper::metrics::TestSuiteMetadata::Compiletest {
                    suite: suite.into(),
                    mode: mode.into(),
                    compare_mode: Some(compare_mode.into()),
                    target: self.target.triple.to_string(),
                    host: self.compiler.host.triple.to_string(),
                    stage: self.compiler.stage,
                },
                builder,
            );

            builder.info(&format!(
                "Check compiletest suite={} mode={} compare_mode={} ({} -> {})",
                suite, mode, compare_mode, &compiler.host, target
            ));
            let _time = helpers::timeit(builder);
            try_run_tests(builder, &mut cmd, false);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BookTest {
    compiler: Compiler,
    path: PathBuf,
    name: &'static str,
    is_ext_doc: bool,
    dependencies: Vec<&'static str>,
}

impl Step for BookTest {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Runs the documentation tests for a book in `src/doc`.
    ///
    /// This uses the `rustdoc` that sits next to `compiler`.
    fn run(self, builder: &Builder<'_>) {
        // External docs are different from local because:
        // - Some books need pre-processing by mdbook before being tested.
        // - They need to save their state to toolstate.
        // - They are only tested on the "checktools" builders.
        //
        // The local docs are tested by default, and we don't want to pay the
        // cost of building mdbook, so they use `rustdoc --test` directly.
        // Also, the unstable book is special because SUMMARY.md is generated,
        // so it is easier to just run `rustdoc` on its files.
        if self.is_ext_doc {
            self.run_ext_doc(builder);
        } else {
            self.run_local_doc(builder);
        }
    }
}

impl BookTest {
    /// This runs the equivalent of `mdbook test` (via the rustbook wrapper)
    /// which in turn runs `rustdoc --test` on each file in the book.
    fn run_ext_doc(self, builder: &Builder<'_>) {
        let compiler = self.compiler;

        builder.ensure(compile::Std::new(compiler, compiler.host));

        // mdbook just executes a binary named "rustdoc", so we need to update
        // PATH so that it points to our rustdoc.
        let mut rustdoc_path = builder.rustdoc(compiler);
        rustdoc_path.pop();
        let old_path = env::var_os("PATH").unwrap_or_default();
        let new_path = env::join_paths(iter::once(rustdoc_path).chain(env::split_paths(&old_path)))
            .expect("could not add rustdoc to PATH");

        let mut rustbook_cmd = builder.tool_cmd(Tool::Rustbook);
        let path = builder.src.join(&self.path);
        // Books often have feature-gated example text.
        rustbook_cmd.env("RUSTC_BOOTSTRAP", "1");
        rustbook_cmd.env("PATH", new_path).arg("test").arg(path);

        // Books may also need to build dependencies. For example, `TheBook` has
        // code samples which use the `trpl` crate. For the `rustdoc` invocation
        // to find them them successfully, they need to be built first and their
        // paths used to generate the
        let libs = if !self.dependencies.is_empty() {
            let mut lib_paths = vec![];
            for dep in self.dependencies {
                let mode = Mode::ToolRustc;
                let target = builder.config.host_target;
                let cargo = tool::prepare_tool_cargo(
                    builder,
                    compiler,
                    mode,
                    target,
                    Kind::Build,
                    dep,
                    SourceType::Submodule,
                    &[],
                );

                let stamp = BuildStamp::new(&builder.cargo_out(compiler, mode, target))
                    .with_prefix(PathBuf::from(dep).file_name().and_then(|v| v.to_str()).unwrap());

                let output_paths = run_cargo(builder, cargo, vec![], &stamp, vec![], false, false);
                let directories = output_paths
                    .into_iter()
                    .filter_map(|p| p.parent().map(ToOwned::to_owned))
                    .fold(HashSet::new(), |mut set, dir| {
                        set.insert(dir);
                        set
                    });

                lib_paths.extend(directories);
            }
            lib_paths
        } else {
            vec![]
        };

        if !libs.is_empty() {
            let paths = libs
                .into_iter()
                .map(|path| path.into_os_string())
                .collect::<Vec<OsString>>()
                .join(OsStr::new(","));
            rustbook_cmd.args([OsString::from("--library-path"), paths]);
        }

        builder.add_rust_test_threads(&mut rustbook_cmd);
        let _guard = builder.msg(
            Kind::Test,
            compiler.stage,
            format_args!("mdbook {}", self.path.display()),
            compiler.host,
            compiler.host,
        );
        let _time = helpers::timeit(builder);
        let toolstate = if rustbook_cmd.delay_failure().run(builder) {
            ToolState::TestPass
        } else {
            ToolState::TestFail
        };
        builder.save_toolstate(self.name, toolstate);
    }

    /// This runs `rustdoc --test` on all `.md` files in the path.
    fn run_local_doc(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let host = self.compiler.host;

        builder.ensure(compile::Std::new(compiler, host));

        let _guard =
            builder.msg(Kind::Test, compiler.stage, format!("book {}", self.name), host, host);

        // Do a breadth-first traversal of the `src/doc` directory and just run
        // tests for all files that end in `*.md`
        let mut stack = vec![builder.src.join(self.path)];
        let _time = helpers::timeit(builder);
        let mut files = Vec::new();
        while let Some(p) = stack.pop() {
            if p.is_dir() {
                stack.extend(t!(p.read_dir()).map(|p| t!(p).path()));
                continue;
            }

            if p.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }

            files.push(p);
        }

        files.sort();

        for file in files {
            markdown_test(builder, compiler, &file);
        }
    }
}

macro_rules! test_book {
    ($(
        $name:ident, $path:expr, $book_name:expr,
        default=$default:expr
        $(,submodules = $submodules:expr)?
        $(,dependencies=$dependencies:expr)?
        ;
    )+) => {
        $(
            #[derive(Debug, Clone, PartialEq, Eq, Hash)]
            pub struct $name {
                compiler: Compiler,
            }

            impl Step for $name {
                type Output = ();
                const DEFAULT: bool = $default;
                const ONLY_HOSTS: bool = true;

                fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                    run.path($path)
                }

                fn make_run(run: RunConfig<'_>) {
                    run.builder.ensure($name {
                        compiler: run.builder.compiler(run.builder.top_stage, run.target),
                    });
                }

                fn run(self, builder: &Builder<'_>) {
                    $(
                        for submodule in $submodules {
                            builder.require_submodule(submodule, None);
                        }
                    )*

                    let dependencies = vec![];
                    $(
                        let mut dependencies = dependencies;
                        for dep in $dependencies {
                            dependencies.push(dep);
                        }
                    )?

                    builder.ensure(BookTest {
                        compiler: self.compiler,
                        path: PathBuf::from($path),
                        name: $book_name,
                        is_ext_doc: !$default,
                        dependencies,
                    });
                }
            }
        )+
    }
}

test_book!(
    Nomicon, "src/doc/nomicon", "nomicon", default=false, submodules=["src/doc/nomicon"];
    Reference, "src/doc/reference", "reference", default=false, submodules=["src/doc/reference"];
    RustdocBook, "src/doc/rustdoc", "rustdoc", default=true;
    RustcBook, "src/doc/rustc", "rustc", default=true;
    RustByExample, "src/doc/rust-by-example", "rust-by-example", default=false, submodules=["src/doc/rust-by-example"];
    EmbeddedBook, "src/doc/embedded-book", "embedded-book", default=false, submodules=["src/doc/embedded-book"];
    TheBook, "src/doc/book", "book", default=false, submodules=["src/doc/book"], dependencies=["src/doc/book/packages/trpl"];
    UnstableBook, "src/doc/unstable-book", "unstable-book", default=true;
    EditionGuide, "src/doc/edition-guide", "edition-guide", default=false, submodules=["src/doc/edition-guide"];
);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ErrorIndex {
    compiler: Compiler,
}

impl Step for ErrorIndex {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        // Also add `error-index` here since that is what appears in the error message
        // when this fails.
        run.path("src/tools/error_index_generator").alias("error-index")
    }

    fn make_run(run: RunConfig<'_>) {
        // error_index_generator depends on librustdoc. Use the compiler that
        // is normally used to build rustdoc for other tests (like compiletest
        // tests in tests/rustdoc) so that it shares the same artifacts.
        let compiler = run.builder.compiler(run.builder.top_stage, run.builder.config.host_target);
        run.builder.ensure(ErrorIndex { compiler });
    }

    /// Runs the error index generator tool to execute the tests located in the error
    /// index.
    ///
    /// The `error_index_generator` tool lives in `src/tools` and is used to
    /// generate a markdown file from the error indexes of the code base which is
    /// then passed to `rustdoc --test`.
    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;

        let dir = testdir(builder, compiler.host);
        t!(fs::create_dir_all(&dir));
        let output = dir.join("error-index.md");

        let mut tool = tool::ErrorIndex::command(builder);
        tool.arg("markdown").arg(&output);

        let guard =
            builder.msg(Kind::Test, compiler.stage, "error-index", compiler.host, compiler.host);
        let _time = helpers::timeit(builder);
        tool.run_capture(builder);
        drop(guard);
        // The tests themselves need to link to std, so make sure it is
        // available.
        builder.ensure(compile::Std::new(compiler, compiler.host));
        markdown_test(builder, compiler, &output);
    }
}

fn markdown_test(builder: &Builder<'_>, compiler: Compiler, markdown: &Path) -> bool {
    if let Ok(contents) = fs::read_to_string(markdown)
        && !contents.contains("```")
    {
        return true;
    }

    builder.verbose(|| println!("doc tests for: {}", markdown.display()));
    let mut cmd = builder.rustdoc_cmd(compiler);
    builder.add_rust_test_threads(&mut cmd);
    // allow for unstable options such as new editions
    cmd.arg("-Z");
    cmd.arg("unstable-options");
    cmd.arg("--test");
    cmd.arg(markdown);
    cmd.env("RUSTC_BOOTSTRAP", "1");

    let test_args = builder.config.test_args().join(" ");
    cmd.arg("--test-args").arg(test_args);

    cmd = cmd.delay_failure();
    if !builder.config.verbose_tests {
        cmd.run_capture(builder).is_success()
    } else {
        cmd.run(builder)
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

/// Given a `cargo test` subcommand, add the appropriate flags and run it.
///
/// Returns whether the test succeeded.
fn run_cargo_test<'a>(
    cargo: builder::Cargo,
    libtest_args: &[&str],
    crates: &[String],
    description: impl Into<Option<&'a str>>,
    target: TargetSelection,
    builder: &Builder<'_>,
) -> bool {
    let compiler = cargo.compiler();
    let mut cargo = prepare_cargo_test(cargo, libtest_args, crates, target, builder);
    let _time = helpers::timeit(builder);
    let _group = description.into().and_then(|what| {
        builder.msg_sysroot_tool(Kind::Test, compiler.stage, what, compiler.host, target)
    });

    #[cfg(feature = "build-metrics")]
    builder.metrics.begin_test_suite(
        build_helper::metrics::TestSuiteMetadata::CargoPackage {
            crates: crates.iter().map(|c| c.to_string()).collect(),
            target: target.triple.to_string(),
            host: compiler.host.triple.to_string(),
            stage: compiler.stage,
        },
        builder,
    );
    add_flags_and_try_run_tests(builder, &mut cargo)
}

/// Given a `cargo test` subcommand, pass it the appropriate test flags given a `builder`.
fn prepare_cargo_test(
    cargo: builder::Cargo,
    libtest_args: &[&str],
    crates: &[String],
    target: TargetSelection,
    builder: &Builder<'_>,
) -> BootstrapCommand {
    let compiler = cargo.compiler();
    let mut cargo: BootstrapCommand = cargo.into();

    // Propagate `--bless` if it has not already been set/unset
    // Any tools that want to use this should bless if `RUSTC_BLESS` is set to
    // anything other than `0`.
    if builder.config.cmd.bless() && !cargo.get_envs().any(|v| v.0 == "RUSTC_BLESS") {
        cargo.env("RUSTC_BLESS", "Gesundheit");
    }

    // Pass in some standard flags then iterate over the graph we've discovered
    // in `cargo metadata` with the maps above and figure out what `-p`
    // arguments need to get passed.
    if builder.kind == Kind::Test && !builder.fail_fast {
        cargo.arg("--no-fail-fast");
    }

    if builder.config.json_output {
        cargo.arg("--message-format=json");
    }

    match builder.doc_tests {
        DocTests::Only => {
            cargo.arg("--doc");
        }
        DocTests::No => {
            cargo.args(["--bins", "--examples", "--tests", "--benches"]);
        }
        DocTests::Yes => {}
    }

    for krate in crates {
        cargo.arg("-p").arg(krate);
    }

    cargo.arg("--").args(builder.config.test_args()).args(libtest_args);
    if !builder.config.verbose_tests {
        cargo.arg("--quiet");
    }

    // The tests are going to run with the *target* libraries, so we need to
    // ensure that those libraries show up in the LD_LIBRARY_PATH equivalent.
    //
    // Note that to run the compiler we need to run with the *host* libraries,
    // but our wrapper scripts arrange for that to be the case anyway.
    //
    // We skip everything on Miri as then this overwrites the libdir set up
    // by `Cargo::new` and that actually makes things go wrong.
    if builder.kind != Kind::Miri {
        let mut dylib_paths = builder.rustc_lib_paths(compiler);
        dylib_paths.push(PathBuf::from(&builder.sysroot_target_libdir(compiler, target)));
        helpers::add_dylib_path(dylib_paths, &mut cargo);
    }

    if builder.remote_tested(target) {
        cargo.env(
            format!("CARGO_TARGET_{}_RUNNER", envify(&target.triple)),
            format!("{} run 0", builder.tool_exe(Tool::RemoteTestClient).display()),
        );
    } else if let Some(tool) = builder.runner(target) {
        cargo.env(format!("CARGO_TARGET_{}_RUNNER", envify(&target.triple)), tool);
    }

    cargo
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
        run.crate_or_deps("sysroot").crate_or_deps("coretests").crate_or_deps("alloctests")
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
            if !builder.config.is_host_target(target) {
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
                }
            }
            Mode::Rustc => {
                compile::rustc_cargo(builder, &mut cargo, target, &compiler, &self.crates);
            }
            _ => panic!("can only test libraries"),
        };

        let mut crates = self.crates.clone();
        // The core and alloc crates can't directly be tested. We
        // could silently ignore them, but adding their own test
        // crates is less confusing for users. We still keep core and
        // alloc themself for doctests
        if crates.iter().any(|crate_| crate_ == "core") {
            crates.push("coretests".to_owned());
        }
        if crates.iter().any(|crate_| crate_ == "alloc") {
            crates.push("alloctests".to_owned());
        }

        run_cargo_test(cargo, &[], &crates, &*crate_description(&self.crates), target, builder);
    }
}

/// Rustdoc is special in various ways, which is why this step is different from `Crate`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateRustdoc {
    host: TargetSelection,
}

impl Step for CrateRustdoc {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.paths(&["src/librustdoc", "src/tools/rustdoc"])
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;

        builder.ensure(CrateRustdoc { host: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let target = self.host;

        let compiler = if builder.download_rustc() {
            builder.compiler(builder.top_stage, target)
        } else {
            // Use the previous stage compiler to reuse the artifacts that are
            // created when running compiletest for tests/rustdoc. If this used
            // `compiler`, then it would cause rustdoc to be built *again*, which
            // isn't really necessary.
            builder.compiler_for(builder.top_stage, target, target)
        };
        // NOTE: normally `ensure(Rustc)` automatically runs `ensure(Std)` for us. However, when
        // using `download-rustc`, the rustc_private artifacts may be in a *different sysroot* from
        // the target rustdoc (`ci-rustc-sysroot` vs `stage2`). In that case, we need to ensure this
        // explicitly to make sure it ends up in the stage2 sysroot.
        builder.ensure(compile::Std::new(compiler, target));
        builder.ensure(compile::Rustc::new(compiler, target));

        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            target,
            builder.kind,
            "src/tools/rustdoc",
            SourceType::InTree,
            &[],
        );
        if self.host.contains("musl") {
            cargo.arg("'-Ctarget-feature=-crt-static'");
        }

        // This is needed for running doctests on librustdoc. This is a bit of
        // an unfortunate interaction with how bootstrap works and how cargo
        // sets up the dylib path, and the fact that the doctest (in
        // html/markdown.rs) links to rustc-private libs. For stage1, the
        // compiler host dylibs (in stage1/lib) are not the same as the target
        // dylibs (in stage1/lib/rustlib/...). This is different from a normal
        // rust distribution where they are the same.
        //
        // On the cargo side, normal tests use `target_process` which handles
        // setting up the dylib for a *target* (stage1/lib/rustlib/... in this
        // case). However, for doctests it uses `rustdoc_process` which only
        // sets up the dylib path for the *host* (stage1/lib), which is the
        // wrong directory.
        //
        // Recall that we special-cased `compiler_for(top_stage)` above, so we always use stage1.
        //
        // It should be considered to just stop running doctests on
        // librustdoc. There is only one test, and it doesn't look too
        // important. There might be other ways to avoid this, but it seems
        // pretty convoluted.
        //
        // See also https://github.com/rust-lang/rust/issues/13983 where the
        // host vs target dylibs for rustdoc are consistently tricky to deal
        // with.
        //
        // Note that this set the host libdir for `download_rustc`, which uses a normal rust distribution.
        let libdir = if builder.download_rustc() {
            builder.rustc_libdir(compiler)
        } else {
            builder.sysroot_target_libdir(compiler, target).to_path_buf()
        };
        let mut dylib_path = dylib_path();
        dylib_path.insert(0, PathBuf::from(&*libdir));
        cargo.env(dylib_path_var(), env::join_paths(&dylib_path).unwrap());

        run_cargo_test(cargo, &[], &["rustdoc:0.0.0".to_string()], "rustdoc", target, builder);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateRustdocJsonTypes {
    host: TargetSelection,
}

impl Step for CrateRustdocJsonTypes {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/rustdoc-json-types")
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;

        builder.ensure(CrateRustdocJsonTypes { host: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let target = self.host;

        // Use the previous stage compiler to reuse the artifacts that are
        // created when running compiletest for tests/rustdoc. If this used
        // `compiler`, then it would cause rustdoc to be built *again*, which
        // isn't really necessary.
        let compiler = builder.compiler_for(builder.top_stage, target, target);
        builder.ensure(compile::Rustc::new(compiler, target));

        let cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            target,
            builder.kind,
            "src/rustdoc-json-types",
            SourceType::InTree,
            &[],
        );

        // FIXME: this looks very wrong, libtest doesn't accept `-C` arguments and the quotes are fishy.
        let libtest_args = if self.host.contains("musl") {
            ["'-Ctarget-feature=-crt-static'"].as_slice()
        } else {
            &[]
        };

        run_cargo_test(
            cargo,
            libtest_args,
            &["rustdoc-json-types".to_string()],
            "rustdoc-json-types",
            target,
            builder,
        );
    }
}

/// Some test suites are run inside emulators or on remote devices, and most
/// of our test binaries are linked dynamically which means we need to ship
/// the standard library and such to the emulator ahead of time. This step
/// represents this and is a dependency of all test suites.
///
/// Most of the time this is a no-op. For some steps such as shipping data to
/// QEMU we have to build our own tools so we've got conditional dependencies
/// on those programs as well. Note that the remote test client is built for
/// the build target (us) and the server is built for the target.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RemoteCopyLibs {
    compiler: Compiler,
    target: TargetSelection,
}

impl Step for RemoteCopyLibs {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target = self.target;
        if !builder.remote_tested(target) {
            return;
        }

        builder.ensure(compile::Std::new(compiler, target));

        builder.info(&format!("REMOTE copy libs to emulator ({target})"));

        let remote_test_server = builder.ensure(tool::RemoteTestServer { compiler, target });

        // Spawn the emulator and wait for it to come online
        let tool = builder.tool_exe(Tool::RemoteTestClient);
        let mut cmd = command(&tool);
        cmd.arg("spawn-emulator")
            .arg(target.triple)
            .arg(&remote_test_server.tool_path)
            .arg(builder.tempdir());
        if let Some(rootfs) = builder.qemu_rootfs(target) {
            cmd.arg(rootfs);
        }
        cmd.run(builder);

        // Push all our dylibs to the emulator
        for f in t!(builder.sysroot_target_libdir(compiler, target).read_dir()) {
            let f = t!(f);
            if helpers::is_dylib(&f.path()) {
                command(&tool).arg("push").arg(f.path()).run(builder);
            }
        }
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

    /// Runs `distcheck`, which is a collection of smoke tests:
    ///
    /// - Run `make check` from an unpacked dist tarball to make sure we can at the minimum run
    ///   check steps from those sources.
    /// - Check that selected dist components (`rust-src` only at the moment) at least have expected
    ///   directory shape and crate manifests that cargo can generate a lockfile from.
    ///
    /// FIXME(#136822): dist components are under-tested.
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
        command(helpers::make(&builder.config.host_target.triple))
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
pub struct Bootstrap;

impl Step for Bootstrap {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    /// Tests the build system itself.
    fn run(self, builder: &Builder<'_>) {
        let host = builder.config.host_target;
        let compiler = builder.compiler(0, host);
        let _guard = builder.msg(Kind::Test, 0, "bootstrap", host, host);

        // Some tests require cargo submodule to be present.
        builder.build.require_submodule("src/tools/cargo", None);

        let mut check_bootstrap = command(builder.python());
        check_bootstrap
            .args(["-m", "unittest", "bootstrap_test.py"])
            .env("BUILD_DIR", &builder.out)
            .env("BUILD_PLATFORM", builder.build.host_target.triple)
            .env("BOOTSTRAP_TEST_RUSTC_BIN", &builder.initial_rustc)
            .env("BOOTSTRAP_TEST_CARGO_BIN", &builder.initial_cargo)
            .current_dir(builder.src.join("src/bootstrap/"));
        // NOTE: we intentionally don't pass test_args here because the args for unittest and cargo test are mutually incompatible.
        // Use `python -m unittest` manually if you want to pass arguments.
        check_bootstrap.delay_failure().run(builder);

        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolBootstrap,
            host,
            Kind::Test,
            "src/bootstrap",
            SourceType::InTree,
            &[],
        );

        cargo.release_build(false);

        cargo
            .rustflag("-Cdebuginfo=2")
            .env("CARGO_TARGET_DIR", builder.out.join("bootstrap"))
            // Needed for insta to correctly write pending snapshots to the right directories.
            .env("INSTA_WORKSPACE_ROOT", &builder.src)
            .env("RUSTC_BOOTSTRAP", "1");

        // bootstrap tests are racy on directory creation so just run them one at a time.
        // Since there's not many this shouldn't be a problem.
        run_cargo_test(cargo, &["--test-threads=1"], &[], None, host, builder);
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/bootstrap")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Bootstrap);
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
        let compiler = run.builder.compiler_for(
            run.builder.top_stage,
            run.builder.build.host_target,
            run.target,
        );
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
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.host_target),
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
        let bootstrap_host = builder.config.host_target;
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
        run_cargo_test(cargo, &[], &[], None, bootstrap_host, builder);

        // We currently don't support running the test.sh script outside linux(?) environments.
        // Eventually this should likely migrate to #[test]s in rust-installer proper rather than a
        // set of scripts, which will likely allow dropping this if.
        if bootstrap_host != "x86_64-unknown-linux-gnu" {
            return;
        }

        let mut cmd = command(builder.src.join("src/tools/rust-installer/test.sh"));
        let tmpdir = testdir(builder, compiler.host).join("rust-installer");
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
pub struct TestHelpers {
    pub target: TargetSelection,
}

impl Step for TestHelpers {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("tests/auxiliary/rust_test_helpers.c")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(TestHelpers { target: run.target })
    }

    /// Compiles the `rust_test_helpers.c` library which we used in various
    /// `run-pass` tests for ABI testing.
    fn run(self, builder: &Builder<'_>) {
        if builder.config.dry_run() {
            return;
        }
        // The x86_64-fortanix-unknown-sgx target doesn't have a working C
        // toolchain. However, some x86_64 ELF objects can be linked
        // without issues. Use this hack to compile the test helpers.
        let target = if self.target == "x86_64-fortanix-unknown-sgx" {
            TargetSelection::from_user("x86_64-unknown-linux-gnu")
        } else {
            self.target
        };
        let dst = builder.test_helpers_out(target);
        let src = builder.src.join("tests/auxiliary/rust_test_helpers.c");
        if up_to_date(&src, &dst.join("librust_test_helpers.a")) {
            return;
        }

        let _guard = builder.msg_unstaged(Kind::Build, "test helpers", target);
        t!(fs::create_dir_all(&dst));
        let mut cfg = cc::Build::new();

        // We may have found various cross-compilers a little differently due to our
        // extra configuration, so inform cc of these compilers. Note, though, that
        // on MSVC we still need cc's detection of env vars (ugh).
        if !target.is_msvc() {
            if let Some(ar) = builder.ar(target) {
                cfg.archiver(ar);
            }
            cfg.compiler(builder.cc(target));
        }
        cfg.cargo_metadata(false)
            .out_dir(&dst)
            .target(&target.triple)
            .host(&builder.config.host_target.triple)
            .opt_level(0)
            .warnings(false)
            .debug(false)
            .file(builder.src.join("tests/auxiliary/rust_test_helpers.c"))
            .compile("rust_test_helpers");
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
        #[expect(deprecated)]
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

        let gcc = builder.ensure(Gcc { target });

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
            add_cg_gcc_cargo_flags(&mut cargo, &gcc);

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
        #[expect(deprecated)]
        builder.config.try_run(&mut prepare_cargo.into()).unwrap();
        */

        let mut cargo = build_cargo();

        cargo
            // cg_gcc's build system ignores RUSTFLAGS. pass some flags through CG_RUSTFLAGS instead.
            .env("CG_RUSTFLAGS", "-Alinker-messages")
            .arg("--")
            .arg("test")
            .arg("--use-backend")
            .arg("gcc")
            .arg("--gcc-path")
            .arg(gcc.libgccjit.parent().unwrap())
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
/// - Runs `cargo test` for the `src/tools/test-float-parse` tool.
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
        run.path("src/tools/test-float-parse")
    }

    fn make_run(run: RunConfig<'_>) {
        for path in run.paths {
            let path = path.assert_single_path().path.clone();
            run.builder.ensure(Self { path, host: run.target });
        }
    }

    fn run(self, builder: &Builder<'_>) {
        let bootstrap_host = builder.config.host_target;
        let compiler = builder.compiler(builder.top_stage, bootstrap_host);
        let path = self.path.to_str().unwrap();
        let crate_name = self.path.iter().next_back().unwrap().to_str().unwrap();

        builder.ensure(tool::TestFloatParse { host: self.host });

        // Run any unit tests in the crate
        let mut cargo_test = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolStd,
            bootstrap_host,
            Kind::Test,
            path,
            SourceType::InTree,
            &[],
        );
        cargo_test.allow_features(tool::TestFloatParse::ALLOW_FEATURES);

        run_cargo_test(cargo_test, &[], &[], crate_name, bootstrap_host, builder);

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
        cargo_run.allow_features(tool::TestFloatParse::ALLOW_FEATURES);

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
