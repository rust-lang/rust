//! Test suites of the various devtools (cargo, rustfmt, rust-analyzer, etc.). Note that rustdoc is
//! not here because it has special logic that isn't required by other devtool tests.

use std::ffi::OsString;
use std::{env, fs, iter};

use super::test_helpers::{prepare_cargo_test, run_cargo_test};
use crate::core::build_steps::compile;
use crate::core::build_steps::tool::{self, SourceType};
use crate::core::builder::{Builder, Compiler, Kind, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::helpers;
use crate::utils::render_tests::add_flags_and_try_run_tests;
use crate::{Mode, t};

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
        run.builder.ensure(Cargo { stage: run.builder.top_stage, host: run.target });
    }

    /// Runs `cargo test` for `cargo` packaged with Rust.
    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(self.stage, self.host);

        builder.ensure(tool::Cargo { compiler, target: self.host });
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
        let mut cargo = prepare_cargo_test(cargo, &[], &[], "cargo", self.host, builder);

        // Don't run cross-compile tests, we may not have cross-compiled libstd libs
        // available.
        cargo.env("CFG_DISABLE_CROSS_TESTS", "1");
        // Forcibly disable tests using nightly features since any changes to
        // those features won't be able to land.
        cargo.env("CARGO_TEST_DISABLE_NIGHTLY", "1");

        fn path_for_cargo(builder: &Builder<'_>, compiler: Compiler) -> OsString {
            // Configure PATH to find the right rustc. NB. we have to use PATH
            // and not RUSTC because the Cargo test suite has tests that will
            // fail if rustc is not spelled `rustc`.
            let path = builder.sysroot(compiler).join("bin");
            let old_path = env::var_os("PATH").unwrap_or_default();
            env::join_paths(iter::once(path).chain(env::split_paths(&old_path))).expect("")
        }

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
                stage: self.stage,
            },
            builder,
        );

        let _time = helpers::timeit(builder);
        add_flags_and_try_run_tests(builder, &mut cargo);
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

        builder.ensure(tool::Rustfmt { compiler, target: self.host });

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

        let dir = builder.test_out(compiler.host);
        t!(fs::create_dir_all(&dir));
        cargo.env("RUSTFMT_TEST_DIR", dir);

        cargo.add_rustc_lib_path(builder);

        run_cargo_test(cargo, &[], &[], "rustfmt", "rustfmt", host, builder);
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
        run_cargo_test(cargo, &[], &[], "rust-analyzer", "rust-analyzer", host, builder);
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
        run.path("src/tools/clippy")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Clippy { stage: run.builder.top_stage, host: run.target });
    }

    /// Runs `cargo test` for clippy.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let host = self.host;
        let compiler = builder.compiler(stage, host);

        builder.ensure(tool::Clippy { compiler, target: self.host });
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

        cargo.add_rustc_lib_path(builder);
        let cargo = prepare_cargo_test(cargo, &[], &[], "clippy", host, builder);

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
