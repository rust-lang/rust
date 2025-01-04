use std::ffi::OsString;
use std::path::PathBuf;
use std::{env, fs, iter};

use super::shared::{prepare_cargo_test, run_cargo_test, testdir};
use crate::core::build_steps::compile;
use crate::core::build_steps::tool::{self, SourceType};
use crate::core::builder::{Builder, Compiler, Kind, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::helpers::{self, dylib_path, dylib_path_var};
use crate::utils::render_tests::add_flags_and_try_run_tests;
use crate::{Mode, t};

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
        run.path("src/tools/jsondoclint")
            .path("src/tools/suggest-tests")
            .path("src/tools/replace-version-placeholder")
            .alias("tidyselftest")
    }

    fn make_run(run: RunConfig<'_>) {
        for path in run.paths {
            let path = path.assert_single_path().path.clone();
            run.builder.ensure(CrateBootstrap { host: run.target, path });
        }
    }

    fn run(self, builder: &Builder<'_>) {
        let bootstrap_host = builder.config.build;
        let compiler = builder.compiler(0, bootstrap_host);
        let mut path = self.path.to_str().unwrap();
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
        run_cargo_test(cargo, &[], &[], crate_name, crate_name, compiler, bootstrap_host, builder);
    }
}

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
        let mut cargo = prepare_cargo_test(cargo, &[], &[], "cargo", compiler, self.host, builder);

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
                stage: self.stage,
            },
            builder,
        );

        let _time = helpers::timeit(builder);
        add_flags_and_try_run_tests(builder, &mut cargo);
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
        run_cargo_test(cargo, &[], &[], "rust-analyzer", "rust-analyzer", compiler, host, builder);
    }
}

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

        let dir = testdir(builder, compiler.host);
        t!(fs::create_dir_all(&dir));
        cargo.env("RUSTFMT_TEST_DIR", dir);

        cargo.add_rustc_lib_path(builder);

        run_cargo_test(cargo, &[], &[], "rustfmt", "rustfmt", compiler, host, builder);
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
        cargo.allow_features("test");
        run_cargo_test(
            cargo,
            &[],
            &[],
            "compiletest",
            "compiletest self test",
            compiler,
            host,
            builder,
        );
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
        let cargo = prepare_cargo_test(cargo, &[], &[], "clippy", compiler, host, builder);

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
        run_cargo_test(
            cargo,
            &[],
            &[],
            "run-make-support",
            "run-make-support self test",
            compiler,
            host,
            builder,
        );
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
        run_cargo_test(
            cargo,
            &[],
            &[],
            "build_helper",
            "build_helper self test",
            compiler,
            host,
            builder,
        );
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

        run_cargo_test(
            cargo,
            &[],
            &["rustdoc:0.0.0".to_string()],
            "rustdoc",
            "rustdoc",
            compiler,
            target,
            builder,
        );
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
            "rustdoc-json-types",
            compiler,
            target,
            builder,
        );
    }
}
