//! Implementation of the test-related targets of the build system.
//!
//! This file implements the various regression test suites that we execute on
//! our CI.

use std::env;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::iter;
use std::path::{Path, PathBuf};
use std::process::Command;

use build_helper::{self, output, t};

use crate::builder::{Builder, Compiler, Kind, RunConfig, ShouldRun, Step};
use crate::cache::{Interned, INTERNER};
use crate::compile;
use crate::config::TargetSelection;
use crate::dist;
use crate::flags::Subcommand;
use crate::native;
use crate::tool::{self, SourceType, Tool};
use crate::toolstate::ToolState;
use crate::util::{self, add_link_lib_path, dylib_path, dylib_path_var};
use crate::Crate as CargoCrate;
use crate::{envify, DocTests, GitRepo, Mode};

const ADB_TEST_DIR: &str = "/data/tmp/work";

/// The two modes of the test runner; tests or benchmarks.
#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone, PartialOrd, Ord)]
pub enum TestKind {
    /// Run `cargo test`.
    Test,
    /// Run `cargo bench`.
    Bench,
}

impl From<Kind> for TestKind {
    fn from(kind: Kind) -> Self {
        match kind {
            Kind::Test => TestKind::Test,
            Kind::Bench => TestKind::Bench,
            _ => panic!("unexpected kind in crate: {:?}", kind),
        }
    }
}

impl TestKind {
    // Return the cargo subcommand for this test kind
    fn subcommand(self) -> &'static str {
        match self {
            TestKind::Test => "test",
            TestKind::Bench => "bench",
        }
    }
}

impl fmt::Display for TestKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            TestKind::Test => "Testing",
            TestKind::Bench => "Benchmarking",
        })
    }
}

fn try_run(builder: &Builder<'_>, cmd: &mut Command) -> bool {
    if !builder.fail_fast {
        if !builder.try_run(cmd) {
            let mut failures = builder.delayed_failures.borrow_mut();
            failures.push(format!("{:?}", cmd));
            return false;
        }
    } else {
        builder.run(cmd);
    }
    true
}

fn try_run_quiet(builder: &Builder<'_>, cmd: &mut Command) -> bool {
    if !builder.fail_fast {
        if !builder.try_run_quiet(cmd) {
            let mut failures = builder.delayed_failures.borrow_mut();
            failures.push(format!("{:?}", cmd));
            return false;
        }
    } else {
        builder.run_quiet(cmd);
    }
    true
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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

        builder.info(&format!("Linkcheck ({})", host));

        builder.default_doc(None);

        let _time = util::timeit(&builder);
        try_run(
            builder,
            builder.tool_cmd(Tool::Linkchecker).arg(builder.out.join(host.triple).join("doc")),
        );
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/tools/linkchecker").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Linkcheck { host: run.target });
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
        builder.ensure(compile::Rustc { compiler, target: compiler.host });
        let cargo = builder.ensure(tool::Cargo { compiler, target: compiler.host });

        // Note that this is a short, cryptic, and not scoped directory name. This
        // is currently to minimize the length of path on Windows where we otherwise
        // quickly run into path name limit constraints.
        let out_dir = builder.out.join("ct");
        t!(fs::create_dir_all(&out_dir));

        let _time = util::timeit(&builder);
        let mut cmd = builder.tool_cmd(Tool::CargoTest);
        try_run(
            builder,
            cmd.arg(&cargo)
                .arg(&out_dir)
                .env("RUSTC", builder.rustc(compiler))
                .env("RUSTDOC", builder.rustdoc(compiler)),
        );
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Cargo {
    stage: u32,
    host: TargetSelection,
}

impl Step for Cargo {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/cargo")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Cargo { stage: run.builder.top_stage, host: run.target });
    }

    /// Runs `cargo test` for `cargo` packaged with Rust.
    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(self.stage, self.host);

        builder.ensure(tool::Cargo { compiler, target: self.host });
        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            self.host,
            "test",
            "src/tools/cargo",
            SourceType::Submodule,
            &[],
        );

        if !builder.fail_fast {
            cargo.arg("--no-fail-fast");
        }

        // Don't run cross-compile tests, we may not have cross-compiled libstd libs
        // available.
        cargo.env("CFG_DISABLE_CROSS_TESTS", "1");
        // Disable a test that has issues with mingw.
        cargo.env("CARGO_TEST_DISABLE_GIT_CLI", "1");
        // Forcibly disable tests using nightly features since any changes to
        // those features won't be able to land.
        cargo.env("CARGO_TEST_DISABLE_NIGHTLY", "1");

        cargo.env("PATH", &path_for_cargo(builder, compiler));

        try_run(builder, &mut cargo.into());
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Rls {
    stage: u32,
    host: TargetSelection,
}

impl Step for Rls {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/rls")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rls { stage: run.builder.top_stage, host: run.target });
    }

    /// Runs `cargo test` for the rls.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let host = self.host;
        let compiler = builder.compiler(stage, host);

        let build_result =
            builder.ensure(tool::Rls { compiler, target: self.host, extra_features: Vec::new() });
        if build_result.is_none() {
            eprintln!("failed to test rls: could not build");
            return;
        }

        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            host,
            "test",
            "src/tools/rls",
            SourceType::Submodule,
            &[],
        );

        cargo.add_rustc_lib_path(builder, compiler);
        cargo.arg("--").args(builder.config.cmd.test_args());

        if try_run(builder, &mut cargo.into()) {
            builder.save_toolstate("rls", ToolState::TestPass);
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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

        let build_result = builder.ensure(tool::Rustfmt {
            compiler,
            target: self.host,
            extra_features: Vec::new(),
        });
        if build_result.is_none() {
            eprintln!("failed to test rustfmt: could not build");
            return;
        }

        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            host,
            "test",
            "src/tools/rustfmt",
            SourceType::Submodule,
            &[],
        );

        let dir = testdir(builder, compiler.host);
        t!(fs::create_dir_all(&dir));
        cargo.env("RUSTFMT_TEST_DIR", dir);

        cargo.add_rustc_lib_path(builder, compiler);

        if try_run(builder, &mut cargo.into()) {
            builder.save_toolstate("rustfmt", ToolState::TestPass);
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Miri {
    stage: u32,
    host: TargetSelection,
}

impl Step for Miri {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/miri")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Miri { stage: run.builder.top_stage, host: run.target });
    }

    /// Runs `cargo test` for miri.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let host = self.host;
        let compiler = builder.compiler(stage, host);

        let miri =
            builder.ensure(tool::Miri { compiler, target: self.host, extra_features: Vec::new() });
        let cargo_miri = builder.ensure(tool::CargoMiri {
            compiler,
            target: self.host,
            extra_features: Vec::new(),
        });
        if let (Some(miri), Some(_cargo_miri)) = (miri, cargo_miri) {
            let mut cargo =
                builder.cargo(compiler, Mode::ToolRustc, SourceType::Submodule, host, "install");
            cargo.arg("xargo");
            // Configure `cargo install` path. cargo adds a `bin/`.
            cargo.env("CARGO_INSTALL_ROOT", &builder.out);

            let mut cargo = Command::from(cargo);
            if !try_run(builder, &mut cargo) {
                return;
            }

            // # Run `cargo miri setup`.
            let mut cargo = tool::prepare_tool_cargo(
                builder,
                compiler,
                Mode::ToolRustc,
                host,
                "run",
                "src/tools/miri/cargo-miri",
                SourceType::Submodule,
                &[],
            );
            cargo.arg("--").arg("miri").arg("setup");

            // Tell `cargo miri setup` where to find the sources.
            cargo.env("XARGO_RUST_SRC", builder.src.join("library"));
            // Tell it where to find Miri.
            cargo.env("MIRI", &miri);
            // Debug things.
            cargo.env("RUST_BACKTRACE", "1");
            // Let cargo-miri know where xargo ended up.
            cargo.env("XARGO_CHECK", builder.out.join("bin").join("xargo-check"));

            let mut cargo = Command::from(cargo);
            if !try_run(builder, &mut cargo) {
                return;
            }

            // # Determine where Miri put its sysroot.
            // To this end, we run `cargo miri setup --print-sysroot` and capture the output.
            // (We do this separately from the above so that when the setup actually
            // happens we get some output.)
            // We re-use the `cargo` from above.
            cargo.arg("--print-sysroot");

            // FIXME: Is there a way in which we can re-use the usual `run` helpers?
            let miri_sysroot = if builder.config.dry_run {
                String::new()
            } else {
                builder.verbose(&format!("running: {:?}", cargo));
                let out = cargo
                    .output()
                    .expect("We already ran `cargo miri setup` before and that worked");
                assert!(out.status.success(), "`cargo miri setup` returned with non-0 exit code");
                // Output is "<sysroot>\n".
                let stdout = String::from_utf8(out.stdout)
                    .expect("`cargo miri setup` stdout is not valid UTF-8");
                let sysroot = stdout.trim_end();
                builder.verbose(&format!("`cargo miri setup --print-sysroot` said: {:?}", sysroot));
                sysroot.to_owned()
            };

            // # Run `cargo test`.
            let mut cargo = tool::prepare_tool_cargo(
                builder,
                compiler,
                Mode::ToolRustc,
                host,
                "test",
                "src/tools/miri",
                SourceType::Submodule,
                &[],
            );

            // miri tests need to know about the stage sysroot
            cargo.env("MIRI_SYSROOT", miri_sysroot);
            cargo.env("RUSTC_LIB_PATH", builder.rustc_libdir(compiler));
            cargo.env("MIRI", miri);

            cargo.arg("--").args(builder.config.cmd.test_args());

            cargo.add_rustc_lib_path(builder, compiler);

            if !try_run(builder, &mut cargo.into()) {
                return;
            }

            // # Done!
            builder.save_toolstate("miri", ToolState::TestPass);
        } else {
            eprintln!("failed to test miri: could not build");
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
        let compiler = builder.compiler(0, host);

        // We need `ToolStd` for the locally-built sysroot because
        // compiletest uses unstable features of the `test` crate.
        builder.ensure(compile::Std { compiler, target: host });
        let cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolStd,
            host,
            "test",
            "src/tools/compiletest",
            SourceType::InTree,
            &[],
        );

        try_run(builder, &mut cargo.into());
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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

        let clippy = builder
            .ensure(tool::Clippy { compiler, target: self.host, extra_features: Vec::new() })
            .expect("in-tree tool");
        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            host,
            "test",
            "src/tools/clippy",
            SourceType::InTree,
            &[],
        );

        // clippy tests need to know about the stage sysroot
        cargo.env("SYSROOT", builder.sysroot(compiler));
        cargo.env("RUSTC_TEST_SUITE", builder.rustc(compiler));
        cargo.env("RUSTC_LIB_PATH", builder.rustc_libdir(compiler));
        let host_libs = builder.stage_out(compiler, Mode::ToolRustc).join(builder.cargo_dir());
        let target_libs = builder
            .stage_out(compiler, Mode::ToolRustc)
            .join(&self.host.triple)
            .join(builder.cargo_dir());
        cargo.env("HOST_LIBS", host_libs);
        cargo.env("TARGET_LIBS", target_libs);
        // clippy tests need to find the driver
        cargo.env("CLIPPY_DRIVER_PATH", clippy);

        cargo.arg("--").args(builder.config.cmd.test_args());

        cargo.add_rustc_lib_path(builder, compiler);

        builder.run(&mut cargo.into());
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

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
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
        let rustdoc = builder.out.join("bootstrap/debug/rustdoc");
        let mut cmd = builder.tool_cmd(Tool::RustdocTheme);
        cmd.arg(rustdoc.to_str().unwrap())
            .arg(builder.src.join("src/librustdoc/html/static/themes").to_str().unwrap())
            .env("RUSTC_STAGE", self.compiler.stage.to_string())
            .env("RUSTC_SYSROOT", builder.sysroot(self.compiler))
            .env("RUSTDOC_LIBDIR", builder.sysroot_libdir(self.compiler, self.compiler.host))
            .env("CFG_RELEASE_CHANNEL", &builder.config.channel)
            .env("RUSTDOC_REAL", builder.rustdoc(self.compiler))
            .env("RUSTC_BOOTSTRAP", "1");
        if let Some(linker) = builder.linker(self.compiler.host) {
            cmd.env("RUSTDOC_LINKER", linker);
        }
        if builder.is_fuse_ld_lld(self.compiler.host) {
            cmd.env("RUSTDOC_FUSE_LD_LLD", "1");
        }
        try_run(builder, &mut cmd);
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct RustdocJSStd {
    pub target: TargetSelection,
}

impl Step for RustdocJSStd {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/test/rustdoc-js-std")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustdocJSStd { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        if let Some(ref nodejs) = builder.config.nodejs {
            let mut command = Command::new(nodejs);
            command
                .arg(builder.src.join("src/tools/rustdoc-js/tester.js"))
                .arg("--crate-name")
                .arg("std")
                .arg("--resource-suffix")
                .arg(&builder.version)
                .arg("--doc-folder")
                .arg(builder.doc_out(self.target))
                .arg("--test-folder")
                .arg(builder.src.join("src/test/rustdoc-js-std"));
            builder.ensure(crate::doc::Std { target: self.target, stage: builder.top_stage });
            builder.run(&mut command);
        } else {
            builder.info("No nodejs found, skipping \"src/test/rustdoc-js-std\" tests");
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct RustdocJSNotStd {
    pub target: TargetSelection,
    pub compiler: Compiler,
}

impl Step for RustdocJSNotStd {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/test/rustdoc-js")
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler = run.builder.compiler(run.builder.top_stage, run.build_triple());
        run.builder.ensure(RustdocJSNotStd { target: run.target, compiler });
    }

    fn run(self, builder: &Builder<'_>) {
        if builder.config.nodejs.is_some() {
            builder.ensure(Compiletest {
                compiler: self.compiler,
                target: self.target,
                mode: "js-doc-test",
                suite: "rustdoc-js",
                path: "src/test/rustdoc-js",
                compare_mode: None,
            });
        } else {
            builder.info("No nodejs found, skipping \"src/test/rustdoc-js\" tests");
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
        if builder.is_verbose() {
            cmd.arg("--verbose");
        }

        builder.info("tidy check");
        try_run(builder, &mut cmd);

        if builder.config.channel == "dev" || builder.config.channel == "nightly" {
            builder.info("fmt check");
            crate::format::format(&builder.build, !builder.config.cmd.bless());
        }
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/tidy")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Tidy);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ExpandYamlAnchors;

impl Step for ExpandYamlAnchors {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    /// Ensure the `generate-ci-config` tool was run locally.
    ///
    /// The tool in `src/tools` reads the CI definition in `src/ci/builders.yml` and generates the
    /// appropriate configuration for all our CI providers. This step ensures the tool was called
    /// by the user before committing CI changes.
    fn run(self, builder: &Builder<'_>) {
        builder.info("Ensuring the YAML anchors in the GitHub Actions config were expanded");
        try_run(
            builder,
            &mut builder.tool_cmd(Tool::ExpandYamlAnchors).arg("check").arg(&builder.src),
        );
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/expand-yaml-anchors")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(ExpandYamlAnchors);
    }
}

fn testdir(builder: &Builder<'_>, host: TargetSelection) -> PathBuf {
    builder.out.join(host.triple).join("test")
}

macro_rules! default_test {
    ($name:ident { path: $path:expr, mode: $mode:expr, suite: $suite:expr }) => {
        test!($name { path: $path, mode: $mode, suite: $suite, default: true, host: false });
    };
}

macro_rules! default_test_with_compare_mode {
    ($name:ident { path: $path:expr, mode: $mode:expr, suite: $suite:expr,
                   compare_mode: $compare_mode:expr }) => {
        test_with_compare_mode!($name {
            path: $path,
            mode: $mode,
            suite: $suite,
            default: true,
            host: false,
            compare_mode: $compare_mode
        });
    };
}

macro_rules! host_test {
    ($name:ident { path: $path:expr, mode: $mode:expr, suite: $suite:expr }) => {
        test!($name { path: $path, mode: $mode, suite: $suite, default: true, host: true });
    };
}

macro_rules! test {
    ($name:ident { path: $path:expr, mode: $mode:expr, suite: $suite:expr, default: $default:expr,
                   host: $host:expr }) => {
        test_definitions!($name {
            path: $path,
            mode: $mode,
            suite: $suite,
            default: $default,
            host: $host,
            compare_mode: None
        });
    };
}

macro_rules! test_with_compare_mode {
    ($name:ident { path: $path:expr, mode: $mode:expr, suite: $suite:expr, default: $default:expr,
                   host: $host:expr, compare_mode: $compare_mode:expr }) => {
        test_definitions!($name {
            path: $path,
            mode: $mode,
            suite: $suite,
            default: $default,
            host: $host,
            compare_mode: Some($compare_mode)
        });
    };
}

macro_rules! test_definitions {
    ($name:ident {
        path: $path:expr,
        mode: $mode:expr,
        suite: $suite:expr,
        default: $default:expr,
        host: $host:expr,
        compare_mode: $compare_mode:expr
    }) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
        pub struct $name {
            pub compiler: Compiler,
            pub target: TargetSelection,
        }

        impl Step for $name {
            type Output = ();
            const DEFAULT: bool = $default;
            const ONLY_HOSTS: bool = $host;

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
                    compare_mode: $compare_mode,
                })
            }
        }
    };
}

default_test_with_compare_mode!(Ui {
    path: "src/test/ui",
    mode: "ui",
    suite: "ui",
    compare_mode: "nll"
});

default_test!(RunPassValgrind {
    path: "src/test/run-pass-valgrind",
    mode: "run-pass-valgrind",
    suite: "run-pass-valgrind"
});

default_test!(MirOpt { path: "src/test/mir-opt", mode: "mir-opt", suite: "mir-opt" });

default_test!(Codegen { path: "src/test/codegen", mode: "codegen", suite: "codegen" });

default_test!(CodegenUnits {
    path: "src/test/codegen-units",
    mode: "codegen-units",
    suite: "codegen-units"
});

default_test!(Incremental {
    path: "src/test/incremental",
    mode: "incremental",
    suite: "incremental"
});

default_test_with_compare_mode!(Debuginfo {
    path: "src/test/debuginfo",
    mode: "debuginfo",
    suite: "debuginfo",
    compare_mode: "split-dwarf"
});

host_test!(UiFullDeps { path: "src/test/ui-fulldeps", mode: "ui", suite: "ui-fulldeps" });

host_test!(Rustdoc { path: "src/test/rustdoc", mode: "rustdoc", suite: "rustdoc" });
host_test!(RustdocUi { path: "src/test/rustdoc-ui", mode: "ui", suite: "rustdoc-ui" });

host_test!(RustdocJson {
    path: "src/test/rustdoc-json",
    mode: "rustdoc-json",
    suite: "rustdoc-json"
});

host_test!(Pretty { path: "src/test/pretty", mode: "pretty", suite: "pretty" });

default_test!(RunMake { path: "src/test/run-make", mode: "run-make", suite: "run-make" });

host_test!(RunMakeFullDeps {
    path: "src/test/run-make-fulldeps",
    mode: "run-make",
    suite: "run-make-fulldeps"
});

default_test!(Assembly { path: "src/test/assembly", mode: "assembly", suite: "assembly" });

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
        if builder.top_stage == 0 && env::var("COMPILETEST_FORCE_STAGE0").is_err() {
            eprintln!("\
error: `--stage 0` runs compiletest on the beta compiler, not your local changes, and will almost always cause tests to fail
help: to test the compiler, use `--stage 1` instead
help: to test the standard library, use `--stage 0 library/std` instead
note: if you're sure you want to do this, please open an issue as to why. In the meantime, you can override this with `COMPILETEST_FORCE_STAGE0=1`."
            );
            std::process::exit(1);
        }

        let compiler = self.compiler;
        let target = self.target;
        let mode = self.mode;
        let suite = self.suite;

        // Path for test suite
        let suite_path = self.path;

        // Skip codegen tests if they aren't enabled in configuration.
        if !builder.config.codegen_tests && suite == "codegen" {
            return;
        }

        if suite == "debuginfo" {
            builder
                .ensure(dist::DebuggerScripts { sysroot: builder.sysroot(compiler), host: target });
        }

        if suite.ends_with("fulldeps") {
            builder.ensure(compile::Rustc { compiler, target });
        }

        builder.ensure(compile::Std { compiler, target });
        // ensure that `libproc_macro` is available on the host.
        builder.ensure(compile::Std { compiler, target: compiler.host });

        // Also provide `rust_test_helpers` for the host.
        builder.ensure(native::TestHelpers { target: compiler.host });

        // As well as the target, except for plain wasm32, which can't build it
        if !target.contains("wasm32") || target.contains("emscripten") {
            builder.ensure(native::TestHelpers { target });
        }

        builder.ensure(RemoteCopyLibs { compiler, target });

        let mut cmd = builder.tool_cmd(Tool::Compiletest);

        // compiletest currently has... a lot of arguments, so let's just pass all
        // of them!

        cmd.arg("--compile-lib-path").arg(builder.rustc_libdir(compiler));
        cmd.arg("--run-lib-path").arg(builder.sysroot_libdir(compiler, target));
        cmd.arg("--rustc-path").arg(builder.rustc(compiler));

        let is_rustdoc = suite.ends_with("rustdoc-ui") || suite.ends_with("rustdoc-js");

        // Avoid depending on rustdoc when we don't need it.
        if mode == "rustdoc"
            || (mode == "run-make" && suite.ends_with("fulldeps"))
            || (mode == "ui" && is_rustdoc)
            || mode == "js-doc-test"
            || mode == "rustdoc-json"
        {
            cmd.arg("--rustdoc-path").arg(builder.rustdoc(compiler));
        }

        if mode == "run-make" && suite.ends_with("fulldeps") {
            cmd.arg("--rust-demangler-path").arg(builder.tool_exe(Tool::RustDemangler));
        }

        cmd.arg("--src-base").arg(builder.src.join("src/test").join(suite));
        cmd.arg("--build-base").arg(testdir(builder, compiler.host).join(suite));
        cmd.arg("--stage-id").arg(format!("stage{}-{}", compiler.stage, target));
        cmd.arg("--suite").arg(suite);
        cmd.arg("--mode").arg(mode);
        cmd.arg("--target").arg(target.rustc_target_arg());
        cmd.arg("--host").arg(&*compiler.host.triple);
        cmd.arg("--llvm-filecheck").arg(builder.llvm_filecheck(builder.config.build));

        if builder.config.cmd.bless() {
            cmd.arg("--bless");
        }

        let compare_mode =
            builder.config.cmd.compare_mode().or_else(|| {
                if builder.config.test_compare_mode { self.compare_mode } else { None }
            });

        if let Some(ref pass) = builder.config.cmd.pass() {
            cmd.arg("--pass");
            cmd.arg(pass);
        }

        if let Some(ref nodejs) = builder.config.nodejs {
            cmd.arg("--nodejs").arg(nodejs);
        }

        let mut flags = if is_rustdoc { Vec::new() } else { vec!["-Crpath".to_string()] };
        if !is_rustdoc {
            if builder.config.rust_optimize_tests {
                flags.push("-O".to_string());
            }
        }
        flags.push(format!("-Cdebuginfo={}", builder.config.rust_debuginfo_level_tests));
        flags.push("-Zunstable-options".to_string());
        flags.push(builder.config.cmd.rustc_args().join(" "));

        if let Some(linker) = builder.linker(target) {
            cmd.arg("--linker").arg(linker);
        }

        let mut hostflags = flags.clone();
        hostflags.push(format!("-Lnative={}", builder.test_helpers_out(compiler.host).display()));
        if builder.is_fuse_ld_lld(compiler.host) {
            hostflags.push("-Clink-args=-fuse-ld=lld".to_string());
        }
        cmd.arg("--host-rustcflags").arg(hostflags.join(" "));

        let mut targetflags = flags;
        targetflags.push(format!("-Lnative={}", builder.test_helpers_out(target).display()));
        if builder.is_fuse_ld_lld(target) {
            targetflags.push("-Clink-args=-fuse-ld=lld".to_string());
        }
        cmd.arg("--target-rustcflags").arg(targetflags.join(" "));

        cmd.arg("--docck-python").arg(builder.python());

        if builder.config.build.ends_with("apple-darwin") {
            // Force /usr/bin/python3 on macOS for LLDB tests because we're loading the
            // LLDB plugin's compiled module which only works with the system python
            // (namely not Homebrew-installed python)
            cmd.arg("--lldb-python").arg("/usr/bin/python3");
        } else {
            cmd.arg("--lldb-python").arg(builder.python());
        }

        if let Some(ref gdb) = builder.config.gdb {
            cmd.arg("--gdb").arg(gdb);
        }

        let run = |cmd: &mut Command| {
            cmd.output().map(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .next()
                    .unwrap_or_else(|| panic!("{:?} failed {:?}", cmd, output))
                    .to_string()
            })
        };
        let lldb_exe = "lldb";
        let lldb_version = Command::new(lldb_exe)
            .arg("--version")
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).to_string())
            .ok();
        if let Some(ref vers) = lldb_version {
            cmd.arg("--lldb-version").arg(vers);
            let lldb_python_dir = run(Command::new(lldb_exe).arg("-P")).ok();
            if let Some(ref dir) = lldb_python_dir {
                cmd.arg("--lldb-python-dir").arg(dir);
            }
        }

        if util::forcing_clang_based_tests() {
            let clang_exe = builder.llvm_out(target).join("bin").join("clang");
            cmd.arg("--run-clang-based-tests-with").arg(clang_exe);
        }

        // Get paths from cmd args
        let paths = match &builder.config.cmd {
            Subcommand::Test { ref paths, .. } => &paths[..],
            _ => &[],
        };

        // Get test-args by striping suite path
        let mut test_args: Vec<&str> = paths
            .iter()
            .map(|p| match p.strip_prefix(".") {
                Ok(path) => path,
                Err(_) => p,
            })
            .filter(|p| p.starts_with(suite_path) && (p.is_dir() || p.is_file()))
            .filter_map(|p| {
                // Since test suite paths are themselves directories, if we don't
                // specify a directory or file, we'll get an empty string here
                // (the result of the test suite directory without its suite prefix).
                // Therefore, we need to filter these out, as only the first --test-args
                // flag is respected, so providing an empty --test-args conflicts with
                // any following it.
                match p.strip_prefix(suite_path).ok().and_then(|p| p.to_str()) {
                    Some(s) if s != "" => Some(s),
                    _ => None,
                }
            })
            .collect();

        test_args.append(&mut builder.config.cmd.test_args());

        cmd.args(&test_args);

        if builder.is_verbose() {
            cmd.arg("--verbose");
        }

        if !builder.config.verbose_tests {
            cmd.arg("--quiet");
        }

        let mut llvm_components_passed = false;
        let mut copts_passed = false;
        if builder.config.llvm_enabled() {
            let llvm_config = builder.ensure(native::Llvm { target: builder.config.build });
            if !builder.config.dry_run {
                let llvm_version = output(Command::new(&llvm_config).arg("--version"));
                let llvm_components = output(Command::new(&llvm_config).arg("--components"));
                // Remove trailing newline from llvm-config output.
                cmd.arg("--llvm-version")
                    .arg(llvm_version.trim())
                    .arg("--llvm-components")
                    .arg(llvm_components.trim());
                llvm_components_passed = true;
            }
            if !builder.is_rust_llvm(target) {
                cmd.arg("--system-llvm");
            }

            // Tests that use compiler libraries may inherit the `-lLLVM` link
            // requirement, but the `-L` library path is not propagated across
            // separate compilations. We can add LLVM's library path to the
            // platform-specific environment variable as a workaround.
            if !builder.config.dry_run && suite.ends_with("fulldeps") {
                let llvm_libdir = output(Command::new(&llvm_config).arg("--libdir"));
                add_link_lib_path(vec![llvm_libdir.trim().into()], &mut cmd);
            }

            // Only pass correct values for these flags for the `run-make` suite as it
            // requires that a C++ compiler was configured which isn't always the case.
            if !builder.config.dry_run && matches!(suite, "run-make" | "run-make-fulldeps") {
                // The llvm/bin directory contains many useful cross-platform
                // tools. Pass the path to run-make tests so they can use them.
                let llvm_bin_path = llvm_config
                    .parent()
                    .expect("Expected llvm-config to be contained in directory");
                assert!(llvm_bin_path.is_dir());
                cmd.arg("--llvm-bin-dir").arg(llvm_bin_path);

                // If LLD is available, add it to the PATH
                if builder.config.lld_enabled {
                    let lld_install_root =
                        builder.ensure(native::Lld { target: builder.config.build });

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
        if !builder.config.dry_run && matches!(suite, "run-make" | "run-make-fulldeps") {
            cmd.arg("--cc")
                .arg(builder.cc(target))
                .arg("--cxx")
                .arg(builder.cxx(target).unwrap())
                .arg("--cflags")
                .arg(builder.cflags(target, GitRepo::Rustc).join(" "));
            copts_passed = true;
            if let Some(ar) = builder.ar(target) {
                cmd.arg("--ar").arg(ar);
            }
        }

        if !llvm_components_passed {
            cmd.arg("--llvm-components").arg("");
        }
        if !copts_passed {
            cmd.arg("--cc").arg("").arg("--cxx").arg("").arg("--cflags").arg("");
        }

        if builder.remote_tested(target) {
            cmd.arg("--remote-test-client").arg(builder.tool_exe(Tool::RemoteTestClient));
        }

        // Running a C compiler on MSVC requires a few env vars to be set, to be
        // sure to set them here.
        //
        // Note that if we encounter `PATH` we make sure to append to our own `PATH`
        // rather than stomp over it.
        if target.contains("msvc") {
            for &(ref k, ref v) in builder.cc[&target].env() {
                if k != "PATH" {
                    cmd.env(k, v);
                }
            }
        }
        cmd.env("RUSTC_BOOTSTRAP", "1");
        builder.add_rust_test_threads(&mut cmd);

        if builder.config.sanitizers_enabled(target) {
            cmd.env("RUSTC_SANITIZER_SUPPORT", "1");
        }

        if builder.config.profiler_enabled(target) {
            cmd.env("RUSTC_PROFILER_SUPPORT", "1");
        }

        let tmp = builder.out.join("tmp");
        std::fs::create_dir_all(&tmp).unwrap();
        cmd.env("RUST_TEST_TMPDIR", tmp);

        cmd.arg("--adb-path").arg("adb");
        cmd.arg("--adb-test-dir").arg(ADB_TEST_DIR);
        if target.contains("android") {
            // Assume that cc for this target comes from the android sysroot
            cmd.arg("--android-cross-path")
                .arg(builder.cc(target).parent().unwrap().parent().unwrap());
        } else {
            cmd.arg("--android-cross-path").arg("");
        }

        if builder.config.cmd.rustfix_coverage() {
            cmd.arg("--rustfix-coverage");
        }

        cmd.env("BOOTSTRAP_CARGO", &builder.initial_cargo);

        builder.ci_env.force_coloring_in_ci(&mut cmd);

        builder.info(&format!(
            "Check compiletest suite={} mode={} ({} -> {})",
            suite, mode, &compiler.host, target
        ));
        let _time = util::timeit(&builder);
        try_run(builder, &mut cmd);

        if let Some(compare_mode) = compare_mode {
            cmd.arg("--compare-mode").arg(compare_mode);
            builder.info(&format!(
                "Check compiletest suite={} mode={} compare_mode={} ({} -> {})",
                suite, mode, compare_mode, &compiler.host, target
            ));
            let _time = util::timeit(&builder);
            try_run(builder, &mut cmd);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BookTest {
    compiler: Compiler,
    path: PathBuf,
    name: &'static str,
    is_ext_doc: bool,
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

        builder.ensure(compile::Std { compiler, target: compiler.host });

        // mdbook just executes a binary named "rustdoc", so we need to update
        // PATH so that it points to our rustdoc.
        let mut rustdoc_path = builder.rustdoc(compiler);
        rustdoc_path.pop();
        let old_path = env::var_os("PATH").unwrap_or_default();
        let new_path = env::join_paths(iter::once(rustdoc_path).chain(env::split_paths(&old_path)))
            .expect("could not add rustdoc to PATH");

        let mut rustbook_cmd = builder.tool_cmd(Tool::Rustbook);
        let path = builder.src.join(&self.path);
        rustbook_cmd.env("PATH", new_path).arg("test").arg(path);
        builder.add_rust_test_threads(&mut rustbook_cmd);
        builder.info(&format!("Testing rustbook {}", self.path.display()));
        let _time = util::timeit(&builder);
        let toolstate = if try_run(builder, &mut rustbook_cmd) {
            ToolState::TestPass
        } else {
            ToolState::TestFail
        };
        builder.save_toolstate(self.name, toolstate);
    }

    /// This runs `rustdoc --test` on all `.md` files in the path.
    fn run_local_doc(self, builder: &Builder<'_>) {
        let compiler = self.compiler;

        builder.ensure(compile::Std { compiler, target: compiler.host });

        // Do a breadth-first traversal of the `src/doc` directory and just run
        // tests for all files that end in `*.md`
        let mut stack = vec![builder.src.join(self.path)];
        let _time = util::timeit(&builder);
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
    ($($name:ident, $path:expr, $book_name:expr, default=$default:expr;)+) => {
        $(
            #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
                    builder.ensure(BookTest {
                        compiler: self.compiler,
                        path: PathBuf::from($path),
                        name: $book_name,
                        is_ext_doc: !$default,
                    });
                }
            }
        )+
    }
}

test_book!(
    Nomicon, "src/doc/nomicon", "nomicon", default=false;
    Reference, "src/doc/reference", "reference", default=false;
    RustdocBook, "src/doc/rustdoc", "rustdoc", default=true;
    RustcBook, "src/doc/rustc", "rustc", default=true;
    RustByExample, "src/doc/rust-by-example", "rust-by-example", default=false;
    EmbeddedBook, "src/doc/embedded-book", "embedded-book", default=false;
    TheBook, "src/doc/book", "book", default=false;
    UnstableBook, "src/doc/unstable-book", "unstable-book", default=true;
    EditionGuide, "src/doc/edition-guide", "edition-guide", default=false;
);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ErrorIndex {
    compiler: Compiler,
}

impl Step for ErrorIndex {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/error_index_generator")
    }

    fn make_run(run: RunConfig<'_>) {
        // error_index_generator depends on librustdoc. Use the compiler that
        // is normally used to build rustdoc for other tests (like compiletest
        // tests in src/test/rustdoc) so that it shares the same artifacts.
        let compiler = run.builder.compiler_for(run.builder.top_stage, run.target, run.target);
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

        let mut tool = tool::ErrorIndex::command(builder, compiler);
        tool.arg("markdown").arg(&output);

        // Use the rustdoc that was built by self.compiler. This copy of
        // rustdoc is shared with other tests (like compiletest tests in
        // src/test/rustdoc). This helps avoid building rustdoc multiple
        // times.
        let rustdoc_compiler = builder.compiler(builder.top_stage, builder.config.build);
        builder.info(&format!("Testing error-index stage{}", rustdoc_compiler.stage));
        let _time = util::timeit(&builder);
        builder.run_quiet(&mut tool);
        builder.ensure(compile::Std { compiler: rustdoc_compiler, target: rustdoc_compiler.host });
        markdown_test(builder, rustdoc_compiler, &output);
    }
}

fn markdown_test(builder: &Builder<'_>, compiler: Compiler, markdown: &Path) -> bool {
    if let Ok(contents) = fs::read_to_string(markdown) {
        if !contents.contains("```") {
            return true;
        }
    }

    builder.info(&format!("doc tests for: {}", markdown.display()));
    let mut cmd = builder.rustdoc_cmd(compiler);
    builder.add_rust_test_threads(&mut cmd);
    cmd.arg("--test");
    cmd.arg(markdown);
    cmd.env("RUSTC_BOOTSTRAP", "1");

    let test_args = builder.config.cmd.test_args().join(" ");
    cmd.arg("--test-args").arg(test_args);

    if builder.config.verbose_tests {
        try_run(builder, &mut cmd)
    } else {
        try_run_quiet(builder, &mut cmd)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct RustcGuide;

impl Step for RustcGuide {
    type Output = ();
    const DEFAULT: bool = false;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/doc/rustc-dev-guide")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustcGuide);
    }

    fn run(self, builder: &Builder<'_>) {
        let src = builder.src.join("src/doc/rustc-dev-guide");
        let mut rustbook_cmd = builder.tool_cmd(Tool::Rustbook);
        let toolstate = if try_run(builder, rustbook_cmd.arg("linkcheck").arg(&src)) {
            ToolState::TestPass
        } else {
            ToolState::TestFail
        };
        builder.save_toolstate("rustc-dev-guide", toolstate);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CrateLibrustc {
    compiler: Compiler,
    target: TargetSelection,
    test_kind: TestKind,
    krate: Interned<String>,
}

impl Step for CrateLibrustc {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.krate("rustc-main")
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let compiler = builder.compiler(builder.top_stage, run.build_triple());

        for krate in builder.in_tree_crates("rustc-main", Some(run.target)) {
            if krate.path.ends_with(&run.path) {
                let test_kind = builder.kind.into();

                builder.ensure(CrateLibrustc {
                    compiler,
                    target: run.target,
                    test_kind,
                    krate: krate.name,
                });
            }
        }
    }

    fn run(self, builder: &Builder<'_>) {
        builder.ensure(Crate {
            compiler: self.compiler,
            target: self.target,
            mode: Mode::Rustc,
            test_kind: self.test_kind,
            krate: self.krate,
        });
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CrateNotDefault {
    compiler: Compiler,
    target: TargetSelection,
    test_kind: TestKind,
    krate: &'static str,
}

impl Step for CrateNotDefault {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/librustc_asan")
            .path("src/librustc_lsan")
            .path("src/librustc_msan")
            .path("src/librustc_tsan")
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let compiler = builder.compiler(builder.top_stage, run.build_triple());

        let test_kind = builder.kind.into();

        builder.ensure(CrateNotDefault {
            compiler,
            target: run.target,
            test_kind,
            krate: match run.path {
                _ if run.path.ends_with("src/librustc_asan") => "rustc_asan",
                _ if run.path.ends_with("src/librustc_lsan") => "rustc_lsan",
                _ if run.path.ends_with("src/librustc_msan") => "rustc_msan",
                _ if run.path.ends_with("src/librustc_tsan") => "rustc_tsan",
                _ => panic!("unexpected path {:?}", run.path),
            },
        });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.ensure(Crate {
            compiler: self.compiler,
            target: self.target,
            mode: Mode::Std,
            test_kind: self.test_kind,
            krate: INTERNER.intern_str(self.krate),
        });
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Crate {
    pub compiler: Compiler,
    pub target: TargetSelection,
    pub mode: Mode,
    pub test_kind: TestKind,
    pub krate: Interned<String>,
}

impl Step for Crate {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.krate("test")
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let compiler = builder.compiler(builder.top_stage, run.build_triple());

        let make = |mode: Mode, krate: &CargoCrate| {
            let test_kind = builder.kind.into();

            builder.ensure(Crate {
                compiler,
                target: run.target,
                mode,
                test_kind,
                krate: krate.name,
            });
        };

        for krate in builder.in_tree_crates("test", Some(run.target)) {
            if krate.path.ends_with(&run.path) {
                make(Mode::Std, krate);
            }
        }
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
        let test_kind = self.test_kind;
        let krate = self.krate;

        builder.ensure(compile::Std { compiler, target });
        builder.ensure(RemoteCopyLibs { compiler, target });

        // If we're not doing a full bootstrap but we're testing a stage2
        // version of libstd, then what we're actually testing is the libstd
        // produced in stage1. Reflect that here by updating the compiler that
        // we're working with automatically.
        let compiler = builder.compiler_for(compiler.stage, compiler.host, target);

        let mut cargo =
            builder.cargo(compiler, mode, SourceType::InTree, target, test_kind.subcommand());
        match mode {
            Mode::Std => {
                compile::std_cargo(builder, target, compiler.stage, &mut cargo);
            }
            Mode::Rustc => {
                builder.ensure(compile::Rustc { compiler, target });
                compile::rustc_cargo(builder, &mut cargo, target);
            }
            _ => panic!("can only test libraries"),
        };

        // Build up the base `cargo test` command.
        //
        // Pass in some standard flags then iterate over the graph we've discovered
        // in `cargo metadata` with the maps above and figure out what `-p`
        // arguments need to get passed.
        if test_kind.subcommand() == "test" && !builder.fail_fast {
            cargo.arg("--no-fail-fast");
        }
        match builder.doc_tests {
            DocTests::Only => {
                cargo.arg("--doc");
            }
            DocTests::No => {
                cargo.args(&["--lib", "--bins", "--examples", "--tests", "--benches"]);
            }
            DocTests::Yes => {}
        }

        cargo.arg("-p").arg(krate);

        // The tests are going to run with the *target* libraries, so we need to
        // ensure that those libraries show up in the LD_LIBRARY_PATH equivalent.
        //
        // Note that to run the compiler we need to run with the *host* libraries,
        // but our wrapper scripts arrange for that to be the case anyway.
        let mut dylib_path = dylib_path();
        dylib_path.insert(0, PathBuf::from(&*builder.sysroot_libdir(compiler, target)));
        cargo.env(dylib_path_var(), env::join_paths(&dylib_path).unwrap());

        cargo.arg("--");
        cargo.args(&builder.config.cmd.test_args());

        if !builder.config.verbose_tests {
            cargo.arg("--quiet");
        }

        if target.contains("emscripten") {
            cargo.env(
                format!("CARGO_TARGET_{}_RUNNER", envify(&target.triple)),
                builder.config.nodejs.as_ref().expect("nodejs not configured"),
            );
        } else if target.starts_with("wasm32") {
            let node = builder.config.nodejs.as_ref().expect("nodejs not configured");
            let runner =
                format!("{} {}/src/etc/wasm32-shim.js", node.display(), builder.src.display());
            cargo.env(format!("CARGO_TARGET_{}_RUNNER", envify(&target.triple)), &runner);
        } else if builder.remote_tested(target) {
            cargo.env(
                format!("CARGO_TARGET_{}_RUNNER", envify(&target.triple)),
                format!("{} run 0", builder.tool_exe(Tool::RemoteTestClient).display()),
            );
        }

        builder.info(&format!(
            "{} {} stage{} ({} -> {})",
            test_kind, krate, compiler.stage, &compiler.host, target
        ));
        let _time = util::timeit(&builder);
        try_run(builder, &mut cargo.into());
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CrateRustdoc {
    host: TargetSelection,
    test_kind: TestKind,
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

        let test_kind = builder.kind.into();

        builder.ensure(CrateRustdoc { host: run.target, test_kind });
    }

    fn run(self, builder: &Builder<'_>) {
        let test_kind = self.test_kind;
        let target = self.host;

        // Use the previous stage compiler to reuse the artifacts that are
        // created when running compiletest for src/test/rustdoc. If this used
        // `compiler`, then it would cause rustdoc to be built *again*, which
        // isn't really necessary.
        let compiler = builder.compiler_for(builder.top_stage, target, target);
        builder.ensure(compile::Rustc { compiler, target });

        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            target,
            test_kind.subcommand(),
            "src/tools/rustdoc",
            SourceType::InTree,
            &[],
        );
        if test_kind.subcommand() == "test" && !builder.fail_fast {
            cargo.arg("--no-fail-fast");
        }

        cargo.arg("-p").arg("rustdoc:0.0.0");

        cargo.arg("--");
        cargo.args(&builder.config.cmd.test_args());

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
        // It should be considered to just stop running doctests on
        // librustdoc. There is only one test, and it doesn't look too
        // important. There might be other ways to avoid this, but it seems
        // pretty convoluted.
        //
        // See also https://github.com/rust-lang/rust/issues/13983 where the
        // host vs target dylibs for rustdoc are consistently tricky to deal
        // with.
        let mut dylib_path = dylib_path();
        dylib_path.insert(0, PathBuf::from(&*builder.sysroot_libdir(compiler, target)));
        cargo.env(dylib_path_var(), env::join_paths(&dylib_path).unwrap());

        if !builder.config.verbose_tests {
            cargo.arg("--quiet");
        }

        builder.info(&format!(
            "{} rustdoc stage{} ({} -> {})",
            test_kind, compiler.stage, &compiler.host, target
        ));
        let _time = util::timeit(&builder);

        try_run(builder, &mut cargo.into());
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
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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

        builder.ensure(compile::Std { compiler, target });

        builder.info(&format!("REMOTE copy libs to emulator ({})", target));
        t!(fs::create_dir_all(builder.out.join("tmp")));

        let server =
            builder.ensure(tool::RemoteTestServer { compiler: compiler.with_stage(0), target });

        // Spawn the emulator and wait for it to come online
        let tool = builder.tool_exe(Tool::RemoteTestClient);
        let mut cmd = Command::new(&tool);
        cmd.arg("spawn-emulator").arg(target.triple).arg(&server).arg(builder.out.join("tmp"));
        if let Some(rootfs) = builder.qemu_rootfs(target) {
            cmd.arg(rootfs);
        }
        builder.run(&mut cmd);

        // Push all our dylibs to the emulator
        for f in t!(builder.sysroot_libdir(compiler, target).read_dir()) {
            let f = t!(f);
            let name = f.file_name().into_string().unwrap();
            if util::is_dylib(&name) {
                builder.run(Command::new(&tool).arg("push").arg(f.path()));
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Distcheck;

impl Step for Distcheck {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("distcheck")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Distcheck);
    }

    /// Runs "distcheck", a 'make check' from a tarball
    fn run(self, builder: &Builder<'_>) {
        builder.info("Distcheck");
        let dir = builder.out.join("tmp").join("distcheck");
        let _ = fs::remove_dir_all(&dir);
        t!(fs::create_dir_all(&dir));

        // Guarantee that these are built before we begin running.
        builder.ensure(dist::PlainSourceTarball);
        builder.ensure(dist::Src);

        let mut cmd = Command::new("tar");
        cmd.arg("-xf")
            .arg(builder.ensure(dist::PlainSourceTarball))
            .arg("--strip-components=1")
            .current_dir(&dir);
        builder.run(&mut cmd);
        builder.run(
            Command::new("./configure")
                .args(&builder.config.configure_args)
                .arg("--enable-vendor")
                .current_dir(&dir),
        );
        builder.run(
            Command::new(build_helper::make(&builder.config.build.triple))
                .arg("check")
                .current_dir(&dir),
        );

        // Now make sure that rust-src has all of libstd's dependencies
        builder.info("Distcheck rust-src");
        let dir = builder.out.join("tmp").join("distcheck-src");
        let _ = fs::remove_dir_all(&dir);
        t!(fs::create_dir_all(&dir));

        let mut cmd = Command::new("tar");
        cmd.arg("-xf").arg(builder.ensure(dist::Src)).arg("--strip-components=1").current_dir(&dir);
        builder.run(&mut cmd);

        let toml = dir.join("rust-src/lib/rustlib/src/rust/library/std/Cargo.toml");
        builder.run(
            Command::new(&builder.initial_cargo)
                .arg("generate-lockfile")
                .arg("--manifest-path")
                .arg(&toml)
                .current_dir(&dir),
        );
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Bootstrap;

impl Step for Bootstrap {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    /// Tests the build system itself.
    fn run(self, builder: &Builder<'_>) {
        let mut cmd = Command::new(&builder.initial_cargo);
        cmd.arg("test")
            .current_dir(builder.src.join("src/bootstrap"))
            .env("RUSTFLAGS", "-Cdebuginfo=2")
            .env("CARGO_TARGET_DIR", builder.out.join("bootstrap"))
            .env("BOOTSTRAP_OUTPUT_DIRECTORY", &builder.config.out)
            .env("BOOTSTRAP_INITIAL_CARGO", &builder.config.initial_cargo)
            .env("RUSTC_BOOTSTRAP", "1")
            .env("RUSTC", &builder.initial_rustc);
        if let Some(flags) = option_env!("RUSTFLAGS") {
            // Use the same rustc flags for testing as for "normal" compilation,
            // so that Cargo doesn’t recompile the entire dependency graph every time:
            // https://github.com/rust-lang/rust/issues/49215
            cmd.env("RUSTFLAGS", flags);
        }
        if !builder.fail_fast {
            cmd.arg("--no-fail-fast");
        }
        cmd.arg("--").args(&builder.config.cmd.test_args());
        // rustbuild tests are racy on directory creation so just run them one at a time.
        // Since there's not many this shouldn't be a problem.
        cmd.arg("--test-threads=1");
        try_run(builder, &mut cmd);
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/bootstrap")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Bootstrap);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
        builder.ensure(compile::Std { compiler: self.compiler, target: self.compiler.host });
        let mut cargo = tool::prepare_tool_cargo(
            builder,
            self.compiler,
            Mode::ToolStd,
            self.compiler.host,
            "run",
            "src/tools/tier-check",
            SourceType::InTree,
            &[],
        );
        cargo.arg(builder.src.join("src/doc/rustc/src/platform-support.md"));
        cargo.arg(&builder.rustc(self.compiler));
        if builder.is_verbose() {
            cargo.arg("--verbose");
        }

        builder.info("platform support check");
        try_run(builder, &mut cargo.into());
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
        builder.ensure(crate::doc::RustcBook {
            compiler: self.compiler,
            target: self.target,
            validate: true,
        });
    }
}
