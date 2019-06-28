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
use crate::dist;
use crate::flags::Subcommand;
use crate::native;
use crate::tool::{self, Tool, SourceType};
use crate::toolstate::ToolState;
use crate::util::{self, dylib_path, dylib_path_var};
use crate::Crate as CargoCrate;
use crate::{DocTests, Mode, GitRepo};

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
    host: Interned<String>,
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
            builder
                .tool_cmd(Tool::Linkchecker)
                .arg(builder.out.join(host).join("doc")),
        );
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/tools/linkchecker")
            .default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Linkcheck { host: run.target });
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Cargotest {
    stage: u32,
    host: Interned<String>,
}

impl Step for Cargotest {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/cargotest")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Cargotest {
            stage: run.builder.top_stage,
            host: run.target,
        });
    }

    /// Runs the `cargotest` tool as compiled in `stage` by the `host` compiler.
    ///
    /// This tool in `src/tools` will check out a few Rust projects and run `cargo
    /// test` to ensure that we don't regress the test suites there.
    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(self.stage, self.host);
        builder.ensure(compile::Rustc {
            compiler,
            target: compiler.host,
        });

        // Note that this is a short, cryptic, and not scoped directory name. This
        // is currently to minimize the length of path on Windows where we otherwise
        // quickly run into path name limit constraints.
        let out_dir = builder.out.join("ct");
        t!(fs::create_dir_all(&out_dir));

        let _time = util::timeit(&builder);
        let mut cmd = builder.tool_cmd(Tool::CargoTest);
        try_run(
            builder,
            cmd.arg(&builder.initial_cargo)
                .arg(&out_dir)
                .env("RUSTC", builder.rustc(compiler))
                .env("RUSTDOC", builder.rustdoc(compiler)),
        );
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Cargo {
    stage: u32,
    host: Interned<String>,
}

impl Step for Cargo {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/cargo")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Cargo {
            stage: run.builder.top_stage,
            host: run.target,
        });
    }

    /// Runs `cargo test` for `cargo` packaged with Rust.
    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(self.stage, self.host);

        builder.ensure(tool::Cargo {
            compiler,
            target: self.host,
        });
        let mut cargo = tool::prepare_tool_cargo(builder,
                                                 compiler,
                                                 Mode::ToolRustc,
                                                 self.host,
                                                 "test",
                                                 "src/tools/cargo",
                                                 SourceType::Submodule,
                                                 &[]);

        if !builder.fail_fast {
            cargo.arg("--no-fail-fast");
        }

        // Don't run cross-compile tests, we may not have cross-compiled libstd libs
        // available.
        cargo.env("CFG_DISABLE_CROSS_TESTS", "1");
        // Disable a test that has issues with mingw.
        cargo.env("CARGO_TEST_DISABLE_GIT_CLI", "1");

        try_run(
            builder,
            cargo.env("PATH", &path_for_cargo(builder, compiler)),
        );
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Rls {
    stage: u32,
    host: Interned<String>,
}

impl Step for Rls {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/rls")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rls {
            stage: run.builder.top_stage,
            host: run.target,
        });
    }

    /// Runs `cargo test` for the rls.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let host = self.host;
        let compiler = builder.compiler(stage, host);

        let build_result = builder.ensure(tool::Rls {
            compiler,
            target: self.host,
            extra_features: Vec::new(),
        });
        if build_result.is_none() {
            eprintln!("failed to test rls: could not build");
            return;
        }

        let mut cargo = tool::prepare_tool_cargo(builder,
                                                 compiler,
                                                 Mode::ToolRustc,
                                                 host,
                                                 "test",
                                                 "src/tools/rls",
                                                 SourceType::Submodule,
                                                 &[]);

        builder.add_rustc_lib_path(compiler, &mut cargo);
        cargo.arg("--")
            .args(builder.config.cmd.test_args());

        if try_run(builder, &mut cargo) {
            builder.save_toolstate("rls", ToolState::TestPass);
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Rustfmt {
    stage: u32,
    host: Interned<String>,
}

impl Step for Rustfmt {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/rustfmt")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rustfmt {
            stage: run.builder.top_stage,
            host: run.target,
        });
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

        let mut cargo = tool::prepare_tool_cargo(builder,
                                                 compiler,
                                                 Mode::ToolRustc,
                                                 host,
                                                 "test",
                                                 "src/tools/rustfmt",
                                                 SourceType::Submodule,
                                                 &[]);

        let dir = testdir(builder, compiler.host);
        t!(fs::create_dir_all(&dir));
        cargo.env("RUSTFMT_TEST_DIR", dir);

        builder.add_rustc_lib_path(compiler, &mut cargo);

        if try_run(builder, &mut cargo) {
            builder.save_toolstate("rustfmt", ToolState::TestPass);
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Miri {
    stage: u32,
    host: Interned<String>,
}

impl Step for Miri {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let test_miri = run.builder.config.test_miri;
        run.path("src/tools/miri").default_condition(test_miri)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Miri {
            stage: run.builder.top_stage,
            host: run.target,
        });
    }

    /// Runs `cargo test` for miri.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let host = self.host;
        let compiler = builder.compiler(stage, host);

        let miri = builder.ensure(tool::Miri {
            compiler,
            target: self.host,
            extra_features: Vec::new(),
        });
        if let Some(miri) = miri {
            let mut cargo = tool::prepare_tool_cargo(builder,
                                                 compiler,
                                                 Mode::ToolRustc,
                                                 host,
                                                 "test",
                                                 "src/tools/miri",
                                                 SourceType::Submodule,
                                                 &[]);

            // miri tests need to know about the stage sysroot
            cargo.env("MIRI_SYSROOT", builder.sysroot(compiler));
            cargo.env("RUSTC_TEST_SUITE", builder.rustc(compiler));
            cargo.env("RUSTC_LIB_PATH", builder.rustc_libdir(compiler));
            cargo.env("MIRI_PATH", miri);

            builder.add_rustc_lib_path(compiler, &mut cargo);

            if try_run(builder, &mut cargo) {
                builder.save_toolstate("miri", ToolState::TestPass);
            }
        } else {
            eprintln!("failed to test miri: could not build");
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CompiletestTest {
    host: Interned<String>,
}

impl Step for CompiletestTest {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/compiletest")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(CompiletestTest {
            host: run.target,
        });
    }

    /// Runs `cargo test` for compiletest.
    fn run(self, builder: &Builder<'_>) {
        let host = self.host;
        let compiler = builder.compiler(0, host);

        let mut cargo = tool::prepare_tool_cargo(builder,
                                                 compiler,
                                                 Mode::ToolBootstrap,
                                                 host,
                                                 "test",
                                                 "src/tools/compiletest",
                                                 SourceType::InTree,
                                                 &[]);

        try_run(builder, &mut cargo);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Clippy {
    stage: u32,
    host: Interned<String>,
}

impl Step for Clippy {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/clippy")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Clippy {
            stage: run.builder.top_stage,
            host: run.target,
        });
    }

    /// Runs `cargo test` for clippy.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let host = self.host;
        let compiler = builder.compiler(stage, host);

        let clippy = builder.ensure(tool::Clippy {
            compiler,
            target: self.host,
            extra_features: Vec::new(),
        });
        if let Some(clippy) = clippy {
            let mut cargo = tool::prepare_tool_cargo(builder,
                                                 compiler,
                                                 Mode::ToolRustc,
                                                 host,
                                                 "test",
                                                 "src/tools/clippy",
                                                 SourceType::Submodule,
                                                 &[]);

            // clippy tests need to know about the stage sysroot
            cargo.env("SYSROOT", builder.sysroot(compiler));
            cargo.env("RUSTC_TEST_SUITE", builder.rustc(compiler));
            cargo.env("RUSTC_LIB_PATH", builder.rustc_libdir(compiler));
            let host_libs = builder
                .stage_out(compiler, Mode::ToolRustc)
                .join(builder.cargo_dir());
            cargo.env("HOST_LIBS", host_libs);
            // clippy tests need to find the driver
            cargo.env("CLIPPY_DRIVER_PATH", clippy);

            builder.add_rustc_lib_path(compiler, &mut cargo);

            if try_run(builder, &mut cargo) {
                builder.save_toolstate("clippy-driver", ToolState::TestPass);
            }
        } else {
            eprintln!("failed to test clippy: could not build");
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
        let compiler = run.builder.compiler(run.builder.top_stage, run.host);

        run.builder.ensure(RustdocTheme { compiler });
    }

    fn run(self, builder: &Builder<'_>) {
        let rustdoc = builder.out.join("bootstrap/debug/rustdoc");
        let mut cmd = builder.tool_cmd(Tool::RustdocTheme);
        cmd.arg(rustdoc.to_str().unwrap())
            .arg(
                builder
                    .src
                    .join("src/librustdoc/html/static/themes")
                    .to_str()
                    .unwrap(),
            )
            .env("RUSTC_STAGE", self.compiler.stage.to_string())
            .env("RUSTC_SYSROOT", builder.sysroot(self.compiler))
            .env(
                "RUSTDOC_LIBDIR",
                builder.sysroot_libdir(self.compiler, self.compiler.host),
            )
            .env("CFG_RELEASE_CHANNEL", &builder.config.channel)
            .env("RUSTDOC_REAL", builder.rustdoc(self.compiler))
            .env("RUSTDOC_CRATE_VERSION", builder.rust_version())
            .env("RUSTC_BOOTSTRAP", "1");
        if let Some(linker) = builder.linker(self.compiler.host) {
            cmd.env("RUSTC_TARGET_LINKER", linker);
        }
        try_run(builder, &mut cmd);
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct RustdocJSStd {
    pub host: Interned<String>,
    pub target: Interned<String>,
}

impl Step for RustdocJSStd {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/test/rustdoc-js-std")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustdocJSStd {
            host: run.host,
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        if let Some(ref nodejs) = builder.config.nodejs {
            let mut command = Command::new(nodejs);
            command.args(&["src/tools/rustdoc-js-std/tester.js", &*self.host]);
            builder.ensure(crate::doc::Std {
                target: self.target,
                stage: builder.top_stage,
            });
            builder.run(&mut command);
        } else {
            builder.info(
                "No nodejs found, skipping \"src/test/rustdoc-js-std\" tests"
            );
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct RustdocJSNotStd {
    pub host: Interned<String>,
    pub target: Interned<String>,
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
        let compiler = run.builder.compiler(run.builder.top_stage, run.host);
        run.builder.ensure(RustdocJSNotStd {
            host: run.host,
            target: run.target,
            compiler,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        if builder.config.nodejs.is_some() {
            builder.ensure(Compiletest {
                compiler: self.compiler,
                target: self.target,
                mode: "js-doc-test",
                suite: "rustdoc-js",
                path: None,
                compare_mode: None,
            });
        } else {
            builder.info(
                "No nodejs found, skipping \"src/test/rustdoc-js\" tests"
            );
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct RustdocUi {
    pub host: Interned<String>,
    pub target: Interned<String>,
    pub compiler: Compiler,
}

impl Step for RustdocUi {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/test/rustdoc-ui")
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler = run.builder.compiler(run.builder.top_stage, run.host);
        run.builder.ensure(RustdocUi {
            host: run.host,
            target: run.target,
            compiler,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.ensure(Compiletest {
            compiler: self.compiler,
            target: self.target,
            mode: "ui",
            suite: "rustdoc-ui",
            path: Some("src/test/rustdoc-ui"),
            compare_mode: None,
        })
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
    fn run(self, builder: &Builder<'_>) {
        let mut cmd = builder.tool_cmd(Tool::Tidy);
        cmd.arg(builder.src.join("src"));
        cmd.arg(&builder.initial_cargo);
        if !builder.config.vendor {
            cmd.arg("--no-vendor");
        }
        if builder.is_verbose() {
            cmd.arg("--verbose");
        }

        let _folder = builder.fold_output(|| "tidy");
        builder.info("tidy check");
        try_run(builder, &mut cmd);
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/tidy")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Tidy);
    }
}

fn testdir(builder: &Builder<'_>, host: Interned<String>) -> PathBuf {
    builder.out.join(host).join("test")
}

macro_rules! default_test {
    ($name:ident { path: $path:expr, mode: $mode:expr, suite: $suite:expr }) => {
        test!($name { path: $path, mode: $mode, suite: $suite, default: true, host: false });
    }
}

macro_rules! default_test_with_compare_mode {
    ($name:ident { path: $path:expr, mode: $mode:expr, suite: $suite:expr,
                   compare_mode: $compare_mode:expr }) => {
        test_with_compare_mode!($name { path: $path, mode: $mode, suite: $suite, default: true,
                                        host: false, compare_mode: $compare_mode });
    }
}

macro_rules! host_test {
    ($name:ident { path: $path:expr, mode: $mode:expr, suite: $suite:expr }) => {
        test!($name { path: $path, mode: $mode, suite: $suite, default: true, host: true });
    }
}

macro_rules! test {
    ($name:ident { path: $path:expr, mode: $mode:expr, suite: $suite:expr, default: $default:expr,
                   host: $host:expr }) => {
        test_definitions!($name { path: $path, mode: $mode, suite: $suite, default: $default,
                                  host: $host, compare_mode: None });
    }
}

macro_rules! test_with_compare_mode {
    ($name:ident { path: $path:expr, mode: $mode:expr, suite: $suite:expr, default: $default:expr,
                   host: $host:expr, compare_mode: $compare_mode:expr }) => {
        test_definitions!($name { path: $path, mode: $mode, suite: $suite, default: $default,
                                  host: $host, compare_mode: Some($compare_mode) });
    }
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
            pub target: Interned<String>,
        }

        impl Step for $name {
            type Output = ();
            const DEFAULT: bool = $default;
            const ONLY_HOSTS: bool = $host;

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                run.suite_path($path)
            }

            fn make_run(run: RunConfig<'_>) {
                let compiler = run.builder.compiler(run.builder.top_stage, run.host);

                run.builder.ensure($name {
                    compiler,
                    target: run.target,
                });
            }

            fn run(self, builder: &Builder<'_>) {
                builder.ensure(Compiletest {
                    compiler: self.compiler,
                    target: self.target,
                    mode: $mode,
                    suite: $suite,
                    path: Some($path),
                    compare_mode: $compare_mode,
                })
            }
        }
    }
}

default_test_with_compare_mode!(Ui {
    path: "src/test/ui",
    mode: "ui",
    suite: "ui",
    compare_mode: "nll"
});

default_test_with_compare_mode!(RunPass {
    path: "src/test/run-pass",
    mode: "run-pass",
    suite: "run-pass",
    compare_mode: "nll"
});

default_test!(CompileFail {
    path: "src/test/compile-fail",
    mode: "compile-fail",
    suite: "compile-fail"
});

default_test!(RunFail {
    path: "src/test/run-fail",
    mode: "run-fail",
    suite: "run-fail"
});

default_test!(RunPassValgrind {
    path: "src/test/run-pass-valgrind",
    mode: "run-pass-valgrind",
    suite: "run-pass-valgrind"
});

default_test!(MirOpt {
    path: "src/test/mir-opt",
    mode: "mir-opt",
    suite: "mir-opt"
});

default_test!(Codegen {
    path: "src/test/codegen",
    mode: "codegen",
    suite: "codegen"
});

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

default_test!(Debuginfo {
    path: "src/test/debuginfo",
    mode: "debuginfo",
    suite: "debuginfo"
});

host_test!(UiFullDeps {
    path: "src/test/ui-fulldeps",
    mode: "ui",
    suite: "ui-fulldeps"
});

host_test!(RunPassFullDeps {
    path: "src/test/run-pass-fulldeps",
    mode: "run-pass",
    suite: "run-pass-fulldeps"
});

host_test!(Rustdoc {
    path: "src/test/rustdoc",
    mode: "rustdoc",
    suite: "rustdoc"
});

host_test!(Pretty {
    path: "src/test/pretty",
    mode: "pretty",
    suite: "pretty"
});
test!(RunPassPretty {
    path: "src/test/run-pass/pretty",
    mode: "pretty",
    suite: "run-pass",
    default: false,
    host: true
});
test!(RunFailPretty {
    path: "src/test/run-fail/pretty",
    mode: "pretty",
    suite: "run-fail",
    default: false,
    host: true
});
test!(RunPassValgrindPretty {
    path: "src/test/run-pass-valgrind/pretty",
    mode: "pretty",
    suite: "run-pass-valgrind",
    default: false,
    host: true
});

default_test!(RunMake {
    path: "src/test/run-make",
    mode: "run-make",
    suite: "run-make"
});

host_test!(RunMakeFullDeps {
    path: "src/test/run-make-fulldeps",
    mode: "run-make",
    suite: "run-make-fulldeps"
});

default_test!(Assembly {
    path: "src/test/assembly",
    mode: "assembly",
    suite: "assembly"
});

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct Compiletest {
    compiler: Compiler,
    target: Interned<String>,
    mode: &'static str,
    suite: &'static str,
    path: Option<&'static str>,
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
        let compiler = self.compiler;
        let target = self.target;
        let mode = self.mode;
        let suite = self.suite;

        // Path for test suite
        let suite_path = self.path.unwrap_or("");

        // Skip codegen tests if they aren't enabled in configuration.
        if !builder.config.codegen_tests && suite == "codegen" {
            return;
        }

        if suite == "debuginfo" {
            let msvc = builder.config.build.contains("msvc");
            if mode == "debuginfo" {
                return builder.ensure(Compiletest {
                    mode: if msvc { "debuginfo-cdb" } else { "debuginfo-gdb+lldb" },
                    ..self
                });
            }

            builder.ensure(dist::DebuggerScripts {
                sysroot: builder.sysroot(compiler),
                host: target,
            });
        }

        if suite.ends_with("fulldeps") {
            builder.ensure(compile::Rustc { compiler, target });
        }

        if builder.no_std(target) == Some(true) {
            // the `test` doesn't compile for no-std targets
            builder.ensure(compile::Std { compiler, target });
        } else {
            builder.ensure(compile::Test { compiler, target });
        }

        if builder.no_std(target) == Some(true) {
            // for no_std run-make (e.g., thumb*),
            // we need a host compiler which is called by cargo.
            builder.ensure(compile::Std { compiler, target: compiler.host });
        }

        // HACK(eddyb) ensure that `libproc_macro` is available on the host.
        builder.ensure(compile::Test { compiler, target: compiler.host });
        // Also provide `rust_test_helpers` for the host.
        builder.ensure(native::TestHelpers { target: compiler.host });

        // wasm32 can't build the test helpers
        if !target.contains("wasm32") {
            builder.ensure(native::TestHelpers { target });
        }
        builder.ensure(RemoteCopyLibs { compiler, target });

        let mut cmd = builder.tool_cmd(Tool::Compiletest);

        // compiletest currently has... a lot of arguments, so let's just pass all
        // of them!

        cmd.arg("--compile-lib-path")
            .arg(builder.rustc_libdir(compiler));
        cmd.arg("--run-lib-path")
            .arg(builder.sysroot_libdir(compiler, target));
        cmd.arg("--rustc-path").arg(builder.rustc(compiler));

        let is_rustdoc = suite.ends_with("rustdoc-ui") || suite.ends_with("rustdoc-js");

        // Avoid depending on rustdoc when we don't need it.
        if mode == "rustdoc"
            || (mode == "run-make" && suite.ends_with("fulldeps"))
            || (mode == "ui" && is_rustdoc)
            || mode == "js-doc-test"
        {
            cmd.arg("--rustdoc-path")
                .arg(builder.rustdoc(compiler));
        }

        cmd.arg("--src-base")
            .arg(builder.src.join("src/test").join(suite));
        cmd.arg("--build-base")
            .arg(testdir(builder, compiler.host).join(suite));
        cmd.arg("--stage-id")
            .arg(format!("stage{}-{}", compiler.stage, target));
        cmd.arg("--mode").arg(mode);
        cmd.arg("--target").arg(target);
        cmd.arg("--host").arg(&*compiler.host);
        cmd.arg("--llvm-filecheck")
            .arg(builder.llvm_filecheck(builder.config.build));

        if builder.config.cmd.bless() {
            cmd.arg("--bless");
        }

        let compare_mode = builder.config.cmd.compare_mode().or_else(|| {
            if builder.config.test_compare_mode {
                self.compare_mode
            } else {
                None
            }
        });

        if let Some(ref pass) = builder.config.cmd.pass() {
            cmd.arg("--pass");
            cmd.arg(pass);
        }

        if let Some(ref nodejs) = builder.config.nodejs {
            cmd.arg("--nodejs").arg(nodejs);
        }

        let mut flags = if is_rustdoc {
            Vec::new()
        } else {
            vec!["-Crpath".to_string()]
        };
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
        hostflags.push(format!(
            "-Lnative={}",
            builder.test_helpers_out(compiler.host).display()
        ));
        cmd.arg("--host-rustcflags").arg(hostflags.join(" "));

        let mut targetflags = flags;
        targetflags.push(format!(
            "-Lnative={}",
            builder.test_helpers_out(target).display()
        ));
        cmd.arg("--target-rustcflags").arg(targetflags.join(" "));

        cmd.arg("--docck-python").arg(builder.python());

        if builder.config.build.ends_with("apple-darwin") {
            // Force /usr/bin/python on macOS for LLDB tests because we're loading the
            // LLDB plugin's compiled module which only works with the system python
            // (namely not Homebrew-installed python)
            cmd.arg("--lldb-python").arg("/usr/bin/python");
        } else {
            cmd.arg("--lldb-python").arg(builder.python());
        }

        if let Some(ref gdb) = builder.config.gdb {
            cmd.arg("--gdb").arg(gdb);
        }

        let run = |cmd: &mut Command| {
            cmd.output().map(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .lines().next().unwrap_or_else(|| {
                        panic!("{:?} failed {:?}", cmd, output)
                    }).to_string()
            })
        };
        let lldb_exe = if builder.config.lldb_enabled && !target.contains("emscripten") {
            // Test against the lldb that was just built.
            builder.llvm_out(target).join("bin").join("lldb")
        } else {
            PathBuf::from("lldb")
        };
        let lldb_version = Command::new(&lldb_exe)
            .arg("--version")
            .output()
            .map(|output| { String::from_utf8_lossy(&output.stdout).to_string() })
            .ok();
        if let Some(ref vers) = lldb_version {
            cmd.arg("--lldb-version").arg(vers);
            let lldb_python_dir = run(Command::new(&lldb_exe).arg("-P")).ok();
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
            .map(|p| {
                match p.strip_prefix(".") {
                    Ok(path) => path,
                    Err(_) => p,
                }
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

        if builder.config.llvm_enabled() {
            let llvm_config = builder.ensure(native::Llvm {
                target: builder.config.build,
                emscripten: false,
            });
            if !builder.config.dry_run {
                let llvm_version = output(Command::new(&llvm_config).arg("--version"));
                cmd.arg("--llvm-version").arg(llvm_version);
            }
            if !builder.is_rust_llvm(target) {
                cmd.arg("--system-llvm");
            }

            // Only pass correct values for these flags for the `run-make` suite as it
            // requires that a C++ compiler was configured which isn't always the case.
            if !builder.config.dry_run && suite == "run-make-fulldeps" {
                let llvm_components = output(Command::new(&llvm_config).arg("--components"));
                let llvm_cxxflags = output(Command::new(&llvm_config).arg("--cxxflags"));
                cmd.arg("--cc")
                    .arg(builder.cc(target))
                    .arg("--cxx")
                    .arg(builder.cxx(target).unwrap())
                    .arg("--cflags")
                    .arg(builder.cflags(target, GitRepo::Rustc).join(" "))
                    .arg("--llvm-components")
                    .arg(llvm_components.trim())
                    .arg("--llvm-cxxflags")
                    .arg(llvm_cxxflags.trim());
                if let Some(ar) = builder.ar(target) {
                    cmd.arg("--ar").arg(ar);
                }

                // The llvm/bin directory contains many useful cross-platform
                // tools. Pass the path to run-make tests so they can use them.
                let llvm_bin_path = llvm_config.parent()
                    .expect("Expected llvm-config to be contained in directory");
                assert!(llvm_bin_path.is_dir());
                cmd.arg("--llvm-bin-dir").arg(llvm_bin_path);

                // If LLD is available, add it to the PATH
                if builder.config.lld_enabled {
                    let lld_install_root = builder.ensure(native::Lld {
                        target: builder.config.build,
                    });

                    let lld_bin_path = lld_install_root.join("bin");

                    let old_path = env::var_os("PATH").unwrap_or_default();
                    let new_path = env::join_paths(std::iter::once(lld_bin_path)
                        .chain(env::split_paths(&old_path)))
                        .expect("Could not add LLD bin path to PATH");
                    cmd.env("PATH", new_path);
                }
            }
        }

        if suite != "run-make-fulldeps" {
            cmd.arg("--cc")
                .arg("")
                .arg("--cxx")
                .arg("")
                .arg("--cflags")
                .arg("")
                .arg("--llvm-components")
                .arg("")
                .arg("--llvm-cxxflags")
                .arg("");
        }

        if builder.remote_tested(target) {
            cmd.arg("--remote-test-client")
                .arg(builder.tool_exe(Tool::RemoteTestClient));
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

        if builder.config.sanitizers {
            cmd.env("RUSTC_SANITIZER_SUPPORT", "1");
        }

        if builder.config.profiler {
            cmd.env("RUSTC_PROFILER_SUPPORT", "1");
        }

        cmd.env("RUST_TEST_TMPDIR", builder.out.join("tmp"));

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

        builder.ci_env.force_coloring_in_ci(&mut cmd);

        let _folder = builder.fold_output(|| format!("test_{}", suite));
        builder.info(&format!(
            "Check compiletest suite={} mode={} ({} -> {})",
            suite, mode, &compiler.host, target
        ));
        let _time = util::timeit(&builder);
        try_run(builder, &mut cmd);

        if let Some(compare_mode) = compare_mode {
            cmd.arg("--compare-mode").arg(compare_mode);
            let _folder = builder.fold_output(|| format!("test_{}_{}", suite, compare_mode));
            builder.info(&format!(
                "Check compiletest suite={} mode={} compare_mode={} ({} -> {})",
                suite, mode, compare_mode, &compiler.host, target
            ));
            let _time = util::timeit(&builder);
            try_run(builder, &mut cmd);
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct DocTest {
    compiler: Compiler,
    path: &'static str,
    name: &'static str,
    is_ext_doc: bool,
}

impl Step for DocTest {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Runs `rustdoc --test` for all documentation in `src/doc`.
    ///
    /// This will run all tests in our markdown documentation (e.g., the book)
    /// located in `src/doc`. The `rustdoc` that's run is the one that sits next to
    /// `compiler`.
    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;

        builder.ensure(compile::Test {
            compiler,
            target: compiler.host,
        });

        // Do a breadth-first traversal of the `src/doc` directory and just run
        // tests for all files that end in `*.md`
        let mut stack = vec![builder.src.join(self.path)];
        let _time = util::timeit(&builder);
        let _folder = builder.fold_output(|| format!("test_{}", self.name));

        let mut files = Vec::new();
        while let Some(p) = stack.pop() {
            if p.is_dir() {
                stack.extend(t!(p.read_dir()).map(|p| t!(p).path()));
                continue;
            }

            if p.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }

            // The nostarch directory in the book is for no starch, and so isn't
            // guaranteed to builder. We don't care if it doesn't build, so skip it.
            if p.to_str().map_or(false, |p| p.contains("nostarch")) {
                continue;
            }

            files.push(p);
        }

        files.sort();

        let mut toolstate = ToolState::TestPass;
        for file in files {
            if !markdown_test(builder, compiler, &file) {
                toolstate = ToolState::TestFail;
            }
        }
        if self.is_ext_doc {
            builder.save_toolstate(self.name, toolstate);
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
                        compiler: run.builder.compiler(run.builder.top_stage, run.host),
                    });
                }

                fn run(self, builder: &Builder<'_>) {
                    builder.ensure(DocTest {
                        compiler: self.compiler,
                        path: $path,
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
        run.builder.ensure(ErrorIndex {
            compiler: run.builder.compiler(run.builder.top_stage, run.host),
        });
    }

    /// Runs the error index generator tool to execute the tests located in the error
    /// index.
    ///
    /// The `error_index_generator` tool lives in `src/tools` and is used to
    /// generate a markdown file from the error indexes of the code base which is
    /// then passed to `rustdoc --test`.
    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;

        builder.ensure(compile::Std {
            compiler,
            target: compiler.host,
        });

        let dir = testdir(builder, compiler.host);
        t!(fs::create_dir_all(&dir));
        let output = dir.join("error-index.md");

        let mut tool = tool::ErrorIndex::command(
            builder,
            builder.compiler(compiler.stage, builder.config.build),
        );
        tool.arg("markdown")
            .arg(&output)
            .env("CFG_BUILD", &builder.config.build)
            .env("RUSTC_ERROR_METADATA_DST", builder.extended_error_dir());

        let _folder = builder.fold_output(|| "test_error_index");
        builder.info(&format!("Testing error-index stage{}", compiler.stage));
        let _time = util::timeit(&builder);
        builder.run(&mut tool);
        markdown_test(builder, compiler, &output);
    }
}

fn markdown_test(builder: &Builder<'_>, compiler: Compiler, markdown: &Path) -> bool {
    match fs::read_to_string(markdown) {
        Ok(contents) => {
            if !contents.contains("```") {
                return true;
            }
        }
        Err(_) => {}
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
pub struct CrateLibrustc {
    compiler: Compiler,
    target: Interned<String>,
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
        let compiler = builder.compiler(builder.top_stage, run.host);

        for krate in builder.in_tree_crates("rustc-main") {
            if run.path.ends_with(&krate.path) {
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
    target: Interned<String>,
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
        let compiler = builder.compiler(builder.top_stage, run.host);

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
    pub target: Interned<String>,
    pub mode: Mode,
    pub test_kind: TestKind,
    pub krate: Interned<String>,
}

impl Step for Crate {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(mut run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run = run.krate("test");
        for krate in run.builder.in_tree_crates("std") {
            if !(krate.name.starts_with("rustc_") && krate.name.ends_with("san")) {
                run = run.path(krate.local_path(&builder).to_str().unwrap());
            }
        }
        run
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let compiler = builder.compiler(builder.top_stage, run.host);

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

        for krate in builder.in_tree_crates("std") {
            if run.path.ends_with(&krate.local_path(&builder)) {
                make(Mode::Std, krate);
            }
        }
        for krate in builder.in_tree_crates("test") {
            if run.path.ends_with(&krate.local_path(&builder)) {
                make(Mode::Test, krate);
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

        builder.ensure(compile::Test { compiler, target });
        builder.ensure(RemoteCopyLibs { compiler, target });

        // If we're not doing a full bootstrap but we're testing a stage2
        // version of libstd, then what we're actually testing is the libstd
        // produced in stage1. Reflect that here by updating the compiler that
        // we're working with automatically.
        let compiler = builder.compiler_for(compiler.stage, compiler.host, target);

        let mut cargo = builder.cargo(compiler, mode, target, test_kind.subcommand());
        match mode {
            Mode::Std => {
                compile::std_cargo(builder, &compiler, target, &mut cargo);
            }
            Mode::Test => {
                compile::test_cargo(builder, &compiler, target, &mut cargo);
            }
            Mode::Rustc => {
                builder.ensure(compile::Rustc { compiler, target });
                compile::rustc_cargo(builder, &mut cargo);
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
                format!("CARGO_TARGET_{}_RUNNER", envify(&target)),
                builder
                    .config
                    .nodejs
                    .as_ref()
                    .expect("nodejs not configured"),
            );
        } else if target.starts_with("wasm32") {
            // Warn about running tests without the `wasm_syscall` feature enabled.
            // The javascript shim implements the syscall interface so that test
            // output can be correctly reported.
            if !builder.config.wasm_syscall {
                builder.info(
                    "Libstd was built without `wasm_syscall` feature enabled: \
                     test output may not be visible."
                );
            }

            // On the wasm32-unknown-unknown target we're using LTO which is
            // incompatible with `-C prefer-dynamic`, so disable that here
            cargo.env("RUSTC_NO_PREFER_DYNAMIC", "1");

            let node = builder
                .config
                .nodejs
                .as_ref()
                .expect("nodejs not configured");
            let runner = format!(
                "{} {}/src/etc/wasm32-shim.js",
                node.display(),
                builder.src.display()
            );
            cargo.env(format!("CARGO_TARGET_{}_RUNNER", envify(&target)), &runner);
        } else if builder.remote_tested(target) {
            cargo.env(
                format!("CARGO_TARGET_{}_RUNNER", envify(&target)),
                format!("{} run", builder.tool_exe(Tool::RemoteTestClient).display()),
            );
        }

        let _folder = builder.fold_output(|| {
            format!(
                "{}_stage{}-{}",
                test_kind.subcommand(),
                compiler.stage,
                krate
            )
        });
        builder.info(&format!(
            "{} {} stage{} ({} -> {})",
            test_kind, krate, compiler.stage, &compiler.host, target
        ));
        let _time = util::timeit(&builder);
        try_run(builder, &mut cargo);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CrateRustdoc {
    host: Interned<String>,
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

        builder.ensure(CrateRustdoc {
            host: run.host,
            test_kind,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        let test_kind = self.test_kind;

        let compiler = builder.compiler(builder.top_stage, self.host);
        let target = compiler.host;
        builder.ensure(compile::Rustc { compiler, target });

        let mut cargo = tool::prepare_tool_cargo(builder,
                                                 compiler,
                                                 Mode::ToolRustc,
                                                 target,
                                                 test_kind.subcommand(),
                                                 "src/tools/rustdoc",
                                                 SourceType::InTree,
                                                 &[]);
        if test_kind.subcommand() == "test" && !builder.fail_fast {
            cargo.arg("--no-fail-fast");
        }

        cargo.arg("-p").arg("rustdoc:0.0.0");

        cargo.arg("--");
        cargo.args(&builder.config.cmd.test_args());

        if self.host.contains("musl") {
            cargo.arg("'-Ctarget-feature=-crt-static'");
        }

        if !builder.config.verbose_tests {
            cargo.arg("--quiet");
        }

        let _folder = builder
            .fold_output(|| format!("{}_stage{}-rustdoc", test_kind.subcommand(), compiler.stage));
        builder.info(&format!(
            "{} rustdoc stage{} ({} -> {})",
            test_kind, compiler.stage, &compiler.host, target
        ));
        let _time = util::timeit(&builder);

        try_run(builder, &mut cargo);
    }
}

fn envify(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '-' => '_',
            c => c,
        })
        .flat_map(|c| c.to_uppercase())
        .collect()
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
    target: Interned<String>,
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

        builder.ensure(compile::Test { compiler, target });

        builder.info(&format!("REMOTE copy libs to emulator ({})", target));
        t!(fs::create_dir_all(builder.out.join("tmp")));

        let server = builder.ensure(tool::RemoteTestServer {
            compiler: compiler.with_stage(0),
            target,
        });

        // Spawn the emulator and wait for it to come online
        let tool = builder.tool_exe(Tool::RemoteTestClient);
        let mut cmd = Command::new(&tool);
        cmd.arg("spawn-emulator")
            .arg(target)
            .arg(&server)
            .arg(builder.out.join("tmp"));
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
        cmd.arg("-xzf")
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
            Command::new(build_helper::make(&builder.config.build))
                .arg("check")
                .current_dir(&dir),
        );

        // Now make sure that rust-src has all of libstd's dependencies
        builder.info("Distcheck rust-src");
        let dir = builder.out.join("tmp").join("distcheck-src");
        let _ = fs::remove_dir_all(&dir);
        t!(fs::create_dir_all(&dir));

        let mut cmd = Command::new("tar");
        cmd.arg("-xzf")
            .arg(builder.ensure(dist::Src))
            .arg("--strip-components=1")
            .current_dir(&dir);
        builder.run(&mut cmd);

        let toml = dir.join("rust-src/lib/rustlib/src/rust/src/libstd/Cargo.toml");
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
