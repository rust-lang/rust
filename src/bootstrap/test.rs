// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of the test-related targets of the build system.
//!
//! This file implements the various regression test suites that we execute on
//! our CI.

use std::env;
use std::ffi::OsString;
use std::iter;
use std::fmt;
use std::fs::{self, File};
use std::path::{PathBuf, Path};
use std::process::Command;
use std::io::Read;

use build_helper::{self, output};

use builder::{Kind, RunConfig, ShouldRun, Builder, Compiler, Step};
use Crate as CargoCrate;
use cache::{INTERNER, Interned};
use compile;
use dist;
use native;
use tool::{self, Tool};
use util::{self, dylib_path, dylib_path_var};
use {Build, Mode};
use toolstate::ToolState;

const ADB_TEST_DIR: &str = "/data/tmp/work";

/// The two modes of the test runner; tests or benchmarks.
#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub enum TestKind {
    /// Run `cargo test`
    Test,
    /// Run `cargo bench`
    Bench,
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match *self {
            TestKind::Test => "Testing",
            TestKind::Bench => "Benchmarking",
        })
    }
}

fn try_run(build: &Build, cmd: &mut Command) -> bool {
    if !build.fail_fast {
        if !build.try_run(cmd) {
            let mut failures = build.delayed_failures.borrow_mut();
            failures.push(format!("{:?}", cmd));
            return false;
        }
    } else {
        build.run(cmd);
    }
    true
}

fn try_run_quiet(build: &Build, cmd: &mut Command) -> bool {
    if !build.fail_fast {
        if !build.try_run_quiet(cmd) {
            let mut failures = build.delayed_failures.borrow_mut();
            failures.push(format!("{:?}", cmd));
            return false;
        }
    } else {
        build.run_quiet(cmd);
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
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let host = self.host;

        build.info(&format!("Linkcheck ({})", host));

        builder.default_doc(None);

        let _time = util::timeit(&build);
        try_run(build, builder.tool_cmd(Tool::Linkchecker)
                              .arg(build.out.join(host).join("doc")));
    }

    fn should_run(run: ShouldRun) -> ShouldRun {
        let builder = run.builder;
        run.path("src/tools/linkchecker").default_condition(builder.build.config.docs)
    }

    fn make_run(run: RunConfig) {
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/tools/cargotest")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Cargotest {
            stage: run.builder.top_stage,
            host: run.target,
        });
    }

    /// Runs the `cargotest` tool as compiled in `stage` by the `host` compiler.
    ///
    /// This tool in `src/tools` will check out a few Rust projects and run `cargo
    /// test` to ensure that we don't regress the test suites there.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = builder.compiler(self.stage, self.host);
        builder.ensure(compile::Rustc { compiler, target: compiler.host });

        // Note that this is a short, cryptic, and not scoped directory name. This
        // is currently to minimize the length of path on Windows where we otherwise
        // quickly run into path name limit constraints.
        let out_dir = build.out.join("ct");
        t!(fs::create_dir_all(&out_dir));

        let _time = util::timeit(&build);
        let mut cmd = builder.tool_cmd(Tool::CargoTest);
        try_run(build, cmd.arg(&build.initial_cargo)
                          .arg(&out_dir)
                          .env("RUSTC", builder.rustc(compiler))
                          .env("RUSTDOC", builder.rustdoc(compiler.host)));
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/tools/cargo")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Cargo {
            stage: run.builder.top_stage,
            host: run.target,
        });
    }

    /// Runs `cargo test` for `cargo` packaged with Rust.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = builder.compiler(self.stage, self.host);

        builder.ensure(tool::Cargo { compiler, target: self.host });
        let mut cargo = builder.cargo(compiler, Mode::Tool, self.host, "test");
        cargo.arg("--manifest-path").arg(build.src.join("src/tools/cargo/Cargo.toml"));
        if !build.fail_fast {
            cargo.arg("--no-fail-fast");
        }

        // Don't build tests dynamically, just a pain to work with
        cargo.env("RUSTC_NO_PREFER_DYNAMIC", "1");

        // Don't run cross-compile tests, we may not have cross-compiled libstd libs
        // available.
        cargo.env("CFG_DISABLE_CROSS_TESTS", "1");

        try_run(build, cargo.env("PATH", &path_for_cargo(builder, compiler)));
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/tools/rls")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Rls {
            stage: run.builder.top_stage,
            host: run.target,
        });
    }

    /// Runs `cargo test` for the rls.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let stage = self.stage;
        let host = self.host;
        let compiler = builder.compiler(stage, host);

        builder.ensure(tool::Rls { compiler, target: self.host, extra_features: Vec::new() });
        let mut cargo = tool::prepare_tool_cargo(builder,
                                                 compiler,
                                                 host,
                                                 "test",
                                                 "src/tools/rls");

        // Don't build tests dynamically, just a pain to work with
        cargo.env("RUSTC_NO_PREFER_DYNAMIC", "1");

        builder.add_rustc_lib_path(compiler, &mut cargo);

        if try_run(build, &mut cargo) {
            build.save_toolstate("rls", ToolState::TestPass);
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/tools/rustfmt")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Rustfmt {
            stage: run.builder.top_stage,
            host: run.target,
        });
    }

    /// Runs `cargo test` for rustfmt.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let stage = self.stage;
        let host = self.host;
        let compiler = builder.compiler(stage, host);

        builder.ensure(tool::Rustfmt { compiler, target: self.host, extra_features: Vec::new() });
        let mut cargo = tool::prepare_tool_cargo(builder,
                                                 compiler,
                                                 host,
                                                 "test",
                                                 "src/tools/rustfmt");

        // Don't build tests dynamically, just a pain to work with
        cargo.env("RUSTC_NO_PREFER_DYNAMIC", "1");

        builder.add_rustc_lib_path(compiler, &mut cargo);

        if try_run(build, &mut cargo) {
            build.save_toolstate("rustfmt", ToolState::TestPass);
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        let test_miri = run.builder.build.config.test_miri;
        run.path("src/tools/miri").default_condition(test_miri)
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Miri {
            stage: run.builder.top_stage,
            host: run.target,
        });
    }

    /// Runs `cargo test` for miri.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let stage = self.stage;
        let host = self.host;
        let compiler = builder.compiler(stage, host);

        let miri = builder.ensure(tool::Miri {
            compiler,
            target: self.host,
            extra_features: Vec::new(),
        });
        if let Some(miri) = miri {
            let mut cargo = builder.cargo(compiler, Mode::Tool, host, "test");
            cargo.arg("--manifest-path").arg(build.src.join("src/tools/miri/Cargo.toml"));

            // Don't build tests dynamically, just a pain to work with
            cargo.env("RUSTC_NO_PREFER_DYNAMIC", "1");
            // miri tests need to know about the stage sysroot
            cargo.env("MIRI_SYSROOT", builder.sysroot(compiler));
            cargo.env("RUSTC_TEST_SUITE", builder.rustc(compiler));
            cargo.env("RUSTC_LIB_PATH", builder.rustc_libdir(compiler));
            cargo.env("MIRI_PATH", miri);

            builder.add_rustc_lib_path(compiler, &mut cargo);

            if try_run(build, &mut cargo) {
                build.save_toolstate("miri", ToolState::TestPass);
            }
        } else {
            eprintln!("failed to test miri: could not build");
        }
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/tools/clippy")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Clippy {
            stage: run.builder.top_stage,
            host: run.target,
        });
    }

    /// Runs `cargo test` for clippy.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let stage = self.stage;
        let host = self.host;
        let compiler = builder.compiler(stage, host);

        let clippy = builder.ensure(tool::Clippy {
            compiler,
            target: self.host,
            extra_features: Vec::new(),
        });
        if let Some(clippy) = clippy {
            let mut cargo = builder.cargo(compiler, Mode::Tool, host, "test");
            cargo.arg("--manifest-path").arg(build.src.join("src/tools/clippy/Cargo.toml"));

            // Don't build tests dynamically, just a pain to work with
            cargo.env("RUSTC_NO_PREFER_DYNAMIC", "1");
            // clippy tests need to know about the stage sysroot
            cargo.env("SYSROOT", builder.sysroot(compiler));
            cargo.env("RUSTC_TEST_SUITE", builder.rustc(compiler));
            cargo.env("RUSTC_LIB_PATH", builder.rustc_libdir(compiler));
            let host_libs = builder.stage_out(compiler, Mode::Tool).join(builder.cargo_dir());
            cargo.env("HOST_LIBS", host_libs);
            // clippy tests need to find the driver
            cargo.env("CLIPPY_DRIVER_PATH", clippy);

            builder.add_rustc_lib_path(compiler, &mut cargo);

            if try_run(build, &mut cargo) {
                build.save_toolstate("clippy-driver", ToolState::TestPass);
            }
        } else {
            eprintln!("failed to test clippy: could not build");
        }
    }
}

fn path_for_cargo(builder: &Builder, compiler: Compiler) -> OsString {
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/tools/rustdoc-themes")
    }

    fn make_run(run: RunConfig) {
        let compiler = run.builder.compiler(run.builder.top_stage, run.host);

        run.builder.ensure(RustdocTheme {
            compiler: compiler,
        });
    }

    fn run(self, builder: &Builder) {
        let rustdoc = builder.rustdoc(self.compiler.host);
        let mut cmd = builder.tool_cmd(Tool::RustdocTheme);
        cmd.arg(rustdoc.to_str().unwrap())
           .arg(builder.src.join("src/librustdoc/html/static/themes").to_str().unwrap())
           .env("RUSTC_STAGE", self.compiler.stage.to_string())
           .env("RUSTC_SYSROOT", builder.sysroot(self.compiler))
           .env("RUSTDOC_LIBDIR", builder.sysroot_libdir(self.compiler, self.compiler.host))
           .env("CFG_RELEASE_CHANNEL", &builder.build.config.channel)
           .env("RUSTDOC_REAL", builder.rustdoc(self.compiler.host))
           .env("RUSTDOC_CRATE_VERSION", builder.build.rust_version())
           .env("RUSTC_BOOTSTRAP", "1");
        if let Some(linker) = builder.build.linker(self.compiler.host) {
            cmd.env("RUSTC_TARGET_LINKER", linker);
        }
        try_run(builder.build, &mut cmd);
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct RustdocJS {
    pub host: Interned<String>,
    pub target: Interned<String>,
}

impl Step for RustdocJS {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/test/rustdoc-js")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(RustdocJS {
            host: run.host,
            target: run.target,
        });
    }

    fn run(self, builder: &Builder) {
        if let Some(ref nodejs) = builder.config.nodejs {
            let mut command = Command::new(nodejs);
            command.args(&["src/tools/rustdoc-js/tester.js", &*self.host]);
            builder.ensure(::doc::Std {
                target: self.target,
                stage: builder.top_stage,
            });
            builder.run(&mut command);
        } else {
            builder.info(&format!("No nodejs found, skipping \"src/test/rustdoc-js\" tests"));
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
    fn run(self, builder: &Builder) {
        let build = builder.build;

        let mut cmd = builder.tool_cmd(Tool::Tidy);
        cmd.arg(build.src.join("src"));
        cmd.arg(&build.initial_cargo);
        if !build.config.vendor {
            cmd.arg("--no-vendor");
        }
        if build.config.quiet_tests {
            cmd.arg("--quiet");
        }

        let _folder = build.fold_output(|| "tidy");
        builder.info(&format!("tidy check"));
        try_run(build, &mut cmd);
    }

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/tools/tidy")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Tidy);
    }
}

fn testdir(build: &Build, host: Interned<String>) -> PathBuf {
    build.out.join(host).join("test")
}

macro_rules! default_test {
    ($name:ident { path: $path:expr, mode: $mode:expr, suite: $suite:expr }) => {
        test!($name { path: $path, mode: $mode, suite: $suite, default: true, host: false });
    }
}

macro_rules! host_test {
    ($name:ident { path: $path:expr, mode: $mode:expr, suite: $suite:expr }) => {
        test!($name { path: $path, mode: $mode, suite: $suite, default: true, host: true });
    }
}

macro_rules! test {
    ($name:ident {
        path: $path:expr,
        mode: $mode:expr,
        suite: $suite:expr,
        default: $default:expr,
        host: $host:expr
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

            fn should_run(run: ShouldRun) -> ShouldRun {
                run.path($path)
            }

            fn make_run(run: RunConfig) {
                let compiler = run.builder.compiler(run.builder.top_stage, run.host);

                run.builder.ensure($name {
                    compiler,
                    target: run.target,
                });
            }

            fn run(self, builder: &Builder) {
                builder.ensure(Compiletest {
                    compiler: self.compiler,
                    target: self.target,
                    mode: $mode,
                    suite: $suite,
                })
            }
        }
    }
}

default_test!(Ui {
    path: "src/test/ui",
    mode: "ui",
    suite: "ui"
});

default_test!(RunPass {
    path: "src/test/run-pass",
    mode: "run-pass",
    suite: "run-pass"
});

default_test!(CompileFail {
    path: "src/test/compile-fail",
    mode: "compile-fail",
    suite: "compile-fail"
});

default_test!(ParseFail {
    path: "src/test/parse-fail",
    mode: "parse-fail",
    suite: "parse-fail"
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
    // What this runs varies depending on the native platform being apple
    mode: "debuginfo-XXX",
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

host_test!(RunFailFullDeps {
    path: "src/test/run-fail-fulldeps",
    mode: "run-fail",
    suite: "run-fail-fulldeps"
});

host_test!(CompileFailFullDeps {
    path: "src/test/compile-fail-fulldeps",
    mode: "compile-fail",
    suite: "compile-fail-fulldeps"
});

host_test!(IncrementalFullDeps {
    path: "src/test/incremental-fulldeps",
    mode: "incremental",
    suite: "incremental-fulldeps"
});

host_test!(Rustdoc {
    path: "src/test/rustdoc",
    mode: "rustdoc",
    suite: "rustdoc"
});

test!(Pretty {
    path: "src/test/pretty",
    mode: "pretty",
    suite: "pretty",
    default: false,
    host: true
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
test!(RunPassFullDepsPretty {
    path: "src/test/run-pass-fulldeps/pretty",
    mode: "pretty",
    suite: "run-pass-fulldeps",
    default: false,
    host: true
});
test!(RunFailFullDepsPretty {
    path: "src/test/run-fail-fulldeps/pretty",
    mode: "pretty",
    suite: "run-fail-fulldeps",
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct Compiletest {
    compiler: Compiler,
    target: Interned<String>,
    mode: &'static str,
    suite: &'static str,
}

impl Step for Compiletest {
    type Output = ();

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.never()
    }

    /// Executes the `compiletest` tool to run a suite of tests.
    ///
    /// Compiles all tests with `compiler` for `target` with the specified
    /// compiletest `mode` and `suite` arguments. For example `mode` can be
    /// "run-pass" or `suite` can be something like `debuginfo`.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = self.compiler;
        let target = self.target;
        let mode = self.mode;
        let suite = self.suite;

        // Skip codegen tests if they aren't enabled in configuration.
        if !build.config.codegen_tests && suite == "codegen" {
            return;
        }

        if suite == "debuginfo" {
            // Skip debuginfo tests on MSVC
            if build.build.contains("msvc") {
                return;
            }

            if mode == "debuginfo-XXX" {
                return if build.build.contains("apple") {
                    builder.ensure(Compiletest {
                        mode: "debuginfo-lldb",
                        ..self
                    });
                } else {
                    builder.ensure(Compiletest {
                        mode: "debuginfo-gdb",
                        ..self
                    });
                };
            }

            builder.ensure(dist::DebuggerScripts {
                sysroot: builder.sysroot(compiler),
                host: target
            });
        }

        if suite.ends_with("fulldeps") ||
            // FIXME: Does pretty need librustc compiled? Note that there are
            // fulldeps test suites with mode = pretty as well.
            mode == "pretty" ||
            mode == "rustdoc" {
            builder.ensure(compile::Rustc { compiler, target });
        }

        builder.ensure(compile::Test { compiler, target });
        builder.ensure(native::TestHelpers { target });
        builder.ensure(RemoteCopyLibs { compiler, target });

        let mut cmd = builder.tool_cmd(Tool::Compiletest);

        // compiletest currently has... a lot of arguments, so let's just pass all
        // of them!

        cmd.arg("--compile-lib-path").arg(builder.rustc_libdir(compiler));
        cmd.arg("--run-lib-path").arg(builder.sysroot_libdir(compiler, target));
        cmd.arg("--rustc-path").arg(builder.rustc(compiler));

        // Avoid depending on rustdoc when we don't need it.
        if mode == "rustdoc" || (mode == "run-make" && suite.ends_with("fulldeps")) {
            cmd.arg("--rustdoc-path").arg(builder.rustdoc(compiler.host));
        }

        cmd.arg("--src-base").arg(build.src.join("src/test").join(suite));
        cmd.arg("--build-base").arg(testdir(build, compiler.host).join(suite));
        cmd.arg("--stage-id").arg(format!("stage{}-{}", compiler.stage, target));
        cmd.arg("--mode").arg(mode);
        cmd.arg("--target").arg(target);
        cmd.arg("--host").arg(&*compiler.host);
        cmd.arg("--llvm-filecheck").arg(build.llvm_filecheck(build.build));

        if let Some(ref nodejs) = build.config.nodejs {
            cmd.arg("--nodejs").arg(nodejs);
        }

        let mut flags = vec!["-Crpath".to_string()];
        if build.config.rust_optimize_tests {
            flags.push("-O".to_string());
        }
        if build.config.rust_debuginfo_tests {
            flags.push("-g".to_string());
        }
        flags.push("-Zmiri -Zunstable-options".to_string());
        flags.push(build.config.cmd.rustc_args().join(" "));

        if let Some(linker) = build.linker(target) {
            cmd.arg("--linker").arg(linker);
        }

        let hostflags = flags.clone();
        cmd.arg("--host-rustcflags").arg(hostflags.join(" "));

        let mut targetflags = flags.clone();
        targetflags.push(format!("-Lnative={}",
                                 build.test_helpers_out(target).display()));
        cmd.arg("--target-rustcflags").arg(targetflags.join(" "));

        cmd.arg("--docck-python").arg(build.python());

        if build.build.ends_with("apple-darwin") {
            // Force /usr/bin/python on macOS for LLDB tests because we're loading the
            // LLDB plugin's compiled module which only works with the system python
            // (namely not Homebrew-installed python)
            cmd.arg("--lldb-python").arg("/usr/bin/python");
        } else {
            cmd.arg("--lldb-python").arg(build.python());
        }

        if let Some(ref gdb) = build.config.gdb {
            cmd.arg("--gdb").arg(gdb);
        }
        if let Some(ref vers) = build.lldb_version {
            cmd.arg("--lldb-version").arg(vers);
        }
        if let Some(ref dir) = build.lldb_python_dir {
            cmd.arg("--lldb-python-dir").arg(dir);
        }

        cmd.args(&build.config.cmd.test_args());

        if build.is_verbose() {
            cmd.arg("--verbose");
        }

        if build.config.quiet_tests {
            cmd.arg("--quiet");
        }

        if build.config.llvm_enabled {
            let llvm_config = builder.ensure(native::Llvm {
                target: build.config.build,
                emscripten: false,
            });
            if !build.config.dry_run {
                let llvm_version = output(Command::new(&llvm_config).arg("--version"));
                cmd.arg("--llvm-version").arg(llvm_version);
            }
            if !build.is_rust_llvm(target) {
                cmd.arg("--system-llvm");
            }

            // Only pass correct values for these flags for the `run-make` suite as it
            // requires that a C++ compiler was configured which isn't always the case.
            if !build.config.dry_run && suite == "run-make-fulldeps" {
                let llvm_components = output(Command::new(&llvm_config).arg("--components"));
                let llvm_cxxflags = output(Command::new(&llvm_config).arg("--cxxflags"));
                cmd.arg("--cc").arg(build.cc(target))
                .arg("--cxx").arg(build.cxx(target).unwrap())
                .arg("--cflags").arg(build.cflags(target).join(" "))
                .arg("--llvm-components").arg(llvm_components.trim())
                .arg("--llvm-cxxflags").arg(llvm_cxxflags.trim());
                if let Some(ar) = build.ar(target) {
                    cmd.arg("--ar").arg(ar);
                }
            }
        }
        if suite == "run-make-fulldeps" && !build.config.llvm_enabled {
            builder.info(
                &format!("Ignoring run-make test suite as they generally don't work without LLVM"));
            return;
        }

        if suite != "run-make-fulldeps" {
            cmd.arg("--cc").arg("")
               .arg("--cxx").arg("")
               .arg("--cflags").arg("")
               .arg("--llvm-components").arg("")
               .arg("--llvm-cxxflags").arg("");
        }

        if build.remote_tested(target) {
            cmd.arg("--remote-test-client").arg(builder.tool_exe(Tool::RemoteTestClient));
        }

        // Running a C compiler on MSVC requires a few env vars to be set, to be
        // sure to set them here.
        //
        // Note that if we encounter `PATH` we make sure to append to our own `PATH`
        // rather than stomp over it.
        if target.contains("msvc") {
            for &(ref k, ref v) in build.cc[&target].env() {
                if k != "PATH" {
                    cmd.env(k, v);
                }
            }
        }
        cmd.env("RUSTC_BOOTSTRAP", "1");
        build.add_rust_test_threads(&mut cmd);

        if build.config.sanitizers {
            cmd.env("SANITIZER_SUPPORT", "1");
        }

        if build.config.profiler {
            cmd.env("PROFILER_SUPPORT", "1");
        }

        cmd.env("RUST_TEST_TMPDIR", build.out.join("tmp"));

        cmd.arg("--adb-path").arg("adb");
        cmd.arg("--adb-test-dir").arg(ADB_TEST_DIR);
        if target.contains("android") {
            // Assume that cc for this target comes from the android sysroot
            cmd.arg("--android-cross-path")
               .arg(build.cc(target).parent().unwrap().parent().unwrap());
        } else {
            cmd.arg("--android-cross-path").arg("");
        }

        build.ci_env.force_coloring_in_ci(&mut cmd);

        let _folder = build.fold_output(|| format!("test_{}", suite));
        builder.info(&format!("Check compiletest suite={} mode={} ({} -> {})",
                 suite, mode, &compiler.host, target));
        let _time = util::timeit(&build);
        try_run(build, &mut cmd);
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.never()
    }

    /// Run `rustdoc --test` for all documentation in `src/doc`.
    ///
    /// This will run all tests in our markdown documentation (e.g. the book)
    /// located in `src/doc`. The `rustdoc` that's run is the one that sits next to
    /// `compiler`.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = self.compiler;

        builder.ensure(compile::Test { compiler, target: compiler.host });

        // Do a breadth-first traversal of the `src/doc` directory and just run
        // tests for all files that end in `*.md`
        let mut stack = vec![build.src.join(self.path)];
        let _time = util::timeit(&build);
        let _folder = build.fold_output(|| format!("test_{}", self.name));

        let mut files = Vec::new();
        while let Some(p) = stack.pop() {
            if p.is_dir() {
                stack.extend(t!(p.read_dir()).map(|p| t!(p).path()));
                continue
            }

            if p.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }

            // The nostarch directory in the book is for no starch, and so isn't
            // guaranteed to build. We don't care if it doesn't build, so skip it.
            if p.to_str().map_or(false, |p| p.contains("nostarch")) {
                continue;
            }

            files.push(p);
        }

        files.sort();

        for file in files {
            let test_result = markdown_test(builder, compiler, &file);
            if self.is_ext_doc {
                let toolstate = if test_result {
                    ToolState::TestPass
                } else {
                    ToolState::TestFail
                };
                build.save_toolstate(self.name, toolstate);
            }
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

                fn should_run(run: ShouldRun) -> ShouldRun {
                    run.path($path)
                }

                fn make_run(run: RunConfig) {
                    run.builder.ensure($name {
                        compiler: run.builder.compiler(run.builder.top_stage, run.host),
                    });
                }

                fn run(self, builder: &Builder) {
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
    RustByExample, "src/doc/rust-by-example", "rust-by-example", default=false;
    TheBook, "src/doc/book", "book", default=false;
    UnstableBook, "src/doc/unstable-book", "unstable-book", default=true;
);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ErrorIndex {
    compiler: Compiler,
}

impl Step for ErrorIndex {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/tools/error_index_generator")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(ErrorIndex {
            compiler: run.builder.compiler(run.builder.top_stage, run.host),
        });
    }

    /// Run the error index generator tool to execute the tests located in the error
    /// index.
    ///
    /// The `error_index_generator` tool lives in `src/tools` and is used to
    /// generate a markdown file from the error indexes of the code base which is
    /// then passed to `rustdoc --test`.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = self.compiler;

        builder.ensure(compile::Std { compiler, target: compiler.host });

        let dir = testdir(build, compiler.host);
        t!(fs::create_dir_all(&dir));
        let output = dir.join("error-index.md");

        let mut tool = builder.tool_cmd(Tool::ErrorIndex);
        tool.arg("markdown")
            .arg(&output)
            .env("CFG_BUILD", &build.build)
            .env("RUSTC_ERROR_METADATA_DST", build.extended_error_dir());


        let _folder = build.fold_output(|| "test_error_index");
        build.info(&format!("Testing error-index stage{}", compiler.stage));
        let _time = util::timeit(&build);
        build.run(&mut tool);
        markdown_test(builder, compiler, &output);
    }
}

fn markdown_test(builder: &Builder, compiler: Compiler, markdown: &Path) -> bool {
    let build = builder.build;
    match File::open(markdown) {
        Ok(mut file) => {
            let mut contents = String::new();
            t!(file.read_to_string(&mut contents));
            if !contents.contains("```") {
                return true;
            }
        }
        Err(_) => {},
    }

    build.info(&format!("doc tests for: {}", markdown.display()));
    let mut cmd = builder.rustdoc_cmd(compiler.host);
    build.add_rust_test_threads(&mut cmd);
    cmd.arg("--test");
    cmd.arg(markdown);
    cmd.env("RUSTC_BOOTSTRAP", "1");

    let test_args = build.config.cmd.test_args().join(" ");
    cmd.arg("--test-args").arg(test_args);

    if build.config.quiet_tests {
        try_run_quiet(build, &mut cmd)
    } else {
        try_run(build, &mut cmd)
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.krate("rustc-main")
    }

    fn make_run(run: RunConfig) {
        let builder = run.builder;
        let compiler = builder.compiler(builder.top_stage, run.host);

        for krate in builder.in_tree_crates("rustc-main") {
            if run.path.ends_with(&krate.path) {
                let test_kind = if builder.kind == Kind::Test {
                    TestKind::Test
                } else if builder.kind == Kind::Bench {
                    TestKind::Bench
                } else {
                    panic!("unexpected builder.kind in crate: {:?}", builder.kind);
                };

                builder.ensure(CrateLibrustc {
                    compiler,
                    target: run.target,
                    test_kind,
                    krate: krate.name,
                });
            }
        }
    }

    fn run(self, builder: &Builder) {
        builder.ensure(Crate {
            compiler: self.compiler,
            target: self.target,
            mode: Mode::Librustc,
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/liballoc_jemalloc")
            .path("src/librustc_asan")
            .path("src/librustc_lsan")
            .path("src/librustc_msan")
            .path("src/librustc_tsan")
    }

    fn make_run(run: RunConfig) {
        let builder = run.builder;
        let compiler = builder.compiler(builder.top_stage, run.host);

        let test_kind = if builder.kind == Kind::Test {
            TestKind::Test
        } else if builder.kind == Kind::Bench {
            TestKind::Bench
        } else {
            panic!("unexpected builder.kind in crate: {:?}", builder.kind);
        };

        builder.ensure(CrateNotDefault {
            compiler,
            target: run.target,
            test_kind,
            krate: match run.path {
                _ if run.path.ends_with("src/liballoc_jemalloc") => "alloc_jemalloc",
                _ if run.path.ends_with("src/librustc_asan") => "rustc_asan",
                _ if run.path.ends_with("src/librustc_lsan") => "rustc_lsan",
                _ if run.path.ends_with("src/librustc_msan") => "rustc_msan",
                _ if run.path.ends_with("src/librustc_tsan") => "rustc_tsan",
                _ => panic!("unexpected path {:?}", run.path),
            },
        });
    }

    fn run(self, builder: &Builder) {
        builder.ensure(Crate {
            compiler: self.compiler,
            target: self.target,
            mode: Mode::Libstd,
            test_kind: self.test_kind,
            krate: INTERNER.intern_str(self.krate),
        });
    }
}


#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Crate {
    compiler: Compiler,
    target: Interned<String>,
    mode: Mode,
    test_kind: TestKind,
    krate: Interned<String>,
}

impl Step for Crate {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(mut run: ShouldRun) -> ShouldRun {
        let builder = run.builder;
        run = run.krate("test");
        for krate in run.builder.in_tree_crates("std") {
            if krate.is_local(&run.builder) &&
                !krate.name.contains("jemalloc") &&
                !(krate.name.starts_with("rustc_") && krate.name.ends_with("san")) &&
                krate.name != "dlmalloc" {
                run = run.path(krate.local_path(&builder).to_str().unwrap());
            }
        }
        run
    }

    fn make_run(run: RunConfig) {
        let builder = run.builder;
        let compiler = builder.compiler(builder.top_stage, run.host);

        let make = |mode: Mode, krate: &CargoCrate| {
            let test_kind = if builder.kind == Kind::Test {
                TestKind::Test
            } else if builder.kind == Kind::Bench {
                TestKind::Bench
            } else {
                panic!("unexpected builder.kind in crate: {:?}", builder.kind);
            };

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
                make(Mode::Libstd, krate);
            }
        }
        for krate in builder.in_tree_crates("test") {
            if run.path.ends_with(&krate.local_path(&builder)) {
                make(Mode::Libtest, krate);
            }
        }
    }

    /// Run all unit tests plus documentation tests for a given crate defined
    /// by a `Cargo.toml` (single manifest)
    ///
    /// This is what runs tests for crates like the standard library, compiler, etc.
    /// It essentially is the driver for running `cargo test`.
    ///
    /// Currently this runs all tests for a DAG by passing a bunch of `-p foo`
    /// arguments, and those arguments are discovered from `cargo metadata`.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = self.compiler;
        let target = self.target;
        let mode = self.mode;
        let test_kind = self.test_kind;
        let krate = self.krate;

        builder.ensure(compile::Test { compiler, target });
        builder.ensure(RemoteCopyLibs { compiler, target });

        // If we're not doing a full bootstrap but we're testing a stage2 version of
        // libstd, then what we're actually testing is the libstd produced in
        // stage1. Reflect that here by updating the compiler that we're working
        // with automatically.
        let compiler = if build.force_use_stage1(compiler, target) {
            builder.compiler(1, compiler.host)
        } else {
            compiler.clone()
        };

        let mut cargo = builder.cargo(compiler, mode, target, test_kind.subcommand());
        match mode {
            Mode::Libstd => {
                compile::std_cargo(builder, &compiler, target, &mut cargo);
            }
            Mode::Libtest => {
                compile::test_cargo(build, &compiler, target, &mut cargo);
            }
            Mode::Librustc => {
                builder.ensure(compile::Rustc { compiler, target });
                compile::rustc_cargo(build, &mut cargo);
            }
            _ => panic!("can only test libraries"),
        };

        // Build up the base `cargo test` command.
        //
        // Pass in some standard flags then iterate over the graph we've discovered
        // in `cargo metadata` with the maps above and figure out what `-p`
        // arguments need to get passed.
        if test_kind.subcommand() == "test" && !build.fail_fast {
            cargo.arg("--no-fail-fast");
        }
        if build.doc_tests {
            cargo.arg("--doc");
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
        cargo.args(&build.config.cmd.test_args());

        if build.config.quiet_tests {
            cargo.arg("--quiet");
        }

        if target.contains("emscripten") {
            cargo.env(format!("CARGO_TARGET_{}_RUNNER", envify(&target)),
                      build.config.nodejs.as_ref().expect("nodejs not configured"));
        } else if target.starts_with("wasm32") {
            // Warn about running tests without the `wasm_syscall` feature enabled.
            // The javascript shim implements the syscall interface so that test
            // output can be correctly reported.
            if !build.config.wasm_syscall {
                build.info(&format!("Libstd was built without `wasm_syscall` feature enabled: \
                          test output may not be visible."));
            }

            // On the wasm32-unknown-unknown target we're using LTO which is
            // incompatible with `-C prefer-dynamic`, so disable that here
            cargo.env("RUSTC_NO_PREFER_DYNAMIC", "1");

            let node = build.config.nodejs.as_ref()
                .expect("nodejs not configured");
            let runner = format!("{} {}/src/etc/wasm32-shim.js",
                                 node.display(),
                                 build.src.display());
            cargo.env(format!("CARGO_TARGET_{}_RUNNER", envify(&target)), &runner);
        } else if build.remote_tested(target) {
            cargo.env(format!("CARGO_TARGET_{}_RUNNER", envify(&target)),
                      format!("{} run",
                              builder.tool_exe(Tool::RemoteTestClient).display()));
        }

        let _folder = build.fold_output(|| {
            format!("{}_stage{}-{}", test_kind.subcommand(), compiler.stage, krate)
        });
        build.info(&format!("{} {} stage{} ({} -> {})", test_kind, krate, compiler.stage,
                &compiler.host, target));
        let _time = util::timeit(&build);
        try_run(build, &mut cargo);
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.paths(&["src/librustdoc", "src/tools/rustdoc"])
    }

    fn make_run(run: RunConfig) {
        let builder = run.builder;

        let test_kind = if builder.kind == Kind::Test {
            TestKind::Test
        } else if builder.kind == Kind::Bench {
            TestKind::Bench
        } else {
            panic!("unexpected builder.kind in crate: {:?}", builder.kind);
        };

        builder.ensure(CrateRustdoc {
            host: run.host,
            test_kind,
        });
    }

    fn run(self, builder: &Builder) {
        let build = builder.build;
        let test_kind = self.test_kind;

        let compiler = builder.compiler(builder.top_stage, self.host);
        let target = compiler.host;

        let mut cargo = tool::prepare_tool_cargo(builder,
                                                 compiler,
                                                 target,
                                                 test_kind.subcommand(),
                                                 "src/tools/rustdoc");
        if test_kind.subcommand() == "test" && !build.fail_fast {
            cargo.arg("--no-fail-fast");
        }

        cargo.arg("-p").arg("rustdoc:0.0.0");

        cargo.arg("--");
        cargo.args(&build.config.cmd.test_args());

        if build.config.quiet_tests {
            cargo.arg("--quiet");
        }

        let _folder = build.fold_output(|| {
            format!("{}_stage{}-rustdoc", test_kind.subcommand(), compiler.stage)
        });
        build.info(&format!("{} rustdoc stage{} ({} -> {})", test_kind, compiler.stage,
                &compiler.host, target));
        let _time = util::timeit(&build);

        try_run(build, &mut cargo);
    }
}

fn envify(s: &str) -> String {
    s.chars().map(|c| {
        match c {
            '-' => '_',
            c => c,
        }
    }).flat_map(|c| c.to_uppercase()).collect()
}

/// Some test suites are run inside emulators or on remote devices, and most
/// of our test binaries are linked dynamically which means we need to ship
/// the standard library and such to the emulator ahead of time. This step
/// represents this and is a dependency of all test suites.
///
/// Most of the time this is a noop. For some steps such as shipping data to
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.never()
    }

    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = self.compiler;
        let target = self.target;
        if !build.remote_tested(target) {
            return
        }

        builder.ensure(compile::Test { compiler, target });

        build.info(&format!("REMOTE copy libs to emulator ({})", target));
        t!(fs::create_dir_all(build.out.join("tmp")));

        let server = builder.ensure(tool::RemoteTestServer { compiler, target });

        // Spawn the emulator and wait for it to come online
        let tool = builder.tool_exe(Tool::RemoteTestClient);
        let mut cmd = Command::new(&tool);
        cmd.arg("spawn-emulator")
           .arg(target)
           .arg(&server)
           .arg(build.out.join("tmp"));
        if let Some(rootfs) = build.qemu_rootfs(target) {
            cmd.arg(rootfs);
        }
        build.run(&mut cmd);

        // Push all our dylibs to the emulator
        for f in t!(builder.sysroot_libdir(compiler, target).read_dir()) {
            let f = t!(f);
            let name = f.file_name().into_string().unwrap();
            if util::is_dylib(&name) {
                build.run(Command::new(&tool)
                                  .arg("push")
                                  .arg(f.path()));
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Distcheck;

impl Step for Distcheck {
    type Output = ();

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("distcheck")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Distcheck);
    }

    /// Run "distcheck", a 'make check' from a tarball
    fn run(self, builder: &Builder) {
        let build = builder.build;

        build.info(&format!("Distcheck"));
        let dir = build.out.join("tmp").join("distcheck");
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
        build.run(&mut cmd);
        build.run(Command::new("./configure")
                         .args(&build.config.configure_args)
                         .arg("--enable-vendor")
                         .current_dir(&dir));
        build.run(Command::new(build_helper::make(&build.build))
                         .arg("check")
                         .current_dir(&dir));

        // Now make sure that rust-src has all of libstd's dependencies
        build.info(&format!("Distcheck rust-src"));
        let dir = build.out.join("tmp").join("distcheck-src");
        let _ = fs::remove_dir_all(&dir);
        t!(fs::create_dir_all(&dir));

        let mut cmd = Command::new("tar");
        cmd.arg("-xzf")
           .arg(builder.ensure(dist::Src))
           .arg("--strip-components=1")
           .current_dir(&dir);
        build.run(&mut cmd);

        let toml = dir.join("rust-src/lib/rustlib/src/rust/src/libstd/Cargo.toml");
        build.run(Command::new(&build.initial_cargo)
                         .arg("generate-lockfile")
                         .arg("--manifest-path")
                         .arg(&toml)
                         .current_dir(&dir));
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Bootstrap;

impl Step for Bootstrap {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    /// Test the build system itself
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let mut cmd = Command::new(&build.initial_cargo);
        cmd.arg("test")
           .current_dir(build.src.join("src/bootstrap"))
           .env("RUSTFLAGS", "-Cdebuginfo=2")
           .env("CARGO_TARGET_DIR", build.out.join("bootstrap"))
           .env("RUSTC_BOOTSTRAP", "1")
           .env("RUSTC", &build.initial_rustc);
        if let Some(flags) = option_env!("RUSTFLAGS") {
            // Use the same rustc flags for testing as for "normal" compilation,
            // so that Cargo doesn’t recompile the entire dependency graph every time:
            // https://github.com/rust-lang/rust/issues/49215
            cmd.env("RUSTFLAGS", flags);
        }
        if !build.fail_fast {
            cmd.arg("--no-fail-fast");
        }
        cmd.arg("--").args(&build.config.cmd.test_args());
        try_run(build, &mut cmd);
    }

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/bootstrap")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Bootstrap);
    }
}
