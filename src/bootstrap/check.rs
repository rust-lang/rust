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

use std::collections::HashSet;
use std::env;
use std::ffi::OsString;
use std::iter;
use std::fmt;
use std::fs::{self, File};
use std::path::{PathBuf, Path};
use std::process::Command;
use std::io::Read;

use build_helper::{self, output};

use {Build, Mode};
use dist;
use util::{self, dylib_path, dylib_path_var};

use compile;
use native;
use builder::{Kind, Builder, Compiler, Step};
use tool::Tool;

const ADB_TEST_DIR: &str = "/data/tmp/work";

/// The two modes of the test runner; tests or benchmarks.
#[derive(Copy, Clone)]
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

fn try_run(build: &Build, cmd: &mut Command) {
    if !build.fail_fast {
        if !build.try_run(cmd) {
            let failures = build.delayed_failures.get();
            build.delayed_failures.set(failures + 1);
        }
    } else {
        build.run(cmd);
    }
}

fn try_run_quiet(build: &Build, cmd: &mut Command) {
    if !build.fail_fast {
        if !build.try_run_quiet(cmd) {
            let failures = build.delayed_failures.get();
            build.delayed_failures.set(failures + 1);
        }
    } else {
        build.run_quiet(cmd);
    }
}

// rules.test("check-linkchecker", "src/tools/linkchecker")
//      .dep(|s| s.name("tool-linkchecker").stage(0))
//      .dep(|s| s.name("default:doc"))
//      .default(build.config.docs)
//      .host(true)
//      .run(move |s| check::linkcheck(build, s.target));

#[derive(Serialize)]
pub struct Linkcheck<'a> {
    host: &'a str,
}

impl<'a> Step<'a> for Linkcheck<'a> {
    type Output = ();

    /// Runs the `linkchecker` tool as compiled in `stage` by the `host` compiler.
    ///
    /// This tool in `src/tools` will verify the validity of all our links in the
    /// documentation to ensure we don't have a bunch of dead ones.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let host = self.host;

        println!("Linkcheck ({})", host);
        let compiler = Compiler::new(0, host);

        let _time = util::timeit();
        try_run(build, build.tool_cmd(&compiler, "linkchecker")
                            .arg(build.out.join(host).join("doc")));
    }
}

// rules.test("check-cargotest", "src/tools/cargotest")
//      .dep(|s| s.name("tool-cargotest").stage(0))
//      .dep(|s| s.name("librustc"))
//      .host(true)
//      .run(move |s| check::cargotest(build, s.stage, s.target));

#[derive(Serialize)]
pub struct Cargotest<'a> {
    stage: u32,
    host: &'a str,
}

impl<'a> Step<'a> for Cargotest<'a> {
    type Output = ();

    /// Runs the `cargotest` tool as compiled in `stage` by the `host` compiler.
    ///
    /// This tool in `src/tools` will check out a few Rust projects and run `cargo
    /// test` to ensure that we don't regress the test suites there.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let stage = self.stage;
        let host = self.host;
        let compiler = Compiler::new(stage, host);

        // Note that this is a short, cryptic, and not scoped directory name. This
        // is currently to minimize the length of path on Windows where we otherwise
        // quickly run into path name limit constraints.
        let out_dir = build.out.join("ct");
        t!(fs::create_dir_all(&out_dir));

        let _time = util::timeit();
        let mut cmd = Command::new(build.tool(&Compiler::new(0, host), "cargotest"));
        build.prepare_tool_cmd(&compiler, &mut cmd);
        try_run(build, cmd.arg(&build.initial_cargo)
                          .arg(&out_dir)
                          .env("RUSTC", build.compiler_path(&compiler))
                          .env("RUSTDOC", build.rustdoc(&compiler)));
    }
}

//rules.test("check-cargo", "cargo")
//     .dep(|s| s.name("tool-cargo"))
//     .host(true)
//     .run(move |s| check::cargo(build, s.stage, s.target));

#[derive(Serialize)]
pub struct Cargo<'a> {
    stage: u32,
    host: &'a str,
}

impl<'a> Step<'a> for Cargo<'a> {
    type Output = ();

    /// Runs `cargo test` for `cargo` packaged with Rust.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let stage = self.stage;
        let ref compiler = Compiler::new(stage, host);

        // Configure PATH to find the right rustc. NB. we have to use PATH
        // and not RUSTC because the Cargo test suite has tests that will
        // fail if rustc is not spelled `rustc`.
        let path = build.sysroot(compiler).join("bin");
        let old_path = env::var_os("PATH").unwrap_or_default();
        let newpath = env::join_paths(iter::once(path).chain(env::split_paths(&old_path))).expect("");

        let mut cargo = build.cargo(compiler, Mode::Tool, host, "test");
        cargo.arg("--manifest-path").arg(build.src.join("src/tools/cargo/Cargo.toml"));
        if !build.fail_fast {
            cargo.arg("--no-fail-fast");
        }

        let compiler = &Compiler::new(stage, host);

        let mut cargo = build.cargo(compiler, Mode::Tool, host, "test");
        cargo.arg("--manifest-path").arg(build.src.join("src/tools/cargo/Cargo.toml"));
        if !build.fail_fast {
            cargo.arg("--no-fail-fast");
        }

        // Don't build tests dynamically, just a pain to work with
        cargo.env("RUSTC_NO_PREFER_DYNAMIC", "1");

        // Don't run cross-compile tests, we may not have cross-compiled libstd libs
        // available.
        cargo.env("CFG_DISABLE_CROSS_TESTS", "1");

        try_run(build, cargo.env("PATH", &path_for_cargo(build, compiler)));
    }
}

#[derive(Serialize)]
pub struct Rls<'a> {
    stage: u32,
    host: &'a str,
}

impl<'a> Step<'a> for Rls<'a> {
    type Output = ();

    /// Runs `cargo test` for the rls.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let stage = self.stage;
        let host = self.host;
        let compiler = &Compiler::new(stage, host);

        let mut cargo = build.cargo(compiler, Mode::Tool, host, "test");
        cargo.arg("--manifest-path").arg(build.src.join("src/tools/rls/Cargo.toml"));

        // Don't build tests dynamically, just a pain to work with
        cargo.env("RUSTC_NO_PREFER_DYNAMIC", "1");

        build.add_rustc_lib_path(compiler, &mut cargo);

        try_run(build, &mut cargo);
    }
}

fn path_for_cargo(build: &Build, compiler: &Compiler) -> OsString {
    // Configure PATH to find the right rustc. NB. we have to use PATH
    // and not RUSTC because the Cargo test suite has tests that will
    // fail if rustc is not spelled `rustc`.
    let path = build.sysroot(compiler).join("bin");
    let old_path = env::var_os("PATH").unwrap_or_default();
    env::join_paths(iter::once(path).chain(env::split_paths(&old_path))).expect("")
||||||| parent of adabe3889e... Move code into Step trait implementations.
    try_run(build, cargo.env("PATH", newpath));
=======
        try_run(build, cargo.env("PATH", newpath));
        let host = self.host;
    }
>>>>>>> adabe3889e... Move code into Step trait implementations.
}

//rules.test("check-tidy", "src/tools/tidy")
//     .dep(|s| s.name("tool-tidy").stage(0))
//     .default(true)
//     .host(true)
//     .only_build(true)
//     .run(move |s| check::tidy(build, s.target));

#[derive(Serialize)]
pub struct Tidy<'a> {
    host: &'a str,
}

impl<'a> Step<'a> for Tidy<'a> {
    type Output = ();

    /// Runs the `tidy` tool as compiled in `stage` by the `host` compiler.
    ///
    /// This tool in `src/tools` checks up on various bits and pieces of style and
    /// otherwise just implements a few lint-like checks that are specific to the
    /// compiler itself.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let host = self.host;

        let _folder = build.fold_output(|| "tidy");
        println!("tidy check ({})", host);
        let compiler = Compiler::new(0, host);
        let mut cmd = build.tool_cmd(&compiler, "tidy");
        cmd.arg(build.src.join("src"));
        if !build.config.vendor {
            cmd.arg("--no-vendor");
        }
        if build.config.quiet_tests {
            cmd.arg("--quiet");
        }
        try_run(build, &mut cmd);
    }
}

fn testdir(build: &Build, host: &str) -> PathBuf {
    build.out.join(host).join("test")
}

//    // ========================================================================
//    // Test targets
//    //
//    // Various unit tests and tests suites we can run
//    {
//        let mut suite = |name, path, mode, dir| {
//            rules.test(name, path)
//                 .dep(|s| s.name("libtest"))
//                 .dep(|s| s.name("tool-compiletest").target(s.host).stage(0))
//                 .dep(|s| s.name("test-helpers"))
//                 .dep(|s| s.name("remote-copy-libs"))
//                 .default(mode != "pretty") // pretty tests don't run everywhere
//                 .run(move |s| {
//                     check::compiletest(build, &s.compiler(), s.target, mode, dir)
//                 });
//        };
//
//        suite("check-ui", "src/test/ui", "ui", "ui");
//        suite("check-rpass", "src/test/run-pass", "run-pass", "run-pass");
//        suite("check-cfail", "src/test/compile-fail", "compile-fail", "compile-fail");
//        suite("check-pfail", "src/test/parse-fail", "parse-fail", "parse-fail");
//        suite("check-rfail", "src/test/run-fail", "run-fail", "run-fail");
//        suite("check-rpass-valgrind", "src/test/run-pass-valgrind",
//              "run-pass-valgrind", "run-pass-valgrind");
//        suite("check-mir-opt", "src/test/mir-opt", "mir-opt", "mir-opt");
//        if build.config.codegen_tests {
//            suite("check-codegen", "src/test/codegen", "codegen", "codegen");
//        }
//        suite("check-codegen-units", "src/test/codegen-units", "codegen-units",
//              "codegen-units");
//        suite("check-incremental", "src/test/incremental", "incremental",
//              "incremental");
//    }
//
//    if build.build.contains("msvc") {
//        // nothing to do for debuginfo tests
//    } else {
//        rules.test("check-debuginfo-lldb", "src/test/debuginfo-lldb")
//             .dep(|s| s.name("libtest"))
//             .dep(|s| s.name("tool-compiletest").target(s.host).stage(0))
//             .dep(|s| s.name("test-helpers"))
//             .dep(|s| s.name("debugger-scripts"))
//             .run(move |s| check::compiletest(build, &s.compiler(), s.target,
//                                         "debuginfo-lldb", "debuginfo"));
//        rules.test("check-debuginfo-gdb", "src/test/debuginfo-gdb")
//             .dep(|s| s.name("libtest"))
//             .dep(|s| s.name("tool-compiletest").target(s.host).stage(0))
//             .dep(|s| s.name("test-helpers"))
//             .dep(|s| s.name("debugger-scripts"))
//             .dep(|s| s.name("remote-copy-libs"))
//             .run(move |s| check::compiletest(build, &s.compiler(), s.target,
//                                         "debuginfo-gdb", "debuginfo"));
//        let mut rule = rules.test("check-debuginfo", "src/test/debuginfo");
//        rule.default(true);
//        if build.build.contains("apple") {
//            rule.dep(|s| s.name("check-debuginfo-lldb"));
//        } else {
//            rule.dep(|s| s.name("check-debuginfo-gdb"));
//        }
//    }
//
//
//
//    {
//        let mut suite = |name, path, mode, dir| {
//            rules.test(name, path)
//                 .dep(|s| s.name("librustc"))
//                 .dep(|s| s.name("test-helpers"))
//                 .dep(|s| s.name("tool-compiletest").target(s.host).stage(0))
//                 .default(mode != "pretty")
//                 .host(true)
//                 .run(move |s| {
//                     check::compiletest(build, &s.compiler(), s.target, mode, dir)
//                 });
//        };
//
//        suite("check-ui-full", "src/test/ui-fulldeps", "ui", "ui-fulldeps");
//        suite("check-rpass-full", "src/test/run-pass-fulldeps",
//              "run-pass", "run-pass-fulldeps");
//        suite("check-rfail-full", "src/test/run-fail-fulldeps",
//              "run-fail", "run-fail-fulldeps");
//        suite("check-cfail-full", "src/test/compile-fail-fulldeps",
//              "compile-fail", "compile-fail-fulldeps");
//        suite("check-rmake", "src/test/run-make", "run-make", "run-make");
//        suite("check-rustdoc", "src/test/rustdoc", "rustdoc", "rustdoc");
//        suite("check-pretty", "src/test/pretty", "pretty", "pretty");
//        suite("check-pretty-rpass", "src/test/run-pass/pretty", "pretty",
//              "run-pass");
//        suite("check-pretty-rfail", "src/test/run-fail/pretty", "pretty",
//              "run-fail");
//        suite("check-pretty-valgrind", "src/test/run-pass-valgrind/pretty", "pretty",
//              "run-pass-valgrind");
//        suite("check-pretty-rpass-full", "src/test/run-pass-fulldeps/pretty",
//              "pretty", "run-pass-fulldeps");
//        suite("check-pretty-rfail-full", "src/test/run-fail-fulldeps/pretty",
//              "pretty", "run-fail-fulldeps");
//    }

#[derive(Serialize)]
pub struct Compiletest<'a> {
    compiler: Compiler<'a>,
    target: &'a str,
    mode: &'a str,
    suite: &'a str,
}

impl<'a> Step<'a> for Compiletest<'a> {
    type Output = ();

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
        let _folder = build.fold_output(|| format!("test_{}", suite));
        println!("Check compiletest suite={} mode={} ({} -> {})",
                 suite, mode, compiler.host, target);
        let mut cmd = Command::new(build.tool(&Compiler::new(0, compiler.host),
                                              "compiletest"));
        build.prepare_tool_cmd(compiler, &mut cmd);

        // compiletest currently has... a lot of arguments, so let's just pass all
        // of them!

        cmd.arg("--compile-lib-path").arg(build.rustc_libdir(compiler));
        cmd.arg("--run-lib-path").arg(build.sysroot_libdir(compiler, target));
        cmd.arg("--rustc-path").arg(build.compiler_path(compiler));
        cmd.arg("--rustdoc-path").arg(build.rustdoc(compiler));
        cmd.arg("--src-base").arg(build.src.join("src/test").join(suite));
        cmd.arg("--build-base").arg(testdir(build, compiler.host).join(suite));
        cmd.arg("--stage-id").arg(format!("stage{}-{}", compiler.stage, target));
        cmd.arg("--mode").arg(mode);
        cmd.arg("--target").arg(target);
        cmd.arg("--host").arg(compiler.host);
        cmd.arg("--llvm-filecheck").arg(build.llvm_filecheck(&build.build));

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

        let mut hostflags = build.rustc_flags(&compiler.host);
        hostflags.extend(flags.clone());
        cmd.arg("--host-rustcflags").arg(hostflags.join(" "));

        let mut targetflags = build.rustc_flags(&target);
        targetflags.extend(flags);
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
        let llvm_config = build.llvm_config(target);
        let llvm_version = output(Command::new(&llvm_config).arg("--version"));
        cmd.arg("--llvm-version").arg(llvm_version);
        if !build.is_rust_llvm(target) {
            cmd.arg("--system-llvm");
        }

        cmd.args(&build.flags.cmd.test_args());

        if build.is_verbose() {
            cmd.arg("--verbose");
        }

        if build.config.quiet_tests {
            cmd.arg("--quiet");
        }

        // Only pass correct values for these flags for the `run-make` suite as it
        // requires that a C++ compiler was configured which isn't always the case.
        if suite == "run-make" {
            let llvm_components = output(Command::new(&llvm_config).arg("--components"));
            let llvm_cxxflags = output(Command::new(&llvm_config).arg("--cxxflags"));
            cmd.arg("--cc").arg(build.cc(target))
               .arg("--cxx").arg(build.cxx(target).unwrap())
               .arg("--cflags").arg(build.cflags(target).join(" "))
               .arg("--llvm-components").arg(llvm_components.trim())
               .arg("--llvm-cxxflags").arg(llvm_cxxflags.trim());
        } else {
            cmd.arg("--cc").arg("")
               .arg("--cxx").arg("")
               .arg("--cflags").arg("")
               .arg("--llvm-components").arg("")
               .arg("--llvm-cxxflags").arg("");
        }

        if build.remote_tested(target) {
            cmd.arg("--remote-test-client")
               .arg(build.tool(&Compiler::new(0, &build.build),
                               "remote-test-client"));
        }

        // Running a C compiler on MSVC requires a few env vars to be set, to be
        // sure to set them here.
        //
        // Note that if we encounter `PATH` we make sure to append to our own `PATH`
        // rather than stomp over it.
        if target.contains("msvc") {
            for &(ref k, ref v) in build.cc[target].0.env() {
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

        let _time = util::timeit();
        try_run(build, &mut cmd);
    }
}

// rules.test("check-docs", "src/doc")
//     .dep(|s| s.name("libtest"))
//     .default(true)
//     .host(true)
//     .run(move |s| check::docs(build, &s.compiler()));
/// Run `rustdoc --test` for all documentation in `src/doc`.
///
/// This will run all tests in our markdown documentation (e.g. the book)
/// located in `src/doc`. The `rustdoc` that's run is the one that sits next to
/// `compiler`.
pub fn docs(build: &Build, compiler: &Compiler) {
    // Do a breadth-first traversal of the `src/doc` directory and just run
    // tests for all files that end in `*.md`
    let mut stack = vec![build.src.join("src/doc")];
    let _time = util::timeit();
    let _folder = build.fold_output(|| "test_docs");

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

        markdown_test(build, compiler, &p);
    }
}

//rules.test("check-error-index", "src/tools/error_index_generator")
//     .dep(|s| s.name("libstd"))
//     .dep(|s| s.name("tool-error-index").host(s.host).stage(0))
//     .default(true)
//     .host(true)
//     .run(move |s| check::error_index(build, &s.compiler()));

#[derive(Serialize)]
pub struct ErrorIndex<'a> {
    compiler: Compiler<'a>,
}

impl<'a> Step<'a> for ErrorIndex<'a> {
    type Output = ();

    /// Run the error index generator tool to execute the tests located in the error
    /// index.
    ///
    /// The `error_index_generator` tool lives in `src/tools` and is used to
    /// generate a markdown file from the error indexes of the code base which is
    /// then passed to `rustdoc --test`.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = self.compiler;

        let _folder = build.fold_output(|| "test_error_index");
        println!("Testing error-index stage{}", compiler.stage);

        let dir = testdir(build, compiler.host);
        t!(fs::create_dir_all(&dir));
        let output = dir.join("error-index.md");

        let _time = util::timeit();
        build.run(build.tool_cmd(&Compiler::new(0, compiler.host),
                                "error_index_generator")
                    .arg("markdown")
                    .arg(&output)
                    .env("CFG_BUILD", &build.build));

        markdown_test(build, compiler, &output);
    }
}

fn markdown_test(build: &Build, compiler: &Compiler, markdown: &Path) {
    let mut file = t!(File::open(markdown));
    let mut contents = String::new();
    t!(file.read_to_string(&mut contents));
    if !contents.contains("```") {
        return;
    }

    println!("doc tests for: {}", markdown.display());
    let mut cmd = Command::new(build.rustdoc(compiler));
    build.add_rustc_lib_path(compiler, &mut cmd);
    build.add_rust_test_threads(&mut cmd);
    cmd.arg("--test");
    cmd.arg(markdown);
    cmd.env("RUSTC_BOOTSTRAP", "1");

    let test_args = build.flags.cmd.test_args().join(" ");
    cmd.arg("--test-args").arg(test_args);

    if build.config.quiet_tests {
        try_run_quiet(build, &mut cmd);
    } else {
        try_run(build, &mut cmd);
    }
}

//    for (krate, path, _default) in krates("std") {
//        rules.test(&krate.test_step, path)
//             .dep(|s| s.name("libtest"))
//             .dep(|s| s.name("remote-copy-libs"))
//             .run(move |s| check::krate(build, &s.compiler(), s.target,
//                                        Mode::Libstd, TestKind::Test,
//                                        Some(&krate.name)));
//    }
//    rules.test("check-std-all", "path/to/nowhere")
//         .dep(|s| s.name("libtest"))
//         .dep(|s| s.name("remote-copy-libs"))
//         .default(true)
//         .run(move |s| check::krate(build, &s.compiler(), s.target,
//                                    Mode::Libstd, TestKind::Test, None));
//
//    // std benchmarks
//    for (krate, path, _default) in krates("std") {
//        rules.bench(&krate.bench_step, path)
//             .dep(|s| s.name("libtest"))
//             .dep(|s| s.name("remote-copy-libs"))
//             .run(move |s| check::krate(build, &s.compiler(), s.target,
//                                        Mode::Libstd, TestKind::Bench,
//                                        Some(&krate.name)));
//    }
//    rules.bench("bench-std-all", "path/to/nowhere")
//         .dep(|s| s.name("libtest"))
//         .dep(|s| s.name("remote-copy-libs"))
//         .default(true)
//         .run(move |s| check::krate(build, &s.compiler(), s.target,
//                                    Mode::Libstd, TestKind::Bench, None));
//
//    for (krate, path, _default) in krates("test") {
//        rules.test(&krate.test_step, path)
//             .dep(|s| s.name("libtest"))
//             .dep(|s| s.name("remote-copy-libs"))
//             .run(move |s| check::krate(build, &s.compiler(), s.target,
//                                        Mode::Libtest, TestKind::Test,
//                                        Some(&krate.name)));
//    }
//    rules.test("check-test-all", "path/to/nowhere")
//         .dep(|s| s.name("libtest"))
//         .dep(|s| s.name("remote-copy-libs"))
//         .default(true)
//         .run(move |s| check::krate(build, &s.compiler(), s.target,
//                                    Mode::Libtest, TestKind::Test, None));
//    for (krate, path, _default) in krates("rustc-main") {
//        rules.test(&krate.test_step, path)
//             .dep(|s| s.name("librustc"))
//             .dep(|s| s.name("remote-copy-libs"))
//             .host(true)
//             .run(move |s| check::krate(build, &s.compiler(), s.target,
//                                        Mode::Librustc, TestKind::Test,
//                                        Some(&krate.name)));
//    }
//    rules.test("check-rustc-all", "path/to/nowhere")
//         .dep(|s| s.name("librustc"))
//         .dep(|s| s.name("remote-copy-libs"))
//         .default(true)
//         .host(true)
//         .run(move |s| check::krate(build, &s.compiler(), s.target,
//                                    Mode::Librustc, TestKind::Test, None));

#[derive(Serialize)]
pub struct Krate<'a> {
    compiler: Compiler<'a>,
    target: &'a str,
    mode: Mode,
    test_kind: TestKind,
    krate: Option<&'a str>,
}

impl<'a> Step<'a> for Krate<'a> {
    type Output = ();

    /// Run all unit tests plus documentation tests for an entire crate DAG defined
    /// by a `Cargo.toml`
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

        let (name, path, features, root) = match mode {
            Mode::Libstd => {
                ("libstd", "src/libstd", build.std_features(), "std")
            }
            Mode::Libtest => {
                ("libtest", "src/libtest", String::new(), "test")
            }
            Mode::Librustc => {
                ("librustc", "src/rustc", build.rustc_features(), "rustc-main")
            }
            _ => panic!("can only test libraries"),
        };
        let _folder = build.fold_output(|| {
            format!("{}_stage{}-{}", test_kind.subcommand(), compiler.stage, name)
        });
        println!("{} {} stage{} ({} -> {})", test_kind, name, compiler.stage,
                compiler.host, target);

        // If we're not doing a full bootstrap but we're testing a stage2 version of
        // libstd, then what we're actually testing is the libstd produced in
        // stage1. Reflect that here by updating the compiler that we're working
        // with automatically.
        let compiler = if build.force_use_stage1(compiler, target) {
            Compiler::new(1, compiler.host)
        } else {
            compiler.clone()
        };

        // Build up the base `cargo test` command.
        //
        // Pass in some standard flags then iterate over the graph we've discovered
        // in `cargo metadata` with the maps above and figure out what `-p`
        // arguments need to get passed.
        let mut cargo = build.cargo(&compiler, mode, target, test_kind.subcommand());
        cargo.arg("--manifest-path")
            .arg(build.src.join(path).join("Cargo.toml"))
            .arg("--features").arg(features);
        if test_kind.subcommand() == "test" && !build.fail_fast {
            cargo.arg("--no-fail-fast");
        }

        match krate {
            Some(krate) => {
                cargo.arg("-p").arg(krate);
            }
            None => {
                let mut visited = HashSet::new();
                let mut next = vec![root];
                while let Some(name) = next.pop() {
                    // Right now jemalloc is our only target-specific crate in the
                    // sense that it's not present on all platforms. Custom skip it
                    // here for now, but if we add more this probably wants to get
                    // more generalized.
                    //
                    // Also skip `build_helper` as it's not compiled normally for
                    // target during the bootstrap and it's just meant to be a
                    // helper crate, not tested. If it leaks through then it ends up
                    // messing with various mtime calculations and such.
                    if !name.contains("jemalloc") && name != "build_helper" {
                        cargo.arg("-p").arg(&format!("{}:0.0.0", name));
                    }
                    for dep in build.crates[name].deps.iter() {
                        if visited.insert(dep) {
                            next.push(dep);
                        }
                    }
                }
            }
        }

        // The tests are going to run with the *target* libraries, so we need to
        // ensure that those libraries show up in the LD_LIBRARY_PATH equivalent.
        //
        // Note that to run the compiler we need to run with the *host* libraries,
        // but our wrapper scripts arrange for that to be the case anyway.
        let mut dylib_path = dylib_path();
        dylib_path.insert(0, build.sysroot_libdir(&compiler, target));
        cargo.env(dylib_path_var(), env::join_paths(&dylib_path).unwrap());

        if target.contains("emscripten") || build.remote_tested(target) {
            cargo.arg("--no-run");
        }

        cargo.arg("--");

        if build.config.quiet_tests {
            cargo.arg("--quiet");
        }

        let _time = util::timeit();

        if target.contains("emscripten") {
            build.run(&mut cargo);
            krate_emscripten(build, &compiler, target, mode);
        } else if build.remote_tested(target) {
            build.run(&mut cargo);
            krate_remote(build, &compiler, target, mode);
        } else {
            cargo.args(&build.flags.cmd.test_args());
            try_run(build, &mut cargo);
        }
    }
}

fn krate_emscripten(build: &Build,
                    compiler: &Compiler,
                    target: &str,
                    mode: Mode) {
    let out_dir = build.cargo_out(compiler, mode, target);
    let tests = find_tests(&out_dir.join("deps"), target);

    let nodejs = build.config.nodejs.as_ref().expect("nodejs not configured");
    for test in tests {
        println!("running {}", test.display());
        let mut cmd = Command::new(nodejs);
        cmd.arg(&test);
        if build.config.quiet_tests {
            cmd.arg("--quiet");
        }
        try_run(build, &mut cmd);
    }
}

fn krate_remote(build: &Build,
                compiler: &Compiler,
                target: &str,
                mode: Mode) {
    let out_dir = build.cargo_out(compiler, mode, target);
    let tests = find_tests(&out_dir.join("deps"), target);

    let tool = build.tool(&Compiler::new(0, &build.build),
                          "remote-test-client");
    for test in tests {
        let mut cmd = Command::new(&tool);
        cmd.arg("run")
           .arg(&test);
        if build.config.quiet_tests {
            cmd.arg("--quiet");
        }
        cmd.args(&build.flags.cmd.test_args());
        try_run(build, &mut cmd);
    }
}

fn find_tests(dir: &Path, target: &str) -> Vec<PathBuf> {
    let mut dst = Vec::new();
    for e in t!(dir.read_dir()).map(|e| t!(e)) {
        let file_type = t!(e.file_type());
        if !file_type.is_file() {
            continue
        }
        let filename = e.file_name().into_string().unwrap();
        if (target.contains("windows") && filename.ends_with(".exe")) ||
           (!target.contains("windows") && !filename.contains(".")) ||
           (target.contains("emscripten") &&
            filename.ends_with(".js") &&
            !filename.ends_with(".asm.js")) {
            dst.push(e.path());
        }
    }
    dst
}

//    // Some test suites are run inside emulators or on remote devices, and most
//    // of our test binaries are linked dynamically which means we need to ship
//    // the standard library and such to the emulator ahead of time. This step
//    // represents this and is a dependency of all test suites.
//    //
//    // Most of the time this step is a noop (the `check::emulator_copy_libs`
//    // only does work if necessary). For some steps such as shipping data to
//    // QEMU we have to build our own tools so we've got conditional dependencies
//    // on those programs as well. Note that the remote test client is built for
//    // the build target (us) and the server is built for the target.
//    rules.test("remote-copy-libs", "path/to/nowhere")
//         .dep(|s| s.name("libtest"))
//         .dep(move |s| {
//             if build.remote_tested(s.target) {
//                s.name("tool-remote-test-client").target(s.host).stage(0)
//             } else {
//                 Step::noop()
//             }
//         })
//         .dep(move |s| {
//             if build.remote_tested(s.target) {
//                s.name("tool-remote-test-server")
//             } else {
//                 Step::noop()
//             }
//         })
//         .run(move |s| check::remote_copy_libs(build, &s.compiler(), s.target));
//

#[derive(Serialize)]
pub struct RemoteCopyLibs<'a> {
    compiler: Compiler<'a>,
    target: &'a str,
}

impl<'a> Step<'a> for RemoteCopyLibs<'a> {
    type Output = ();

    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = self.compiler;
        let target = self.target;
        if !build.remote_tested(target) {
            return
        }

        println!("REMOTE copy libs to emulator ({})", target);
        t!(fs::create_dir_all(build.out.join("tmp")));

        let server = build.cargo_out(compiler, Mode::Tool, target)
                          .join(exe("remote-test-server", target));

        // Spawn the emulator and wait for it to come online
        let tool = build.tool(&Compiler::new(0, &build.build),
                              "remote-test-client");
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
        for f in t!(build.sysroot_libdir(compiler, target).read_dir()) {
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

//rules.test("check-distcheck", "distcheck")
//     .dep(|s| s.name("dist-plain-source-tarball"))
//     .dep(|s| s.name("dist-src"))
//     .run(move |_| check::distcheck(build));

#[derive(Serialize)]
pub struct Distcheck;

impl<'a> Step<'a> for Distcheck {
    type Output = ();

    /// Run "distcheck", a 'make check' from a tarball
    fn run(self, builder: &Builder) {
        let build = builder.build;

        if build.build != "x86_64-unknown-linux-gnu" {
            return
        }
        if !build.config.host.iter().any(|s| s == "x86_64-unknown-linux-gnu") {
            return
        }
        if !build.config.target.iter().any(|s| s == "x86_64-unknown-linux-gnu") {
            return
        }

        println!("Distcheck");
        let dir = build.out.join("tmp").join("distcheck");
        let _ = fs::remove_dir_all(&dir);
        t!(fs::create_dir_all(&dir));

        let mut cmd = Command::new("tar");
        cmd.arg("-xzf")
           .arg(dist::rust_src_location(build))
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
        println!("Distcheck rust-src");
        let dir = build.out.join("tmp").join("distcheck-src");
        let _ = fs::remove_dir_all(&dir);
        t!(fs::create_dir_all(&dir));

        let mut cmd = Command::new("tar");
        cmd.arg("-xzf")
           .arg(dist::rust_src_installer(build))
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

//rules.test("check-bootstrap", "src/bootstrap")
//     .default(true)
//     .host(true)
//     .only_build(true)
//     .run(move |_| check::bootstrap(build));

#[derive(Serialize)]
pub struct Bootstrap;

impl<'a> for Step<'a> Bootstrap {
    type Output = ();

    /// Test the build system itself
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let mut cmd = Command::new(&build.initial_cargo);
        cmd.arg("test")
           .current_dir(build.src.join("src/bootstrap"))
           .env("CARGO_TARGET_DIR", build.out.join("bootstrap"))
           .env("RUSTC_BOOTSTRAP", "1")
           .env("RUSTC", &build.initial_rustc);
        if !build.fail_fast {
            cmd.arg("--no-fail-fast");
        }
        cmd.arg("--").args(&build.flags.cmd.test_args());
        try_run(build, &mut cmd);
    }
}
