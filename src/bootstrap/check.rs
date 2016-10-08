// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of the various `check-*` targets of the build system.
//!
//! This file implements the various regression test suites that we execute on
//! our CI.

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{PathBuf, Path};
use std::process::Command;

use build_helper::output;
use rustc_serialize::json;

use {Build, Compiler, Mode};
use util::{self, dylib_path, dylib_path_var};

const ADB_TEST_DIR: &'static str = "/data/tmp";

#[derive(RustcDecodable)]
struct Output {
    packages: Vec<Package>,
    resolve: Resolve,
}

#[derive(RustcDecodable)]
struct Package {
    id: String,
    name: String,
    source: Option<String>,
}

#[derive(RustcDecodable)]
struct Resolve {
    nodes: Vec<ResolveNode>,
}

#[derive(RustcDecodable)]
struct ResolveNode {
    id: String,
    dependencies: Vec<String>,
}

/// Runs the `linkchecker` tool as compiled in `stage` by the `host` compiler.
///
/// This tool in `src/tools` will verify the validity of all our links in the
/// documentation to ensure we don't have a bunch of dead ones.
pub fn linkcheck(build: &Build, stage: u32, host: &str) {
    println!("Linkcheck stage{} ({})", stage, host);
    let compiler = Compiler::new(stage, host);
    build.run(build.tool_cmd(&compiler, "linkchecker")
                   .arg(build.out.join(host).join("doc")));
}

/// Runs the `cargotest` tool as compiled in `stage` by the `host` compiler.
///
/// This tool in `src/tools` will check out a few Rust projects and run `cargo
/// test` to ensure that we don't regress the test suites there.
pub fn cargotest(build: &Build, stage: u32, host: &str) {
    let ref compiler = Compiler::new(stage, host);

    // Configure PATH to find the right rustc. NB. we have to use PATH
    // and not RUSTC because the Cargo test suite has tests that will
    // fail if rustc is not spelled `rustc`.
    let path = build.sysroot(compiler).join("bin");
    let old_path = ::std::env::var("PATH").expect("");
    let sep = if cfg!(windows) { ";" } else {":" };
    let ref newpath = format!("{}{}{}", path.display(), sep, old_path);

    // Note that this is a short, cryptic, and not scoped directory name. This
    // is currently to minimize the length of path on Windows where we otherwise
    // quickly run into path name limit constraints.
    let out_dir = build.out.join("ct");
    t!(fs::create_dir_all(&out_dir));

    build.run(build.tool_cmd(compiler, "cargotest")
                   .env("PATH", newpath)
                   .arg(&build.cargo)
                   .arg(&out_dir));
}

/// Runs the `tidy` tool as compiled in `stage` by the `host` compiler.
///
/// This tool in `src/tools` checks up on various bits and pieces of style and
/// otherwise just implements a few lint-like checks that are specific to the
/// compiler itself.
pub fn tidy(build: &Build, stage: u32, host: &str) {
    println!("tidy check stage{} ({})", stage, host);
    let compiler = Compiler::new(stage, host);
    build.run(build.tool_cmd(&compiler, "tidy")
                   .arg(build.src.join("src")));
}

fn testdir(build: &Build, host: &str) -> PathBuf {
    build.out.join(host).join("test")
}

/// Executes the `compiletest` tool to run a suite of tests.
///
/// Compiles all tests with `compiler` for `target` with the specified
/// compiletest `mode` and `suite` arguments. For example `mode` can be
/// "run-pass" or `suite` can be something like `debuginfo`.
pub fn compiletest(build: &Build,
                   compiler: &Compiler,
                   target: &str,
                   mode: &str,
                   suite: &str) {
    println!("Check compiletest {} ({} -> {})", suite, compiler.host, target);
    let mut cmd = build.tool_cmd(compiler, "compiletest");

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
    cmd.arg("--llvm-filecheck").arg(build.llvm_filecheck(&build.config.build));

    if let Some(nodejs) = build.config.nodejs.as_ref() {
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

    // FIXME: CFG_PYTHON should probably be detected more robustly elsewhere
    let python_default = "python";
    cmd.arg("--docck-python").arg(python_default);

    if build.config.build.ends_with("apple-darwin") {
        // Force /usr/bin/python on OSX for LLDB tests because we're loading the
        // LLDB plugin's compiled module which only works with the system python
        // (namely not Homebrew-installed python)
        cmd.arg("--lldb-python").arg("/usr/bin/python");
    } else {
        cmd.arg("--lldb-python").arg(python_default);
    }

    if let Some(ref vers) = build.gdb_version {
        cmd.arg("--gdb-version").arg(vers);
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

    cmd.args(&build.flags.args);

    if build.config.verbose || build.flags.verbose {
        cmd.arg("--verbose");
    }

    // Only pass correct values for these flags for the `run-make` suite as it
    // requires that a C++ compiler was configured which isn't always the case.
    if suite == "run-make" {
        let llvm_components = output(Command::new(&llvm_config).arg("--components"));
        let llvm_cxxflags = output(Command::new(&llvm_config).arg("--cxxflags"));
        cmd.arg("--cc").arg(build.cc(target))
           .arg("--cxx").arg(build.cxx(target))
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

    // Running a C compiler on MSVC requires a few env vars to be set, to be
    // sure to set them here.
    if target.contains("msvc") {
        for &(ref k, ref v) in build.cc[target].0.env() {
            if k != "PATH" {
                cmd.env(k, v);
            }
        }
    }
    build.add_bootstrap_key(compiler, &mut cmd);

    cmd.arg("--adb-path").arg("adb");
    cmd.arg("--adb-test-dir").arg(ADB_TEST_DIR);
    if target.contains("android") {
        // Assume that cc for this target comes from the android sysroot
        cmd.arg("--android-cross-path")
           .arg(build.cc(target).parent().unwrap().parent().unwrap());
    } else {
        cmd.arg("--android-cross-path").arg("");
    }

    build.run(&mut cmd);
}

/// Run `rustdoc --test` for all documentation in `src/doc`.
///
/// This will run all tests in our markdown documentation (e.g. the book)
/// located in `src/doc`. The `rustdoc` that's run is the one that sits next to
/// `compiler`.
pub fn docs(build: &Build, compiler: &Compiler) {
    // Do a breadth-first traversal of the `src/doc` directory and just run
    // tests for all files that end in `*.md`
    let mut stack = vec![build.src.join("src/doc")];

    while let Some(p) = stack.pop() {
        if p.is_dir() {
            stack.extend(t!(p.read_dir()).map(|p| t!(p).path()));
            continue
        }

        if p.extension().and_then(|s| s.to_str()) != Some("md") {
            continue
        }

        println!("doc tests for: {}", p.display());
        markdown_test(build, compiler, &p);
    }
}

/// Run the error index generator tool to execute the tests located in the error
/// index.
///
/// The `error_index_generator` tool lives in `src/tools` and is used to
/// generate a markdown file from the error indexes of the code base which is
/// then passed to `rustdoc --test`.
pub fn error_index(build: &Build, compiler: &Compiler) {
    println!("Testing error-index stage{}", compiler.stage);

    let output = testdir(build, compiler.host).join("error-index.md");
    build.run(build.tool_cmd(compiler, "error_index_generator")
                   .arg("markdown")
                   .arg(&output)
                   .env("CFG_BUILD", &build.config.build));

    markdown_test(build, compiler, &output);
}

fn markdown_test(build: &Build, compiler: &Compiler, markdown: &Path) {
    let mut cmd = Command::new(build.rustdoc(compiler));
    build.add_rustc_lib_path(compiler, &mut cmd);
    cmd.arg("--test");
    cmd.arg(markdown);
    cmd.arg("--test-args").arg(build.flags.args.join(" "));
    build.run(&mut cmd);
}

/// Run all unit tests plus documentation tests for an entire crate DAG defined
/// by a `Cargo.toml`
///
/// This is what runs tests for crates like the standard library, compiler, etc.
/// It essentially is the driver for running `cargo test`.
///
/// Currently this runs all tests for a DAG by passing a bunch of `-p foo`
/// arguments, and those arguments are discovered from `cargo metadata`.
pub fn krate(build: &Build,
             compiler: &Compiler,
             target: &str,
             mode: Mode) {
    let (name, path, features, root) = match mode {
        Mode::Libstd => {
            ("libstd", "src/rustc/std_shim", build.std_features(), "std_shim")
        }
        Mode::Libtest => {
            ("libtest", "src/rustc/test_shim", String::new(), "test_shim")
        }
        Mode::Librustc => {
            ("librustc", "src/rustc", build.rustc_features(), "rustc-main")
        }
        _ => panic!("can only test libraries"),
    };
    println!("Testing {} stage{} ({} -> {})", name, compiler.stage,
             compiler.host, target);

    // Run `cargo metadata` to figure out what crates we're testing.
    //
    // Down below we're going to call `cargo test`, but to test the right set
    // of packages we're going to have to know what `-p` arguments to pass it
    // to know what crates to test. Here we run `cargo metadata` to learn about
    // the dependency graph and what `-p` arguments there are.
    let mut cargo = Command::new(&build.cargo);
    cargo.arg("metadata")
         .arg("--manifest-path").arg(build.src.join(path).join("Cargo.toml"));
    let output = output(&mut cargo);
    let output: Output = json::decode(&output).unwrap();
    let id2pkg = output.packages.iter()
                        .map(|pkg| (&pkg.id, pkg))
                        .collect::<HashMap<_, _>>();
    let id2deps = output.resolve.nodes.iter()
                        .map(|node| (&node.id, &node.dependencies))
                        .collect::<HashMap<_, _>>();

    // Build up the base `cargo test` command.
    //
    // Pass in some standard flags then iterate over the graph we've discovered
    // in `cargo metadata` with the maps above and figure out what `-p`
    // arguments need to get passed.
    let mut cargo = build.cargo(compiler, mode, target, "test");
    cargo.arg("--manifest-path")
         .arg(build.src.join(path).join("Cargo.toml"))
         .arg("--features").arg(features);

    let mut visited = HashSet::new();
    let root_pkg = output.packages.iter().find(|p| p.name == root).unwrap();
    let mut next = vec![&root_pkg.id];
    while let Some(id) = next.pop() {
        // Skip any packages with sources listed, as these come from crates.io
        // and we shouldn't be testing them.
        if id2pkg[id].source.is_some() {
            continue
        }
        // Right now jemalloc is our only target-specific crate in the sense
        // that it's not present on all platforms. Custom skip it here for now,
        // but if we add more this probably wants to get more generalized.
        if !id.contains("jemalloc") {
            cargo.arg("-p").arg(&id2pkg[id].name);
        }
        for dep in id2deps[id] {
            if visited.insert(dep) {
                next.push(dep);
            }
        }
    }

    // The tests are going to run with the *target* libraries, so we need to
    // ensure that those libraries show up in the LD_LIBRARY_PATH equivalent.
    //
    // Note that to run the compiler we need to run with the *host* libraries,
    // but our wrapper scripts arrange for that to be the case anyway.
    let mut dylib_path = dylib_path();
    dylib_path.insert(0, build.sysroot_libdir(compiler, target));
    cargo.env(dylib_path_var(), env::join_paths(&dylib_path).unwrap());

    if target.contains("android") {
        build.run(cargo.arg("--no-run"));
        krate_android(build, compiler, target, mode);
    } else if target.contains("emscripten") {
        build.run(cargo.arg("--no-run"));
        krate_emscripten(build, compiler, target, mode);
    } else {
        cargo.args(&build.flags.args);
        build.run(&mut cargo);
    }
}

fn krate_android(build: &Build,
                 compiler: &Compiler,
                 target: &str,
                 mode: Mode) {
    let mut tests = Vec::new();
    let out_dir = build.cargo_out(compiler, mode, target);
    find_tests(&out_dir, target, &mut tests);
    find_tests(&out_dir.join("deps"), target, &mut tests);

    for test in tests {
        build.run(Command::new("adb").arg("push").arg(&test).arg(ADB_TEST_DIR));

        let test_file_name = test.file_name().unwrap().to_string_lossy();
        let log = format!("{}/check-stage{}-T-{}-H-{}-{}.log",
                          ADB_TEST_DIR,
                          compiler.stage,
                          target,
                          compiler.host,
                          test_file_name);
        let program = format!("(cd {dir}; \
                                LD_LIBRARY_PATH=./{target} ./{test} \
                                    --logfile {log} \
                                    {args})",
                              dir = ADB_TEST_DIR,
                              target = target,
                              test = test_file_name,
                              log = log,
                              args = build.flags.args.join(" "));

        let output = output(Command::new("adb").arg("shell").arg(&program));
        println!("{}", output);
        build.run(Command::new("adb")
                          .arg("pull")
                          .arg(&log)
                          .arg(build.out.join("tmp")));
        build.run(Command::new("adb").arg("shell").arg("rm").arg(&log));
        if !output.contains("result: ok") {
            panic!("some tests failed");
        }
    }
}

fn krate_emscripten(build: &Build,
                    compiler: &Compiler,
                    target: &str,
                    mode: Mode) {
     let mut tests = Vec::new();
     let out_dir = build.cargo_out(compiler, mode, target);
     find_tests(&out_dir, target, &mut tests);
     find_tests(&out_dir.join("deps"), target, &mut tests);

     for test in tests {
         let test_file_name = test.to_string_lossy().into_owned();
         println!("running {}", test_file_name);
         let nodejs = build.config.nodejs.as_ref().expect("nodejs not configured");
         let status = Command::new(nodejs)
             .arg(&test_file_name)
             .stderr(::std::process::Stdio::inherit())
             .status();
         match status {
             Ok(status) => {
                 if !status.success() {
                     panic!("some tests failed");
                 }
             }
             Err(e) => panic!(format!("failed to execute command: {}", e)),
         };
     }
 }


fn find_tests(dir: &Path,
              target: &str,
              dst: &mut Vec<PathBuf>) {
    for e in t!(dir.read_dir()).map(|e| t!(e)) {
        let file_type = t!(e.file_type());
        if !file_type.is_file() {
            continue
        }
        let filename = e.file_name().into_string().unwrap();
        if (target.contains("windows") && filename.ends_with(".exe")) ||
           (!target.contains("windows") && !filename.contains(".")) ||
           (target.contains("emscripten") && filename.contains(".js")){
            dst.push(e.path());
        }
    }
}

pub fn android_copy_libs(build: &Build,
                         compiler: &Compiler,
                         target: &str) {
    println!("Android copy libs to emulator ({})", target);
    build.run(Command::new("adb").arg("remount"));
    build.run(Command::new("adb").args(&["shell", "rm", "-r", ADB_TEST_DIR]));
    build.run(Command::new("adb").args(&["shell", "mkdir", ADB_TEST_DIR]));
    build.run(Command::new("adb")
                      .arg("push")
                      .arg(build.src.join("src/etc/adb_run_wrapper.sh"))
                      .arg(ADB_TEST_DIR));

    let target_dir = format!("{}/{}", ADB_TEST_DIR, target);
    build.run(Command::new("adb").args(&["shell", "mkdir", &target_dir[..]]));

    for f in t!(build.sysroot_libdir(compiler, target).read_dir()) {
        let f = t!(f);
        let name = f.file_name().into_string().unwrap();
        if util::is_dylib(&name) {
            build.run(Command::new("adb")
                              .arg("push")
                              .arg(f.path())
                              .arg(&target_dir));
        }
    }
}
