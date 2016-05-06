// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fs;
use std::path::{PathBuf, Path};
use std::process::Command;

use build_helper::output;

use build::{Build, Compiler};

pub fn linkcheck(build: &Build, stage: u32, host: &str) {
    println!("Linkcheck stage{} ({})", stage, host);
    let compiler = Compiler::new(stage, host);
    build.run(build.tool_cmd(&compiler, "linkchecker")
                   .arg(build.out.join(host).join("doc")));
}

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

pub fn tidy(build: &Build, stage: u32, host: &str) {
    println!("tidy check stage{} ({})", stage, host);
    let compiler = Compiler::new(stage, host);
    build.run(build.tool_cmd(&compiler, "tidy")
                   .arg(build.src.join("src")));
}

fn testdir(build: &Build, host: &str) -> PathBuf {
    build.out.join(host).join("test")
}

pub fn compiletest(build: &Build,
                   compiler: &Compiler,
                   target: &str,
                   mode: &str,
                   suite: &str) {
    let mut cmd = build.tool_cmd(compiler, "compiletest");

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

    let linkflag = format!("-Lnative={}", build.test_helpers_out(target).display());
    cmd.arg("--host-rustcflags").arg("-Crpath");
    cmd.arg("--target-rustcflags").arg(format!("-Crpath {}", linkflag));

    // FIXME: needs android support
    cmd.arg("--android-cross-path").arg("");

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

    cmd.args(&build.flags.args);

    if build.config.verbose || build.flags.verbose {
        cmd.arg("--verbose");
    }

    if suite == "run-make" {
        let llvm_config = build.llvm_config(target);
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

    build.run(&mut cmd);
}

pub fn docs(build: &Build, compiler: &Compiler) {
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
