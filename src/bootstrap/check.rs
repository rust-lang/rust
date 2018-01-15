// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of compiling the compiler and standard library, in "check" mode.

use compile::{run_cargo, std_cargo, test_cargo, rustc_cargo, add_to_sysroot};
use builder::{RunConfig, Builder, ShouldRun, Step};
use {Build, Compiler, Mode};
use cache::Interned;
use std::path::PathBuf;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Std {
    pub target: Interned<String>,
}

impl Step for Std {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/libstd").krate("std")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Std {
            target: run.target,
        });
    }

    fn run(self, builder: &Builder) {
        let build = builder.build;
        let target = self.target;
        let compiler = builder.compiler(0, build.build);

        let _folder = build.fold_output(|| format!("stage{}-std", compiler.stage));
        println!("Checking std artifacts ({} -> {})", &compiler.host, target);

        let out_dir = build.stage_out(compiler, Mode::Libstd);
        build.clear_if_dirty(&out_dir, &builder.rustc(compiler));
        let mut cargo = builder.cargo(compiler, Mode::Libstd, target, "check");
        std_cargo(build, &compiler, target, &mut cargo);
        run_cargo(build,
                  &mut cargo,
                  &libstd_stamp(build, compiler, target),
                  true);
        let libdir = builder.sysroot_libdir(compiler, target);
        add_to_sysroot(&libdir, &libstd_stamp(build, compiler, target));
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Rustc {
    pub target: Interned<String>,
}

impl Step for Rustc {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/librustc").krate("rustc-main")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Rustc {
            target: run.target,
        });
    }

    /// Build the compiler.
    ///
    /// This will build the compiler for a particular stage of the build using
    /// the `compiler` targeting the `target` architecture. The artifacts
    /// created will also be linked into the sysroot directory.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = builder.compiler(0, build.build);
        let target = self.target;

        let _folder = build.fold_output(|| format!("stage{}-rustc", compiler.stage));
        println!("Checking compiler artifacts ({} -> {})", &compiler.host, target);

        let stage_out = builder.stage_out(compiler, Mode::Librustc);
        build.clear_if_dirty(&stage_out, &libstd_stamp(build, compiler, target));
        build.clear_if_dirty(&stage_out, &libtest_stamp(build, compiler, target));

        let mut cargo = builder.cargo(compiler, Mode::Librustc, target, "check");
        rustc_cargo(build, target, &mut cargo);
        run_cargo(build,
                  &mut cargo,
                  &librustc_stamp(build, compiler, target),
                  true);
        let libdir = builder.sysroot_libdir(compiler, target);
        add_to_sysroot(&libdir, &librustc_stamp(build, compiler, target));
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Test {
    pub target: Interned<String>,
}

impl Step for Test {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/libtest").krate("test")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Test {
            target: run.target,
        });
    }

    fn run(self, builder: &Builder) {
        let build = builder.build;
        let target = self.target;
        let compiler = builder.compiler(0, build.build);

        let _folder = build.fold_output(|| format!("stage{}-test", compiler.stage));
        println!("Checking test artifacts ({} -> {})", &compiler.host, target);
        let out_dir = build.stage_out(compiler, Mode::Libtest);
        build.clear_if_dirty(&out_dir, &libstd_stamp(build, compiler, target));
        let mut cargo = builder.cargo(compiler, Mode::Libtest, target, "check");
        test_cargo(build, &compiler, target, &mut cargo);
        run_cargo(build,
                  &mut cargo,
                  &libtest_stamp(build, compiler, target),
                  true);
        let libdir = builder.sysroot_libdir(compiler, target);
        add_to_sysroot(&libdir, &libtest_stamp(build, compiler, target));
    }
}

/// Cargo's output path for the standard library in a given stage, compiled
/// by a particular compiler for the specified target.
pub fn libstd_stamp(build: &Build, compiler: Compiler, target: Interned<String>) -> PathBuf {
    build.cargo_out(compiler, Mode::Libstd, target).join(".libstd-check.stamp")
}

/// Cargo's output path for libtest in a given stage, compiled by a particular
/// compiler for the specified target.
pub fn libtest_stamp(build: &Build, compiler: Compiler, target: Interned<String>) -> PathBuf {
    build.cargo_out(compiler, Mode::Libtest, target).join(".libtest-check.stamp")
}

/// Cargo's output path for librustc in a given stage, compiled by a particular
/// compiler for the specified target.
pub fn librustc_stamp(build: &Build, compiler: Compiler, target: Interned<String>) -> PathBuf {
    build.cargo_out(compiler, Mode::Librustc, target).join(".librustc-check.stamp")
}

