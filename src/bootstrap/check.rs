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

use compile::{add_to_sysroot};
use builder::{Builder, RunConfig, ShouldRun, Step};
use Mode;
use cache::Interned;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Std {
    pub target: Interned<String>,
}

impl Step for Std {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.all_krates("std")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Std { target: run.target });
    }

    fn run(self, builder: &Builder) {
        let target = self.target;
        let compiler = builder.compiler(0, builder.config.general.build);

        let _folder = builder.fold_output(|| format!("stage{}-std", compiler.stage));
        println!("Checking std artifacts ({} -> {})", &compiler.host, target);

        let mut cargo = builder.cargo(compiler, Mode::Libstd, target, "check");
        builder.run_cargo(
            &mut cargo,
            &builder.libstd_stamp(compiler, target),
            true,
        );
        let libdir = builder.sysroot_libdir(compiler, target);
        add_to_sysroot(&libdir, &builder.libstd_stamp(compiler, target));
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
        run.all_krates("rustc-main")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Rustc { target: run.target });
    }

    /// Build the compiler.
    ///
    /// This will build the compiler for a particular stage of the build using
    /// the `compiler` targeting the `target` architecture. The artifacts
    /// created will also be linked into the sysroot directory.
    fn run(self, builder: &Builder) {
        let compiler = builder.compiler(0, builder.config.general.build);
        let target = self.target;

        let _folder = builder.fold_output(|| format!("stage{}-rustc", compiler.stage));
        println!(
            "Checking compiler artifacts ({} -> {})",
            &compiler.host, target
        );

        let mut cargo = builder.cargo(compiler, Mode::Librustc, target, "check");
        builder.run_cargo(
            &mut cargo,
            &builder.librustc_stamp(compiler, target),
            true,
        );
        let libdir = builder.sysroot_libdir(compiler, target);
        add_to_sysroot(&libdir, &builder.librustc_stamp(compiler, target));
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
        run.all_krates("test")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Test { target: run.target });
    }

    fn run(self, builder: &Builder) {
        let target = self.target;
        let compiler = builder.compiler(0, builder.config.general.build);

        let _folder = builder.fold_output(|| format!("stage{}-test", compiler.stage));
        println!("Checking test artifacts ({} -> {})", &compiler.host, target);
        let mut cargo = builder.cargo(compiler, Mode::Libtest, target, "check");
        builder.run_cargo(
            &mut cargo,
            &builder.libtest_stamp(compiler, target),
            true,
        );
        let libdir = builder.sysroot_libdir(compiler, target);
        add_to_sysroot(&libdir, &builder.libtest_stamp(compiler, target));
    }
}
