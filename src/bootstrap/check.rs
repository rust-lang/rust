//! Implementation of compiling the compiler and standard library, in "check"-based modes.

use crate::compile::{run_cargo, std_cargo, rustc_cargo, add_to_sysroot};
use crate::builder::{RunConfig, Builder, Kind, ShouldRun, Step};
use crate::tool::{prepare_tool_cargo, SourceType};
use crate::{Compiler, Mode};
use crate::cache::Interned;
use std::path::PathBuf;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Std {
    pub target: Interned<String>,
}

fn args(kind: Kind) -> Vec<String> {
    match kind {
        Kind::Clippy => vec!["--".to_owned(), "--cap-lints".to_owned(), "warn".to_owned()],
        _ => Vec::new()
    }
}

fn cargo_subcommand(kind: Kind) -> &'static str {
    match kind {
        Kind::Check => "check",
        Kind::Clippy => "clippy",
        Kind::Fix => "fix",
        _ => unreachable!()
    }
}

impl Step for Std {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.all_krates("test")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Std {
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        let target = self.target;
        let compiler = builder.compiler(0, builder.config.build);

        let mut cargo = builder.cargo(compiler, Mode::Std, target, cargo_subcommand(builder.kind));
        std_cargo(builder, &compiler, target, &mut cargo);

        builder.info(&format!("Checking std artifacts ({} -> {})", &compiler.host, target));
        run_cargo(builder,
                  cargo,
                  args(builder.kind),
                  &libstd_stamp(builder, compiler, target),
                  vec![],
                  true);

        let libdir = builder.sysroot_libdir(compiler, target);
        let hostdir = builder.sysroot_libdir(compiler, compiler.host);
        add_to_sysroot(&builder, &libdir, &hostdir, &libstd_stamp(builder, compiler, target));
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

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.all_krates("rustc-main")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rustc {
            target: run.target,
        });
    }

    /// Builds the compiler.
    ///
    /// This will build the compiler for a particular stage of the build using
    /// the `compiler` targeting the `target` architecture. The artifacts
    /// created will also be linked into the sysroot directory.
    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(0, builder.config.build);
        let target = self.target;

        builder.ensure(Std { target });

        let mut cargo = builder.cargo(compiler, Mode::Rustc, target,
            cargo_subcommand(builder.kind));
        rustc_cargo(builder, &mut cargo, target);

        builder.info(&format!("Checking compiler artifacts ({} -> {})", &compiler.host, target));
        run_cargo(builder,
                  cargo,
                  args(builder.kind),
                  &librustc_stamp(builder, compiler, target),
                  vec![],
                  true);

        let libdir = builder.sysroot_libdir(compiler, target);
        let hostdir = builder.sysroot_libdir(compiler, compiler.host);
        add_to_sysroot(&builder, &libdir, &hostdir, &librustc_stamp(builder, compiler, target));
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Rustdoc {
    pub target: Interned<String>,
}

impl Step for Rustdoc {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/rustdoc")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rustdoc {
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(0, builder.config.build);
        let target = self.target;

        builder.ensure(Rustc { target });

        let cargo = prepare_tool_cargo(builder,
                                       compiler,
                                       Mode::ToolRustc,
                                       target,
                                       cargo_subcommand(builder.kind),
                                       "src/tools/rustdoc",
                                       SourceType::InTree,
                                       &[]);

        println!("Checking rustdoc artifacts ({} -> {})", &compiler.host, target);
        run_cargo(builder,
                  cargo,
                  args(builder.kind),
                  &rustdoc_stamp(builder, compiler, target),
                  vec![],
                  true);

        let libdir = builder.sysroot_libdir(compiler, target);
        let hostdir = builder.sysroot_libdir(compiler, compiler.host);
        add_to_sysroot(&builder, &libdir, &hostdir, &rustdoc_stamp(builder, compiler, target));
    }
}

/// Cargo's output path for the standard library in a given stage, compiled
/// by a particular compiler for the specified target.
pub fn libstd_stamp(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: Interned<String>,
) -> PathBuf {
    builder.cargo_out(compiler, Mode::Std, target).join(".libstd-check.stamp")
}

/// Cargo's output path for librustc in a given stage, compiled by a particular
/// compiler for the specified target.
pub fn librustc_stamp(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: Interned<String>,
) -> PathBuf {
    builder.cargo_out(compiler, Mode::Rustc, target).join(".librustc-check.stamp")
}

/// Cargo's output path for rustdoc in a given stage, compiled by a particular
/// compiler for the specified target.
pub fn rustdoc_stamp(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: Interned<String>,
) -> PathBuf {
    builder.cargo_out(compiler, Mode::ToolRustc, target)
        .join(".rustdoc-check.stamp")
}
