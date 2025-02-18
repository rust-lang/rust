use std::env;
use std::path::PathBuf;

use crate::core::build_steps::compile;
use crate::core::build_steps::tool::{self, SourceType, Tool};
use crate::core::builder::{
    self, Builder, Compiler, Kind, RunConfig, ShouldRun, Step, crate_description,
};
use crate::core::config::TargetSelection;
use crate::utils::exec::{BootstrapCommand, command};
use crate::utils::helpers::{self, dylib_path, dylib_path_var, t};
use crate::utils::render_tests::add_flags_and_try_run_tests;
use crate::{DocTests, Mode, envify};

/// Given a `cargo test` subcommand, add the appropriate flags and run it.
///
/// Returns whether the test succeeded.
pub(super) fn run_cargo_test<'a>(
    cargo: builder::Cargo,
    libtest_args: &[&str],
    crates: &[String],
    primary_crate: &str,
    description: impl Into<Option<&'a str>>,
    target: TargetSelection,
    builder: &Builder<'_>,
) -> bool {
    let compiler = cargo.compiler();
    let mut cargo = prepare_cargo_test(cargo, libtest_args, crates, primary_crate, target, builder);
    let _time = helpers::timeit(builder);
    let _group = description.into().and_then(|what| {
        builder.msg_sysroot_tool(Kind::Test, compiler.stage, what, compiler.host, target)
    });

    #[cfg(feature = "build-metrics")]
    builder.metrics.begin_test_suite(
        build_helper::metrics::TestSuiteMetadata::CargoPackage {
            crates: crates.iter().map(|c| c.to_string()).collect(),
            target: target.triple.to_string(),
            host: compiler.host.triple.to_string(),
            stage: compiler.stage,
        },
        builder,
    );
    add_flags_and_try_run_tests(builder, &mut cargo)
}

/// Given a `cargo test` subcommand, pass it the appropriate test flags given a `builder`.
pub(super) fn prepare_cargo_test(
    cargo: builder::Cargo,
    libtest_args: &[&str],
    crates: &[String],
    primary_crate: &str,
    target: TargetSelection,
    builder: &Builder<'_>,
) -> BootstrapCommand {
    let compiler = cargo.compiler();
    let mut cargo: BootstrapCommand = cargo.into();

    // Propagate `--bless` if it has not already been set/unset
    // Any tools that want to use this should bless if `RUSTC_BLESS` is set to
    // anything other than `0`.
    if builder.config.cmd.bless() && !cargo.get_envs().any(|v| v.0 == "RUSTC_BLESS") {
        cargo.env("RUSTC_BLESS", "Gesundheit");
    }

    // Pass in some standard flags then iterate over the graph we've discovered
    // in `cargo metadata` with the maps above and figure out what `-p`
    // arguments need to get passed.
    if builder.kind == Kind::Test && !builder.fail_fast {
        cargo.arg("--no-fail-fast");
    }

    if builder.config.json_output {
        cargo.arg("--message-format=json");
    }

    match builder.doc_tests {
        DocTests::Only => {
            cargo.arg("--doc");
        }
        DocTests::No => {
            let krate = &builder
                .crates
                .get(primary_crate)
                .unwrap_or_else(|| panic!("missing crate {primary_crate}"));
            if krate.has_lib {
                cargo.arg("--lib");
            }
            cargo.args(["--bins", "--examples", "--tests", "--benches"]);
        }
        DocTests::Yes => {}
    }

    for krate in crates {
        cargo.arg("-p").arg(krate);
    }

    cargo.arg("--").args(builder.config.test_args()).args(libtest_args);
    if !builder.config.verbose_tests {
        cargo.arg("--quiet");
    }

    // The tests are going to run with the *target* libraries, so we need to
    // ensure that those libraries show up in the LD_LIBRARY_PATH equivalent.
    //
    // Note that to run the compiler we need to run with the *host* libraries,
    // but our wrapper scripts arrange for that to be the case anyway.
    //
    // We skip everything on Miri as then this overwrites the libdir set up
    // by `Cargo::new` and that actually makes things go wrong.
    if builder.kind != Kind::Miri {
        let mut dylib_path = dylib_path();
        dylib_path.insert(0, PathBuf::from(&*builder.sysroot_target_libdir(compiler, target)));
        cargo.env(dylib_path_var(), env::join_paths(&dylib_path).unwrap());
    }

    if builder.remote_tested(target) {
        cargo.env(
            format!("CARGO_TARGET_{}_RUNNER", envify(&target.triple)),
            format!("{} run 0", builder.tool_exe(Tool::RemoteTestClient).display()),
        );
    } else if let Some(tool) = builder.runner(target) {
        cargo.env(format!("CARGO_TARGET_{}_RUNNER", envify(&target.triple)), tool);
    }

    cargo
}

/// Some test suites are run inside emulators or on remote devices, and most of our test binaries
/// are linked dynamically which means we need to ship the standard library and such to the emulator
/// ahead of time. This step represents this and is a dependency of all test suites.
///
/// Most of the time this is a no-op. For some steps such as shipping data to QEMU we have to build
/// our own tools so we've got conditional dependencies on those programs as well. Note that the
/// remote test client is built for the build target (us) and the server is built for the target.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct RemoteCopyLibs {
    pub(super) compiler: Compiler,
    pub(super) target: TargetSelection,
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

        builder.ensure(compile::Std::new(compiler, target));

        builder.info(&format!("REMOTE copy libs to emulator ({target})"));

        let server = builder.ensure(tool::RemoteTestServer { compiler, target });

        // Spawn the emulator and wait for it to come online
        let tool = builder.tool_exe(Tool::RemoteTestClient);
        let mut cmd = command(&tool);
        cmd.arg("spawn-emulator").arg(target.triple).arg(&server).arg(builder.tempdir());
        if let Some(rootfs) = builder.qemu_rootfs(target) {
            cmd.arg(rootfs);
        }
        cmd.run(builder);

        // Push all our dylibs to the emulator
        for f in t!(builder.sysroot_target_libdir(compiler, target).read_dir()) {
            let f = t!(f);
            if helpers::is_dylib(&f.path()) {
                command(&tool).arg("push").arg(f.path()).run(builder);
            }
        }
    }
}

// FIXME(#137178): `Crate` is very confusing, probably need to be split into two steps?

/// Runs `cargo test` for standard library crates.
///
/// (Also used internally to run `cargo test` for compiler crates.)
///
/// FIXME(Zalathar): Try to split this into two separate steps: a user-visible step for testing
/// standard library crates, and an internal step used for both library crates and compiler crates.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Crate {
    pub compiler: Compiler,
    pub target: TargetSelection,
    pub mode: Mode,
    pub crates: Vec<String>,
}

impl Step for Crate {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.crate_or_deps("sysroot").crate_or_deps("coretests")
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let host = run.build_triple();
        let compiler = builder.compiler_for(builder.top_stage, host, host);
        let crates = run
            .paths
            .iter()
            .map(|p| builder.crate_paths[&p.assert_single_path().path].clone())
            .collect();

        builder.ensure(Crate { compiler, target: run.target, mode: Mode::Std, crates });
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

        // Prepare sysroot
        // See [field@compile::Std::force_recompile].
        builder.ensure(compile::Std::new(compiler, compiler.host).force_recompile(true));

        // If we're not doing a full bootstrap but we're testing a stage2
        // version of libstd, then what we're actually testing is the libstd
        // produced in stage1. Reflect that here by updating the compiler that
        // we're working with automatically.
        let compiler = builder.compiler_for(compiler.stage, compiler.host, target);

        let mut cargo = if builder.kind == Kind::Miri {
            if builder.top_stage == 0 {
                eprintln!("ERROR: `x.py miri` requires stage 1 or higher");
                std::process::exit(1);
            }

            // Build `cargo miri test` command
            // (Implicitly prepares target sysroot)
            let mut cargo = builder::Cargo::new(
                builder,
                compiler,
                mode,
                SourceType::InTree,
                target,
                Kind::MiriTest,
            );
            // This hack helps bootstrap run standard library tests in Miri. The issue is as
            // follows: when running `cargo miri test` on libcore, cargo builds a local copy of core
            // and makes it a dependency of the integration test crate. This copy duplicates all the
            // lang items, so the build fails. (Regular testing avoids this because the sysroot is a
            // literal copy of what `cargo build` produces, but since Miri builds its own sysroot
            // this does not work for us.) So we need to make it so that the locally built libcore
            // contains all the items from `core`, but does not re-define them -- we want to replace
            // the entire crate but a re-export of the sysroot crate. We do this by swapping out the
            // source file: if `MIRI_REPLACE_LIBRS_IF_NOT_TEST` is set and we are building a
            // `lib.rs` file, and a `lib.miri.rs` file exists in the same folder, we build that
            // instead. But crucially we only do that for the library, not the test builds.
            cargo.env("MIRI_REPLACE_LIBRS_IF_NOT_TEST", "1");
            // std needs to be built with `-Zforce-unstable-if-unmarked`. For some reason the builder
            // does not set this directly, but relies on the rustc wrapper to set it, and we are not using
            // the wrapper -- hence we have to set it ourselves.
            cargo.rustflag("-Zforce-unstable-if-unmarked");
            cargo
        } else {
            // Also prepare a sysroot for the target.
            if !builder.is_builder_target(target) {
                builder.ensure(compile::Std::new(compiler, target).force_recompile(true));
                builder.ensure(RemoteCopyLibs { compiler, target });
            }

            // Build `cargo test` command
            builder::Cargo::new(builder, compiler, mode, SourceType::InTree, target, builder.kind)
        };

        match mode {
            Mode::Std => {
                if builder.kind == Kind::Miri {
                    // We can't use `std_cargo` as that uses `optimized-compiler-builtins` which
                    // needs host tools for the given target. This is similar to what `compile::Std`
                    // does when `is_for_mir_opt_tests` is true. There's probably a chance for
                    // de-duplication here... `std_cargo` should support a mode that avoids needing
                    // host tools.
                    cargo
                        .arg("--manifest-path")
                        .arg(builder.src.join("library/sysroot/Cargo.toml"));
                } else {
                    compile::std_cargo(builder, target, compiler.stage, &mut cargo);
                    // `std_cargo` actually does the wrong thing: it passes `--sysroot build/host/stage2`,
                    // but we want to use the force-recompile std we just built in `build/host/stage2-test-sysroot`.
                    // Override it.
                    if builder.download_rustc() && compiler.stage > 0 {
                        let sysroot = builder
                            .out
                            .join(compiler.host)
                            .join(format!("stage{}-test-sysroot", compiler.stage));
                        cargo.env("RUSTC_SYSROOT", sysroot);
                    }
                }
            }
            Mode::Rustc => {
                compile::rustc_cargo(builder, &mut cargo, target, &compiler, &self.crates);
            }
            _ => panic!("can only test libraries"),
        };

        run_cargo_test(
            cargo,
            &[],
            &self.crates,
            &self.crates[0],
            &*crate_description(&self.crates),
            target,
            builder,
        );
    }
}
