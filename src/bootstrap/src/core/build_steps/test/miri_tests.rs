use std::path::PathBuf;

use super::test_helpers::prepare_cargo_test;
use crate::core::build_steps::compile;
use crate::core::build_steps::tool::{self, SourceType};
use crate::core::builder::{self, Builder, Compiler, Kind, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::build_stamp::{self};
use crate::utils::exec::BootstrapCommand;
use crate::utils::helpers;
use crate::{DocTests, Mode};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Miri {
    target: TargetSelection,
}

impl Miri {
    /// Run `cargo miri setup` for the given target, return where the Miri sysroot was put.
    pub fn build_miri_sysroot(
        builder: &Builder<'_>,
        compiler: Compiler,
        target: TargetSelection,
    ) -> PathBuf {
        let miri_sysroot = builder.out.join(compiler.host).join("miri-sysroot");
        let mut cargo = builder::Cargo::new(
            builder,
            compiler,
            Mode::Std,
            SourceType::Submodule,
            target,
            Kind::MiriSetup,
        );

        // Tell `cargo miri setup` where to find the sources.
        cargo.env("MIRI_LIB_SRC", builder.src.join("library"));
        // Tell it where to put the sysroot.
        cargo.env("MIRI_SYSROOT", &miri_sysroot);

        let mut cargo = BootstrapCommand::from(cargo);
        let _guard =
            builder.msg(Kind::Build, compiler.stage, "miri sysroot", compiler.host, target);
        cargo.run(builder);

        // # Determine where Miri put its sysroot.
        // To this end, we run `cargo miri setup --print-sysroot` and capture the output.
        // (We do this separately from the above so that when the setup actually
        // happens we get some output.)
        // We re-use the `cargo` from above.
        cargo.arg("--print-sysroot");

        builder.verbose(|| println!("running: {cargo:?}"));
        let stdout = cargo.run_capture_stdout(builder).stdout();
        // Output is "<sysroot>\n".
        let sysroot = stdout.trim_end();
        builder.verbose(|| println!("`cargo miri setup --print-sysroot` said: {sysroot:?}"));
        PathBuf::from(sysroot)
    }
}

impl Step for Miri {
    type Output = ();
    const ONLY_HOSTS: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/miri")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Miri { target: run.target });
    }

    /// Runs `cargo test` for miri.
    fn run(self, builder: &Builder<'_>) {
        let host = builder.build.build;
        let target = self.target;
        let stage = builder.top_stage;
        if stage == 0 {
            eprintln!("miri cannot be tested at stage 0");
            std::process::exit(1);
        }

        // This compiler runs on the host, we'll just use it for the target.
        let target_compiler = builder.compiler(stage, host);
        // Similar to `compile::Assemble`, build with the previous stage's compiler. Otherwise
        // we'd have stageN/bin/rustc and stageN/bin/rustdoc be effectively different stage
        // compilers, which isn't what we want. Rustdoc should be linked in the same way as the
        // rustc compiler it's paired with, so it must be built with the previous stage compiler.
        let host_compiler = builder.compiler(stage - 1, host);

        // Build our tools.
        let miri = builder.ensure(tool::Miri { compiler: host_compiler, target: host });
        // the ui tests also assume cargo-miri has been built
        builder.ensure(tool::CargoMiri { compiler: host_compiler, target: host });

        // We also need sysroots, for Miri and for the host (the latter for build scripts).
        // This is for the tests so everything is done with the target compiler.
        let miri_sysroot = Miri::build_miri_sysroot(builder, target_compiler, target);
        builder.ensure(compile::Std::new(target_compiler, host));
        let host_sysroot = builder.sysroot(target_compiler);

        // Miri has its own "target dir" for ui test dependencies. Make sure it gets cleared when
        // the sysroot gets rebuilt, to avoid "found possibly newer version of crate `std`" errors.
        if !builder.config.dry_run() {
            let ui_test_dep_dir = builder.stage_out(host_compiler, Mode::ToolStd).join("miri_ui");
            // The mtime of `miri_sysroot` changes when the sysroot gets rebuilt (also see
            // <https://github.com/RalfJung/rustc-build-sysroot/commit/10ebcf60b80fe2c3dc765af0ff19fdc0da4b7466>).
            // We can hence use that directly as a signal to clear the ui test dir.
            build_stamp::clear_if_dirty(builder, &ui_test_dep_dir, &miri_sysroot);
        }

        // Run `cargo test`.
        // This is with the Miri crate, so it uses the host compiler.
        let mut cargo = tool::prepare_tool_cargo(
            builder,
            host_compiler,
            Mode::ToolRustc,
            host,
            Kind::Test,
            "src/tools/miri",
            SourceType::InTree,
            &[],
        );

        cargo.add_rustc_lib_path(builder);

        // We can NOT use `run_cargo_test` since Miri's integration tests do not use the usual test
        // harness and therefore do not understand the flags added by `add_flags_and_try_run_test`.
        let mut cargo = prepare_cargo_test(cargo, &[], &[], "miri", host, builder);

        // miri tests need to know about the stage sysroot
        cargo.env("MIRI_SYSROOT", &miri_sysroot);
        cargo.env("MIRI_HOST_SYSROOT", &host_sysroot);
        cargo.env("MIRI", &miri);

        // Set the target.
        cargo.env("MIRI_TEST_TARGET", target.rustc_target_arg());

        {
            let _guard = builder.msg_sysroot_tool(Kind::Test, stage, "miri", host, target);
            let _time = helpers::timeit(builder);
            cargo.run(builder);
        }

        // Run it again for mir-opt-level 4 to catch some miscompilations.
        if builder.config.test_args().is_empty() {
            cargo.env("MIRIFLAGS", "-O -Zmir-opt-level=4 -Cdebug-assertions=yes");
            // Optimizations can change backtraces
            cargo.env("MIRI_SKIP_UI_CHECKS", "1");
            // `MIRI_SKIP_UI_CHECKS` and `RUSTC_BLESS` are incompatible
            cargo.env_remove("RUSTC_BLESS");
            // Optimizations can change error locations and remove UB so don't run `fail` tests.
            cargo.args(["tests/pass", "tests/panic"]);

            {
                let _guard = builder.msg_sysroot_tool(
                    Kind::Test,
                    stage,
                    "miri (mir-opt-level 4)",
                    host,
                    target,
                );
                let _time = helpers::timeit(builder);
                cargo.run(builder);
            }
        }
    }
}

/// Runs `cargo miri test` to demonstrate that `src/tools/miri/cargo-miri`
/// works and that libtest works under miri.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CargoMiri {
    target: TargetSelection,
}

impl Step for CargoMiri {
    type Output = ();
    const ONLY_HOSTS: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/miri/cargo-miri")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(CargoMiri { target: run.target });
    }

    /// Tests `cargo miri test`.
    fn run(self, builder: &Builder<'_>) {
        let host = builder.build.build;
        let target = self.target;
        let stage = builder.top_stage;
        if stage == 0 {
            eprintln!("cargo-miri cannot be tested at stage 0");
            std::process::exit(1);
        }

        // This compiler runs on the host, we'll just use it for the target.
        let compiler = builder.compiler(stage, host);

        // Run `cargo miri test`.
        // This is just a smoke test (Miri's own CI invokes this in a bunch of different ways and ensures
        // that we get the desired output), but that is sufficient to make sure that the libtest harness
        // itself executes properly under Miri, and that all the logic in `cargo-miri` does not explode.
        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolStd, // it's unclear what to use here, we're not building anything just doing a smoke test!
            target,
            Kind::MiriTest,
            "src/tools/miri/test-cargo-miri",
            SourceType::Submodule,
            &[],
        );

        // We're not using `prepare_cargo_test` so we have to do this ourselves.
        // (We're not using that as the test-cargo-miri crate is not known to bootstrap.)
        match builder.doc_tests {
            DocTests::Yes => {}
            DocTests::No => {
                cargo.args(["--lib", "--bins", "--examples", "--tests", "--benches"]);
            }
            DocTests::Only => {
                cargo.arg("--doc");
            }
        }

        // Finally, pass test-args and run everything.
        cargo.arg("--").args(builder.config.test_args());
        let mut cargo = BootstrapCommand::from(cargo);
        {
            let _guard = builder.msg_sysroot_tool(Kind::Test, stage, "cargo-miri", host, target);
            let _time = helpers::timeit(builder);
            cargo.run(builder);
        }
    }
}
