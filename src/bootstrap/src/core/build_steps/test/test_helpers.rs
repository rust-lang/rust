use std::env;
use std::path::PathBuf;

use crate::core::build_steps::tool::Tool;
use crate::core::builder::{self, Builder, Kind};
use crate::core::config::TargetSelection;
use crate::utils::exec::BootstrapCommand;
use crate::utils::helpers::{self, dylib_path, dylib_path_var};
use crate::utils::render_tests::add_flags_and_try_run_tests;
use crate::{DocTests, envify};

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
