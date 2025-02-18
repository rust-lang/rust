//! Self-tests for bootstrap itself and tools used by bootstrap.

use std::path::PathBuf;

use super::test_helpers::run_cargo_test;
use crate::Mode;
use crate::core::build_steps::tool::{self, SourceType};
use crate::core::builder::{Builder, Kind, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::exec::command;

// FIXME(#137178): `CrateBootstrap` is not a great name, on cursory glance it sounds like it would
// be testing bootstrap-the-crate itself, but actually tests "some tools used by bootstrap". Its
// `should_run` condition is also not at all clear...

/// Runs `cargo test` on various internal tools used by bootstrap.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateBootstrap {
    path: PathBuf,
    host: TargetSelection,
}

impl Step for CrateBootstrap {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        // This step is responsible for several different tool paths. By default it will test all of
        // them, but requesting specific tools on the command-line (e.g. `./x test suggest-tests`)
        // will test only the specified tools.
        run.path("src/tools/jsondoclint")
            .path("src/tools/suggest-tests")
            .path("src/tools/replace-version-placeholder")
            // We want `./x test tidy` to _run_ the tidy tool, not its tests. So we need a separate
            // alias to test the tidy tool itself.
            .alias("tidyselftest")
    }

    fn make_run(run: RunConfig<'_>) {
        // Create and ensure a separate instance of this step for each path that was selected on the
        // command-line (or selected by default).
        for path in run.paths {
            let path = path.assert_single_path().path.clone();
            run.builder.ensure(CrateBootstrap { host: run.target, path });
        }
    }

    fn run(self, builder: &Builder<'_>) {
        let bootstrap_host = builder.config.build;
        let compiler = builder.compiler(0, bootstrap_host);
        let mut path = self.path.to_str().unwrap();

        // Map alias `tidyselftest` back to the actual crate path of tidy.
        if path == "tidyselftest" {
            path = "src/tools/tidy";
        }

        let cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolBootstrap,
            bootstrap_host,
            Kind::Test,
            path,
            SourceType::InTree,
            &[],
        );

        let crate_name = path.rsplit_once('/').unwrap().1;
        run_cargo_test(cargo, &[], &[], crate_name, crate_name, bootstrap_host, builder);
    }
}

// FIXME(#137178): `Bootstrap` is not a great name either, because it's easily confused with
// `CrateBootstrap`.

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Bootstrap;

impl Step for Bootstrap {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/bootstrap")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Bootstrap);
    }

    /// Tests the build system itself.
    fn run(self, builder: &Builder<'_>) {
        let host = builder.config.build;
        let compiler = builder.compiler(0, host);
        let _guard = builder.msg(Kind::Test, 0, "bootstrap", host, host);

        // Some tests require cargo submodule to be present.
        builder.build.require_submodule("src/tools/cargo", None);

        let mut check_bootstrap = command(builder.python());
        check_bootstrap
            .args(["-m", "unittest", "bootstrap_test.py"])
            .env("BUILD_DIR", &builder.out)
            .env("BUILD_PLATFORM", builder.build.build.triple)
            .env("BOOTSTRAP_TEST_RUSTC_BIN", &builder.initial_rustc)
            .env("BOOTSTRAP_TEST_CARGO_BIN", &builder.initial_cargo)
            .current_dir(builder.src.join("src/bootstrap/"));
        // NOTE: we intentionally don't pass test_args here because the args for unittest and cargo test are mutually incompatible.
        // Use `python -m unittest` manually if you want to pass arguments.
        check_bootstrap.delay_failure().run(builder);

        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolBootstrap,
            host,
            Kind::Test,
            "src/bootstrap",
            SourceType::InTree,
            &[],
        );

        cargo.release_build(false);

        cargo
            .rustflag("-Cdebuginfo=2")
            .env("CARGO_TARGET_DIR", builder.out.join("bootstrap"))
            .env("RUSTC_BOOTSTRAP", "1");

        // bootstrap tests are racy on directory creation so just run them one at a time. Since
        // there's not many this shouldn't be a problem.
        run_cargo_test(cargo, &["--test-threads=1"], &[], "bootstrap", None, host, builder);
    }
}
