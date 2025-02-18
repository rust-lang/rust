//! Standard library tests.

use std::env;
use std::path::PathBuf;

use super::test_helpers::run_cargo_test;
use crate::Mode;
use crate::core::build_steps::tool::{self, SourceType};
use crate::core::builder::{Builder, Kind, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;

// FIXME(#137178): `Crate` is also used for standard library crate tests?

// FIXME(#137178): break up `TestFloatParse` into two steps: one for testing the tool itself, and
// one for testing std float parsing.

/// Test step that does two things:
/// - Runs `cargo test` for the `src/etc/test-float-parse` tool.
/// - Invokes the `test-float-parse` tool to test the standard library's float parsing routines.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TestFloatParse {
    path: PathBuf,
    host: TargetSelection,
}

impl Step for TestFloatParse {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/etc/test-float-parse")
    }

    fn make_run(run: RunConfig<'_>) {
        for path in run.paths {
            let path = path.assert_single_path().path.clone();
            run.builder.ensure(Self { path, host: run.target });
        }
    }

    fn run(self, builder: &Builder<'_>) {
        let bootstrap_host = builder.config.build;
        let compiler = builder.compiler(builder.top_stage, bootstrap_host);
        let path = self.path.to_str().unwrap();
        let crate_name = self.path.components().last().unwrap().as_os_str().to_str().unwrap();

        builder.ensure(tool::TestFloatParse { host: self.host });

        // Run any unit tests in the crate
        let cargo_test = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolStd,
            bootstrap_host,
            Kind::Test,
            path,
            SourceType::InTree,
            &[],
        );

        run_cargo_test(cargo_test, &[], &[], crate_name, crate_name, bootstrap_host, builder);

        // Run the actual parse tests.
        let mut cargo_run = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolStd,
            bootstrap_host,
            Kind::Run,
            path,
            SourceType::InTree,
            &[],
        );

        if !matches!(env::var("FLOAT_PARSE_TESTS_NO_SKIP_HUGE").as_deref(), Ok("1") | Ok("true")) {
            cargo_run.args(["--", "--skip-huge"]);
        }

        cargo_run.into_cmd().run(builder);
    }
}
