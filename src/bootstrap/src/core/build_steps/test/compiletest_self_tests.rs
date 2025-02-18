use super::test_helpers::run_cargo_test;
use crate::core::build_steps::compile;
use crate::core::build_steps::tool::{self, SourceType};
use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::{Kind, Mode};

// FIXME(#137178): this step is named inconsistently versus other crate self-test steps.

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompiletestTest {
    host: TargetSelection,
}

impl Step for CompiletestTest {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/compiletest")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(CompiletestTest { host: run.target });
    }

    /// Runs `cargo test` for compiletest.
    fn run(self, builder: &Builder<'_>) {
        let host = self.host;
        let compiler = builder.compiler(builder.top_stage, host);

        // We need `ToolStd` for the locally-built sysroot because
        // compiletest uses unstable features of the `test` crate.
        builder.ensure(compile::Std::new(compiler, host));
        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            // compiletest uses libtest internals; make it use the in-tree std to make sure it never breaks
            // when std sources change.
            Mode::ToolStd,
            host,
            Kind::Test,
            "src/tools/compiletest",
            SourceType::InTree,
            &[],
        );
        cargo.allow_features("test");
        run_cargo_test(cargo, &[], &[], "compiletest", "compiletest self test", host, builder);
    }
}
