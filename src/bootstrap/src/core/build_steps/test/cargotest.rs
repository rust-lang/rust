use std::fs;

use crate::core::build_steps::compile;
use crate::core::build_steps::tool::{self, Tool};
use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::helpers::{self, LldThreads, add_rustdoc_cargo_linker_args, t};

/// Builds cargo and then runs the `src/tools/cargotest` tool, which checks out some representative
/// crate repositories and runs `cargo test` on them, in order to test cargo.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cargotest {
    stage: u32,
    host: TargetSelection,
}

impl Step for Cargotest {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/cargotest")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Cargotest { stage: run.builder.top_stage, host: run.target });
    }

    /// Runs the `cargotest` tool as compiled in `stage` by the `host` compiler.
    ///
    /// This tool in `src/tools` will check out a few Rust projects and run `cargo
    /// test` to ensure that we don't regress the test suites there.
    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(self.stage, self.host);
        builder.ensure(compile::Rustc::new(compiler, compiler.host));
        let cargo = builder.ensure(tool::Cargo { compiler, target: compiler.host });

        // Note that this is a short, cryptic, and not scoped directory name. This
        // is currently to minimize the length of path on Windows where we otherwise
        // quickly run into path name limit constraints.
        let out_dir = builder.out.join("ct");
        t!(fs::create_dir_all(&out_dir));

        let _time = helpers::timeit(builder);
        let mut cmd = builder.tool_cmd(Tool::CargoTest);
        cmd.arg(&cargo)
            .arg(&out_dir)
            .args(builder.config.test_args())
            .env("RUSTC", builder.rustc(compiler))
            .env("RUSTDOC", builder.rustdoc(compiler));
        add_rustdoc_cargo_linker_args(&mut cmd, builder, compiler.host, LldThreads::No);
        cmd.delay_failure().run(builder);
    }
}
