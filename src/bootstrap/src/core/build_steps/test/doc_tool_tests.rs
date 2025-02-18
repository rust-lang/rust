//! Auxiliary tools for checking various documentation.

use super::test_helpers::run_cargo_test;
use crate::core::build_steps::tool::{self, SourceType, Tool};
use crate::core::builder::{Builder, Kind, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::exec::command;
use crate::utils::helpers::{self};
use crate::{DocTests, Mode};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Linkcheck {
    host: TargetSelection,
}

impl Step for Linkcheck {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    /// Runs the `linkchecker` tool as compiled in `stage` by the `host` compiler.
    ///
    /// This tool in `src/tools` will verify the validity of all our links in the
    /// documentation to ensure we don't have a bunch of dead ones.
    fn run(self, builder: &Builder<'_>) {
        let host = self.host;
        let hosts = &builder.hosts;
        let targets = &builder.targets;

        // if we have different hosts and targets, some things may be built for
        // the host (e.g. rustc) and others for the target (e.g. std). The
        // documentation built for each will contain broken links to
        // docs built for the other platform (e.g. rustc linking to cargo)
        if (hosts != targets) && !hosts.is_empty() && !targets.is_empty() {
            panic!(
                "Linkcheck currently does not support builds with different hosts and targets.
You can skip linkcheck with --skip src/tools/linkchecker"
            );
        }

        builder.info(&format!("Linkcheck ({host})"));

        // Test the linkchecker itself.
        let bootstrap_host = builder.config.build;
        let compiler = builder.compiler(0, bootstrap_host);

        let cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolBootstrap,
            bootstrap_host,
            Kind::Test,
            "src/tools/linkchecker",
            SourceType::InTree,
            &[],
        );
        run_cargo_test(
            cargo,
            &[],
            &[],
            "linkchecker",
            "linkchecker self tests",
            bootstrap_host,
            builder,
        );

        if builder.doc_tests == DocTests::No {
            return;
        }

        // Build all the default documentation.
        builder.default_doc(&[]);

        // Build the linkchecker before calling `msg`, since GHA doesn't support nested groups.
        let linkchecker = builder.tool_cmd(Tool::Linkchecker);

        // Run the linkchecker.
        let _guard =
            builder.msg(Kind::Test, compiler.stage, "Linkcheck", bootstrap_host, bootstrap_host);
        let _time = helpers::timeit(builder);
        linkchecker.delay_failure().arg(builder.out.join(host).join("doc")).run(builder);
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        let run = run.path("src/tools/linkchecker");
        run.default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Linkcheck { host: run.target });
    }
}

fn check_if_tidy_is_installed(builder: &Builder<'_>) -> bool {
    command("tidy").allow_failure().arg("--version").run_capture_stdout(builder).is_success()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HtmlCheck {
    target: TargetSelection,
}

impl Step for HtmlCheck {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        let run = run.path("src/tools/html-checker");
        run.lazy_default_condition(Box::new(|| check_if_tidy_is_installed(builder)))
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(HtmlCheck { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        if !check_if_tidy_is_installed(builder) {
            eprintln!("not running HTML-check tool because `tidy` is missing");
            eprintln!(
                "You need the HTML tidy tool https://www.html-tidy.org/, this tool is *not* part of the rust project and needs to be installed separately, for example via your package manager."
            );
            panic!("Cannot run html-check tests");
        }
        // Ensure that a few different kinds of documentation are available.
        builder.default_doc(&[]);
        builder.ensure(crate::core::build_steps::doc::Rustc::new(
            builder.top_stage,
            self.target,
            builder,
        ));

        builder
            .tool_cmd(Tool::HtmlChecker)
            .delay_failure()
            .arg(builder.doc_out(self.target))
            .run(builder);
    }
}
