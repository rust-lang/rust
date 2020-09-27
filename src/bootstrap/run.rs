use crate::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::tool::Tool;
use std::process::Command;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ExpandYamlAnchors;

impl Step for ExpandYamlAnchors {
    type Output = ();

    /// Runs the `expand-yaml_anchors` tool.
    ///
    /// This tool in `src/tools` reads the CI configuration files written in YAML and expands the
    /// anchors in them, since GitHub Actions doesn't support them.
    fn run(self, builder: &Builder<'_>) {
        builder.info("Expanding YAML anchors in the GitHub Actions configuration");
        try_run(
            builder,
            &mut builder.tool_cmd(Tool::ExpandYamlAnchors).arg("generate").arg(&builder.src),
        );
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/expand-yaml-anchors")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(ExpandYamlAnchors);
    }
}

fn try_run(builder: &Builder<'_>, cmd: &mut Command) -> bool {
    if !builder.fail_fast {
        if !builder.try_run(cmd) {
            let mut failures = builder.delayed_failures.borrow_mut();
            failures.push(format!("{:?}", cmd));
            return false;
        }
    } else {
        builder.run(cmd);
    }
    true
}
