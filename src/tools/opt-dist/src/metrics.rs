use crate::timer::TimerSection;
use build_helper::metrics::{JsonNode, JsonRoot};
use camino::Utf8Path;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct BuildStep {
    r#type: String,
    children: Vec<BuildStep>,
    duration: Duration,
}

impl BuildStep {
    pub fn find_all_by_type(&self, r#type: &str) -> Vec<&BuildStep> {
        let mut result = Vec::new();
        self.find_by_type(r#type, &mut result);
        result
    }
    fn find_by_type<'a>(&'a self, r#type: &str, result: &mut Vec<&'a BuildStep>) {
        if self.r#type == r#type {
            result.push(self);
        }
        for child in &self.children {
            child.find_by_type(r#type, result);
        }
    }
}

/// Loads the metrics of the most recent bootstrap execution from a metrics.json file.
pub fn load_metrics(path: &Utf8Path) -> anyhow::Result<BuildStep> {
    let content = std::fs::read(path.as_std_path())?;
    let mut metrics = serde_json::from_slice::<JsonRoot>(&content)?;
    let invocation = metrics
        .invocations
        .pop()
        .ok_or_else(|| anyhow::anyhow!("No bootstrap invocation found in metrics file"))?;

    fn parse(node: JsonNode) -> Option<BuildStep> {
        match node {
            JsonNode::RustbuildStep {
                type_: kind,
                children,
                duration_excluding_children_sec,
                ..
            } => {
                let children: Vec<_> = children.into_iter().filter_map(parse).collect();
                let children_duration = children.iter().map(|c| c.duration).sum::<Duration>();
                Some(BuildStep {
                    r#type: kind.to_string(),
                    children,
                    duration: children_duration
                        + Duration::from_secs_f64(duration_excluding_children_sec),
                })
            }
            JsonNode::TestSuite(_) => None,
        }
    }

    let duration = Duration::from_secs_f64(invocation.duration_including_children_sec);
    let children: Vec<_> = invocation.children.into_iter().filter_map(parse).collect();
    Ok(BuildStep { r#type: "root".to_string(), children, duration })
}

/// Logs the individual metrics in a table and add Rustc and LLVM durations to the passed
/// timer.
pub fn record_metrics(metrics: &BuildStep, timer: &mut TimerSection) {
    let llvm_steps = metrics.find_all_by_type("bootstrap::llvm::Llvm");
    let llvm_duration: Duration = llvm_steps.into_iter().map(|s| s.duration).sum();

    let rustc_steps = metrics.find_all_by_type("bootstrap::compile::Rustc");
    let rustc_duration: Duration = rustc_steps.into_iter().map(|s| s.duration).sum();

    // The LLVM step is part of the Rustc step
    let rustc_duration = rustc_duration.saturating_sub(llvm_duration);

    if !llvm_duration.is_zero() {
        timer.add_duration("LLVM", llvm_duration);
    }
    if !rustc_duration.is_zero() {
        timer.add_duration("Rustc", rustc_duration);
    }

    log_metrics(metrics);
}

fn log_metrics(metrics: &BuildStep) {
    use std::fmt::Write;

    let mut substeps: Vec<(u32, &BuildStep)> = Vec::new();

    fn visit<'a>(step: &'a BuildStep, level: u32, substeps: &mut Vec<(u32, &'a BuildStep)>) {
        substeps.push((level, step));
        for child in &step.children {
            visit(child, level + 1, substeps);
        }
    }

    visit(metrics, 0, &mut substeps);

    let mut output = String::new();
    for (level, step) in substeps {
        let label = format!("{}{}", ".".repeat(level as usize), step.r#type);
        writeln!(output, "{label:<65}{:>8.2}s", step.duration.as_secs_f64()).unwrap();
    }
    log::info!("Build step durations\n{output}");
}
