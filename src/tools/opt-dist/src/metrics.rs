use std::time::Duration;

use build_helper::metrics::{BuildStep, JsonRoot, format_build_steps};
use camino::Utf8Path;

use crate::timer::TimerSection;

/// Loads the metrics of the most recent bootstrap execution from a metrics.json file.
pub fn load_metrics(path: &Utf8Path) -> anyhow::Result<BuildStep> {
    let content = std::fs::read(path.as_std_path())?;
    let mut metrics = serde_json::from_slice::<JsonRoot>(&content)?;
    let invocation = metrics
        .invocations
        .pop()
        .ok_or_else(|| anyhow::anyhow!("No bootstrap invocation found in metrics file"))?;

    Ok(BuildStep::from_invocation(&invocation))
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

    let output = format_build_steps(metrics);
    log::info!("Build step durations\n{output}");
}
