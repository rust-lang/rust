use super::super::types::*;
use super::ArtifactCollector;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// Architecture-level README generation.
pub fn generate_arch_readme(collector: &ArtifactCollector) -> std::io::Result<PathBuf> {
    let readme_path = collector.base_dir.join("README.md");
    let mut file = fs::File::create(&readme_path)?;

    writeln!(file, "# BDD Test Results: {}", collector.arch)?;
    writeln!(file)?;
    writeln!(
        file,
        "> Last run: {}",
        collector.start_time.format("%Y-%m-%d %H:%M:%S")
    )?;
    writeln!(file)?;

    writeln!(file, "## Features")?;
    writeln!(file)?;
    writeln!(file, "| Feature | Scenarios | Status |")?;
    writeln!(file, "|---------|-----------|--------|")?;

    for feature in &collector.features {
        let passed_scenarios = feature.scenarios.iter().filter(|s| s.passed).count();
        let total_scenarios = feature.scenarios.len();

        // Feature passes if all scenarios pass AND there's at least one scenario
        let passed = total_scenarios > 0 && passed_scenarios == total_scenarios;
        let icon = if passed { "✅" } else { "❌" };

        let rel_path = ArtifactCollector::slugify(&feature.name); // Using simple slugify for link
        let link = format!("[{}]({}/README.md)", feature.name, rel_path);

        writeln!(
            file,
            "| {} | {}/{} | {} |",
            link, passed_scenarios, total_scenarios, icon
        )?;
    }

    Ok(readme_path)
}

/// Feature-level README generation.
pub fn write_feature_readme(
    collector: &ArtifactCollector,
    feature: &FeatureArtifacts,
) -> std::io::Result<()> {
    let readme_path = feature.dir.join("README.md");
    let mut file = fs::File::create(&readme_path)?;

    writeln!(file, "# Feature: {}", feature.name)?;
    writeln!(file)?;
    writeln!(
        file,
        "> Last run: {}",
        collector.start_time.format("%Y-%m-%d %H:%M:%S")
    )?;
    writeln!(file)?;

    writeln!(file, "## Scenarios")?;
    writeln!(file)?;
    writeln!(file, "| Scenario | Steps | Status | Link |")?;
    writeln!(file, "|----------|-------|--------|------|")?;

    for scenario in &feature.scenarios {
        let passed_steps = scenario
            .steps
            .iter()
            .filter(|s| s.result == StepResult::Passed)
            .count();
        let total_steps = scenario.steps.len();
        let icon = if scenario.passed { "✅" } else { "❌" };

        let rel_path = ArtifactCollector::slugify(&scenario.name);

        writeln!(
            file,
            "| {} | {}/{} | {} | [View Details]({}/README.md) |",
            scenario.name, passed_steps, total_steps, icon, rel_path
        )?;
    }

    Ok(())
}

pub fn write_scenario_readme(
    collector: &ArtifactCollector,
    scenario: &ScenarioArtifacts,
) -> std::io::Result<()> {
    let readme_path = scenario.dir.join("README.md");
    let mut file = fs::File::create(&readme_path)?;

    let icon = if scenario.passed { "✅" } else { "❌" };

    writeln!(file, "# {} Scenario: {}", icon, scenario.name)?;
    writeln!(file)?;
    writeln!(
        file,
        "> Last run: {}",
        collector.start_time.format("%Y-%m-%d %H:%M:%S")
    )?;
    writeln!(file)?;

    writeln!(file, "## Steps")?;
    writeln!(file)?;
    writeln!(file, "| # | Step | Result | Duration | Artifacts |")?;
    writeln!(file, "|---|------|--------|----------|-----------|")?;

    for (i, step) in scenario.steps.iter().enumerate() {
        let step_dir = format!("{:02}", i + 1);
        let screenshot_link = if step.screenshot_after.is_some() {
            format!(
                "<a href=\"./{}/after.png\"><img src=\"./{}/after.png\" width=\"150\" /></a>",
                step_dir, step_dir
            )
        } else {
            "-".to_string()
        };
        let log_link = if step.serial_log.is_some() {
            format!("[📜](./{}/serial.log)", step_dir)
        } else {
            "-".to_string()
        };
        let reg_link = if step.registers.is_some() {
            format!("[💾](./{}/registers.txt)", step_dir)
        } else {
            "-".to_string()
        };

        writeln!(
            file,
            "| {} | {} {} | {} | {}ms | {} {} {} |",
            i + 1,
            step.keyword,
            step.name,
            step.result.emoji(),
            step.duration_ms,
            screenshot_link,
            log_link,
            reg_link
        )?;
    }
    writeln!(file)?;

    let log_path = scenario.dir.join("serial.log");
    if log_path.exists() {
        if let Ok(content) = fs::read_to_string(log_path) {
            writeln!(file, "<details>")?;
            writeln!(file, "<summary>📜 Full Serial Log</summary>")?;
            writeln!(file)?;
            writeln!(file, "```")?;
            writeln!(file, "{}", content)?;
            writeln!(file, "```")?;
            writeln!(file, "</details>")?;
        }
    }

    Ok(())
}

/// Step-level README generation is disabled - the scenario README contains all needed info.
pub fn write_step_readme(_step: &StepArtifacts) -> std::io::Result<()> {
    // No-op: step READMEs are not generated - the scenario README contains the step table
    Ok(())
}
