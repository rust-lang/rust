//! Custom BDD Reporter
//!
//! Custom cucumber reporter with Thing-OS formatting
//! and artifact collection integration.

use cucumber::{
    Event, Writer, cli,
    event::{self, Cucumber},
    gherkin, parser,
    writer::{NonTransforming, Normalized},
};

use crate::artifacts::{self, StepResult};

/// Custom reporter that adds artifact collection and custom formatting.
pub struct ThingOsReporter {
    /// Whether the current scenario has any failed steps
    scenario_failed: bool,
    /// Serial log length at step start (for computing excerpt)
    step_start_serial_len: usize,
    /// Step start time for duration
    step_start_time: Option<std::time::Instant>,
    /// Track if we've seen scenario start without end (for crash recovery)
    in_scenario: bool,
}

impl ThingOsReporter {
    /// Creates a new reporter for the given architecture.
    pub fn new(arch: &str) -> Self {
        // Initialize the global artifact collector
        artifacts::init_global(arch);

        Self {
            scenario_failed: false,
            step_start_serial_len: 0,
            step_start_time: None,
            in_scenario: false,
        }
    }
}

impl<World> Writer<World> for ThingOsReporter
where
    World: cucumber::World + std::fmt::Debug,
{
    type Cli = cli::Empty;

    async fn handle_event(&mut self, ev: parser::Result<Event<Cucumber<World>>>, _cli: &Self::Cli) {
        use event::Feature;

        let Ok(event) = ev else {
            if let Err(ref e) = ev {
                eprintln!("Parse error: {:?}", e);
            }
            return;
        };

        match &event.value {
            Cucumber::Started => {
                eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
                eprintln!("║             Thing-OS BDD Test Suite                          ║");
                eprintln!("╚══════════════════════════════════════════════════════════════╝\n");
            }
            Cucumber::Finished => {
                // Ensure any pending scenario is closed
                if self.in_scenario {
                    eprintln!("│  │  │  └─ ⚠️ scenario ended unexpectedly");
                    let serial = artifacts::get_latest_serial().await;
                    let mut collector = artifacts::global().lock().await;
                    collector.on_scenario_end(false, &serial);
                    self.in_scenario = false;
                }

                let collector = artifacts::global().lock().await;
                let (passed, failed) = collector.count_scenarios();
                eprintln!("\n══════════════════════════════════════════════════════════════════");
                eprintln!("                    Test Run Complete");
                eprintln!("══════════════════════════════════════════════════════════════════");
                eprintln!("\n📊 Scenarios: {} passed, {} failed", passed, failed);
                eprintln!("📁 Results: {}\n", collector.base_dir.display());
            }
            Cucumber::Feature(feature, feat_event) => {
                match feat_event {
                    Feature::Started => {
                        eprintln!("┌─ Feature: {}", feature.name);
                        let mut collector = artifacts::global().lock().await;
                        collector.on_feature_start(&feature.name);
                    }
                    Feature::Finished => {
                        eprintln!("└─ Feature complete\n");
                        let mut collector = artifacts::global().lock().await;
                        collector.on_feature_end();
                    }
                    Feature::Scenario(scenario, retryable) => {
                        match &retryable.event {
                            event::Scenario::Started => {
                                self.scenario_failed = false;
                                self.in_scenario = true;
                                eprintln!("│  ├─ Scenario: {}", scenario.name);
                                let mut collector = artifacts::global().lock().await;
                                collector.on_scenario_start(&scenario.name);
                            }
                            event::Scenario::Finished => {
                                // Save scenario serial log
                                let serial = artifacts::get_latest_serial().await;
                                let mut collector = artifacts::global().lock().await;
                                collector.on_scenario_end(!self.scenario_failed, &serial);
                                self.in_scenario = false;
                            }
                            event::Scenario::Step(step, step_event) => {
                                self.handle_step(step, step_event).await;
                            }
                            event::Scenario::Background(_, bg_event) => {
                                if let event::Step::Started = bg_event {
                                    eprintln!("│  │  ├─ Background");
                                }
                            }
                            _ => {}
                        }
                    }
                    Feature::Rule(rule, rule_event) => match rule_event {
                        event::Rule::Started => {
                            eprintln!("│  ├─ Rule: {}", rule.name);
                        }
                        event::Rule::Finished => {}
                        event::Rule::Scenario(scenario, retryable) => match &retryable.event {
                            event::Scenario::Started => {
                                self.scenario_failed = false;
                                self.in_scenario = true;
                                eprintln!("│  │  ├─ Scenario: {}", scenario.name);
                                let mut collector = artifacts::global().lock().await;
                                collector.on_scenario_start(&scenario.name);
                            }
                            event::Scenario::Finished => {
                                let serial = artifacts::get_latest_serial().await;
                                let mut collector = artifacts::global().lock().await;
                                collector.on_scenario_end(!self.scenario_failed, &serial);
                                self.in_scenario = false;
                            }
                            _ => {}
                        },
                    },
                }
            }
            Cucumber::ParsingFinished { .. } => {}
        }
    }
}

impl ThingOsReporter {
    async fn handle_step(
        &mut self,
        step: &gherkin::Step,
        step_event: &event::Step<impl cucumber::World>,
    ) {
        match step_event {
            event::Step::Started => {
                eprintln!("│  │  ├─ {} {}", step.keyword.trim(), step.value);

                // Record step start
                let serial = artifacts::get_latest_serial().await;
                self.step_start_serial_len = serial.len();
                self.step_start_time = Some(std::time::Instant::now());

                // Try to capture a "before" screenshot
                let screenshot_before = None;

                let mut collector = artifacts::global().lock().await;
                collector.on_step_start(
                    step.keyword.trim(),
                    &step.value,
                    self.step_start_serial_len,
                    screenshot_before,
                );
            }
            event::Step::Passed(..) => {
                eprintln!("│  │  │  └─ ✅ passed");
                self.finish_step(StepResult::Passed).await;
            }
            event::Step::Skipped => {
                eprintln!("│  │  │  └─ ⏭️ skipped");
                self.finish_step(StepResult::Skipped).await;
            }
            event::Step::Failed(_, _, _, err) => {
                eprintln!(">>> REPORTER CAUGHT FAILED STEP: {:?} <<<", err);
                self.scenario_failed = true;
                eprintln!("│  │  │  └─ ❌ FAILED");
                eprintln!("│  │  │      {:?}", err);
                self.finish_step(StepResult::Failed).await;
            }
        }
    }

    async fn finish_step(&mut self, result: StepResult) {
        eprintln!(">>> FINISH STEP START: {:?} <<<", result);
        let serial = artifacts::get_latest_serial().await;

        // Try to capture a screenshot
        let screenshot_after = None;

        // Try to dump registers
        let registers = None;

        let mut collector = artifacts::global().lock().await;
        collector.on_step_end(result, None, screenshot_after, registers, &serial);

        // Update scenario's full serial log
        collector.set_scenario_serial(&serial);
    }
}

// Marker traits for compatibility with cucumber's writer utilities
impl NonTransforming for ThingOsReporter {}
impl Normalized for ThingOsReporter {}
