//! BDD Test Runner for Thing-OS
//!
//! Runs cucumber-rs tests against the OS in QEMU.
//!
//! Configuration via environment variables:
//! - BDD_ARCH: Target architecture (default: x86_64)
//! - BDD_FEATURE: Specific feature file to run (optional)

mod artifacts;
mod reporter;
mod steps;
mod world;

use cucumber::World;
use reporter::ThingOsReporter;
use std::fs;
use std::path::PathBuf;
use world::ThingOsWorld;

fn main() {
    // Get configuration from environment
    let arch = std::env::var("BDD_ARCH").unwrap_or_else(|_| "x86_64".to_string());
    let feature = std::env::var("BDD_FEATURE").ok();

    eprintln!("[bdd] Running tests for architecture: {}", arch);
    eprintln!("[bdd] CARGO_MANIFEST_DIR: {}", env!("CARGO_MANIFEST_DIR"));

    // Build features path
    // features are in docs/behavior/features relative to workspace root.
    // CARGO_MANIFEST_DIR is tools/bdd. So we need ../../docs/behavior/features
    let features_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap() // tools
        .parent()
        .unwrap() // root
        .join("docs/behavior/features");
    let features_path = if let Some(f) = feature {
        features_dir.join(format!("{}.feature", f))
    } else {
        features_dir
    };
    eprintln!("[bdd] Features path: {:?}", features_path);

    // Create custom reporter with artifact collection
    let reporter = ThingOsReporter::new(&arch);

    // Create output directory and JSON file
    let output_dir = PathBuf::from("docs/behavior").join(&arch);
    let _ = fs::create_dir_all(&output_dir);
    let json_file =
        fs::File::create(output_dir.join("results.json")).expect("Failed to create results.json");

    // Create JSON writer for structured output
    let json_writer = cucumber::writer::Json::new(json_file);

    // Run cucumber with tokio runtime and both reporters
    // Use catch_unwind to ensure we generate reports even if tests panic
    let run_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tokio::runtime::Runtime::new().unwrap().block_on(
            ThingOsWorld::cucumber()
                .max_concurrent_scenarios(1) // Force sequential execution to avoid global artifact race conditions
                .with_writer(cucumber::writer::Tee::new(reporter, json_writer))
                .after(
                    |_feature, _rule, _scenario, _ev, world: Option<&mut ThingOsWorld>| {
                        Box::pin(async move {
                            if let Some(w) = world {
                                w.shutdown().await;
                            }
                        })
                    },
                )
                .run(features_path),
        )
    }));

    // ALWAYS generate the report, even if tests panicked
    let failed = tokio::runtime::Runtime::new().unwrap().block_on(async {
        let collector = artifacts::global().lock().await;
        // Generate architecture report
        match collector.generate_arch_readme() {
            Ok(path) => eprintln!("[bdd] Generated: {}", path.display()),
            Err(e) => eprintln!("[bdd] WARNING: Failed to generate README: {}", e),
        }

        let (_, features_failed) = collector.count_features();
        features_failed > 0
    });

    // Check for panics first
    if run_result.is_err() {
        eprintln!("[bdd] Test run panicked - report generated before exit");
        std::process::exit(2);
    }

    if failed {
        std::process::exit(1);
    }
}
