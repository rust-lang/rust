use anyhow::Context;
use log::LevelFilter;

use crate::environment::{create_environment, Environment};
use crate::exec::Bootstrap;
use crate::tests::run_tests;
use crate::timer::Timer;
use crate::training::{gather_llvm_bolt_profiles, gather_llvm_profiles, gather_rustc_profiles};
use crate::utils::io::reset_directory;
use crate::utils::{clear_llvm_files, format_env_variables, print_free_disk_space};

mod environment;
mod exec;
mod metrics;
mod tests;
mod timer;
mod training;
mod utils;

fn is_try_build() -> bool {
    std::env::var("DIST_TRY_BUILD").unwrap_or_else(|_| "0".to_string()) != "0"
}

fn execute_pipeline(
    env: &dyn Environment,
    timer: &mut Timer,
    dist_args: Vec<String>,
) -> anyhow::Result<()> {
    reset_directory(&env.opt_artifacts())?;
    env.prepare_rustc_perf()?;

    // Stage 1: Build PGO instrumented rustc
    // We use a normal build of LLVM, because gathering PGO profiles for LLVM and `rustc` at the
    // same time can cause issues, because the host and in-tree LLVM versions can diverge.
    let rustc_pgo_profile = timer.section("Stage 1 (Rustc PGO)", |stage| {
        let rustc_profile_dir_root = env.opt_artifacts().join("rustc-pgo");

        stage.section("Build PGO instrumented rustc and LLVM", |section| {
            Bootstrap::build(env).rustc_pgo_instrument(&rustc_profile_dir_root).run(section)
        })?;

        let profile = stage
            .section("Gather profiles", |_| gather_rustc_profiles(env, &rustc_profile_dir_root))?;
        print_free_disk_space()?;

        stage.section("Build PGO optimized rustc", |section| {
            Bootstrap::build(env).rustc_pgo_optimize(&profile).run(section)
        })?;

        Ok(profile)
    })?;

    // Stage 2: Gather LLVM PGO profiles
    // Here we build a PGO instrumented LLVM, reusing the previously PGO optimized rustc.
    // Then we use the instrumented LLVM to gather LLVM PGO profiles.
    let llvm_pgo_profile = timer.section("Stage 2 (LLVM PGO)", |stage| {
        // Remove the previous, uninstrumented build of LLVM.
        clear_llvm_files(env)?;

        let llvm_profile_dir_root = env.opt_artifacts().join("llvm-pgo");

        stage.section("Build PGO instrumented LLVM", |section| {
            Bootstrap::build(env)
                .llvm_pgo_instrument(&llvm_profile_dir_root)
                .avoid_rustc_rebuild()
                .run(section)
        })?;

        let profile = stage
            .section("Gather profiles", |_| gather_llvm_profiles(env, &llvm_profile_dir_root))?;

        print_free_disk_space()?;

        // Proactively delete the instrumented artifacts, to avoid using them by accident in
        // follow-up stages.
        clear_llvm_files(env)?;

        Ok(profile)
    })?;

    let llvm_bolt_profile = if env.supports_bolt() {
        // Stage 3: Build BOLT instrumented LLVM
        // We build a PGO optimized LLVM in this step, then instrument it with BOLT and gather BOLT profiles.
        // Note that we don't remove LLVM artifacts after this step, so that they are reused in the final dist build.
        // BOLT instrumentation is performed "on-the-fly" when the LLVM library is copied to the sysroot of rustc,
        // therefore the LLVM artifacts on disk are not "tainted" with BOLT instrumentation and they can be reused.
        timer.section("Stage 3 (LLVM BOLT)", |stage| {
            stage.section("Build BOLT instrumented LLVM", |stage| {
                Bootstrap::build(env)
                    .llvm_bolt_instrument()
                    .llvm_pgo_optimize(&llvm_pgo_profile)
                    .avoid_rustc_rebuild()
                    .run(stage)
            })?;

            let profile = stage.section("Gather profiles", |_| gather_llvm_bolt_profiles(env))?;
            print_free_disk_space()?;

            // LLVM is not being cleared here, we want to reuse the previous PGO-optimized build

            Ok(Some(profile))
        })?
    } else {
        None
    };

    let mut dist = Bootstrap::dist(env, &dist_args)
        .llvm_pgo_optimize(&llvm_pgo_profile)
        .rustc_pgo_optimize(&rustc_pgo_profile)
        .avoid_rustc_rebuild();

    if let Some(llvm_bolt_profile) = llvm_bolt_profile {
        dist = dist.llvm_bolt_optimize(&llvm_bolt_profile);
    }

    // Final stage: Assemble the dist artifacts
    // The previous PGO optimized rustc build and PGO optimized LLVM builds should be reused.
    timer.section("Stage 4 (final build)", |stage| dist.run(stage))?;

    // After dist has finished, run a subset of the test suite on the optimized artifacts to discover
    // possible regressions.
    // The tests are not executed for try builds, which can be in various broken states, so we don't
    // want to gatekeep them with tests.
    if !is_try_build() {
        timer.section("Run tests", |_| run_tests(env))?;
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    // Make sure that we get backtraces for easier debugging in CI
    std::env::set_var("RUST_BACKTRACE", "1");

    env_logger::builder()
        .filter_level(LevelFilter::Info)
        .format_timestamp_millis()
        .parse_default_env()
        .init();

    let mut build_args: Vec<String> = std::env::args().skip(1).collect();
    log::info!("Running optimized build pipeline with args `{}`", build_args.join(" "));
    log::info!("Environment values\n{}", format_env_variables());

    if let Ok(config) = std::fs::read_to_string("config.toml") {
        log::info!("Contents of `config.toml`:\n{config}");
    }

    // Skip components that are not needed for try builds to speed them up
    if is_try_build() {
        log::info!("Skipping building of unimportant components for a try build");
        for target in [
            "rust-docs",
            "rustc-docs",
            "rust-docs-json",
            "rust-analyzer",
            "rustc-src",
            "clippy",
            "miri",
            "rustfmt",
        ] {
            build_args.extend(["--exclude".to_string(), target.to_string()]);
        }
    }

    let mut timer = Timer::new();
    let env = create_environment();

    let result = execute_pipeline(env.as_ref(), &mut timer, build_args);
    log::info!("Timer results\n{}", timer.format_stats());

    print_free_disk_space()?;

    result.context("Optimized build pipeline has failed")
}
