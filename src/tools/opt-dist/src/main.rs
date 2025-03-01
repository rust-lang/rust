use anyhow::Context;
use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;
use log::LevelFilter;
use utils::io;

use crate::bolt::{bolt_optimize, with_bolt_instrumented};
use crate::environment::{Environment, EnvironmentBuilder};
use crate::exec::{Bootstrap, cmd};
use crate::tests::run_tests;
use crate::timer::Timer;
use crate::training::{
    gather_bolt_profiles, gather_llvm_profiles, gather_rustc_profiles, llvm_benchmarks,
    rustc_benchmarks,
};
use crate::utils::artifact_size::print_binary_sizes;
use crate::utils::io::{copy_directory, reset_directory};
use crate::utils::{
    clear_llvm_files, format_env_variables, print_free_disk_space, with_log_group,
    write_timer_to_summary,
};

mod bolt;
mod environment;
mod exec;
mod metrics;
mod tests;
mod timer;
mod training;
mod utils;

#[derive(clap::Parser, Debug)]
struct Args {
    #[clap(subcommand)]
    env: EnvironmentCmd,
}

#[derive(clap::Parser, Clone, Debug)]
struct SharedArgs {
    // Arguments passed to `x` to perform the final (dist) build.
    build_args: Vec<String>,
}

#[derive(clap::Parser, Clone, Debug)]
enum EnvironmentCmd {
    /// Perform a custom local PGO/BOLT optimized build.
    Local {
        /// Target triple of the host.
        #[arg(long)]
        target_triple: String,

        /// Checkout directory of `rustc`.
        #[arg(long)]
        checkout_dir: Utf8PathBuf,

        /// Host LLVM installation directory.
        #[arg(long)]
        llvm_dir: Utf8PathBuf,

        /// Python binary to use in bootstrap invocations.
        #[arg(long, default_value = "python3")]
        python: String,

        /// Directory where artifacts (like PGO profiles or rustc-perf) of this workflow
        /// will be stored.
        #[arg(long, default_value = "opt-artifacts")]
        artifact_dir: Utf8PathBuf,

        /// Checkout directory of `rustc-perf`.
        ///
        /// If unspecified, defaults to the rustc-perf submodule in the rustc checkout dir
        /// (`src/tools/rustc-perf`), which should have been initialized when building this tool.
        // FIXME: Move update_submodule into build_helper, that way we can also ensure the submodule
        // is updated when _running_ opt-dist, rather than building.
        #[arg(long)]
        rustc_perf_checkout_dir: Option<Utf8PathBuf>,

        /// Is LLVM for `rustc` built in shared library mode?
        #[arg(long, default_value_t = true)]
        llvm_shared: bool,

        /// Should BOLT optimization be used? If yes, host LLVM must have BOLT binaries
        /// (`llvm-bolt` and `merge-fdata`) available.
        #[arg(long, default_value_t = false)]
        use_bolt: bool,

        /// Tests that should be skipped when testing the optimized compiler.
        #[arg(long)]
        skipped_tests: Vec<String>,

        #[clap(flatten)]
        shared: SharedArgs,

        /// Arguments passed to `rustc-perf --cargo-config <value>` when running benchmarks.
        #[arg(long)]
        benchmark_cargo_config: Vec<String>,
    },
    /// Perform an optimized build on Linux CI, from inside Docker.
    LinuxCi {
        #[clap(flatten)]
        shared: SharedArgs,
    },
    /// Perform an optimized build on Windows CI, directly inside Github Actions.
    WindowsCi {
        #[clap(flatten)]
        shared: SharedArgs,
    },
}

fn is_try_build() -> bool {
    std::env::var("DIST_TRY_BUILD").unwrap_or_else(|_| "0".to_string()) != "0"
}

fn create_environment(args: Args) -> anyhow::Result<(Environment, Vec<String>)> {
    let (env, args) = match args.env {
        EnvironmentCmd::Local {
            target_triple,
            checkout_dir,
            llvm_dir,
            python,
            artifact_dir,
            rustc_perf_checkout_dir,
            llvm_shared,
            use_bolt,
            skipped_tests,
            benchmark_cargo_config,
            shared,
        } => {
            let env = EnvironmentBuilder::default()
                .host_tuple(target_triple)
                .python_binary(python)
                .checkout_dir(checkout_dir.clone())
                .host_llvm_dir(llvm_dir)
                .artifact_dir(artifact_dir)
                .build_dir(checkout_dir)
                .prebuilt_rustc_perf(rustc_perf_checkout_dir)
                .shared_llvm(llvm_shared)
                .use_bolt(use_bolt)
                .skipped_tests(skipped_tests)
                .benchmark_cargo_config(benchmark_cargo_config)
                .build()?;

            (env, shared.build_args)
        }
        EnvironmentCmd::LinuxCi { shared } => {
            let target_triple =
                std::env::var("PGO_HOST").expect("PGO_HOST environment variable missing");

            let is_aarch64 = target_triple.starts_with("aarch64");

            let checkout_dir = Utf8PathBuf::from("/checkout");
            let env = EnvironmentBuilder::default()
                .host_tuple(target_triple)
                .python_binary("python3".to_string())
                .checkout_dir(checkout_dir.clone())
                .host_llvm_dir(Utf8PathBuf::from("/rustroot"))
                .artifact_dir(Utf8PathBuf::from("/tmp/tmp-multistage/opt-artifacts"))
                .build_dir(checkout_dir.join("obj"))
                .shared_llvm(true)
                // FIXME: Enable bolt for aarch64 once it's fixed upstream. Broken as of December 2024.
                .use_bolt(!is_aarch64)
                .skipped_tests(vec![])
                .build()?;

            (env, shared.build_args)
        }
        EnvironmentCmd::WindowsCi { shared } => {
            let target_triple =
                std::env::var("PGO_HOST").expect("PGO_HOST environment variable missing");

            let checkout_dir: Utf8PathBuf = std::env::current_dir()?.try_into()?;
            let env = EnvironmentBuilder::default()
                .host_tuple(target_triple)
                .python_binary("python".to_string())
                .checkout_dir(checkout_dir.clone())
                .host_llvm_dir(checkout_dir.join("citools").join("clang-rust"))
                .artifact_dir(checkout_dir.join("opt-artifacts"))
                .build_dir(checkout_dir)
                .shared_llvm(false)
                .use_bolt(false)
                .skipped_tests(vec![])
                .build()?;

            (env, shared.build_args)
        }
    };
    Ok((env, args))
}

fn execute_pipeline(
    env: &Environment,
    timer: &mut Timer,
    dist_args: Vec<String>,
) -> anyhow::Result<()> {
    reset_directory(&env.artifact_dir())?;

    with_log_group("Building rustc-perf", || {
        let rustc_perf_checkout_dir = match env.prebuilt_rustc_perf() {
            Some(dir) => dir,
            None => env.checkout_path().join("src").join("tools").join("rustc-perf"),
        };
        copy_rustc_perf(env, &rustc_perf_checkout_dir)
    })?;

    // Stage 1: Build PGO instrumented rustc
    // We use a normal build of LLVM, because gathering PGO profiles for LLVM and `rustc` at the
    // same time can cause issues, because the host and in-tree LLVM versions can diverge.
    let rustc_pgo_profile = timer.section("Stage 1 (Rustc PGO)", |stage| {
        let rustc_profile_dir_root = env.artifact_dir().join("rustc-pgo");

        stage.section("Build PGO instrumented rustc and LLVM", |section| {
            let mut builder = Bootstrap::build(env).rustc_pgo_instrument(&rustc_profile_dir_root);

            if env.supports_shared_llvm() {
                // This first LLVM that we build will be thrown away after this stage, and it
                // doesn't really need LTO. Without LTO, it builds in ~1 minute thanks to sccache,
                // with LTO it takes almost 10 minutes. It makes the followup Rustc PGO
                // instrumented/optimized build a bit slower, but it seems to be worth it.
                builder = builder.without_llvm_lto();
            }

            builder.run(section)
        })?;

        let profile = stage
            .section("Gather profiles", |_| gather_rustc_profiles(env, &rustc_profile_dir_root))?;
        print_free_disk_space()?;

        stage.section("Build PGO optimized rustc", |section| {
            let mut cmd = Bootstrap::build(env).rustc_pgo_optimize(&profile);
            if env.use_bolt() {
                cmd = cmd.with_rustc_bolt_ldflags();
            }

            cmd.run(section)
        })?;

        Ok(profile)
    })?;

    // Stage 2: Gather LLVM PGO profiles
    // Here we build a PGO instrumented LLVM, reusing the previously PGO optimized rustc.
    // Then we use the instrumented LLVM to gather LLVM PGO profiles.
    let llvm_pgo_profile = timer.section("Stage 2 (LLVM PGO)", |stage| {
        // Remove the previous, uninstrumented build of LLVM.
        clear_llvm_files(env)?;

        let llvm_profile_dir_root = env.artifact_dir().join("llvm-pgo");

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

    let bolt_profiles = if env.use_bolt() {
        // Stage 3: Build BOLT instrumented LLVM
        // We build a PGO optimized LLVM in this step, then instrument it with BOLT and gather BOLT profiles.
        // Note that we don't remove LLVM artifacts after this step, so that they are reused in the final dist build.
        // BOLT instrumentation is performed "on-the-fly" when the LLVM library is copied to the sysroot of rustc,
        // therefore the LLVM artifacts on disk are not "tainted" with BOLT instrumentation and they can be reused.
        timer.section("Stage 3 (BOLT)", |stage| {
            stage.section("Build PGO optimized LLVM", |stage| {
                Bootstrap::build(env)
                    .with_llvm_bolt_ldflags()
                    .llvm_pgo_optimize(&llvm_pgo_profile)
                    .avoid_rustc_rebuild()
                    .run(stage)
            })?;

            let libdir = env.build_artifacts().join("stage2").join("lib");
            // The actual name will be something like libLLVM.so.18.1-rust-dev.
            let llvm_lib = io::find_file_in_dir(&libdir, "libLLVM.so", "")?;

            log::info!("Optimizing {llvm_lib} with BOLT");

            // FIXME(kobzol): try gather profiles together, at once for LLVM and rustc
            // Instrument the libraries and gather profiles
            let llvm_profile = with_bolt_instrumented(&llvm_lib, |llvm_profile_dir| {
                stage.section("Gather profiles", |_| {
                    gather_bolt_profiles(env, "LLVM", llvm_benchmarks(env), llvm_profile_dir)
                })
            })?;
            print_free_disk_space()?;

            // Now optimize the library with BOLT. The `libLLVM-XXX.so` library is actually hard-linked
            // from several places, and this specific path (`llvm_lib`) will *not* be packaged into
            // the final dist build. However, when BOLT optimizes an artifact, it does so *in-place*,
            // therefore it will actually optimize all the hard links, which means that the final
            // packaged `libLLVM.so` file *will* be BOLT optimized.
            bolt_optimize(&llvm_lib, &llvm_profile, env)
                .context("Could not optimize LLVM with BOLT")?;

            let rustc_lib = io::find_file_in_dir(&libdir, "librustc_driver", ".so")?;

            log::info!("Optimizing {rustc_lib} with BOLT");

            // Instrument it and gather profiles
            let rustc_profile = with_bolt_instrumented(&rustc_lib, |rustc_profile_dir| {
                stage.section("Gather profiles", |_| {
                    gather_bolt_profiles(env, "rustc", rustc_benchmarks(env), rustc_profile_dir)
                })
            })?;
            print_free_disk_space()?;

            // Now optimize the library with BOLT.
            bolt_optimize(&rustc_lib, &rustc_profile, env)
                .context("Could not optimize rustc with BOLT")?;

            // LLVM is not being cleared here, we want to use the BOLT-optimized LLVM
            Ok(vec![llvm_profile, rustc_profile])
        })?
    } else {
        vec![]
    };

    let mut dist = Bootstrap::dist(env, &dist_args)
        .llvm_pgo_optimize(&llvm_pgo_profile)
        .rustc_pgo_optimize(&rustc_pgo_profile)
        .avoid_rustc_rebuild();

    for bolt_profile in bolt_profiles {
        dist = dist.with_bolt_profile(bolt_profile);
    }

    // Final stage: Assemble the dist artifacts
    // The previous PGO optimized rustc build and PGO optimized LLVM builds should be reused.
    timer.section("Stage 5 (final build)", |stage| dist.run(stage))?;

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

    let args = Args::parse();

    println!("Running optimized build pipeline with args `{:?}`", args);

    with_log_group("Environment values", || {
        println!("Environment values\n{}", format_env_variables());
    });

    with_log_group("Printing bootstrap.toml", || {
        let config_file = if std::path::Path::new("bootstrap.toml").exists() {
            "bootstrap.toml"
        } else {
            "config.toml" // Fall back for backward compatibility
        };

        if let Ok(config) = std::fs::read_to_string(config_file) {
            println!("Contents of `bootstrap.toml`:\n{config}");
        } else {
            eprintln!("Failed to read `{}`", config_file);
        }
    });

    let (env, mut build_args) = create_environment(args).context("Cannot create environment")?;

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
            build_args.extend(["--skip".to_string(), target.to_string()]);
        }
    }

    let mut timer = Timer::new();

    let result = execute_pipeline(&env, &mut timer, build_args);
    log::info!("Timer results\n{}", timer.format_stats());

    if let Ok(summary_path) = std::env::var("GITHUB_STEP_SUMMARY") {
        write_timer_to_summary(&summary_path, &timer)?;
    }

    print_free_disk_space()?;
    result.context("Optimized build pipeline has failed")?;
    print_binary_sizes(&env)?;

    Ok(())
}

// Copy rustc-perf from the given path into the environment and build it.
fn copy_rustc_perf(env: &Environment, dir: &Utf8Path) -> anyhow::Result<()> {
    copy_directory(dir, &env.rustc_perf_dir())?;
    build_rustc_perf(env)
}

fn build_rustc_perf(env: &Environment) -> anyhow::Result<()> {
    cmd(&[env.cargo_stage_0().as_str(), "build", "-p", "collector"])
        .workdir(&env.rustc_perf_dir())
        .env("RUSTC", &env.rustc_stage_0().into_string())
        .env("RUSTC_BOOTSTRAP", "1")
        .run()?;
    Ok(())
}
