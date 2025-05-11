use anyhow::Context;
use build_helper::{LLVM_PGO_CRATES, RUSTC_PGO_CRATES};
use camino::{Utf8Path, Utf8PathBuf};
use humansize::BINARY;

use crate::environment::{Environment, executable_extension};
use crate::exec::{CmdBuilder, cmd};
use crate::utils::io::{count_files, delete_directory};
use crate::utils::with_log_group;

fn init_compiler_benchmarks(
    env: &Environment,
    profiles: &[&str],
    scenarios: &[&str],
    crates: &[&str],
) -> CmdBuilder {
    // Run rustc-perf benchmarks
    // Benchmark using profile_local with eprintln, which essentially just means
    // don't actually benchmark -- just make sure we run rustc a bunch of times.
    let mut cmd = cmd(&[
        env.cargo_stage_0().as_str(),
        "run",
        "-p",
        "collector",
        "--bin",
        "collector",
        "--",
        "profile_local",
        "eprintln",
        env.rustc_stage_2().as_str(),
        "--id",
        "Test",
        "--cargo",
        env.cargo_stage_0().as_str(),
        "--profiles",
        profiles.join(",").as_str(),
        "--scenarios",
        scenarios.join(",").as_str(),
        "--include",
        crates.join(",").as_str(),
    ])
    .env("RUST_LOG", "collector=debug")
    .env("RUSTC", env.rustc_stage_0().as_str())
    .env("RUSTC_BOOTSTRAP", "1")
    .workdir(&env.rustc_perf_dir());

    // This propagates cargo configs to `rustc-perf --cargo-config`,
    // which is particularly useful when the environment is air-gapped,
    // and you want to use the default set of training crates vendored
    // in the rustc-src tarball.
    for config in env.benchmark_cargo_config() {
        cmd = cmd.arg("--cargo-config").arg(config);
    }

    cmd
}

/// Describes which `llvm-profdata` binary should be used for merging PGO profiles.
enum LlvmProfdata {
    /// Use llvm-profdata from the host toolchain (i.e. from LLVM provided externally).
    Host,
    /// Use llvm-profdata from the target toolchain (i.e. from LLVM built from `src/llvm-project`).
    Target,
}

fn merge_llvm_profiles(
    env: &Environment,
    merged_path: &Utf8Path,
    profile_dir: &Utf8Path,
    profdata: LlvmProfdata,
) -> anyhow::Result<()> {
    let llvm_profdata = match profdata {
        LlvmProfdata::Host => {
            env.host_llvm_dir().join(format!("bin/llvm-profdata{}", executable_extension()))
        }
        LlvmProfdata::Target => env
            .build_artifacts()
            .join("llvm")
            .join("build")
            .join(format!("bin/llvm-profdata{}", executable_extension())),
    };

    cmd(&[llvm_profdata.as_str(), "merge", "-o", merged_path.as_str(), profile_dir.as_str()])
        .run()
        .context("Cannot merge LLVM profiles")?;
    Ok(())
}

fn log_profile_stats(
    name: &str,
    merged_profile: &Utf8Path,
    profile_root: &Utf8Path,
) -> anyhow::Result<()> {
    log::info!("{name} PGO statistics");
    log::info!(
        "{merged_profile}: {}",
        humansize::format_size(std::fs::metadata(merged_profile.as_std_path())?.len(), BINARY)
    );
    log::info!(
        "{profile_root}: {}",
        humansize::format_size(fs_extra::dir::get_size(profile_root.as_std_path())?, BINARY)
    );
    log::info!("Profile file count: {}", count_files(profile_root)?);
    Ok(())
}

pub fn llvm_benchmarks(env: &Environment) -> CmdBuilder {
    init_compiler_benchmarks(env, &["Debug", "Opt"], &["Full"], LLVM_PGO_CRATES)
}

pub fn rustc_benchmarks(env: &Environment) -> CmdBuilder {
    init_compiler_benchmarks(env, &["Check", "Debug", "Opt"], &["All"], RUSTC_PGO_CRATES)
}

pub struct LlvmPGOProfile(pub Utf8PathBuf);

pub fn gather_llvm_profiles(
    env: &Environment,
    profile_root: &Utf8Path,
) -> anyhow::Result<LlvmPGOProfile> {
    log::info!("Running benchmarks with PGO instrumented LLVM");

    with_log_group("Running benchmarks", || {
        llvm_benchmarks(env).run().context("Cannot gather LLVM PGO profiles")
    })?;

    let merged_profile = env.artifact_dir().join("llvm-pgo.profdata");
    log::info!("Merging LLVM PGO profiles to {merged_profile}");

    merge_llvm_profiles(env, &merged_profile, profile_root, LlvmProfdata::Host)?;
    log_profile_stats("LLVM", &merged_profile, profile_root)?;

    // We don't need the individual .profraw files now that they have been merged
    // into a final .profdata
    delete_directory(profile_root)?;

    Ok(LlvmPGOProfile(merged_profile))
}

pub struct RustcPGOProfile(pub Utf8PathBuf);

pub fn gather_rustc_profiles(
    env: &Environment,
    profile_root: &Utf8Path,
) -> anyhow::Result<RustcPGOProfile> {
    log::info!("Running benchmarks with PGO instrumented rustc");

    // The profile data is written into a single filepath that is being repeatedly merged when each
    // rustc invocation ends. Empirically, this can result in some profiling data being lost. That's
    // why we override the profile path to include the PID. This will produce many more profiling
    // files, but the resulting profile will produce a slightly faster rustc binary.
    let profile_template = profile_root.join("default_%m_%p.profraw");

    // Here we're profiling the `rustc` frontend, so we also include `Check`.
    // The benchmark set includes various stress tests that put the frontend under pressure.
    with_log_group("Running benchmarks", || {
        rustc_benchmarks(env)
            .env("LLVM_PROFILE_FILE", profile_template.as_str())
            .run()
            .context("Cannot gather rustc PGO profiles")
    })?;

    let merged_profile = env.artifact_dir().join("rustc-pgo.profdata");
    log::info!("Merging Rustc PGO profiles to {merged_profile}");

    merge_llvm_profiles(env, &merged_profile, profile_root, LlvmProfdata::Target)?;
    log_profile_stats("Rustc", &merged_profile, profile_root)?;

    // We don't need the individual .profraw files now that they have been merged
    // into a final .profdata
    delete_directory(profile_root)?;

    Ok(RustcPGOProfile(merged_profile))
}

pub struct BoltProfile(pub Utf8PathBuf);

pub fn gather_bolt_profiles(
    env: &Environment,
    name: &str,
    benchmarks: CmdBuilder,
    profile_prefix: &Utf8Path,
) -> anyhow::Result<BoltProfile> {
    log::info!("Running benchmarks with BOLT instrumented {name}");

    with_log_group("Running benchmarks", || {
        benchmarks.run().with_context(|| "Cannot gather {name} BOLT profiles")
    })?;

    let merged_profile = env.artifact_dir().join(format!("{name}-bolt.profdata"));
    log::info!("Merging {name} BOLT profiles from {profile_prefix} to {merged_profile}");

    let profiles: Vec<_> =
        glob::glob(&format!("{profile_prefix}*"))?.collect::<Result<Vec<_>, _>>()?;

    let mut merge_args = vec!["merge-fdata"];
    merge_args.extend(profiles.iter().map(|p| p.to_str().unwrap()));

    with_log_group("Merging BOLT profiles", || {
        cmd(&merge_args)
            .redirect_output(merged_profile.clone())
            .run()
            .context("Cannot merge BOLT profiles")
    })?;

    log::info!("{name} BOLT statistics");
    log::info!(
        "{merged_profile}: {}",
        humansize::format_size(std::fs::metadata(merged_profile.as_std_path())?.len(), BINARY)
    );

    let size = profiles
        .iter()
        .map(|p| std::fs::metadata(p).map(|metadata| metadata.len()))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum::<u64>();
    log::info!("{profile_prefix}: {}", humansize::format_size(size, BINARY));
    log::info!("Profile file count: {}", profiles.len());

    // Delete the gathered profiles
    for profile in glob::glob(&format!("{profile_prefix}*"))?.flatten() {
        if let Err(error) = std::fs::remove_file(&profile) {
            log::error!("Cannot delete BOLT profile {}: {error:?}", profile.display());
        }
    }

    Ok(BoltProfile(merged_profile))
}
