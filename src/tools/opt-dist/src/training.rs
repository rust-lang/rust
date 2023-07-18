use crate::environment::Environment;
use crate::exec::{cmd, CmdBuilder};
use crate::utils::io::{count_files, delete_directory};
use crate::utils::with_log_group;
use anyhow::Context;
use camino::{Utf8Path, Utf8PathBuf};
use humansize::BINARY;

const LLVM_PGO_CRATES: &[&str] = &[
    "syn-1.0.89",
    "cargo-0.60.0",
    "serde-1.0.136",
    "ripgrep-13.0.0",
    "regex-1.5.5",
    "clap-3.1.6",
    "hyper-0.14.18",
];

const RUSTC_PGO_CRATES: &[&str] = &[
    "externs",
    "ctfe-stress-5",
    "cargo-0.60.0",
    "token-stream-stress",
    "match-stress",
    "tuple-stress",
    "diesel-1.4.8",
    "bitmaps-3.1.0",
];

const LLVM_BOLT_CRATES: &[&str] = LLVM_PGO_CRATES;

fn init_compiler_benchmarks(
    env: &dyn Environment,
    profiles: &[&str],
    scenarios: &[&str],
    crates: &[&str],
) -> CmdBuilder {
    // Run rustc-perf benchmarks
    // Benchmark using profile_local with eprintln, which essentially just means
    // don't actually benchmark -- just make sure we run rustc a bunch of times.
    cmd(&[
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
    .workdir(&env.rustc_perf_dir())
}

fn merge_llvm_profiles(
    env: &dyn Environment,
    merged_path: &Utf8Path,
    profile_dir: &Utf8Path,
) -> anyhow::Result<()> {
    cmd(&[
        env.downloaded_llvm_dir().join("bin/llvm-profdata").as_str(),
        "merge",
        "-o",
        merged_path.as_str(),
        profile_dir.as_str(),
    ])
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

pub struct LlvmPGOProfile(pub Utf8PathBuf);

pub fn gather_llvm_profiles(
    env: &dyn Environment,
    profile_root: &Utf8Path,
) -> anyhow::Result<LlvmPGOProfile> {
    log::info!("Running benchmarks with PGO instrumented LLVM");

    with_log_group("Running benchmarks", || {
        init_compiler_benchmarks(env, &["Debug", "Opt"], &["Full"], LLVM_PGO_CRATES)
            .run()
            .context("Cannot gather LLVM PGO profiles")
    })?;

    let merged_profile = env.opt_artifacts().join("llvm-pgo.profdata");
    log::info!("Merging LLVM PGO profiles to {merged_profile}");

    merge_llvm_profiles(env, &merged_profile, profile_root)?;
    log_profile_stats("LLVM", &merged_profile, profile_root)?;

    // We don't need the individual .profraw files now that they have been merged
    // into a final .profdata
    delete_directory(profile_root)?;

    Ok(LlvmPGOProfile(merged_profile))
}

pub struct RustcPGOProfile(pub Utf8PathBuf);

pub fn gather_rustc_profiles(
    env: &dyn Environment,
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
        init_compiler_benchmarks(env, &["Check", "Debug", "Opt"], &["All"], RUSTC_PGO_CRATES)
            .env("LLVM_PROFILE_FILE", profile_template.as_str())
            .run()
            .context("Cannot gather rustc PGO profiles")
    })?;

    let merged_profile = env.opt_artifacts().join("rustc-pgo.profdata");
    log::info!("Merging Rustc PGO profiles to {merged_profile}");

    merge_llvm_profiles(env, &merged_profile, profile_root)?;
    log_profile_stats("Rustc", &merged_profile, profile_root)?;

    // We don't need the individual .profraw files now that they have been merged
    // into a final .profdata
    delete_directory(profile_root)?;

    Ok(RustcPGOProfile(merged_profile))
}

pub struct LlvmBoltProfile(pub Utf8PathBuf);

pub fn gather_llvm_bolt_profiles(env: &dyn Environment) -> anyhow::Result<LlvmBoltProfile> {
    log::info!("Running benchmarks with BOLT instrumented LLVM");

    with_log_group("Running benchmarks", || {
        init_compiler_benchmarks(env, &["Check", "Debug", "Opt"], &["Full"], LLVM_BOLT_CRATES)
            .run()
            .context("Cannot gather LLVM BOLT profiles")
    })?;

    let merged_profile = env.opt_artifacts().join("bolt.profdata");
    let profile_root = Utf8PathBuf::from("/tmp/prof.fdata");
    log::info!("Merging LLVM BOLT profiles to {merged_profile}");

    let profiles: Vec<_> =
        glob::glob(&format!("{profile_root}*"))?.into_iter().collect::<Result<Vec<_>, _>>()?;

    let mut merge_args = vec!["merge-fdata"];
    merge_args.extend(profiles.iter().map(|p| p.to_str().unwrap()));

    with_log_group("Merging BOLT profiles", || {
        cmd(&merge_args)
            .redirect_output(merged_profile.clone())
            .run()
            .context("Cannot merge BOLT profiles")
    })?;

    log::info!("LLVM BOLT statistics");
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
    log::info!("{profile_root}: {}", humansize::format_size(size, BINARY));
    log::info!("Profile file count: {}", profiles.len());

    Ok(LlvmBoltProfile(merged_profile))
}
