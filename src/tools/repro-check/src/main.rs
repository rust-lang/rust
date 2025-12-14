//! Tool to check if Rust compiler builds are reproducible.
//! Copies source to two dirs, builds each, hashes artifacts, compares.
//! Handy for spotting non-determinism.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use log::{debug, info, trace};
use repro_check::{build, compare, config, fs_utils};

#[derive(Parser, Debug)]
#[command(author, version, about = "Checks Rust build reproducibility", long_about = None)]
struct Args {
    #[arg(long, default_value = ".")]
    src_root: PathBuf,

    #[arg(long)]
    target: Option<String>,

    #[arg(long, default_value = "repro_report.html")]
    html_output: PathBuf,

    #[arg(short, long, default_value_t = num_cpus::get() as u32)]
    jobs: u32,

    #[arg(long)]
    skip_copy: bool,

    #[arg(long, default_value_t = 10)]
    path_delta: usize,

    #[arg(long)]
    full_dist: bool,

    #[arg(long)]
    clean: bool,

    #[arg(long)]
    exclude_pattern: Vec<String>,

    #[arg(long)]
    verbose: bool,
}

// Extracted to keep main cleaner - runs a single build.
fn run_one_build(
    env_dir: &Path,
    jobs: u32,
    target: Option<&str>,
    full_dist: bool,
    label: &str,
) -> Result<()> {
    info!("Starting {} build in {:?}", label, env_dir);
    let start = Instant::now();
    build::run_xpy(env_dir, jobs, target, full_dist)?;
    info!("{} build done in {:?}", label, start.elapsed());
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let log_level = if args.verbose { log::LevelFilter::Debug } else { log::LevelFilter::Info };
    env_logger::builder().filter_level(log_level).init();

    let mut excludes = HashSet::from([
        "metrics.json".to_string(),
        ".lock".to_string(),
        "git-commit-info".to_string(),
        "Cargo.lock".to_string(),
        ".log".to_string(),
    ]);
    for pat in args.exclude_pattern {
        excludes.insert(pat);
    }
    debug!("Excludes: {:?}", excludes);

    let src_root = std::fs::canonicalize(&args.src_root).context("Bad source root path")?;
    info!("Repro check starting from {:?}", src_root);

    let workspace = src_root.join("build/repro_workspace");
    if args.clean {
        fs_utils::clean_workspace(&workspace)?;
    }

    let (env_a, env_b) =
        fs_utils::prepare_workspace(&workspace, &src_root, args.path_delta, args.skip_copy)?;

    // Pass the requested target (if any) into the generated bootstrap.toml
    let target_for_config = args.target.as_deref();
    config::write_bootstrap_toml(&env_a, target_for_config)?;
    config::write_bootstrap_toml(&env_b, target_for_config)?;

    info!("-----------------------------------------------");

    run_one_build(&env_a, args.jobs, args.target.as_deref(), args.full_dist, "A")?;

    info!("-----------------------------------------------");

    run_one_build(&env_b, args.jobs, args.target.as_deref(), args.full_dist, "B")?;

    info!("-----------------------------------------------");
    info!("Now comparing...");

    // The stage2 compiler and sysroot are *always* built under the host triple,
    // even when cross-compiling. The target-specific libraries live inside
    // `lib/rustlib/<target>/` under the host stage2 directory.
    let host = build::detect_host(&src_root)?;
    let stage2_path = Path::new("build").join(&host).join("stage2");

    let path_a = env_a.join(&stage2_path);
    let path_b = env_b.join(&stage2_path);

    if !path_a.exists() || !path_b.exists() {
        bail!(
            "Missing stage2 directories at {} — build may have failed or been incomplete",
            stage2_path.display()
        );
    }
    let report = compare::compare_directories(&path_a, &path_b, &host, &excludes)?;

    if args.verbose {
        debug!("Ignored:");
        for (p, pat) in &report.ignored_files {
            trace!("- {} (via {})", p.display(), pat);
        }
        debug!("Compared:");
        for p in &report.compared_files {
            trace!("- {}", p.display());
        }
    }

    compare::generate_html_report(&report, &args.html_output)?;

    if report.mismatches.is_empty() {
        info!("All good - builds match!");
    } else {
        bail!("Mismatches: {} - see {}", report.mismatches.len(), args.html_output.display());
    }

    Ok(())
}
