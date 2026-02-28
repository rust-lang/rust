use std::io::Error;
use std::process::ExitCode;

use check_diff::{
    Edition, StyleEdition, check_diff, clone_repositories_for_diff_check, compile_rustfmt,
};
use clap::Parser;
use tempfile::tempdir;
use tracing::{error, info};

/// A curated set of `rust-lang/*` and popular ecosystem repositories to compare `rustfmt`s against.
const REPOS: &[&str] = &[
    // `rust-lang/*` repositories.
    "https://github.com/rust-lang/cargo.git",
    "https://github.com/rust-lang/futures-rs.git",
    "https://github.com/rust-lang/log.git",
    "https://github.com/rust-lang/mdBook.git",
    "https://github.com/rust-lang/miri.git",
    "https://github.com/rust-lang/packed_simd.git",
    "https://github.com/rust-lang/rust-analyzer.git",
    "https://github.com/rust-lang/rust-bindgen.git",
    "https://github.com/rust-lang/rust-clippy.git",
    "https://github.com/rust-lang/rust-semverver.git",
    "https://github.com/rust-lang/rustfmt.git",
    "https://github.com/rust-lang/rust.git",
    "https://github.com/rust-lang/rustlings.git",
    "https://github.com/rust-lang/rustup.git",
    // Ecosystem repositories
    "https://github.com/actix/actix.git",
    "https://github.com/bitflags/bitflags.git",
    "https://github.com/denoland/deno.git",
    "https://github.com/dtolnay/anyhow.git",
    "https://github.com/dtolnay/syn.git",
    "https://github.com/dtolnay/thiserror.git",
    "https://github.com/hyperium/hyper.git",
    "https://github.com/rustls/rustls.git",
    "https://github.com/serde-rs/serde.git",
    "https://github.com/SergioBenitez/Rocket.git",
    "https://github.com/Stebalien/tempfile.git",
];

/// Inputs for the check_diff script
#[derive(Parser)]
struct CliInputs {
    /// Git url of a rustfmt fork to compare against the latest main rustfmt
    remote_repo_url: String,
    /// Name of the feature branch on the forked repo
    feature_branch: String,
    /// Rust language `edition` used to parse code. Possible values {2015, 2018, 2021, 2024}
    #[arg(short, long, default_value = "2015")]
    edition: Edition,
    /// rustfmt `style_edition` used when formatting code. Possible vales {2015, 2018, 2021, 2024}.
    #[arg(short, long, default_value = "2021")]
    style_edition: StyleEdition,
    /// Optional commit hash from the feature branch
    #[arg(short, long)]
    commit_hash: Option<String>,
    /// Optional comma separated list of rustfmt config options to
    /// pass when running the feature branch
    #[arg(value_delimiter = ',', short, long, num_args = 1..)]
    rustfmt_config: Option<Vec<String>>,
    /// How many threads should check for formatting diffs.
    // Choosing 16 as the default since that's a common multiple of available CPU cores.
    #[arg(short, long, default_value_t = std::num::NonZeroU8::new(16).unwrap())]
    worker_threads: std::num::NonZeroU8,
}

fn main() -> Result<ExitCode, Error> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_env("CHECK_DIFF_LOG"))
        .init();
    let args = CliInputs::parse();
    let tmp_dir = tempdir()?;
    info!("Created tmp_dir {:?}", tmp_dir);

    let compilation_result = compile_rustfmt(
        tmp_dir.path(),
        args.remote_repo_url,
        args.feature_branch,
        args.edition,
        args.style_edition,
        args.commit_hash,
        args.rustfmt_config.as_deref(),
    );

    let check_diff_runners = match compilation_result {
        Ok(runner) => runner,
        Err(e) => {
            error!("Failed to compile rustfmt:\n{e:?}");
            return Ok(ExitCode::FAILURE);
        }
    };

    // Clone all repositories we plan to check
    let repositories = clone_repositories_for_diff_check(REPOS);

    info!("Starting the Diff Check");
    let errors = check_diff(&check_diff_runners, &repositories, args.worker_threads);

    if errors.is_empty() {
        info!("No diff found 😊");
        return Ok(ExitCode::SUCCESS);
    }

    for (diff, file, repo) in errors.iter() {
        let repo_name = repo.name();
        let relative_path = repo.relative_path(&file);

        error!(
            "Diff found in '{0}' when formatting {0}/{1}\n{2}",
            repo_name,
            relative_path.display(),
            diff,
        );
    }

    error!("{} formatting diffs found 💔", errors.len());
    Ok(ExitCode::FAILURE)
}
