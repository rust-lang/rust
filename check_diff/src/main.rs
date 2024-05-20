use clap::Parser;
/// Inputs for the check_diff script
#[derive(Parser)]
struct CliInputs {
    /// Git url of a rustfmt fork to compare against the latest master rustfmt
    remote_repo_url: String,
    /// Name of the feature branch on the forked repo
    feature_branch: String,
    /// Optional commit hash from the feature branch
    #[arg(short, long)]
    commit_hash: Option<String>,
    /// Optional comma separated list of rustfmt config options to
    /// pass when running the feature branch
    #[arg(value_delimiter = ',', short, long, num_args = 1..)]
    rustfmt_config: Option<Vec<String>>,
}

fn main() {
    let args = CliInputs::parse();
    println!(
        "remote_repo_url: {:?}, feature_branch: {:?},
        optional_commit_hash: {:?}, optional_rustfmt_config: {:?}",
        args.remote_repo_url, args.feature_branch, args.commit_hash, args.rustfmt_config
    );
}
