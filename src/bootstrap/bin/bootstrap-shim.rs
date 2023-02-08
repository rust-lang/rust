use std::{env, process::Command};

use bootstrap::{Flags, MinimalConfig};

#[path = "../../../src/tools/x/src/main.rs"]
mod run_python;

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let flags = Flags::parse(&args);

    // If there are no untracked changes to bootstrap, download it from CI.
    // Otherwise, build it from source. Use python to build to avoid duplicating the code between python and rust.
    let config = MinimalConfig::parse(&flags, None);
    let bootstrap_bin = if let Some(commit) = last_modified_bootstrap_commit(&config) {
        config.download_bootstrap(&commit)
    } else {
        return run_python::main();
    };

    let args: Vec<_> = std::env::args().skip(1).collect();
    Command::new(bootstrap_bin).args(args).status().expect("failed to spawn bootstrap binairy");
}

fn last_modified_bootstrap_commit(config: &MinimalConfig) -> Option<String> {
    config.last_modified_commit(
        &["src/bootstrap", "src/tools/build_helper"],
        "download-bootstrap",
        true,
    )
}
