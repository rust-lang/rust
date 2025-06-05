use std::io::{Read, Write};
use std::process::exit;
use std::{env, io};

use crate::sync::{GitSync, Josh};

mod sync;

const USAGE: &str = r#"Utility for synchroniing compiler-builtins with rust-lang/rust

Usage:

    josh-sync rustc-pull

        Pull from rust-lang/rust to compiler-builtins. Creates a commit
        updating the version file, followed by a merge commit.

    josh-sync rustc-push GITHUB_USERNAME [BRANCH]

        Create a branch off of rust-lang/rust updating compiler-builtins.
"#;

fn main() {
    let sync = GitSync::from_current_dir();

    // Collect args, then recollect as str refs so we can match on them
    let args: Vec<_> = env::args().collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();

    match args.as_slice()[1..] {
        ["rustc-pull"] => sync.rustc_pull(None),
        ["rustc-push", github_user, branch] => sync.rustc_push(github_user, Some(branch)),
        ["rustc-push", github_user] => sync.rustc_push(github_user, None),
        ["start-josh"] => {
            let _josh = Josh::start();
            println!("press enter to stop");
            io::stdout().flush().unwrap();
            let _ = io::stdin().read(&mut [0u8]).unwrap();
        }
        _ => {
            println!("{USAGE}");
            exit(1);
        }
    }
}
