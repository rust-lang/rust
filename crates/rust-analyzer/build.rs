//! Construct version in the `commit-hash date chanel` format

use std::{env, path::PathBuf, process::Command};

fn main() {
    set_rerun();
    println!("cargo:rustc-env=REV={}", rev())
}

fn set_rerun() {
    println!("cargo:rerun-if-env-changed=RUST_ANALYZER_REV");

    let mut manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("`CARGO_MANIFEST_DIR` is always set by cargo."),
    );

    while manifest_dir.parent().is_some() {
        let head_ref = manifest_dir.join(".git/HEAD");
        if head_ref.exists() {
            println!("cargo:rerun-if-changed={}", head_ref.display());
            return;
        }

        manifest_dir.pop();
    }

    println!("cargo:warning=Could not find `.git/HEAD` from manifest dir!");
}

fn rev() -> String {
    if let Ok(rev) = env::var("RUST_ANALYZER_REV") {
        return rev;
    }

    if let Some(commit_hash) = commit_hash() {
        let mut buf = commit_hash;

        if let Some(date) = build_date() {
            buf.push(' ');
            buf.push_str(&date);
        }

        let channel = env::var("RUST_ANALYZER_CHANNEL").unwrap_or_else(|_| "dev".to_string());
        buf.push(' ');
        buf.push_str(&channel);

        return buf;
    }

    "???????".to_string()
}

fn commit_hash() -> Option<String> {
    output_to_string("git rev-parse --short HEAD")
}

fn build_date() -> Option<String> {
    output_to_string("date --iso --utc")
}

fn output_to_string(command: &str) -> Option<String> {
    let args = command.split_ascii_whitespace().collect::<Vec<_>>();
    let output = Command::new(args[0]).args(&args[1..]).output().ok()?;
    let stdout = String::from_utf8(output.stdout).ok()?;
    Some(stdout.trim().to_string())
}
