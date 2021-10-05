//! Construct version in the `commit-hash date chanel` format

use std::{env, path::PathBuf, process::Command};

fn main() {
    set_rerun();
    println!("cargo:rustc-env=REV={}", rev());
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
    exec("git rev-parse --short HEAD").ok()
}

fn build_date() -> Option<String> {
    exec("date -u +%Y-%m-%d").ok()
}

fn exec(command: &str) -> std::io::Result<String> {
    let args = command.split_ascii_whitespace().collect::<Vec<_>>();
    let output = Command::new(args[0]).args(&args[1..]).output()?;
    if !output.status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("command {:?} returned non-zero code", command,),
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
    Ok(stdout.trim().to_string())
}
