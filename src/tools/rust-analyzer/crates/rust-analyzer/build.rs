//! Construct version in the `commit-hash date channel` format

use std::{env, path::PathBuf, process::Command};

fn main() {
    set_rerun();
    set_commit_info();
    println!("cargo::rustc-check-cfg=cfg(rust_analyzer)");
    if option_env!("CFG_RELEASE").is_none() {
        println!("cargo:rustc-env=POKE_RA_DEVS=1");
    }
}

fn set_rerun() {
    println!("cargo:rerun-if-env-changed=CFG_RELEASE");

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

fn set_commit_info() {
    #[allow(clippy::disallowed_methods)]
    let output = match Command::new("git")
        .arg("log")
        .arg("-1")
        .arg("--date=short")
        .arg("--format=%H %h %cd")
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => return,
    };
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut parts = stdout.split_whitespace();
    let mut next = || parts.next().unwrap();
    println!("cargo:rustc-env=RA_COMMIT_HASH={}", next());
    println!("cargo:rustc-env=RA_COMMIT_SHORT_HASH={}", next());
    println!("cargo:rustc-env=RA_COMMIT_DATE={}", next())
}
