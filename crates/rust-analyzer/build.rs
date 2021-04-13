//! Just embed git-hash to `--version`

use std::{env, path::PathBuf, process::Command};

fn main() {
    set_rerun();

    let rev =
        env::var("RUST_ANALYZER_REV").ok().or_else(rev).unwrap_or_else(|| "???????".to_string());
    println!("cargo:rustc-env=REV={}", rev)
}

fn set_rerun() {
    println!("cargo:rerun-if-env-changed=RUST_ANALYZER_REV");

    let mut manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("`CARGO_MANIFEST_DIR` is always set by cargo."),
    );

    while manifest_dir.parent().is_some() {
        if manifest_dir.join(".git/HEAD").exists() {
            let git_dir = manifest_dir.join(".git");

            println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
            // current branch ref
            if let Ok(output) =
                Command::new("git").args(&["rev-parse", "--symbolic-full-name", "HEAD"]).output()
            {
                if let Ok(ref_link) = String::from_utf8(output.stdout) {
                    println!("cargo:rerun-if-changed={}", git_dir.join(ref_link).display());
                }
            }
            return;
        }

        manifest_dir.pop();
    }
    println!("cargo:warning=Could not find `.git/HEAD` from manifest dir!");
}

fn rev() -> Option<String> {
    let output =
        Command::new("git").args(&["describe", "--tags", "--exclude", "nightly"]).output().ok()?;
    let stdout = String::from_utf8(output.stdout).ok()?;
    Some(stdout)
}
