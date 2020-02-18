//! Just embed git-hash to `--version`

use std::process::Command;

fn main() {
    let rev = rev().unwrap_or_else(|| "???????".to_string());
    println!("cargo:rustc-env=REV={}", rev)
}

fn rev() -> Option<String> {
    let output = Command::new("git").args(&["rev-parse", "HEAD"]).output().ok()?;
    let stdout = String::from_utf8(output.stdout).ok()?;
    let short_hash = stdout.get(0..7)?;
    Some(short_hash.to_owned())
}
