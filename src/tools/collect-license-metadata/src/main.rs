mod licenses;
mod path_tree;
mod reuse;

use std::path::PathBuf;

use anyhow::{Context, Error};

use crate::licenses::LicensesInterner;

/// The entry point to the binary.
///
/// You should probably let `bootstrap` execute this program instead of running it directly.
///
/// * Run `x.py run collect-license-metadata` to re-regenerate the file.
/// * Run `x.py test collect-license-metadata` to check if the file you have is correct.
fn main() -> Result<(), Error> {
    let reuse_exe: PathBuf = std::env::var_os("REUSE_EXE").expect("Missing REUSE_EXE").into();
    let dest: PathBuf = std::env::var_os("DEST").expect("Missing DEST").into();
    let only_check = std::env::var_os("ONLY_CHECK").is_some();

    let mut interner = LicensesInterner::new();
    let paths = crate::reuse::collect(&reuse_exe, &mut interner)?;

    let mut tree = crate::path_tree::build(paths);
    tree.simplify();

    let output = serde_json::json!({
        "files": crate::path_tree::expand_interned_licenses(tree, &interner)
    });

    if only_check {
        println!("loading existing license information");
        let existing = std::fs::read_to_string(&dest).with_context(|| {
            format!("Failed to read existing license JSON at {}", dest.display())
        })?;
        let existing_json: serde_json::Value =
            serde_json::from_str(&existing).with_context(|| {
                format!("Failed to read existing license JSON at {}", dest.display())
            })?;
        if existing_json != output {
            eprintln!("The existing {} file is out of date.", dest.display());
            eprintln!("Run ./x run collect-license-metadata to update it.");
            anyhow::bail!("The existing {} file doesn't match what REUSE reports.", dest.display());
        }
        println!("license information matches");
    } else {
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&dest, &serde_json::to_vec_pretty(&output)?)?;
        println!("license information written to {}", dest.display());
    }

    Ok(())
}
