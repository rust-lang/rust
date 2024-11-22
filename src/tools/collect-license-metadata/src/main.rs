mod licenses;
mod path_tree;
mod reuse;

use std::path::PathBuf;

use anyhow::Error;

use crate::licenses::LicensesInterner;

/// The entry point to the binary.
///
/// You should probably let `bootstrap` execute this program instead of running it directly.
///
/// Run `x.py run collect-license-metadata`
fn main() -> Result<(), Error> {
    let reuse_exe: PathBuf = std::env::var_os("REUSE_EXE").expect("Missing REUSE_EXE").into();
    let dest: PathBuf = std::env::var_os("DEST").expect("Missing DEST").into();

    if dest.exists() {
        println!("{} exists, skipping REUSE data collection", dest.display());
        return Ok(());
    }

    let mut interner = LicensesInterner::new();
    let paths = crate::reuse::collect(&reuse_exe, &mut interner)?;

    let mut tree = crate::path_tree::build(paths);
    tree.simplify();

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        &dest,
        &serde_json::to_vec_pretty(&serde_json::json!({
            "files": crate::path_tree::expand_interned_licenses(tree, &interner),
        }))?,
    )?;

    Ok(())
}
