mod licenses;
mod path_tree;
mod reuse;

use crate::licenses::LicensesInterner;
use anyhow::Error;
use std::path::PathBuf;

fn main() -> Result<(), Error> {
    let reuse_exe: PathBuf = std::env::var_os("REUSE_EXE").expect("Missing REUSE_EXE").into();
    let dest: PathBuf = std::env::var_os("DEST").expect("Missing DEST").into();

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
