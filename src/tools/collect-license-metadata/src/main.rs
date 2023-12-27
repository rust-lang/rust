mod licenses;
mod path_tree;
mod reuse;

use crate::licenses::LicensesInterner;
use anyhow::Error;
use std::path::PathBuf;

// Some directories have too many slight license differences that'd result in a
// huge report, and could be considered a standalone project anyway. Those
// directories are "condensed" into a single licensing block for ease of
// reading, merging the licensing information.
//
// For every `(dir, file)``, every file in `dir` is considered to have the
// license info of `file`.
const CONDENSED_DIRECTORIES: &[(&str, &str)] =
    &[("./src/llvm-project/", "./src/llvm-project/README.md")];

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
