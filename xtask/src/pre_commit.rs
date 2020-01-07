//! pre-commit hook for code formatting.

use std::{fs, path::PathBuf};

use anyhow::{bail, Result};

use crate::{cmd::run_with_output, project_root, run, run_rustfmt, Mode};

// FIXME: if there are changed `.ts` files, also reformat TypeScript (by
// shelling out to `npm fmt`).
pub fn run_hook() -> Result<()> {
    run_rustfmt(Mode::Overwrite)?;

    let diff = run_with_output("git diff --diff-filter=MAR --name-only --cached", ".")?;

    let root = project_root();
    for line in String::from_utf8(diff.stdout)?.lines() {
        run(&format!("git update-index --add {}", root.join(line).to_string_lossy()), ".")?;
    }

    Ok(())
}

pub fn install_hook() -> Result<()> {
    let hook_path: PathBuf =
        format!("./.git/hooks/pre-commit{}", std::env::consts::EXE_SUFFIX).into();

    if hook_path.exists() {
        bail!("Git hook already created");
    }

    let me = std::env::current_exe()?;
    fs::copy(me, hook_path)?;

    Ok(())
}
