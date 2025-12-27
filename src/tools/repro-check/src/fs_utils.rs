use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::{Walk, WalkBuilder};
use log::{info, warn};

// Sets up two build envs by copying source (if not skipped), cleaning first if asked.
// The path_delta is for testing path sensitivity - adds extra dir levels to B.
pub fn prepare_workspace(
    workspace: &Path,
    src_root: &Path,
    path_delta: usize,
    skip_copy: bool,
) -> Result<(PathBuf, PathBuf)> {
    if !skip_copy {
        info!("Setting up fresh workspace at {:?}", workspace);
        clean_workspace(workspace)?;
        fs::create_dir_all(workspace)?;
    }

    let env_a = workspace.join("build-a");
    let mut env_b = workspace.join("build-b");

    if path_delta > 0 {
        for i in 1..=path_delta {
            env_b = env_b.join(format!("extra{}", i));
        }
    }

    if cfg!(windows) && path_delta > 10 {
        warn!("Watch out - long paths on Windows might need registry tweaks.");
    }

    if !skip_copy {
        info!("Copying sources to A: {:?}", env_a);
        copy_source_tree(src_root, &env_a)?;

        info!("Copying sources to B: {:?}", env_b);
        copy_source_tree(src_root, &env_b)?;
    }

    Ok((env_a, env_b))
}

pub fn clean_workspace(workspace: &Path) -> Result<()> {
    if workspace.exists() {
        info!("Cleaning up old workspace: {:?}", workspace);
        fs::remove_dir_all(workspace).context("Workspace clean failed")?;
    }
    Ok(())
}

// Copies the source tree, respecting .gitignore and skipping build/.git.
// Uses ignore crate for git-like filtering.
fn copy_source_tree(src: &Path, dest: &Path) -> Result<()> {
    let walker: Walk = WalkBuilder::new(src)
        .hidden(false)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(true)
        .require_git(false)
        .build();

    for entry_res in walker {
        let entry = entry_res?;
        let from_path = entry.path();

        if from_path == src {
            continue;
        }

        if entry.file_type().map_or(false, |ft| ft.is_symlink()) {
            continue;
        }

        let rel_path = from_path.strip_prefix(src).context("Bad strip prefix")?;

        if rel_path.starts_with(".git") || rel_path.starts_with("build") {
            continue;
        }

        let to_path = dest.join(rel_path);

        if let Some(ft) = entry.file_type() {
            if ft.is_dir() {
                fs::create_dir_all(&to_path)?;
            } else if ft.is_file() {
                if let Some(parent) = to_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(from_path, &to_path).context(format!("Copy failed: {:?}", from_path))?;
            }
        }
    }

    Ok(())
}
