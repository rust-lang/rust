//! Tidy check to ensure that rustdoc templates didn't forget a `{# #}` to strip extra whitespace
//! characters.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use ignore::DirEntry;

use crate::walk::walk_no_read;

fn run_eslint(eslint_version: &str, args: &[PathBuf], config_folder: PathBuf, bad: &mut bool) {
    let mut child = match Command::new("npx")
        .arg(format!("eslint@{eslint_version}"))
        .arg("-c")
        .arg(config_folder.join(".eslintrc.js"))
        .args(args)
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            *bad = true;
            eprintln!("failed to run eslint: {error:?}");
            return;
        }
    };
    match child.wait() {
        Ok(exit_status) => {
            if exit_status.success() {
                return;
            }
            eprintln!("eslint command failed");
        }
        Err(error) => eprintln!("eslint command failed: {error:?}"),
    }
    *bad = true;
}

fn has_npx() -> bool {
    Command::new("npx").arg("--version").output().is_ok_and(|output| output.status.success())
}

pub fn check(librustdoc_path: &Path, tools_path: &Path, src_path: &Path, bad: &mut bool) {
    let eslint_version_path =
        src_path.join("ci/docker/host-x86_64/mingw-check-tidy/eslint.version");
    let eslint_version = match std::fs::read_to_string(&eslint_version_path) {
        Ok(version) => version.trim().to_string(),
        Err(error) => {
            *bad = true;
            eprintln!("failed to read `{}`: {error:?}", eslint_version_path.display());
            return;
        }
    };
    if !has_npx() {
        eprintln!("`npx` doesn't seem to be installed. Skipping tidy check for JS files.");
        eprintln!("Installing `node` should also install `npm` and `npx` on your system.");
        return;
    }
    let mut files_to_check = Vec::new();
    walk_no_read(
        &[&librustdoc_path.join("html/static/js")],
        |path, is_dir| is_dir || !path.extension().is_some_and(|ext| ext == OsStr::new("js")),
        &mut |path: &DirEntry| {
            files_to_check.push(path.path().into());
        },
    );
    println!("Running eslint on rustdoc JS files");
    run_eslint(&eslint_version, &files_to_check, librustdoc_path.join("html/static"), bad);

    run_eslint(
        &eslint_version,
        &[tools_path.join("rustdoc-js/tester.js")],
        tools_path.join("rustdoc-js"),
        bad,
    );
    run_eslint(
        &eslint_version,
        &[tools_path.join("rustdoc-gui/tester.js")],
        tools_path.join("rustdoc-gui"),
        bad,
    );
}
