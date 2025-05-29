//! Tidy check to ensure that rustdoc templates didn't forget a `{# #}` to strip extra whitespace
//! characters.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use ignore::DirEntry;

use crate::walk::walk_no_read;

fn run_eslint(args: &[PathBuf], config_folder: PathBuf, bad: &mut bool) {
    let mut child = match Command::new("npx")
        .arg("eslint")
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

fn get_eslint_version_inner(global: bool) -> Option<String> {
    let mut command = Command::new("npm");
    command.arg("list").arg("--parseable").arg("--long").arg("--depth=0");
    if global {
        command.arg("--global");
    }
    let output = command.output().ok()?;
    let lines = String::from_utf8_lossy(&output.stdout);
    lines.lines().find_map(|l| l.split(':').nth(1)?.strip_prefix("eslint@")).map(|v| v.to_owned())
}

fn get_eslint_version() -> Option<String> {
    get_eslint_version_inner(false).or_else(|| get_eslint_version_inner(true))
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
    match get_eslint_version() {
        Some(version) => {
            if version != eslint_version {
                *bad = true;
                eprintln!(
                    "⚠️ Installed version of eslint (`{version}`) is different than the \
                     one used in the CI (`{eslint_version}`)",
                );
                eprintln!(
                    "You can install this version using `npm update eslint` or by using \
                     `npm install eslint@{eslint_version}`",
                );
                return;
            }
        }
        None => {
            eprintln!("`eslint` doesn't seem to be installed. Skipping tidy check for JS files.");
            eprintln!("You can install it using `npm install eslint@{eslint_version}`");
            return;
        }
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
    run_eslint(&files_to_check, librustdoc_path.join("html/static"), bad);

    run_eslint(&[tools_path.join("rustdoc-js/tester.js")], tools_path.join("rustdoc-js"), bad);
    run_eslint(&[tools_path.join("rustdoc-gui/tester.js")], tools_path.join("rustdoc-gui"), bad);
}
