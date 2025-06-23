//! Tidy check to ensure that rustdoc templates didn't forget a `{# #}` to strip extra whitespace
//! characters.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use ignore::DirEntry;

use crate::walk::walk_no_read;

fn rustdoc_js_files(librustdoc_path: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    walk_no_read(
        &[&librustdoc_path.join("html/static/js")],
        |path, is_dir| is_dir || path.extension().is_none_or(|ext| ext != OsStr::new("js")),
        &mut |path: &DirEntry| {
            files.push(path.path().into());
        },
    );
    return files;
}

fn run_eslint(args: &[PathBuf], config_folder: PathBuf) -> Result<(), super::Error> {
    let mut child = Command::new("npx")
        .arg("eslint")
        .arg("-c")
        .arg(config_folder.join(".eslintrc.js"))
        .args(args)
        .spawn()?;
    match child.wait() {
        Ok(exit_status) => {
            if exit_status.success() {
                return Ok(());
            }
            Err(super::Error::FailedCheck("eslint command failed"))
        }
        Err(error) => Err(super::Error::Generic(format!("eslint command failed: {error:?}"))),
    }
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

pub(super) fn lint(
    librustdoc_path: &Path,
    tools_path: &Path,
    src_path: &Path,
) -> Result<(), super::Error> {
    let eslint_version_path = src_path.join("ci/docker/host-x86_64/tidy/eslint.version");
    let eslint_version = match std::fs::read_to_string(&eslint_version_path) {
        Ok(version) => version.trim().to_string(),
        Err(error) => {
            eprintln!("failed to read `{}`: {error:?}", eslint_version_path.display());
            return Err(error.into());
        }
    };
    // Having the correct `eslint` version installed via `npm` isn't strictly necessary, since we're invoking it via `npx`,
    // but this check allows the vast majority that is not working on the rustdoc frontend to avoid the penalty of running
    // `eslint` in tidy. See also: https://github.com/rust-lang/rust/pull/142851
    match get_eslint_version() {
        Some(version) => {
            if version != eslint_version {
                // unfortunatly we can't use `Error::Version` here becuse `str::trim` isn't const and
                // Version::required must be a static str
                return Err(super::Error::Generic(format!(
                    "⚠️ Installed version of eslint (`{version}`) is different than the \
                     one used in the CI (`{eslint_version}`)\n\
                    You can install this version using `npm update eslint` or by using \
                    `npm install eslint@{eslint_version}`\n
"
                )));
            }
        }
        None => {
            //eprintln!("`eslint` doesn't seem to be installed. Skipping tidy check for JS files.");
            //eprintln!("You can install it using `npm install eslint@{eslint_version}`");
            return Err(super::Error::MissingReq(
                "eslint",
                "js lint checks",
                Some(format!("You can install it using `npm install eslint@{eslint_version}`")),
            ));
        }
    }
    let files_to_check = rustdoc_js_files(librustdoc_path);
    println!("Running eslint on rustdoc JS files");
    run_eslint(&files_to_check, librustdoc_path.join("html/static"))?;

    run_eslint(&[tools_path.join("rustdoc-js/tester.js")], tools_path.join("rustdoc-js"))?;
    run_eslint(&[tools_path.join("rustdoc-gui/tester.js")], tools_path.join("rustdoc-gui"))?;
    Ok(())
}

pub(super) fn typecheck(librustdoc_path: &Path) -> Result<(), super::Error> {
    // use npx to ensure correct version
    let mut child = Command::new("npx")
        .arg("tsc")
        .arg("-p")
        .arg(librustdoc_path.join("html/static/js/tsconfig.json"))
        .spawn()?;
    match child.wait() {
        Ok(exit_status) => {
            if exit_status.success() {
                return Ok(());
            }
            Err(super::Error::FailedCheck("tsc command failed"))
        }
        Err(error) => Err(super::Error::Generic(format!("tsc command failed: {error:?}"))),
    }
}

pub(super) fn es_check(librustdoc_path: &Path) -> Result<(), super::Error> {
    let files_to_check = rustdoc_js_files(librustdoc_path);
    // use npx to ensure correct version
    let mut cmd = Command::new("npx");
    cmd.arg("es-check").arg("es2019");
    for f in files_to_check {
        cmd.arg(f);
    }
    match cmd.spawn()?.wait() {
        Ok(exit_status) => {
            if exit_status.success() {
                return Ok(());
            }
            Err(super::Error::FailedCheck("es-check command failed"))
        }
        Err(error) => Err(super::Error::Generic(format!("es-check command failed: {error:?}"))),
    }
}
