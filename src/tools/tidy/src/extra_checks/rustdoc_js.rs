//! Tidy check to ensure that rustdoc templates didn't forget a `{# #}` to strip extra whitespace
//! characters.

use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};

use build_helper::npm;
use ignore::DirEntry;

use crate::walk::walk_no_read;

fn node_module_bin(outdir: &Path, name: &str) -> PathBuf {
    outdir.join("node_modules/.bin").join(name)
}

fn spawn_cmd(cmd: &mut Command) -> Result<Child, io::Error> {
    cmd.spawn().map_err(|err| {
        eprintln!("unable to run {cmd:?} due to {err:?}");
        err
    })
}

/// install all js dependencies from package.json.
pub(super) fn npm_install(root_path: &Path, outdir: &Path, npm: &Path) -> Result<(), super::Error> {
    npm::install(root_path, outdir, npm)?;
    Ok(())
}

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

fn run_eslint(
    outdir: &Path,
    args: &[PathBuf],
    config_folder: PathBuf,
    bless: bool,
) -> Result<(), super::Error> {
    let mut cmd = Command::new(node_module_bin(outdir, "eslint"));
    if bless {
        cmd.arg("--fix");
    }
    cmd.arg("-c").arg(config_folder.join(".eslintrc.js")).args(args);
    let mut child = spawn_cmd(&mut cmd)?;
    match child.wait() {
        Ok(exit_status) => {
            if exit_status.success() {
                return Ok(());
            }
            Err(super::Error::FailedCheck("eslint"))
        }
        Err(error) => Err(super::Error::Generic(format!("eslint command failed: {error:?}"))),
    }
}

pub(super) fn lint(
    outdir: &Path,
    librustdoc_path: &Path,
    tools_path: &Path,
    bless: bool,
) -> Result<(), super::Error> {
    let files_to_check = rustdoc_js_files(librustdoc_path);
    println!("Running eslint on rustdoc JS files");
    run_eslint(outdir, &files_to_check, librustdoc_path.join("html/static"), bless)?;

    run_eslint(
        outdir,
        &[tools_path.join("rustdoc-js/tester.js")],
        tools_path.join("rustdoc-js"),
        bless,
    )?;
    Ok(())
}

pub(super) fn typecheck(outdir: &Path, librustdoc_path: &Path) -> Result<(), super::Error> {
    // use npx to ensure correct version
    let mut child = spawn_cmd(
        Command::new(node_module_bin(outdir, "tsc"))
            .arg("-p")
            .arg(librustdoc_path.join("html/static/js/tsconfig.json")),
    )?;
    match child.wait() {
        Ok(exit_status) => {
            if exit_status.success() {
                return Ok(());
            }
            Err(super::Error::FailedCheck("tsc"))
        }
        Err(error) => Err(super::Error::Generic(format!("tsc command failed: {error:?}"))),
    }
}

pub(super) fn es_check(outdir: &Path, librustdoc_path: &Path) -> Result<(), super::Error> {
    let files_to_check = rustdoc_js_files(librustdoc_path);
    let mut cmd = Command::new(node_module_bin(outdir, "es-check"));
    cmd.arg("es2019");
    for f in files_to_check {
        cmd.arg(f);
    }
    match spawn_cmd(&mut cmd)?.wait() {
        Ok(exit_status) => {
            if exit_status.success() {
                return Ok(());
            }
            Err(super::Error::FailedCheck("es-check"))
        }
        Err(error) => Err(super::Error::Generic(format!("es-check command failed: {error:?}"))),
    }
}
