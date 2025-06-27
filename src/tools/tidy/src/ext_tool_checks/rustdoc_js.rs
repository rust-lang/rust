//! Tidy check to ensure that rustdoc templates didn't forget a `{# #}` to strip extra whitespace
//! characters.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::{fs, io};

use build_helper::ci::CiEnv;
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
pub(super) fn npm_install(root_path: &Path, outdir: &Path) -> Result<(), super::Error> {
    let copy_to_build = |p| {
        fs::copy(root_path.join(p), outdir.join(p)).map_err(|e| {
            eprintln!("unable to copy {p:?} to build directory: {e:?}");
            e
        })
    };
    // copy stuff to the output directory to make node_modules get put there.
    copy_to_build("package.json")?;
    copy_to_build("package-lock.json")?;

    let mut cmd = Command::new("npm");
    if CiEnv::is_ci() {
        // `npm ci` redownloads every time and thus is too slow for local development.
        cmd.arg("ci");
    } else {
        cmd.arg("install");
    }
    // disable a bunch of things we don't want.
    // this makes tidy output less noisy, and also significantly improves runtime
    // of repeated tidy invokations.
    cmd.args(&["--audit=false", "--save=false", "--fund=false"]);
    cmd.current_dir(outdir);
    let mut child = spawn_cmd(&mut cmd)?;
    match child.wait() {
        Ok(exit_status) => {
            if exit_status.success() {
                return Ok(());
            }
            Err(super::Error::FailedCheck("npm install"))
        }
        Err(error) => Err(super::Error::Generic(format!("npm install failed: {error:?}"))),
    }
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

fn run_eslint(outdir: &Path, args: &[PathBuf], config_folder: PathBuf) -> Result<(), super::Error> {
    let mut child = spawn_cmd(
        Command::new(node_module_bin(outdir, "eslint"))
            .arg("-c")
            .arg(config_folder.join(".eslintrc.js"))
            .args(args),
    )?;
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

pub(super) fn lint(
    outdir: &Path,
    librustdoc_path: &Path,
    tools_path: &Path,
) -> Result<(), super::Error> {
    let files_to_check = rustdoc_js_files(librustdoc_path);
    println!("Running eslint on rustdoc JS files");
    run_eslint(outdir, &files_to_check, librustdoc_path.join("html/static"))?;

    run_eslint(outdir, &[tools_path.join("rustdoc-js/tester.js")], tools_path.join("rustdoc-js"))?;
    run_eslint(
        outdir,
        &[tools_path.join("rustdoc-gui/tester.js")],
        tools_path.join("rustdoc-gui"),
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
            Err(super::Error::FailedCheck("tsc command failed"))
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
            Err(super::Error::FailedCheck("es-check command failed"))
        }
        Err(error) => Err(super::Error::Generic(format!("es-check command failed: {error:?}"))),
    }
}
