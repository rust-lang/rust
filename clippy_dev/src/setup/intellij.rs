use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};

// This module takes an absolute path to a rustc repo and alters the dependencies to point towards
// the respective rustc subcrates instead of using extern crate xyz.
// This allows IntelliJ to analyze rustc internals and show proper information inside Clippy
// code. See https://github.com/rust-lang/rust-clippy/issues/5514 for details

const RUSTC_PATH_SECTION: &str = "[target.'cfg(NOT_A_PLATFORM)'.dependencies]";
const DEPENDENCIES_SECTION: &str = "[dependencies]";

const CLIPPY_PROJECTS: &[ClippyProjectInfo] = &[
    ClippyProjectInfo::new("root", "Cargo.toml", "src/driver.rs"),
    ClippyProjectInfo::new("clippy_lints", "clippy_lints/Cargo.toml", "clippy_lints/src/lib.rs"),
    ClippyProjectInfo::new("clippy_utils", "clippy_utils/Cargo.toml", "clippy_utils/src/lib.rs"),
];

/// Used to store clippy project information to later inject the dependency into.
struct ClippyProjectInfo {
    /// Only used to display information to the user
    name: &'static str,
    cargo_file: &'static str,
    lib_rs_file: &'static str,
}

impl ClippyProjectInfo {
    const fn new(name: &'static str, cargo_file: &'static str, lib_rs_file: &'static str) -> Self {
        Self {
            name,
            cargo_file,
            lib_rs_file,
        }
    }
}

pub fn setup_rustc_src(rustc_path: &str) {
    let Ok(rustc_source_dir) = check_and_get_rustc_dir(rustc_path) else {
        return;
    };

    for project in CLIPPY_PROJECTS {
        if inject_deps_into_project(&rustc_source_dir, project).is_err() {
            return;
        }
    }

    println!("info: the source paths can be removed again with `cargo dev remove intellij`");
}

fn check_and_get_rustc_dir(rustc_path: &str) -> Result<PathBuf, ()> {
    let mut path = PathBuf::from(rustc_path);

    if path.is_relative() {
        match path.canonicalize() {
            Ok(absolute_path) => {
                println!("info: the rustc path was resolved to: `{}`", absolute_path.display());
                path = absolute_path;
            },
            Err(err) => {
                eprintln!("error: unable to get the absolute path of rustc ({err})");
                return Err(());
            },
        };
    }

    let path = path.join("compiler");
    println!("info: looking for compiler sources at: {}", path.display());

    if !path.exists() {
        eprintln!("error: the given path does not exist");
        return Err(());
    }

    if !path.is_dir() {
        eprintln!("error: the given path is not a directory");
        return Err(());
    }

    Ok(path)
}

fn inject_deps_into_project(rustc_source_dir: &Path, project: &ClippyProjectInfo) -> Result<(), ()> {
    let cargo_content = read_project_file(project.cargo_file)?;
    let lib_content = read_project_file(project.lib_rs_file)?;

    if inject_deps_into_manifest(rustc_source_dir, project.cargo_file, &cargo_content, &lib_content).is_err() {
        eprintln!(
            "error: unable to inject dependencies into {} with the Cargo file {}",
            project.name, project.cargo_file
        );
        Err(())
    } else {
        Ok(())
    }
}

/// `clippy_dev` expects to be executed in the root directory of Clippy. This function
/// loads the given file or returns an error. Having it in this extra function ensures
/// that the error message looks nice.
fn read_project_file(file_path: &str) -> Result<String, ()> {
    let path = Path::new(file_path);
    if !path.exists() {
        eprintln!("error: unable to find the file `{file_path}`");
        return Err(());
    }

    match fs::read_to_string(path) {
        Ok(content) => Ok(content),
        Err(err) => {
            eprintln!("error: the file `{file_path}` could not be read ({err})");
            Err(())
        },
    }
}

fn inject_deps_into_manifest(
    rustc_source_dir: &Path,
    manifest_path: &str,
    cargo_toml: &str,
    lib_rs: &str,
) -> std::io::Result<()> {
    // do not inject deps if we have already done so
    if cargo_toml.contains(RUSTC_PATH_SECTION) {
        eprintln!("warn: dependencies are already setup inside {manifest_path}, skipping file");
        return Ok(());
    }

    let extern_crates = lib_rs
        .lines()
        // only take dependencies starting with `rustc_`
        .filter(|line| line.starts_with("extern crate rustc_"))
        // we have something like "extern crate foo;", we only care about the "foo"
        // extern crate rustc_middle;
        //              ^^^^^^^^^^^^
        .map(|s| &s[13..(s.len() - 1)]);

    let new_deps = extern_crates.map(|dep| {
        // format the dependencies that are going to be put inside the Cargo.toml
        format!("{dep} = {{ path = \"{}/{dep}\" }}\n", rustc_source_dir.display())
    });

    // format a new [dependencies]-block with the new deps we need to inject
    let mut all_deps = String::from("[target.'cfg(NOT_A_PLATFORM)'.dependencies]\n");
    new_deps.for_each(|dep_line| {
        all_deps.push_str(&dep_line);
    });
    all_deps.push_str("\n[dependencies]\n");

    // replace "[dependencies]" with
    // [dependencies]
    // dep1 = { path = ... }
    // dep2 = { path = ... }
    // etc
    let new_manifest = cargo_toml.replacen("[dependencies]\n", &all_deps, 1);

    // println!("{new_manifest}");
    let mut file = File::create(manifest_path)?;
    file.write_all(new_manifest.as_bytes())?;

    println!("info: successfully setup dependencies inside {manifest_path}");

    Ok(())
}

pub fn remove_rustc_src() {
    for project in CLIPPY_PROJECTS {
        remove_rustc_src_from_project(project);
    }
}

fn remove_rustc_src_from_project(project: &ClippyProjectInfo) -> bool {
    let Ok(mut cargo_content) = read_project_file(project.cargo_file) else {
        return false;
    };
    let Some(section_start) = cargo_content.find(RUSTC_PATH_SECTION) else {
        println!(
            "info: dependencies could not be found in `{}` for {}, skipping file",
            project.cargo_file, project.name
        );
        return true;
    };

    let Some(end_point) = cargo_content.find(DEPENDENCIES_SECTION) else {
        eprintln!(
            "error: the end of the rustc dependencies section could not be found in `{}`",
            project.cargo_file
        );
        return false;
    };

    cargo_content.replace_range(section_start..end_point, "");

    match File::create(project.cargo_file) {
        Ok(mut file) => {
            file.write_all(cargo_content.as_bytes()).unwrap();
            println!("info: successfully removed dependencies inside {}", project.cargo_file);
            true
        },
        Err(err) => {
            eprintln!(
                "error: unable to open file `{}` to remove rustc dependencies for {} ({err})",
                project.cargo_file, project.name
            );
            false
        },
    }
}
