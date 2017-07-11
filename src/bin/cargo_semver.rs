#![feature(box_syntax)]

extern crate cargo;
extern crate crates_io;

use crates_io::{Crate, Registry};

use cargo::exit_with_error;
use cargo::core::{Package, PackageId, Source, SourceId, Workspace};
use cargo::ops::{compile, CompileMode, CompileOptions};
use cargo::sources::registry::RegistrySource;
use cargo::util::{human, CargoResult, CliError};
use cargo::util::config::Config;
use cargo::util::important_paths::find_root_manifest_for_wd;

use std::io::Write;
use std::path::PathBuf;
use std::process::{Stdio, Command};

/// Given a crate name, try to locate the corresponding crate on `crates.io`.
///
/// If no crate with the exact name is present, error out.
fn exact_search(query: &str) -> CargoResult<Crate> {
    // TODO: maybe we can get this with less constants :)
    let mut registry = Registry::new("https://crates.io".to_owned(), None);

    registry
        .search(query, 1)
        .map_err(|e|
                 human(format!("failed to retrieve search results from the registry: {}", e)))
        .and_then(|(mut crates, _)| {
            crates
                .drain(..)
                .find(|krate| krate.name == query)
                .ok_or_else(|| human(format!("failed to find a matching crate `{}`", query)))
        })
}

/// A specification of package name and version.
struct NameAndVersion<'a> {
    /// The package's name.
    name: &'a str,
    /// The package's version, as a semver-string.
    version: &'a str,
}

/// A specification of a package source to fetch remote packages from.
struct SourceInfo<'a> {
    /// The source id to be used.
    id: SourceId,
    /// The registry to be used.
    registry: RegistrySource<'a>,
}

impl<'a> SourceInfo<'a> {
    /// Construct a new source info for `crates.io`.
    fn new(config: &'a Config) -> CargoResult<SourceInfo<'a>> {
        let source_id = SourceId::crates_io(config)?;
        let registry = RegistrySource::remote(&source_id, config);
        Ok(SourceInfo {
            id: source_id,
            registry: registry,
        })
    }
}

/// A specification of a package and it's workspace.
struct WorkInfo<'a> {
    /// The package to be compiled.
    package: Package,
    /// The package's workspace.
    workspace: Workspace<'a>,
}

impl<'a> WorkInfo<'a> {
    /// Construct a package/workspace pair for the local directory.
    fn local(config: &'a Config) -> CargoResult<WorkInfo<'a>> {
        let manifest_path = find_root_manifest_for_wd(None, config.cwd())?;

        Ok(WorkInfo {
            package: Package::for_path(&manifest_path, config)?,
            workspace: Workspace::new(&manifest_path, config)?,
        })
    }

    /// Construct a package/workspace pair by fetching the package of a specified name and
    /// version.
    fn remote(config: &'a Config, source: &mut SourceInfo<'a>, info: NameAndVersion)
        -> CargoResult<WorkInfo<'a>>
    {
        let package_id = PackageId::new(info.name, info.version, &source.id)?;
        let package = source.registry.download(&package_id)?;
        let workspace = Workspace::ephemeral(package.clone(), config, None, false)?;

        Ok(WorkInfo {
            package: package,
            workspace: workspace,
        })
    }

    /// Obtain the paths to the rlib produced and to the output directory for dependencies.
    fn rlib_and_dep_output(&self, config: &'a Config, name: &str)
        -> CargoResult<(PathBuf, PathBuf)>
    {
        let opts = CompileOptions::default(config, CompileMode::Build);
        let compilation = compile(&self.workspace, &opts)?;
        let rlib = compilation.libraries[self.package.package_id()]
            .iter()
            .find(|t| t.0.name() == name)
            .ok_or_else(|| human("lost a build artifact"))?;

        Ok((rlib.1.clone(), compilation.deps_output))
    }

}

/// Perform the heavy lifting.
///
/// Obtain the local crate and compile it, then fetch the latest version from the registry, and
/// build it as well.
///
/// TODO:
/// * possibly reduce the complexity by investigating where some of the info can be sourced from
/// in a more direct fashion
/// * add proper support to compare two arbitrary versions
fn do_main() -> CargoResult<()> {
    use std::env::var;
    let config = Config::default()?;
    let mut source = SourceInfo::new(&config)?;

    let current =
        if let (Ok(n), Ok(v)) = (var("RUST_SEMVER_NAME"), var("RUST_SEMVER_CURRENT")) {
            let info = NameAndVersion { name: &n, version: &v };
            WorkInfo::remote(&config, &mut source, info)?
        } else {
            WorkInfo::local(&config)?
        };

    let name = current.package.name().to_owned();

    let (stable, stable_version) = if let Ok(v) = var("RUST_SEMVER_STABLE") {
        let info = NameAndVersion { name: &name, version: &v };
        let work_info = WorkInfo::remote(&config, &mut source, info)?;
        (work_info, v.clone())
    } else {
        let stable_crate = exact_search(&name)?;
        let info = NameAndVersion { name: &name, version: &stable_crate.max_version };
        let work_info = WorkInfo::remote(&config, &mut source, info)?;
        (work_info, stable_crate.max_version.clone())
    };

    let (current_rlib, current_deps_output) = current.rlib_and_dep_output(&config, &name)?;
    let (stable_rlib, stable_deps_output) = stable.rlib_and_dep_output(&config, &name)?;

    let mut child = Command::new("rust-semverver")
        .arg("--crate-type=lib")
        .args(&["--extern", &*format!("old={}", stable_rlib.display())])
        .args(&[format!("-L{}", stable_deps_output.display())])
        .args(&["--extern", &*format!("new={}", current_rlib.display())])
        .args(&[format!("-L{}", current_deps_output.display())])
        .arg("-")
        .stdin(Stdio::piped())
        .env("RUST_SEMVER_CRATE_VERSION", stable_version)
        .spawn()
        .map_err(|e| human(format!("could not spawn rustc: {}", e)))?;

    if let Some(ref mut stdin) = child.stdin {
        stdin
            .write_fmt(format_args!("extern crate new; extern crate old;"))?;
    } else {
        return Err(human("could not pipe to rustc (wtf?)"));
    }

    child
        .wait()
        .map_err(|e| human(format!("failed to wait for rustc: {}", e)))?;

    Ok(())
}

const CARGO_SEMVER_HELP: &str = r#"Checks a package's SemVer compatibility with already published versions.

Usage:
    cargo semver [options]

Common options:
    -h, --help               Print this message
    -V, --version            Print version info and exit

Currently, no other options are supported (this will change in the future)
"#;

fn help() {
    println!("{}", CARGO_SEMVER_HELP);
}

fn version() {
    println!("{}", env!("CARGO_PKG_VERSION"));
}

fn main() {
    if std::env::args().any(|arg| arg == "-h" || arg == "--help") {
        help();
        return;
    }

    if std::env::args().any(|arg| arg == "-V" || arg == "--version") {
        version();
        return;
    }

    if let Err(e) = do_main() {
        if let Ok(config) = Config::default() {
            exit_with_error(CliError::new(e, 1), &mut config.shell());
        } else {
            panic!("ffs, we can't get a config and errors happened :/");
        }
    }
}
