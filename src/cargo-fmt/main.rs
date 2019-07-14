// Inspired by Paul Woolcock's cargo-fmt (https://github.com/pwoolcoc/cargo-fmt/).

#![deny(warnings)]

use cargo_metadata;

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, Write};
use std::iter::FromIterator;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str;

use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(
    bin_name = "cargo fmt",
    author = "",
    about = "This utility formats all bin and lib files of \
             the current crate using rustfmt."
)]
pub struct Opts {
    /// No output printed to stdout
    #[structopt(short = "q", long = "quiet")]
    quiet: bool,

    /// Use verbose output
    #[structopt(short = "v", long = "verbose")]
    verbose: bool,

    /// Print rustfmt version and exit
    #[structopt(long = "version")]
    version: bool,

    /// Specify package to format (only usable in workspaces)
    #[structopt(short = "p", long = "package", value_name = "package")]
    packages: Vec<String>,

    /// Specify path to Cargo.toml
    #[structopt(long = "manifest-path", value_name = "manifest-path")]
    manifest_path: Option<String>,

    /// Options passed to rustfmt
    // 'raw = true' to make `--` explicit.
    #[structopt(name = "rustfmt_options", raw(raw = "true"))]
    rustfmt_options: Vec<String>,

    /// Format all packages (only usable in workspaces)
    #[structopt(long = "all")]
    format_all: bool,
}

fn main() {
    let exit_status = execute();
    std::io::stdout().flush().unwrap();
    std::process::exit(exit_status);
}

const SUCCESS: i32 = 0;
const FAILURE: i32 = 1;

fn execute() -> i32 {
    // Drop extra `fmt` argument provided by `cargo`.
    let mut found_fmt = false;
    let args = env::args().filter(|x| {
        if found_fmt {
            true
        } else {
            found_fmt = x == "fmt";
            x != "fmt"
        }
    });

    let opts = Opts::from_iter(args);

    let verbosity = match (opts.verbose, opts.quiet) {
        (false, false) => Verbosity::Normal,
        (false, true) => Verbosity::Quiet,
        (true, false) => Verbosity::Verbose,
        (true, true) => {
            print_usage_to_stderr("quiet mode and verbose mode are not compatible");
            return FAILURE;
        }
    };

    if opts.version {
        return handle_command_status(get_rustfmt_info(&[String::from("--version")]));
    }
    if opts.rustfmt_options.iter().any(|s| {
        ["--print-config", "-h", "--help", "-V", "--version"].contains(&s.as_str())
            || s.starts_with("--help=")
            || s.starts_with("--print-config=")
    }) {
        return handle_command_status(get_rustfmt_info(&opts.rustfmt_options));
    }

    let strategy = CargoFmtStrategy::from_opts(&opts);

    if opts.manifest_path.is_some() {
        let specified_manifest_path = opts.manifest_path.unwrap();
        if !specified_manifest_path.ends_with("Cargo.toml") {
            print_usage_to_stderr("the manifest-path must be a path to a Cargo.toml file");
            return FAILURE;
        }
        let manifest_path = PathBuf::from(specified_manifest_path);
        handle_command_status(format_crate(
            verbosity,
            &strategy,
            opts.rustfmt_options,
            Some(&manifest_path),
        ))
    } else {
        handle_command_status(format_crate(
            verbosity,
            &strategy,
            opts.rustfmt_options,
            None,
        ))
    }
}

fn print_usage_to_stderr(reason: &str) {
    eprintln!("{}", reason);
    let app = Opts::clap();
    app.after_help("")
        .write_help(&mut io::stderr())
        .expect("failed to write to stderr");
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verbosity {
    Verbose,
    Normal,
    Quiet,
}

fn handle_command_status(status: Result<i32, io::Error>) -> i32 {
    match status {
        Err(e) => {
            print_usage_to_stderr(&e.to_string());
            FAILURE
        }
        Ok(status) => status,
    }
}

fn get_rustfmt_info(args: &[String]) -> Result<i32, io::Error> {
    let mut command = Command::new("rustfmt")
        .stdout(std::process::Stdio::inherit())
        .args(args)
        .spawn()
        .map_err(|e| match e.kind() {
            io::ErrorKind::NotFound => io::Error::new(
                io::ErrorKind::Other,
                "Could not run rustfmt, please make sure it is in your PATH.",
            ),
            _ => e,
        })?;
    let result = command.wait()?;
    if result.success() {
        Ok(SUCCESS)
    } else {
        Ok(result.code().unwrap_or(SUCCESS))
    }
}

fn format_crate(
    verbosity: Verbosity,
    strategy: &CargoFmtStrategy,
    rustfmt_args: Vec<String>,
    manifest_path: Option<&Path>,
) -> Result<i32, io::Error> {
    let targets = get_targets(strategy, manifest_path)?;

    // Currently only bin and lib files get formatted.
    run_rustfmt(&targets, &rustfmt_args, verbosity)
}

/// Target uses a `path` field for equality and hashing.
#[derive(Debug)]
pub struct Target {
    /// A path to the main source file of the target.
    path: PathBuf,
    /// A kind of target (e.g., lib, bin, example, ...).
    kind: String,
    /// Rust edition for this target.
    edition: String,
}

impl Target {
    pub fn from_target(target: &cargo_metadata::Target) -> Self {
        let path = PathBuf::from(&target.src_path);
        let canonicalized = fs::canonicalize(&path).unwrap_or(path);

        Target {
            path: canonicalized,
            kind: target.kind[0].clone(),
            edition: target.edition.clone(),
        }
    }
}

impl PartialEq for Target {
    fn eq(&self, other: &Target) -> bool {
        self.path == other.path
    }
}

impl PartialOrd for Target {
    fn partial_cmp(&self, other: &Target) -> Option<Ordering> {
        Some(self.path.cmp(&other.path))
    }
}

impl Ord for Target {
    fn cmp(&self, other: &Target) -> Ordering {
        self.path.cmp(&other.path)
    }
}

impl Eq for Target {}

impl Hash for Target {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.path.hash(state);
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum CargoFmtStrategy {
    /// Format every packages and dependencies.
    All,
    /// Format packages that are specified by the command line argument.
    Some(Vec<String>),
    /// Format the root packages only.
    Root,
}

impl CargoFmtStrategy {
    pub fn from_opts(opts: &Opts) -> CargoFmtStrategy {
        match (opts.format_all, opts.packages.is_empty()) {
            (false, true) => CargoFmtStrategy::Root,
            (true, _) => CargoFmtStrategy::All,
            (false, false) => CargoFmtStrategy::Some(opts.packages.clone()),
        }
    }
}

/// Based on the specified `CargoFmtStrategy`, returns a set of main source files.
fn get_targets(
    strategy: &CargoFmtStrategy,
    manifest_path: Option<&Path>,
) -> Result<BTreeSet<Target>, io::Error> {
    let mut targets = BTreeSet::new();

    match *strategy {
        CargoFmtStrategy::Root => get_targets_root_only(manifest_path, &mut targets)?,
        CargoFmtStrategy::All => {
            get_targets_recursive(manifest_path, &mut targets, &mut BTreeSet::new())?
        }
        CargoFmtStrategy::Some(ref hitlist) => {
            get_targets_with_hitlist(manifest_path, hitlist, &mut targets)?
        }
    }

    if targets.is_empty() {
        Err(io::Error::new(
            io::ErrorKind::Other,
            "Failed to find targets".to_owned(),
        ))
    } else {
        Ok(targets)
    }
}

fn get_targets_root_only(
    manifest_path: Option<&Path>,
    targets: &mut BTreeSet<Target>,
) -> Result<(), io::Error> {
    let metadata = get_cargo_metadata(manifest_path, false)?;
    let workspace_root_path = PathBuf::from(&metadata.workspace_root).canonicalize()?;
    let (in_workspace_root, current_dir_manifest) = if manifest_path.is_some() {
        let target_manifest = manifest_path.unwrap().canonicalize()?;
        (workspace_root_path == target_manifest, target_manifest)
    } else {
        let current_dir = env::current_dir()?.canonicalize()?;
        (
            workspace_root_path == current_dir,
            current_dir.join("Cargo.toml"),
        )
    };

    let package_targets = match metadata.packages.len() {
        1 => metadata.packages.into_iter().next().unwrap().targets,
        _ => metadata
            .packages
            .into_iter()
            .filter(|p| {
                in_workspace_root
                    || PathBuf::from(&p.manifest_path)
                        .canonicalize()
                        .unwrap_or_default()
                        == current_dir_manifest
            })
            .map(|p| p.targets)
            .flatten()
            .collect(),
    };

    for target in package_targets {
        targets.insert(Target::from_target(&target));
    }

    Ok(())
}

fn get_targets_recursive(
    manifest_path: Option<&Path>,
    mut targets: &mut BTreeSet<Target>,
    visited: &mut BTreeSet<String>,
) -> Result<(), io::Error> {
    let metadata = get_cargo_metadata(manifest_path, false)?;
    let metadata_with_deps = get_cargo_metadata(manifest_path, true)?;

    for package in metadata.packages {
        add_targets(&package.targets, &mut targets);

        // Look for local dependencies.
        for dependency in package.dependencies {
            if dependency.source.is_some() || visited.contains(&dependency.name) {
                continue;
            }

            let dependency_package = metadata_with_deps
                .packages
                .iter()
                .find(|p| p.name == dependency.name);
            let manifest_path = if dependency_package.is_some() {
                PathBuf::from(&dependency_package.unwrap().manifest_path)
            } else {
                let mut package_manifest_path = PathBuf::from(&package.manifest_path);
                package_manifest_path.pop();
                package_manifest_path.push(&dependency.name);
                package_manifest_path.push("Cargo.toml");
                package_manifest_path
            };

            if manifest_path.exists() {
                visited.insert(dependency.name);
                get_targets_recursive(Some(&manifest_path), &mut targets, visited)?;
            }
        }
    }

    Ok(())
}

fn get_targets_with_hitlist(
    manifest_path: Option<&Path>,
    hitlist: &[String],
    targets: &mut BTreeSet<Target>,
) -> Result<(), io::Error> {
    let metadata = get_cargo_metadata(manifest_path, false)?;

    let mut workspace_hitlist: BTreeSet<&String> = BTreeSet::from_iter(hitlist);

    for package in metadata.packages {
        if workspace_hitlist.remove(&package.name) {
            for target in package.targets {
                targets.insert(Target::from_target(&target));
            }
        }
    }

    if workspace_hitlist.is_empty() {
        Ok(())
    } else {
        let package = workspace_hitlist.iter().next().unwrap();
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("package `{}` is not a member of the workspace", package),
        ))
    }
}

fn add_targets(target_paths: &[cargo_metadata::Target], targets: &mut BTreeSet<Target>) {
    for target in target_paths {
        targets.insert(Target::from_target(target));
    }
}

fn run_rustfmt(
    targets: &BTreeSet<Target>,
    fmt_args: &[String],
    verbosity: Verbosity,
) -> Result<i32, io::Error> {
    let by_edition = targets
        .iter()
        .inspect(|t| {
            if verbosity == Verbosity::Verbose {
                println!("[{} ({})] {:?}", t.kind, t.edition, t.path)
            }
        })
        .fold(BTreeMap::new(), |mut h, t| {
            h.entry(&t.edition).or_insert_with(Vec::new).push(&t.path);
            h
        });

    let mut status = vec![];
    for (edition, files) in by_edition {
        let stdout = if verbosity == Verbosity::Quiet {
            std::process::Stdio::null()
        } else {
            std::process::Stdio::inherit()
        };

        if verbosity == Verbosity::Verbose {
            print!("rustfmt");
            print!(" --edition {}", edition);
            fmt_args.iter().for_each(|f| print!(" {}", f));
            files.iter().for_each(|f| print!(" {}", f.display()));
            println!();
        }

        let mut command = Command::new("rustfmt")
            .stdout(stdout)
            .args(files)
            .args(&["--edition", edition])
            .args(fmt_args)
            .spawn()
            .map_err(|e| match e.kind() {
                io::ErrorKind::NotFound => io::Error::new(
                    io::ErrorKind::Other,
                    "Could not run rustfmt, please make sure it is in your PATH.",
                ),
                _ => e,
            })?;

        status.push(command.wait()?);
    }

    Ok(status
        .iter()
        .filter_map(|s| if s.success() { None } else { s.code() })
        .next()
        .unwrap_or(SUCCESS))
}

fn get_cargo_metadata(
    manifest_path: Option<&Path>,
    include_deps: bool,
) -> Result<cargo_metadata::Metadata, io::Error> {
    let mut cmd = cargo_metadata::MetadataCommand::new();
    if !include_deps {
        cmd.no_deps();
    }
    if let Some(manifest_path) = manifest_path {
        cmd.manifest_path(manifest_path);
    }
    match cmd.exec() {
        Ok(metadata) => Ok(metadata),
        Err(error) => Err(io::Error::new(io::ErrorKind::Other, error.to_string())),
    }
}

#[cfg(test)]
mod cargo_fmt_tests {
    use super::*;

    #[test]
    fn default_options() {
        let empty: Vec<String> = vec![];
        let o = Opts::from_iter(&empty);
        assert_eq!(false, o.quiet);
        assert_eq!(false, o.verbose);
        assert_eq!(false, o.version);
        assert_eq!(empty, o.packages);
        assert_eq!(empty, o.rustfmt_options);
        assert_eq!(false, o.format_all);
    }

    #[test]
    fn good_options() {
        let o = Opts::from_iter(&[
            "test",
            "-q",
            "-p",
            "p1",
            "-p",
            "p2",
            "--",
            "--edition",
            "2018",
        ]);
        assert_eq!(true, o.quiet);
        assert_eq!(false, o.verbose);
        assert_eq!(false, o.version);
        assert_eq!(vec!["p1", "p2"], o.packages);
        assert_eq!(vec!["--edition", "2018"], o.rustfmt_options);
        assert_eq!(false, o.format_all);
    }

    #[test]
    fn unexpected_option() {
        assert!(
            Opts::clap()
                .get_matches_from_safe(&["test", "unexpected"])
                .is_err()
        );
    }

    #[test]
    fn unexpected_flag() {
        assert!(
            Opts::clap()
                .get_matches_from_safe(&["test", "--flag"])
                .is_err()
        );
    }

    #[test]
    fn mandatory_separator() {
        assert!(
            Opts::clap()
                .get_matches_from_safe(&["test", "--check"])
                .is_err()
        );
        assert!(
            !Opts::clap()
                .get_matches_from_safe(&["test", "--", "--check"])
                .is_err()
        );
    }

    #[test]
    fn multiple_packages_one_by_one() {
        let o = Opts::from_iter(&[
            "test",
            "-p",
            "package1",
            "--package",
            "package2",
            "-p",
            "package3",
        ]);
        assert_eq!(3, o.packages.len());
    }

    #[test]
    fn multiple_packages_grouped() {
        let o = Opts::from_iter(&[
            "test",
            "--package",
            "package1",
            "package2",
            "-p",
            "package3",
            "package4",
        ]);
        assert_eq!(4, o.packages.len());
    }

    #[test]
    fn empty_packages_1() {
        assert!(Opts::clap().get_matches_from_safe(&["test", "-p"]).is_err());
    }

    #[test]
    fn empty_packages_2() {
        assert!(
            Opts::clap()
                .get_matches_from_safe(&["test", "-p", "--", "--check"])
                .is_err()
        );
    }

    #[test]
    fn empty_packages_3() {
        assert!(
            Opts::clap()
                .get_matches_from_safe(&["test", "-p", "--verbose"])
                .is_err()
        );
    }

    #[test]
    fn empty_packages_4() {
        assert!(
            Opts::clap()
                .get_matches_from_safe(&["test", "-p", "--check"])
                .is_err()
        );
    }
}
