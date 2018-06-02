#![feature(box_syntax)]
#![feature(rustc_private)]

extern crate cargo;
extern crate crates_io;
extern crate env_logger;
extern crate getopts;
#[macro_use]
extern crate log;

use crates_io::{Crate, Registry};

use cargo::exit_with_error;
use cargo::core::{Package, PackageId, Source, SourceId, Workspace};
use cargo::ops::{compile, CompileMode, CompileOptions};
use cargo::util::{CargoError, CargoResult, CliError};
use cargo::util::config::Config;
use cargo::util::important_paths::find_root_manifest_for_wd;

use getopts::{Matches, Options};

use std::env;
use std::error;
use std::fmt;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Stdio, Command};

/// Very simple error representation.
#[derive(Debug)]
struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.0)
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        &self.0
    }
}

/// Given a crate name, try to locate the corresponding crate on `crates.io`.
///
/// If no crate with the exact name is present, error out.
fn exact_search(query: &str) -> CargoResult<Crate> {
    let mut registry = Registry::new("https://crates.io".to_owned(), None);

    registry
        .search(query, 1)
        .map_err(|e|
                 Error(format!("failed to retrieve search results from the registry: {}", e))
                     .into())
        .and_then(|(mut crates, _)| {
            crates
                .drain(..)
                .find(|krate| krate.name == query)
                .ok_or_else(|| Error(format!("failed to find a matching crate `{}`", query))
                                .into())
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
    /// The source to be used.
    source: Box<Source + 'a>,
}

impl<'a> SourceInfo<'a> {
    /// Construct a new source info for `crates.io`.
    fn new(config: &'a Config) -> CargoResult<SourceInfo<'a>> {
        let source_id = SourceId::crates_io(config)?;
        let source = source_id.load(config)?;

        debug!("source id loaded: {:?}", source_id);

        Ok(SourceInfo {
            id: source_id,
            source: source,
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
    fn local(config: &'a Config, explicit_path: Option<PathBuf>) -> CargoResult<WorkInfo<'a>> {
        let manifest_path = if let Some(path) = explicit_path {
            find_root_manifest_for_wd(&path)?
        } else {
            find_root_manifest_for_wd(config.cwd())?
        };

        let workspace = Workspace::new(&manifest_path, config)?;
        let package = workspace.load(&manifest_path)?;

        Ok(WorkInfo {
            package,
            workspace,
        })
    }

    /// Construct a package/workspace pair by fetching the package of a specified name and
    /// version.
    fn remote(config: &'a Config, source: &mut SourceInfo<'a>, info: &NameAndVersion)
        -> CargoResult<WorkInfo<'a>>
    {
        // TODO: fall back to locally cached package instance, or better yet, search for it
        // first.
        let package_id = PackageId::new(info.name, info.version, &source.id)?;
        debug!("(remote) package id: {:?}", package_id);
        let package = source.source.download(&package_id)?;
        let workspace = Workspace::ephemeral(package.clone(), config, None, false)?;

        Ok(WorkInfo {
            package: package,
            workspace: workspace,
        })
    }

    /// Obtain the paths to the produced rlib and the dependency output directory.
    fn rlib_and_dep_output(&self, config: &'a Config, name: &str, current: bool)
        -> CargoResult<(PathBuf, PathBuf)>
    {
        let opts = CompileOptions::default(config, CompileMode::Build);

        if current {
            env::set_var("RUSTFLAGS", "-C metadata=new");
        } else {
            env::set_var("RUSTFLAGS", "-C metadata=old");
        }

        let compilation = compile(&self.workspace, &opts)?;

        env::remove_var("RUSTFLAGS");

        let rlib = compilation.libraries[self.package.package_id()]
            .iter()
            .find(|t| t.0.name() == name)
            .ok_or_else(|| Error("lost a build artifact".to_owned()))?;

        Ok((rlib.1.clone(), compilation.deps_output))
    }

}

/// Perform the heavy lifting.
///
/// Obtain the two versions of the crate to be analyzed as specified by command line arguments
/// and/or defaults, and dispatch the actual analysis.
// TODO: possibly reduce the complexity by finding where some info can be taken from directly
fn do_main(config: &Config, matches: &Matches, explain: bool) -> CargoResult<()> {
    debug!("running cargo-semver");
    fn parse_arg(opt: &str) -> CargoResult<NameAndVersion> {
        let mut split = opt.split(':');
        let name = if let Some(n) = split.next() {
            n
        } else {
            return Err(Error("spec has to be of form `name:version`".to_owned()).into());
        };
        let version = if let Some(v) = split.next() {
            v
        } else {
            return Err(Error("spec has to be of form `name:version`".to_owned()).into());
        };

        if split.next().is_some() {
            return Err(Error("spec has to be of form `name:version`".to_owned()).into());
        }

        Ok(NameAndVersion { name: name, version: version })
    }

    let mut source = SourceInfo::new(config)?;

    let current = if let Some(opt) = matches.opt_str("C") {
        WorkInfo::remote(config, &mut source, &parse_arg(&opt)?)?
    } else {
        WorkInfo::local(config, matches.opt_str("c").map(PathBuf::from))?
    };

    let name = current.package.name().to_owned();

    let (stable, stable_version) = if let Some(opt) = matches.opt_str("S") {
        let info = parse_arg(&opt)?;
        let version = info.version.to_owned();

        let work_info = WorkInfo::remote(config, &mut source, &info)?;

        (work_info, version)
    } else if let Some(path) = matches.opt_str("s") {
        let work_info = WorkInfo::local(config, Some(PathBuf::from(path)))?;
        let version = format!("{}", work_info.package.version());
        (work_info, version)
    } else {
        let stable_crate = exact_search(&name)?;
        let info = NameAndVersion { name: &name, version: &stable_crate.max_version };
        let work_info = WorkInfo::remote(config, &mut source, &info)?;

        (work_info, stable_crate.max_version.clone())
    };

    let (current_rlib, current_deps_output) = current.rlib_and_dep_output(config, &name, true)?;
    let (stable_rlib, stable_deps_output) = stable.rlib_and_dep_output(config, &name, false)?;

    if matches.opt_present("d") {
        println!("--extern old={} -L{} --extern new={} -L{}",
                 stable_rlib.display(),
                 stable_deps_output.display(),
                 current_rlib.display(),
                 current_deps_output.display());
        return Ok(());
    }

    debug!("running rust-semverver on compiled crates");

    let mut child = Command::new("rust-semverver")
        .arg("--crate-type=lib")
        .args(&["--extern", &*format!("old={}", stable_rlib.display())])
        .args(&[format!("-L{}", stable_deps_output.display())])
        .args(&["--extern", &*format!("new={}", current_rlib.display())])
        .args(&[format!("-L{}", current_deps_output.display())])
        .arg("-")
        .stdin(Stdio::piped())
        .env("RUST_SEMVER_CRATE_VERSION", stable_version)
        .env("RUST_SEMVER_VERBOSE", format!("{}", explain))
        .spawn()
        .map_err(|e| Error(format!("could not spawn rustc: {}", e)))?;

    if let Some(ref mut stdin) = child.stdin {
        stdin.write_fmt(format_args!("#[allow(unused_extern_crates)] \
                                     extern crate new; \
                                     #[allow(unused_extern_crates)] \
                                     extern crate old;"))?;
    } else {
        return Err(Error("could not pipe to rustc (wtf?)".to_owned()).into());
    }

    child
        .wait()
        .map_err(|e| Error(format!("failed to wait for rustc: {}", e)))?;

    Ok(())
}

/// Print a help message.
fn help(opts: &Options) {
    let brief = "usage: cargo semver [options] [-- cargo options]";
    print!("{}", opts.usage(brief));
}

/// Print a version message.
fn version() {
    println!("{}", env!("CARGO_PKG_VERSION"));
}

/// Main entry point.
///
/// Parse CLI arguments, handle their semantics, and provide for proper error handling.
fn main() {
    fn err(config: &Config, e: CargoError) -> ! {
        exit_with_error(CliError::new(e, 1), &mut config.shell());
    }

    if env_logger::try_init().is_err() {
        eprintln!("ERROR: could not initialize logger");
    }

    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut opts = Options::new();

    opts.optflag("h", "help", "print this message and exit");
    opts.optflag("V", "version", "print version information and exit");
    opts.optflag("e", "explain", "print detailed error explanations");
    opts.optflag("d", "debug", "print command to debug and exit");
    opts.optopt("s", "stable-path", "use local path as stable/old crate", "PATH");
    opts.optopt("c", "current-path", "use local path as current/new crate", "PATH");
    opts.optopt("S", "stable-pkg", "use a `name:version` string as stable/old crate",
                "NAME:VERSION");
    opts.optopt("C", "current-pkg", "use a `name:version` string as current/new crate",
                "NAME:VERSION");

    let config = match Config::default() {
        Ok(cfg) => cfg,
        Err(e) => panic!("can't obtain config: {:?}", e),
    };

    let matches = match opts.parse(&args) {
        Ok(m) => m,
        Err(f) => err(&config, f.to_owned().into()),
    };

    if matches.opt_present("h") {
        help(&opts);
        return;
    }

    if matches.opt_present("V") {
        version();
        return;
    }

    if (matches.opt_present("s") && matches.opt_present("S")) ||
        matches.opt_count("s") > 1 || matches.opt_count("S") > 1
    {
        let msg = "at most one of `-s,--stable-path` and `-S,--stable-pkg` allowed";
        err(&config, Error(msg.to_owned()).into());
    }

    if (matches.opt_present("c") && matches.opt_present("C")) ||
        matches.opt_count("c") > 1 || matches.opt_count("C") > 1
    {
        let msg = "at most one of `-c,--current-path` and `-C,--current-pkg` allowed";
        err(&config, Error(msg.to_owned()).into());
    }

    if let Err(e) = do_main(&config, &matches, matches.opt_present("e")) {
        err(&config, e);
    }
}
