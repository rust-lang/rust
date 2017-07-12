#![feature(box_syntax)]
#![feature(rustc_private)]

extern crate cargo;
extern crate crates_io;
extern crate getopts;

use crates_io::{Crate, Registry};

use cargo::exit_with_error;
use cargo::core::{Package, PackageId, Source, SourceId, Workspace};
use cargo::ops::{compile, CompileMode, CompileOptions};
use cargo::sources::registry::RegistrySource;
use cargo::util::{human, CargoError, CargoResult, CliError};
use cargo::util::config::Config;
use cargo::util::important_paths::find_root_manifest_for_wd;

use getopts::{Matches, Options};

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
    fn local(config: &'a Config, explicit_path: Option<PathBuf>) -> CargoResult<WorkInfo<'a>> {
        let manifest_path = if let Some(path) = explicit_path {
            find_root_manifest_for_wd(None, &path)?
        } else {
            find_root_manifest_for_wd(None, config.cwd())?
        };

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
fn do_main(config: &Config, matches: &Matches) -> CargoResult<()> {
    fn parse_arg(opt: &str) -> CargoResult<(&str, &str)> {
        let mut split = opt.split('-');
        let name = if let Some(n) = split.next() {
            n
        } else {
            return Err(human("spec has to be of form `name-version`".to_owned()));
        };
        let version = if let Some(v) = split.next() {
            v
        } else {
            return Err(human("spec has to be of form `name-version`".to_owned()));
        };

        if split.next().is_some() {
            return Err(human("spec has to be of form `name-version`".to_owned()));
        }

        Ok((name, version))
    }

    let mut source = SourceInfo::new(config)?;

    let current = if let Some(opt) = matches.opt_str("C") {
        let (name, version) = parse_arg(&opt)?;

        let info = NameAndVersion { name: name, version: version };
        WorkInfo::remote(config, &mut source, info)?
    } else {
        WorkInfo::local(config, matches.opt_str("c").map(PathBuf::from))?
    };

    let name = current.package.name().to_owned();

    let (stable, stable_version) = if let Some(opt) = matches.opt_str("S") {
        let (name, version) = parse_arg(&opt)?;

        let info = NameAndVersion { name: name, version: version };
        let work_info = WorkInfo::remote(config, &mut source, info)?;

        (work_info, version.to_owned())
    } else if let Some(path) = matches.opt_str("s") {
        let work_info = WorkInfo::local(config, Some(PathBuf::from(path)))?;
        let version = format!("{}", work_info.package.version());
        (work_info, version)
    } else {
        let stable_crate = exact_search(&name)?;
        let info = NameAndVersion { name: &name, version: &stable_crate.max_version };
        let work_info = WorkInfo::remote(config, &mut source, info)?;

        (work_info, stable_crate.max_version.clone())
    };

    let (current_rlib, current_deps_output) = current.rlib_and_dep_output(config, &name)?;
    let (stable_rlib, stable_deps_output) = stable.rlib_and_dep_output(config, &name)?;

    if matches.opt_present("d") {
        println!("--crate-type=lib --extern old={} -L{} --extern new={} -L{} tests/helper/test2.rs",
                 stable_rlib.display(),
                 stable_deps_output.display(),
                 current_rlib.display(),
                 current_deps_output.display());
        return Ok(());
    }

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

fn help(opts: &Options) {
    let brief = "usage: cargo semver [options] [-- cargo options]";
    print!("{}", opts.usage(brief));
}

fn version() {
    println!("{}", env!("CARGO_PKG_VERSION"));
}

fn main() {
    fn err(config: &Config, e: Box<CargoError>) -> ! {
        exit_with_error(CliError::new(e, 1), &mut config.shell());
    }

    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut opts = Options::new();

    opts.optflag("h", "help", "print this message and exit");
    opts.optflag("V", "version", "print version information and exit");
    opts.optflag("d", "debug", "print command to debug and exit");
    opts.optopt("s", "stable-path", "use local path as stable/old crate", "PATH");
    opts.optopt("c", "current-path", "use local path as current/new crate", "PATH");
    opts.optopt("S", "stable-pkg", "use a name-version string as stable/old crate", "SPEC");
    opts.optopt("C", "current-pkg", "use a name-version string as current/new crate", "SPEC");

    let config = match Config::default() {
        Ok(cfg) => cfg,
        Err(e) => panic!("can't obtain config: {:?}", e),
    };

    let matches = match opts.parse(&args) {
        Ok(m) => m,
        Err(f) => err(&config, human(f.to_string())),
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
        let msg = "at most one of `-s,--stable-path` and `-S,--stable-pkg` allowed".to_owned();
        err(&config, human(msg));
    }

    if (matches.opt_present("c") && matches.opt_present("C")) ||
        matches.opt_count("c") > 1 || matches.opt_count("C") > 1
    {
        let msg = "at most one of `-c,--current-path` and `-C,--current-pkg` allowed".to_owned();
        err(&config, human(msg));
    }

    if let Err(e) = do_main(&config, &matches) {
        err(&config, e);
    }
}
