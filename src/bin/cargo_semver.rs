#![feature(rustc_private)]
#![feature(set_stdio)]

extern crate getopts;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;

use cargo::core::{Package, PackageId, PackageSet, Source, SourceId, SourceMap, Workspace};
use log::debug;
use std::{
    env,
    fs::File,
    io::BufReader,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

pub type Result<T> = cargo::util::CargoResult<T>;

#[derive(Debug, Deserialize)]
struct Invocation {
    package_name: String,
    outputs: Vec<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct BuildPlan {
    invocations: Vec<Invocation>,
}

/// Main entry point.
///
/// Parse CLI arguments, handle their semantics, and provide for proper error handling.
fn main() {
    if env_logger::try_init().is_err() {
        eprintln!("ERROR: could not initialize logger");
    }

    let config = match cargo::Config::default() {
        Ok(cfg) => cfg,
        Err(e) => panic!("can't obtain cargo config: {:?}", e),
    };

    let opts = cli::options();

    let matches = match cli::parse_args(&opts) {
        Ok(m) => m,
        Err(f) => cli::exit_with_error(&config, f.to_owned().into()),
    };

    if matches.opt_present("h") {
        cli::print_help(&opts);
        return;
    }

    if matches.opt_present("V") {
        cli::print_version();
        return;
    }

    if let Err(e) = cli::validate_args(&matches) {
        cli::exit_with_error(&config, e);
    }

    if let Err(e) = run(&config, &matches, matches.opt_present("e")) {
        cli::exit_with_error(&config, e);
    }
}

/// Obtain two versions of the same crate, the "current" version, and the
/// "stable" version, compile them both into `rlib`s, and report the breaking
/// introduced in the "current" version with respect to the "stable" version.
// TODO: possibly reduce the complexity by finding where some info can be taken from directly
fn run(config: &cargo::Config, matches: &getopts::Matches, explain: bool) -> Result<()> {
    use cargo::util::important_paths::find_root_manifest_for_wd;
    debug!("running cargo-semver");

    // Obtain WorkInfo for the "current"
    let current = if let Some(name_and_version) = matches.opt_str("C") {
        // -C "name:version" requires fetching the appropriate package:
        WorkInfo::remote(
            config,
            SourceInfo::new(config)?,
            &PackageNameAndVersion::parse(&name_and_version)?,
        )?
    } else if let Some(path) = matches.opt_str("c").map(PathBuf::from) {
        // -c "local_path":
        WorkInfo::local(config, &find_root_manifest_for_wd(&path)?)?
    } else {
        // default: if neither -c / -C are used, use the workspace at the
        // current working directory:
        WorkInfo::local(config, &find_root_manifest_for_wd(config.cwd())?)?
    };
    let name = current.package.name().to_owned();

    // Obtain WorkInfo for the "stable" version
    let (stable, stable_version) = if let Some(name_and_version) = matches.opt_str("S") {
        // -S "name:version" requires fetching the appropriate package:
        let info = PackageNameAndVersion::parse(&name_and_version)?;
        let version = info.version.to_owned();
        let work_info = WorkInfo::remote(config, SourceInfo::new(config)?, &info)?;
        (work_info, version)
    } else if let Some(path) = matches.opt_str("s") {
        // -s "local_path":
        let work_info = WorkInfo::local(config, &PathBuf::from(path))?;
        let version = format!("{}", work_info.package.version());
        (work_info, version)
    } else {
        // default: if neither -s / -S are used, use the current's crate name to find the
        // latest stable version of the crate on crates.io and use that one:
        let stable_crate = find_on_crates_io(&name)?;
        let info = PackageNameAndVersion {
            name: &name,
            version: &stable_crate.max_version,
        };
        let work_info = WorkInfo::remote(config, SourceInfo::new(config)?, &info)?;
        (work_info, stable_crate.max_version.clone())
    };

    let (current_rlib, current_deps_output) = current.rlib_and_dep_output(config, &name, true)?;
    let (stable_rlib, stable_deps_output) = stable.rlib_and_dep_output(config, &name, false)?;

    println!("current_rlib: {:?}", current_rlib);
    println!("stable_rlib: {:?}", stable_rlib);

    if matches.opt_present("d") {
        println!(
            "--extern old={} -L{} --extern new={} -L{}",
            stable_rlib.display(),
            stable_deps_output.display(),
            current_rlib.display(),
            current_deps_output.display()
        );
        return Ok(());
    }

    debug!("running rust-semverver on compiled crates");

    let mut child = Command::new("rust-semverver");
    child
        .arg("--crate-type=lib")
        .args(&["--extern", &*format!("old={}", stable_rlib.display())])
        .args(&[format!("-L{}", stable_deps_output.display())])
        .args(&["--extern", &*format!("new={}", current_rlib.display())])
        .args(&[format!("-L{}", current_deps_output.display())]);

    if let Some(target) = matches.opt_str("target") {
        child.args(&["--target", &target]);
    }

    let child = child
        .arg("-")
        .stdin(Stdio::piped())
        .env("RUST_SEMVER_CRATE_VERSION", stable_version)
        .env("RUST_SEMVER_VERBOSE", format!("{}", explain))
        .env(
            "RUST_SEMVER_API_GUIDELINES",
            if matches.opt_present("a") {
                "true"
            } else {
                "false"
            },
        );

    let mut child = child
        .spawn()
        .map_err(|e| failure::err_msg(format!("could not spawn rustc: {}", e)))?;

    if let Some(ref mut stdin) = child.stdin {
        // The order of the `extern crate` declaration is important here: it will later
        // be used to select the `old` and `new` crates.
        stdin.write_fmt(format_args!(
            "#[allow(unused_extern_crates)] \
             extern crate old; \
             #[allow(unused_extern_crates)] \
             extern crate new;"
        ))?;
    } else {
        return Err(failure::err_msg("could not pipe to rustc (wtf?)".to_owned()).into());
    }

    let exit_status = child
        .wait()
        .map_err(|e| failure::err_msg(format!("failed to wait for rustc: {}", e)))?;

    if exit_status.success() {
        Ok(())
    } else {
        Err(failure::err_msg("rustc-semverver errored".to_owned()))
    }
}

/// CLI utils
mod cli {
    extern crate getopts;
    use cargo::util::CliError;

    /// CLI options
    pub fn options() -> getopts::Options {
        let mut opts = getopts::Options::new();

        opts.optflag("h", "help", "print this message and exit");
        opts.optflag("V", "version", "print version information and exit");
        opts.optflag("e", "explain", "print detailed error explanations");
        opts.optflag("d", "debug", "print command to debug and exit");
        opts.optflag(
            "a",
            "api-guidelines",
            "report only changes that are breaking according to the API-guidelines",
        );
        opts.optopt(
            "s",
            "stable-path",
            "use local path as stable/old crate",
            "PATH",
        );
        opts.optopt(
            "c",
            "current-path",
            "use local path as current/new crate",
            "PATH",
        );
        opts.optopt(
            "S",
            "stable-pkg",
            "use a `name:version` string as stable/old crate",
            "NAME:VERSION",
        );
        opts.optopt(
            "C",
            "current-pkg",
            "use a `name:version` string as current/new crate",
            "NAME:VERSION",
        );
        opts.optopt("", "target", "Build for the target triple", "<TRIPLE>");
        opts
    }

    /// Parse CLI arguments
    pub fn parse_args(opts: &getopts::Options) -> Result<getopts::Matches, getopts::Fail> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        opts.parse(&args)
    }

    /// Validate CLI arguments
    pub fn validate_args(matches: &getopts::Matches) -> Result<(), failure::Error> {
        if (matches.opt_present("s") && matches.opt_present("S"))
            || matches.opt_count("s") > 1
            || matches.opt_count("S") > 1
        {
            let msg = "at most one of `-s,--stable-path` and `-S,--stable-pkg` allowed";
            return Err(failure::err_msg(msg.to_owned()));
        }

        if (matches.opt_present("c") && matches.opt_present("C"))
            || matches.opt_count("c") > 1
            || matches.opt_count("C") > 1
        {
            let msg = "at most one of `-c,--current-path` and `-C,--current-pkg` allowed";
            return Err(failure::err_msg(msg.to_owned()));
        }

        Ok(())
    }

    /// Print a help message
    pub fn print_help(opts: &getopts::Options) {
        let brief = "usage: cargo semver [options] [-- cargo options]";
        print!("{}", opts.usage(brief));
    }

    /// Print a version message.
    pub fn print_version() {
        println!("{}", env!("CARGO_PKG_VERSION"));
    }

    /// Exit with error `e`.
    pub fn exit_with_error(config: &cargo::Config, e: failure::Error) -> ! {
        cargo::exit_with_error(CliError::new(e, 1), &mut config.shell());
    }
}

/// A package's name and version.
pub struct PackageNameAndVersion<'a> {
    /// The crate's name.
    pub name: &'a str,
    /// The package's version, as a semver-string.
    pub version: &'a str,
}

impl<'a> PackageNameAndVersion<'a> {
    /// Parses the string "name:version" into `Self`
    pub fn parse(s: &'a str) -> Result<Self> {
        let err = || {
            failure::err_msg(format!(
                "spec has to be of form `name:version` but is `{}`",
                s
            ))
        };
        let mut split = s.split(':');
        let name = split.next().ok_or_else(err)?;
        let version = split.next().ok_or_else(err)?;
        if split.next().is_some() {
            Err(err())
        } else {
            Ok(Self { name, version })
        }
    }
}

/// A specification of a package source to fetch remote packages from.
pub struct SourceInfo<'a> {
    /// The source id to be used.
    id: SourceId,
    /// The source to be used.
    source: Box<Source + 'a>,
}

impl<'a> SourceInfo<'a> {
    /// Construct a new source info for `crates.io`.
    pub fn new(config: &'a cargo::Config) -> Result<SourceInfo<'a>> {
        let source_id = SourceId::crates_io(config)?;
        let source = source_id.load(config)?;

        debug!("source id loaded: {:?}", source_id);

        Ok(Self {
            id: source_id,
            source,
        })
    }
}

/// A specification of a package and it's workspace.
pub struct WorkInfo<'a> {
    /// The package to be compiled.
    pub package: Package,
    /// The package's workspace.
    workspace: Workspace<'a>,
}

impl<'a> WorkInfo<'a> {
    /// Construct a package/workspace pair for the `manifest_path`
    pub fn local(config: &'a cargo::Config, manifest_path: &Path) -> Result<WorkInfo<'a>> {
        let workspace = Workspace::new(&manifest_path, config)?;
        let package = workspace.load(&manifest_path)?;
        Ok(Self { package, workspace })
    }

    /// Construct a package/workspace pair by fetching the package of a
    /// specified `PackageNameAndVersion` from the `source`.
    pub fn remote(
        config: &'a cargo::Config,
        source: SourceInfo<'a>,
        &PackageNameAndVersion { name, version }: &PackageNameAndVersion,
    ) -> Result<WorkInfo<'a>> {
        // TODO: fall back to locally cached package instance, or better yet, search for it
        // first.
        let package_ids = [PackageId::new(name, version, &source.id)?];
        debug!("(remote) package id: {:?}", package_ids[0]);
        let sources = {
            let mut s = SourceMap::new();
            s.insert(source.source);
            s
        };

        let package_set = PackageSet::new(&package_ids, sources, config)?;
        let package = package_set.get_one(&package_ids[0])?;
        let workspace = Workspace::ephemeral(package.clone(), config, None, false)?;

        Ok(Self {
            package: package.clone(),
            workspace,
        })
    }

    /// Obtain the paths to the produced rlib and the dependency output directory.
    pub fn rlib_and_dep_output(
        &self,
        config: &'a cargo::Config,
        name: &str,
        current: bool,
    ) -> Result<(PathBuf, PathBuf)> {
        let mut opts =
            cargo::ops::CompileOptions::new(config, cargo::core::compiler::CompileMode::Build)?;
        // we need the build plan to find our build artifacts
        opts.build_config.build_plan = true;
        // TODO: this is where we could insert feature flag builds (or using the CLI mechanisms)

        env::set_var(
            "RUSTFLAGS",
            format!("-C metadata={}", if current { "new" } else { "old" }),
        );

        let mut outdir = env::temp_dir();
        outdir.push(&format!("cargo_semver_{}_{}", name, current));

        // redirection gang
        let outfile = File::create(&outdir)?;
        let old_stdio = std::io::set_print(Some(Box::new(outfile)));

        let _ = cargo::ops::compile(&self.workspace, &opts)?;

        std::io::set_print(old_stdio);

        // actually compile things now
        opts.build_config.build_plan = false;

        let compilation = cargo::ops::compile(&self.workspace, &opts)?;
        env::remove_var("RUSTFLAGS");

        let build_plan: BuildPlan = serde_json::from_reader(BufReader::new(File::open(&outdir)?))?;

        // TODO: handle multiple outputs gracefully
        for i in &build_plan.invocations {
            if i.package_name == name {
                return Ok((i.outputs[0].clone(), compilation.deps_output));
            }
        }

        Err(failure::err_msg("lost build artifact".to_owned()))
    }
}

/// Given a `crate_name`, try to locate the corresponding crate on `crates.io`.
///
/// If no crate with the exact name is present, error out.
pub fn find_on_crates_io(crate_name: &str) -> Result<crates_io::Crate> {
    let mut registry = crates_io::Registry::new("https://crates.io".to_owned(), None);

    registry
        .search(crate_name, 1)
        .map_err(|e| {
            failure::err_msg(format!(
                "failed to retrieve search results from the registry: {}",
                e
            ))
            .into()
        })
        .and_then(|(mut crates, _)| {
            crates
                .drain(..)
                .find(|krate| krate.name == crate_name)
                .ok_or_else(|| {
                    failure::err_msg(format!("failed to find a matching crate `{}`", crate_name))
                        .into()
                })
        })
}
