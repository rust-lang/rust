#![feature(box_syntax)]
// #![feature(rustc_private)]

// extern crate cargo_metadata;
// extern crate getopts;
// extern crate rustc;
// extern crate rustc_driver;
// extern crate rustc_errors;
// extern crate syntax;

extern crate cargo;
extern crate crates_io;

use crates_io::{Crate, Registry};

use cargo::exit_with_error;
use cargo::core::{Package, PackageId, Source, SourceId, Workspace};
use cargo::ops::{compile, Compilation, CompileMode, CompileOptions};
use cargo::sources::registry::RegistrySource;
use cargo::util::{human, CargoResult, CliError};
use cargo::util::config::Config;
use cargo::util::important_paths::find_root_manifest_for_wd;

// use rustc::session::{config, Session};
// use rustc::session::config::{Input, ErrorOutputType};

// use rustc_driver::{driver, CompilerCalls, RustcDefaultCalls, Compilation};

// use std::io::{self, Write};
// use std::path::{Path, PathBuf};
// use std::process::Command;

// use syntax::ast;

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
                          .ok_or(human(format!("failed to find a matching crate `{}`", query)))
                  })
}

/// Compile a crate given it's workspace.
///
/// The results returned are then used to locate the build artefacts, which in turn are linked
/// together for the actual analysis.
fn generate_rlib<'a>(config: &'a Config,
                     workspace: Workspace<'a>) -> CargoResult<Compilation<'a>> {
    let opts = CompileOptions::default(config, CompileMode::Build);
    compile(&workspace, &opts).map(|c| c)
}

/// Perform the heavy lifting.
///
/// Obtain the local crate and compile it, then fetch the latest version from the registry, and
/// build it as well.
///
/// TODO:
/// * split this up
/// * no, seriously, split this up
/// * give some structure to the build artefact gathering
/// * possibly reduce the complexity by investigating where some of the info can be sourced from
/// in a more direct fashion
fn do_main() -> CargoResult<()> {
    let config = Config::default()?;

    let manifest_path = find_root_manifest_for_wd(None, config.cwd())?;

    let local_package = Package::for_path(&manifest_path, &config)?;
    let local_workspace = Workspace::new(&manifest_path, &config)?;
    let local_compilation = generate_rlib(&config, local_workspace)?;

    let name = local_package.name();

    for i in &local_compilation.libraries[local_package.package_id()] {
        if i.0.name() == name {
            println!("{:?}", i.1);
        }
    }

    let source_id = SourceId::crates_io(&config)?;
    let mut registry_source = RegistrySource::remote(&source_id, &config);

    let remote_crate = exact_search(name)?;

    let package_id = PackageId::new(name, &remote_crate.max_version, &source_id)?;

    let stable_package = registry_source.download(&package_id)?;
    let stable_workspace = Workspace::ephemeral(stable_package, &config, None, false)?;
    let stable_compilation = generate_rlib(&config, stable_workspace)?;

    for i in &stable_compilation.libraries[&package_id] {
        if i.0.name() == name {
            println!("{:?}", i.1);
        }
    }

    Ok(())
}

fn main() {
    if let Err(e) = do_main() {
        if let Ok(config) = Config::default() {
            exit_with_error(CliError::new(e, 1), &mut config.shell());
        } else {
            panic!("ffs, we can't get a config and errors happened :/");
        }
    }
}

/*
struct SemVerVerCompilerCalls {
    default: RustcDefaultCalls,
    enabled: bool
}

impl SemVerVerCompilerCalls {
    pub fn new(enabled: bool) -> SemVerVerCompilerCalls {
        SemVerVerCompilerCalls {
            default: RustcDefaultCalls,
            enabled: enabled,
        }
    }
}

impl<'a> CompilerCalls<'a> for SemVerVerCompilerCalls {
    fn early_callback(&mut self,
                      matches: &getopts::Matches,
                      sopts: &config::Options,
                      cfg: &ast::CrateConfig,
                      descriptions: &rustc_errors::registry::Registry,
                      output: ErrorOutputType)
                      -> Compilation {
        self.default
            .early_callback(matches, sopts, cfg, descriptions, output)
    }

    fn no_input(&mut self,
                matches: &getopts::Matches,
                sopts: &config::Options,
                cfg: &ast::CrateConfig,
                odir: &Option<PathBuf>,
                ofile: &Option<PathBuf>,
                descriptions: &rustc_errors::registry::Registry)
                -> Option<(Input, Option<PathBuf>)> {
        self.default
            .no_input(matches, sopts, cfg, odir, ofile, descriptions)
    }

    fn late_callback(&mut self,
                     matches: &getopts::Matches,
                     sess: &Session,
                     input: &Input,
                     odir: &Option<PathBuf>,
                     ofile: &Option<PathBuf>)
                     -> Compilation {
        self.default
            .late_callback(matches, sess, input, odir, ofile)
    }

    fn build_controller(&mut self,
                        sess: &Session,
                        matches: &getopts::Matches)
                        -> driver::CompileController<'a> {
        let mut controller = self.default.build_controller(sess, matches);

        if self.enabled {
            let old_callback = std::mem::replace(&mut controller.after_hir_lowering.callback,
                                                 box |_| {});
            controller.after_hir_lowering.callback = box move |state| { old_callback(state); };
        }

        controller
    }
}

const CARGO_SEMVER_HELP: &str = r#"Checks a package's SemVer compatibility with already published versions.

Usage:
    cargo semver [options] [--] [<opts>...]

Common options:
    -h, --help               Print this message
    --features               Features to compile for the package
    -V, --version            Print version info and exit

Other options are the same as `cargo rustc`.
"#;

fn help() {
    println!("{}", CARGO_SEMVER_HELP);
}

fn version() {
    println!("{}", env!("CARGO_PKG_VERSION"));
}

pub fn main() {
    // TODO: maybe don't use cargo_metadata, as it pulls in tons of deps

    if std::env::args().any(|arg| arg == "-h" || arg == "--help") {
        help();
        return;
    }

    if std::env::args().any(|arg| arg == "-V" || arg == "--version") {
        version();
        return;
    }

    if std::env::args()
           .nth(1)
           .map(|arg| arg == "semver")
           .unwrap_or(false) {
        // first run (we blatantly copy clippy's code structure here)
        // we are being run as `cargo semver`
        //
        // TODO: maybe it would make sense to reuse cargo internals here to avoid the quite ugly
        // dance this turns out to be :)

        let manifest_path_arg = std::env::args()
            .skip(2)
            .find(|val| val.starts_with("--manifest-path="));

        let mut metadata = if let Ok(data) =
            cargo_metadata::metadata(manifest_path_arg.as_ref().map(AsRef::as_ref)) {
            data
        } else {
            let _ = io::stderr()
                .write_fmt(format_args!("error: could not obtain cargo metadata.\n"));
            std::process::exit(1);
        };

        let manifest_path = manifest_path_arg.map(|arg| PathBuf::from(
                Path::new(&arg["--manifest-path=".len()..])));

        let current_dir = std::env::current_dir();

        let package_index = metadata
            .packages
            .iter()
            .position(|package| {
                let package_manifest_path = Path::new(&package.manifest_path);
                if let Some(ref path) = manifest_path {
                    package_manifest_path == path
                } else {
                    let current_dir = current_dir
                        .as_ref()
                        .expect("could not read current directory");
                    let package_manifest_directory = package_manifest_path
                        .parent()
                        .expect("could not find parent directory of package manifest");
                    package_manifest_directory == current_dir
                }
            })
            .expect("could not find matching package");

        let package = metadata.packages.remove(package_index);

        for target in package.targets {
            let args = std::env::args().skip(2);

            if let Some(first) = target.kind.get(0) {
                if target.kind.len() > 1 || first.ends_with("lib") {
                    if let Err(code) = process(std::iter::once("--lib".to_owned()).chain(args)) {
                        std::process::exit(code);
                    }
                } else if ["bin", "example", "test", "bench"].contains(&&**first) {
                    if let Err(code) = process(vec![format!("--{}", first), target.name]
                                                   .into_iter()
                                                   .chain(args)) {
                        std::process::exit(code);
                    }
                }
            } else {
                panic!("badly formatted cargo metadata: target::kind is an empty array");
            }
        }
    } else {
        // second run: we're being run by `cargo rustc` as we set it up to happen

        let home = option_env!("RUSTUP_HOME");
        let toolchain = option_env!("RUSTUP_TOOLCHAIN");
        let sys_root = if let (Some(home), Some(toolchain)) = (home, toolchain) {
            format!("{}/toolchains/{}", home, toolchain)
        } else {
            option_env!("SYSROOT")
                .map(|s| s.to_owned())
                .or_else(|| {
                    Command::new("rustc")
                        .args(&["--print", "sysroot"])
                        .output()
                        .ok()
                        .and_then(|out| String::from_utf8(out.stdout).ok())
                        .map(|s| s.trim().to_owned())
                })
                .expect("need to specify SYSROOT env var during compilation, or use rustup")
        };

        rustc_driver::in_rustc_thread(|| {
            // make it possible to call `cargo-semver` directly without having to pass
            // --sysroot or anything
            let args: Vec<String> = if std::env::args().any(|s| s == "--sysroot") {
                std::env::args().collect()
            } else {
                std::env::args()
                    .chain(Some("--sysroot".to_owned()))
                    .chain(Some(sys_root))
                    .collect()
            };

            // this check ensures that dependencies are built but not checked and the final
            // crate is checked but not built
            let checks_enabled = std::env::args().any(|s| s == "-Zno-trans");

            let mut cc = SemVerVerCompilerCalls::new(checks_enabled);
            // TODO: the second result is a `Session` - maybe we'll need it
            let (result, _) = rustc_driver::run_compiler(&args, &mut cc, None, None);

            if let Err(count) = result {
                if count > 0 {
                    std::process::exit(1);
                }
            }
        })
        .expect("rustc thread failed");
    }
}

// run `cargo rustc` with `RUSTC` set to our path
fn process<I>(old_args: I) -> Result<(), i32>
    where I: Iterator<Item = String>
{
    let mut args = vec!["rustc".to_owned()];

    let found_dashes = old_args.fold(false, |mut found, arg| {
        found |= arg == "--";
        args.push(arg);
        found
    });

    if !found_dashes {
        args.push("--".to_owned());
    }

    args.push("-Zno-trans".to_owned());

    let path = std::env::current_exe().expect("current executable path invalid");

    let exit_status = std::process::Command::new("cargo")
        .args(&args)
        .env("RUSTC", path)
        .spawn()
        .expect("could not run cargo")
        .wait()
        .expect("failed to wait for cargo?");

    if exit_status.success() {
        Ok(())
    } else {
        Err(exit_status.code().unwrap_or(-1))
    }
}
*/
