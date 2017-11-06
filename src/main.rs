// error-pattern:yummy
#![feature(box_syntax)]
#![feature(rustc_private)]
#![allow(unknown_lints, missing_docs_in_private_items)]

use std::collections::HashMap;
use std::process;
use std::io::{self, Write};

extern crate cargo_metadata;

use std::path::Path;

const CARGO_CLIPPY_HELP: &str = r#"Checks a package to catch common mistakes and improve your Rust code.

Usage:
    cargo clippy [options] [--] [<opts>...]

Common options:
    -h, --help               Print this message
    --features               Features to compile for the package
    -V, --version            Print version info and exit
    --all                    Run over all packages in the current workspace

Other options are the same as `cargo rustc`.

To allow or deny a lint from the command line you can use `cargo clippy --`
with:

    -W --warn OPT       Set lint warnings
    -A --allow OPT      Set lint allowed
    -D --deny OPT       Set lint denied
    -F --forbid OPT     Set lint forbidden

The feature `cargo-clippy` is automatically defined for convenience. You can use
it to allow or deny lints from the code, eg.:

    #[cfg_attr(feature = "cargo-clippy", allow(needless_lifetimes))]
"#;

#[allow(print_stdout)]
fn show_help() {
    println!("{}", CARGO_CLIPPY_HELP);
}

#[allow(print_stdout)]
fn show_version() {
    println!("{}", env!("CARGO_PKG_VERSION"));
}

pub fn main() {
    use std::env;

    if env::var("CLIPPY_DOGFOOD").is_ok() {
        panic!("yummy");
    }

    // Check for version and help flags even when invoked as 'cargo-clippy'
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        show_help();
        return;
    }
    if std::env::args().any(|a| a == "--version" || a == "-V") {
        show_version();
        return;
    }

    let manifest_path_arg = std::env::args()
        .skip(2)
        .find(|val| val.starts_with("--manifest-path="));

    let mut metadata = if let Ok(metadata) = cargo_metadata::metadata(manifest_path_arg.as_ref().map(AsRef::as_ref)) {
        metadata
    } else {
        let _ = io::stderr().write_fmt(format_args!("error: Could not obtain cargo metadata.\n"));
        process::exit(101);
    };

    let manifest_path = manifest_path_arg.map(|arg| {
        Path::new(&arg["--manifest-path=".len()..])
            .canonicalize()
            .expect("manifest path could not be canonicalized")
    });

    let packages = if std::env::args().any(|a| a == "--all") {
        metadata.packages
    } else {
        let package_index = {
            if let Some(manifest_path) = manifest_path {
                metadata.packages.iter().position(|package| {
                    let package_manifest_path = Path::new(&package.manifest_path)
                        .canonicalize()
                        .expect("package manifest path could not be canonicalized");
                    package_manifest_path == manifest_path
                })
            } else {
                let package_manifest_paths: HashMap<_, _> = metadata
                    .packages
                    .iter()
                    .enumerate()
                    .map(|(i, package)| {
                        let package_manifest_path = Path::new(&package.manifest_path)
                            .parent()
                            .expect("could not find parent directory of package manifest")
                            .canonicalize()
                            .expect("package directory cannot be canonicalized");
                        (package_manifest_path, i)
                    })
                    .collect();

                let current_dir = std::env::current_dir()
                    .expect("could not read current directory")
                    .canonicalize()
                    .expect("current directory cannot be canonicalized");

                let mut current_path: &Path = &current_dir;

                // This gets the most-recent parent (the one that takes the fewest `cd ..`s to
                // reach).
                loop {
                    if let Some(&package_index) = package_manifest_paths.get(current_path) {
                        break Some(package_index);
                    } else {
                        // We'll never reach the filesystem root, because to get to this point in the
                        // code
                        // the call to `cargo_metadata::metadata` must have succeeded. So it's okay to
                        // unwrap the current path's parent.
                        current_path = current_path
                            .parent()
                            .unwrap_or_else(|| panic!("could not find parent of path {}", current_path.display()));
                    }
                }
            }
        }.expect("could not find matching package");

        vec![metadata.packages.remove(package_index)]
    };

    for package in packages {
        let manifest_path = package.manifest_path;

        for target in package.targets {
            let args = std::env::args()
                .skip(2)
                .filter(|a| a != "--all" && !a.starts_with("--manifest-path="));

            let args = std::iter::once(format!("--manifest-path={}", manifest_path)).chain(args);
            if let Some(first) = target.kind.get(0) {
                if target.kind.len() > 1 || first.ends_with("lib") {
                    if let Err(code) = process(std::iter::once("--lib".to_owned()).chain(args)) {
                        std::process::exit(code);
                    }
                } else if ["bin", "example", "test", "bench"].contains(&&**first) {
                    if let Err(code) = process(
                        vec![format!("--{}", first), target.name]
                            .into_iter()
                            .chain(args),
                    ) {
                        std::process::exit(code);
                    }
                }
            } else {
                panic!("badly formatted cargo metadata: target::kind is an empty array");
            }
        }
    }
}

fn process<I>(old_args: I) -> Result<(), i32>
where
    I: Iterator<Item = String>,
{
    let mut args = vec!["rustc".to_owned()];

    let mut found_dashes = false;
    for arg in old_args {
        found_dashes |= arg == "--";
        args.push(arg);
    }
    if !found_dashes {
        args.push("--".to_owned());
    }
    args.push("--emit=metadata".to_owned());
    args.push("--cfg".to_owned());
    args.push(r#"feature="cargo-clippy""#.to_owned());

    let path = std::env::current_exe()
        .expect("current executable path invalid")
        .with_file_name("clippy-driver");
    let exit_status = std::process::Command::new("cargo")
        .args(&args)
        .env("RUSTC_WRAPPER", path)
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
