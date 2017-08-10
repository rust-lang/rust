extern crate cargo_metadata;

use std::path::{PathBuf, Path};
use std::io::Write;
use std::process::Command;


const CARGO_MIRI_HELP: &str = r#"Interprets bin crates

Usage:
    cargo miri [options] [--] [<opts>...]

Common options:
    -h, --help               Print this message
    --features               Features to compile for the package
    -V, --version            Print version info and exit

Other options are the same as `cargo rustc`.

The feature `cargo-miri` is automatically defined for convenience. You can use
it to configure the resource limits

    #![cfg_attr(feature = "cargo-miri", memory_size = 42)]

available resource limits are `memory_size`, `step_limit`, `stack_limit`
"#;

fn show_help() {
    println!("{}", CARGO_MIRI_HELP);
}

fn show_version() {
    println!("{}", env!("CARGO_PKG_VERSION"));
}

fn main() {
    // Check for version and help flags even when invoked as 'cargo-miri'
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        show_help();
        return;
    }
    if std::env::args().any(|a| a == "--version" || a == "-V") {
        show_version();
        return;
    }

    if let Some("miri") = std::env::args().nth(1).as_ref().map(AsRef::as_ref) {
        // this arm is when `cargo miri` is called

        let test = std::env::args().nth(2).map_or(false, |text| text == "test");
        let skip = if test { 3 } else { 2 };

        let manifest_path_arg = std::env::args().skip(skip).find(|val| {
            val.starts_with("--manifest-path=")
        });

        let mut metadata = if let Ok(metadata) = cargo_metadata::metadata(
            manifest_path_arg.as_ref().map(AsRef::as_ref),
        )
        {
            metadata
        } else {
            let _ = std::io::stderr().write_fmt(format_args!(
                "error: Could not obtain cargo metadata."
            ));
            std::process::exit(101);
        };

        let manifest_path = manifest_path_arg.map(|arg| {
            PathBuf::from(Path::new(&arg["--manifest-path=".len()..]))
        });

        let current_dir = std::env::current_dir();

        let package_index = metadata
            .packages
            .iter()
            .position(|package| {
                let package_manifest_path = Path::new(&package.manifest_path);
                if let Some(ref manifest_path) = manifest_path {
                    package_manifest_path == manifest_path
                } else {
                    let current_dir = current_dir.as_ref().expect(
                        "could not read current directory",
                    );
                    let package_manifest_directory = package_manifest_path.parent().expect(
                        "could not find parent directory of package manifest",
                    );
                    package_manifest_directory == current_dir
                }
            })
            .expect("could not find matching package");
        let package = metadata.packages.remove(package_index);
        for target in package.targets {
            let args = std::env::args().skip(skip);
            let kind = target.kind.get(0).expect(
                "badly formatted cargo metadata: target::kind is an empty array",
            );
            if test && kind == "test" {
                if let Err(code) = process(
                    vec!["--test".to_string(), target.name].into_iter().chain(
                        args,
                    ),
                )
                {
                    std::process::exit(code);
                }
            } else if !test && kind == "bin" {
                if let Err(code) = process(
                    vec!["--bin".to_string(), target.name].into_iter().chain(
                        args,
                    ),
                )
                {
                    std::process::exit(code);
                }
            }
        }
    } else {
        // this arm is executed when cargo-miri runs `cargo rustc` with the `RUSTC` env var set to itself

        let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
        let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
        let sys_root = if let (Some(home), Some(toolchain)) = (home, toolchain) {
            format!("{}/toolchains/{}", home, toolchain)
        } else {
            option_env!("RUST_SYSROOT")
                .map(|s| s.to_owned())
                .or_else(|| {
                    Command::new("rustc")
                        .arg("--print")
                        .arg("sysroot")
                        .output()
                        .ok()
                        .and_then(|out| String::from_utf8(out.stdout).ok())
                        .map(|s| s.trim().to_owned())
                })
                .expect("need to specify RUST_SYSROOT env var during miri compilation, or use rustup or multirust")
        };

        // this conditional check for the --sysroot flag is there so users can call `cargo-miri` directly
        // without having to pass --sysroot or anything
        let mut args: Vec<String> = if std::env::args().any(|s| s == "--sysroot") {
            std::env::args().skip(1).collect()
        } else {
            std::env::args()
                .skip(1)
                .chain(Some("--sysroot".to_owned()))
                .chain(Some(sys_root))
                .collect()
        };

        // this check ensures that dependencies are built but not interpreted and the final crate is
        // interpreted but not built
        let miri_enabled = std::env::args().any(|s| s == "-Zno-trans");

        let mut command = if miri_enabled {
            let mut path = std::env::current_exe().expect("current executable path invalid");
            path.set_file_name("miri");
            Command::new(path)
        } else {
            Command::new("rustc")
        };

        args.extend_from_slice(&["-Z".to_owned(), "always-encode-mir".to_owned()]);
        args.extend_from_slice(&["--cfg".to_owned(), r#"feature="cargo-miri""#.to_owned()]);

        match command.args(&args).status() {
            Ok(exit) => {
                if !exit.success() {
                    std::process::exit(exit.code().unwrap_or(42));
                }
            }
            Err(ref e) if miri_enabled => panic!("error during miri run: {:?}", e),
            Err(ref e) => panic!("error during rustc call: {:?}", e),
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
    args.push("-Zno-trans".to_owned());
    args.push("--cfg".to_owned());
    args.push(r#"feature="cargo-miri""#.to_owned());

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
