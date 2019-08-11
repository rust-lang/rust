#![cfg_attr(feature = "deny-warnings", deny(warnings))]

use rustc_tools_util::*;

const CARGO_CLIPPY_HELP: &str = r#"Checks a package to catch common mistakes and improve your Rust code.

Usage:
    cargo clippy [options] [--] [<opts>...]

Common options:
    -h, --help               Print this message
    -V, --version            Print version info and exit

Other options are the same as `cargo check`.

To allow or deny a lint from the command line you can use `cargo clippy --`
with:

    -W --warn OPT       Set lint warnings
    -A --allow OPT      Set lint allowed
    -D --deny OPT       Set lint denied
    -F --forbid OPT     Set lint forbidden

You can use tool lints to allow or deny lints from your code, eg.:

    #[allow(clippy::needless_lifetimes)]
"#;

fn show_help() {
    println!("{}", CARGO_CLIPPY_HELP);
}

fn show_version() {
    let version_info = rustc_tools_util::get_version_info!();
    println!("{}", version_info);
}

pub fn main() {
    // Check for version and help flags even when invoked as 'cargo-clippy'
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        show_help();
        return;
    }

    if std::env::args().any(|a| a == "--version" || a == "-V") {
        show_version();
        return;
    }

    if let Err(code) = process(std::env::args().skip(2)) {
        std::process::exit(code);
    }
}

fn process<I>(mut old_args: I) -> Result<(), i32>
where
    I: Iterator<Item = String>,
{
    let mut args = vec!["check".to_owned()];

    for arg in old_args.by_ref() {
        if arg == "--" {
            break;
        }
        args.push(arg);
    }

    let clippy_args: String = old_args.map(|arg| format!("{}__CLIPPY_HACKERY__", arg)).collect();

    let mut path = std::env::current_exe()
        .expect("current executable path invalid")
        .with_file_name("clippy-driver");
    if cfg!(windows) {
        path.set_extension("exe");
    }

    let target_dir = std::env::var_os("CLIPPY_DOGFOOD")
        .map(|_| {
            std::env::var_os("CARGO_MANIFEST_DIR").map_or_else(
                || std::ffi::OsString::from("clippy_dogfood"),
                |d| {
                    std::path::PathBuf::from(d)
                        .join("target")
                        .join("dogfood")
                        .into_os_string()
                },
            )
        })
        .map(|p| ("CARGO_TARGET_DIR", p));

    // Run the dogfood tests directly on nightly cargo. This is required due
    // to a bug in rustup.rs when running cargo on custom toolchains. See issue #3118.
    if std::env::var_os("CLIPPY_DOGFOOD").is_some() && cfg!(windows) {
        args.insert(0, "+nightly".to_string());
    }

    let exit_status = std::process::Command::new("cargo")
        .args(&args)
        .env("RUSTC_WRAPPER", path)
        .env("CLIPPY_ARGS", clippy_args)
        .envs(target_dir)
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
