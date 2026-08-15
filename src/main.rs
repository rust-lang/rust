// We need this feature as it changes `dylib` linking behavior.
#![feature(rustc_private)]
// warn on lints, that are included in `rust-lang/rust`s bootstrap
#![warn(rust_2018_idioms, unused_lifetimes)]

use std::env;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let args = match Args::parse(env::args()) {
        Ok(args) => args,
        Err(e) => {
            e.print();
            return ExitCode::FAILURE;
        },
    };

    if args.help {
        return match anstream::stdout().write_all(HELP_MSG.as_bytes()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ExitCode::FAILURE,
        };
    }
    if args.version {
        return match writeln!(anstream::stdout(), "{}", rustc_tools_util::get_version_info!()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ExitCode::FAILURE,
        };
    }
    if let Some(lint) = &args.explain {
        return clippy_lints::explain(lint);
    }

    let exit_status = Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .env("RUSTC_WORKSPACE_WRAPPER", driver_path())
        .env("CLIPPY_ARGS", &args.clippy_args)
        .arg(if args.fix { "fix" } else { "check" })
        .args(&args.cargo_args)
        .spawn()
        .expect("could not run cargo")
        .wait()
        .expect("failed to wait for cargo?");

    if exit_status.success() {
        ExitCode::SUCCESS
    } else {
        let code = exit_status.code().unwrap_or(-1);
        cfg_select! {
            // Unlike unix platforms, Windows uses the full 32-bits for exit codes.
            windows => {
                use std::os::windows::process::ExitCodeExt;
                ExitCode::from_raw(code as u32)
            },
            _ =>
            {
                #[expect(clippy::cast_possible_truncation)]
                ExitCode::from(code.cast_unsigned() as u8)
            },
        }
    }
}

#[expect(clippy::struct_field_names)]
struct Args {
    /// The arguments to forward to cargo.
    cargo_args: Vec<String>,
    /// Arguments to pass through to `clippy-driver`. Arguments are separated by
    /// `__CLIPPY_HACKERY__` and passed via an environment variable.
    clippy_args: String,
    /// Whether to stop and print info for the specified lint.
    explain: Option<String>,
    /// Whether to stop and print help info.
    help: bool,
    /// Whether to stop and print version info.
    version: bool,
    /// Whether to run in fix mode.
    fix: bool,
}
impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, ArgError> {
        fn parse_explain(arg: Option<String>) -> Result<String, ArgError> {
            if let Some(mut arg) = arg
                && !arg.starts_with('-')
            {
                arg.make_ascii_lowercase();
                Ok(arg.strip_prefix("clippy::").unwrap_or(&*arg).replace('-', "_"))
            } else {
                Err(ArgError::NoExplainArg)
            }
        }

        let mut parsed = Args {
            cargo_args: Vec::new(),
            clippy_args: String::new(),
            explain: None,
            help: false,
            version: false,
            fix: false,
        };
        let mut no_deps = false;

        // Cargo forwards to sub-commands as `cargo sub-command args`.
        // Special case handling help and version in case this is called directly,
        // but otherwise ignore the second argument.
        match args.nth(1) {
            Some(x) => match &*x {
                "-h" | "--help" => parsed.help = true,
                "-V" | "--version" => parsed.version = true,
                "--explain" => parsed.explain = Some(parse_explain(args.next())?),
                // FIXME(@Jarcho): Handle all arguments here without breaking too many existing callers.
                _ => {},
            },
            None => return Ok(parsed),
        }
        while let Some(arg) = args.next() {
            match &*arg {
                "-h" | "--help" => parsed.help = true,
                "-V" | "--version" => parsed.version = true,
                "--explain" if parsed.explain.is_some() => return Err(ArgError::MultipleExplain),
                "--explain" => parsed.explain = Some(parse_explain(args.next())?),
                "--fix" => parsed.fix = true,
                "--no-deps" => {
                    no_deps = true;
                    parsed.clippy_args.push_str("--no-deps__CLIPPY_HACKERY__");
                },
                "--" => {
                    for arg in args {
                        no_deps |= arg == "--no-deps";
                        parsed.clippy_args.extend([&*arg, "__CLIPPY_HACKERY__"]);
                    }
                    break;
                },
                _ => parsed.cargo_args.push(arg),
            }
        }
        if parsed.fix && !no_deps {
            parsed.clippy_args.push_str("--no-deps__CLIPPY_HACKERY__");
        }
        Ok(parsed)
    }
}

/// Gets the driver path relative to the current cargo-clippy.
fn driver_path() -> PathBuf {
    const DRIVER_NAME: &str = if cfg!(windows) {
        "clippy-driver.exe"
    } else {
        "clippy-driver"
    };

    let mut path = env::current_exe().expect("current executable path invalid");
    path.set_file_name(DRIVER_NAME);
    path
}

#[derive(Debug)]
enum ArgError {
    MultipleExplain,
    NoExplainArg,
}
impl ArgError {
    fn print(&self) {
        let msg = match self {
            Self::MultipleExplain => "multiple `--explain` arguments",
            Self::NoExplainArg => "missing value for `--explain`",
        };
        let mut dst = anstream::stderr().lock();
        let _ = dst
            .write_all(color_print::cstr!("<bold,red>error</>: ").as_bytes())
            .and_then(|()| dst.write_all(msg.as_bytes()))
            .and_then(|()| dst.write_all(HELP_USAGE.as_bytes()));
    }
}

const HELP_MSG: &str = color_print::cstr!(
"Checks a package to catch common mistakes and improve your Rust code.

<green,bold>Usage</>:
    <cyan,bold>cargo clippy</> <cyan>[OPTIONS] [--] [<<ARGS>>...]</>

<green,bold>Common options:</>
    <cyan,bold>--no-deps</>                Run Clippy only on the given crate, without linting the dependencies
    <cyan,bold>--fix</>                    Automatically apply lint suggestions. This flag implies <cyan>--no-deps</> and <cyan>--all-targets</>
    <cyan,bold>-h</>, <cyan,bold>--help</>               Print this message
    <cyan,bold>-V</>, <cyan,bold>--version</>            Print version info and exit
    <cyan,bold>--explain [LINT]</>         Print the documentation for a given lint

See all options with <cyan,bold>cargo check --help</>.

<green,bold>Allowing / Denying lints</>

To allow or deny a lint from the command line you can use <cyan,bold>cargo clippy --</> with:

    <cyan,bold>-W</> / <cyan,bold>--warn</> <cyan>[LINT]</>       Set lint warnings
    <cyan,bold>-A</> / <cyan,bold>--allow</> <cyan>[LINT]</>      Set lint allowed
    <cyan,bold>-D</> / <cyan,bold>--deny</> <cyan>[LINT]</>       Set lint denied
    <cyan,bold>-F</> / <cyan,bold>--forbid</> <cyan>[LINT]</>     Set lint forbidden

You can use tool lints to allow or deny lints from your code, e.g.:

    <yellow,bold>#[allow(clippy::needless_lifetimes)]</>

<green,bold>Manifest Options:</>
    <cyan,bold>--manifest-path</> <cyan><<PATH>></>  Path to Cargo.toml
    <cyan,bold>--frozen</>                Require Cargo.lock and cache are up to date
    <cyan,bold>--locked</>                Require Cargo.lock is up to date
    <cyan,bold>--offline</>               Run without accessing the network

");
const HELP_USAGE: &str = {
    let mut i = 0usize;
    while HELP_MSG.as_bytes()[i] != b'\n' {
        i += 1;
    }
    HELP_MSG.split_at(i).1 // include leading line ends
};

#[cfg(test)]
mod tests {
    use super::Args;

    fn parse(args: &[&str]) -> Args {
        Args::parse(args.iter().copied().map(String::from)).unwrap()
    }
    fn clippy_args(args: &Args) -> impl Iterator<Item = &str> {
        args.clippy_args.split("__CLIPPY_HACKERY__")
    }

    #[test]
    fn fix_implies_no_deps() {
        let args = parse(&["cargo", "clippy", "--fix"]);
        assert!(clippy_args(&args).any(|arg| arg == "--no-deps"));
    }

    #[test]
    fn no_deps_not_duplicated_with_fix() {
        let args = parse(&["cargo", "clippy", "--fix", "--no-deps"]);
        assert_eq!(clippy_args(&args).filter(|&arg| arg == "--no-deps").count(), 1);
    }

    #[test]
    fn no_deps_not_duplicated_with_fix_extra() {
        let args = parse(&["cargo", "clippy", "--fix", "--", "--no-deps"]);
        assert_eq!(clippy_args(&args).filter(|&arg| arg == "--no-deps").count(), 1);
    }
}
