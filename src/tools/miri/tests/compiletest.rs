use colored::*;
use regex::bytes::Regex;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::{env, process::Command};
use ui_test::{color_eyre::Result, Config, Match, Mode, OutputConflictHandling};
use ui_test::{status_emitter, CommandBuilder};

fn miri_path() -> PathBuf {
    PathBuf::from(option_env!("MIRI").unwrap_or(env!("CARGO_BIN_EXE_miri")))
}

fn get_host() -> String {
    rustc_version::VersionMeta::for_command(std::process::Command::new(miri_path()))
        .expect("failed to parse rustc version info")
        .host
}

// Build the shared object file for testing external C function calls.
fn build_so_for_c_ffi_tests() -> PathBuf {
    let cc = option_env!("CC").unwrap_or("cc");
    // Target directory that we can write to.
    let so_target_dir = Path::new(&env::var_os("CARGO_TARGET_DIR").unwrap()).join("miri-extern-so");
    // Create the directory if it does not already exist.
    std::fs::create_dir_all(&so_target_dir)
        .expect("Failed to create directory for shared object file");
    let so_file_path = so_target_dir.join("libtestlib.so");
    let cc_output = Command::new(cc)
        .args([
            "-shared",
            "-o",
            so_file_path.to_str().unwrap(),
            "tests/extern-so/test.c",
            // Only add the functions specified in libcode.version to the shared object file.
            // This is to avoid automatically adding `malloc`, etc.
            // Source: https://anadoxin.org/blog/control-over-symbol-exports-in-gcc.html/
            "-fPIC",
            "-Wl,--version-script=tests/extern-so/libcode.version",
        ])
        .output()
        .expect("failed to generate shared object file for testing external C function calls");
    if !cc_output.status.success() {
        panic!("error in generating shared object file for testing external C function calls");
    }
    so_file_path
}

fn test_config(target: &str, path: &str, mode: Mode, with_dependencies: bool) -> Config {
    // Miri is rustc-like, so we create a default builder for rustc and modify it
    let mut program = CommandBuilder::rustc();
    program.program = miri_path();

    // Add some flags we always want.
    program.args.push("-Dwarnings".into());
    program.args.push("-Dunused".into());
    if let Ok(extra_flags) = env::var("MIRIFLAGS") {
        for flag in extra_flags.split_whitespace() {
            program.args.push(flag.into());
        }
    }
    program.args.push("-Zui-testing".into());
    program.args.push("--target".into());
    program.args.push(target.into());

    // If we're on linux, and we're testing the extern-so functionality,
    // then build the shared object file for testing external C function calls
    // and push the relevant compiler flag.
    if cfg!(target_os = "linux") && path.starts_with("tests/extern-so/") {
        let so_file_path = build_so_for_c_ffi_tests();
        let mut flag = std::ffi::OsString::from("-Zmiri-extern-so-file=");
        flag.push(so_file_path.into_os_string());
        program.args.push(flag);
    }

    let skip_ui_checks = env::var_os("MIRI_SKIP_UI_CHECKS").is_some();

    let output_conflict_handling = match (env::var_os("MIRI_BLESS").is_some(), skip_ui_checks) {
        (false, false) => OutputConflictHandling::Error("./miri bless".into()),
        (true, false) => OutputConflictHandling::Bless,
        (false, true) => OutputConflictHandling::Ignore,
        (true, true) => panic!("cannot use MIRI_BLESS and MIRI_SKIP_UI_CHECKS at the same time"),
    };

    let mut config = Config {
        target: Some(target.to_owned()),
        stderr_filters: STDERR.clone(),
        stdout_filters: STDOUT.clone(),
        mode,
        program,
        output_conflict_handling,
        out_dir: PathBuf::from(std::env::var_os("CARGO_TARGET_DIR").unwrap()).join("ui"),
        edition: Some("2021".into()),
        ..Config::rustc(path.into())
    };

    let use_std = env::var_os("MIRI_NO_STD").is_none();

    if with_dependencies && use_std {
        config.dependencies_crate_manifest_path =
            Some(Path::new("test_dependencies").join("Cargo.toml"));
        config.dependency_builder.args = vec![
            "run".into(),
            "--manifest-path".into(),
            "cargo-miri/Cargo.toml".into(),
            "--".into(),
            "miri".into(),
            "run".into(), // There is no `cargo miri build` so we just use `cargo miri run`.
        ];
    }
    config
}

fn run_tests(mode: Mode, path: &str, target: &str, with_dependencies: bool) -> Result<()> {
    let config = test_config(target, path, mode, with_dependencies);

    // Handle command-line arguments.
    let mut after_dashdash = false;
    let mut quiet = false;
    let filters = std::env::args()
        .skip(1)
        .filter(|arg| {
            if after_dashdash {
                // Just propagate everything.
                return true;
            }
            match &**arg {
                "--quiet" => {
                    quiet = true;
                    false
                }
                "--" => {
                    after_dashdash = true;
                    false
                }
                s if s.starts_with('-') => {
                    panic!("unknown compiletest flag `{s}`");
                }
                _ => true,
            }
        })
        .collect::<Vec<_>>();
    eprintln!("   Compiler: {}", config.program.display());
    ui_test::run_tests_generic(
        config,
        // The files we're actually interested in (all `.rs` files).
        |path| {
            path.extension().is_some_and(|ext| ext == "rs")
                && (filters.is_empty()
                    || filters.iter().any(|f| path.display().to_string().contains(f)))
        },
        // This could be used to overwrite the `Config` on a per-test basis.
        |_, _| None,
        (
            if quiet {
                Box::<status_emitter::Quiet>::default()
                    as Box<dyn status_emitter::StatusEmitter + Send>
            } else {
                Box::new(status_emitter::Text)
            },
            status_emitter::Gha::</* GHA Actions groups*/ false> {
                name: format!("{mode:?} {path} ({target})"),
            },
        ),
    )
}

macro_rules! regexes {
    ($name:ident: $($regex:expr => $replacement:expr,)*) => {lazy_static::lazy_static! {
        static ref $name: Vec<(Match, &'static [u8])> = vec![
            $((Regex::new($regex).unwrap().into(), $replacement.as_bytes()),)*
        ];
    }};
}

regexes! {
    STDOUT:
    // Windows file paths
    r"\\"                           => "/",
    // erase borrow tags
    "<[0-9]+>"                      => "<TAG>",
    "<[0-9]+="                      => "<TAG=",
}

regexes! {
    STDERR:
    // erase line and column info
    r"\.rs:[0-9]+:[0-9]+(: [0-9]+:[0-9]+)?" => ".rs:LL:CC",
    // erase alloc ids
    "alloc[0-9]+"                    => "ALLOC",
    // erase borrow tags
    "<[0-9]+>"                       => "<TAG>",
    "<[0-9]+="                       => "<TAG=",
    // erase whitespace that differs between platforms
    r" +at (.*\.rs)"                 => " at $1",
    // erase generics in backtraces
    "([0-9]+: .*)::<.*>"             => "$1",
    // erase addresses in backtraces
    "([0-9]+: ) +0x[0-9a-f]+ - (.*)" => "$1$2",
    // erase long hexadecimals
    r"0x[0-9a-fA-F]+[0-9a-fA-F]{2,2}" => "$$HEX",
    // erase specific alignments
    "alignment [0-9]+"               => "alignment ALIGN",
    // erase thread caller ids
    r"call [0-9]+"                  => "call ID",
    // erase platform module paths
    "sys::[a-z]+::"                  => "sys::PLATFORM::",
    // Windows file paths
    r"\\"                           => "/",
    // erase Rust stdlib path
    "[^ `]*/(rust[^/]*|checkout)/library/" => "RUSTLIB/",
    // erase platform file paths
    "sys/[a-z]+/"                    => "sys/PLATFORM/",
    // erase paths into the crate registry
    r"[^ ]*/\.?cargo/registry/.*/(.*\.rs)"  => "CARGO_REGISTRY/.../$1",
}

enum Dependencies {
    WithDependencies,
    WithoutDependencies,
}

use Dependencies::*;

fn ui(mode: Mode, path: &str, target: &str, with_dependencies: Dependencies) -> Result<()> {
    let msg = format!("## Running ui tests in {path} against miri for {target}");
    eprintln!("{}", msg.green().bold());

    let with_dependencies = match with_dependencies {
        WithDependencies => true,
        WithoutDependencies => false,
    };
    run_tests(mode, path, target, with_dependencies)
}

fn get_target() -> String {
    env::var("MIRI_TEST_TARGET").ok().unwrap_or_else(get_host)
}

fn main() -> Result<()> {
    ui_test::color_eyre::install()?;

    let target = get_target();

    let mut args = std::env::args_os();

    // Skip the program name and check whether this is a `./miri run-dep` invocation
    if let Some(first) = args.nth(1) {
        if first == "--miri-run-dep-mode" {
            return run_dep_mode(target, args);
        }
    }

    // Add a test env var to do environment communication tests.
    env::set_var("MIRI_ENV_VAR_TEST", "0");
    // Let the tests know where to store temp files (they might run for a different target, which can make this hard to find).
    env::set_var("MIRI_TEMP", env::temp_dir());

    ui(Mode::Pass, "tests/pass", &target, WithoutDependencies)?;
    ui(Mode::Pass, "tests/pass-dep", &target, WithDependencies)?;
    ui(Mode::Panic, "tests/panic", &target, WithDependencies)?;
    ui(Mode::Fail { require_patterns: true }, "tests/fail", &target, WithDependencies)?;
    if cfg!(target_os = "linux") {
        ui(Mode::Pass, "tests/extern-so/pass", &target, WithoutDependencies)?;
        ui(
            Mode::Fail { require_patterns: true },
            "tests/extern-so/fail",
            &target,
            WithoutDependencies,
        )?;
    }

    Ok(())
}

fn run_dep_mode(target: String, mut args: impl Iterator<Item = OsString>) -> Result<()> {
    let path = args.next().expect("./miri run-dep must be followed by a file name");
    let mut config = test_config(&target, "", Mode::Yolo, /* with dependencies */ true);
    config.program.args.clear(); // We want to give the user full control over flags
    config.build_dependencies_and_link_them()?;

    let mut cmd = config.program.build(&config.out_dir);

    cmd.arg(path);

    cmd.args(args);
    if cmd.spawn()?.wait()?.success() { Ok(()) } else { std::process::exit(1) }
}
