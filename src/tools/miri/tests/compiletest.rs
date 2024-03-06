use colored::*;
use regex::bytes::Regex;
use std::ffi::OsString;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::{env, process::Command};
use ui_test::{color_eyre::Result, Config, Match, Mode, OutputConflictHandling};
use ui_test::{status_emitter, CommandBuilder, Format, RustfixMode};

fn miri_path() -> PathBuf {
    PathBuf::from(option_env!("MIRI").unwrap_or(env!("CARGO_BIN_EXE_miri")))
}

fn get_host() -> String {
    rustc_version::VersionMeta::for_command(std::process::Command::new(miri_path()))
        .expect("failed to parse rustc version info")
        .host
}

pub fn flagsplit(flags: &str) -> Vec<String> {
    // This code is taken from `RUSTFLAGS` handling in cargo.
    flags.split(' ').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect()
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
    program.args.push("-Ainternal_features".into());
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

    // Add a test env var to do environment communication tests.
    program.envs.push(("MIRI_ENV_VAR_TEST".into(), Some("0".into())));
    // Let the tests know where to store temp files (they might run for a different target, which can make this hard to find).
    program.envs.push(("MIRI_TEMP".into(), Some(env::temp_dir().into())));

    let mut config = Config {
        target: Some(target.to_owned()),
        stderr_filters: STDERR.clone(),
        stdout_filters: STDOUT.clone(),
        mode,
        program,
        out_dir: PathBuf::from(std::env::var_os("CARGO_TARGET_DIR").unwrap()).join("ui"),
        edition: Some("2021".into()), // keep in sync with `./miri run`
        threads: std::env::var("MIRI_TEST_THREADS")
            .ok()
            .map(|threads| NonZeroUsize::new(threads.parse().unwrap()).unwrap()),
        ..Config::rustc(path)
    };

    if with_dependencies {
        // Set the `cargo-miri` binary, which we expect to be in the same folder as the `miri` binary.
        // (It's a separate crate, so we don't get an env var from cargo.)
        let mut prog = miri_path();
        prog.set_file_name("cargo-miri");
        config.dependency_builder.program = prog;
        let builder_args = ["miri", "run"]; // There is no `cargo miri build` so we just use `cargo miri run`.
        config.dependency_builder.args = builder_args.into_iter().map(Into::into).collect();
        config.dependencies_crate_manifest_path =
            Some(Path::new("test_dependencies").join("Cargo.toml"));
        // Reset `RUSTFLAGS` to work around <https://github.com/rust-lang/rust/pull/119574#issuecomment-1876878344>.
        config.dependency_builder.envs.push(("RUSTFLAGS".into(), None));
    }
    config
}

fn run_tests(mode: Mode, path: &str, target: &str, with_dependencies: bool) -> Result<()> {
    let mut config = test_config(target, path, mode, with_dependencies);

    // Handle command-line arguments.
    let args = ui_test::Args::test()?;
    let default_bless = env::var_os("RUSTC_BLESS").is_some_and(|v| v != "0");
    config.with_args(&args, default_bless);
    if let OutputConflictHandling::Error(msg) = &mut config.output_conflict_handling {
        *msg = "./miri test --bless".into();
    }
    if env::var_os("MIRI_SKIP_UI_CHECKS").is_some() {
        assert!(!default_bless, "cannot use RUSTC_BLESS and MIRI_SKIP_UI_CHECKS at the same time");
        config.output_conflict_handling = OutputConflictHandling::Ignore;
    }
    eprintln!("   Compiler: {}", config.program.display());
    ui_test::run_tests_generic(
        // Only run one test suite. In the future we can add all test suites to one `Vec` and run
        // them all at once, making best use of systems with high parallelism.
        vec![config],
        // The files we're actually interested in (all `.rs` files).
        ui_test::default_file_filter,
        // This could be used to overwrite the `Config` on a per-test basis.
        |_, _, _| {},
        (
            match args.format {
                Format::Terse => status_emitter::Text::quiet(),
                Format::Pretty => status_emitter::Text::verbose(),
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
    // erase thread ids
    r"unnamed-[0-9]+"               => "unnamed-ID",
    // erase borrow tags
    "<[0-9]+>"                       => "<TAG>",
    "<[0-9]+="                       => "<TAG=",
    // normalize width of Tree Borrows diagnostic borders (which otherwise leak borrow tag info)
    "(─{50})─+"                      => "$1",
    // erase whitespace that differs between platforms
    r" +at (.*\.rs)"                 => " at $1",
    // erase generics in backtraces
    "([0-9]+: .*)::<.*>"             => "$1",
    // erase long hexadecimals
    r"0x[0-9a-fA-F]+[0-9a-fA-F]{2,2}" => "$$HEX",
    // erase specific alignments
    "alignment [0-9]+"               => "alignment ALIGN",
    "[0-9]+ byte alignment but found [0-9]+" => "ALIGN byte alignment but found ALIGN",
    // erase thread caller ids
    r"call [0-9]+"                  => "call ID",
    // erase platform module paths
    "sys::pal::[a-z]+::"                  => "sys::pal::PLATFORM::",
    // Windows file paths
    r"\\"                           => "/",
    // erase Rust stdlib path
    "[^ \n`]*/(rust[^/]*|checkout)/library/" => "RUSTLIB/",
    // erase platform file paths
    "sys/pal/[a-z]+/"                    => "sys/pal/PLATFORM/",
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

    ui(Mode::Pass, "tests/pass", &target, WithoutDependencies)?;
    ui(Mode::Pass, "tests/pass-dep", &target, WithDependencies)?;
    ui(Mode::Panic, "tests/panic", &target, WithDependencies)?;
    ui(
        Mode::Fail { require_patterns: true, rustfix: RustfixMode::Disabled },
        "tests/fail",
        &target,
        WithoutDependencies,
    )?;
    ui(
        Mode::Fail { require_patterns: true, rustfix: RustfixMode::Disabled },
        "tests/fail-dep",
        &target,
        WithDependencies,
    )?;
    if cfg!(target_os = "linux") {
        ui(Mode::Pass, "tests/extern-so/pass", &target, WithoutDependencies)?;
        ui(
            Mode::Fail { require_patterns: true, rustfix: RustfixMode::Disabled },
            "tests/extern-so/fail",
            &target,
            WithoutDependencies,
        )?;
    }

    Ok(())
}

fn run_dep_mode(target: String, mut args: impl Iterator<Item = OsString>) -> Result<()> {
    let path = args.next().expect("./miri run-dep must be followed by a file name");
    let mut config = test_config(
        &target,
        "",
        Mode::Yolo { rustfix: RustfixMode::Disabled },
        /* with dependencies */ true,
    );
    config.program.args.clear(); // We want to give the user full control over flags
    let dep_args = config.build_dependencies()?;

    let mut cmd = config.program.build(&config.out_dir);
    cmd.args(dep_args);

    cmd.arg(path);

    cmd.args(args);
    if cmd.spawn()?.wait()?.success() { Ok(()) } else { std::process::exit(1) }
}
