use colored::*;
use regex::bytes::Regex;
use std::path::{Path, PathBuf};
use std::{env, process::Command};
use ui_test::status_emitter::StatusEmitter;
use ui_test::CommandBuilder;
use ui_test::{color_eyre::Result, Config, Match, Mode, OutputConflictHandling};

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

fn run_tests(mode: Mode, path: &str, target: &str, with_dependencies: bool) -> Result<()> {
    // Miri is rustc-like, so we create a default builder for rustc and modify it
    let mut program = CommandBuilder::rustc();
    program.program = miri_path();

    let in_rustc_test_suite = option_env!("RUSTC_STAGE").is_some();

    // Add some flags we always want.
    if in_rustc_test_suite {
        // Less aggressive warnings to make the rustc toolstate management less painful.
        // (We often get warnings when e.g. a feature gets stabilized or some lint gets added/improved.)
        program.args.push("-Astable-features".into());
        program.args.push("-Aunused".into());
    } else {
        program.args.push("-Dwarnings".into());
        program.args.push("-Dunused".into());
    }
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
        (false, false) => OutputConflictHandling::Error,
        (true, false) => OutputConflictHandling::Bless,
        (false, true) => OutputConflictHandling::Ignore,
        (true, true) => panic!("cannot use MIRI_BLESS and MIRI_SKIP_UI_CHECKS at the same time"),
    };

    let mut config = Config {
        target: Some(target.to_owned()),
        stderr_filters: STDERR.clone(),
        stdout_filters: STDOUT.clone(),
        root_dir: PathBuf::from(path),
        mode,
        program,
        output_conflict_handling,
        quiet: false,
        edition: Some("2021".into()),
        ..Config::default()
    };

    // Handle command-line arguments.
    let mut after_dashdash = false;
    config.path_filter.extend(std::env::args().skip(1).filter(|arg| {
        if after_dashdash {
            // Just propagate everything.
            return true;
        }
        match &**arg {
            "--quiet" => {
                config.quiet = true;
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
    }));

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
    ui_test::run_tests_generic(
        config,
        // The files we're actually interested in (all `.rs` files).
        |path| path.extension().is_some_and(|ext| ext == "rs"),
        // This could be used to overwrite the `Config` on a per-test basis.
        |_, _| None,
        TextAndGha,
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

/// This is a custom renderer for `ui_test` output that does not emit github actions
/// `group`s, while still producing regular github actions messages on test failures.
struct TextAndGha;
impl StatusEmitter for TextAndGha {
    fn failed_test<'a>(
        &'a self,
        revision: &'a str,
        path: &'a Path,
        cmd: &'a Command,
        stderr: &'a [u8],
    ) -> Box<dyn std::fmt::Debug + 'a> {
        Box::new((
            ui_test::status_emitter::Gha::<false>.failed_test(revision, path, cmd, stderr),
            ui_test::status_emitter::Text.failed_test(revision, path, cmd, stderr),
        ))
    }

    fn run_tests(&self, _config: &Config) -> Box<dyn ui_test::status_emitter::DuringTestRun> {
        Box::new(TextAndGha)
    }

    fn finalize(
        &self,
        failures: usize,
        succeeded: usize,
        ignored: usize,
        filtered: usize,
    ) -> Box<dyn ui_test::status_emitter::Summary> {
        Box::new((
            ui_test::status_emitter::Gha::<false>.finalize(failures, succeeded, ignored, filtered),
            ui_test::status_emitter::Text.finalize(failures, succeeded, ignored, filtered),
        ))
    }
}

impl ui_test::status_emitter::DuringTestRun for TextAndGha {
    fn test_result(&mut self, path: &Path, revision: &str, result: &ui_test::TestResult) {
        ui_test::status_emitter::Text.test_result(path, revision, result);
        ui_test::status_emitter::Gha::<false>.test_result(path, revision, result);
    }
}
