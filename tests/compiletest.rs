use colored::*;
use regex::Regex;
use std::path::{Path, PathBuf};
use std::{env, ffi::OsString};
use ui_test::{color_eyre::Result, Config, DependencyBuilder, Mode, OutputConflictHandling};

fn miri_path() -> PathBuf {
    PathBuf::from(option_env!("MIRI").unwrap_or(env!("CARGO_BIN_EXE_miri")))
}

fn run_tests(mode: Mode, path: &str, target: Option<String>) -> Result<()> {
    let in_rustc_test_suite = option_env!("RUSTC_STAGE").is_some();

    // Add some flags we always want.
    let mut flags: Vec<OsString> = Vec::new();
    flags.push("--edition".into());
    flags.push("2018".into());
    if in_rustc_test_suite {
        // Less aggressive warnings to make the rustc toolstate management less painful.
        // (We often get warnings when e.g. a feature gets stabilized or some lint gets added/improved.)
        flags.push("-Astable-features".into());
        flags.push("-Aunused".into());
    } else {
        flags.push("-Dwarnings".into());
        flags.push("-Dunused".into());
    }
    if let Some(sysroot) = env::var_os("MIRI_SYSROOT") {
        flags.push("--sysroot".into());
        flags.push(sysroot);
    }
    if let Ok(extra_flags) = env::var("MIRIFLAGS") {
        for flag in extra_flags.split_whitespace() {
            flags.push(flag.into());
        }
    }
    flags.push("-Zui-testing".into());
    if let Some(target) = &target {
        flags.push("--target".into());
        flags.push(target.into());
    }

    let skip_ui_checks = env::var_os("MIRI_SKIP_UI_CHECKS").is_some();

    let output_conflict_handling = match (env::var_os("MIRI_BLESS").is_some(), skip_ui_checks) {
        (false, false) => OutputConflictHandling::Error,
        (true, false) => OutputConflictHandling::Bless,
        (false, true) => OutputConflictHandling::Ignore,
        (true, true) => panic!("cannot use MIRI_BLESS and MIRI_SKIP_UI_CHECKS at the same time"),
    };

    // Pass on all arguments as filters.
    let path_filter = std::env::args().skip(1);

    let use_std = env::var_os("MIRI_NO_STD").is_none();

    let config = Config {
        args: flags,
        target,
        stderr_filters: STDERR.clone(),
        stdout_filters: STDOUT.clone(),
        root_dir: PathBuf::from(path),
        mode,
        path_filter: path_filter.collect(),
        program: miri_path(),
        output_conflict_handling,
        dependencies_crate_manifest_path: use_std
            .then(|| Path::new("test_dependencies").join("Cargo.toml")),
        dependency_builder: Some(DependencyBuilder {
            program: std::env::var_os("CARGO").unwrap().into(),
            args: vec![
                "run".into(),
                "--manifest-path".into(),
                "cargo-miri/Cargo.toml".into(),
                "--".into(),
                "miri".into(),
            ],
            envs: vec![],
        }),
    };
    ui_test::run_tests(config)
}

macro_rules! regexes {
    ($name:ident: $($regex:expr => $replacement:expr,)*) => {lazy_static::lazy_static! {
        static ref $name: Vec<(Regex, &'static str)> = vec![
            $((Regex::new($regex).unwrap(), $replacement),)*
        ];
    }};
}

regexes! {
    STDOUT:
    // Windows file paths
    r"\\"                           => "/",
}

regexes! {
    STDERR:
    // erase line and column info
    r"\.rs:[0-9]+:[0-9]+(: [0-9]+:[0-9]+)?" => ".rs:LL:CC",
    // erase alloc ids
    "alloc[0-9]+"                    => "ALLOC",
    // erase Stacked Borrows tags
    "<[0-9]+>"                       => "<TAG>",
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
    r"[^ ]*/\.cargo/registry/.*/(.*\.rs)"  => "CARGO_REGISTRY/$1",
}

fn ui(mode: Mode, path: &str) -> Result<()> {
    let target = get_target();

    let msg = format!(
        "## Running ui tests in {path} against miri for {}",
        target.as_deref().unwrap_or("host")
    );
    eprintln!("{}", msg.green().bold());

    run_tests(mode, path, target)
}

fn get_target() -> Option<String> {
    env::var("MIRI_TEST_TARGET").ok()
}

fn main() -> Result<()> {
    ui_test::color_eyre::install()?;

    // Add a test env var to do environment communication tests.
    env::set_var("MIRI_ENV_VAR_TEST", "0");
    // Let the tests know where to store temp files (they might run for a different target, which can make this hard to find).
    env::set_var("MIRI_TEMP", env::temp_dir());

    ui(Mode::Pass, "tests/pass")?;
    ui(Mode::Panic, "tests/panic")?;
    ui(Mode::Fail, "tests/fail")?;

    Ok(())
}
