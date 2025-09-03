use std::env;
use std::ffi::OsString;
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use colored::*;
use regex::bytes::Regex;
use ui_test::build_manager::BuildManager;
use ui_test::color_eyre::eyre::{Context, Result};
use ui_test::custom_flags::edition::Edition;
use ui_test::dependencies::DependencyBuilder;
use ui_test::per_test_config::TestConfig;
use ui_test::spanned::Spanned;
use ui_test::status_emitter::StatusEmitter;
use ui_test::{CommandBuilder, Config, Match, ignore_output_conflict};

#[derive(Copy, Clone, Debug)]
enum Mode {
    Pass,
    /// Requires annotations
    Fail,
    /// Not used for tests, but for `miri run --dep`
    RunDep,
    Panic,
}

fn miri_path() -> PathBuf {
    PathBuf::from(env::var("MIRI").unwrap_or_else(|_| env!("CARGO_BIN_EXE_miri").into()))
}

pub fn flagsplit(flags: &str) -> Vec<String> {
    // This code is taken from `RUSTFLAGS` handling in cargo.
    flags.split(' ').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect()
}

// Build the shared object file for testing native function calls.
fn build_native_lib(target: &str) -> PathBuf {
    // Loosely follow the logic of the `cc` crate for finding the compiler.
    let cc = env::var(format!("CC_{target}"))
        .or_else(|_| env::var("CC"))
        .unwrap_or_else(|_| "cc".into());
    // Target directory that we can write to.
    let so_target_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("miri-native-lib");
    // Create the directory if it does not already exist.
    std::fs::create_dir_all(&so_target_dir)
        .expect("Failed to create directory for shared object file");
    // We use a platform-neutral file extension to avoid having to hard-code alternatives.
    let native_lib_path = so_target_dir.join("native-lib.module");
    let cc_output = Command::new(cc)
        .args([
            "-shared",
            "-fPIC",
            // We hide all symbols by default and export just the ones we need
            // This is to future-proof against a more complex shared object which eg defines its own malloc
            // (but we wouldn't want miri to call that, as it would if it was exported).
            "-fvisibility=hidden",
            "-o",
            native_lib_path.to_str().unwrap(),
            // FIXME: Automate gathering of all relevant C source files in the directory.
            "tests/native-lib/scalar_arguments.c",
            "tests/native-lib/aggregate_arguments.c",
            "tests/native-lib/ptr_read_access.c",
            "tests/native-lib/ptr_write_access.c",
            // Ensure we notice serious problems in the C code.
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
        ])
        .output()
        .expect("failed to generate shared object file for testing native function calls");
    if !cc_output.status.success() {
        panic!(
            "error generating shared object file for testing native function calls:\n{}",
            String::from_utf8_lossy(&cc_output.stderr),
        );
    }
    native_lib_path
}

struct WithDependencies {
    bless: bool,
}

/// Does *not* set any args or env vars, since it is shared between the test runner and
/// run_dep_mode.
fn miri_config(
    target: &str,
    path: &str,
    mode: Mode,
    with_dependencies: Option<WithDependencies>,
) -> Config {
    // Miri is rustc-like, so we create a default builder for rustc and modify it
    let mut program = CommandBuilder::rustc();
    program.program = miri_path();

    let mut config = Config {
        target: Some(target.to_owned()),
        program,
        // When changing this, remember to also adjust the logic in bootstrap, in Miri's test step,
        // that deletes the `miri_ui` dir when it needs a rebuild.
        out_dir: PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("miri_ui"),
        threads: std::env::var("MIRI_TEST_THREADS")
            .ok()
            .map(|threads| NonZero::new(threads.parse().unwrap()).unwrap()),
        ..Config::rustc(path)
    };

    config.comment_defaults.base().exit_status = match mode {
        Mode::Pass => Some(0),
        Mode::Fail => Some(1),
        Mode::RunDep => None,
        Mode::Panic => Some(101),
    }
    .map(Spanned::dummy)
    .into();

    config.comment_defaults.base().require_annotations =
        Spanned::dummy(matches!(mode, Mode::Fail)).into();

    config.comment_defaults.base().normalize_stderr =
        stderr_filters().iter().map(|(m, p)| (m.clone(), p.to_vec())).collect();
    config.comment_defaults.base().normalize_stdout =
        stdout_filters().iter().map(|(m, p)| (m.clone(), p.to_vec())).collect();

    // keep in sync with `./miri run`
    config.comment_defaults.base().add_custom("edition", Edition("2021".into()));

    if let Some(WithDependencies { bless }) = with_dependencies {
        config.comment_defaults.base().set_custom(
            "dependencies",
            DependencyBuilder {
                program: CommandBuilder {
                    // Set the `cargo-miri` binary, which we expect to be in the same folder as the `miri` binary.
                    // (It's a separate crate, so we don't get an env var from cargo.)
                    program: miri_path()
                        .with_file_name(format!("cargo-miri{}", env::consts::EXE_SUFFIX)),
                    // There is no `cargo miri build` so we just use `cargo miri run`.
                    args: ["miri", "run"].into_iter().map(Into::into).collect(),
                    // Reset `RUSTFLAGS` to work around <https://github.com/rust-lang/rust/pull/119574#issuecomment-1876878344>.
                    envs: vec![("RUSTFLAGS".into(), None)],
                    ..CommandBuilder::cargo()
                },
                crate_manifest_path: Path::new("tests/deps").join("Cargo.toml"),
                build_std: None,
                bless_lockfile: bless,
            },
        );
    }
    config
}

fn run_tests(
    mode: Mode,
    path: &str,
    target: &str,
    with_dependencies: bool,
    tmpdir: &Path,
) -> Result<()> {
    // Handle command-line arguments.
    let mut args = ui_test::Args::test()?;
    args.bless |= env::var_os("RUSTC_BLESS").is_some_and(|v| v != "0");

    let with_dependencies = with_dependencies.then_some(WithDependencies { bless: args.bless });

    let mut config = miri_config(target, path, mode, with_dependencies);
    config.with_args(&args);
    config.bless_command = Some("./miri test --bless".into());

    if env::var_os("MIRI_SKIP_UI_CHECKS").is_some() {
        assert!(!args.bless, "cannot use RUSTC_BLESS and MIRI_SKIP_UI_CHECKS at the same time");
        config.output_conflict_handling = ignore_output_conflict;
    }

    // Add a test env var to do environment communication tests.
    config.program.envs.push(("MIRI_ENV_VAR_TEST".into(), Some("0".into())));
    // Let the tests know where to store temp files (they might run for a different target, which can make this hard to find).
    config.program.envs.push(("MIRI_TEMP".into(), Some(tmpdir.to_owned().into())));
    // If a test ICEs, we want to see a backtrace.
    config.program.envs.push(("RUST_BACKTRACE".into(), Some("1".into())));

    // Add some flags we always want.
    config.program.args.push(
        format!(
            "--sysroot={}",
            env::var("MIRI_SYSROOT").expect("MIRI_SYSROOT must be set to run the ui test suite")
        )
        .into(),
    );
    config.program.args.push("-Dwarnings".into());
    config.program.args.push("-Dunused".into());
    config.program.args.push("-Ainternal_features".into());
    if let Ok(extra_flags) = env::var("MIRIFLAGS") {
        for flag in extra_flags.split_whitespace() {
            config.program.args.push(flag.into());
        }
    }
    config.program.args.push("-Zui-testing".into());

    // If we're testing the native-lib functionality, then build the shared object file for testing
    // external C function calls and push the relevant compiler flag.
    if path.starts_with("tests/native-lib/") {
        let native_lib = build_native_lib(target);
        let mut flag = std::ffi::OsString::from("-Zmiri-native-lib=");
        flag.push(native_lib.into_os_string());
        config.program.args.push(flag);
    }

    eprintln!("   Compiler: {}", config.program.display());
    ui_test::run_tests_generic(
        // Only run one test suite. In the future we can add all test suites to one `Vec` and run
        // them all at once, making best use of systems with high parallelism.
        vec![config],
        // The files we're actually interested in (all `.rs` files).
        ui_test::default_file_filter,
        // This could be used to overwrite the `Config` on a per-test basis.
        |_, _| {},
        // No GHA output as that would also show in the main rustc repo.
        Box::<dyn StatusEmitter>::from(args.format),
    )
}

macro_rules! regexes {
    ($name:ident: $($regex:expr => $replacement:expr,)*) => {
        fn $name() -> &'static [(Match, &'static [u8])] {
            static S: OnceLock<Vec<(Match, &'static [u8])>> = OnceLock::new();
            S.get_or_init(|| vec![
                $((Regex::new($regex).unwrap().into(), $replacement.as_bytes()),)*
            ])
        }
    };
}

regexes! {
    stdout_filters:
    // Windows file paths
    r"\\"                           => "/",
    // erase borrow tags
    "<[0-9]+>"                      => "<TAG>",
    "<[0-9]+="                      => "<TAG=",
}

regexes! {
    stderr_filters:
    // erase line and column info
    r"\.rs:[0-9]+:[0-9]+(: [0-9]+:[0-9]+)?" => ".rs:LL:CC",
    // erase alloc ids
    "alloc[0-9]+"                    => "ALLOC",
    // erase thread ids
    r"unnamed-[0-9]+"                => "unnamed-ID",
    r"thread '(?P<name>.*?)' \(\d+\) panicked" => "thread '$name' ($$TID) panicked",
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
    r"\bsys::([a-z_]+)::[a-z]+::"   => "sys::$1::PLATFORM::",
    // Windows file paths
    r"\\"                           => "/",
    // erase Rust stdlib path
    "[^ \n`]*/(rust[^/]*|checkout)/library/" => "RUSTLIB/",
    // erase platform file paths
    r"\bsys/([a-z_]+)/[a-z]+\b"     => "sys/$1/PLATFORM",
    // erase paths into the crate registry
    r"[^ ]*/\.?cargo/registry/.*/(.*\.rs)"  => "CARGO_REGISTRY/.../$1",
}

enum Dependencies {
    WithDependencies,
    WithoutDependencies,
}

use Dependencies::*;

fn ui(
    mode: Mode,
    path: &str,
    target: &str,
    with_dependencies: Dependencies,
    tmpdir: &Path,
) -> Result<()> {
    let msg = format!("## Running ui tests in {path} for {target}");
    eprintln!("{}", msg.green().bold());

    let with_dependencies = match with_dependencies {
        WithDependencies => true,
        WithoutDependencies => false,
    };
    run_tests(mode, path, target, with_dependencies, tmpdir)
        .with_context(|| format!("ui tests in {path} for {target} failed"))
}

fn get_host() -> String {
    rustc_version::VersionMeta::for_command(std::process::Command::new(miri_path()))
        .expect("failed to parse rustc version info")
        .host
}

fn get_target(host: &str) -> String {
    env::var("MIRI_TEST_TARGET").ok().unwrap_or_else(|| host.to_owned())
}

fn main() -> Result<()> {
    ui_test::color_eyre::install()?;

    let host = get_host();
    let target = get_target(&host);
    let tmpdir = tempfile::Builder::new().prefix("miri-uitest-").tempdir()?;

    let mut args = std::env::args_os();

    // Skip the program name and check whether this is a `./miri run-dep` invocation
    if let Some(first) = args.nth(1)
        && first == "--miri-run-dep-mode"
    {
        return run_dep_mode(target, args);
    }

    ui(Mode::Pass, "tests/pass", &target, WithoutDependencies, tmpdir.path())?;
    ui(Mode::Pass, "tests/pass-dep", &target, WithDependencies, tmpdir.path())?;
    ui(Mode::Panic, "tests/panic", &target, WithDependencies, tmpdir.path())?;
    ui(Mode::Fail, "tests/fail", &target, WithoutDependencies, tmpdir.path())?;
    ui(Mode::Fail, "tests/fail-dep", &target, WithDependencies, tmpdir.path())?;
    if cfg!(all(unix, feature = "native-lib")) && target == host {
        ui(Mode::Pass, "tests/native-lib/pass", &target, WithoutDependencies, tmpdir.path())?;
        ui(Mode::Fail, "tests/native-lib/fail", &target, WithoutDependencies, tmpdir.path())?;
    }

    // We only enable GenMC tests when the `genmc` feature is enabled, but also only on platforms we support:
    // FIXME(genmc,macos): Add `target_os = "macos"` once `https://github.com/dtolnay/cxx/issues/1535` is fixed.
    // FIXME(genmc,cross-platform): remove `host == target` check once cross-platform support with GenMC is possible.
    if cfg!(all(
        feature = "genmc",
        target_os = "linux",
        target_pointer_width = "64",
        target_endian = "little"
    )) && host == target
    {
        ui(Mode::Pass, "tests/genmc/pass", &target, WithDependencies, tmpdir.path())?;
        ui(Mode::Fail, "tests/genmc/fail", &target, WithDependencies, tmpdir.path())?;
    }

    Ok(())
}

fn run_dep_mode(target: String, args: impl Iterator<Item = OsString>) -> Result<()> {
    let mut config =
        miri_config(&target, "", Mode::RunDep, Some(WithDependencies { bless: false }));
    config.comment_defaults.base().custom.remove("edition"); // `./miri` adds an `--edition` in `args`, so don't set it twice
    config.fill_host_and_target()?;
    config.program.args = args.collect();

    let test_config = TestConfig::one_off_runner(config.clone(), PathBuf::new());

    let build_manager = BuildManager::one_off(config);
    let mut cmd = test_config.config.program.build(&test_config.config.out_dir);
    cmd.arg("--target").arg(test_config.config.target.as_ref().unwrap());
    // Build dependencies
    test_config.apply_custom(&mut cmd, &build_manager).unwrap();

    if cmd.spawn()?.wait()?.success() { Ok(()) } else { std::process::exit(1) }
}
