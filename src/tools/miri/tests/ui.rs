#![allow(clippy::let_and_return)]
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::{env, fmt};

use colored::*;
use regex::bytes::Regex;
use ui_test::build_manager::BuildManager;
use ui_test::color_eyre::eyre::{Context, Result};
use ui_test::custom_flags::Flag;
use ui_test::custom_flags::edition::Edition;
use ui_test::dependencies::DependencyBuilder;
use ui_test::per_test_config::TestConfig;
use ui_test::spanned::Spanned;
use ui_test::status_emitter::StatusEmitter;
use ui_test::{CommandBuilder, Config, Match, ignore_output_conflict};

#[derive(Copy, Clone, Debug)]
enum Mode {
    Pass {
        native: bool,
    },
    /// Requires annotations
    Fail,
    /// Not used for tests, but for `miri run --dep`
    RunDep {
        native: bool,
    },
    /// Test must panic.
    Panic,
}

impl Mode {
    fn native(self) -> bool {
        matches!(self, Mode::Pass { native: true } | Mode::RunDep { native: true })
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mode::Pass { native: false } => write!(f, "pass"),
            Mode::Pass { native: true } => write!(f, "pass-native"),
            Mode::Fail => write!(f, "fail"),
            Mode::Panic => write!(f, "panic"),
            Mode::RunDep { .. } => unreachable!(),
        }
    }
}

fn miri_path() -> PathBuf {
    env!("CARGO_BIN_EXE_miri").into()
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
    let native_lib_path = so_target_dir.join("native-lib-tests.so");
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
            "tests/native-lib/fn_ptr.c",
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

/// Does *not* set args or (most) env vars, since it is shared between the test runner and
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
    if mode.native() {
        // This means we build the program instead of running Miri.
        program.envs.push(("MIRI_BE_RUSTC".into(), Some("host".into())));
        // Use the right linker, if necessary. We use the `CC_*` variable as that is set by CI and
        // unlike `CARGO_TARGET_*_LINKER` it does not require upper-casing the target.
        if let Ok(linker) = env::var(format!("CC_{target}")) {
            program.args.push(format!("-Clinker={linker}").into());
        }
    }

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

    // Register custom comments.
    config.custom_comments.insert("run-native", |parser, _args, span| {
        // Just remember that this is present.
        parser.set_custom_once("run-native", (), span);
    });

    // Adjust comment defaults.
    config.comment_defaults.base().exit_status = match mode {
        Mode::Pass { .. } => Some(0),
        Mode::Fail => Some(1),
        Mode::RunDep { .. } => None,
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

    // Building dependencies is also a "comment default".
    if let Some(WithDependencies { bless }) = with_dependencies {
        let crate_manifest_path = Path::new("tests/deps").join("Cargo.toml");
        let dep_builder = if mode.native() {
            DependencyBuilder { crate_manifest_path, ..Default::default() }
        } else {
            DependencyBuilder {
                program: CommandBuilder {
                    // Set the `cargo-miri` binary, which we expect to be in the same folder as the `miri` binary.
                    // (It's a separate crate, so we don't get an env var from cargo.)
                    program: miri_path()
                        .with_file_name(format!("cargo-miri{}", env::consts::EXE_SUFFIX)),
                    // Add `-Zbinary-dep-depinfo` since it is needed for bootstrap builds (and doesn't harm otherwise).
                    args: ["miri", "build", "-Zbinary-dep-depinfo"]
                        .into_iter()
                        .map(Into::into)
                        .collect(),
                    envs: vec![
                        // Reset `RUSTFLAGS`/`CARGO_ENCODED_RUSTFLAGS` to work around <https://github.com/rust-lang/rust/pull/119574#issuecomment-1876878344>.
                        ("RUSTFLAGS".into(), None),
                        ("CARGO_ENCODED_RUSTFLAGS".into(), None),
                        // Reset `MIRIFLAGS` because it caused trouble in the past and should not be needed.
                        ("MIRIFLAGS".into(), None),
                        // Allow `cargo miri build`.
                        ("MIRI_BUILD_TEST_DEPS".into(), Some("1".into())),
                    ],
                    ..CommandBuilder::cargo()
                },
                crate_manifest_path,
                build_std: None,
                bless_lockfile: bless,
            }
        };
        config.comment_defaults.base().set_custom("dependencies", dep_builder);
    }

    // We only want this for actual test runs, not native run-dep mode.
    if matches!(mode, Mode::Pass { native: true }) {
        // Overwrite "compile-flags" so that it does nothing.
        // FIXME: make it just skip `-Zmiri` flags.
        config.custom_comments.insert("compile-flags", |_parser, _args, _span| {});

        // Add a default comment that interprets our custom `run-native` comment.
        #[derive(Debug)]
        struct NativeRunner;
        config.comment_defaults.base().set_custom("native-runner", NativeRunner);

        impl Flag for NativeRunner {
            fn clone_inner(&self) -> Box<dyn Flag> {
                Box::new(NativeRunner)
            }
            fn must_be_unique(&self) -> bool {
                true
            }

            fn test_condition(
                &self,
                _config: &Config,
                comments: &ui_test::Comments,
                revision: &str,
            ) -> bool {
                let should_run = comments
                    .for_revision(revision)
                    .any(|r| r.custom.iter().any(|(k, _v)| *k == "run-native"));
                // We return `true` when the test should be ignored.
                let ignore = !should_run;
                ignore
            }

            fn post_test_action(
                &self,
                config: &TestConfig,
                output: &std::process::Output,
                build_manager: &BuildManager,
            ) -> Result<(), ui_test::Errored> {
                // Delegate to the native run support.
                use ui_test::custom_flags::run::Run;
                Run::post_test_action(
                    &Run { exit_code: 0, output_conflict_handling: None },
                    config,
                    output,
                    build_manager,
                )
            }
        }
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
    if mode.native() {
        config.output_conflict_handling = ignore_output_conflict;
    }

    // Add a test env var to do environment communication tests.
    config.program.envs.push(("MIRI_ENV_VAR_TEST".into(), Some("0".into())));
    // Let the tests know where to store temp files (they might run for a different target, which can make this hard to find).
    config.program.envs.push(("MIRI_TEMP".into(), Some(tmpdir.to_owned().into())));
    // If a test ICEs, we want to see a backtrace.
    config.program.envs.push(("RUST_BACKTRACE".into(), Some("1".into())));

    // Add rustc/Miri flags.
    config.program.args.push("-Dwarnings".into());
    config.program.args.push("-Dunused".into());
    config.program.args.push("-Ainternal_features".into());
    config.program.args.push("-Zui-testing".into());
    if !mode.native() {
        config.program.args.push(
            format!(
                "--sysroot={}",
                env::var("MIRI_SYSROOT")
                    .expect("MIRI_SYSROOT must be set to run the ui test suite")
            )
            .into(),
        );
        if let Ok(extra_flags) = env::var("MIRIFLAGS") {
            for flag in extra_flags.split_whitespace() {
                config.program.args.push(flag.into());
            }
        }
        config.program.args.push("-Zjit-mode".into());
    }

    // If we're testing the native-lib functionality, then build the shared object file for testing
    // external C function calls and push the relevant compiler flag.
    if path.starts_with("tests/native-lib/") {
        let native_lib = build_native_lib(target);
        let mut flag = std::ffi::OsString::from("-Zmiri-native-lib=");
        flag.push(native_lib.into_os_string());
        config.program.args.push(flag);
    }
    // For GenMC tests, add the relevant flags.
    if path.starts_with("tests/genmc/") {
        config.program.args.push("-Zmiri-genmc".into());
        // FIXME(genmc): remove this when GenMC and SB can be used together.
        config.program.args.push("-Zmiri-disable-stacked-borrows".into());
    }

    println!("   Compiler: {}", config.program.display());
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
    // remove time print from GenMC estimation mode output.
    "\nExpected verification time: .* ± .*" => "\nExpected verification time: [MEAN] ± [SD]",
}

enum Dependencies {
    WithDeps,
    WithoutDeps,
}

use Dependencies::*;

fn ui(
    mode: Mode,
    path: &str,
    target: &str,
    with_dependencies: Dependencies,
    tmpdir: &Path,
) -> Result<()> {
    let msg = format!("## Running {mode} ui tests in {path} for {target}");
    println!("{}", msg.green().bold());

    let with_dependencies = match with_dependencies {
        WithDeps => true,
        WithoutDeps => false,
    };
    run_tests(mode, path, target, with_dependencies, tmpdir)
        .with_context(|| format!("{mode} ui tests in {path} for {target} failed"))
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

    // Check whether this is a `./miri run` invocation
    if let Ok(mode) = env::var("MIRI_RUN_MODE") {
        return run_with_deps(target, mode);
    }

    ui(Mode::Pass { native: false }, "tests/pass", &target, WithoutDeps, tmpdir.path())?;
    ui(Mode::Pass { native: false }, "tests/pass-dep", &target, WithDeps, tmpdir.path())?;
    if target == host {
        ui(Mode::Pass { native: true }, "tests/pass", &target, WithoutDeps, tmpdir.path())?;
        ui(Mode::Pass { native: true }, "tests/pass-dep", &target, WithDeps, tmpdir.path())?;
    }
    ui(Mode::Panic, "tests/panic", &target, WithDeps, tmpdir.path())?;
    ui(Mode::Fail, "tests/fail", &target, WithoutDeps, tmpdir.path())?;
    ui(Mode::Fail, "tests/fail-dep", &target, WithDeps, tmpdir.path())?;
    if cfg!(all(unix, feature = "native-lib")) && target == host {
        ui(
            Mode::Pass { native: false },
            "tests/native-lib/pass",
            &target,
            WithoutDeps,
            tmpdir.path(),
        )?;
        ui(Mode::Fail, "tests/native-lib/fail", &target, WithoutDeps, tmpdir.path())?;
    }

    // We only enable GenMC tests when the `genmc` feature is enabled, but also only on platforms we support:
    // FIXME(genmc,cross-platform): Technically we do support cross-target execution as long as the
    // target is also 64bit little-endian, so `host == target` is too strict.
    if cfg!(all(
        feature = "genmc",
        target_os = "linux",
        target_pointer_width = "64",
        target_endian = "little"
    )) && host == target
    {
        ui(Mode::Pass { native: false }, "tests/genmc/pass", &target, WithDeps, tmpdir.path())?;
        ui(Mode::Fail, "tests/genmc/fail", &target, WithDeps, tmpdir.path())?;
    }

    Ok(())
}

fn run_with_deps(target: String, mode: String) -> Result<()> {
    let native = mode == "native";

    let mut config =
        miri_config(&target, "", Mode::RunDep { native }, Some(WithDependencies { bless: false }));
    config.comment_defaults.base().custom.remove("edition"); // `./miri` adds an `--edition` in `args`, so don't set it twice
    config.fill_host_and_target()?;
    // Reset `args` (otherwise we'll get JSON output).
    config.program.args = vec![];

    // Compute the actual Miri invocation command.
    let test_config = TestConfig::one_off_runner(config.clone(), PathBuf::new());
    let mut cmd = test_config.config.program.build(&test_config.config.out_dir);
    // We are not using `test_config.build_command` (as that would require us to know the filename
    // we are invoking), so we need to set the target ourselves.
    cmd.arg("--target").arg(&target);
    // Also forward arguments to the program (skipping the binary name).
    // We don't put this in the `config` since we don't want it to affect the dependency build.
    cmd.args({
        let mut args = env::args_os();
        args.next().unwrap();
        args
    });

    // Build dependencies (which will mutate that command)
    test_config
        .apply_custom(&mut cmd, &BuildManager::one_off(config.clone()))
        .expect("failed to build dependencies");
    // Finally, actually run Miri.
    let exit_status = cmd.spawn()?.wait()?;
    if !exit_status.success() {
        std::process::exit(1)
    }

    if native {
        // We just built the program, we still have to run it. We can't use the ui_test `Run` flag
        // as (a) that always captures the output, and (b) that needs an actual BuildManager, not
        // just the one-off stub we have here. So we implement the core logic ourselves.

        // First, figure out the output binary by re-running the compiler with `--print`.
        cmd.arg("--print").arg("file-names");
        let output = cmd.output()?;
        let exe = std::str::from_utf8(&output.stdout).unwrap().trim();
        let exe = config.out_dir.join(exe);
        // Then run that binary.
        let mut cmd = Command::new(exe);
        let exit_status = cmd.spawn()?.wait()?;
        if !exit_status.success() {
            std::process::exit(1)
        }
    }

    Ok(())
}
