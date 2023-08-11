#![feature(test)] // compiletest_rs requires this attribute
#![feature(lazy_cell)]
#![feature(is_sorted)]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![warn(rust_2018_idioms, unused_lifetimes)]
#![allow(unused_extern_crates)]

use compiletest::{status_emitter, Args, CommandBuilder, OutputConflictHandling};
use ui_test as compiletest;
use ui_test::Mode as TestMode;

use std::collections::BTreeMap;
use std::env::{self, remove_var, set_var, var_os};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use test_utils::IS_RUSTC_TEST_SUITE;

// Test dependencies may need an `extern crate` here to ensure that they show up
// in the depinfo file (otherwise cargo thinks they are unused)
extern crate clippy_lints;
extern crate clippy_utils;
extern crate derive_new;
extern crate futures;
extern crate if_chain;
extern crate itertools;
extern crate parking_lot;
extern crate quote;
extern crate syn;
extern crate tokio;

/// All crates used in UI tests are listed here
static TEST_DEPENDENCIES: &[&str] = &[
    "clippy_lints",
    "clippy_utils",
    "derive_new",
    "futures",
    "if_chain",
    "itertools",
    "parking_lot",
    "quote",
    "regex",
    "serde_derive",
    "serde",
    "syn",
    "tokio",
];

/// Produces a string with an `--extern` flag for all UI test crate
/// dependencies.
///
/// The dependency files are located by parsing the depinfo file for this test
/// module. This assumes the `-Z binary-dep-depinfo` flag is enabled. All test
/// dependencies must be added to Cargo.toml at the project root. Test
/// dependencies that are not *directly* used by this test module require an
/// `extern crate` declaration.
static EXTERN_FLAGS: LazyLock<Vec<String>> = LazyLock::new(|| {
    let current_exe_depinfo = {
        let mut path = env::current_exe().unwrap();
        path.set_extension("d");
        fs::read_to_string(path).unwrap()
    };
    let mut crates = BTreeMap::<&str, &str>::new();
    for line in current_exe_depinfo.lines() {
        // each dependency is expected to have a Makefile rule like `/path/to/crate-hash.rlib:`
        let parse_name_path = || {
            if line.starts_with(char::is_whitespace) {
                return None;
            }
            let path_str = line.strip_suffix(':')?;
            let path = Path::new(path_str);
            if !matches!(path.extension()?.to_str()?, "rlib" | "so" | "dylib" | "dll") {
                return None;
            }
            let (name, _hash) = path.file_stem()?.to_str()?.rsplit_once('-')?;
            // the "lib" prefix is not present for dll files
            let name = name.strip_prefix("lib").unwrap_or(name);
            Some((name, path_str))
        };
        if let Some((name, path)) = parse_name_path() {
            if TEST_DEPENDENCIES.contains(&name) {
                // A dependency may be listed twice if it is available in sysroot,
                // and the sysroot dependencies are listed first. As of the writing,
                // this only seems to apply to if_chain.
                crates.insert(name, path);
            }
        }
    }
    let not_found: Vec<&str> = TEST_DEPENDENCIES
        .iter()
        .copied()
        .filter(|n| !crates.contains_key(n))
        .collect();
    assert!(
        not_found.is_empty(),
        "dependencies not found in depinfo: {not_found:?}\n\
        help: Make sure the `-Z binary-dep-depinfo` rust flag is enabled\n\
        help: Try adding to dev-dependencies in Cargo.toml\n\
        help: Be sure to also add `extern crate ...;` to tests/compile-test.rs",
    );
    crates
        .into_iter()
        .map(|(name, path)| format!("--extern={name}={path}"))
        .collect()
});

mod test_utils;

// whether to run internal tests or not
const RUN_INTERNAL_TESTS: bool = cfg!(feature = "internal");

fn canonicalize(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    fs::create_dir_all(path).unwrap();
    fs::canonicalize(path).unwrap_or_else(|err| panic!("{} cannot be canonicalized: {err}", path.display()))
}

fn base_config(test_dir: &str) -> (compiletest::Config, Args) {
    let args = Args::test().unwrap();
    let mut config = compiletest::Config {
        mode: TestMode::Yolo { rustfix: true },
        stderr_filters: vec![],
        stdout_filters: vec![],
        output_conflict_handling: if var_os("GITHUB_ACTION").is_none()
            && (var_os("RUSTC_BLESS").is_some_and(|v| v != "0") || !args.check)
        {
            OutputConflictHandling::Bless
        } else {
            OutputConflictHandling::Error("cargo uibless".into())
        },
        target: None,
        out_dir: canonicalize(
            std::env::var_os("CARGO_TARGET_DIR")
                .map_or_else(|| std::env::current_dir().unwrap().join("target"), PathBuf::from),
        )
        .join("ui_test"),
        ..compiletest::Config::rustc(Path::new("tests").join(test_dir))
    };
    let current_exe_path = env::current_exe().unwrap();
    let deps_path = current_exe_path.parent().unwrap();
    let profile_path = deps_path.parent().unwrap();

    config.program.args.extend(
        [
            "--emit=metadata",
            "-Aunused",
            "-Ainternal_features",
            "-Zui-testing",
            "-Dwarnings",
            &format!("-Ldependency={}", deps_path.display()),
        ]
        .map(OsString::from),
    );

    config.program.args.extend(EXTERN_FLAGS.iter().map(OsString::from));

    if let Some(host_libs) = option_env!("HOST_LIBS") {
        let dep = format!("-Ldependency={}", Path::new(host_libs).join("deps").display());
        config.program.args.push(dep.into());
    }

    // Normalize away slashes in windows paths.
    config.stderr_filter(r"\\", "/");

    //config.build_base = profile_path.join("test").join(test_dir);
    config.program.program = profile_path.join(if cfg!(windows) {
        "clippy-driver.exe"
    } else {
        "clippy-driver"
    });
    (config, args)
}

fn test_filter() -> Box<dyn Sync + Fn(&Path) -> bool> {
    if let Ok(filters) = env::var("TESTNAME") {
        let filters: Vec<_> = filters.split(',').map(ToString::to_string).collect();
        Box::new(move |path| filters.iter().any(|f| path.to_string_lossy().contains(f)))
    } else {
        Box::new(|_| true)
    }
}

fn run_ui() {
    let (config, args) = base_config("ui");
    //config.rustfix_coverage = true;
    // use tests/clippy.toml
    let _g = VarGuard::set("CARGO_MANIFEST_DIR", canonicalize("tests"));
    let _threads = VarGuard::set(
        "RUST_TEST_THREADS",
        // if RUST_TEST_THREADS is set, adhere to it, otherwise override it
        env::var("RUST_TEST_THREADS").unwrap_or_else(|_| {
            std::thread::available_parallelism()
                .map_or(1, std::num::NonZeroUsize::get)
                .to_string()
        }),
    );

    let test_filter = test_filter();

    let quiet = args.quiet;

    compiletest::run_tests_generic(
        vec![config],
        args,
        move |path, args, config| compiletest::default_file_filter(path, args, config) && test_filter(path),
        compiletest::default_per_file_config,
        if quiet {
            status_emitter::Text::quiet()
        } else {
            status_emitter::Text::verbose()
        },
    )
    .unwrap();
}

fn run_internal_tests() {
    // only run internal tests with the internal-tests feature
    if !RUN_INTERNAL_TESTS {
        return;
    }
    let (mut config, args) = base_config("ui-internal");
    if let OutputConflictHandling::Error(err) = &mut config.output_conflict_handling {
        *err = "cargo uitest --features internal".into();
    }
    let test_filter = test_filter();
    let quiet = args.quiet;

    compiletest::run_tests_generic(
        vec![config],
        args,
        move |path, args, config| compiletest::default_file_filter(path, args, config) && test_filter(path),
        compiletest::default_per_file_config,
        if quiet {
            status_emitter::Text::quiet()
        } else {
            status_emitter::Text::verbose()
        },
    )
    .unwrap();
}

fn run_ui_toml() {
    let (mut config, args) = base_config("ui-toml");

    config.stderr_filter(
        &regex::escape(
            &canonicalize("tests")
                .parent()
                .unwrap()
                .display()
                .to_string()
                .replace('\\', "/"),
        ),
        "$$DIR",
    );

    let test_filter = test_filter();
    let quiet = args.quiet;

    ui_test::run_tests_generic(
        vec![config],
        args,
        |path, args, config| compiletest::default_file_filter(path, args, config) && test_filter(path),
        |config, path, _file_contents| {
            config
                .program
                .envs
                .push(("CLIPPY_CONF_DIR".into(), Some(path.parent().unwrap().into())));
        },
        if quiet {
            status_emitter::Text::quiet()
        } else {
            status_emitter::Text::verbose()
        },
    )
    .unwrap();
}

fn run_ui_cargo() {
    if IS_RUSTC_TEST_SUITE {
        return;
    }

    let (mut config, args) = base_config("ui-cargo");
    config.program.input_file_flag = CommandBuilder::cargo().input_file_flag;
    config.program.out_dir_flag = CommandBuilder::cargo().out_dir_flag;
    config.program.args = vec!["clippy".into(), "--color".into(), "never".into(), "--quiet".into()];
    config
        .program
        .envs
        .push(("RUSTFLAGS".into(), Some("-Dwarnings".into())));
    // We need to do this while we still have a rustc in the `program` field.
    config.fill_host_and_target().unwrap();
    config.dependencies_crate_manifest_path = None;
    config.program.program.set_file_name(if cfg!(windows) {
        "cargo-clippy.exe"
    } else {
        "cargo-clippy"
    });
    config.edition = None;

    config.stderr_filter(
        &regex::escape(
            &canonicalize("tests")
                .parent()
                .unwrap()
                .display()
                .to_string()
                .replace('\\', "/"),
        ),
        "$$DIR",
    );

    let test_filter = test_filter();
    let quiet = args.quiet;

    ui_test::run_tests_generic(
        vec![config],
        args,
        |path, _args, _config| test_filter(path) && path.ends_with("Cargo.toml"),
        |config, path, _file_contents| {
            config.out_dir = canonicalize(
                std::env::current_dir()
                    .unwrap()
                    .join("target")
                    .join("ui_test_cargo/")
                    .join(path.parent().unwrap()),
            );
        },
        if quiet {
            status_emitter::Text::quiet()
        } else {
            status_emitter::Text::verbose()
        },
    )
    .unwrap();
}

fn main() {
    // Support being run by cargo nextest - https://nexte.st/book/custom-test-harnesses.html
    if env::args().any(|arg| arg == "--list") {
        if !env::args().any(|arg| arg == "--ignored") {
            println!("compile_test: test");
        }

        return;
    }

    set_var("CLIPPY_DISABLE_DOCS_LINKS", "true");
    // The SPEEDTEST_* env variables can be used to check Clippy's performance on your PR. It runs the
    // affected test 1000 times and gets the average.
    if let Ok(speedtest) = std::env::var("SPEEDTEST") {
        println!("----------- STARTING SPEEDTEST -----------");
        let f = match speedtest.as_str() {
            "ui" => run_ui as fn(),
            "cargo" => run_ui_cargo as fn(),
            "toml" => run_ui_toml as fn(),
            "internal" => run_internal_tests as fn(),
            "ui-cargo-toml-metadata" => ui_cargo_toml_metadata as fn(),

            _ => panic!("unknown speedtest: {speedtest} || accepted speedtests are: [ui, cargo, toml, internal]"),
        };

        let iterations;
        if let Ok(iterations_str) = std::env::var("SPEEDTEST_ITERATIONS") {
            iterations = iterations_str
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("Couldn't parse `{iterations_str}`, please use a valid u64"));
        } else {
            iterations = 1000;
        }

        let mut sum = 0;
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            f();
            sum += start.elapsed().as_millis();
        }
        println!("average {} time: {} millis.", speedtest.to_uppercase(), sum / 1000);
    } else {
        run_ui();
        run_ui_toml();
        run_ui_cargo();
        run_internal_tests();
        ui_cargo_toml_metadata();
    }
}

fn ui_cargo_toml_metadata() {
    let ui_cargo_path = Path::new("tests/ui-cargo");
    let cargo_common_metadata_path = ui_cargo_path.join("cargo_common_metadata");
    let publish_exceptions =
        ["fail_publish", "fail_publish_true", "pass_publish_empty"].map(|path| cargo_common_metadata_path.join(path));

    for entry in walkdir::WalkDir::new(ui_cargo_path) {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.file_name() != Some(OsStr::new("Cargo.toml")) {
            continue;
        }

        let toml = fs::read_to_string(path).unwrap().parse::<toml::Value>().unwrap();

        let package = toml.as_table().unwrap().get("package").unwrap().as_table().unwrap();

        let name = package.get("name").unwrap().as_str().unwrap().replace('-', "_");
        assert!(
            path.parent()
                .unwrap()
                .components()
                .map(|component| component.as_os_str().to_string_lossy().replace('-', "_"))
                .any(|s| *s == name)
                || path.starts_with(&cargo_common_metadata_path),
            "{path:?} has incorrect package name"
        );

        let publish = package.get("publish").and_then(toml::Value::as_bool).unwrap_or(true);
        assert!(
            !publish || publish_exceptions.contains(&path.parent().unwrap().to_path_buf()),
            "{path:?} lacks `publish = false`"
        );
    }
}

/// Restores an env var on drop
#[must_use]
struct VarGuard {
    key: &'static str,
    value: Option<OsString>,
}

impl VarGuard {
    fn set(key: &'static str, val: impl AsRef<OsStr>) -> Self {
        let value = var_os(key);
        set_var(key, val);
        Self { key, value }
    }
}

impl Drop for VarGuard {
    fn drop(&mut self) {
        match self.value.as_deref() {
            None => remove_var(self.key),
            Some(value) => set_var(self.key, value),
        }
    }
}
