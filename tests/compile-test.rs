#![feature(lazy_cell)]
#![feature(is_sorted)]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![warn(rust_2018_idioms, unused_lifetimes)]
#![allow(unused_extern_crates)]

use ui_test::spanned::Spanned;
use ui_test::{status_emitter, Args, CommandBuilder, Config, Match, Mode, OutputConflictHandling};

use std::collections::BTreeMap;
use std::env::{self, set_var, var_os};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use test_utils::IS_RUSTC_TEST_SUITE;

// Test dependencies may need an `extern crate` here to ensure that they show up
// in the depinfo file (otherwise cargo thinks they are unused)
extern crate clippy_lints;
extern crate clippy_utils;
extern crate futures;
extern crate if_chain;
extern crate itertools;
extern crate parking_lot;
extern crate quote;
extern crate syn;
extern crate tokio;

mod test_utils;

/// All crates used in UI tests are listed here
static TEST_DEPENDENCIES: &[&str] = &[
    "clippy_config",
    "clippy_lints",
    "clippy_utils",
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

// whether to run internal tests or not
const RUN_INTERNAL_TESTS: bool = cfg!(feature = "internal");

fn base_config(test_dir: &str) -> (Config, Args) {
    let mut args = Args::test().unwrap();
    args.bless |= var_os("RUSTC_BLESS").is_some_and(|v| v != "0");

    let target_dir = PathBuf::from(var_os("CARGO_TARGET_DIR").unwrap_or_else(|| "target".into()));
    let mut config = Config {
        output_conflict_handling: OutputConflictHandling::Error,
        filter_files: env::var("TESTNAME")
            .map(|filters| filters.split(',').map(str::to_string).collect())
            .unwrap_or_default(),
        target: None,
        bless_command: Some("cargo uibless".into()),
        out_dir: target_dir.join("ui_test"),
        ..Config::rustc(Path::new("tests").join(test_dir))
    };
    config.comment_defaults.base().mode = Some(Spanned::dummy(Mode::Yolo {
        rustfix: ui_test::RustfixMode::Everything,
    }))
    .into();
    config.comment_defaults.base().diagnostic_code_prefix = Some(Spanned::dummy("clippy::".into())).into();
    config.with_args(&args);
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

    config.program.program = profile_path.join(if cfg!(windows) {
        "clippy-driver.exe"
    } else {
        "clippy-driver"
    });
    (config, args)
}

fn run_ui() {
    let (mut config, args) = base_config("ui");
    config
        .program
        .envs
        .push(("CLIPPY_CONF_DIR".into(), Some("tests".into())));

    ui_test::run_tests_generic(
        vec![config],
        ui_test::default_file_filter,
        ui_test::default_per_file_config,
        status_emitter::Text::from(args.format),
    )
    .unwrap();
}

fn run_internal_tests() {
    // only run internal tests with the internal-tests feature
    if !RUN_INTERNAL_TESTS {
        return;
    }
    let (mut config, args) = base_config("ui-internal");
    config.bless_command = Some("cargo uitest --features internal -- -- --bless".into());

    ui_test::run_tests_generic(
        vec![config],
        ui_test::default_file_filter,
        ui_test::default_per_file_config,
        status_emitter::Text::from(args.format),
    )
    .unwrap();
}

fn run_ui_toml() {
    let (mut config, args) = base_config("ui-toml");

    config
        .comment_defaults
        .base()
        .normalize_stderr
        .push((Match::from(env::current_dir().unwrap().as_path()), b"$DIR".into()));

    ui_test::run_tests_generic(
        vec![config],
        ui_test::default_file_filter,
        |config, path, _file_contents| {
            config
                .program
                .envs
                .push(("CLIPPY_CONF_DIR".into(), Some(path.parent().unwrap().into())));
        },
        status_emitter::Text::from(args.format),
    )
    .unwrap();
}

// Allow `Default::default` as `OptWithSpan` is not nameable
#[allow(clippy::default_trait_access)]
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
    config.comment_defaults.base().edition = Default::default();

    config
        .comment_defaults
        .base()
        .normalize_stderr
        .push((Match::from(env::current_dir().unwrap().as_path()), b"$DIR".into()));

    let ignored_32bit = |path: &Path| {
        // FIXME: for some reason the modules are linted in a different order for this test
        cfg!(target_pointer_width = "32") && path.ends_with("tests/ui-cargo/module_style/fail_mod/Cargo.toml")
    };

    ui_test::run_tests_generic(
        vec![config],
        |path, config| {
            path.ends_with("Cargo.toml")
                .then(|| ui_test::default_any_file_filter(path, config) && !ignored_32bit(path))
        },
        |_config, _path, _file_contents| {},
        status_emitter::Text::from(args.format),
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
        println!(
            "average {} time: {} millis.",
            speedtest.to_uppercase(),
            sum / u128::from(iterations)
        );
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
