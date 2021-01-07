#![feature(test)] // compiletest_rs requires this attribute
#![feature(once_cell)]

use compiletest_rs as compiletest;
use compiletest_rs::common::Mode as TestMode;

use std::env::{self, set_var, var};
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

mod cargo;

// whether to run internal tests or not
const RUN_INTERNAL_TESTS: bool = cfg!(feature = "internal-lints");

fn host_lib() -> PathBuf {
    option_env!("HOST_LIBS").map_or(cargo::CARGO_TARGET_DIR.join(env!("PROFILE")), PathBuf::from)
}

fn clippy_driver_path() -> PathBuf {
    option_env!("CLIPPY_DRIVER_PATH").map_or(cargo::TARGET_LIB.join("clippy-driver"), PathBuf::from)
}

// When we'll want to use `extern crate ..` for a dependency that is used
// both by the crate and the compiler itself, we can't simply pass -L flags
// as we'll get a duplicate matching versions. Instead, disambiguate with
// `--extern dep=path`.
// See https://github.com/rust-lang/rust-clippy/issues/4015.
//
// FIXME: We cannot use `cargo build --message-format=json` to resolve to dependency files.
//        Because it would force-rebuild if the options passed to `build` command is not the same
//        as what we manually pass to `cargo` invocation
fn third_party_crates() -> String {
    use std::collections::HashMap;
    static CRATES: &[&str] = &["serde", "serde_derive", "regex", "clippy_lints", "syn", "quote"];
    let dep_dir = cargo::TARGET_LIB.join("deps");
    let mut crates: HashMap<&str, PathBuf> = HashMap::with_capacity(CRATES.len());
    for entry in fs::read_dir(dep_dir).unwrap() {
        let path = match entry {
            Ok(entry) => entry.path(),
            Err(_) => continue,
        };
        if let Some(name) = path.file_name().and_then(OsStr::to_str) {
            for dep in CRATES {
                if name.starts_with(&format!("lib{}-", dep)) && name.ends_with(".rlib") {
                    if let Some(old) = crates.insert(dep, path.clone()) {
                        panic!("Found multiple rlibs for crate `{}`: `{:?}` and `{:?}", dep, old, path);
                    }
                    break;
                }
            }
        }
    }

    let v: Vec<_> = crates
        .into_iter()
        .map(|(dep, path)| format!("--extern {}={}", dep, path.display()))
        .collect();
    v.join(" ")
}

fn default_config() -> compiletest::Config {
    let mut config = compiletest::Config::default();

    if let Ok(name) = env::var("TESTNAME") {
        config.filter = Some(name);
    }

    if let Some(path) = option_env!("RUSTC_LIB_PATH") {
        let path = PathBuf::from(path);
        config.run_lib_path = path.clone();
        config.compile_lib_path = path;
    }

    config.target_rustcflags = Some(format!(
        "--emit=metadata -L {0} -L {1} -Dwarnings -Zui-testing {2}",
        host_lib().join("deps").display(),
        cargo::TARGET_LIB.join("deps").display(),
        third_party_crates(),
    ));

    config.build_base = if cargo::is_rustc_test_suite() {
        // This make the stderr files go to clippy OUT_DIR on rustc repo build dir
        let mut path = PathBuf::from(env!("OUT_DIR"));
        path.push("test_build_base");
        path
    } else {
        host_lib().join("test_build_base")
    };
    config.rustc_path = clippy_driver_path();
    config
}

fn run_mode(cfg: &mut compiletest::Config) {
    cfg.mode = TestMode::Ui;
    cfg.src_base = Path::new("tests").join("ui");
    compiletest::run_tests(&cfg);
}

fn run_internal_tests(cfg: &mut compiletest::Config) {
    // only run internal tests with the internal-tests feature
    if !RUN_INTERNAL_TESTS {
        return;
    }
    cfg.mode = TestMode::Ui;
    cfg.src_base = Path::new("tests").join("ui-internal");
    compiletest::run_tests(&cfg);
}

fn run_ui_toml(config: &mut compiletest::Config) {
    fn run_tests(config: &compiletest::Config, mut tests: Vec<tester::TestDescAndFn>) -> Result<bool, io::Error> {
        let mut result = true;
        let opts = compiletest::test_opts(config);
        for dir in fs::read_dir(&config.src_base)? {
            let dir = dir?;
            if !dir.file_type()?.is_dir() {
                continue;
            }
            let dir_path = dir.path();
            set_var("CARGO_MANIFEST_DIR", &dir_path);
            for file in fs::read_dir(&dir_path)? {
                let file = file?;
                let file_path = file.path();
                if file.file_type()?.is_dir() {
                    continue;
                }
                if file_path.extension() != Some(OsStr::new("rs")) {
                    continue;
                }
                let paths = compiletest::common::TestPaths {
                    file: file_path,
                    base: config.src_base.clone(),
                    relative_dir: dir_path.file_name().unwrap().into(),
                };
                let test_name = compiletest::make_test_name(&config, &paths);
                let index = tests
                    .iter()
                    .position(|test| test.desc.name == test_name)
                    .expect("The test should be in there");
                result &= tester::run_tests_console(&opts, vec![tests.swap_remove(index)])?;
            }
        }
        Ok(result)
    }

    config.mode = TestMode::Ui;
    config.src_base = Path::new("tests").join("ui-toml").canonicalize().unwrap();

    let tests = compiletest::make_tests(&config);

    let manifest_dir = var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let res = run_tests(&config, tests);
    set_var("CARGO_MANIFEST_DIR", &manifest_dir);
    match res {
        Ok(true) => {},
        Ok(false) => panic!("Some tests failed"),
        Err(e) => {
            panic!("I/O failure during tests: {:?}", e);
        },
    }
}

fn run_ui_cargo(config: &mut compiletest::Config) {
    fn run_tests(
        config: &compiletest::Config,
        filter: &Option<String>,
        mut tests: Vec<tester::TestDescAndFn>,
    ) -> Result<bool, io::Error> {
        let mut result = true;
        let opts = compiletest::test_opts(config);

        for dir in fs::read_dir(&config.src_base)? {
            let dir = dir?;
            if !dir.file_type()?.is_dir() {
                continue;
            }

            // Use the filter if provided
            let dir_path = dir.path();
            match &filter {
                Some(name) if !dir_path.ends_with(name) => continue,
                _ => {},
            }

            for case in fs::read_dir(&dir_path)? {
                let case = case?;
                if !case.file_type()?.is_dir() {
                    continue;
                }

                let src_path = case.path().join("src");

                // When switching between branches, if the previous branch had a test
                // that the current branch does not have, the directory is not removed
                // because an ignored Cargo.lock file exists.
                if !src_path.exists() {
                    continue;
                }

                env::set_current_dir(&src_path)?;
                for file in fs::read_dir(&src_path)? {
                    let file = file?;
                    if file.file_type()?.is_dir() {
                        continue;
                    }

                    // Search for the main file to avoid running a test for each file in the project
                    let file_path = file.path();
                    match file_path.file_name().and_then(OsStr::to_str) {
                        Some("main.rs") => {},
                        _ => continue,
                    }
                    let paths = compiletest::common::TestPaths {
                        file: file_path,
                        base: config.src_base.clone(),
                        relative_dir: src_path.strip_prefix(&config.src_base).unwrap().into(),
                    };
                    let test_name = compiletest::make_test_name(&config, &paths);
                    let index = tests
                        .iter()
                        .position(|test| test.desc.name == test_name)
                        .expect("The test should be in there");
                    result &= tester::run_tests_console(&opts, vec![tests.swap_remove(index)])?;
                }
            }
        }
        Ok(result)
    }

    if cargo::is_rustc_test_suite() {
        return;
    }

    config.mode = TestMode::Ui;
    config.src_base = Path::new("tests").join("ui-cargo").canonicalize().unwrap();

    let tests = compiletest::make_tests(&config);

    let current_dir = env::current_dir().unwrap();
    let filter = env::var("TESTNAME").ok();
    let res = run_tests(&config, &filter, tests);
    env::set_current_dir(current_dir).unwrap();

    match res {
        Ok(true) => {},
        Ok(false) => panic!("Some tests failed"),
        Err(e) => {
            panic!("I/O failure during tests: {:?}", e);
        },
    }
}

fn prepare_env() {
    set_var("CLIPPY_DISABLE_DOCS_LINKS", "true");
    set_var("__CLIPPY_INTERNAL_TESTS", "true");
    //set_var("RUST_BACKTRACE", "0");
}

#[test]
fn compile_test() {
    prepare_env();
    let mut config = default_config();
    run_mode(&mut config);
    run_ui_toml(&mut config);
    run_ui_cargo(&mut config);
    run_internal_tests(&mut config);
}
