#![feature(test)] // compiletest_rs requires this attribute
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![warn(rust_2018_idioms, unused_lifetimes)]

use compiletest_rs as compiletest;
use compiletest_rs::common::Mode as TestMode;

use std::collections::HashMap;
use std::env::{self, remove_var, set_var, var_os};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

mod cargo;

// whether to run internal tests or not
const RUN_INTERNAL_TESTS: bool = cfg!(feature = "internal-lints");

/// All crates used in UI tests are listed here
static TEST_DEPENDENCIES: &[&str] = &[
    "clippy_utils",
    "derive_new",
    "if_chain",
    "itertools",
    "quote",
    "regex",
    "serde",
    "serde_derive",
    "syn",
];

// Test dependencies may need an `extern crate` here to ensure that they show up
// in the depinfo file (otherwise cargo thinks they are unused)
#[allow(unused_extern_crates)]
extern crate clippy_utils;
#[allow(unused_extern_crates)]
extern crate derive_new;
#[allow(unused_extern_crates)]
extern crate if_chain;
#[allow(unused_extern_crates)]
extern crate itertools;
#[allow(unused_extern_crates)]
extern crate quote;
#[allow(unused_extern_crates)]
extern crate syn;

/// Produces a string with an `--extern` flag for all UI test crate
/// dependencies.
///
/// The dependency files are located by parsing the depinfo file for this test
/// module. This assumes the `-Z binary-dep-depinfo` flag is enabled. All test
/// dependencies must be added to Cargo.toml at the project root. Test
/// dependencies that are not *directly* used by this test module require an
/// `extern crate` declaration.
fn extern_flags() -> String {
    let current_exe_depinfo = {
        let mut path = env::current_exe().unwrap();
        path.set_extension("d");
        std::fs::read_to_string(path).unwrap()
    };
    let mut crates: HashMap<&str, &str> = HashMap::with_capacity(TEST_DEPENDENCIES.len());
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
        "dependencies not found in depinfo: {:?}\n\
        help: Make sure the `-Z binary-dep-depinfo` rust flag is enabled\n\
        help: Try adding to dev-dependencies in Cargo.toml",
        not_found
    );
    crates
        .into_iter()
        .map(|(name, path)| format!(" --extern {}={}", name, path))
        .collect()
}

fn default_config() -> compiletest::Config {
    let mut config = compiletest::Config {
        edition: Some("2021".into()),
        ..compiletest::Config::default()
    };

    if let Ok(filters) = env::var("TESTNAME") {
        config.filters = filters.split(',').map(std::string::ToString::to_string).collect();
    }

    if let Some(path) = option_env!("RUSTC_LIB_PATH") {
        let path = PathBuf::from(path);
        config.run_lib_path = path.clone();
        config.compile_lib_path = path;
    }
    let current_exe_path = std::env::current_exe().unwrap();
    let deps_path = current_exe_path.parent().unwrap();
    let profile_path = deps_path.parent().unwrap();

    // Using `-L dependency={}` enforces that external dependencies are added with `--extern`.
    // This is valuable because a) it allows us to monitor what external dependencies are used
    // and b) it ensures that conflicting rlibs are resolved properly.
    let host_libs = option_env!("HOST_LIBS")
        .map(|p| format!(" -L dependency={}", Path::new(p).join("deps").display()))
        .unwrap_or_default();
    config.target_rustcflags = Some(format!(
        "--emit=metadata -Dwarnings -Zui-testing -L dependency={}{}{}",
        deps_path.display(),
        host_libs,
        extern_flags(),
    ));

    config.build_base = profile_path.join("test");
    config.rustc_path = profile_path.join(if cfg!(windows) {
        "clippy-driver.exe"
    } else {
        "clippy-driver"
    });
    config
}

fn run_ui(cfg: &mut compiletest::Config) {
    cfg.mode = TestMode::Ui;
    cfg.src_base = Path::new("tests").join("ui");
    // use tests/clippy.toml
    let _g = VarGuard::set("CARGO_MANIFEST_DIR", std::fs::canonicalize("tests").unwrap());
    compiletest::run_tests(cfg);
}

fn run_ui_test(cfg: &mut compiletest::Config) {
    cfg.mode = TestMode::Ui;
    cfg.src_base = Path::new("tests").join("ui_test");
    let _g = VarGuard::set("CARGO_MANIFEST_DIR", std::fs::canonicalize("tests").unwrap());
    let rustcflags = cfg.target_rustcflags.get_or_insert_with(Default::default);
    let len = rustcflags.len();
    rustcflags.push_str(" --test");
    compiletest::run_tests(cfg);
    if let Some(ref mut flags) = &mut cfg.target_rustcflags {
        flags.truncate(len);
    }
}

fn run_internal_tests(cfg: &mut compiletest::Config) {
    // only run internal tests with the internal-tests feature
    if !RUN_INTERNAL_TESTS {
        return;
    }
    cfg.mode = TestMode::Ui;
    cfg.src_base = Path::new("tests").join("ui-internal");
    compiletest::run_tests(cfg);
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
            let _g = VarGuard::set("CARGO_MANIFEST_DIR", &dir_path);
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
                let test_name = compiletest::make_test_name(config, &paths);
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

    let tests = compiletest::make_tests(config);

    let res = run_tests(config, tests);
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
        filters: &[String],
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
            for filter in filters {
                if !dir_path.ends_with(filter) {
                    continue;
                }
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
                    let _g = VarGuard::set("CLIPPY_CONF_DIR", case.path());
                    let paths = compiletest::common::TestPaths {
                        file: file_path,
                        base: config.src_base.clone(),
                        relative_dir: src_path.strip_prefix(&config.src_base).unwrap().into(),
                    };
                    let test_name = compiletest::make_test_name(config, &paths);
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

    let tests = compiletest::make_tests(config);

    let current_dir = env::current_dir().unwrap();
    let res = run_tests(config, &config.filters, tests);
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
    run_ui(&mut config);
    run_ui_test(&mut config);
    run_ui_toml(&mut config);
    run_ui_cargo(&mut config);
    run_internal_tests(&mut config);
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
