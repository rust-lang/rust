#![feature(test)]

use compiletest_rs as compiletest;
use compiletest_rs::common::Mode as TestMode;

use std::env::{self, set_var};
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

mod cargo;

#[must_use]
fn rustc_test_suite() -> Option<PathBuf> {
    option_env!("RUSTC_TEST_SUITE").map(PathBuf::from)
}

#[must_use]
fn rustc_lib_path() -> PathBuf {
    option_env!("RUSTC_LIB_PATH").unwrap().into()
}

fn default_config() -> compiletest::Config {
    let build_info = cargo::BuildInfo::new();
    let mut config = compiletest::Config::default();

    if let Ok(name) = env::var("TESTNAME") {
        config.filter = Some(name);
    }

    if rustc_test_suite().is_some() {
        let path = rustc_lib_path();
        config.run_lib_path = path.clone();
        config.compile_lib_path = path;
    }

    let disambiguated: Vec<_> = cargo::BuildInfo::third_party_crates()
        .iter()
        .map(|(krate, path)| format!("--extern {}={}", krate, path.display()))
        .collect();

    config.target_rustcflags = Some(format!(
        "-L {0} -L {1} -Dwarnings -Zui-testing {2}",
        build_info.host_lib().join("deps").display(),
        build_info.target_lib().join("deps").display(),
        disambiguated.join(" ")
    ));

    config.build_base = if rustc_test_suite().is_some() {
        // we don't need access to the stderr files on travis
        let mut path = PathBuf::from(env!("OUT_DIR"));
        path.push("test_build_base");
        path
    } else {
        build_info.host_lib().join("test_build_base")
    };
    config.rustc_path = build_info.clippy_driver_path();
    config
}

fn run_mode(cfg: &mut compiletest::Config) {
    cfg.mode = TestMode::Ui;
    cfg.src_base = Path::new("tests").join("ui");
    compiletest::run_tests(&cfg);
}

#[allow(clippy::identity_conversion)]
fn run_ui_toml_tests(config: &compiletest::Config, mut tests: Vec<tester::TestDescAndFn>) -> Result<bool, io::Error> {
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

fn run_ui_toml(config: &mut compiletest::Config) {
    config.mode = TestMode::Ui;
    config.src_base = Path::new("tests").join("ui-toml").canonicalize().unwrap();

    let tests = compiletest::make_tests(&config);

    let res = run_ui_toml_tests(&config, tests);
    match res {
        Ok(true) => {},
        Ok(false) => panic!("Some tests failed"),
        Err(e) => {
            println!("I/O failure during tests: {:?}", e);
        },
    }
}

fn prepare_env() {
    set_var("CLIPPY_DISABLE_DOCS_LINKS", "true");
    set_var("CLIPPY_TESTS", "true");
    //set_var("RUST_BACKTRACE", "0");
}

#[test]
fn compile_test() {
    prepare_env();
    let mut config = default_config();
    run_mode(&mut config);
    run_ui_toml(&mut config);
}
