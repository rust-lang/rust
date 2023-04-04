#![feature(test)] // compiletest_rs requires this attribute
#![feature(lazy_cell)]
#![feature(is_sorted)]
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
use std::sync::LazyLock;
use test_utils::IS_RUSTC_TEST_SUITE;

mod test_utils;

// whether to run internal tests or not
const RUN_INTERNAL_TESTS: bool = cfg!(feature = "internal");

/// All crates used in UI tests are listed here
static TEST_DEPENDENCIES: &[&str] = &[
    "clippy_lints",
    "clippy_utils",
    "derive_new",
    "futures",
    "if_chain",
    "itertools",
    "quote",
    "regex",
    "serde",
    "serde_derive",
    "syn",
    "tokio",
    "parking_lot",
    "rustc_semver",
];

// Test dependencies may need an `extern crate` here to ensure that they show up
// in the depinfo file (otherwise cargo thinks they are unused)
#[allow(unused_extern_crates)]
extern crate clippy_lints;
#[allow(unused_extern_crates)]
extern crate clippy_utils;
#[allow(unused_extern_crates)]
extern crate derive_new;
#[allow(unused_extern_crates)]
extern crate futures;
#[allow(unused_extern_crates)]
extern crate if_chain;
#[allow(unused_extern_crates)]
extern crate itertools;
#[allow(unused_extern_crates)]
extern crate parking_lot;
#[allow(unused_extern_crates)]
extern crate quote;
#[allow(unused_extern_crates)]
extern crate rustc_semver;
#[allow(unused_extern_crates)]
extern crate syn;
#[allow(unused_extern_crates)]
extern crate tokio;

/// Produces a string with an `--extern` flag for all UI test crate
/// dependencies.
///
/// The dependency files are located by parsing the depinfo file for this test
/// module. This assumes the `-Z binary-dep-depinfo` flag is enabled. All test
/// dependencies must be added to Cargo.toml at the project root. Test
/// dependencies that are not *directly* used by this test module require an
/// `extern crate` declaration.
static EXTERN_FLAGS: LazyLock<String> = LazyLock::new(|| {
    let current_exe_depinfo = {
        let mut path = env::current_exe().unwrap();
        path.set_extension("d");
        fs::read_to_string(path).unwrap()
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
        "dependencies not found in depinfo: {not_found:?}\n\
        help: Make sure the `-Z binary-dep-depinfo` rust flag is enabled\n\
        help: Try adding to dev-dependencies in Cargo.toml\n\
        help: Be sure to also add `extern crate ...;` to tests/compile-test.rs",
    );
    crates
        .into_iter()
        .map(|(name, path)| format!(" --extern {name}={path}"))
        .collect()
});

fn base_config(test_dir: &str) -> compiletest::Config {
    let mut config = compiletest::Config {
        edition: Some("2021".into()),
        mode: TestMode::Ui,
        ..Default::default()
    };

    if let Ok(filters) = env::var("TESTNAME") {
        config.filters = filters.split(',').map(ToString::to_string).collect();
    }

    if let Some(path) = option_env!("RUSTC_LIB_PATH") {
        let path = PathBuf::from(path);
        config.run_lib_path = path.clone();
        config.compile_lib_path = path;
    }
    let current_exe_path = env::current_exe().unwrap();
    let deps_path = current_exe_path.parent().unwrap();
    let profile_path = deps_path.parent().unwrap();

    // Using `-L dependency={}` enforces that external dependencies are added with `--extern`.
    // This is valuable because a) it allows us to monitor what external dependencies are used
    // and b) it ensures that conflicting rlibs are resolved properly.
    let host_libs = option_env!("HOST_LIBS")
        .map(|p| format!(" -L dependency={}", Path::new(p).join("deps").display()))
        .unwrap_or_default();
    config.target_rustcflags = Some(format!(
        "--emit=metadata -Dwarnings -Zui-testing -L dependency={}{host_libs}{}",
        deps_path.display(),
        &*EXTERN_FLAGS,
    ));

    config.src_base = Path::new("tests").join(test_dir);
    config.build_base = profile_path.join("test").join(test_dir);
    config.rustc_path = profile_path.join(if cfg!(windows) {
        "clippy-driver.exe"
    } else {
        "clippy-driver"
    });
    config
}

fn run_ui() {
    let mut config = base_config("ui");
    config.rustfix_coverage = true;
    // use tests/clippy.toml
    let _g = VarGuard::set("CARGO_MANIFEST_DIR", fs::canonicalize("tests").unwrap());
    let _threads = VarGuard::set(
        "RUST_TEST_THREADS",
        // if RUST_TEST_THREADS is set, adhere to it, otherwise override it
        env::var("RUST_TEST_THREADS").unwrap_or_else(|_| {
            std::thread::available_parallelism()
                .map_or(1, std::num::NonZeroUsize::get)
                .to_string()
        }),
    );
    compiletest::run_tests(&config);
    check_rustfix_coverage();
}

fn run_internal_tests() {
    // only run internal tests with the internal-tests feature
    if !RUN_INTERNAL_TESTS {
        return;
    }
    let config = base_config("ui-internal");
    compiletest::run_tests(&config);
}

fn run_ui_toml() {
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

    let mut config = base_config("ui-toml");
    config.src_base = config.src_base.canonicalize().unwrap();

    let tests = compiletest::make_tests(&config);

    let res = run_tests(&config, tests);
    match res {
        Ok(true) => {},
        Ok(false) => panic!("Some tests failed"),
        Err(e) => {
            panic!("I/O failure during tests: {e:?}");
        },
    }
}

fn run_ui_cargo() {
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

                let cargo_toml_path = case.path().join("Cargo.toml");
                let cargo_content = fs::read(cargo_toml_path)?;
                let cargo_parsed: toml::Value = toml::from_str(
                    std::str::from_utf8(&cargo_content).expect("`Cargo.toml` is not a valid utf-8 file!"),
                )
                .expect("Can't parse `Cargo.toml`");

                let _g = VarGuard::set("CARGO_MANIFEST_DIR", case.path());
                let _h = VarGuard::set(
                    "CARGO_PKG_RUST_VERSION",
                    cargo_parsed
                        .get("package")
                        .and_then(|p| p.get("rust-version"))
                        .and_then(toml::Value::as_str)
                        .unwrap_or(""),
                );

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

    if IS_RUSTC_TEST_SUITE {
        return;
    }

    let mut config = base_config("ui-cargo");
    config.src_base = config.src_base.canonicalize().unwrap();

    let tests = compiletest::make_tests(&config);

    let current_dir = env::current_dir().unwrap();
    let res = run_tests(&config, &config.filters, tests);
    env::set_current_dir(current_dir).unwrap();

    match res {
        Ok(true) => {},
        Ok(false) => panic!("Some tests failed"),
        Err(e) => {
            panic!("I/O failure during tests: {e:?}");
        },
    }
}

#[test]
fn compile_test() {
    set_var("CLIPPY_DISABLE_DOCS_LINKS", "true");
    run_ui();
    run_ui_toml();
    run_ui_cargo();
    run_internal_tests();
}

const RUSTFIX_COVERAGE_KNOWN_EXCEPTIONS: &[&str] = &[
    "assign_ops2.rs",
    "borrow_deref_ref_unfixable.rs",
    "cast_size_32bit.rs",
    "char_lit_as_u8.rs",
    "cmp_owned/without_suggestion.rs",
    "dbg_macro.rs",
    "deref_addrof_double_trigger.rs",
    "doc/unbalanced_ticks.rs",
    "eprint_with_newline.rs",
    "explicit_counter_loop.rs",
    "iter_skip_next_unfixable.rs",
    "let_and_return.rs",
    "literals.rs",
    "map_flatten.rs",
    "map_unwrap_or.rs",
    "match_bool.rs",
    "mem_replace_macro.rs",
    "needless_arbitrary_self_type_unfixable.rs",
    "needless_borrow_pat.rs",
    "needless_for_each_unfixable.rs",
    "nonminimal_bool.rs",
    "print_literal.rs",
    "print_with_newline.rs",
    "redundant_static_lifetimes_multiple.rs",
    "ref_binding_to_reference.rs",
    "repl_uninit.rs",
    "result_map_unit_fn_unfixable.rs",
    "search_is_some.rs",
    "single_component_path_imports_nested_first.rs",
    "string_add.rs",
    "suspicious_to_owned.rs",
    "toplevel_ref_arg_non_rustfix.rs",
    "unit_arg.rs",
    "unnecessary_clone.rs",
    "unnecessary_lazy_eval_unfixable.rs",
    "write_literal.rs",
    "write_literal_2.rs",
    "write_with_newline.rs",
];

fn check_rustfix_coverage() {
    let missing_coverage_path = Path::new("debug/test/ui/rustfix_missing_coverage.txt");
    let missing_coverage_path = if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        PathBuf::from(target_dir).join(missing_coverage_path)
    } else {
        missing_coverage_path.to_path_buf()
    };

    if let Ok(missing_coverage_contents) = std::fs::read_to_string(missing_coverage_path) {
        assert!(RUSTFIX_COVERAGE_KNOWN_EXCEPTIONS.iter().is_sorted_by_key(Path::new));

        for rs_file in missing_coverage_contents.lines() {
            let rs_path = Path::new(rs_file);
            if rs_path.starts_with("tests/ui/crashes") {
                continue;
            }
            assert!(rs_path.starts_with("tests/ui/"), "{rs_file:?}");
            let filename = rs_path.strip_prefix("tests/ui/").unwrap();
            assert!(
                RUSTFIX_COVERAGE_KNOWN_EXCEPTIONS
                    .binary_search_by_key(&filename, Path::new)
                    .is_ok(),
                "`{rs_file}` runs `MachineApplicable` diagnostics but is missing a `run-rustfix` annotation. \
                Please either add `// run-rustfix` at the top of the file or add the file to \
                `RUSTFIX_COVERAGE_KNOWN_EXCEPTIONS` in `tests/compile-test.rs`.",
            );
        }
    }
}

#[test]
fn rustfix_coverage_known_exceptions_accuracy() {
    for filename in RUSTFIX_COVERAGE_KNOWN_EXCEPTIONS {
        let rs_path = Path::new("tests/ui").join(filename);
        assert!(
            rs_path.exists(),
            "`{}` does not exist",
            rs_path.strip_prefix(env!("CARGO_MANIFEST_DIR")).unwrap().display()
        );
        let fixed_path = rs_path.with_extension("fixed");
        assert!(
            !fixed_path.exists(),
            "`{}` exists",
            fixed_path.strip_prefix(env!("CARGO_MANIFEST_DIR")).unwrap().display()
        );
    }
}

#[test]
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
