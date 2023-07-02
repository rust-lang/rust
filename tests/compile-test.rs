#![feature(test)] // compiletest_rs requires this attribute
#![feature(lazy_cell)]
#![feature(is_sorted)]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![warn(rust_2018_idioms, unused_lifetimes)]

use compiletest::{status_emitter, CommandBuilder};
use ui_test as compiletest;
use ui_test::Mode as TestMode;

use std::env::{self, remove_var, set_var, var_os};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use test_utils::IS_RUSTC_TEST_SUITE;

mod test_utils;

// whether to run internal tests or not
const RUN_INTERNAL_TESTS: bool = cfg!(feature = "internal");

fn base_config(test_dir: &str) -> compiletest::Config {
    let mut config = compiletest::Config {
        mode: TestMode::Yolo,
        stderr_filters: vec![],
        stdout_filters: vec![],
        output_conflict_handling: if var_os("BLESS").is_some() || env::args().any(|arg| arg == "--bless") {
            compiletest::OutputConflictHandling::Bless
        } else {
            compiletest::OutputConflictHandling::Error("cargo test -- -- --bless".into())
        },
        dependencies_crate_manifest_path: Some("clippy_test_deps/Cargo.toml".into()),
        target: None,
        out_dir: PathBuf::from(std::env::var_os("CARGO_TARGET_DIR").unwrap_or("target".into())).join("ui_test"),
        ..compiletest::Config::rustc(Path::new("tests").join(test_dir))
    };

    if let Some(_path) = option_env!("RUSTC_LIB_PATH") {
        //let path = PathBuf::from(path);
        //config.run_lib_path = path.clone();
        //config.compile_lib_path = path;
    }
    let current_exe_path = env::current_exe().unwrap();
    let deps_path = current_exe_path.parent().unwrap();
    let profile_path = deps_path.parent().unwrap();

    config.program.args.push("--emit=metadata".into());
    config.program.args.push("-Aunused".into());
    config.program.args.push("-Zui-testing".into());
    config.program.args.push("-Dwarnings".into());

    // Normalize away slashes in windows paths.
    config.stderr_filter(r"\\", "/");

    //config.build_base = profile_path.join("test").join(test_dir);
    config.program.program = profile_path.join(if cfg!(windows) {
        "clippy-driver.exe"
    } else {
        "clippy-driver"
    });
    config
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
    let config = base_config("ui");
    //config.rustfix_coverage = true;
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
    eprintln!("   Compiler: {}", config.program.display());

    let name = config.root_dir.display().to_string();

    let test_filter = test_filter();

    compiletest::run_tests_generic(
        config,
        move |path| compiletest::default_file_filter(path) && test_filter(path),
        compiletest::default_per_file_config,
        (status_emitter::Text, status_emitter::Gha::<true> { name }),
    )
    .unwrap();
    check_rustfix_coverage();
}

fn run_internal_tests() {
    // only run internal tests with the internal-tests feature
    if !RUN_INTERNAL_TESTS {
        return;
    }
    let mut config = base_config("ui-internal");
    config.dependency_builder.args.push("--features".into());
    config.dependency_builder.args.push("internal".into());
    compiletest::run_tests(config).unwrap();
}

fn run_ui_toml() {
    let mut config = base_config("ui-toml");

    config.stderr_filter(
        &regex::escape(
            &fs::canonicalize("tests")
                .unwrap()
                .parent()
                .unwrap()
                .display()
                .to_string()
                .replace('\\', "/"),
        ),
        "$$DIR",
    );

    let name = config.root_dir.display().to_string();

    let test_filter = test_filter();

    ui_test::run_tests_generic(
        config,
        |path| test_filter(path) && path.extension() == Some("rs".as_ref()),
        |config, path| {
            let mut config = config.clone();
            config
                .program
                .envs
                .push(("CLIPPY_CONF_DIR".into(), Some(path.parent().unwrap().into())));
            Some(config)
        },
        (status_emitter::Text, status_emitter::Gha::<true> { name }),
    )
    .unwrap();
}

fn run_ui_cargo() {
    if IS_RUSTC_TEST_SUITE {
        return;
    }

    let mut config = base_config("ui-cargo");
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
            &fs::canonicalize("tests")
                .unwrap()
                .parent()
                .unwrap()
                .display()
                .to_string()
                .replace('\\', "/"),
        ),
        "$$DIR",
    );

    let name = config.root_dir.display().to_string();

    let test_filter = test_filter();

    ui_test::run_tests_generic(
        config,
        |path| test_filter(path) && path.ends_with("Cargo.toml"),
        |config, path| {
            let mut config = config.clone();
            config.out_dir = PathBuf::from("target/ui_test_cargo/").join(path.parent().unwrap());
            Some(config)
        },
        (status_emitter::Text, status_emitter::Gha::<true> { name }),
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
    run_ui();
    run_ui_toml();
    run_ui_cargo();
    run_internal_tests();
    rustfix_coverage_known_exceptions_accuracy();
    ui_cargo_toml_metadata();
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
                Please either add `//@run-rustfix` at the top of the file or add the file to \
                `RUSTFIX_COVERAGE_KNOWN_EXCEPTIONS` in `tests/compile-test.rs`.",
            );
        }
    }
}

fn rustfix_coverage_known_exceptions_accuracy() {
    for filename in RUSTFIX_COVERAGE_KNOWN_EXCEPTIONS {
        let rs_path = Path::new("tests/ui").join(filename);
        assert!(rs_path.exists(), "`{}` does not exist", rs_path.display());
        let fixed_path = rs_path.with_extension("fixed");
        assert!(!fixed_path.exists(), "`{}` exists", fixed_path.display());
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
