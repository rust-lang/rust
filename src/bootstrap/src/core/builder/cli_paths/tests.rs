use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::Build;
use crate::core::builder::cli_paths::match_paths_to_steps_and_run;
use crate::core::builder::{Builder, StepDescription};
use crate::utils::tests::TestCtx;

fn render_steps_for_cli_args(args_str: &str) -> String {
    // Split a single string into a step kind and subsequent arguments.
    // E.g. "test ui" => ("test", &["ui"])
    let args = args_str.split_ascii_whitespace().collect::<Vec<_>>();
    let (kind, args) = args.split_first().unwrap();

    // Arbitrary tuple to represent the host system.
    let hosts = &["x86_64-unknown-linux-gnu"];
    // Arbitrary tuple to represent the target system, which might not be the host.
    let targets = &["aarch64-unknown-linux-gnu"];

    let config = TestCtx::new()
        .config(kind)
        // `test::Bootstrap` is only run by default in CI, causing inconsistency.
        .arg("--ci=false")
        .args(args)
        .hosts(hosts)
        .targets(targets)
        .create_config();
    let mut build = Build::new(config);
    // Some rustdoc test steps are only run by default if nodejs is
    // configured/discovered, causing inconsistency.
    build.config.nodejs = Some(PathBuf::from("node"));
    let mut builder = Builder::new(&build);

    // Tell the builder to log steps that it would run, instead of running them.
    let mut buf = Arc::new(Mutex::new(String::new()));
    let buf2 = Arc::clone(&buf);
    builder.log_cli_step_for_tests = Some(Box::new(move |step_desc, pathsets, targets| {
        use std::fmt::Write;
        let mut buf = buf2.lock().unwrap();

        let StepDescription { name, kind, .. } = step_desc;
        // Strip boilerplate to make step names easier to read.
        let name = name.strip_prefix("bootstrap::core::build_steps::").unwrap_or(name);

        writeln!(buf, "[{kind:?}] {name}").unwrap();
        writeln!(buf, "    targets: {targets:?}").unwrap();
        for pathset in pathsets {
            // Normalize backslashes in paths, to avoid snapshot differences on Windows.
            // FIXME(Zalathar): Doing a string-replace on <PathSet as Debug>
            // is a bit unprincipled, but it's good enough for now.
            let pathset_str = format!("{pathset:?}").replace('\\', "/");
            writeln!(buf, "    - {pathset_str}").unwrap();
        }
    }));

    builder.execute_cli();

    String::clone(&buf.lock().unwrap())
}

fn snapshot_test_inner(name: &str, args_str: &str) {
    let mut settings = insta::Settings::clone_current();
    // Use the test name as the snapshot filename, not its whole fully-qualified name.
    settings.set_prepend_module_to_snapshot(false);
    settings.bind(|| {
        insta::assert_snapshot!(name, render_steps_for_cli_args(args_str), args_str);
    });
}

/// Keep the snapshots directory tidy by forbidding `.snap` files that don't
/// correspond to a test name.
fn no_unused_snapshots_inner(known_test_names: &[&str]) {
    let known_test_names = known_test_names.iter().copied().collect::<HashSet<&str>>();

    let mut unexpected_file_names = BTreeSet::new();

    // FIXME(Zalathar): Is there a better way to locate the snapshots dir?
    for entry in walkdir::WalkDir::new("src/core/builder/cli_paths/snapshots")
        .into_iter()
        .map(Result::unwrap)
    {
        let meta = entry.metadata().unwrap();
        if !meta.is_file() {
            continue;
        }

        let name = entry.file_name().to_str().unwrap();
        if let Some(name_stub) = name.strip_suffix(".snap")
            && !known_test_names.contains(name_stub)
        {
            unexpected_file_names.insert(name.to_owned());
        }
    }

    assert!(
        unexpected_file_names.is_empty(),
        "Found snapshot files that don't correspond to a test name: {unexpected_file_names:#?}",
    );
}

macro_rules! declare_tests {
    (
        $( ($name:ident, $args:literal) ),* $(,)?
    ) => {
        $(
            #[test]
            fn $name() {
                snapshot_test_inner(stringify!($name), $args);
            }
        )*

        #[test]
        fn no_unused_snapshots() {
            let known_test_names = &[ $( stringify!($name), )* ];
            no_unused_snapshots_inner(known_test_names);
        }
    };
}

// Snapshot tests for bootstrap's command-line path-to-step handling.
//
// To bless these tests as necessary, choose one:
// - Run `INSTA_UPDATE=always ./x test bootstrap`
// - Run `./x test bootstrap --bless`
// - Follow the instructions for `cargo-insta` in bootstrap's README.md
//
// These snapshot tests capture _current_ behavior, to prevent unintended
// changes or regressions. If the current behavior is wrong or undersirable,
// then any fix will necessarily have to re-bless the affected tests!
declare_tests!(
    // tidy-alphabetical-start
    (x_bench, "bench"),
    (x_build, "build"),
    (x_build_compiler, "build compiler"),
    (x_build_compiletest, "build compiletest"),
    (x_build_library, "build library"),
    (x_build_llvm, "build llvm"),
    (x_build_rustc, "build rustc"),
    (x_build_rustc_llvm, "build rustc_llvm"),
    (x_build_rustdoc, "build rustdoc"),
    (x_build_sysroot, "build sysroot"),
    (x_check, "check"),
    (x_check_bootstrap, "check bootstrap"),
    (x_check_compiler, "check compiler"),
    (x_check_compiletest, "check compiletest"),
    (x_check_compiletest_include_default_paths, "check compiletest --include-default-paths"),
    (x_check_library, "check library"),
    (x_check_rustc, "check rustc"),
    (x_check_rustdoc, "check rustdoc"),
    (x_clean, "clean"),
    (x_clippy, "clippy"),
    (x_dist, "dist"),
    (x_doc, "doc"),
    (x_fix, "fix"),
    (x_fmt, "fmt"),
    (x_install, "install"),
    (x_miri, "miri"),
    (x_run, "run"),
    (x_setup, "setup"),
    (x_test, "test"),
    (x_test_coverage, "test coverage"),
    (x_test_coverage_map, "test coverage-map"),
    (x_test_coverage_run, "test coverage-run"),
    // FIXME(Zalathar): Currently this doesn't actually skip the coverage-run tests!
    (x_test_coverage_skip_coverage_run, "test coverage --skip=coverage-run"),
    (x_test_debuginfo, "test debuginfo"),
    (x_test_library, "test library"),
    (x_test_librustdoc, "test librustdoc"),
    (x_test_librustdoc_rustdoc, "test librustdoc rustdoc"),
    (x_test_rustdoc, "test rustdoc"),
    (x_test_skip_coverage, "test --skip=coverage"),
    // FIXME(Zalathar): This doesn't skip the coverage-map or coverage-run tests.
    (x_test_skip_tests, "test --skip=tests"),
    // From `src/ci/docker/scripts/stage_2_test_set2.sh`.
    (
        x_test_skip_tests_etc,
        "test --skip=tests --skip=coverage-map --skip=coverage-run --skip=library --skip=tidyselftest"
    ),
    (x_test_tests, "test tests"),
    (x_test_tests_skip_coverage, "test tests --skip=coverage"),
    (x_test_tests_ui, "test tests/ui"),
    (x_test_tidy, "test tidy"),
    (x_test_tidyselftest, "test tidyselftest"),
    (x_test_ui, "test ui"),
    (x_vendor, "vendor"),
    // tidy-alphabetical-end
);
