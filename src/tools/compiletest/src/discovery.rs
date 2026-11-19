use core::panic;
use std::collections::HashSet;
use std::io::{self, ErrorKind};
use std::sync::Arc;
use std::time::SystemTime;
use std::{fs, vec};

use build_helper::git::{get_git_modified_files, get_git_untracked_files};
use camino::{Utf8Component, Utf8Path, Utf8PathBuf};
use rayon::iter::{ParallelBridge, ParallelIterator};
use tracing::debug;
use walkdir::WalkDir;

use crate::common::{
    CodegenBackend, Config, Debugger, TestMode, TestPaths, UI_EXTENSIONS, expected_output_path,
    output_base_dir, output_relative_path,
};
use crate::directives::{
    self, AuxProps, DirectivesCache, EarlyProps, FileDirectives, make_test_description,
};
use crate::executor::{CollectedTest, TestVariant};
use crate::{fatal, runtest, warning};

/// Called by `main` after the config has been parsed.
pub(crate) fn discover_tests(config: &Arc<Config>) -> Vec<CollectedTest> {
    debug!(?config, "run_tests");

    // Debugging emscripten code doesn't make sense today
    let ignore_tests = config.mode == TestMode::DebugInfo && config.target.contains("emscripten");

    // Discover all of the tests in the test suite directory, and build a `CollectedTest`
    // structure for each test (or each revision of a multi-revision test).
    let mut tests = Vec::new();
    if !ignore_tests {
        tests.extend(collect_and_make_tests(config.clone()));
    }

    tests.sort_by(|a, b| Ord::cmp(&a.desc.name, &b.desc.name));

    tests
}

/// Read-only context data used during test collection.
struct TestCollectorCx {
    config: Arc<Config>,
    cache: DirectivesCache,
    common_inputs_stamp: Stamp,
    modified_tests: Vec<Utf8PathBuf>,
}

/// Mutable state used during test collection.
struct TestCollector {
    tests: Vec<CollectedTest>,
    found_path_stems: HashSet<Utf8PathBuf>,
    poisoned: bool,
}

impl TestCollector {
    fn new() -> Self {
        TestCollector { tests: vec![], found_path_stems: HashSet::new(), poisoned: false }
    }

    fn merge(&mut self, mut other: Self) {
        self.tests.append(&mut other.tests);
        self.found_path_stems.extend(other.found_path_stems);
        self.poisoned |= other.poisoned;
    }
}

/// Creates test structures for every test/revision in the test suite directory.
///
/// This always inspects _all_ test files in the suite (e.g. all 17k+ ui tests),
/// regardless of whether any filters/tests were specified on the command-line,
/// because filtering is handled later by code that was copied from libtest.
///
/// FIXME(Zalathar): Now that we no longer rely on libtest, try to overhaul
/// test discovery to take into account the filters/tests specified on the
/// command-line, instead of having to enumerate everything.
fn collect_and_make_tests(config: Arc<Config>) -> Vec<CollectedTest> {
    debug!("making tests from {}", config.src_test_suite_root);
    let common_inputs_stamp = common_inputs_stamp(&config);
    let modified_tests =
        modified_tests(&config, &config.src_test_suite_root).unwrap_or_else(|err| {
            fatal!("modified_tests: {}: {err}", config.src_test_suite_root);
        });
    let cache = DirectivesCache::load(&config);

    let cx = TestCollectorCx { config, cache, common_inputs_stamp, modified_tests };
    let collector = collect_tests_from_dir(&cx, &cx.config.src_test_suite_root, Utf8Path::new(""))
        .unwrap_or_else(|reason| {
            panic!("Could not read tests from {}: {reason}", cx.config.src_test_suite_root)
        });

    let TestCollector { tests, found_path_stems, poisoned } = collector;

    if poisoned {
        eprintln!();
        panic!("there are errors in tests");
    }

    check_for_overlapping_test_paths(&found_path_stems);

    tests
}

/// Returns the most recent last-modified timestamp from among the input files
/// that are considered relevant to all tests (e.g. the compiler, std, and
/// compiletest itself).
///
/// (Some of these inputs aren't actually relevant to _all_ tests, but they are
/// common to some subset of tests, and are hopefully unlikely to be modified
/// while working on other tests.)
fn common_inputs_stamp(config: &Config) -> Stamp {
    let src_root = &config.src_root;

    let mut stamp = Stamp::from_path(&config.rustc_path);

    // Relevant pretty printer files
    let pretty_printer_files = [
        "src/etc/rust_types.py",
        "src/etc/gdb_load_rust_pretty_printers.py",
        "src/etc/gdb_lookup.py",
        "src/etc/gdb_providers.py",
        "src/etc/lldb_lookup.py",
        "src/etc/lldb_providers.py",
    ];
    for file in &pretty_printer_files {
        let path = src_root.join(file);
        stamp.add_path(&path);
    }

    stamp.add_dir(&src_root.join("src/etc/debugger_tester"));
    stamp.add_dir(&src_root.join("src/etc/natvis"));

    stamp.add_dir(&config.target_run_lib_path);

    if let Some(ref rustdoc_path) = config.rustdoc_path {
        stamp.add_path(&rustdoc_path);
        stamp.add_path(&src_root.join("src/etc/htmldocck.py"));
    }

    // Re-run coverage tests if the `coverage-dump` tool was modified,
    // because its output format might have changed.
    if let Some(coverage_dump_path) = &config.coverage_dump_path {
        stamp.add_path(coverage_dump_path)
    }

    stamp.add_dir(&src_root.join("src/tools/run-make-support"));

    // Compiletest itself.
    stamp.add_dir(&src_root.join("src/tools/compiletest"));

    stamp
}

/// Returns a list of modified/untracked test files that should be run when
/// the `--only-modified` flag is in use.
///
/// (Might be inaccurate in some cases.)
fn modified_tests(config: &Config, dir: &Utf8Path) -> Result<Vec<Utf8PathBuf>, String> {
    // If `--only-modified` wasn't passed, the list of modified tests won't be
    // used for anything, so avoid some work and just return an empty list.
    if !config.only_modified {
        return Ok(vec![]);
    }

    let files = get_git_modified_files(
        &config.git_config(),
        Some(dir.as_std_path()),
        &vec!["rs", "stderr", "fixed"],
    )?;
    // Add new test cases to the list, it will be convenient in daily development.
    let untracked_files = get_git_untracked_files(Some(dir.as_std_path()))?.unwrap_or(vec![]);

    let all_paths = [&files[..], &untracked_files[..]].concat();
    let full_paths = {
        let mut full_paths: Vec<Utf8PathBuf> = all_paths
            .into_iter()
            .map(|f| Utf8PathBuf::from(f).with_extension("").with_extension("rs"))
            .filter_map(
                |f| if Utf8Path::new(&f).exists() { f.canonicalize_utf8().ok() } else { None },
            )
            .collect();
        full_paths.dedup();
        full_paths.sort_unstable();
        full_paths
    };
    Ok(full_paths)
}

/// Recursively scans a directory to find test files and create test structures
/// that will be handed over to the executor.
fn collect_tests_from_dir(
    cx: &TestCollectorCx,
    dir: &Utf8Path,
    relative_dir_path: &Utf8Path,
) -> io::Result<TestCollector> {
    // Ignore directories that contain a file named `compiletest-ignore-dir`.
    if dir.join("compiletest-ignore-dir").exists() {
        return Ok(TestCollector::new());
    }

    let mut components = dir.components().rev();
    if let Some(Utf8Component::Normal(last)) = components.next()
        && let Some(("assembly" | "codegen", backend)) = last.split_once('-')
        && let Some(Utf8Component::Normal(parent)) = components.next()
        && parent == "tests"
        && let Ok(backend) = backend.parse::<CodegenBackend>()
        && backend != cx.config.default_codegen_backend
    {
        // We ignore asm tests which don't match the current codegen backend.
        warning!(
            "Ignoring tests in `{dir}` because they don't match the configured codegen \
             backend (`{}`)",
            cx.config.default_codegen_backend.as_str(),
        );
        return Ok(TestCollector::new());
    }

    // For run-make tests, a "test file" is actually a directory that contains an `rmake.rs`.
    if cx.config.mode == TestMode::RunMake {
        let mut collector = TestCollector::new();
        if dir.join("rmake.rs").exists() {
            let paths = TestPaths {
                file: dir.to_path_buf(),
                relative_dir: relative_dir_path.parent().unwrap().to_path_buf(),
            };
            make_test(cx, &mut collector, &paths);
            // This directory is a test, so don't try to find other tests inside it.
            return Ok(collector);
        }
    }

    // If we find a test foo/bar.rs, we have to build the
    // output directory `$build/foo` so we can write
    // `$build/foo/bar` into it. We do this *now* in this
    // sequential loop because otherwise, if we do it in the
    // tests themselves, they race for the privilege of
    // creating the directories and sometimes fail randomly.
    let build_dir = output_relative_path(&cx.config, relative_dir_path);
    fs::create_dir_all(&build_dir).unwrap();

    // Add each `.rs` file as a test, and recurse further on any
    // subdirectories we find, except for `auxiliary` directories.
    // FIXME: this walks full tests tree, even if we have something to ignore
    // use walkdir/ignore like in tidy?
    fs::read_dir(dir.as_std_path())?
        .par_bridge()
        .map(|file| {
            let mut collector = TestCollector::new();
            let file = file?;
            let file_path = Utf8PathBuf::try_from(file.path()).unwrap();
            let file_name = file_path.file_name().unwrap();

            if is_test(file_name)
                && (!cx.config.only_modified || cx.modified_tests.contains(&file_path))
            {
                // We found a test file, so create the corresponding test structures.
                debug!(%file_path, "found test file");

                // Record the stem of the test file, to check for overlaps later.
                let rel_test_path = relative_dir_path.join(file_path.file_stem().unwrap());
                collector.found_path_stems.insert(rel_test_path);

                let paths =
                    TestPaths { file: file_path, relative_dir: relative_dir_path.to_path_buf() };
                make_test(cx, &mut collector, &paths);
            } else if file_path.is_dir() {
                // Recurse to find more tests in a subdirectory.
                let relative_file_path = relative_dir_path.join(file_name);
                if file_name != "auxiliary" {
                    debug!(%file_path, "found directory");
                    collector.merge(collect_tests_from_dir(cx, &file_path, &relative_file_path)?);
                }
            } else {
                debug!(%file_path, "found other file/directory");
            }
            Ok(collector)
        })
        .reduce(
            || Ok(TestCollector::new()),
            |a, b| {
                let mut a = a?;
                a.merge(b?);
                Ok(a)
            },
        )
}

/// Returns true if `file_name` looks like a proper test file name.
pub(crate) fn is_test(file_name: &str) -> bool {
    if !file_name.ends_with(".rs") {
        return false;
    }

    // `.`, `#`, and `~` are common temp-file prefixes.
    let invalid_prefixes = &[".", "#", "~"];
    !invalid_prefixes.iter().any(|p| file_name.starts_with(p))
}

/// For a single test file, creates one or more test structures (one per revision) that can be
/// handed over to the executor to run, possibly in parallel.
fn make_test(cx: &TestCollectorCx, collector: &mut TestCollector, testpaths: &TestPaths) {
    // For run-make tests, each "test file" is actually a _directory_ containing an `rmake.rs`. But
    // for the purposes of directive parsing, we want to look at that recipe file, not the directory
    // itself.
    let test_path = if cx.config.mode == TestMode::RunMake {
        testpaths.file.join("rmake.rs")
    } else {
        testpaths.file.clone()
    };

    // Scan the test file to discover its revisions, if any.
    let file_contents =
        fs::read_to_string(&test_path).expect("reading test file for directives should succeed");
    let file_directives = FileDirectives::from_file_contents(&test_path, &file_contents);

    if let Err(message) = directives::do_early_directives_check(cx.config.mode, &file_directives) {
        // FIXME(Zalathar): Overhaul compiletest error handling so that we
        // don't have to resort to ad-hoc panics everywhere.
        panic!("directives check failed:\n{message}");
    }
    let early_props = EarlyProps::from_file_directives(&cx.config, &file_directives);

    // Normally we create one structure per revision, with two exceptions:
    // - If a test doesn't use revisions, create a dummy revision (None) so that
    //   the test can still run.
    // - Incremental tests inherently can't run their revisions in parallel, so
    //   we treat them like non-revisioned tests here. Incremental revisions are
    //   handled internally by `runtest::run` instead.
    let revisions = if early_props.revisions.is_empty() || cx.config.mode == TestMode::Incremental {
        vec![None]
    } else {
        early_props.revisions.iter().map(|r| Some(r.as_str())).collect()
    };

    // For debuginfo tests, we have to run them once for each debugger.
    // We thus create a cartesian product of each revision and each supported debugger here.
    let debuggers = if cx.config.mode == TestMode::DebugInfo {
        vec![Some(Debugger::Cdb), Some(Debugger::Gdb), Some(Debugger::Lldb)]
    } else {
        vec![None]
    };

    // For each revision (or the sole dummy revision) and each debugger, create and append a
    // `CollectedTest` that can be handed over to the test executor.
    for debugger in debuggers {
        collector.tests.extend(revisions.iter().map(|&revision| {
            let revision = revision.map(str::to_owned);
            let variant = TestVariant { revision, debugger };

            // Create a test name and description to hand over to the executor.
            let (test_name, filterable_path) =
                make_test_name_and_filterable_path(&cx.config, testpaths, &variant);

            // While scanning for ignore/only/needs directives, also collect aux
            // paths for up-to-date checking.
            let mut aux_props = AuxProps::default();

            // Create a description struct for the test/revision.
            // This is where `ignore-*`/`only-*`/`needs-*` directives are handled,
            // because they historically needed to set the libtest ignored flag.
            let mut desc = make_test_description(
                &cx.config,
                &cx.cache,
                test_name,
                &test_path,
                &filterable_path,
                &file_directives,
                &variant,
                &mut collector.poisoned,
                &mut aux_props,
            );

            // If a test's inputs haven't changed since the last time it ran,
            // mark it as ignored so that the executor will skip it.
            if !desc.is_ignored()
                && !cx.config.force_rerun
                && is_up_to_date(cx, testpaths, &aux_props, &variant)
            {
                // Keep this in sync with the "up-to-date" message detected by bootstrap.
                // FIXME(Zalathar): Now that we are no longer tied to libtest, we could
                // find a less fragile way to communicate this status to bootstrap.
                desc.ignore_message = Some("up-to-date".into());
            }

            let config = Arc::clone(&cx.config);
            let testpaths = testpaths.clone();

            CollectedTest { desc, config, testpaths, variant }
        }));
    }
}

/// The path of the `stamp` file that gets created or updated whenever a
/// particular test completes successfully.
pub(crate) fn stamp_file_path(
    config: &Config,
    testpaths: &TestPaths,
    variant: &TestVariant,
) -> Utf8PathBuf {
    output_base_dir(config, testpaths, variant).join("stamp")
}

/// Returns a list of files that, if modified, would cause this test to no
/// longer be up-to-date.
///
/// (Might be inaccurate in some cases.)
fn files_related_to_test(
    config: &Config,
    testpaths: &TestPaths,
    aux_props: &AuxProps,
    variant: &TestVariant,
) -> Vec<Utf8PathBuf> {
    let mut related = vec![];

    if testpaths.file.is_dir() {
        // run-make tests use their individual directory
        for entry in WalkDir::new(&testpaths.file) {
            let path = entry.unwrap().into_path();
            if path.is_file() {
                related.push(Utf8PathBuf::try_from(path).unwrap());
            }
        }
    } else {
        related.push(testpaths.file.clone());
    }

    for aux in aux_props.all_aux_path_strings() {
        // FIXME(Zalathar): Perform all `auxiliary` path resolution in one place.
        // FIXME(Zalathar): This only finds auxiliary files used _directly_ by
        // the test file; if a transitive auxiliary is modified, the test might
        // be treated as "up-to-date" even though it should run.
        let path = testpaths.file.parent().unwrap().join("auxiliary").join(aux);
        related.push(path);
    }

    // UI test files.
    for extension in UI_EXTENSIONS {
        let path =
            expected_output_path(testpaths, variant.revision(), &config.compare_mode, extension);
        related.push(path);
    }

    // `minicore.rs` test auxiliary: we need to make sure tests get rerun if this changes.
    related.push(config.src_root.join("tests").join("auxiliary").join("minicore.rs"));

    // `tests/debuginfo` blessed files
    match variant.debugger {
        Some(debugger @ Debugger::Lldb | debugger @ Debugger::Gdb) => {
            let bless_path: Utf8PathBuf =
                testpaths.file.parent().unwrap().join(format!("{}_input", debugger.to_str()));
            if bless_path.is_dir() {
                related.extend(
                    WalkDir::new(bless_path)
                        .into_iter()
                        .map(|entry| Utf8PathBuf::from(entry.unwrap().path().to_str().unwrap())),
                );
            }
        }
        Some(Debugger::Cdb) | None => {}
    }

    related
}

/// Checks whether a particular test/revision is "up-to-date", meaning that no
/// relevant files/settings have changed since the last time the test succeeded.
///
/// (This is not very reliable in some circumstances, so the `--force-rerun`
/// flag can be used to ignore up-to-date checking and always re-run tests.)
fn is_up_to_date(
    cx: &TestCollectorCx,
    testpaths: &TestPaths,
    aux_props: &AuxProps,
    variant: &TestVariant,
) -> bool {
    let stamp_file_path = stamp_file_path(&cx.config, testpaths, variant);
    // Check the config hash inside the stamp file.
    let contents = match fs::read_to_string(&stamp_file_path) {
        Ok(f) => f,
        Err(ref e) if e.kind() == ErrorKind::InvalidData => panic!("Can't read stamp contents"),
        // The test hasn't succeeded yet, so it is not up-to-date.
        Err(_) => return false,
    };
    let expected_hash = runtest::compute_stamp_hash(&cx.config, variant);
    if contents != expected_hash {
        // Some part of compiletest configuration has changed since the test
        // last succeeded, so it is not up-to-date.
        return false;
    }

    // Check the timestamp of the stamp file against the last modified time
    // of all files known to be relevant to the test.
    let mut inputs_stamp = cx.common_inputs_stamp.clone();
    for path in files_related_to_test(&cx.config, testpaths, aux_props, variant) {
        inputs_stamp.add_path(&path);
    }

    // If no relevant files have been modified since the stamp file was last
    // written, the test is up-to-date.
    inputs_stamp < Stamp::from_path(&stamp_file_path)
}

/// The maximum of a set of file-modified timestamps.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Stamp {
    time: SystemTime,
}

impl Stamp {
    /// Creates a timestamp holding the last-modified time of the specified file.
    fn from_path(path: &Utf8Path) -> Self {
        let mut stamp = Stamp { time: SystemTime::UNIX_EPOCH };
        stamp.add_path(path);
        stamp
    }

    /// Updates this timestamp to the last-modified time of the specified file,
    /// if it is later than the currently-stored timestamp.
    fn add_path(&mut self, path: &Utf8Path) {
        let modified = fs::metadata(path.as_std_path())
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        self.time = self.time.max(modified);
    }

    /// Updates this timestamp to the most recent last-modified time of all files
    /// recursively contained in the given directory, if it is later than the
    /// currently-stored timestamp.
    fn add_dir(&mut self, path: &Utf8Path) {
        let path = path.as_std_path();
        for entry in WalkDir::new(path) {
            let entry = entry.unwrap();
            if entry.file_type().is_file() {
                let modified = entry
                    .metadata()
                    .ok()
                    .and_then(|metadata| metadata.modified().ok())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                self.time = self.time.max(modified);
            }
        }
    }
}

/// Creates a name for this test/revision that can be handed over to the executor.
fn make_test_name_and_filterable_path(
    config: &Config,
    testpaths: &TestPaths,
    variant: &TestVariant,
) -> (String, Utf8PathBuf) {
    // Print the name of the file, relative to the sources root.
    let path = testpaths.file.strip_prefix(&config.src_root).unwrap();
    let debugger = match variant.debugger.as_ref() {
        Some(d) => format!("-{d}"),
        None => String::new(),
    };
    let mode_suffix = match config.compare_mode {
        Some(ref mode) => format!(" ({})", mode.to_str()),
        None => String::new(),
    };

    let name = format!(
        "[{}{}{}] {}{}",
        config.mode,
        debugger,
        mode_suffix,
        path,
        variant.revision().map_or("".to_string(), |rev| format!("#{}", rev))
    );

    // `path` is the full path from the repo root like, `tests/ui/foo/bar.rs`.
    // Filtering is applied without the `tests/ui/` part, so strip that off.
    // First strip off "tests" to make sure we don't have some unexpected path.
    let mut filterable_path = path.strip_prefix("tests").unwrap().to_owned();
    // Now strip off e.g. "ui" or "run-make" component.
    filterable_path = filterable_path.components().skip(1).collect();

    (name, filterable_path)
}

/// Checks that test discovery didn't find any tests whose name stem is a prefix
/// of some other tests's name.
///
/// For example, suppose the test suite contains these two test files:
/// - `tests/rustdoc-html/primitive.rs`
/// - `tests/rustdoc-html/primitive/no_std.rs`
///
/// The test runner might put the output from those tests in these directories:
/// - `$build/test/rustdoc/primitive/`
/// - `$build/test/rustdoc/primitive/no_std/`
///
/// Because one output path is a subdirectory of the other, the two tests might
/// interfere with each other in unwanted ways, especially if the test runner
/// decides to delete test output directories to clean them between runs.
/// To avoid problems, we forbid test names from overlapping in this way.
///
/// See <https://github.com/rust-lang/rust/pull/109509> for more context.
fn check_for_overlapping_test_paths(found_path_stems: &HashSet<Utf8PathBuf>) {
    let mut collisions = Vec::new();
    for path in found_path_stems {
        for ancestor in path.ancestors().skip(1) {
            if found_path_stems.contains(ancestor) {
                collisions.push((path, ancestor));
            }
        }
    }
    if !collisions.is_empty() {
        collisions.sort();
        let collisions: String = collisions
            .into_iter()
            .map(|(path, check_parent)| format!("test {path} clashes with {check_parent}\n"))
            .collect();
        panic!(
            "{collisions}\n\
            Tests cannot have overlapping names. Make sure they use unique prefixes."
        );
    }
}
