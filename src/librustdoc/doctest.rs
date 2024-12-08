mod make;
mod markdown;
mod runner;
mod rust;

use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::{panic, str};

pub(crate) use make::DocTestBuilder;
pub(crate) use markdown::test as test_markdown;
use rustc_ast as ast;
use rustc_data_structures::fx::{FxHashMap, FxIndexMap, FxIndexSet};
use rustc_errors::emitter::HumanReadableErrorType;
use rustc_errors::{ColorConfig, DiagCtxtHandle, ErrorGuaranteed, FatalError};
use rustc_hir::CRATE_HIR_ID;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_interface::interface;
use rustc_session::config::{self, CrateType, ErrorOutputType, Input};
use rustc_session::lint;
use rustc_span::FileName;
use rustc_span::edition::Edition;
use rustc_span::symbol::sym;
use rustc_target::spec::{Target, TargetTuple};
use tempfile::{Builder as TempFileBuilder, TempDir};
use tracing::debug;

use self::rust::HirCollector;
use crate::config::Options as RustdocOptions;
use crate::html::markdown::{ErrorCodes, Ignore, LangString, MdRelLine};
use crate::lint::init_lints;

/// Options that apply to all doctests in a crate or Markdown file (for `rustdoc foo.md`).
#[derive(Clone)]
pub(crate) struct GlobalTestOptions {
    /// Name of the crate (for regular `rustdoc`) or Markdown file (for `rustdoc foo.md`).
    pub(crate) crate_name: String,
    /// Whether to disable the default `extern crate my_crate;` when creating doctests.
    pub(crate) no_crate_inject: bool,
    /// Whether inserting extra indent spaces in code block,
    /// default is `false`, only `true` for generating code link of Rust playground
    pub(crate) insert_indent_space: bool,
    /// Additional crate-level attributes to add to doctests.
    pub(crate) attrs: Vec<String>,
    /// Path to file containing arguments for the invocation of rustc.
    pub(crate) args_file: PathBuf,
}

pub(crate) fn generate_args_file(file_path: &Path, options: &RustdocOptions) -> Result<(), String> {
    let mut file = File::create(file_path)
        .map_err(|error| format!("failed to create args file: {error:?}"))?;

    // We now put the common arguments into the file we created.
    let mut content = vec!["--crate-type=bin".to_string()];

    for cfg in &options.cfgs {
        content.push(format!("--cfg={cfg}"));
    }
    for check_cfg in &options.check_cfgs {
        content.push(format!("--check-cfg={check_cfg}"));
    }

    for lib_str in &options.lib_strs {
        content.push(format!("-L{lib_str}"));
    }
    for extern_str in &options.extern_strs {
        content.push(format!("--extern={extern_str}"));
    }
    content.push("-Ccodegen-units=1".to_string());
    for codegen_options_str in &options.codegen_options_strs {
        content.push(format!("-C{codegen_options_str}"));
    }
    for unstable_option_str in &options.unstable_opts_strs {
        content.push(format!("-Z{unstable_option_str}"));
    }

    let content = content.join("\n");

    file.write_all(content.as_bytes())
        .map_err(|error| format!("failed to write arguments to temporary file: {error:?}"))?;
    Ok(())
}

fn get_doctest_dir() -> io::Result<TempDir> {
    TempFileBuilder::new().prefix("rustdoctest").tempdir()
}

pub(crate) fn run(
    dcx: DiagCtxtHandle<'_>,
    input: Input,
    options: RustdocOptions,
) -> Result<(), ErrorGuaranteed> {
    let invalid_codeblock_attributes_name = crate::lint::INVALID_CODEBLOCK_ATTRIBUTES.name;

    // See core::create_config for what's going on here.
    let allowed_lints = vec![
        invalid_codeblock_attributes_name.to_owned(),
        lint::builtin::UNKNOWN_LINTS.name.to_owned(),
        lint::builtin::RENAMED_AND_REMOVED_LINTS.name.to_owned(),
    ];

    let (lint_opts, lint_caps) = init_lints(allowed_lints, options.lint_opts.clone(), |lint| {
        if lint.name == invalid_codeblock_attributes_name {
            None
        } else {
            Some((lint.name_lower(), lint::Allow))
        }
    });

    debug!(?lint_opts);

    let crate_types =
        if options.proc_macro_crate { vec![CrateType::ProcMacro] } else { vec![CrateType::Rlib] };

    let sessopts = config::Options {
        maybe_sysroot: options.maybe_sysroot.clone(),
        search_paths: options.libs.clone(),
        crate_types,
        lint_opts,
        lint_cap: Some(options.lint_cap.unwrap_or(lint::Forbid)),
        cg: options.codegen_options.clone(),
        externs: options.externs.clone(),
        unstable_features: options.unstable_features,
        actually_rustdoc: true,
        edition: options.edition,
        target_triple: options.target.clone(),
        crate_name: options.crate_name.clone(),
        remap_path_prefix: options.remap_path_prefix.clone(),
        ..config::Options::default()
    };

    let mut cfgs = options.cfgs.clone();
    cfgs.push("doc".to_owned());
    cfgs.push("doctest".to_owned());
    let config = interface::Config {
        opts: sessopts,
        crate_cfg: cfgs,
        crate_check_cfg: options.check_cfgs.clone(),
        input: input.clone(),
        output_file: None,
        output_dir: None,
        file_loader: None,
        locale_resources: rustc_driver::DEFAULT_LOCALE_RESOURCES.to_vec(),
        lint_caps,
        psess_created: None,
        hash_untracked_state: None,
        register_lints: Some(Box::new(crate::lint::register_lints)),
        override_queries: None,
        make_codegen_backend: None,
        registry: rustc_driver::diagnostics_registry(),
        ice_file: None,
        using_internal_features: Arc::default(),
        expanded_args: options.expanded_args.clone(),
    };

    let externs = options.externs.clone();
    let json_unused_externs = options.json_unused_externs;

    let temp_dir = match get_doctest_dir()
        .map_err(|error| format!("failed to create temporary directory: {error:?}"))
    {
        Ok(temp_dir) => temp_dir,
        Err(error) => return crate::wrap_return(dcx, Err(error)),
    };
    let args_path = temp_dir.path().join("rustdoc-cfgs");
    crate::wrap_return(dcx, generate_args_file(&args_path, &options))?;

    let CreateRunnableDocTests {
        standalone_tests,
        mergeable_tests,
        rustdoc_options,
        opts,
        unused_extern_reports,
        compiling_test_count,
        ..
    } = interface::run_compiler(config, |compiler| {
        compiler.enter(|queries| {
            let collector = queries.global_ctxt()?.enter(|tcx| {
                let crate_name = tcx.crate_name(LOCAL_CRATE).to_string();
                let crate_attrs = tcx.hir().attrs(CRATE_HIR_ID);
                let opts = scrape_test_config(crate_name, crate_attrs, args_path);
                let enable_per_target_ignores = options.enable_per_target_ignores;

                let mut collector = CreateRunnableDocTests::new(options, opts);
                let hir_collector = HirCollector::new(
                    ErrorCodes::from(compiler.sess.opts.unstable_features.is_nightly_build()),
                    enable_per_target_ignores,
                    tcx,
                );
                let tests = hir_collector.collect_crate();
                tests.into_iter().for_each(|t| collector.add_test(t));

                collector
            });
            if compiler.sess.dcx().has_errors().is_some() {
                FatalError.raise();
            }

            Ok(collector)
        })
    })?;

    run_tests(opts, &rustdoc_options, &unused_extern_reports, standalone_tests, mergeable_tests);

    let compiling_test_count = compiling_test_count.load(Ordering::SeqCst);

    // Collect and warn about unused externs, but only if we've gotten
    // reports for each doctest
    if json_unused_externs.is_enabled() {
        let unused_extern_reports: Vec<_> =
            std::mem::take(&mut unused_extern_reports.lock().unwrap());
        if unused_extern_reports.len() == compiling_test_count {
            let extern_names =
                externs.iter().map(|(name, _)| name).collect::<FxIndexSet<&String>>();
            let mut unused_extern_names = unused_extern_reports
                .iter()
                .map(|uexts| uexts.unused_extern_names.iter().collect::<FxIndexSet<&String>>())
                .fold(extern_names, |uextsa, uextsb| {
                    uextsa.intersection(&uextsb).copied().collect::<FxIndexSet<&String>>()
                })
                .iter()
                .map(|v| (*v).clone())
                .collect::<Vec<String>>();
            unused_extern_names.sort();
            // Take the most severe lint level
            let lint_level = unused_extern_reports
                .iter()
                .map(|uexts| uexts.lint_level.as_str())
                .max_by_key(|v| match *v {
                    "warn" => 1,
                    "deny" => 2,
                    "forbid" => 3,
                    // The allow lint level is not expected,
                    // as if allow is specified, no message
                    // is to be emitted.
                    v => unreachable!("Invalid lint level '{v}'"),
                })
                .unwrap_or("warn")
                .to_string();
            let uext = UnusedExterns { lint_level, unused_extern_names };
            let unused_extern_json = serde_json::to_string(&uext).unwrap();
            eprintln!("{unused_extern_json}");
        }
    }

    Ok(())
}

pub(crate) fn run_tests(
    opts: GlobalTestOptions,
    rustdoc_options: &Arc<RustdocOptions>,
    unused_extern_reports: &Arc<Mutex<Vec<UnusedExterns>>>,
    mut standalone_tests: Vec<test::TestDescAndFn>,
    mergeable_tests: FxIndexMap<Edition, Vec<(DocTestBuilder, ScrapedDocTest)>>,
) {
    let mut test_args = Vec::with_capacity(rustdoc_options.test_args.len() + 1);
    test_args.insert(0, "rustdoctest".to_string());
    test_args.extend_from_slice(&rustdoc_options.test_args);
    if rustdoc_options.nocapture {
        test_args.push("--nocapture".to_string());
    }

    let mut nb_errors = 0;
    let mut ran_edition_tests = 0;
    let target_str = rustdoc_options.target.to_string();

    for (edition, mut doctests) in mergeable_tests {
        if doctests.is_empty() {
            continue;
        }
        doctests.sort_by(|(_, a), (_, b)| a.name.cmp(&b.name));

        let mut tests_runner = runner::DocTestRunner::new();

        let rustdoc_test_options = IndividualTestOptions::new(
            rustdoc_options,
            &Some(format!("merged_doctest_{edition}")),
            PathBuf::from(format!("doctest_{edition}.rs")),
        );

        for (doctest, scraped_test) in &doctests {
            tests_runner.add_test(doctest, scraped_test, &target_str);
        }
        if let Ok(success) = tests_runner.run_merged_tests(
            rustdoc_test_options,
            edition,
            &opts,
            &test_args,
            rustdoc_options,
        ) {
            ran_edition_tests += 1;
            if !success {
                nb_errors += 1;
            }
            continue;
        }
        // We failed to compile all compatible tests as one so we push them into the
        // `standalone_tests` doctests.
        debug!("Failed to compile compatible doctests for edition {} all at once", edition);
        for (doctest, scraped_test) in doctests {
            doctest.generate_unique_doctest(
                &scraped_test.text,
                scraped_test.langstr.test_harness,
                &opts,
                Some(&opts.crate_name),
            );
            standalone_tests.push(generate_test_desc_and_fn(
                doctest,
                scraped_test,
                opts.clone(),
                Arc::clone(rustdoc_options),
                unused_extern_reports.clone(),
            ));
        }
    }

    // We need to call `test_main` even if there is no doctest to run to get the output
    // `running 0 tests...`.
    if ran_edition_tests == 0 || !standalone_tests.is_empty() {
        standalone_tests.sort_by(|a, b| a.desc.name.as_slice().cmp(b.desc.name.as_slice()));
        test::test_main(&test_args, standalone_tests, None);
    }
    if nb_errors != 0 {
        // libtest::ERROR_EXIT_CODE is not public but it's the same value.
        std::process::exit(101);
    }
}

// Look for `#![doc(test(no_crate_inject))]`, used by crates in the std facade.
fn scrape_test_config(
    crate_name: String,
    attrs: &[ast::Attribute],
    args_file: PathBuf,
) -> GlobalTestOptions {
    use rustc_ast_pretty::pprust;

    let mut opts = GlobalTestOptions {
        crate_name,
        no_crate_inject: false,
        attrs: Vec::new(),
        insert_indent_space: false,
        args_file,
    };

    let test_attrs: Vec<_> = attrs
        .iter()
        .filter(|a| a.has_name(sym::doc))
        .flat_map(|a| a.meta_item_list().unwrap_or_default())
        .filter(|a| a.has_name(sym::test))
        .collect();
    let attrs = test_attrs.iter().flat_map(|a| a.meta_item_list().unwrap_or(&[]));

    for attr in attrs {
        if attr.has_name(sym::no_crate_inject) {
            opts.no_crate_inject = true;
        }
        if attr.has_name(sym::attr)
            && let Some(l) = attr.meta_item_list()
        {
            for item in l {
                opts.attrs.push(pprust::meta_list_item_to_string(item));
            }
        }
    }

    opts
}

/// Documentation test failure modes.
enum TestFailure {
    /// The test failed to compile.
    CompileError,
    /// The test is marked `compile_fail` but compiled successfully.
    UnexpectedCompilePass,
    /// The test failed to compile (as expected) but the compiler output did not contain all
    /// expected error codes.
    MissingErrorCodes(Vec<String>),
    /// The test binary was unable to be executed.
    ExecutionError(io::Error),
    /// The test binary exited with a non-zero exit code.
    ///
    /// This typically means an assertion in the test failed or another form of panic occurred.
    ExecutionFailure(process::Output),
    /// The test is marked `should_panic` but the test binary executed successfully.
    UnexpectedRunPass,
}

enum DirState {
    Temp(tempfile::TempDir),
    Perm(PathBuf),
}

impl DirState {
    fn path(&self) -> &std::path::Path {
        match self {
            DirState::Temp(t) => t.path(),
            DirState::Perm(p) => p.as_path(),
        }
    }
}

// NOTE: Keep this in sync with the equivalent structs in rustc
// and cargo.
// We could unify this struct the one in rustc but they have different
// ownership semantics, so doing so would create wasteful allocations.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct UnusedExterns {
    /// Lint level of the unused_crate_dependencies lint
    lint_level: String,
    /// List of unused externs by their names.
    unused_extern_names: Vec<String>,
}

fn add_exe_suffix(input: String, target: &TargetTuple) -> String {
    let exe_suffix = match target {
        TargetTuple::TargetTuple(_) => Target::expect_builtin(target).options.exe_suffix,
        TargetTuple::TargetJson { contents, .. } => {
            Target::from_json(contents.parse().unwrap()).unwrap().0.options.exe_suffix
        }
    };
    input + &exe_suffix
}

fn wrapped_rustc_command(rustc_wrappers: &[PathBuf], rustc_binary: &Path) -> Command {
    let mut args = rustc_wrappers.iter().map(PathBuf::as_path).chain([rustc_binary]);

    let exe = args.next().expect("unable to create rustc command");
    let mut command = Command::new(exe);
    for arg in args {
        command.arg(arg);
    }

    command
}

/// Information needed for running a bundle of doctests.
///
/// This data structure contains the "full" test code, including the wrappers
/// (if multiple doctests are merged), `main` function,
/// and everything needed to calculate the compiler's command-line arguments.
/// The `# ` prefix on boring lines has also been stripped.
pub(crate) struct RunnableDocTest {
    full_test_code: String,
    full_test_line_offset: usize,
    test_opts: IndividualTestOptions,
    global_opts: GlobalTestOptions,
    langstr: LangString,
    line: usize,
    edition: Edition,
    no_run: bool,
    is_multiple_tests: bool,
}

impl RunnableDocTest {
    fn path_for_merged_doctest(&self) -> PathBuf {
        self.test_opts.outdir.path().join(format!("doctest_{}.rs", self.edition))
    }
}

/// Execute a `RunnableDoctest`.
///
/// This is the function that calculates the compiler command line, invokes the compiler, then
/// invokes the test or tests in a separate executable (if applicable).
fn run_test(
    doctest: RunnableDocTest,
    rustdoc_options: &RustdocOptions,
    supports_color: bool,
    report_unused_externs: impl Fn(UnusedExterns),
) -> Result<(), TestFailure> {
    let langstr = &doctest.langstr;
    // Make sure we emit well-formed executable names for our target.
    let rust_out = add_exe_suffix("rust_out".to_owned(), &rustdoc_options.target);
    let output_file = doctest.test_opts.outdir.path().join(rust_out);

    let rustc_binary = rustdoc_options
        .test_builder
        .as_deref()
        .unwrap_or_else(|| rustc_interface::util::rustc_path().expect("found rustc"));
    let mut compiler = wrapped_rustc_command(&rustdoc_options.test_builder_wrappers, rustc_binary);

    compiler.arg(format!("@{}", doctest.global_opts.args_file.display()));

    if let Some(sysroot) = &rustdoc_options.maybe_sysroot {
        compiler.arg(format!("--sysroot={}", sysroot.display()));
    }

    compiler.arg("--edition").arg(doctest.edition.to_string());
    if !doctest.is_multiple_tests {
        // Setting these environment variables is unneeded if this is a merged doctest.
        compiler.env("UNSTABLE_RUSTDOC_TEST_PATH", &doctest.test_opts.path);
        compiler.env(
            "UNSTABLE_RUSTDOC_TEST_LINE",
            format!("{}", doctest.line as isize - doctest.full_test_line_offset as isize),
        );
    }
    compiler.arg("-o").arg(&output_file);
    if langstr.test_harness {
        compiler.arg("--test");
    }
    if rustdoc_options.json_unused_externs.is_enabled() && !langstr.compile_fail {
        compiler.arg("--error-format=json");
        compiler.arg("--json").arg("unused-externs");
        compiler.arg("-W").arg("unused_crate_dependencies");
        compiler.arg("-Z").arg("unstable-options");
    }

    if doctest.no_run && !langstr.compile_fail && rustdoc_options.persist_doctests.is_none() {
        // FIXME: why does this code check if it *shouldn't* persist doctests
        //        -- shouldn't it be the negation?
        compiler.arg("--emit=metadata");
    }
    compiler.arg("--target").arg(match &rustdoc_options.target {
        TargetTuple::TargetTuple(s) => s,
        TargetTuple::TargetJson { path_for_rustdoc, .. } => {
            path_for_rustdoc.to_str().expect("target path must be valid unicode")
        }
    });
    if let ErrorOutputType::HumanReadable(kind, color_config) = rustdoc_options.error_format {
        let short = kind.short();
        let unicode = kind == HumanReadableErrorType::Unicode;

        if short {
            compiler.arg("--error-format").arg("short");
        }
        if unicode {
            compiler.arg("--error-format").arg("human-unicode");
        }

        match color_config {
            ColorConfig::Never => {
                compiler.arg("--color").arg("never");
            }
            ColorConfig::Always => {
                compiler.arg("--color").arg("always");
            }
            ColorConfig::Auto => {
                compiler.arg("--color").arg(if supports_color { "always" } else { "never" });
            }
        }
    }

    // If this is a merged doctest, we need to write it into a file instead of using stdin
    // because if the size of the merged doctests is too big, it'll simply break stdin.
    if doctest.is_multiple_tests {
        // It makes the compilation failure much faster if it is for a combined doctest.
        compiler.arg("--error-format=short");
        let input_file = doctest.path_for_merged_doctest();
        if std::fs::write(&input_file, &doctest.full_test_code).is_err() {
            // If we cannot write this file for any reason, we leave. All combined tests will be
            // tested as standalone tests.
            return Err(TestFailure::CompileError);
        }
        compiler.arg(input_file);
        if !rustdoc_options.nocapture {
            // If `nocapture` is disabled, then we don't display rustc's output when compiling
            // the merged doctests.
            compiler.stderr(Stdio::null());
        }
    } else {
        compiler.arg("-");
        compiler.stdin(Stdio::piped());
        compiler.stderr(Stdio::piped());
    }

    debug!("compiler invocation for doctest: {compiler:?}");

    let mut child = compiler.spawn().expect("Failed to spawn rustc process");
    let output = if doctest.is_multiple_tests {
        let status = child.wait().expect("Failed to wait");
        process::Output { status, stdout: Vec::new(), stderr: Vec::new() }
    } else {
        let stdin = child.stdin.as_mut().expect("Failed to open stdin");
        stdin.write_all(doctest.full_test_code.as_bytes()).expect("could write out test sources");
        child.wait_with_output().expect("Failed to read stdout")
    };

    struct Bomb<'a>(&'a str);
    impl Drop for Bomb<'_> {
        fn drop(&mut self) {
            eprint!("{}", self.0);
        }
    }
    let mut out = str::from_utf8(&output.stderr)
        .unwrap()
        .lines()
        .filter(|l| {
            if let Ok(uext) = serde_json::from_str::<UnusedExterns>(l) {
                report_unused_externs(uext);
                false
            } else {
                true
            }
        })
        .intersperse_with(|| "\n")
        .collect::<String>();

    // Add a \n to the end to properly terminate the last line,
    // but only if there was output to be printed
    if !out.is_empty() {
        out.push('\n');
    }

    let _bomb = Bomb(&out);
    match (output.status.success(), langstr.compile_fail) {
        (true, true) => {
            return Err(TestFailure::UnexpectedCompilePass);
        }
        (true, false) => {}
        (false, true) => {
            if !langstr.error_codes.is_empty() {
                // We used to check if the output contained "error[{}]: " but since we added the
                // colored output, we can't anymore because of the color escape characters before
                // the ":".
                let missing_codes: Vec<String> = langstr
                    .error_codes
                    .iter()
                    .filter(|err| !out.contains(&format!("error[{err}]")))
                    .cloned()
                    .collect();

                if !missing_codes.is_empty() {
                    return Err(TestFailure::MissingErrorCodes(missing_codes));
                }
            }
        }
        (false, false) => {
            return Err(TestFailure::CompileError);
        }
    }

    if doctest.no_run {
        return Ok(());
    }

    // Run the code!
    let mut cmd;

    let output_file = make_maybe_absolute_path(output_file);
    if let Some(tool) = &rustdoc_options.runtool {
        let tool = make_maybe_absolute_path(tool.into());
        cmd = Command::new(tool);
        cmd.args(&rustdoc_options.runtool_args);
        cmd.arg(&output_file);
    } else {
        cmd = Command::new(&output_file);
        if doctest.is_multiple_tests {
            cmd.env("RUSTDOC_DOCTEST_BIN_PATH", &output_file);
        }
    }
    if let Some(run_directory) = &rustdoc_options.test_run_directory {
        cmd.current_dir(run_directory);
    }

    let result = if doctest.is_multiple_tests || rustdoc_options.nocapture {
        cmd.status().map(|status| process::Output {
            status,
            stdout: Vec::new(),
            stderr: Vec::new(),
        })
    } else {
        cmd.output()
    };
    match result {
        Err(e) => return Err(TestFailure::ExecutionError(e)),
        Ok(out) => {
            if langstr.should_panic && out.status.success() {
                return Err(TestFailure::UnexpectedRunPass);
            } else if !langstr.should_panic && !out.status.success() {
                return Err(TestFailure::ExecutionFailure(out));
            }
        }
    }

    Ok(())
}

/// Converts a path intended to use as a command to absolute if it is
/// relative, and not a single component.
///
/// This is needed to deal with relative paths interacting with
/// `Command::current_dir` in a platform-specific way.
fn make_maybe_absolute_path(path: PathBuf) -> PathBuf {
    if path.components().count() == 1 {
        // Look up process via PATH.
        path
    } else {
        std::env::current_dir().map(|c| c.join(&path)).unwrap_or_else(|_| path)
    }
}
struct IndividualTestOptions {
    outdir: DirState,
    path: PathBuf,
}

impl IndividualTestOptions {
    fn new(options: &RustdocOptions, test_id: &Option<String>, test_path: PathBuf) -> Self {
        let outdir = if let Some(ref path) = options.persist_doctests {
            let mut path = path.clone();
            path.push(test_id.as_deref().unwrap_or("<doctest>"));

            if let Err(err) = std::fs::create_dir_all(&path) {
                eprintln!("Couldn't create directory for doctest executables: {err}");
                panic::resume_unwind(Box::new(()));
            }

            DirState::Perm(path)
        } else {
            DirState::Temp(get_doctest_dir().expect("rustdoc needs a tempdir"))
        };

        Self { outdir, path: test_path }
    }
}

/// A doctest scraped from the code, ready to be turned into a runnable test.
///
/// The pipeline goes: [`clean`] AST -> `ScrapedDoctest` -> `RunnableDoctest`.
/// [`run_merged_tests`] converts a bunch of scraped doctests to a single runnable doctest,
/// while [`generate_unique_doctest`] does the standalones.
///
/// [`clean`]: crate::clean
/// [`run_merged_tests`]: crate::doctest::runner::DocTestRunner::run_merged_tests
/// [`generate_unique_doctest`]: crate::doctest::make::DocTestBuilder::generate_unique_doctest
pub(crate) struct ScrapedDocTest {
    filename: FileName,
    line: usize,
    langstr: LangString,
    text: String,
    name: String,
}

impl ScrapedDocTest {
    fn new(
        filename: FileName,
        line: usize,
        logical_path: Vec<String>,
        langstr: LangString,
        text: String,
    ) -> Self {
        let mut item_path = logical_path.join("::");
        item_path.retain(|c| c != ' ');
        if !item_path.is_empty() {
            item_path.push(' ');
        }
        let name =
            format!("{} - {item_path}(line {line})", filename.prefer_remapped_unconditionaly());

        Self { filename, line, langstr, text, name }
    }
    fn edition(&self, opts: &RustdocOptions) -> Edition {
        self.langstr.edition.unwrap_or(opts.edition)
    }

    fn no_run(&self, opts: &RustdocOptions) -> bool {
        self.langstr.no_run || opts.no_run
    }
    fn path(&self) -> PathBuf {
        match &self.filename {
            FileName::Real(path) => {
                if let Some(local_path) = path.local_path() {
                    local_path.to_path_buf()
                } else {
                    // Somehow we got the filename from the metadata of another crate, should never happen
                    unreachable!("doctest from a different crate");
                }
            }
            _ => PathBuf::from(r"doctest.rs"),
        }
    }
}

pub(crate) trait DocTestVisitor {
    fn visit_test(&mut self, test: String, config: LangString, rel_line: MdRelLine);
    fn visit_header(&mut self, _name: &str, _level: u32) {}
}

struct CreateRunnableDocTests {
    standalone_tests: Vec<test::TestDescAndFn>,
    mergeable_tests: FxIndexMap<Edition, Vec<(DocTestBuilder, ScrapedDocTest)>>,

    rustdoc_options: Arc<RustdocOptions>,
    opts: GlobalTestOptions,
    visited_tests: FxHashMap<(String, usize), usize>,
    unused_extern_reports: Arc<Mutex<Vec<UnusedExterns>>>,
    compiling_test_count: AtomicUsize,
    can_merge_doctests: bool,
}

impl CreateRunnableDocTests {
    fn new(rustdoc_options: RustdocOptions, opts: GlobalTestOptions) -> CreateRunnableDocTests {
        let can_merge_doctests = rustdoc_options.edition >= Edition::Edition2024;
        CreateRunnableDocTests {
            standalone_tests: Vec::new(),
            mergeable_tests: FxIndexMap::default(),
            rustdoc_options: Arc::new(rustdoc_options),
            opts,
            visited_tests: FxHashMap::default(),
            unused_extern_reports: Default::default(),
            compiling_test_count: AtomicUsize::new(0),
            can_merge_doctests,
        }
    }

    fn add_test(&mut self, scraped_test: ScrapedDocTest) {
        // For example `module/file.rs` would become `module_file_rs`
        let file = scraped_test
            .filename
            .prefer_local()
            .to_string_lossy()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect::<String>();
        let test_id = format!(
            "{file}_{line}_{number}",
            file = file,
            line = scraped_test.line,
            number = {
                // Increases the current test number, if this file already
                // exists or it creates a new entry with a test number of 0.
                self.visited_tests
                    .entry((file.clone(), scraped_test.line))
                    .and_modify(|v| *v += 1)
                    .or_insert(0)
            },
        );

        let edition = scraped_test.edition(&self.rustdoc_options);
        let doctest = DocTestBuilder::new(
            &scraped_test.text,
            Some(&self.opts.crate_name),
            edition,
            self.can_merge_doctests,
            Some(test_id),
            Some(&scraped_test.langstr),
        );
        let is_standalone = !doctest.can_be_merged
            || scraped_test.langstr.compile_fail
            || scraped_test.langstr.test_harness
            || scraped_test.langstr.standalone_crate
            || self.rustdoc_options.nocapture
            || self.rustdoc_options.test_args.iter().any(|arg| arg == "--show-output");
        if is_standalone {
            let test_desc = self.generate_test_desc_and_fn(doctest, scraped_test);
            self.standalone_tests.push(test_desc);
        } else {
            self.mergeable_tests.entry(edition).or_default().push((doctest, scraped_test));
        }
    }

    fn generate_test_desc_and_fn(
        &mut self,
        test: DocTestBuilder,
        scraped_test: ScrapedDocTest,
    ) -> test::TestDescAndFn {
        if !scraped_test.langstr.compile_fail {
            self.compiling_test_count.fetch_add(1, Ordering::SeqCst);
        }

        generate_test_desc_and_fn(
            test,
            scraped_test,
            self.opts.clone(),
            Arc::clone(&self.rustdoc_options),
            self.unused_extern_reports.clone(),
        )
    }
}

fn generate_test_desc_and_fn(
    test: DocTestBuilder,
    scraped_test: ScrapedDocTest,
    opts: GlobalTestOptions,
    rustdoc_options: Arc<RustdocOptions>,
    unused_externs: Arc<Mutex<Vec<UnusedExterns>>>,
) -> test::TestDescAndFn {
    let target_str = rustdoc_options.target.to_string();
    let rustdoc_test_options =
        IndividualTestOptions::new(&rustdoc_options, &test.test_id, scraped_test.path());

    debug!("creating test {}: {}", scraped_test.name, scraped_test.text);
    test::TestDescAndFn {
        desc: test::TestDesc {
            name: test::DynTestName(scraped_test.name.clone()),
            ignore: match scraped_test.langstr.ignore {
                Ignore::All => true,
                Ignore::None => false,
                Ignore::Some(ref ignores) => ignores.iter().any(|s| target_str.contains(s)),
            },
            ignore_message: None,
            source_file: "",
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            // compiler failures are test failures
            should_panic: test::ShouldPanic::No,
            compile_fail: scraped_test.langstr.compile_fail,
            no_run: scraped_test.no_run(&rustdoc_options),
            test_type: test::TestType::DocTest,
        },
        testfn: test::DynTestFn(Box::new(move || {
            doctest_run_fn(
                rustdoc_test_options,
                opts,
                test,
                scraped_test,
                rustdoc_options,
                unused_externs,
            )
        })),
    }
}

fn doctest_run_fn(
    test_opts: IndividualTestOptions,
    global_opts: GlobalTestOptions,
    doctest: DocTestBuilder,
    scraped_test: ScrapedDocTest,
    rustdoc_options: Arc<RustdocOptions>,
    unused_externs: Arc<Mutex<Vec<UnusedExterns>>>,
) -> Result<(), String> {
    let report_unused_externs = |uext| {
        unused_externs.lock().unwrap().push(uext);
    };
    let (full_test_code, full_test_line_offset) = doctest.generate_unique_doctest(
        &scraped_test.text,
        scraped_test.langstr.test_harness,
        &global_opts,
        Some(&global_opts.crate_name),
    );
    let runnable_test = RunnableDocTest {
        full_test_code,
        full_test_line_offset,
        test_opts,
        global_opts,
        langstr: scraped_test.langstr.clone(),
        line: scraped_test.line,
        edition: scraped_test.edition(&rustdoc_options),
        no_run: scraped_test.no_run(&rustdoc_options),
        is_multiple_tests: false,
    };
    let res =
        run_test(runnable_test, &rustdoc_options, doctest.supports_color, report_unused_externs);

    if let Err(err) = res {
        match err {
            TestFailure::CompileError => {
                eprint!("Couldn't compile the test.");
            }
            TestFailure::UnexpectedCompilePass => {
                eprint!("Test compiled successfully, but it's marked `compile_fail`.");
            }
            TestFailure::UnexpectedRunPass => {
                eprint!("Test executable succeeded, but it's marked `should_panic`.");
            }
            TestFailure::MissingErrorCodes(codes) => {
                eprint!("Some expected error codes were not found: {codes:?}");
            }
            TestFailure::ExecutionError(err) => {
                eprint!("Couldn't run the test: {err}");
                if err.kind() == io::ErrorKind::PermissionDenied {
                    eprint!(" - maybe your tempdir is mounted with noexec?");
                }
            }
            TestFailure::ExecutionFailure(out) => {
                eprintln!("Test executable failed ({reason}).", reason = out.status);

                // FIXME(#12309): An unfortunate side-effect of capturing the test
                // executable's output is that the relative ordering between the test's
                // stdout and stderr is lost. However, this is better than the
                // alternative: if the test executable inherited the parent's I/O
                // handles the output wouldn't be captured at all, even on success.
                //
                // The ordering could be preserved if the test process' stderr was
                // redirected to stdout, but that functionality does not exist in the
                // standard library, so it may not be portable enough.
                let stdout = str::from_utf8(&out.stdout).unwrap_or_default();
                let stderr = str::from_utf8(&out.stderr).unwrap_or_default();

                if !stdout.is_empty() || !stderr.is_empty() {
                    eprintln!();

                    if !stdout.is_empty() {
                        eprintln!("stdout:\n{stdout}");
                    }

                    if !stderr.is_empty() {
                        eprintln!("stderr:\n{stderr}");
                    }
                }
            }
        }

        panic::resume_unwind(Box::new(()));
    }
    Ok(())
}

#[cfg(test)] // used in tests
impl DocTestVisitor for Vec<usize> {
    fn visit_test(&mut self, _test: String, _config: LangString, rel_line: MdRelLine) {
        self.push(1 + rel_line.offset());
    }
}

#[cfg(test)]
mod tests;
