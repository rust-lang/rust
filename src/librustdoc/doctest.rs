mod markdown;
mod rust;

pub(crate) use markdown::test as test_markdown;

use rustc_ast as ast;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lrc;
use rustc_errors::emitter::stderr_destination;
use rustc_errors::{ColorConfig, ErrorGuaranteed, FatalError};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_hir::CRATE_HIR_ID;
use rustc_interface::interface;
use rustc_parse::new_parser_from_source_str;
use rustc_parse::parser::attr::InnerAttrPolicy;
use rustc_session::config::{self, CrateType, ErrorOutputType};
use rustc_session::lint;
use rustc_session::parse::ParseSess;
use rustc_span::edition::Edition;
use rustc_span::source_map::SourceMap;
use rustc_span::symbol::sym;
use rustc_span::FileName;
use rustc_target::spec::{Target, TargetTriple};

use std::fs::File;
use std::io::{self, Write};
use std::panic;
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::str;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tempfile::{Builder as TempFileBuilder, TempDir};

use crate::config::Options as RustdocOptions;
use crate::html::markdown::{ErrorCodes, Ignore, LangString, MdRelLine};
use crate::lint::init_lints;

use self::rust::HirCollector;

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
    dcx: &rustc_errors::DiagCtxt,
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
        ..config::Options::default()
    };

    let mut cfgs = options.cfgs.clone();
    cfgs.push("doc".to_owned());
    cfgs.push("doctest".to_owned());
    let config = interface::Config {
        opts: sessopts,
        crate_cfg: cfgs,
        crate_check_cfg: options.check_cfgs.clone(),
        input: options.input.clone(),
        output_file: None,
        output_dir: None,
        file_loader: None,
        locale_resources: rustc_driver::DEFAULT_LOCALE_RESOURCES,
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

    let test_args = options.test_args.clone();
    let nocapture = options.nocapture;
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

    let (tests, unused_extern_reports, compiling_test_count) =
        interface::run_compiler(config, |compiler| {
            compiler.enter(|queries| {
                let collector = queries.global_ctxt()?.enter(|tcx| {
                    let crate_name = tcx.crate_name(LOCAL_CRATE).to_string();
                    let crate_attrs = tcx.hir().attrs(CRATE_HIR_ID);
                    let opts = scrape_test_config(crate_name, crate_attrs, args_path);
                    let enable_per_target_ignores = options.enable_per_target_ignores;

                    let mut collector = CreateRunnableDoctests::new(options, opts);
                    let hir_collector = HirCollector::new(
                        &compiler.sess,
                        tcx.hir(),
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

                let unused_extern_reports = collector.unused_extern_reports.clone();
                let compiling_test_count = collector.compiling_test_count.load(Ordering::SeqCst);
                Ok((collector.tests, unused_extern_reports, compiling_test_count))
            })
        })?;

    run_tests(test_args, nocapture, tests);

    // Collect and warn about unused externs, but only if we've gotten
    // reports for each doctest
    if json_unused_externs.is_enabled() {
        let unused_extern_reports: Vec<_> =
            std::mem::take(&mut unused_extern_reports.lock().unwrap());
        if unused_extern_reports.len() == compiling_test_count {
            let extern_names = externs.iter().map(|(name, _)| name).collect::<FxHashSet<&String>>();
            let mut unused_extern_names = unused_extern_reports
                .iter()
                .map(|uexts| uexts.unused_extern_names.iter().collect::<FxHashSet<&String>>())
                .fold(extern_names, |uextsa, uextsb| {
                    uextsa.intersection(&uextsb).copied().collect::<FxHashSet<&String>>()
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
    mut test_args: Vec<String>,
    nocapture: bool,
    mut tests: Vec<test::TestDescAndFn>,
) {
    test_args.insert(0, "rustdoctest".to_string());
    if nocapture {
        test_args.push("--nocapture".to_string());
    }
    tests.sort_by(|a, b| a.desc.name.as_slice().cmp(&b.desc.name.as_slice()));
    test::test_main(&test_args, tests, None);
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
struct UnusedExterns {
    /// Lint level of the unused_crate_dependencies lint
    lint_level: String,
    /// List of unused externs by their names.
    unused_extern_names: Vec<String>,
}

fn add_exe_suffix(input: String, target: &TargetTriple) -> String {
    let exe_suffix = match target {
        TargetTriple::TargetTriple(_) => Target::expect_builtin(target).options.exe_suffix,
        TargetTriple::TargetJson { contents, .. } => {
            Target::from_json(contents.parse().unwrap()).unwrap().0.options.exe_suffix
        }
    };
    input + &exe_suffix
}

fn wrapped_rustc_command(rustc_wrappers: &[PathBuf], rustc_binary: &Path) -> Command {
    let mut args = rustc_wrappers.iter().map(PathBuf::as_path).chain([rustc_binary].into_iter());

    let exe = args.next().expect("unable to create rustc command");
    let mut command = Command::new(exe);
    for arg in args {
        command.arg(arg);
    }

    command
}

struct RunnableDoctest {
    full_test_code: String,
    full_test_line_offset: usize,
    test_opts: IndividualTestOptions,
    global_opts: GlobalTestOptions,
    scraped_test: ScrapedDoctest,
}

fn run_test(
    doctest: RunnableDoctest,
    rustdoc_options: &RustdocOptions,
    supports_color: bool,
    report_unused_externs: impl Fn(UnusedExterns),
) -> Result<(), TestFailure> {
    let scraped_test = &doctest.scraped_test;
    let langstr = &scraped_test.langstr;
    // Make sure we emit well-formed executable names for our target.
    let rust_out = add_exe_suffix("rust_out".to_owned(), &rustdoc_options.target);
    let output_file = doctest.test_opts.outdir.path().join(rust_out);

    let rustc_binary = rustdoc_options
        .test_builder
        .as_deref()
        .unwrap_or_else(|| rustc_interface::util::rustc_path().expect("found rustc"));
    let mut compiler = wrapped_rustc_command(&rustdoc_options.test_builder_wrappers, rustc_binary);

    compiler.arg(&format!("@{}", doctest.global_opts.args_file.display()));

    if let Some(sysroot) = &rustdoc_options.maybe_sysroot {
        compiler.arg(format!("--sysroot={}", sysroot.display()));
    }

    compiler.arg("--edition").arg(&scraped_test.edition(rustdoc_options).to_string());
    compiler.env("UNSTABLE_RUSTDOC_TEST_PATH", &doctest.test_opts.path);
    compiler.env(
        "UNSTABLE_RUSTDOC_TEST_LINE",
        format!("{}", scraped_test.line as isize - doctest.full_test_line_offset as isize),
    );
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

    if scraped_test.no_run(rustdoc_options)
        && !langstr.compile_fail
        && rustdoc_options.persist_doctests.is_none()
    {
        // FIXME: why does this code check if it *shouldn't* persist doctests
        //        -- shouldn't it be the negation?
        compiler.arg("--emit=metadata");
    }
    compiler.arg("--target").arg(match &rustdoc_options.target {
        TargetTriple::TargetTriple(s) => s,
        TargetTriple::TargetJson { path_for_rustdoc, .. } => {
            path_for_rustdoc.to_str().expect("target path must be valid unicode")
        }
    });
    if let ErrorOutputType::HumanReadable(kind) = rustdoc_options.error_format {
        let (short, color_config) = kind.unzip();

        if short {
            compiler.arg("--error-format").arg("short");
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

    compiler.arg("-");
    compiler.stdin(Stdio::piped());
    compiler.stderr(Stdio::piped());

    debug!("compiler invocation for doctest: {compiler:?}");

    let mut child = compiler.spawn().expect("Failed to spawn rustc process");
    {
        let stdin = child.stdin.as_mut().expect("Failed to open stdin");
        stdin.write_all(doctest.full_test_code.as_bytes()).expect("could write out test sources");
    }
    let output = child.wait_with_output().expect("Failed to read stdout");

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
                let missing_codes: Vec<String> = scraped_test
                    .langstr
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

    if scraped_test.no_run(rustdoc_options) {
        return Ok(());
    }

    // Run the code!
    let mut cmd;

    let output_file = make_maybe_absolute_path(output_file);
    if let Some(tool) = &rustdoc_options.runtool {
        let tool = make_maybe_absolute_path(tool.into());
        cmd = Command::new(tool);
        cmd.args(&rustdoc_options.runtool_args);
        cmd.arg(output_file);
    } else {
        cmd = Command::new(output_file);
    }
    if let Some(run_directory) = &rustdoc_options.test_run_directory {
        cmd.current_dir(run_directory);
    }

    let result = if rustdoc_options.nocapture {
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

/// Transforms a test into code that can be compiled into a Rust binary, and returns the number of
/// lines before the test code begins as well as if the output stream supports colors or not.
pub(crate) fn make_test(
    s: &str,
    crate_name: Option<&str>,
    dont_insert_main: bool,
    opts: &GlobalTestOptions,
    edition: Edition,
    test_id: Option<&str>,
) -> (String, usize, bool) {
    let (crate_attrs, everything_else, crates) = partition_source(s, edition);
    let everything_else = everything_else.trim();
    let mut line_offset = 0;
    let mut prog = String::new();
    let mut supports_color = false;

    if opts.attrs.is_empty() {
        // If there aren't any attributes supplied by #![doc(test(attr(...)))], then allow some
        // lints that are commonly triggered in doctests. The crate-level test attributes are
        // commonly used to make tests fail in case they trigger warnings, so having this there in
        // that case may cause some tests to pass when they shouldn't have.
        prog.push_str("#![allow(unused)]\n");
        line_offset += 1;
    }

    // Next, any attributes that came from the crate root via #![doc(test(attr(...)))].
    for attr in &opts.attrs {
        prog.push_str(&format!("#![{attr}]\n"));
        line_offset += 1;
    }

    // Now push any outer attributes from the example, assuming they
    // are intended to be crate attributes.
    prog.push_str(&crate_attrs);
    prog.push_str(&crates);

    // Uses librustc_ast to parse the doctest and find if there's a main fn and the extern
    // crate already is included.
    let result = rustc_driver::catch_fatal_errors(|| {
        rustc_span::create_session_if_not_set_then(edition, |_| {
            use rustc_errors::emitter::{Emitter, HumanEmitter};
            use rustc_errors::DiagCtxt;
            use rustc_parse::parser::ForceCollect;
            use rustc_span::source_map::FilePathMapping;

            let filename = FileName::anon_source_code(s);
            let source = crates + everything_else;

            // Any errors in parsing should also appear when the doctest is compiled for real, so just
            // send all the errors that librustc_ast emits directly into a `Sink` instead of stderr.
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let fallback_bundle = rustc_errors::fallback_fluent_bundle(
                rustc_driver::DEFAULT_LOCALE_RESOURCES.to_vec(),
                false,
            );
            supports_color =
                HumanEmitter::new(stderr_destination(ColorConfig::Auto), fallback_bundle.clone())
                    .supports_color();

            let emitter = HumanEmitter::new(Box::new(io::sink()), fallback_bundle);

            // FIXME(misdreavus): pass `-Z treat-err-as-bug` to the doctest parser
            let dcx = DiagCtxt::new(Box::new(emitter)).disable_warnings();
            let psess = ParseSess::with_dcx(dcx, sm);

            let mut found_main = false;
            let mut found_extern_crate = crate_name.is_none();
            let mut found_macro = false;

            let mut parser = match new_parser_from_source_str(&psess, filename, source) {
                Ok(p) => p,
                Err(errs) => {
                    errs.into_iter().for_each(|err| err.cancel());
                    return (found_main, found_extern_crate, found_macro);
                }
            };

            loop {
                match parser.parse_item(ForceCollect::No) {
                    Ok(Some(item)) => {
                        if !found_main
                            && let ast::ItemKind::Fn(..) = item.kind
                            && item.ident.name == sym::main
                        {
                            found_main = true;
                        }

                        if !found_extern_crate
                            && let ast::ItemKind::ExternCrate(original) = item.kind
                        {
                            // This code will never be reached if `crate_name` is none because
                            // `found_extern_crate` is initialized to `true` if it is none.
                            let crate_name = crate_name.unwrap();

                            match original {
                                Some(name) => found_extern_crate = name.as_str() == crate_name,
                                None => found_extern_crate = item.ident.as_str() == crate_name,
                            }
                        }

                        if !found_macro && let ast::ItemKind::MacCall(..) = item.kind {
                            found_macro = true;
                        }

                        if found_main && found_extern_crate {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        e.cancel();
                        break;
                    }
                }

                // The supplied item is only used for diagnostics,
                // which are swallowed here anyway.
                parser.maybe_consume_incorrect_semicolon(None);
            }

            // Reset errors so that they won't be reported as compiler bugs when dropping the
            // dcx. Any errors in the tests will be reported when the test file is compiled,
            // Note that we still need to cancel the errors above otherwise `Diag` will panic on
            // drop.
            psess.dcx.reset_err_count();

            (found_main, found_extern_crate, found_macro)
        })
    });
    let Ok((already_has_main, already_has_extern_crate, found_macro)) = result else {
        // If the parser panicked due to a fatal error, pass the test code through unchanged.
        // The error will be reported during compilation.
        return (s.to_owned(), 0, false);
    };

    // If a doctest's `fn main` is being masked by a wrapper macro, the parsing loop above won't
    // see it. In that case, run the old text-based scan to see if they at least have a main
    // function written inside a macro invocation. See
    // https://github.com/rust-lang/rust/issues/56898
    let already_has_main = if found_macro && !already_has_main {
        s.lines()
            .map(|line| {
                let comment = line.find("//");
                if let Some(comment_begins) = comment { &line[0..comment_begins] } else { line }
            })
            .any(|code| code.contains("fn main"))
    } else {
        already_has_main
    };

    // Don't inject `extern crate std` because it's already injected by the
    // compiler.
    if !already_has_extern_crate && !opts.no_crate_inject && crate_name != Some("std") {
        if let Some(crate_name) = crate_name {
            // Don't inject `extern crate` if the crate is never used.
            // NOTE: this is terribly inaccurate because it doesn't actually
            // parse the source, but only has false positives, not false
            // negatives.
            if s.contains(crate_name) {
                // rustdoc implicitly inserts an `extern crate` item for the own crate
                // which may be unused, so we need to allow the lint.
                prog.push_str("#[allow(unused_extern_crates)]\n");

                prog.push_str(&format!("extern crate r#{crate_name};\n"));
                line_offset += 1;
            }
        }
    }

    // FIXME: This code cannot yet handle no_std test cases yet
    if dont_insert_main || already_has_main || prog.contains("![no_std]") {
        prog.push_str(everything_else);
    } else {
        let returns_result = everything_else.trim_end().ends_with("(())");
        // Give each doctest main function a unique name.
        // This is for example needed for the tooling around `-C instrument-coverage`.
        let inner_fn_name = if let Some(test_id) = test_id {
            format!("_doctest_main_{test_id}")
        } else {
            "_inner".into()
        };
        let inner_attr = if test_id.is_some() { "#[allow(non_snake_case)] " } else { "" };
        let (main_pre, main_post) = if returns_result {
            (
                format!(
                    "fn main() {{ {inner_attr}fn {inner_fn_name}() -> Result<(), impl core::fmt::Debug> {{\n",
                ),
                format!("\n}} {inner_fn_name}().unwrap() }}"),
            )
        } else if test_id.is_some() {
            (
                format!("fn main() {{ {inner_attr}fn {inner_fn_name}() {{\n",),
                format!("\n}} {inner_fn_name}() }}"),
            )
        } else {
            ("fn main() {\n".into(), "\n}".into())
        };
        // Note on newlines: We insert a line/newline *before*, and *after*
        // the doctest and adjust the `line_offset` accordingly.
        // In the case of `-C instrument-coverage`, this means that the generated
        // inner `main` function spans from the doctest opening codeblock to the
        // closing one. For example
        // /// ``` <- start of the inner main
        // /// <- code under doctest
        // /// ``` <- end of the inner main
        line_offset += 1;

        // add extra 4 spaces for each line to offset the code block
        let content = if opts.insert_indent_space {
            everything_else
                .lines()
                .map(|line| format!("    {}", line))
                .collect::<Vec<String>>()
                .join("\n")
        } else {
            everything_else.to_string()
        };
        prog.extend([&main_pre, content.as_str(), &main_post].iter().cloned());
    }

    debug!("final doctest:\n{prog}");

    (prog, line_offset, supports_color)
}

fn check_if_attr_is_complete(source: &str, edition: Edition) -> bool {
    if source.is_empty() {
        // Empty content so nothing to check in here...
        return true;
    }
    rustc_driver::catch_fatal_errors(|| {
        rustc_span::create_session_if_not_set_then(edition, |_| {
            use rustc_errors::emitter::HumanEmitter;
            use rustc_errors::DiagCtxt;
            use rustc_span::source_map::FilePathMapping;

            let filename = FileName::anon_source_code(source);
            // Any errors in parsing should also appear when the doctest is compiled for real, so just
            // send all the errors that librustc_ast emits directly into a `Sink` instead of stderr.
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let fallback_bundle = rustc_errors::fallback_fluent_bundle(
                rustc_driver::DEFAULT_LOCALE_RESOURCES.to_vec(),
                false,
            );

            let emitter = HumanEmitter::new(Box::new(io::sink()), fallback_bundle);

            let dcx = DiagCtxt::new(Box::new(emitter)).disable_warnings();
            let psess = ParseSess::with_dcx(dcx, sm);
            let mut parser = match new_parser_from_source_str(&psess, filename, source.to_owned()) {
                Ok(p) => p,
                Err(errs) => {
                    errs.into_iter().for_each(|err| err.cancel());
                    // If there is an unclosed delimiter, an error will be returned by the
                    // tokentrees.
                    return false;
                }
            };
            // If a parsing error happened, it's very likely that the attribute is incomplete.
            if let Err(e) = parser.parse_attribute(InnerAttrPolicy::Permitted) {
                e.cancel();
                return false;
            }
            true
        })
    })
    .unwrap_or(false)
}

fn partition_source(s: &str, edition: Edition) -> (String, String, String) {
    #[derive(Copy, Clone, PartialEq)]
    enum PartitionState {
        Attrs,
        Crates,
        Other,
    }
    let mut state = PartitionState::Attrs;
    let mut before = String::new();
    let mut crates = String::new();
    let mut after = String::new();

    let mut mod_attr_pending = String::new();

    for line in s.lines() {
        let trimline = line.trim();

        // FIXME(misdreavus): if a doc comment is placed on an extern crate statement, it will be
        // shunted into "everything else"
        match state {
            PartitionState::Attrs => {
                state = if trimline.starts_with("#![") {
                    if !check_if_attr_is_complete(line, edition) {
                        mod_attr_pending = line.to_owned();
                    } else {
                        mod_attr_pending.clear();
                    }
                    PartitionState::Attrs
                } else if trimline.chars().all(|c| c.is_whitespace())
                    || (trimline.starts_with("//") && !trimline.starts_with("///"))
                {
                    PartitionState::Attrs
                } else if trimline.starts_with("extern crate")
                    || trimline.starts_with("#[macro_use] extern crate")
                {
                    PartitionState::Crates
                } else {
                    // First we check if the previous attribute was "complete"...
                    if !mod_attr_pending.is_empty() {
                        // If not, then we append the new line into the pending attribute to check
                        // if this time it's complete...
                        mod_attr_pending.push_str(line);
                        if !trimline.is_empty()
                            && check_if_attr_is_complete(&mod_attr_pending, edition)
                        {
                            // If it's complete, then we can clear the pending content.
                            mod_attr_pending.clear();
                        }
                        // In any case, this is considered as `PartitionState::Attrs` so it's
                        // prepended before rustdoc's inserts.
                        PartitionState::Attrs
                    } else {
                        PartitionState::Other
                    }
                };
            }
            PartitionState::Crates => {
                state = if trimline.starts_with("extern crate")
                    || trimline.starts_with("#[macro_use] extern crate")
                    || trimline.chars().all(|c| c.is_whitespace())
                    || (trimline.starts_with("//") && !trimline.starts_with("///"))
                {
                    PartitionState::Crates
                } else {
                    PartitionState::Other
                };
            }
            PartitionState::Other => {}
        }

        match state {
            PartitionState::Attrs => {
                before.push_str(line);
                before.push('\n');
            }
            PartitionState::Crates => {
                crates.push_str(line);
                crates.push('\n');
            }
            PartitionState::Other => {
                after.push_str(line);
                after.push('\n');
            }
        }
    }

    debug!("before:\n{before}");
    debug!("crates:\n{crates}");
    debug!("after:\n{after}");

    (before, after, crates)
}

pub(crate) struct IndividualTestOptions {
    outdir: DirState,
    test_id: String,
    path: PathBuf,
}

impl IndividualTestOptions {
    fn new(options: &RustdocOptions, test_id: String, test_path: PathBuf) -> Self {
        let outdir = if let Some(ref path) = options.persist_doctests {
            let mut path = path.clone();
            path.push(&test_id);

            if let Err(err) = std::fs::create_dir_all(&path) {
                eprintln!("Couldn't create directory for doctest executables: {err}");
                panic::resume_unwind(Box::new(()));
            }

            DirState::Perm(path)
        } else {
            DirState::Temp(get_doctest_dir().expect("rustdoc needs a tempdir"))
        };

        Self { outdir, test_id, path: test_path }
    }
}

/// A doctest scraped from the code, ready to be turned into a runnable test.
struct ScrapedDoctest {
    filename: FileName,
    line: usize,
    logical_path: Vec<String>,
    langstr: LangString,
    text: String,
}

impl ScrapedDoctest {
    fn edition(&self, opts: &RustdocOptions) -> Edition {
        self.langstr.edition.unwrap_or(opts.edition)
    }

    fn no_run(&self, opts: &RustdocOptions) -> bool {
        self.langstr.no_run || opts.no_run
    }
}

pub(crate) trait DoctestVisitor {
    fn visit_test(&mut self, test: String, config: LangString, rel_line: MdRelLine);
    fn visit_header(&mut self, _name: &str, _level: u32) {}
}

pub(crate) struct CreateRunnableDoctests {
    pub(crate) tests: Vec<test::TestDescAndFn>,

    rustdoc_options: Arc<RustdocOptions>,
    opts: GlobalTestOptions,
    visited_tests: FxHashMap<(String, usize), usize>,
    unused_extern_reports: Arc<Mutex<Vec<UnusedExterns>>>,
    compiling_test_count: AtomicUsize,
}

impl CreateRunnableDoctests {
    pub(crate) fn new(
        rustdoc_options: RustdocOptions,
        opts: GlobalTestOptions,
    ) -> CreateRunnableDoctests {
        CreateRunnableDoctests {
            tests: Vec::new(),
            rustdoc_options: Arc::new(rustdoc_options),
            opts,
            visited_tests: FxHashMap::default(),
            unused_extern_reports: Default::default(),
            compiling_test_count: AtomicUsize::new(0),
        }
    }

    fn generate_name(&self, filename: &FileName, line: usize, logical_path: &[String]) -> String {
        let mut item_path = logical_path.join("::");
        item_path.retain(|c| c != ' ');
        if !item_path.is_empty() {
            item_path.push(' ');
        }
        format!("{} - {item_path}(line {line})", filename.prefer_local())
    }

    fn add_test(&mut self, test: ScrapedDoctest) {
        let name = self.generate_name(&test.filename, test.line, &test.logical_path);
        let opts = self.opts.clone();
        let target_str = self.rustdoc_options.target.to_string();
        let unused_externs = self.unused_extern_reports.clone();
        if !test.langstr.compile_fail {
            self.compiling_test_count.fetch_add(1, Ordering::SeqCst);
        }

        let path = match &test.filename {
            FileName::Real(path) => {
                if let Some(local_path) = path.local_path() {
                    local_path.to_path_buf()
                } else {
                    // Somehow we got the filename from the metadata of another crate, should never happen
                    unreachable!("doctest from a different crate");
                }
            }
            _ => PathBuf::from(r"doctest.rs"),
        };

        // For example `module/file.rs` would become `module_file_rs`
        let file = test
            .filename
            .prefer_local()
            .to_string_lossy()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect::<String>();
        let test_id = format!(
            "{file}_{line}_{number}",
            file = file,
            line = test.line,
            number = {
                // Increases the current test number, if this file already
                // exists or it creates a new entry with a test number of 0.
                self.visited_tests
                    .entry((file.clone(), test.line))
                    .and_modify(|v| *v += 1)
                    .or_insert(0)
            },
        );

        let rustdoc_options = self.rustdoc_options.clone();
        let rustdoc_test_options = IndividualTestOptions::new(&self.rustdoc_options, test_id, path);

        debug!("creating test {name}: {}", test.text);
        self.tests.push(test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::DynTestName(name),
                ignore: match test.langstr.ignore {
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
                compile_fail: test.langstr.compile_fail,
                no_run: test.no_run(&rustdoc_options),
                test_type: test::TestType::DocTest,
            },
            testfn: test::DynTestFn(Box::new(move || {
                doctest_run_fn(rustdoc_test_options, opts, test, rustdoc_options, unused_externs)
            })),
        });
    }
}

fn doctest_run_fn(
    test_opts: IndividualTestOptions,
    global_opts: GlobalTestOptions,
    scraped_test: ScrapedDoctest,
    rustdoc_options: Arc<RustdocOptions>,
    unused_externs: Arc<Mutex<Vec<UnusedExterns>>>,
) -> Result<(), String> {
    let report_unused_externs = |uext| {
        unused_externs.lock().unwrap().push(uext);
    };
    let edition = scraped_test.edition(&rustdoc_options);
    let (full_test_code, full_test_line_offset, supports_color) = make_test(
        &scraped_test.text,
        Some(&global_opts.crate_name),
        scraped_test.langstr.test_harness,
        &global_opts,
        edition,
        Some(&test_opts.test_id),
    );
    let runnable_test = RunnableDoctest {
        full_test_code,
        full_test_line_offset,
        test_opts,
        global_opts,
        scraped_test,
    };
    let res = run_test(runnable_test, &rustdoc_options, supports_color, report_unused_externs);

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
impl DoctestVisitor for Vec<usize> {
    fn visit_test(&mut self, _test: String, _config: LangString, rel_line: MdRelLine) {
        self.push(1 + rel_line.offset());
    }
}

#[cfg(test)]
mod tests;
