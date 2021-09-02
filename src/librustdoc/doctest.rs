use rustc_ast as ast;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lrc;
use rustc_errors::{ColorConfig, ErrorReported};
use rustc_hir as hir;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_hir::intravisit;
use rustc_hir::{HirId, CRATE_HIR_ID};
use rustc_interface::interface;
use rustc_middle::hir::map::Map;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{self, CrateType, ErrorOutputType};
use rustc_session::{lint, DiagnosticOutput, Session};
use rustc_span::edition::Edition;
use rustc_span::source_map::SourceMap;
use rustc_span::symbol::sym;
use rustc_span::Symbol;
use rustc_span::{BytePos, FileName, Pos, Span, DUMMY_SP};
use rustc_target::spec::TargetTriple;
use tempfile::Builder as TempFileBuilder;

use std::env;
use std::io::{self, Write};
use std::panic;
use std::path::PathBuf;
use std::process::{self, Command, Stdio};
use std::str;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::clean::{types::AttributesExt, Attributes};
use crate::config::Options;
use crate::html::markdown::{self, ErrorCodes, Ignore, LangString};
use crate::lint::init_lints;
use crate::passes::span_of_attrs;

#[derive(Clone, Default)]
crate struct TestOptions {
    /// Whether to disable the default `extern crate my_crate;` when creating doctests.
    crate no_crate_inject: bool,
    /// Whether to emit compilation warnings when compiling doctests. Setting this will suppress
    /// the default `#![allow(unused)]`.
    crate display_warnings: bool,
    /// Additional crate-level attributes to add to doctests.
    crate attrs: Vec<String>,
}

crate fn run(options: Options) -> Result<(), ErrorReported> {
    let input = config::Input::File(options.input.clone());

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

    let crate_types =
        if options.proc_macro_crate { vec![CrateType::ProcMacro] } else { vec![CrateType::Rlib] };

    let sessopts = config::Options {
        maybe_sysroot: options.maybe_sysroot.clone(),
        search_paths: options.libs.clone(),
        crate_types,
        lint_opts: if !options.display_warnings { lint_opts } else { vec![] },
        lint_cap: Some(options.lint_cap.clone().unwrap_or_else(|| lint::Forbid)),
        cg: options.codegen_options.clone(),
        externs: options.externs.clone(),
        unstable_features: options.render_options.unstable_features,
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
        crate_cfg: interface::parse_cfgspecs(cfgs),
        input,
        input_path: None,
        output_file: None,
        output_dir: None,
        file_loader: None,
        diagnostic_output: DiagnosticOutput::Default,
        stderr: None,
        lint_caps,
        parse_sess_created: None,
        register_lints: Some(Box::new(crate::lint::register_lints)),
        override_queries: None,
        make_codegen_backend: None,
        registry: rustc_driver::diagnostics_registry(),
    };

    let test_args = options.test_args.clone();
    let display_warnings = options.display_warnings;
    let nocapture = options.nocapture;
    let externs = options.externs.clone();
    let json_unused_externs = options.json_unused_externs;

    let res = interface::run_compiler(config, |compiler| {
        compiler.enter(|queries| {
            let mut global_ctxt = queries.global_ctxt()?.take();

            let collector = global_ctxt.enter(|tcx| {
                let crate_attrs = tcx.hir().attrs(CRATE_HIR_ID);

                let mut opts = scrape_test_config(crate_attrs);
                opts.display_warnings |= options.display_warnings;
                let enable_per_target_ignores = options.enable_per_target_ignores;
                let mut collector = Collector::new(
                    tcx.crate_name(LOCAL_CRATE),
                    options,
                    false,
                    opts,
                    Some(compiler.session().parse_sess.clone_source_map()),
                    None,
                    enable_per_target_ignores,
                );

                let mut hir_collector = HirCollector {
                    sess: compiler.session(),
                    collector: &mut collector,
                    map: tcx.hir(),
                    codes: ErrorCodes::from(
                        compiler.session().opts.unstable_features.is_nightly_build(),
                    ),
                    tcx,
                };
                hir_collector.visit_testable(
                    "".to_string(),
                    CRATE_HIR_ID,
                    tcx.hir().span(CRATE_HIR_ID),
                    |this| tcx.hir().walk_toplevel_module(this),
                );

                collector
            });
            compiler.session().abort_if_errors();

            let unused_extern_reports = collector.unused_extern_reports.clone();
            let compiling_test_count = collector.compiling_test_count.load(Ordering::SeqCst);
            let ret: Result<_, ErrorReported> =
                Ok((collector.tests, unused_extern_reports, compiling_test_count));
            ret
        })
    });
    let (tests, unused_extern_reports, compiling_test_count) = match res {
        Ok(res) => res,
        Err(ErrorReported) => return Err(ErrorReported),
    };

    run_tests(test_args, nocapture, display_warnings, tests);

    // Collect and warn about unused externs, but only if we've gotten
    // reports for each doctest
    if json_unused_externs {
        let unused_extern_reports: Vec<_> =
            std::mem::take(&mut unused_extern_reports.lock().unwrap());
        if unused_extern_reports.len() == compiling_test_count {
            let extern_names = externs.iter().map(|(name, _)| name).collect::<FxHashSet<&String>>();
            let mut unused_extern_names = unused_extern_reports
                .iter()
                .map(|uexts| uexts.unused_extern_names.iter().collect::<FxHashSet<&String>>())
                .fold(extern_names, |uextsa, uextsb| {
                    uextsa.intersection(&uextsb).map(|v| *v).collect::<FxHashSet<&String>>()
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
                    v => unreachable!("Invalid lint level '{}'", v),
                })
                .unwrap_or("warn")
                .to_string();
            let uext = UnusedExterns { lint_level, unused_extern_names };
            let unused_extern_json = serde_json::to_string(&uext).unwrap();
            eprintln!("{}", unused_extern_json);
        }
    }

    Ok(())
}

crate fn run_tests(
    mut test_args: Vec<String>,
    nocapture: bool,
    display_warnings: bool,
    tests: Vec<test::TestDescAndFn>,
) {
    test_args.insert(0, "rustdoctest".to_string());
    if nocapture {
        test_args.push("--nocapture".to_string());
    }
    test::test_main(&test_args, tests, Some(test::Options::new().display_output(display_warnings)));
}

// Look for `#![doc(test(no_crate_inject))]`, used by crates in the std facade.
fn scrape_test_config(attrs: &[ast::Attribute]) -> TestOptions {
    use rustc_ast_pretty::pprust;

    let mut opts =
        TestOptions { no_crate_inject: false, display_warnings: false, attrs: Vec::new() };

    let test_attrs: Vec<_> = attrs
        .iter()
        .filter(|a| a.has_name(sym::doc))
        .flat_map(|a| a.meta_item_list().unwrap_or_else(Vec::new))
        .filter(|a| a.has_name(sym::test))
        .collect();
    let attrs = test_attrs.iter().flat_map(|a| a.meta_item_list().unwrap_or(&[]));

    for attr in attrs {
        if attr.has_name(sym::no_crate_inject) {
            opts.no_crate_inject = true;
        }
        if attr.has_name(sym::attr) {
            if let Some(l) = attr.meta_item_list() {
                for item in l {
                    opts.attrs.push(pprust::meta_list_item_to_string(item));
                }
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

fn run_test(
    test: &str,
    crate_name: &str,
    line: usize,
    options: Options,
    should_panic: bool,
    no_run: bool,
    as_test_harness: bool,
    runtool: Option<String>,
    runtool_args: Vec<String>,
    target: TargetTriple,
    compile_fail: bool,
    mut error_codes: Vec<String>,
    opts: &TestOptions,
    edition: Edition,
    outdir: DirState,
    path: PathBuf,
    test_id: &str,
    report_unused_externs: impl Fn(UnusedExterns),
) -> Result<(), TestFailure> {
    let (test, line_offset, supports_color) =
        make_test(test, Some(crate_name), as_test_harness, opts, edition, Some(test_id));

    let output_file = outdir.path().join("rust_out");

    let rustc_binary = options
        .test_builder
        .as_deref()
        .unwrap_or_else(|| rustc_interface::util::rustc_path().expect("found rustc"));
    let mut compiler = Command::new(&rustc_binary);
    compiler.arg("--crate-type").arg("bin");
    for cfg in &options.cfgs {
        compiler.arg("--cfg").arg(&cfg);
    }
    if let Some(sysroot) = options.maybe_sysroot {
        compiler.arg("--sysroot").arg(sysroot);
    }
    compiler.arg("--edition").arg(&edition.to_string());
    compiler.env("UNSTABLE_RUSTDOC_TEST_PATH", path);
    compiler.env("UNSTABLE_RUSTDOC_TEST_LINE", format!("{}", line as isize - line_offset as isize));
    compiler.arg("-o").arg(&output_file);
    if as_test_harness {
        compiler.arg("--test");
    }
    if options.json_unused_externs && !compile_fail {
        compiler.arg("--error-format=json");
        compiler.arg("--json").arg("unused-externs");
        compiler.arg("-Z").arg("unstable-options");
        compiler.arg("-W").arg("unused_crate_dependencies");
    }
    for lib_str in &options.lib_strs {
        compiler.arg("-L").arg(&lib_str);
    }
    for extern_str in &options.extern_strs {
        compiler.arg("--extern").arg(&extern_str);
    }
    compiler.arg("-Ccodegen-units=1");
    for codegen_options_str in &options.codegen_options_strs {
        compiler.arg("-C").arg(&codegen_options_str);
    }
    for debugging_option_str in &options.debugging_opts_strs {
        compiler.arg("-Z").arg(&debugging_option_str);
    }
    if no_run && !compile_fail && options.persist_doctests.is_none() {
        compiler.arg("--emit=metadata");
    }
    compiler.arg("--target").arg(match target {
        TargetTriple::TargetTriple(s) => s,
        TargetTriple::TargetPath(path) => {
            path.to_str().expect("target path must be valid unicode").to_string()
        }
    });
    if let ErrorOutputType::HumanReadable(kind) = options.error_format {
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

    let mut child = compiler.spawn().expect("Failed to spawn rustc process");
    {
        let stdin = child.stdin.as_mut().expect("Failed to open stdin");
        stdin.write_all(test.as_bytes()).expect("could write out test sources");
    }
    let output = child.wait_with_output().expect("Failed to read stdout");

    struct Bomb<'a>(&'a str);
    impl Drop for Bomb<'_> {
        fn drop(&mut self) {
            eprint!("{}", self.0);
        }
    }
    let mut out_lines = str::from_utf8(&output.stderr)
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
        .collect::<Vec<_>>();

    // Add a \n to the end to properly terminate the last line,
    // but only if there was output to be printed
    if out_lines.len() > 0 {
        out_lines.push("");
    }

    let out = out_lines.join("\n");
    let _bomb = Bomb(&out);
    match (output.status.success(), compile_fail) {
        (true, true) => {
            return Err(TestFailure::UnexpectedCompilePass);
        }
        (true, false) => {}
        (false, true) => {
            if !error_codes.is_empty() {
                // We used to check if the output contained "error[{}]: " but since we added the
                // colored output, we can't anymore because of the color escape characters before
                // the ":".
                error_codes.retain(|err| !out.contains(&format!("error[{}]", err)));

                if !error_codes.is_empty() {
                    return Err(TestFailure::MissingErrorCodes(error_codes));
                }
            }
        }
        (false, false) => {
            return Err(TestFailure::CompileError);
        }
    }

    if no_run {
        return Ok(());
    }

    // Run the code!
    let mut cmd;

    if let Some(tool) = runtool {
        cmd = Command::new(tool);
        cmd.args(runtool_args);
        cmd.arg(output_file);
    } else {
        cmd = Command::new(output_file);
    }
    if let Some(run_directory) = options.test_run_directory {
        cmd.current_dir(run_directory);
    }

    let result = if options.nocapture {
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
            if should_panic && out.status.success() {
                return Err(TestFailure::UnexpectedRunPass);
            } else if !should_panic && !out.status.success() {
                return Err(TestFailure::ExecutionFailure(out));
            }
        }
    }

    Ok(())
}

/// Transforms a test into code that can be compiled into a Rust binary, and returns the number of
/// lines before the test code begins as well as if the output stream supports colors or not.
crate fn make_test(
    s: &str,
    crate_name: Option<&str>,
    dont_insert_main: bool,
    opts: &TestOptions,
    edition: Edition,
    test_id: Option<&str>,
) -> (String, usize, bool) {
    let (crate_attrs, everything_else, crates) = partition_source(s);
    let everything_else = everything_else.trim();
    let mut line_offset = 0;
    let mut prog = String::new();
    let mut supports_color = false;

    if opts.attrs.is_empty() && !opts.display_warnings {
        // If there aren't any attributes supplied by #![doc(test(attr(...)))], then allow some
        // lints that are commonly triggered in doctests. The crate-level test attributes are
        // commonly used to make tests fail in case they trigger warnings, so having this there in
        // that case may cause some tests to pass when they shouldn't have.
        prog.push_str("#![allow(unused)]\n");
        line_offset += 1;
    }

    // Next, any attributes that came from the crate root via #![doc(test(attr(...)))].
    for attr in &opts.attrs {
        prog.push_str(&format!("#![{}]\n", attr));
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
            use rustc_errors::emitter::{Emitter, EmitterWriter};
            use rustc_errors::Handler;
            use rustc_parse::maybe_new_parser_from_source_str;
            use rustc_parse::parser::ForceCollect;
            use rustc_session::parse::ParseSess;
            use rustc_span::source_map::FilePathMapping;

            let filename = FileName::anon_source_code(s);
            let source = crates + everything_else;

            // Any errors in parsing should also appear when the doctest is compiled for real, so just
            // send all the errors that librustc_ast emits directly into a `Sink` instead of stderr.
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            supports_color =
                EmitterWriter::stderr(ColorConfig::Auto, None, false, false, Some(80), false)
                    .supports_color();

            let emitter =
                EmitterWriter::new(Box::new(io::sink()), None, false, false, false, None, false);

            // FIXME(misdreavus): pass `-Z treat-err-as-bug` to the doctest parser
            let handler = Handler::with_emitter(false, None, Box::new(emitter));
            let sess = ParseSess::with_span_handler(handler, sm);

            let mut found_main = false;
            let mut found_extern_crate = crate_name.is_none();
            let mut found_macro = false;

            let mut parser = match maybe_new_parser_from_source_str(&sess, filename, source) {
                Ok(p) => p,
                Err(errs) => {
                    for mut err in errs {
                        err.cancel();
                    }

                    return (found_main, found_extern_crate, found_macro);
                }
            };

            loop {
                match parser.parse_item(ForceCollect::No) {
                    Ok(Some(item)) => {
                        if !found_main {
                            if let ast::ItemKind::Fn(..) = item.kind {
                                if item.ident.name == sym::main {
                                    found_main = true;
                                }
                            }
                        }

                        if !found_extern_crate {
                            if let ast::ItemKind::ExternCrate(original) = item.kind {
                                // This code will never be reached if `crate_name` is none because
                                // `found_extern_crate` is initialized to `true` if it is none.
                                let crate_name = crate_name.unwrap();

                                match original {
                                    Some(name) => found_extern_crate = name.as_str() == crate_name,
                                    None => found_extern_crate = item.ident.as_str() == crate_name,
                                }
                            }
                        }

                        if !found_macro {
                            if let ast::ItemKind::MacCall(..) = item.kind {
                                found_macro = true;
                            }
                        }

                        if found_main && found_extern_crate {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(mut e) => {
                        e.cancel();
                        break;
                    }
                }
            }

            // Reset errors so that they won't be reported as compiler bugs when dropping the
            // handler. Any errors in the tests will be reported when the test file is compiled,
            // Note that we still need to cancel the errors above otherwise `DiagnosticBuilder`
            // will panic on drop.
            sess.span_diagnostic.reset_err_count();

            (found_main, found_extern_crate, found_macro)
        })
    });
    let (already_has_main, already_has_extern_crate, found_macro) = match result {
        Ok(result) => result,
        Err(ErrorReported) => {
            // If the parser panicked due to a fatal error, pass the test code through unchanged.
            // The error will be reported during compilation.
            return (s.to_owned(), 0, false);
        }
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
                prog.push_str(&format!("extern crate r#{};\n", crate_name));
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
        // This is for example needed for the tooling around `-Z instrument-coverage`.
        let inner_fn_name = if let Some(test_id) = test_id {
            format!("_doctest_main_{}", test_id)
        } else {
            "_inner".into()
        };
        let inner_attr = if test_id.is_some() { "#[allow(non_snake_case)] " } else { "" };
        let (main_pre, main_post) = if returns_result {
            (
                format!(
                    "fn main() {{ {}fn {}() -> Result<(), impl core::fmt::Debug> {{\n",
                    inner_attr, inner_fn_name
                ),
                format!("\n}} {}().unwrap() }}", inner_fn_name),
            )
        } else if test_id.is_some() {
            (
                format!("fn main() {{ {}fn {}() {{\n", inner_attr, inner_fn_name),
                format!("\n}} {}() }}", inner_fn_name),
            )
        } else {
            ("fn main() {\n".into(), "\n}".into())
        };
        // Note on newlines: We insert a line/newline *before*, and *after*
        // the doctest and adjust the `line_offset` accordingly.
        // In the case of `-Z instrument-coverage`, this means that the generated
        // inner `main` function spans from the doctest opening codeblock to the
        // closing one. For example
        // /// ``` <- start of the inner main
        // /// <- code under doctest
        // /// ``` <- end of the inner main
        line_offset += 1;

        prog.extend([&main_pre, everything_else, &main_post].iter().cloned());
    }

    debug!("final doctest:\n{}", prog);

    (prog, line_offset, supports_color)
}

// FIXME(aburka): use a real parser to deal with multiline attributes
fn partition_source(s: &str) -> (String, String, String) {
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

    for line in s.lines() {
        let trimline = line.trim();

        // FIXME(misdreavus): if a doc comment is placed on an extern crate statement, it will be
        // shunted into "everything else"
        match state {
            PartitionState::Attrs => {
                state = if trimline.starts_with("#![")
                    || trimline.chars().all(|c| c.is_whitespace())
                    || (trimline.starts_with("//") && !trimline.starts_with("///"))
                {
                    PartitionState::Attrs
                } else if trimline.starts_with("extern crate")
                    || trimline.starts_with("#[macro_use] extern crate")
                {
                    PartitionState::Crates
                } else {
                    PartitionState::Other
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

    debug!("before:\n{}", before);
    debug!("crates:\n{}", crates);
    debug!("after:\n{}", after);

    (before, after, crates)
}

crate trait Tester {
    fn add_test(&mut self, test: String, config: LangString, line: usize);
    fn get_line(&self) -> usize {
        0
    }
    fn register_header(&mut self, _name: &str, _level: u32) {}
}

crate struct Collector {
    crate tests: Vec<test::TestDescAndFn>,

    // The name of the test displayed to the user, separated by `::`.
    //
    // In tests from Rust source, this is the path to the item
    // e.g., `["std", "vec", "Vec", "push"]`.
    //
    // In tests from a markdown file, this is the titles of all headers (h1~h6)
    // of the sections that contain the code block, e.g., if the markdown file is
    // written as:
    //
    // ``````markdown
    // # Title
    //
    // ## Subtitle
    //
    // ```rust
    // assert!(true);
    // ```
    // ``````
    //
    // the `names` vector of that test will be `["Title", "Subtitle"]`.
    names: Vec<String>,

    options: Options,
    use_headers: bool,
    enable_per_target_ignores: bool,
    crate_name: Symbol,
    opts: TestOptions,
    position: Span,
    source_map: Option<Lrc<SourceMap>>,
    filename: Option<PathBuf>,
    visited_tests: FxHashMap<(String, usize), usize>,
    unused_extern_reports: Arc<Mutex<Vec<UnusedExterns>>>,
    compiling_test_count: AtomicUsize,
}

impl Collector {
    crate fn new(
        crate_name: Symbol,
        options: Options,
        use_headers: bool,
        opts: TestOptions,
        source_map: Option<Lrc<SourceMap>>,
        filename: Option<PathBuf>,
        enable_per_target_ignores: bool,
    ) -> Collector {
        Collector {
            tests: Vec::new(),
            names: Vec::new(),
            options,
            use_headers,
            enable_per_target_ignores,
            crate_name,
            opts,
            position: DUMMY_SP,
            source_map,
            filename,
            visited_tests: FxHashMap::default(),
            unused_extern_reports: Default::default(),
            compiling_test_count: AtomicUsize::new(0),
        }
    }

    fn generate_name(&self, line: usize, filename: &FileName) -> String {
        let mut item_path = self.names.join("::");
        if !item_path.is_empty() {
            item_path.push(' ');
        }
        format!("{} - {}(line {})", filename.prefer_local(), item_path, line)
    }

    crate fn set_position(&mut self, position: Span) {
        self.position = position;
    }

    fn get_filename(&self) -> FileName {
        if let Some(ref source_map) = self.source_map {
            let filename = source_map.span_to_filename(self.position);
            if let FileName::Real(ref filename) = filename {
                if let Ok(cur_dir) = env::current_dir() {
                    if let Some(local_path) = filename.local_path() {
                        if let Ok(path) = local_path.strip_prefix(&cur_dir) {
                            return path.to_owned().into();
                        }
                    }
                }
            }
            filename
        } else if let Some(ref filename) = self.filename {
            filename.clone().into()
        } else {
            FileName::Custom("input".to_owned())
        }
    }
}

impl Tester for Collector {
    fn add_test(&mut self, test: String, config: LangString, line: usize) {
        let filename = self.get_filename();
        let name = self.generate_name(line, &filename);
        let crate_name = self.crate_name.to_string();
        let opts = self.opts.clone();
        let edition = config.edition.unwrap_or(self.options.edition);
        let options = self.options.clone();
        let runtool = self.options.runtool.clone();
        let runtool_args = self.options.runtool_args.clone();
        let target = self.options.target.clone();
        let target_str = target.to_string();
        let unused_externs = self.unused_extern_reports.clone();
        let no_run = config.no_run || options.no_run;
        if !config.compile_fail {
            self.compiling_test_count.fetch_add(1, Ordering::SeqCst);
        }

        let path = match &filename {
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
        let file = filename
            .prefer_local()
            .to_string_lossy()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect::<String>();
        let test_id = format!(
            "{file}_{line}_{number}",
            file = file,
            line = line,
            number = {
                // Increases the current test number, if this file already
                // exists or it creates a new entry with a test number of 0.
                self.visited_tests.entry((file.clone(), line)).and_modify(|v| *v += 1).or_insert(0)
            },
        );
        let outdir = if let Some(mut path) = options.persist_doctests.clone() {
            path.push(&test_id);

            std::fs::create_dir_all(&path)
                .expect("Couldn't create directory for doctest executables");

            DirState::Perm(path)
        } else {
            DirState::Temp(
                TempFileBuilder::new()
                    .prefix("rustdoctest")
                    .tempdir()
                    .expect("rustdoc needs a tempdir"),
            )
        };

        debug!("creating test {}: {}", name, test);
        self.tests.push(test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::DynTestName(name),
                ignore: match config.ignore {
                    Ignore::All => true,
                    Ignore::None => false,
                    Ignore::Some(ref ignores) => ignores.iter().any(|s| target_str.contains(s)),
                },
                // compiler failures are test failures
                should_panic: test::ShouldPanic::No,
                allow_fail: config.allow_fail,
                compile_fail: config.compile_fail,
                no_run,
                test_type: test::TestType::DocTest,
            },
            testfn: test::DynTestFn(Box::new(move || {
                let report_unused_externs = |uext| {
                    unused_externs.lock().unwrap().push(uext);
                };
                let res = run_test(
                    &test,
                    &crate_name,
                    line,
                    options,
                    config.should_panic,
                    no_run,
                    config.test_harness,
                    runtool,
                    runtool_args,
                    target,
                    config.compile_fail,
                    config.error_codes,
                    &opts,
                    edition,
                    outdir,
                    path,
                    &test_id,
                    report_unused_externs,
                );

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
                            eprint!("Some expected error codes were not found: {:?}", codes);
                        }
                        TestFailure::ExecutionError(err) => {
                            eprint!("Couldn't run the test: {}", err);
                            if err.kind() == io::ErrorKind::PermissionDenied {
                                eprint!(" - maybe your tempdir is mounted with noexec?");
                            }
                        }
                        TestFailure::ExecutionFailure(out) => {
                            let reason = if let Some(code) = out.status.code() {
                                format!("exit code {}", code)
                            } else {
                                String::from("terminated by signal")
                            };

                            eprintln!("Test executable failed ({}).", reason);

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
                                    eprintln!("stdout:\n{}", stdout);
                                }

                                if !stderr.is_empty() {
                                    eprintln!("stderr:\n{}", stderr);
                                }
                            }
                        }
                    }

                    panic::resume_unwind(Box::new(()));
                }
            })),
        });
    }

    fn get_line(&self) -> usize {
        if let Some(ref source_map) = self.source_map {
            let line = self.position.lo().to_usize();
            let line = source_map.lookup_char_pos(BytePos(line as u32)).line;
            if line > 0 { line - 1 } else { line }
        } else {
            0
        }
    }

    fn register_header(&mut self, name: &str, level: u32) {
        if self.use_headers {
            // We use these headings as test names, so it's good if
            // they're valid identifiers.
            let name = name
                .chars()
                .enumerate()
                .map(|(i, c)| {
                    if (i == 0 && rustc_lexer::is_id_start(c))
                        || (i != 0 && rustc_lexer::is_id_continue(c))
                    {
                        c
                    } else {
                        '_'
                    }
                })
                .collect::<String>();

            // Here we try to efficiently assemble the header titles into the
            // test name in the form of `h1::h2::h3::h4::h5::h6`.
            //
            // Suppose that originally `self.names` contains `[h1, h2, h3]`...
            let level = level as usize;
            if level <= self.names.len() {
                // ... Consider `level == 2`. All headers in the lower levels
                // are irrelevant in this new level. So we should reset
                // `self.names` to contain headers until <h2>, and replace that
                // slot with the new name: `[h1, name]`.
                self.names.truncate(level);
                self.names[level - 1] = name;
            } else {
                // ... On the other hand, consider `level == 5`. This means we
                // need to extend `self.names` to contain five headers. We fill
                // in the missing level (<h4>) with `_`. Thus `self.names` will
                // become `[h1, h2, h3, "_", name]`.
                if level - 1 > self.names.len() {
                    self.names.resize(level - 1, "_".to_owned());
                }
                self.names.push(name);
            }
        }
    }
}

struct HirCollector<'a, 'hir, 'tcx> {
    sess: &'a Session,
    collector: &'a mut Collector,
    map: Map<'hir>,
    codes: ErrorCodes,
    tcx: TyCtxt<'tcx>,
}

impl<'a, 'hir, 'tcx> HirCollector<'a, 'hir, 'tcx> {
    fn visit_testable<F: FnOnce(&mut Self)>(
        &mut self,
        name: String,
        hir_id: HirId,
        sp: Span,
        nested: F,
    ) {
        let ast_attrs = self.tcx.hir().attrs(hir_id);
        let mut attrs = Attributes::from_ast(ast_attrs, None);

        if let Some(ref cfg) = ast_attrs.cfg(self.sess) {
            if !cfg.matches(&self.sess.parse_sess, Some(&self.sess.features_untracked())) {
                return;
            }
        }

        let has_name = !name.is_empty();
        if has_name {
            self.collector.names.push(name);
        }

        attrs.unindent_doc_comments();
        // The collapse-docs pass won't combine sugared/raw doc attributes, or included files with
        // anything else, this will combine them for us.
        if let Some(doc) = attrs.collapsed_doc_value() {
            // Use the outermost invocation, so that doctest names come from where the docs were written.
            let span = ast_attrs
                .span()
                .map(|span| span.ctxt().outer_expn().expansion_cause().unwrap_or(span))
                .unwrap_or(DUMMY_SP);
            self.collector.set_position(span);
            markdown::find_testable_code(
                &doc,
                self.collector,
                self.codes,
                self.collector.enable_per_target_ignores,
                Some(&crate::html::markdown::ExtraInfo::new(
                    self.tcx,
                    hir_id,
                    span_of_attrs(&attrs).unwrap_or(sp),
                )),
            );
        }

        nested(self);

        if has_name {
            self.collector.names.pop();
        }
    }
}

impl<'a, 'hir, 'tcx> intravisit::Visitor<'hir> for HirCollector<'a, 'hir, 'tcx> {
    type Map = Map<'hir>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::All(self.map)
    }

    fn visit_item(&mut self, item: &'hir hir::Item<'_>) {
        let name = match &item.kind {
            hir::ItemKind::Macro(ref macro_def) => {
                // FIXME(#88038): Non exported macros have historically not been tested,
                // but we really ought to start testing them.
                let def_id = item.def_id.to_def_id();
                if macro_def.macro_rules && !self.tcx.has_attr(def_id, sym::macro_export) {
                    intravisit::walk_item(self, item);
                    return;
                }
                item.ident.to_string()
            }
            hir::ItemKind::Impl(impl_) => {
                rustc_hir_pretty::id_to_string(&self.map, impl_.self_ty.hir_id)
            }
            _ => item.ident.to_string(),
        };

        self.visit_testable(name, item.hir_id(), item.span, |this| {
            intravisit::walk_item(this, item);
        });
    }

    fn visit_trait_item(&mut self, item: &'hir hir::TraitItem<'_>) {
        self.visit_testable(item.ident.to_string(), item.hir_id(), item.span, |this| {
            intravisit::walk_trait_item(this, item);
        });
    }

    fn visit_impl_item(&mut self, item: &'hir hir::ImplItem<'_>) {
        self.visit_testable(item.ident.to_string(), item.hir_id(), item.span, |this| {
            intravisit::walk_impl_item(this, item);
        });
    }

    fn visit_foreign_item(&mut self, item: &'hir hir::ForeignItem<'_>) {
        self.visit_testable(item.ident.to_string(), item.hir_id(), item.span, |this| {
            intravisit::walk_foreign_item(this, item);
        });
    }

    fn visit_variant(
        &mut self,
        v: &'hir hir::Variant<'_>,
        g: &'hir hir::Generics<'_>,
        item_id: hir::HirId,
    ) {
        self.visit_testable(v.ident.to_string(), v.id, v.span, |this| {
            intravisit::walk_variant(this, v, g, item_id);
        });
    }

    fn visit_field_def(&mut self, f: &'hir hir::FieldDef<'_>) {
        self.visit_testable(f.ident.to_string(), f.hir_id, f.span, |this| {
            intravisit::walk_field_def(this, f);
        });
    }
}

#[cfg(test)]
mod tests;
