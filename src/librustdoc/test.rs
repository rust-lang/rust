use rustc_data_structures::sync::Lrc;
use rustc_feature::UnstableFeatures;
use rustc_interface::interface;
use rustc_target::spec::TargetTriple;
use rustc::hir;
use rustc::hir::intravisit;
use rustc::session::{self, config, DiagnosticOutput};
use rustc::util::common::ErrorReported;
use syntax::ast;
use syntax::with_globals;
use syntax::source_map::SourceMap;
use syntax::edition::Edition;
use std::env;
use std::io::{self, Write};
use std::panic;
use std::path::PathBuf;
use std::process::{self, Command, Stdio};
use std::str;
use syntax::symbol::sym;
use syntax_pos::{BytePos, DUMMY_SP, Pos, Span, FileName};
use tempfile::Builder as TempFileBuilder;
use testing;

use crate::clean::Attributes;
use crate::config::Options;
use crate::html::markdown::{self, ErrorCodes, LangString, Ignore};

#[derive(Clone, Default)]
pub struct TestOptions {
    /// Whether to disable the default `extern crate my_crate;` when creating doctests.
    pub no_crate_inject: bool,
    /// Whether to emit compilation warnings when compiling doctests. Setting this will suppress
    /// the default `#![allow(unused)]`.
    pub display_warnings: bool,
    /// Additional crate-level attributes to add to doctests.
    pub attrs: Vec<String>,
}

pub fn run(options: Options) -> i32 {
    let input = config::Input::File(options.input.clone());

    let crate_types = if options.proc_macro_crate {
        vec![config::CrateType::ProcMacro]
    } else {
        vec![config::CrateType::Dylib]
    };

    let sessopts = config::Options {
        maybe_sysroot: options.maybe_sysroot.clone(),
        search_paths: options.libs.clone(),
        crate_types,
        cg: options.codegen_options.clone(),
        externs: options.externs.clone(),
        unstable_features: UnstableFeatures::from_environment(),
        lint_cap: Some(::rustc::lint::Level::Allow),
        actually_rustdoc: true,
        debugging_opts: config::DebuggingOptions {
            ..config::basic_debugging_options()
        },
        edition: options.edition,
        target_triple: options.target.clone(),
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
        crate_name: options.crate_name.clone(),
        lint_caps: Default::default(),
        register_lints: None,
        override_queries: None,
        registry: rustc_driver::diagnostics_registry(),
    };

    let mut test_args = options.test_args.clone();
    let display_warnings = options.display_warnings;

    let tests = interface::run_compiler(config, |compiler| compiler.enter(|queries| {
        let lower_to_hir = queries.lower_to_hir()?;

        let mut opts = scrape_test_config(lower_to_hir.peek().0.krate());
        opts.display_warnings |= options.display_warnings;
        let enable_per_target_ignores = options.enable_per_target_ignores;
        let mut collector = Collector::new(
            queries.crate_name()?.peek().to_string(),
            options,
            false,
            opts,
            Some(compiler.source_map().clone()),
            None,
            enable_per_target_ignores,
        );

        let mut global_ctxt = queries.global_ctxt()?.take();

        global_ctxt.enter(|tcx| {
            let krate = tcx.hir().krate();
            let mut hir_collector = HirCollector {
                sess: compiler.session(),
                collector: &mut collector,
                map: tcx.hir(),
                codes: ErrorCodes::from(compiler.session().opts
                                                .unstable_features.is_nightly_build()),
            };
            hir_collector.visit_testable("".to_string(), &krate.attrs, |this| {
                intravisit::walk_crate(this, krate);
            });
        });

        let ret : Result<_, ErrorReported> = Ok(collector.tests);
        ret
    })).expect("compiler aborted in rustdoc!");

    test_args.insert(0, "rustdoctest".to_string());

    testing::test_main(
        &test_args,
        tests,
        Some(testing::Options::new().display_output(display_warnings))
    );

    0
}

// Look for `#![doc(test(no_crate_inject))]`, used by crates in the std facade.
fn scrape_test_config(krate: &::rustc::hir::Crate) -> TestOptions {
    use syntax::print::pprust;

    let mut opts = TestOptions {
        no_crate_inject: false,
        display_warnings: false,
        attrs: Vec::new(),
    };

    let test_attrs: Vec<_> = krate.attrs.iter()
        .filter(|a| a.check_name(sym::doc))
        .flat_map(|a| a.meta_item_list().unwrap_or_else(Vec::new))
        .filter(|a| a.check_name(sym::test))
        .collect();
    let attrs = test_attrs.iter().flat_map(|a| a.meta_item_list().unwrap_or(&[]));

    for attr in attrs {
        if attr.check_name(sym::no_crate_inject) {
            opts.no_crate_inject = true;
        }
        if attr.check_name(sym::attr) {
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

fn run_test(
    test: &str,
    cratename: &str,
    filename: &FileName,
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
) -> Result<(), TestFailure> {
    let (test, line_offset) = match panic::catch_unwind(|| {
        make_test(test, Some(cratename), as_test_harness, opts, edition)
    }) {
        Ok((test, line_offset)) => (test, line_offset),
        Err(cause) if cause.is::<errors::FatalErrorMarker>() => {
            // If the parser used by `make_test` panicked due to a fatal error, pass the test code
            // through unchanged. The error will be reported during compilation.
            (test.to_owned(), 0)
        },
        Err(cause) => panic::resume_unwind(cause),
    };

    // FIXME(#44940): if doctests ever support path remapping, then this filename
    // needs to be the result of `SourceMap::span_to_unmapped_path`.
    let path = match filename {
        FileName::Real(path) => path.clone(),
        _ => PathBuf::from(r"doctest.rs"),
    };

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

    let outdir = if let Some(mut path) = options.persist_doctests {
        path.push(format!("{}_{}",
            filename
                .to_string()
                .rsplit('/')
                .next()
                .unwrap()
                .replace(".", "_"),
                line)
        );
        std::fs::create_dir_all(&path)
            .expect("Couldn't create directory for doctest executables");

        DirState::Perm(path)
    } else {
        DirState::Temp(TempFileBuilder::new()
                        .prefix("rustdoctest")
                        .tempdir()
                        .expect("rustdoc needs a tempdir"))
    };
    let output_file = outdir.path().join("rust_out");

    let rustc_binary = options.test_builder.as_ref().map(|v| &**v).unwrap_or_else(|| {
        rustc_interface::util::rustc_path().expect("found rustc")
    });
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
    compiler.env("UNSTABLE_RUSTDOC_TEST_LINE",
                 format!("{}", line as isize - line_offset as isize));
    compiler.arg("-o").arg(&output_file);
    if as_test_harness {
        compiler.arg("--test");
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
    for debugging_option_str in &options.debugging_options_strs {
        compiler.arg("-Z").arg(&debugging_option_str);
    }
    if no_run {
        compiler.arg("--emit=metadata");
    }
    compiler.arg("--target").arg(target.to_string());

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
            eprint!("{}",self.0);
        }
    }

    let out = str::from_utf8(&output.stderr).unwrap();
    let _bomb = Bomb(&out);
    match (output.status.success(), compile_fail) {
        (true, true) => {
            return Err(TestFailure::UnexpectedCompilePass);
        }
        (true, false) => {}
        (false, true) => {
            if !error_codes.is_empty() {
                error_codes.retain(|err| !out.contains(err));

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
        cmd.arg(output_file);
        cmd.args(runtool_args);
    } else {
        cmd = Command::new(output_file);
    }

    match cmd.output() {
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
/// lines before the test code begins.
///
/// # Panics
///
/// This function uses the compiler's parser internally. The parser will panic if it encounters a
/// fatal error while parsing the test.
pub fn make_test(s: &str,
                 cratename: Option<&str>,
                 dont_insert_main: bool,
                 opts: &TestOptions,
                 edition: Edition)
                 -> (String, usize) {
    let (crate_attrs, everything_else, crates) = partition_source(s);
    let everything_else = everything_else.trim();
    let mut line_offset = 0;
    let mut prog = String::new();

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

    // Uses libsyntax to parse the doctest and find if there's a main fn and the extern
    // crate already is included.
    let (already_has_main, already_has_extern_crate, found_macro) = with_globals(edition, || {
        use crate::syntax::{sess::ParseSess, source_map::FilePathMapping};
        use rustc_parse::maybe_new_parser_from_source_str;
        use errors::emitter::EmitterWriter;
        use errors::Handler;

        let filename = FileName::anon_source_code(s);
        let source = crates + &everything_else;

        // Any errors in parsing should also appear when the doctest is compiled for real, so just
        // send all the errors that libsyntax emits directly into a `Sink` instead of stderr.
        let cm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let emitter = EmitterWriter::new(box io::sink(), None, false, false, false, None, false);
        // FIXME(misdreavus): pass `-Z treat-err-as-bug` to the doctest parser
        let handler = Handler::with_emitter(false, None, box emitter);
        let sess = ParseSess::with_span_handler(handler, cm);

        let mut found_main = false;
        let mut found_extern_crate = cratename.is_none();
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
            match parser.parse_item() {
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
                            // This code will never be reached if `cratename` is none because
                            // `found_extern_crate` is initialized to `true` if it is none.
                            let cratename = cratename.unwrap();

                            match original {
                                Some(name) => found_extern_crate = name.as_str() == cratename,
                                None => found_extern_crate = item.ident.as_str() == cratename,
                            }
                        }
                    }

                    if !found_macro {
                        if let ast::ItemKind::Mac(..) = item.kind {
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

        (found_main, found_extern_crate, found_macro)
    });

    // If a doctest's `fn main` is being masked by a wrapper macro, the parsing loop above won't
    // see it. In that case, run the old text-based scan to see if they at least have a main
    // function written inside a macro invocation. See
    // https://github.com/rust-lang/rust/issues/56898
    let already_has_main = if found_macro && !already_has_main {
        s.lines()
            .map(|line| {
                let comment = line.find("//");
                if let Some(comment_begins) = comment {
                    &line[0..comment_begins]
                } else {
                    line
                }
            })
            .any(|code| code.contains("fn main"))
    } else {
        already_has_main
    };

    // Don't inject `extern crate std` because it's already injected by the
    // compiler.
    if !already_has_extern_crate && !opts.no_crate_inject && cratename != Some("std") {
        if let Some(cratename) = cratename {
            // Make sure its actually used if not included.
            if s.contains(cratename) {
                prog.push_str(&format!("extern crate {};\n", cratename));
                line_offset += 1;
            }
        }
    }

    // FIXME: This code cannot yet handle no_std test cases yet
    if dont_insert_main || already_has_main || prog.contains("![no_std]") {
        prog.push_str(everything_else);
    } else {
        let returns_result = everything_else.trim_end().ends_with("(())");
        let (main_pre, main_post) = if returns_result {
            ("fn main() { fn _inner() -> Result<(), impl core::fmt::Debug> {",
             "}\n_inner().unwrap() }")
        } else {
            ("fn main() {\n", "\n}")
        };
        prog.extend([main_pre, everything_else, main_post].iter().cloned());
        line_offset += 1;
    }

    debug!("final doctest:\n{}", prog);

    (prog, line_offset)
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
                state = if trimline.starts_with("#![") ||
                    trimline.chars().all(|c| c.is_whitespace()) ||
                    (trimline.starts_with("//") && !trimline.starts_with("///"))
                {
                    PartitionState::Attrs
                } else if trimline.starts_with("extern crate") ||
                    trimline.starts_with("#[macro_use] extern crate")
                {
                    PartitionState::Crates
                } else {
                    PartitionState::Other
                };
            }
            PartitionState::Crates => {
                state = if trimline.starts_with("extern crate") ||
                    trimline.starts_with("#[macro_use] extern crate") ||
                    trimline.chars().all(|c| c.is_whitespace()) ||
                    (trimline.starts_with("//") && !trimline.starts_with("///"))
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
                before.push_str("\n");
            }
            PartitionState::Crates => {
                crates.push_str(line);
                crates.push_str("\n");
            }
            PartitionState::Other => {
                after.push_str(line);
                after.push_str("\n");
            }
        }
    }

    debug!("before:\n{}", before);
    debug!("crates:\n{}", crates);
    debug!("after:\n{}", after);

    (before, after, crates)
}

pub trait Tester {
    fn add_test(&mut self, test: String, config: LangString, line: usize);
    fn get_line(&self) -> usize {
        0
    }
    fn register_header(&mut self, _name: &str, _level: u32) {}
}

pub struct Collector {
    pub tests: Vec<testing::TestDescAndFn>,

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
    cratename: String,
    opts: TestOptions,
    position: Span,
    source_map: Option<Lrc<SourceMap>>,
    filename: Option<PathBuf>,
}

impl Collector {
    pub fn new(cratename: String, options: Options, use_headers: bool, opts: TestOptions,
               source_map: Option<Lrc<SourceMap>>, filename: Option<PathBuf>,
               enable_per_target_ignores: bool) -> Collector {
        Collector {
            tests: Vec::new(),
            names: Vec::new(),
            options,
            use_headers,
            enable_per_target_ignores,
            cratename,
            opts,
            position: DUMMY_SP,
            source_map,
            filename,
        }
    }

    fn generate_name(&self, line: usize, filename: &FileName) -> String {
        format!("{} - {} (line {})", filename, self.names.join("::"), line)
    }

    pub fn set_position(&mut self, position: Span) {
        self.position = position;
    }

    fn get_filename(&self) -> FileName {
        if let Some(ref source_map) = self.source_map {
            let filename = source_map.span_to_filename(self.position);
            if let FileName::Real(ref filename) = filename {
                if let Ok(cur_dir) = env::current_dir() {
                    if let Ok(path) = filename.strip_prefix(&cur_dir) {
                        return path.to_owned().into();
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
        let cratename = self.cratename.to_string();
        let opts = self.opts.clone();
        let edition = config.edition.unwrap_or(self.options.edition.clone());
        let options = self.options.clone();
        let runtool = self.options.runtool.clone();
        let runtool_args = self.options.runtool_args.clone();
        let target = self.options.target.clone();
        let target_str = target.to_string();

        debug!("creating test {}: {}", name, test);
        self.tests.push(testing::TestDescAndFn {
            desc: testing::TestDesc {
                name: testing::DynTestName(name.clone()),
                ignore: match config.ignore {
                    Ignore::All => true,
                    Ignore::None => false,
                    Ignore::Some(ref ignores) => {
                        ignores.iter().any(|s| target_str.contains(s))
                    },
                },
                // compiler failures are test failures
                should_panic: testing::ShouldPanic::No,
                allow_fail: config.allow_fail,
                test_type: testing::TestType::DocTest,
            },
            testfn: testing::DynTestFn(box move || {
                let res = run_test(
                    &test,
                    &cratename,
                    &filename,
                    line,
                    options,
                    config.should_panic,
                    config.no_run,
                    config.test_harness,
                    runtool,
                    runtool_args,
                    target,
                    config.compile_fail,
                    config.error_codes,
                    &opts,
                    edition,
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

                    panic::resume_unwind(box ());
                }
            }),
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
            let name = name.chars().enumerate().map(|(i, c)| {
                    if (i == 0 && rustc_lexer::is_id_start(c)) ||
                        (i != 0 && rustc_lexer::is_id_continue(c)) {
                        c
                    } else {
                        '_'
                    }
                }).collect::<String>();

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

struct HirCollector<'a, 'hir> {
    sess: &'a session::Session,
    collector: &'a mut Collector,
    map: &'a hir::map::Map<'hir>,
    codes: ErrorCodes,
}

impl<'a, 'hir> HirCollector<'a, 'hir> {
    fn visit_testable<F: FnOnce(&mut Self)>(&mut self,
                                            name: String,
                                            attrs: &[ast::Attribute],
                                            nested: F) {
        let mut attrs = Attributes::from_ast(self.sess.diagnostic(), attrs);
        if let Some(ref cfg) = attrs.cfg {
            if !cfg.matches(&self.sess.parse_sess, Some(&self.sess.features_untracked())) {
                return;
            }
        }

        let has_name = !name.is_empty();
        if has_name {
            self.collector.names.push(name);
        }

        attrs.collapse_doc_comments();
        attrs.unindent_doc_comments();
        // The collapse-docs pass won't combine sugared/raw doc attributes, or included files with
        // anything else, this will combine them for us.
        if let Some(doc) = attrs.collapsed_doc_value() {
            self.collector.set_position(attrs.span.unwrap_or(DUMMY_SP));
            markdown::find_testable_code(&doc,
                                         self.collector,
                                         self.codes,
                                         self.collector.enable_per_target_ignores);
        }

        nested(self);

        if has_name {
            self.collector.names.pop();
        }
    }
}

impl<'a, 'hir> intravisit::Visitor<'hir> for HirCollector<'a, 'hir> {
    fn nested_visit_map<'this>(&'this mut self) -> intravisit::NestedVisitorMap<'this, 'hir> {
        intravisit::NestedVisitorMap::All(&self.map)
    }

    fn visit_item(&mut self, item: &'hir hir::Item) {
        let name = if let hir::ItemKind::Impl(.., ref ty, _) = item.kind {
            self.map.hir_to_pretty_string(ty.hir_id)
        } else {
            item.ident.to_string()
        };

        self.visit_testable(name, &item.attrs, |this| {
            intravisit::walk_item(this, item);
        });
    }

    fn visit_trait_item(&mut self, item: &'hir hir::TraitItem) {
        self.visit_testable(item.ident.to_string(), &item.attrs, |this| {
            intravisit::walk_trait_item(this, item);
        });
    }

    fn visit_impl_item(&mut self, item: &'hir hir::ImplItem) {
        self.visit_testable(item.ident.to_string(), &item.attrs, |this| {
            intravisit::walk_impl_item(this, item);
        });
    }

    fn visit_foreign_item(&mut self, item: &'hir hir::ForeignItem) {
        self.visit_testable(item.ident.to_string(), &item.attrs, |this| {
            intravisit::walk_foreign_item(this, item);
        });
    }

    fn visit_variant(&mut self,
                     v: &'hir hir::Variant,
                     g: &'hir hir::Generics,
                     item_id: hir::HirId) {
        self.visit_testable(v.ident.to_string(), &v.attrs, |this| {
            intravisit::walk_variant(this, v, g, item_id);
        });
    }

    fn visit_struct_field(&mut self, f: &'hir hir::StructField) {
        self.visit_testable(f.ident.to_string(), &f.attrs, |this| {
            intravisit::walk_struct_field(this, f);
        });
    }

    fn visit_macro_def(&mut self, macro_def: &'hir hir::MacroDef) {
        self.visit_testable(macro_def.name.to_string(), &macro_def.attrs, |_| ());
    }
}

#[cfg(test)]
mod tests;
