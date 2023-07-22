//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(lazy_cell)]
#![feature(decl_macro)]
#![feature(ice_to_disk)]
#![feature(let_chains)]
#![recursion_limit = "256"]
#![allow(rustc::potential_query_instability)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate tracing;

pub extern crate rustc_plugin_impl as plugin;

use rustc_ast as ast;
use rustc_codegen_ssa::{traits::CodegenBackend, CodegenErrors, CodegenResults};
use rustc_data_structures::profiling::{
    get_resident_set_size, print_time_passes_entry, TimePassesFormat,
};
use rustc_data_structures::sync::SeqCst;
use rustc_errors::registry::{InvalidErrorCode, Registry};
use rustc_errors::{markdown, ColorConfig};
use rustc_errors::{
    DiagnosticMessage, ErrorGuaranteed, Handler, PResult, SubdiagnosticMessage, TerminalUrl,
};
use rustc_feature::find_gated_cfg;
use rustc_fluent_macro::fluent_messages;
use rustc_interface::util::{self, collect_crate_types, get_codegen_backend};
use rustc_interface::{interface, Queries};
use rustc_lint::LintStore;
use rustc_metadata::locator;
use rustc_session::config::{nightly_options, CG_OPTIONS, Z_OPTIONS};
use rustc_session::config::{ErrorOutputType, Input, OutFileName, OutputType, TrimmedDefPaths};
use rustc_session::cstore::MetadataLoader;
use rustc_session::getopts::{self, Matches};
use rustc_session::lint::{Lint, LintId};
use rustc_session::{config, EarlyErrorHandler, Session};
use rustc_span::source_map::{FileLoader, FileName};
use rustc_span::symbol::sym;
use rustc_target::json::ToJson;
use rustc_target::spec::{Target, TargetTriple};

use std::cmp::max;
use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::panic::{self, catch_unwind};
use std::path::PathBuf;
use std::process::{self, Command, Stdio};
use std::str;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

#[allow(unused_macros)]
macro do_not_use_print($($t:tt)*) {
    std::compile_error!(
        "Don't use `print` or `println` here, use `safe_print` or `safe_println` instead"
    )
}

#[allow(unused_macros)]
macro do_not_use_safe_print($($t:tt)*) {
    std::compile_error!("Don't use `safe_print` or `safe_println` here, use `println_info` instead")
}

// This import blocks the use of panicking `print` and `println` in all the code
// below. Please use `safe_print` and `safe_println` to avoid ICE when
// encountering an I/O error during print.
#[allow(unused_imports)]
use {do_not_use_print as print, do_not_use_print as println};

pub mod args;
pub mod pretty;
#[macro_use]
mod print;
mod session_diagnostics;

use crate::session_diagnostics::{
    RLinkEmptyVersionNumber, RLinkEncodingVersionMismatch, RLinkRustcVersionMismatch,
    RLinkWrongFileType, RlinkNotAFile, RlinkUnableToRead,
};

fluent_messages! { "../messages.ftl" }

pub static DEFAULT_LOCALE_RESOURCES: &[&str] = &[
    // tidy-alphabetical-start
    crate::DEFAULT_LOCALE_RESOURCE,
    rustc_ast_lowering::DEFAULT_LOCALE_RESOURCE,
    rustc_ast_passes::DEFAULT_LOCALE_RESOURCE,
    rustc_attr::DEFAULT_LOCALE_RESOURCE,
    rustc_borrowck::DEFAULT_LOCALE_RESOURCE,
    rustc_builtin_macros::DEFAULT_LOCALE_RESOURCE,
    rustc_codegen_ssa::DEFAULT_LOCALE_RESOURCE,
    rustc_const_eval::DEFAULT_LOCALE_RESOURCE,
    rustc_error_messages::DEFAULT_LOCALE_RESOURCE,
    rustc_errors::DEFAULT_LOCALE_RESOURCE,
    rustc_expand::DEFAULT_LOCALE_RESOURCE,
    rustc_hir_analysis::DEFAULT_LOCALE_RESOURCE,
    rustc_hir_typeck::DEFAULT_LOCALE_RESOURCE,
    rustc_incremental::DEFAULT_LOCALE_RESOURCE,
    rustc_infer::DEFAULT_LOCALE_RESOURCE,
    rustc_interface::DEFAULT_LOCALE_RESOURCE,
    rustc_lint::DEFAULT_LOCALE_RESOURCE,
    rustc_metadata::DEFAULT_LOCALE_RESOURCE,
    rustc_middle::DEFAULT_LOCALE_RESOURCE,
    rustc_mir_build::DEFAULT_LOCALE_RESOURCE,
    rustc_mir_dataflow::DEFAULT_LOCALE_RESOURCE,
    rustc_mir_transform::DEFAULT_LOCALE_RESOURCE,
    rustc_monomorphize::DEFAULT_LOCALE_RESOURCE,
    rustc_parse::DEFAULT_LOCALE_RESOURCE,
    rustc_passes::DEFAULT_LOCALE_RESOURCE,
    rustc_plugin_impl::DEFAULT_LOCALE_RESOURCE,
    rustc_privacy::DEFAULT_LOCALE_RESOURCE,
    rustc_query_system::DEFAULT_LOCALE_RESOURCE,
    rustc_resolve::DEFAULT_LOCALE_RESOURCE,
    rustc_session::DEFAULT_LOCALE_RESOURCE,
    rustc_symbol_mangling::DEFAULT_LOCALE_RESOURCE,
    rustc_trait_selection::DEFAULT_LOCALE_RESOURCE,
    rustc_ty_utils::DEFAULT_LOCALE_RESOURCE,
    // tidy-alphabetical-end
];

/// Exit status code used for successful compilation and help output.
pub const EXIT_SUCCESS: i32 = 0;

/// Exit status code used for compilation failures and invalid flags.
pub const EXIT_FAILURE: i32 = 1;

pub const DEFAULT_BUG_REPORT_URL: &str = "https://github.com/rust-lang/rust/issues/new\
    ?labels=C-bug%2C+I-ICE%2C+T-compiler&template=ice.md";

const ICE_REPORT_COMPILER_FLAGS: &[&str] = &["-Z", "-C", "--crate-type"];

const ICE_REPORT_COMPILER_FLAGS_EXCLUDE: &[&str] = &["metadata", "extra-filename"];

const ICE_REPORT_COMPILER_FLAGS_STRIP_VALUE: &[&str] = &["incremental"];

pub fn abort_on_err<T>(result: Result<T, ErrorGuaranteed>, sess: &Session) -> T {
    match result {
        Err(..) => {
            sess.abort_if_errors();
            panic!("error reported but abort_if_errors didn't abort???");
        }
        Ok(x) => x,
    }
}

pub trait Callbacks {
    /// Called before creating the compiler instance
    fn config(&mut self, _config: &mut interface::Config) {}
    /// Called after parsing. Return value instructs the compiler whether to
    /// continue the compilation afterwards (defaults to `Compilation::Continue`)
    fn after_parsing<'tcx>(
        &mut self,
        _compiler: &interface::Compiler,
        _queries: &'tcx Queries<'tcx>,
    ) -> Compilation {
        Compilation::Continue
    }
    /// Called after expansion. Return value instructs the compiler whether to
    /// continue the compilation afterwards (defaults to `Compilation::Continue`)
    fn after_expansion<'tcx>(
        &mut self,
        _compiler: &interface::Compiler,
        _queries: &'tcx Queries<'tcx>,
    ) -> Compilation {
        Compilation::Continue
    }
    /// Called after analysis. Return value instructs the compiler whether to
    /// continue the compilation afterwards (defaults to `Compilation::Continue`)
    fn after_analysis<'tcx>(
        &mut self,
        _handler: &EarlyErrorHandler,
        _compiler: &interface::Compiler,
        _queries: &'tcx Queries<'tcx>,
    ) -> Compilation {
        Compilation::Continue
    }
}

#[derive(Default)]
pub struct TimePassesCallbacks {
    time_passes: Option<TimePassesFormat>,
}

impl Callbacks for TimePassesCallbacks {
    // JUSTIFICATION: the session doesn't exist at this point.
    #[allow(rustc::bad_opt_access)]
    fn config(&mut self, config: &mut interface::Config) {
        // If a --print=... option has been given, we don't print the "total"
        // time because it will mess up the --print output. See #64339.
        //
        self.time_passes = (config.opts.prints.is_empty() && config.opts.unstable_opts.time_passes)
            .then(|| config.opts.unstable_opts.time_passes_format);
        config.opts.trimmed_def_paths = TrimmedDefPaths::GoodPath;
    }
}

pub fn diagnostics_registry() -> Registry {
    Registry::new(rustc_error_codes::DIAGNOSTICS)
}

/// This is the primary entry point for rustc.
pub struct RunCompiler<'a, 'b> {
    at_args: &'a [String],
    callbacks: &'b mut (dyn Callbacks + Send),
    file_loader: Option<Box<dyn FileLoader + Send + Sync>>,
    make_codegen_backend:
        Option<Box<dyn FnOnce(&config::Options) -> Box<dyn CodegenBackend> + Send>>,
}

impl<'a, 'b> RunCompiler<'a, 'b> {
    pub fn new(at_args: &'a [String], callbacks: &'b mut (dyn Callbacks + Send)) -> Self {
        Self { at_args, callbacks, file_loader: None, make_codegen_backend: None }
    }

    /// Set a custom codegen backend.
    ///
    /// Has no uses within this repository, but is used by bjorn3 for "the
    /// hotswapping branch of cg_clif" for "setting the codegen backend from a
    /// custom driver where the custom codegen backend has arbitrary data."
    /// (See #102759.)
    pub fn set_make_codegen_backend(
        &mut self,
        make_codegen_backend: Option<
            Box<dyn FnOnce(&config::Options) -> Box<dyn CodegenBackend> + Send>,
        >,
    ) -> &mut Self {
        self.make_codegen_backend = make_codegen_backend;
        self
    }

    /// Load files from sources other than the file system.
    ///
    /// Has no uses within this repository, but may be used in the future by
    /// bjorn3 for "hooking rust-analyzer's VFS into rustc at some point for
    /// running rustc without having to save". (See #102759.)
    pub fn set_file_loader(
        &mut self,
        file_loader: Option<Box<dyn FileLoader + Send + Sync>>,
    ) -> &mut Self {
        self.file_loader = file_loader;
        self
    }

    /// Parse args and run the compiler.
    pub fn run(self) -> interface::Result<()> {
        run_compiler(self.at_args, self.callbacks, self.file_loader, self.make_codegen_backend)
    }
}

fn run_compiler(
    at_args: &[String],
    callbacks: &mut (dyn Callbacks + Send),
    file_loader: Option<Box<dyn FileLoader + Send + Sync>>,
    make_codegen_backend: Option<
        Box<dyn FnOnce(&config::Options) -> Box<dyn CodegenBackend> + Send>,
    >,
) -> interface::Result<()> {
    let mut early_error_handler = EarlyErrorHandler::new(ErrorOutputType::default());

    // Throw away the first argument, the name of the binary.
    // In case of at_args being empty, as might be the case by
    // passing empty argument array to execve under some platforms,
    // just use an empty slice.
    //
    // This situation was possible before due to arg_expand_all being
    // called before removing the argument, enabling a crash by calling
    // the compiler with @empty_file as argv[0] and no more arguments.
    let at_args = at_args.get(1..).unwrap_or_default();

    let args = args::arg_expand_all(&early_error_handler, at_args);

    let Some(matches) = handle_options(&early_error_handler, &args) else { return Ok(()) };

    let sopts = config::build_session_options(&mut early_error_handler, &matches);

    if let Some(ref code) = matches.opt_str("explain") {
        handle_explain(&early_error_handler, diagnostics_registry(), code, sopts.color);
        return Ok(());
    }

    let cfg = interface::parse_cfgspecs(&early_error_handler, matches.opt_strs("cfg"));
    let check_cfg = interface::parse_check_cfg(&early_error_handler, matches.opt_strs("check-cfg"));
    let (odir, ofile) = make_output(&matches);
    let mut config = interface::Config {
        opts: sopts,
        crate_cfg: cfg,
        crate_check_cfg: check_cfg,
        input: Input::File(PathBuf::new()),
        output_file: ofile,
        output_dir: odir,
        ice_file: ice_path().clone(),
        file_loader,
        locale_resources: DEFAULT_LOCALE_RESOURCES,
        lint_caps: Default::default(),
        parse_sess_created: None,
        register_lints: None,
        override_queries: None,
        make_codegen_backend,
        registry: diagnostics_registry(),
    };

    match make_input(&early_error_handler, &matches.free) {
        Err(reported) => return Err(reported),
        Ok(Some(input)) => {
            config.input = input;

            callbacks.config(&mut config);
        }
        Ok(None) => match matches.free.len() {
            0 => {
                callbacks.config(&mut config);

                early_error_handler.abort_if_errors();

                interface::run_compiler(config, |compiler| {
                    let sopts = &compiler.session().opts;
                    let handler = EarlyErrorHandler::new(sopts.error_format);

                    if sopts.describe_lints {
                        let mut lint_store =
                            rustc_lint::new_lint_store(compiler.session().enable_internal_lints());
                        let registered_lints =
                            if let Some(register_lints) = compiler.register_lints() {
                                register_lints(compiler.session(), &mut lint_store);
                                true
                            } else {
                                false
                            };
                        describe_lints(compiler.session(), &lint_store, registered_lints);
                        return;
                    }
                    let should_stop = print_crate_info(
                        &handler,
                        &**compiler.codegen_backend(),
                        compiler.session(),
                        false,
                    );

                    if should_stop == Compilation::Stop {
                        return;
                    }
                    handler.early_error("no input filename given")
                });
                return Ok(());
            }
            1 => panic!("make_input should have provided valid inputs"),
            _ => early_error_handler.early_error(format!(
                "multiple input filenames provided (first two filenames are `{}` and `{}`)",
                matches.free[0], matches.free[1],
            )),
        },
    };

    early_error_handler.abort_if_errors();

    interface::run_compiler(config, |compiler| {
        let sess = compiler.session();
        let handler = EarlyErrorHandler::new(sess.opts.error_format);

        let should_stop = print_crate_info(&handler, &**compiler.codegen_backend(), sess, true)
            .and_then(|| {
                list_metadata(&handler, sess, &*compiler.codegen_backend().metadata_loader())
            })
            .and_then(|| try_process_rlink(sess, compiler));

        if should_stop == Compilation::Stop {
            return sess.compile_status();
        }

        let linker = compiler.enter(|queries| {
            let early_exit = || sess.compile_status().map(|_| None);
            queries.parse()?;

            if let Some(ppm) = &sess.opts.pretty {
                if ppm.needs_ast_map() {
                    queries.global_ctxt()?.enter(|tcx| {
                        tcx.ensure().early_lint_checks(());
                        pretty::print_after_hir_lowering(tcx, *ppm);
                        Ok(())
                    })?;
                } else {
                    let krate = queries.parse()?.steal();
                    pretty::print_after_parsing(sess, &krate, *ppm);
                }
                trace!("finished pretty-printing");
                return early_exit();
            }

            if callbacks.after_parsing(compiler, queries) == Compilation::Stop {
                return early_exit();
            }

            if sess.opts.unstable_opts.parse_only || sess.opts.unstable_opts.show_span.is_some() {
                return early_exit();
            }

            {
                let plugins = queries.register_plugins()?;
                let (.., lint_store) = &*plugins.borrow();

                // Lint plugins are registered; now we can process command line flags.
                if sess.opts.describe_lints {
                    describe_lints(sess, lint_store, true);
                    return early_exit();
                }
            }

            // Make sure name resolution and macro expansion is run.
            queries.global_ctxt()?.enter(|tcx| tcx.resolver_for_lowering(()));

            if callbacks.after_expansion(compiler, queries) == Compilation::Stop {
                return early_exit();
            }

            // Make sure the `output_filenames` query is run for its side
            // effects of writing the dep-info and reporting errors.
            queries.global_ctxt()?.enter(|tcx| tcx.output_filenames(()));

            if sess.opts.output_types.contains_key(&OutputType::DepInfo)
                && sess.opts.output_types.len() == 1
            {
                return early_exit();
            }

            if sess.opts.unstable_opts.no_analysis {
                return early_exit();
            }

            queries.global_ctxt()?.enter(|tcx| tcx.analysis(()))?;

            if callbacks.after_analysis(&handler, compiler, queries) == Compilation::Stop {
                return early_exit();
            }

            let ongoing_codegen = queries.ongoing_codegen()?;

            if sess.opts.unstable_opts.print_type_sizes {
                sess.code_stats.print_type_sizes();
            }

            if sess.opts.unstable_opts.print_vtable_sizes {
                let crate_name =
                    compiler.session().opts.crate_name.as_deref().unwrap_or("<UNKNOWN_CRATE>");

                sess.code_stats.print_vtable_sizes(crate_name);
            }

            let linker = queries.linker(ongoing_codegen)?;
            Ok(Some(linker))
        })?;

        if let Some(linker) = linker {
            let _timer = sess.timer("link");
            linker.link()?
        }

        if sess.opts.unstable_opts.perf_stats {
            sess.print_perf_stats();
        }

        if sess.opts.unstable_opts.print_fuel.is_some() {
            eprintln!(
                "Fuel used by {}: {}",
                sess.opts.unstable_opts.print_fuel.as_ref().unwrap(),
                sess.print_fuel.load(SeqCst)
            );
        }

        Ok(())
    })
}

// Extract output directory and file from matches.
fn make_output(matches: &getopts::Matches) -> (Option<PathBuf>, Option<OutFileName>) {
    let odir = matches.opt_str("out-dir").map(|o| PathBuf::from(&o));
    let ofile = matches.opt_str("o").map(|o| match o.as_str() {
        "-" => OutFileName::Stdout,
        path => OutFileName::Real(PathBuf::from(path)),
    });
    (odir, ofile)
}

// Extract input (string or file and optional path) from matches.
fn make_input(
    handler: &EarlyErrorHandler,
    free_matches: &[String],
) -> Result<Option<Input>, ErrorGuaranteed> {
    if free_matches.len() == 1 {
        let ifile = &free_matches[0];
        if ifile == "-" {
            let mut src = String::new();
            if io::stdin().read_to_string(&mut src).is_err() {
                // Immediately stop compilation if there was an issue reading
                // the input (for example if the input stream is not UTF-8).
                let reported = handler.early_error_no_abort(
                    "couldn't read from stdin, as it did not contain valid UTF-8",
                );
                return Err(reported);
            }
            if let Ok(path) = env::var("UNSTABLE_RUSTDOC_TEST_PATH") {
                let line = env::var("UNSTABLE_RUSTDOC_TEST_LINE").expect(
                    "when UNSTABLE_RUSTDOC_TEST_PATH is set \
                                    UNSTABLE_RUSTDOC_TEST_LINE also needs to be set",
                );
                let line = isize::from_str_radix(&line, 10)
                    .expect("UNSTABLE_RUSTDOC_TEST_LINE needs to be an number");
                let file_name = FileName::doc_test_source_code(PathBuf::from(path), line);
                Ok(Some(Input::Str { name: file_name, input: src }))
            } else {
                Ok(Some(Input::Str { name: FileName::anon_source_code(&src), input: src }))
            }
        } else {
            Ok(Some(Input::File(PathBuf::from(ifile))))
        }
    } else {
        Ok(None)
    }
}

/// Whether to stop or continue compilation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Compilation {
    Stop,
    Continue,
}

impl Compilation {
    pub fn and_then<F: FnOnce() -> Compilation>(self, next: F) -> Compilation {
        match self {
            Compilation::Stop => Compilation::Stop,
            Compilation::Continue => next(),
        }
    }
}

fn handle_explain(handler: &EarlyErrorHandler, registry: Registry, code: &str, color: ColorConfig) {
    let upper_cased_code = code.to_ascii_uppercase();
    let normalised =
        if upper_cased_code.starts_with('E') { upper_cased_code } else { format!("E{code:0>4}") };
    match registry.try_find_description(&normalised) {
        Ok(description) => {
            let mut is_in_code_block = false;
            let mut text = String::new();
            // Slice off the leading newline and print.
            for line in description.lines() {
                let indent_level =
                    line.find(|c: char| !c.is_whitespace()).unwrap_or_else(|| line.len());
                let dedented_line = &line[indent_level..];
                if dedented_line.starts_with("```") {
                    is_in_code_block = !is_in_code_block;
                    text.push_str(&line[..(indent_level + 3)]);
                } else if is_in_code_block && dedented_line.starts_with("# ") {
                    continue;
                } else {
                    text.push_str(line);
                }
                text.push('\n');
            }
            if io::stdout().is_terminal() {
                show_md_content_with_pager(&text, color);
            } else {
                safe_print!("{text}");
            }
        }
        Err(InvalidErrorCode) => {
            handler.early_error(format!("{code} is not a valid error code"));
        }
    }
}

/// If color is always or auto, print formatted & colorized markdown. If color is never or
/// if formatted printing fails, print the raw text.
///
/// Prefers a pager, falls back standard print
fn show_md_content_with_pager(content: &str, color: ColorConfig) {
    let mut fallback_to_println = false;
    let pager_name = env::var_os("PAGER").unwrap_or_else(|| {
        if cfg!(windows) { OsString::from("more.com") } else { OsString::from("less") }
    });

    let mut cmd = Command::new(&pager_name);
    // FIXME: find if other pagers accept color options
    let mut print_formatted = if pager_name == "less" {
        cmd.arg("-r");
        true
    } else if ["bat", "catbat", "delta"].iter().any(|v| *v == pager_name) {
        true
    } else {
        false
    };

    if color == ColorConfig::Never {
        print_formatted = false;
    } else if color == ColorConfig::Always {
        print_formatted = true;
    }

    let mdstream = markdown::MdStream::parse_str(content);
    let bufwtr = markdown::create_stdout_bufwtr();
    let mut mdbuf = bufwtr.buffer();
    if mdstream.write_termcolor_buf(&mut mdbuf).is_err() {
        print_formatted = false;
    }

    if let Ok(mut pager) = cmd.stdin(Stdio::piped()).spawn() {
        if let Some(pipe) = pager.stdin.as_mut() {
            let res = if print_formatted {
                pipe.write_all(mdbuf.as_slice())
            } else {
                pipe.write_all(content.as_bytes())
            };

            if res.is_err() {
                fallback_to_println = true;
            }
        }

        if pager.wait().is_err() {
            fallback_to_println = true;
        }
    } else {
        fallback_to_println = true;
    }

    // If pager fails for whatever reason, we should still print the content
    // to standard output
    if fallback_to_println {
        let fmt_success = match color {
            ColorConfig::Auto => io::stdout().is_terminal() && bufwtr.print(&mdbuf).is_ok(),
            ColorConfig::Always => bufwtr.print(&mdbuf).is_ok(),
            ColorConfig::Never => false,
        };

        if !fmt_success {
            safe_print!("{content}");
        }
    }
}

pub fn try_process_rlink(sess: &Session, compiler: &interface::Compiler) -> Compilation {
    if sess.opts.unstable_opts.link_only {
        if let Input::File(file) = &sess.io.input {
            // FIXME: #![crate_type] and #![crate_name] support not implemented yet
            sess.init_crate_types(collect_crate_types(sess, &[]));
            let outputs = compiler.build_output_filenames(sess, &[]);
            let rlink_data = fs::read(file).unwrap_or_else(|err| {
                sess.emit_fatal(RlinkUnableToRead { err });
            });
            let codegen_results = match CodegenResults::deserialize_rlink(sess, rlink_data) {
                Ok(codegen) => codegen,
                Err(err) => {
                    match err {
                        CodegenErrors::WrongFileType => sess.emit_fatal(RLinkWrongFileType),
                        CodegenErrors::EmptyVersionNumber => {
                            sess.emit_fatal(RLinkEmptyVersionNumber)
                        }
                        CodegenErrors::EncodingVersionMismatch { version_array, rlink_version } => {
                            sess.emit_fatal(RLinkEncodingVersionMismatch {
                                version_array,
                                rlink_version,
                            })
                        }
                        CodegenErrors::RustcVersionMismatch { rustc_version } => {
                            sess.emit_fatal(RLinkRustcVersionMismatch {
                                rustc_version,
                                current_version: sess.cfg_version,
                            })
                        }
                    };
                }
            };
            let result = compiler.codegen_backend().link(sess, codegen_results, &outputs);
            abort_on_err(result, sess);
        } else {
            sess.emit_fatal(RlinkNotAFile {})
        }
        Compilation::Stop
    } else {
        Compilation::Continue
    }
}

pub fn list_metadata(
    handler: &EarlyErrorHandler,
    sess: &Session,
    metadata_loader: &dyn MetadataLoader,
) -> Compilation {
    if sess.opts.unstable_opts.ls {
        match sess.io.input {
            Input::File(ref ifile) => {
                let path = &(*ifile);
                let mut v = Vec::new();
                locator::list_file_metadata(&sess.target, path, metadata_loader, &mut v).unwrap();
                safe_println!("{}", String::from_utf8(v).unwrap());
            }
            Input::Str { .. } => {
                handler.early_error("cannot list metadata for stdin");
            }
        }
        return Compilation::Stop;
    }

    Compilation::Continue
}

fn print_crate_info(
    handler: &EarlyErrorHandler,
    codegen_backend: &dyn CodegenBackend,
    sess: &Session,
    parse_attrs: bool,
) -> Compilation {
    use rustc_session::config::PrintKind::*;

    // This import prevents the following code from using the printing macros
    // used by the rest of the module. Within this function, we only write to
    // the output specified by `sess.io.output_file`.
    #[allow(unused_imports)]
    use {do_not_use_safe_print as safe_print, do_not_use_safe_print as safe_println};

    // NativeStaticLibs and LinkArgs are special - printed during linking
    // (empty iterator returns true)
    if sess.opts.prints.iter().all(|p| p.kind == NativeStaticLibs || p.kind == LinkArgs) {
        return Compilation::Continue;
    }

    let attrs = if parse_attrs {
        let result = parse_crate_attrs(sess);
        match result {
            Ok(attrs) => Some(attrs),
            Err(mut parse_error) => {
                parse_error.emit();
                return Compilation::Stop;
            }
        }
    } else {
        None
    };

    for req in &sess.opts.prints {
        let mut crate_info = String::new();
        macro println_info($($arg:tt)*) {
            crate_info.write_fmt(format_args!("{}\n", format_args!($($arg)*))).unwrap()
        }

        match req.kind {
            TargetList => {
                let mut targets = rustc_target::spec::TARGETS.to_vec();
                targets.sort_unstable();
                println_info!("{}", targets.join("\n"));
            }
            Sysroot => println_info!("{}", sess.sysroot.display()),
            TargetLibdir => println_info!("{}", sess.target_tlib_path.dir.display()),
            TargetSpec => {
                println_info!("{}", serde_json::to_string_pretty(&sess.target.to_json()).unwrap());
            }
            AllTargetSpecs => {
                let mut targets = BTreeMap::new();
                for name in rustc_target::spec::TARGETS {
                    let triple = TargetTriple::from_triple(name);
                    let target = Target::expect_builtin(&triple);
                    targets.insert(name, target.to_json());
                }
                println_info!("{}", serde_json::to_string_pretty(&targets).unwrap());
            }
            FileNames => {
                let Some(attrs) = attrs.as_ref() else {
                    // no crate attributes, print out an error and exit
                    return Compilation::Continue;
                };
                let t_outputs = rustc_interface::util::build_output_filenames(attrs, sess);
                let id = rustc_session::output::find_crate_name(sess, attrs);
                let crate_types = collect_crate_types(sess, attrs);
                for &style in &crate_types {
                    let fname =
                        rustc_session::output::filename_for_input(sess, style, id, &t_outputs);
                    println_info!("{}", fname.as_path().file_name().unwrap().to_string_lossy());
                }
            }
            CrateName => {
                let Some(attrs) = attrs.as_ref() else {
                    // no crate attributes, print out an error and exit
                    return Compilation::Continue;
                };
                let id = rustc_session::output::find_crate_name(sess, attrs);
                println_info!("{id}");
            }
            Cfg => {
                let mut cfgs = sess
                    .parse_sess
                    .config
                    .iter()
                    .filter_map(|&(name, value)| {
                        // Note that crt-static is a specially recognized cfg
                        // directive that's printed out here as part of
                        // rust-lang/rust#37406, but in general the
                        // `target_feature` cfg is gated under
                        // rust-lang/rust#29717. For now this is just
                        // specifically allowing the crt-static cfg and that's
                        // it, this is intended to get into Cargo and then go
                        // through to build scripts.
                        if (name != sym::target_feature || value != Some(sym::crt_dash_static))
                            && !sess.is_nightly_build()
                            && find_gated_cfg(|cfg_sym| cfg_sym == name).is_some()
                        {
                            return None;
                        }

                        if let Some(value) = value {
                            Some(format!("{name}=\"{value}\""))
                        } else {
                            Some(name.to_string())
                        }
                    })
                    .collect::<Vec<String>>();

                cfgs.sort();
                for cfg in cfgs {
                    println_info!("{cfg}");
                }
            }
            CallingConventions => {
                let mut calling_conventions = rustc_target::spec::abi::all_names();
                calling_conventions.sort_unstable();
                println_info!("{}", calling_conventions.join("\n"));
            }
            RelocationModels
            | CodeModels
            | TlsModels
            | TargetCPUs
            | StackProtectorStrategies
            | TargetFeatures => {
                codegen_backend.print(req, &mut crate_info, sess);
            }
            // Any output here interferes with Cargo's parsing of other printed output
            NativeStaticLibs => {}
            LinkArgs => {}
            SplitDebuginfo => {
                use rustc_target::spec::SplitDebuginfo::{Off, Packed, Unpacked};

                for split in &[Off, Packed, Unpacked] {
                    if sess.target.options.supported_split_debuginfo.contains(split) {
                        println_info!("{split}");
                    }
                }
            }
            DeploymentTarget => {
                use rustc_target::spec::current_apple_deployment_target;

                if sess.target.is_like_osx {
                    println_info!(
                        "deployment_target={}",
                        current_apple_deployment_target(&sess.target)
                            .expect("unknown Apple target OS")
                    )
                } else {
                    handler
                        .early_error("only Apple targets currently support deployment version info")
                }
            }
        }

        req.out.overwrite(&crate_info, sess);
    }
    Compilation::Stop
}

/// Prints version information
///
/// NOTE: this is a macro to support drivers built at a different time than the main `rustc_driver` crate.
pub macro version($handler: expr, $binary: literal, $matches: expr) {
    fn unw(x: Option<&str>) -> &str {
        x.unwrap_or("unknown")
    }
    $crate::version_at_macro_invocation(
        $handler,
        $binary,
        $matches,
        unw(option_env!("CFG_VERSION")),
        unw(option_env!("CFG_VER_HASH")),
        unw(option_env!("CFG_VER_DATE")),
        unw(option_env!("CFG_RELEASE")),
    )
}

#[doc(hidden)] // use the macro instead
pub fn version_at_macro_invocation(
    handler: &EarlyErrorHandler,
    binary: &str,
    matches: &getopts::Matches,
    version: &str,
    commit_hash: &str,
    commit_date: &str,
    release: &str,
) {
    let verbose = matches.opt_present("verbose");

    safe_println!("{binary} {version}");

    if verbose {
        safe_println!("binary: {binary}");
        safe_println!("commit-hash: {commit_hash}");
        safe_println!("commit-date: {commit_date}");
        safe_println!("host: {}", config::host_triple());
        safe_println!("release: {release}");

        let debug_flags = matches.opt_strs("Z");
        let backend_name = debug_flags.iter().find_map(|x| x.strip_prefix("codegen-backend="));
        get_codegen_backend(handler, &None, backend_name).print_version();
    }
}

fn usage(verbose: bool, include_unstable_options: bool, nightly_build: bool) {
    let groups = if verbose { config::rustc_optgroups() } else { config::rustc_short_optgroups() };
    let mut options = getopts::Options::new();
    for option in groups.iter().filter(|x| include_unstable_options || x.is_stable()) {
        (option.apply)(&mut options);
    }
    let message = "Usage: rustc [OPTIONS] INPUT";
    let nightly_help = if nightly_build {
        "\n    -Z help             Print unstable compiler options"
    } else {
        ""
    };
    let verbose_help = if verbose {
        ""
    } else {
        "\n    --help -v           Print the full set of options rustc accepts"
    };
    let at_path = if verbose {
        "    @path               Read newline separated options from `path`\n"
    } else {
        ""
    };
    safe_println!(
        "{options}{at_path}\nAdditional help:
    -C help             Print codegen options
    -W help             \
              Print 'lint' options and default settings{nightly}{verbose}\n",
        options = options.usage(message),
        at_path = at_path,
        nightly = nightly_help,
        verbose = verbose_help
    );
}

fn print_wall_help() {
    safe_println!(
        "
The flag `-Wall` does not exist in `rustc`. Most useful lints are enabled by
default. Use `rustc -W help` to see all available lints. It's more common to put
warning settings in the crate root using `#![warn(LINT_NAME)]` instead of using
the command line flag directly.
"
    );
}

/// Write to stdout lint command options, together with a list of all available lints
pub fn describe_lints(sess: &Session, lint_store: &LintStore, loaded_plugins: bool) {
    safe_println!(
        "
Available lint options:
    -W <foo>           Warn about <foo>
    -A <foo>           \
              Allow <foo>
    -D <foo>           Deny <foo>
    -F <foo>           Forbid <foo> \
              (deny <foo> and all attempts to override)

"
    );

    fn sort_lints(sess: &Session, mut lints: Vec<&'static Lint>) -> Vec<&'static Lint> {
        // The sort doesn't case-fold but it's doubtful we care.
        lints.sort_by_cached_key(|x: &&Lint| (x.default_level(sess.edition()), x.name));
        lints
    }

    fn sort_lint_groups(
        lints: Vec<(&'static str, Vec<LintId>, bool)>,
    ) -> Vec<(&'static str, Vec<LintId>)> {
        let mut lints: Vec<_> = lints.into_iter().map(|(x, y, _)| (x, y)).collect();
        lints.sort_by_key(|l| l.0);
        lints
    }

    let (plugin, builtin): (Vec<_>, _) =
        lint_store.get_lints().iter().cloned().partition(|&lint| lint.is_plugin);
    let plugin = sort_lints(sess, plugin);
    let builtin = sort_lints(sess, builtin);

    let (plugin_groups, builtin_groups): (Vec<_>, _) =
        lint_store.get_lint_groups().partition(|&(.., p)| p);
    let plugin_groups = sort_lint_groups(plugin_groups);
    let builtin_groups = sort_lint_groups(builtin_groups);

    let max_name_len =
        plugin.iter().chain(&builtin).map(|&s| s.name.chars().count()).max().unwrap_or(0);
    let padded = |x: &str| {
        let mut s = " ".repeat(max_name_len - x.chars().count());
        s.push_str(x);
        s
    };

    safe_println!("Lint checks provided by rustc:\n");

    let print_lints = |lints: Vec<&Lint>| {
        safe_println!("    {}  {:7.7}  {}", padded("name"), "default", "meaning");
        safe_println!("    {}  {:7.7}  {}", padded("----"), "-------", "-------");
        for lint in lints {
            let name = lint.name_lower().replace('_', "-");
            safe_println!(
                "    {}  {:7.7}  {}",
                padded(&name),
                lint.default_level(sess.edition()).as_str(),
                lint.desc
            );
        }
        safe_println!("\n");
    };

    print_lints(builtin);

    let max_name_len = max(
        "warnings".len(),
        plugin_groups
            .iter()
            .chain(&builtin_groups)
            .map(|&(s, _)| s.chars().count())
            .max()
            .unwrap_or(0),
    );

    let padded = |x: &str| {
        let mut s = " ".repeat(max_name_len - x.chars().count());
        s.push_str(x);
        s
    };

    safe_println!("Lint groups provided by rustc:\n");

    let print_lint_groups = |lints: Vec<(&'static str, Vec<LintId>)>, all_warnings| {
        safe_println!("    {}  sub-lints", padded("name"));
        safe_println!("    {}  ---------", padded("----"));

        if all_warnings {
            safe_println!("    {}  all lints that are set to issue warnings", padded("warnings"));
        }

        for (name, to) in lints {
            let name = name.to_lowercase().replace('_', "-");
            let desc = to
                .into_iter()
                .map(|x| x.to_string().replace('_', "-"))
                .collect::<Vec<String>>()
                .join(", ");
            safe_println!("    {}  {}", padded(&name), desc);
        }
        safe_println!("\n");
    };

    print_lint_groups(builtin_groups, true);

    match (loaded_plugins, plugin.len(), plugin_groups.len()) {
        (false, 0, _) | (false, _, 0) => {
            safe_println!("Lint tools like Clippy can provide additional lints and lint groups.");
        }
        (false, ..) => panic!("didn't load lint plugins but got them anyway!"),
        (true, 0, 0) => safe_println!("This crate does not load any lint plugins or lint groups."),
        (true, l, g) => {
            if l > 0 {
                safe_println!("Lint checks provided by plugins loaded by this crate:\n");
                print_lints(plugin);
            }
            if g > 0 {
                safe_println!("Lint groups provided by plugins loaded by this crate:\n");
                print_lint_groups(plugin_groups, false);
            }
        }
    }
}

/// Show help for flag categories shared between rustdoc and rustc.
///
/// Returns whether a help option was printed.
pub fn describe_flag_categories(handler: &EarlyErrorHandler, matches: &Matches) -> bool {
    // Handle the special case of -Wall.
    let wall = matches.opt_strs("W");
    if wall.iter().any(|x| *x == "all") {
        print_wall_help();
        rustc_errors::FatalError.raise();
    }

    // Don't handle -W help here, because we might first load plugins.
    let debug_flags = matches.opt_strs("Z");
    if debug_flags.iter().any(|x| *x == "help") {
        describe_debug_flags();
        return true;
    }

    let cg_flags = matches.opt_strs("C");
    if cg_flags.iter().any(|x| *x == "help") {
        describe_codegen_flags();
        return true;
    }

    if cg_flags.iter().any(|x| *x == "no-stack-check") {
        handler.early_warn("the --no-stack-check flag is deprecated and does nothing");
    }

    if cg_flags.iter().any(|x| *x == "passes=list") {
        let backend_name = debug_flags.iter().find_map(|x| x.strip_prefix("codegen-backend="));
        get_codegen_backend(handler, &None, backend_name).print_passes();
        return true;
    }

    false
}

fn describe_debug_flags() {
    safe_println!("\nAvailable options:\n");
    print_flag_list("-Z", config::Z_OPTIONS);
}

fn describe_codegen_flags() {
    safe_println!("\nAvailable codegen options:\n");
    print_flag_list("-C", config::CG_OPTIONS);
}

fn print_flag_list<T>(
    cmdline_opt: &str,
    flag_list: &[(&'static str, T, &'static str, &'static str)],
) {
    let max_len = flag_list.iter().map(|&(name, _, _, _)| name.chars().count()).max().unwrap_or(0);

    for &(name, _, _, desc) in flag_list {
        safe_println!(
            "    {} {:>width$}=val -- {}",
            cmdline_opt,
            name.replace('_', "-"),
            desc,
            width = max_len
        );
    }
}

/// Process command line options. Emits messages as appropriate. If compilation
/// should continue, returns a getopts::Matches object parsed from args,
/// otherwise returns `None`.
///
/// The compiler's handling of options is a little complicated as it ties into
/// our stability story. The current intention of each compiler option is to
/// have one of two modes:
///
/// 1. An option is stable and can be used everywhere.
/// 2. An option is unstable, and can only be used on nightly.
///
/// Like unstable library and language features, however, unstable options have
/// always required a form of "opt in" to indicate that you're using them. This
/// provides the easy ability to scan a code base to check to see if anything
/// unstable is being used. Currently, this "opt in" is the `-Z` "zed" flag.
///
/// All options behind `-Z` are considered unstable by default. Other top-level
/// options can also be considered unstable, and they were unlocked through the
/// `-Z unstable-options` flag. Note that `-Z` remains to be the root of
/// instability in both cases, though.
///
/// So with all that in mind, the comments below have some more detail about the
/// contortions done here to get things to work out correctly.
pub fn handle_options(handler: &EarlyErrorHandler, args: &[String]) -> Option<getopts::Matches> {
    if args.is_empty() {
        // user did not write `-v` nor `-Z unstable-options`, so do not
        // include that extra information.
        let nightly_build =
            rustc_feature::UnstableFeatures::from_environment(None).is_nightly_build();
        usage(false, false, nightly_build);
        return None;
    }

    // Parse with *all* options defined in the compiler, we don't worry about
    // option stability here we just want to parse as much as possible.
    let mut options = getopts::Options::new();
    for option in config::rustc_optgroups() {
        (option.apply)(&mut options);
    }
    let matches = options.parse(args).unwrap_or_else(|e| {
        let msg = match e {
            getopts::Fail::UnrecognizedOption(ref opt) => CG_OPTIONS
                .iter()
                .map(|&(name, ..)| ('C', name))
                .chain(Z_OPTIONS.iter().map(|&(name, ..)| ('Z', name)))
                .find(|&(_, name)| *opt == name.replace('_', "-"))
                .map(|(flag, _)| format!("{e}. Did you mean `-{flag} {opt}`?")),
            _ => None,
        };
        handler.early_error(msg.unwrap_or_else(|| e.to_string()));
    });

    // For all options we just parsed, we check a few aspects:
    //
    // * If the option is stable, we're all good
    // * If the option wasn't passed, we're all good
    // * If `-Z unstable-options` wasn't passed (and we're not a -Z option
    //   ourselves), then we require the `-Z unstable-options` flag to unlock
    //   this option that was passed.
    // * If we're a nightly compiler, then unstable options are now unlocked, so
    //   we're good to go.
    // * Otherwise, if we're an unstable option then we generate an error
    //   (unstable option being used on stable)
    nightly_options::check_nightly_options(handler, &matches, &config::rustc_optgroups());

    if matches.opt_present("h") || matches.opt_present("help") {
        // Only show unstable options in --help if we accept unstable options.
        let unstable_enabled = nightly_options::is_unstable_enabled(&matches);
        let nightly_build = nightly_options::match_is_nightly_build(&matches);
        usage(matches.opt_present("verbose"), unstable_enabled, nightly_build);
        return None;
    }

    if describe_flag_categories(handler, &matches) {
        return None;
    }

    if matches.opt_present("version") {
        version!(handler, "rustc", &matches);
        return None;
    }

    Some(matches)
}

fn parse_crate_attrs<'a>(sess: &'a Session) -> PResult<'a, ast::AttrVec> {
    match &sess.io.input {
        Input::File(ifile) => rustc_parse::parse_crate_attrs_from_file(ifile, &sess.parse_sess),
        Input::Str { name, input } => rustc_parse::parse_crate_attrs_from_source_str(
            name.clone(),
            input.clone(),
            &sess.parse_sess,
        ),
    }
}

/// Gets a list of extra command-line flags provided by the user, as strings.
///
/// This function is used during ICEs to show more information useful for
/// debugging, since some ICEs only happens with non-default compiler flags
/// (and the users don't always report them).
fn extra_compiler_flags() -> Option<(Vec<String>, bool)> {
    let mut args = env::args_os().map(|arg| arg.to_string_lossy().to_string()).peekable();

    let mut result = Vec::new();
    let mut excluded_cargo_defaults = false;
    while let Some(arg) = args.next() {
        if let Some(a) = ICE_REPORT_COMPILER_FLAGS.iter().find(|a| arg.starts_with(*a)) {
            let content = if arg.len() == a.len() {
                // A space-separated option, like `-C incremental=foo` or `--crate-type rlib`
                match args.next() {
                    Some(arg) => arg.to_string(),
                    None => continue,
                }
            } else if arg.get(a.len()..a.len() + 1) == Some("=") {
                // An equals option, like `--crate-type=rlib`
                arg[a.len() + 1..].to_string()
            } else {
                // A non-space option, like `-Cincremental=foo`
                arg[a.len()..].to_string()
            };
            let option = content.split_once('=').map(|s| s.0).unwrap_or(&content);
            if ICE_REPORT_COMPILER_FLAGS_EXCLUDE.iter().any(|exc| option == *exc) {
                excluded_cargo_defaults = true;
            } else {
                result.push(a.to_string());
                match ICE_REPORT_COMPILER_FLAGS_STRIP_VALUE.iter().find(|s| option == **s) {
                    Some(s) => result.push(format!("{s}=[REDACTED]")),
                    None => result.push(content),
                }
            }
        }
    }

    if !result.is_empty() { Some((result, excluded_cargo_defaults)) } else { None }
}

/// Runs a closure and catches unwinds triggered by fatal errors.
///
/// The compiler currently unwinds with a special sentinel value to abort
/// compilation on fatal errors. This function catches that sentinel and turns
/// the panic into a `Result` instead.
pub fn catch_fatal_errors<F: FnOnce() -> R, R>(f: F) -> Result<R, ErrorGuaranteed> {
    catch_unwind(panic::AssertUnwindSafe(f)).map_err(|value| {
        if value.is::<rustc_errors::FatalErrorMarker>() {
            #[allow(deprecated)]
            ErrorGuaranteed::unchecked_claim_error_was_emitted()
        } else {
            panic::resume_unwind(value);
        }
    })
}

/// Variant of `catch_fatal_errors` for the `interface::Result` return type
/// that also computes the exit code.
pub fn catch_with_exit_code(f: impl FnOnce() -> interface::Result<()>) -> i32 {
    let result = catch_fatal_errors(f).and_then(|result| result);
    match result {
        Ok(()) => EXIT_SUCCESS,
        Err(_) => EXIT_FAILURE,
    }
}

pub static ICE_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

pub fn ice_path() -> &'static Option<PathBuf> {
    ICE_PATH.get_or_init(|| {
        if !rustc_feature::UnstableFeatures::from_environment(None).is_nightly_build() {
            return None;
        }
        if let Ok("0") = std::env::var("RUST_BACKTRACE").as_deref() {
            return None;
        }
        let mut path = match std::env::var("RUSTC_ICE").as_deref() {
            // Explicitly opting out of writing ICEs to disk.
            Ok("0") => return None,
            Ok(s) => PathBuf::from(s),
            Err(_) => std::env::current_dir().unwrap_or_default(),
        };
        let now: OffsetDateTime = SystemTime::now().into();
        let file_now = now.format(&Rfc3339).unwrap_or(String::new());
        let pid = std::process::id();
        path.push(format!("rustc-ice-{file_now}-{pid}.txt"));
        Some(path)
    })
}

/// Installs a panic hook that will print the ICE message on unexpected panics.
///
/// The hook is intended to be useable even by external tools. You can pass a custom
/// `bug_report_url`, or report arbitrary info in `extra_info`. Note that `extra_info` is called in
/// a context where *the thread is currently panicking*, so it must not panic or the process will
/// abort.
///
/// If you have no extra info to report, pass the empty closure `|_| ()` as the argument to
/// extra_info.
///
/// A custom rustc driver can skip calling this to set up a custom ICE hook.
pub fn install_ice_hook(bug_report_url: &'static str, extra_info: fn(&Handler)) {
    // If the user has not explicitly overridden "RUST_BACKTRACE", then produce
    // full backtraces. When a compiler ICE happens, we want to gather
    // as much information as possible to present in the issue opened
    // by the user. Compiler developers and other rustc users can
    // opt in to less-verbose backtraces by manually setting "RUST_BACKTRACE"
    // (e.g. `RUST_BACKTRACE=1`)
    if std::env::var("RUST_BACKTRACE").is_err() {
        std::env::set_var("RUST_BACKTRACE", "full");
    }

    panic::set_hook(Box::new(move |info| {
        // If the error was caused by a broken pipe then this is not a bug.
        // Write the error and return immediately. See #98700.
        #[cfg(windows)]
        if let Some(msg) = info.payload().downcast_ref::<String>() {
            if msg.starts_with("failed printing to stdout: ") && msg.ends_with("(os error 232)") {
                // the error code is already going to be reported when the panic unwinds up the stack
                let handler = EarlyErrorHandler::new(ErrorOutputType::default());
                let _ = handler.early_error_no_abort(msg.clone());
                return;
            }
        };

        // Invoke the default handler, which prints the actual panic message and optionally a backtrace
        // Don't do this for delayed bugs, which already emit their own more useful backtrace.
        if !info.payload().is::<rustc_errors::DelayedBugPanic>() {
            std::panic_hook_with_disk_dump(info, ice_path().as_deref());

            // Separate the output with an empty line
            eprintln!();
        }

        // Print the ICE message
        report_ice(info, bug_report_url, extra_info);
    }));
}

/// Prints the ICE message, including query stack, but without backtrace.
///
/// The message will point the user at `bug_report_url` to report the ICE.
///
/// When `install_ice_hook` is called, this function will be called as the panic
/// hook.
pub fn report_ice(info: &panic::PanicInfo<'_>, bug_report_url: &str, extra_info: fn(&Handler)) {
    let fallback_bundle =
        rustc_errors::fallback_fluent_bundle(crate::DEFAULT_LOCALE_RESOURCES.to_vec(), false);
    let emitter = Box::new(rustc_errors::emitter::EmitterWriter::stderr(
        rustc_errors::ColorConfig::Auto,
        None,
        None,
        fallback_bundle,
        false,
        false,
        None,
        false,
        false,
        TerminalUrl::No,
    ));
    let handler = rustc_errors::Handler::with_emitter(true, None, emitter, None);

    // a .span_bug or .bug call has already printed what
    // it wants to print.
    if !info.payload().is::<rustc_errors::ExplicitBug>()
        && !info.payload().is::<rustc_errors::DelayedBugPanic>()
    {
        handler.emit_err(session_diagnostics::Ice);
    }

    handler.emit_note(session_diagnostics::IceBugReport { bug_report_url });

    let version = util::version_str!().unwrap_or("unknown_version");
    let triple = config::host_triple();

    static FIRST_PANIC: AtomicBool = AtomicBool::new(true);

    let file = if let Some(path) = ice_path().as_ref() {
        // Create the ICE dump target file.
        match crate::fs::File::options().create(true).append(true).open(&path) {
            Ok(mut file) => {
                handler
                    .emit_note(session_diagnostics::IcePath { path: path.display().to_string() });
                if FIRST_PANIC.swap(false, Ordering::SeqCst) {
                    let _ = write!(file, "\n\nrustc version: {version}\nplatform: {triple}");
                }
                Some(file)
            }
            Err(err) => {
                // The path ICE couldn't be written to disk, provide feedback to the user as to why.
                handler.emit_warning(session_diagnostics::IcePathError {
                    path: path.display().to_string(),
                    error: err.to_string(),
                    env_var: std::env::var("RUSTC_ICE")
                        .ok()
                        .map(|env_var| session_diagnostics::IcePathErrorEnv { env_var }),
                });
                handler.emit_note(session_diagnostics::IceVersion { version, triple });
                None
            }
        }
    } else {
        handler.emit_note(session_diagnostics::IceVersion { version, triple });
        None
    };

    if let Some((flags, excluded_cargo_defaults)) = extra_compiler_flags() {
        handler.emit_note(session_diagnostics::IceFlags { flags: flags.join(" ") });
        if excluded_cargo_defaults {
            handler.emit_note(session_diagnostics::IceExcludeCargoDefaults);
        }
    }

    // If backtraces are enabled, also print the query stack
    let backtrace = env::var_os("RUST_BACKTRACE").is_some_and(|x| &x != "0");

    let num_frames = if backtrace { None } else { Some(2) };

    interface::try_print_query_stack(&handler, num_frames, file);

    // We don't trust this callback not to panic itself, so run it at the end after we're sure we've
    // printed all the relevant info.
    extra_info(&handler);

    #[cfg(windows)]
    if env::var("RUSTC_BREAK_ON_ICE").is_ok() {
        // Trigger a debugger if we crashed during bootstrap
        unsafe { windows::Win32::System::Diagnostics::Debug::DebugBreak() };
    }
}

/// This allows tools to enable rust logging without having to magically match rustc's
/// tracing crate version.
pub fn init_rustc_env_logger(handler: &EarlyErrorHandler) {
    init_env_logger(handler, "RUSTC_LOG");
}

/// This allows tools to enable rust logging without having to magically match rustc's
/// tracing crate version. In contrast to `init_rustc_env_logger` it allows you to choose an env var
/// other than `RUSTC_LOG`.
pub fn init_env_logger(handler: &EarlyErrorHandler, env: &str) {
    if let Err(error) = rustc_log::init_env_logger(env) {
        handler.early_error(error.to_string());
    }
}

#[cfg(all(unix, any(target_env = "gnu", target_os = "macos")))]
mod signal_handler {
    extern "C" {
        fn backtrace_symbols_fd(
            buffer: *const *mut libc::c_void,
            size: libc::c_int,
            fd: libc::c_int,
        );
    }

    extern "C" fn print_stack_trace(_: libc::c_int) {
        const MAX_FRAMES: usize = 256;
        static mut STACK_TRACE: [*mut libc::c_void; MAX_FRAMES] =
            [std::ptr::null_mut(); MAX_FRAMES];
        unsafe {
            let depth = libc::backtrace(STACK_TRACE.as_mut_ptr(), MAX_FRAMES as i32);
            if depth == 0 {
                return;
            }
            backtrace_symbols_fd(STACK_TRACE.as_ptr(), depth, 2);
        }
    }

    /// When an error signal (such as SIGABRT or SIGSEGV) is delivered to the
    /// process, print a stack trace and then exit.
    pub(super) fn install() {
        use std::alloc::{alloc, Layout};

        unsafe {
            let alt_stack_size: usize = min_sigstack_size() + 64 * 1024;
            let mut alt_stack: libc::stack_t = std::mem::zeroed();
            alt_stack.ss_sp = alloc(Layout::from_size_align(alt_stack_size, 1).unwrap()).cast();
            alt_stack.ss_size = alt_stack_size;
            libc::sigaltstack(&alt_stack, std::ptr::null_mut());

            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = print_stack_trace as libc::sighandler_t;
            sa.sa_flags = libc::SA_NODEFER | libc::SA_RESETHAND | libc::SA_ONSTACK;
            libc::sigemptyset(&mut sa.sa_mask);
            libc::sigaction(libc::SIGSEGV, &sa, std::ptr::null_mut());
        }
    }

    /// Modern kernels on modern hardware can have dynamic signal stack sizes.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn min_sigstack_size() -> usize {
        const AT_MINSIGSTKSZ: core::ffi::c_ulong = 51;
        let dynamic_sigstksz = unsafe { libc::getauxval(AT_MINSIGSTKSZ) };
        // If getauxval couldn't find the entry, it returns 0,
        // so take the higher of the "constant" and auxval.
        // This transparently supports older kernels which don't provide AT_MINSIGSTKSZ
        libc::MINSIGSTKSZ.max(dynamic_sigstksz as _)
    }

    /// Not all OS support hardware where this is needed.
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn min_sigstack_size() -> usize {
        libc::MINSIGSTKSZ
    }
}

#[cfg(not(all(unix, any(target_env = "gnu", target_os = "macos"))))]
mod signal_handler {
    pub(super) fn install() {}
}

pub fn main() -> ! {
    let start_time = Instant::now();
    let start_rss = get_resident_set_size();

    let handler = EarlyErrorHandler::new(ErrorOutputType::default());

    init_rustc_env_logger(&handler);
    signal_handler::install();
    let mut callbacks = TimePassesCallbacks::default();
    install_ice_hook(DEFAULT_BUG_REPORT_URL, |_| ());
    let exit_code = catch_with_exit_code(|| {
        let args = env::args_os()
            .enumerate()
            .map(|(i, arg)| {
                arg.into_string().unwrap_or_else(|arg| {
                    handler.early_error(format!("argument {i} is not valid Unicode: {arg:?}"))
                })
            })
            .collect::<Vec<_>>();
        RunCompiler::new(&args, &mut callbacks).run()
    });

    if let Some(format) = callbacks.time_passes {
        let end_rss = get_resident_set_size();
        print_time_passes_entry("total", start_time.elapsed(), start_rss, end_rss, format);
    }

    process::exit(exit_code)
}
