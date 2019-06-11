//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(box_syntax)]
#![cfg_attr(unix, feature(libc))]
#![feature(nll)]
#![feature(rustc_diagnostic_macros)]
#![feature(set_stdio)]
#![feature(no_debug)]
#![feature(integer_atomics)]

#![recursion_limit="256"]

#![deny(rust_2018_idioms)]
#![deny(internal)]
#![deny(unused_lifetimes)]

pub extern crate getopts;
#[cfg(unix)]
extern crate libc;
#[macro_use]
extern crate log;

use pretty::{PpMode, UserIdentifiedItem};

//use rustc_resolve as resolve;
use rustc_save_analysis as save;
use rustc_save_analysis::DumpHandler;
use rustc::session::{config, Session, DiagnosticOutput};
use rustc::session::config::{Input, PrintRequest, ErrorOutputType, OutputType};
use rustc::session::config::nightly_options;
use rustc::session::{early_error, early_warn};
use rustc::lint::Lint;
use rustc::lint;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc::util::common::{time, ErrorReported, install_panic_hook};
use rustc_metadata::locator;
use rustc_metadata::cstore::CStore;
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc_interface::interface;
use rustc_interface::util::get_codegen_sysroot;
use rustc_data_structures::sync::SeqCst;

use serialize::json::ToJson;

use std::borrow::Cow;
use std::cmp::max;
use std::default::Default;
use std::env;
use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::panic::{self, catch_unwind};
use std::path::PathBuf;
use std::process::{self, Command, Stdio};
use std::str;
use std::mem;

use syntax::ast;
use syntax::source_map::FileLoader;
use syntax::feature_gate::{GatedCfg, UnstableFeatures};
use syntax::parse::{self, PResult};
use syntax::symbol::sym;
use syntax_pos::{DUMMY_SP, MultiSpan, FileName};

pub mod pretty;

/// Exit status code used for successful compilation and help output.
pub const EXIT_SUCCESS: i32 = 0;

/// Exit status code used for compilation failures and  invalid flags.
pub const EXIT_FAILURE: i32 = 1;

const BUG_REPORT_URL: &str = "https://github.com/rust-lang/rust/blob/master/CONTRIBUTING.\
                              md#bug-reports";

const ICE_REPORT_COMPILER_FLAGS: &[&str] = &["Z", "C", "crate-type"];

const ICE_REPORT_COMPILER_FLAGS_EXCLUDE: &[&str] = &["metadata", "extra-filename"];

const ICE_REPORT_COMPILER_FLAGS_STRIP_VALUE: &[&str] = &["incremental"];

pub fn source_name(input: &Input) -> FileName {
    match *input {
        Input::File(ref ifile) => ifile.clone().into(),
        Input::Str { ref name, .. } => name.clone(),
    }
}

pub fn abort_on_err<T>(result: Result<T, ErrorReported>, sess: &Session) -> T {
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
    /// Called after parsing and returns true to continue execution
    fn after_parsing(&mut self, _compiler: &interface::Compiler) -> bool {
        true
    }
    /// Called after analysis and returns true to continue execution
    fn after_analysis(&mut self, _compiler: &interface::Compiler) -> bool {
        true
    }
}

pub struct DefaultCallbacks;

impl Callbacks for DefaultCallbacks {}

// Parse args and run the compiler. This is the primary entry point for rustc.
// See comments on CompilerCalls below for details about the callbacks argument.
// The FileLoader provides a way to load files from sources other than the file system.
pub fn run_compiler(
    args: &[String],
    callbacks: &mut (dyn Callbacks + Send),
    file_loader: Option<Box<dyn FileLoader + Send + Sync>>,
    emitter: Option<Box<dyn Write + Send>>
) -> interface::Result<()> {
    let diagnostic_output = emitter.map(|emitter| DiagnosticOutput::Raw(emitter))
                                   .unwrap_or(DiagnosticOutput::Default);
    let matches = match handle_options(args) {
        Some(matches) => matches,
        None => return Ok(()),
    };

    install_panic_hook();

    let (sopts, cfg) = config::build_session_options_and_crate_config(&matches);

    let mut dummy_config = |sopts, cfg, diagnostic_output| {
        let mut config = interface::Config {
            opts: sopts,
            crate_cfg: cfg,
            input: Input::File(PathBuf::new()),
            input_path: None,
            output_file: None,
            output_dir: None,
            file_loader: None,
            diagnostic_output,
            stderr: None,
            crate_name: None,
            lint_caps: Default::default(),
        };
        callbacks.config(&mut config);
        config
    };

    if let Some(ref code) = matches.opt_str("explain") {
        handle_explain(code, sopts.error_format);
        return Ok(());
    }

    let (odir, ofile) = make_output(&matches);
    let (input, input_file_path, input_err) = match make_input(&matches.free) {
        Some(v) => v,
        None => {
            match matches.free.len() {
                0 => {
                    let config = dummy_config(sopts, cfg, diagnostic_output);
                    interface::run_compiler(config, |compiler| {
                        let sopts = &compiler.session().opts;
                        if sopts.describe_lints {
                            describe_lints(
                                compiler.session(),
                                &*compiler.session().lint_store.borrow(),
                                false
                            );
                            return;
                        }
                        let should_stop = RustcDefaultCalls::print_crate_info(
                            &***compiler.codegen_backend(),
                            compiler.session(),
                            None,
                            &odir,
                            &ofile
                        );

                        if should_stop == Compilation::Stop {
                            return;
                        }
                        early_error(sopts.error_format, "no input filename given")
                    });
                    return Ok(());
                }
                1 => panic!("make_input should have provided valid inputs"),
                _ => early_error(sopts.error_format, &format!(
                    "multiple input filenames provided (first two filenames are `{}` and `{}`)",
                    matches.free[0],
                    matches.free[1],
                )),
            }
        }
    };

    if let Some(err) = input_err {
        // Immediately stop compilation if there was an issue reading
        // the input (for example if the input stream is not UTF-8).
        interface::run_compiler(dummy_config(sopts, cfg, diagnostic_output), |compiler| {
            compiler.session().err(&err.to_string());
        });
        return Err(ErrorReported);
    }

    let mut config = interface::Config {
        opts: sopts,
        crate_cfg: cfg,
        input,
        input_path: input_file_path,
        output_file: ofile,
        output_dir: odir,
        file_loader,
        diagnostic_output,
        stderr: None,
        crate_name: None,
        lint_caps: Default::default(),
    };

    callbacks.config(&mut config);

    interface::run_compiler(config, |compiler| {
        let sess = compiler.session();
        let should_stop = RustcDefaultCalls::print_crate_info(
            &***compiler.codegen_backend(),
            sess,
            Some(compiler.input()),
            compiler.output_dir(),
            compiler.output_file(),
        ).and_then(|| RustcDefaultCalls::list_metadata(
            sess,
            compiler.cstore(),
            &matches,
            compiler.input()
        ));

        if should_stop == Compilation::Stop {
            return sess.compile_status();
        }

        let pretty_info = parse_pretty(sess, &matches);

        compiler.parse()?;

        if let Some((ppm, opt_uii)) = pretty_info {
            if ppm.needs_ast_map(&opt_uii) {
                pretty::visit_crate(sess, &mut compiler.parse()?.peek_mut(), ppm);
                compiler.global_ctxt()?.peek_mut().enter(|tcx| {
                    let expanded_crate = compiler.expansion()?.take().0;
                    pretty::print_after_hir_lowering(
                        tcx,
                        compiler.input(),
                        &expanded_crate,
                        ppm,
                        opt_uii.clone(),
                        compiler.output_file().as_ref().map(|p| &**p),
                    );
                    Ok(())
                })?;
                return sess.compile_status();
            } else {
                let mut krate = compiler.parse()?.take();
                pretty::visit_crate(sess, &mut krate, ppm);
                pretty::print_after_parsing(
                    sess,
                    &compiler.input(),
                    &krate,
                    ppm,
                    compiler.output_file().as_ref().map(|p| &**p),
                );
                return sess.compile_status();
            }
        }

        if !callbacks.after_parsing(compiler) {
            return sess.compile_status();
        }

        if sess.opts.debugging_opts.parse_only ||
           sess.opts.debugging_opts.show_span.is_some() ||
           sess.opts.debugging_opts.ast_json_noexpand {
            return sess.compile_status();
        }

        compiler.register_plugins()?;

        // Lint plugins are registered; now we can process command line flags.
        if sess.opts.describe_lints {
            describe_lints(&sess, &sess.lint_store.borrow(), true);
            return sess.compile_status();
        }

        compiler.prepare_outputs()?;

        if sess.opts.output_types.contains_key(&OutputType::DepInfo)
            && sess.opts.output_types.len() == 1
        {
            return sess.compile_status();
        }

        compiler.global_ctxt()?;

        if sess.opts.debugging_opts.no_analysis ||
           sess.opts.debugging_opts.ast_json {
            return sess.compile_status();
        }

        if sess.opts.debugging_opts.save_analysis {
            let expanded_crate = &compiler.expansion()?.peek().0;
            let crate_name = compiler.crate_name()?.peek().clone();
            compiler.global_ctxt()?.peek_mut().enter(|tcx| {
                let result = tcx.analysis(LOCAL_CRATE);

                time(sess, "save analysis", || {
                    save::process_crate(
                        tcx,
                        &expanded_crate,
                        &crate_name,
                        &compiler.input(),
                        None,
                        DumpHandler::new(compiler.output_dir().as_ref().map(|p| &**p), &crate_name)
                    )
                });

                result
                // AST will be dropped *after* the `after_analysis` callback
                // (needed by the RLS)
            })?;
        } else {
            // Drop AST after creating GlobalCtxt to free memory
            mem::drop(compiler.expansion()?.take());
        }

        compiler.global_ctxt()?.peek_mut().enter(|tcx| tcx.analysis(LOCAL_CRATE))?;

        if !callbacks.after_analysis(compiler) {
            return sess.compile_status();
        }

        if sess.opts.debugging_opts.save_analysis {
            mem::drop(compiler.expansion()?.take());
        }

        compiler.ongoing_codegen()?;

        // Drop GlobalCtxt after starting codegen to free memory
        mem::drop(compiler.global_ctxt()?.take());

        if sess.opts.debugging_opts.print_type_sizes {
            sess.code_stats.borrow().print_type_sizes();
        }

        compiler.link()?;

        if sess.opts.debugging_opts.perf_stats {
            sess.print_perf_stats();
        }

        if sess.print_fuel_crate.is_some() {
            eprintln!("Fuel used by {}: {}",
                sess.print_fuel_crate.as_ref().unwrap(),
                sess.print_fuel.load(SeqCst));
        }

        Ok(())
    })
}

#[cfg(unix)]
pub fn set_sigpipe_handler() {
    unsafe {
        // Set the SIGPIPE signal handler, so that an EPIPE
        // will cause rustc to terminate, as expected.
        assert_ne!(libc::signal(libc::SIGPIPE, libc::SIG_DFL), libc::SIG_ERR);
    }
}

#[cfg(windows)]
pub fn set_sigpipe_handler() {}

// Extract output directory and file from matches.
fn make_output(matches: &getopts::Matches) -> (Option<PathBuf>, Option<PathBuf>) {
    let odir = matches.opt_str("out-dir").map(|o| PathBuf::from(&o));
    let ofile = matches.opt_str("o").map(|o| PathBuf::from(&o));
    (odir, ofile)
}

// Extract input (string or file and optional path) from matches.
fn make_input(free_matches: &[String]) -> Option<(Input, Option<PathBuf>, Option<io::Error>)> {
    if free_matches.len() == 1 {
        let ifile = &free_matches[0];
        if ifile == "-" {
            let mut src = String::new();
            let err = if io::stdin().read_to_string(&mut src).is_err() {
                Some(io::Error::new(io::ErrorKind::InvalidData,
                                    "couldn't read from stdin, as it did not contain valid UTF-8"))
            } else {
                None
            };
            Some((Input::Str { name: FileName::anon_source_code(&src), input: src },
                  None, err))
        } else {
            Some((Input::File(PathBuf::from(ifile)),
                  Some(PathBuf::from(ifile)), None))
        }
    } else {
        None
    }
}

fn parse_pretty(sess: &Session,
                matches: &getopts::Matches)
                -> Option<(PpMode, Option<UserIdentifiedItem>)> {
    let pretty = if sess.opts.debugging_opts.unstable_options {
        matches.opt_default("pretty", "normal").map(|a| {
            // stable pretty-print variants only
            pretty::parse_pretty(sess, &a, false)
        })
    } else {
        None
    };

    if pretty.is_none() {
        sess.opts.debugging_opts.unpretty.as_ref().map(|a| {
            // extended with unstable pretty-print variants
            pretty::parse_pretty(sess, &a, true)
        })
    } else {
        pretty
    }
}

// Whether to stop or continue compilation.
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

/// CompilerCalls instance for a regular rustc build.
#[derive(Copy, Clone)]
pub struct RustcDefaultCalls;

// FIXME remove these and use winapi 0.3 instead
// Duplicates: bootstrap/compile.rs, librustc_errors/emitter.rs
#[cfg(unix)]
fn stdout_isatty() -> bool {
    unsafe { libc::isatty(libc::STDOUT_FILENO) != 0 }
}

#[cfg(windows)]
fn stdout_isatty() -> bool {
    type DWORD = u32;
    type BOOL = i32;
    type HANDLE = *mut u8;
    type LPDWORD = *mut u32;
    const STD_OUTPUT_HANDLE: DWORD = -11i32 as DWORD;
    extern "system" {
        fn GetStdHandle(which: DWORD) -> HANDLE;
        fn GetConsoleMode(hConsoleHandle: HANDLE, lpMode: LPDWORD) -> BOOL;
    }
    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut out = 0;
        GetConsoleMode(handle, &mut out) != 0
    }
}

fn handle_explain(code: &str,
                  output: ErrorOutputType) {
    let descriptions = rustc_interface::util::diagnostics_registry();
    let normalised = if code.starts_with("E") {
        code.to_string()
    } else {
        format!("E{0:0>4}", code)
    };
    match descriptions.find_description(&normalised) {
        Some(ref description) => {
            let mut is_in_code_block = false;
            let mut text = String::new();

            // Slice off the leading newline and print.
            for line in description[1..].lines() {
                let indent_level = line.find(|c: char| !c.is_whitespace())
                    .unwrap_or_else(|| line.len());
                let dedented_line = &line[indent_level..];
                if dedented_line.starts_with("```") {
                    is_in_code_block = !is_in_code_block;
                    text.push_str(&line[..(indent_level+3)]);
                } else if is_in_code_block && dedented_line.starts_with("# ") {
                    continue;
                } else {
                    text.push_str(line);
                }
                text.push('\n');
            }

            if stdout_isatty() {
                show_content_with_pager(&text);
            } else {
                print!("{}", text);
            }
        }
        None => {
            early_error(output, &format!("no extended information for {}", code));
        }
    }
}

fn show_content_with_pager(content: &String) {
    let pager_name = env::var_os("PAGER").unwrap_or_else(|| if cfg!(windows) {
        OsString::from("more.com")
    } else {
        OsString::from("less")
    });

    let mut fallback_to_println = false;

    match Command::new(pager_name).stdin(Stdio::piped()).spawn() {
        Ok(mut pager) => {
            if let Some(pipe) = pager.stdin.as_mut() {
                if pipe.write_all(content.as_bytes()).is_err() {
                    fallback_to_println = true;
                }
            }

            if pager.wait().is_err() {
                fallback_to_println = true;
            }
        }
        Err(_) => {
            fallback_to_println = true;
        }
    }

    // If pager fails for whatever reason, we should still print the content
    // to standard output
    if fallback_to_println {
        print!("{}", content);
    }
}

impl RustcDefaultCalls {
    pub fn list_metadata(sess: &Session,
                         cstore: &CStore,
                         matches: &getopts::Matches,
                         input: &Input)
                         -> Compilation {
        let r = matches.opt_strs("Z");
        if r.iter().any(|s| *s == "ls") {
            match input {
                &Input::File(ref ifile) => {
                    let path = &(*ifile);
                    let mut v = Vec::new();
                    locator::list_file_metadata(&sess.target.target,
                                                path,
                                                &*cstore.metadata_loader,
                                                &mut v)
                            .unwrap();
                    println!("{}", String::from_utf8(v).unwrap());
                }
                &Input::Str { .. } => {
                    early_error(ErrorOutputType::default(), "cannot list metadata for stdin");
                }
            }
            return Compilation::Stop;
        }

        Compilation::Continue
    }


    fn print_crate_info(codegen_backend: &dyn CodegenBackend,
                        sess: &Session,
                        input: Option<&Input>,
                        odir: &Option<PathBuf>,
                        ofile: &Option<PathBuf>)
                        -> Compilation {
        use rustc::session::config::PrintRequest::*;
        // PrintRequest::NativeStaticLibs is special - printed during linking
        // (empty iterator returns true)
        if sess.opts.prints.iter().all(|&p| p == PrintRequest::NativeStaticLibs) {
            return Compilation::Continue;
        }

        let attrs = match input {
            None => None,
            Some(input) => {
                let result = parse_crate_attrs(sess, input);
                match result {
                    Ok(attrs) => Some(attrs),
                    Err(mut parse_error) => {
                        parse_error.emit();
                        return Compilation::Stop;
                    }
                }
            }
        };
        for req in &sess.opts.prints {
            match *req {
                TargetList => {
                    let mut targets = rustc_target::spec::get_targets().collect::<Vec<String>>();
                    targets.sort();
                    println!("{}", targets.join("\n"));
                },
                Sysroot => println!("{}", sess.sysroot.display()),
                TargetSpec => println!("{}", sess.target.target.to_json().pretty()),
                FileNames | CrateName => {
                    let input = input.unwrap_or_else(||
                        early_error(ErrorOutputType::default(), "no input file provided"));
                    let attrs = attrs.as_ref().unwrap();
                    let t_outputs = rustc_interface::util::build_output_filenames(
                        input,
                        odir,
                        ofile,
                        attrs,
                        sess
                    );
                    let id = rustc_codegen_utils::link::find_crate_name(Some(sess), attrs, input);
                    if *req == PrintRequest::CrateName {
                        println!("{}", id);
                        continue;
                    }
                    let crate_types = rustc_interface::util::collect_crate_types(sess, attrs);
                    for &style in &crate_types {
                        let fname = rustc_codegen_utils::link::filename_for_input(
                            sess,
                            style,
                            &id,
                            &t_outputs
                        );
                        println!("{}", fname.file_name().unwrap().to_string_lossy());
                    }
                }
                Cfg => {
                    let allow_unstable_cfg = UnstableFeatures::from_environment()
                        .is_nightly_build();

                    let mut cfgs = sess.parse_sess.config.iter().filter_map(|&(name, ref value)| {
                        let gated_cfg = GatedCfg::gate(&ast::MetaItem {
                            path: ast::Path::from_ident(ast::Ident::with_empty_ctxt(name)),
                            node: ast::MetaItemKind::Word,
                            span: DUMMY_SP,
                        });

                        // Note that crt-static is a specially recognized cfg
                        // directive that's printed out here as part of
                        // rust-lang/rust#37406, but in general the
                        // `target_feature` cfg is gated under
                        // rust-lang/rust#29717. For now this is just
                        // specifically allowing the crt-static cfg and that's
                        // it, this is intended to get into Cargo and then go
                        // through to build scripts.
                        let value = value.as_ref().map(|s| s.as_str());
                        let value = value.as_ref().map(|s| s.as_ref());
                        if name != sym::target_feature || value != Some("crt-static") {
                            if !allow_unstable_cfg && gated_cfg.is_some() {
                                return None
                            }
                        }

                        if let Some(value) = value {
                            Some(format!("{}=\"{}\"", name, value))
                        } else {
                            Some(name.to_string())
                        }
                    }).collect::<Vec<String>>();

                    cfgs.sort();
                    for cfg in cfgs {
                        println!("{}", cfg);
                    }
                }
                RelocationModels | CodeModels | TlsModels | TargetCPUs | TargetFeatures => {
                    codegen_backend.print(*req, sess);
                }
                // Any output here interferes with Cargo's parsing of other printed output
                PrintRequest::NativeStaticLibs => {}
            }
        }
        return Compilation::Stop;
    }
}

/// Returns a version string such as "0.12.0-dev".
fn release_str() -> Option<&'static str> {
    option_env!("CFG_RELEASE")
}

/// Returns the full SHA1 hash of HEAD of the Git repo from which rustc was built.
fn commit_hash_str() -> Option<&'static str> {
    option_env!("CFG_VER_HASH")
}

/// Returns the "commit date" of HEAD of the Git repo from which rustc was built as a static string.
fn commit_date_str() -> Option<&'static str> {
    option_env!("CFG_VER_DATE")
}

/// Prints version information
pub fn version(binary: &str, matches: &getopts::Matches) {
    let verbose = matches.opt_present("verbose");

    println!("{} {}", binary, option_env!("CFG_VERSION").unwrap_or("unknown version"));

    if verbose {
        fn unw(x: Option<&str>) -> &str {
            x.unwrap_or("unknown")
        }
        println!("binary: {}", binary);
        println!("commit-hash: {}", unw(commit_hash_str()));
        println!("commit-date: {}", unw(commit_date_str()));
        println!("host: {}", config::host_triple());
        println!("release: {}", unw(release_str()));
        get_codegen_sysroot("llvm")().print_version();
    }
}

fn usage(verbose: bool, include_unstable_options: bool) {
    let groups = if verbose {
        config::rustc_optgroups()
    } else {
        config::rustc_short_optgroups()
    };
    let mut options = getopts::Options::new();
    for option in groups.iter().filter(|x| include_unstable_options || x.is_stable()) {
        (option.apply)(&mut options);
    }
    let message = "Usage: rustc [OPTIONS] INPUT";
    let nightly_help = if nightly_options::is_nightly_build() {
        "\n    -Z help             Print unstable compiler options"
    } else {
        ""
    };
    let verbose_help = if verbose {
        ""
    } else {
        "\n    --help -v           Print the full set of options rustc accepts"
    };
    println!("{}\nAdditional help:
    -C help             Print codegen options
    -W help             \
              Print 'lint' options and default settings{}{}\n",
             options.usage(message),
             nightly_help,
             verbose_help);
}

fn print_wall_help() {
    println!("
The flag `-Wall` does not exist in `rustc`. Most useful lints are enabled by
default. Use `rustc -W help` to see all available lints. It's more common to put
warning settings in the crate root using `#![warn(LINT_NAME)]` instead of using
the command line flag directly.
");
}

fn describe_lints(sess: &Session, lint_store: &lint::LintStore, loaded_plugins: bool) {
    println!("
Available lint options:
    -W <foo>           Warn about <foo>
    -A <foo>           \
              Allow <foo>
    -D <foo>           Deny <foo>
    -F <foo>           Forbid <foo> \
              (deny <foo> and all attempts to override)

");

    fn sort_lints(sess: &Session, lints: Vec<(&'static Lint, bool)>) -> Vec<&'static Lint> {
        let mut lints: Vec<_> = lints.into_iter().map(|(x, _)| x).collect();
        // The sort doesn't case-fold but it's doubtful we care.
        lints.sort_by_cached_key(|x: &&Lint| (x.default_level(sess), x.name));
        lints
    }

    fn sort_lint_groups(lints: Vec<(&'static str, Vec<lint::LintId>, bool)>)
                        -> Vec<(&'static str, Vec<lint::LintId>)> {
        let mut lints: Vec<_> = lints.into_iter().map(|(x, y, _)| (x, y)).collect();
        lints.sort_by_key(|l| l.0);
        lints
    }

    let (plugin, builtin): (Vec<_>, _) = lint_store.get_lints()
                                                   .iter()
                                                   .cloned()
                                                   .partition(|&(_, p)| p);
    let plugin = sort_lints(sess, plugin);
    let builtin = sort_lints(sess, builtin);

    let (plugin_groups, builtin_groups): (Vec<_>, _) = lint_store.get_lint_groups()
                                                                 .iter()
                                                                 .cloned()
                                                                 .partition(|&(.., p)| p);
    let plugin_groups = sort_lint_groups(plugin_groups);
    let builtin_groups = sort_lint_groups(builtin_groups);

    let max_name_len = plugin.iter()
                             .chain(&builtin)
                             .map(|&s| s.name.chars().count())
                             .max()
                             .unwrap_or(0);
    let padded = |x: &str| {
        let mut s = " ".repeat(max_name_len - x.chars().count());
        s.push_str(x);
        s
    };

    println!("Lint checks provided by rustc:\n");
    println!("    {}  {:7.7}  {}", padded("name"), "default", "meaning");
    println!("    {}  {:7.7}  {}", padded("----"), "-------", "-------");

    let print_lints = |lints: Vec<&Lint>| {
        for lint in lints {
            let name = lint.name_lower().replace("_", "-");
            println!("    {}  {:7.7}  {}",
                     padded(&name),
                     lint.default_level.as_str(),
                     lint.desc);
        }
        println!("\n");
    };

    print_lints(builtin);

    let max_name_len = max("warnings".len(),
                           plugin_groups.iter()
                                        .chain(&builtin_groups)
                                        .map(|&(s, _)| s.chars().count())
                                        .max()
                                        .unwrap_or(0));

    let padded = |x: &str| {
        let mut s = " ".repeat(max_name_len - x.chars().count());
        s.push_str(x);
        s
    };

    println!("Lint groups provided by rustc:\n");
    println!("    {}  {}", padded("name"), "sub-lints");
    println!("    {}  {}", padded("----"), "---------");
    println!("    {}  {}", padded("warnings"), "all lints that are set to issue warnings");

    let print_lint_groups = |lints: Vec<(&'static str, Vec<lint::LintId>)>| {
        for (name, to) in lints {
            let name = name.to_lowercase().replace("_", "-");
            let desc = to.into_iter()
                         .map(|x| x.to_string().replace("_", "-"))
                         .collect::<Vec<String>>()
                         .join(", ");
            println!("    {}  {}", padded(&name), desc);
        }
        println!("\n");
    };

    print_lint_groups(builtin_groups);

    match (loaded_plugins, plugin.len(), plugin_groups.len()) {
        (false, 0, _) | (false, _, 0) => {
            println!("Compiler plugins can provide additional lints and lint groups. To see a \
                      listing of these, re-run `rustc -W help` with a crate filename.");
        }
        (false, ..) => panic!("didn't load lint plugins but got them anyway!"),
        (true, 0, 0) => println!("This crate does not load any lint plugins or lint groups."),
        (true, l, g) => {
            if l > 0 {
                println!("Lint checks provided by plugins loaded by this crate:\n");
                print_lints(plugin);
            }
            if g > 0 {
                println!("Lint groups provided by plugins loaded by this crate:\n");
                print_lint_groups(plugin_groups);
            }
        }
    }
}

fn describe_debug_flags() {
    println!("\nAvailable options:\n");
    print_flag_list("-Z", config::DB_OPTIONS);
}

fn describe_codegen_flags() {
    println!("\nAvailable codegen options:\n");
    print_flag_list("-C", config::CG_OPTIONS);
}

fn print_flag_list<T>(cmdline_opt: &str,
                      flag_list: &[(&'static str, T, Option<&'static str>, &'static str)]) {
    let max_len = flag_list.iter()
                           .map(|&(name, _, opt_type_desc, _)| {
                               let extra_len = match opt_type_desc {
                                   Some(..) => 4,
                                   None => 0,
                               };
                               name.chars().count() + extra_len
                           })
                           .max()
                           .unwrap_or(0);

    for &(name, _, opt_type_desc, desc) in flag_list {
        let (width, extra) = match opt_type_desc {
            Some(..) => (max_len - 4, "=val"),
            None => (max_len, ""),
        };
        println!("    {} {:>width$}{} -- {}",
                 cmdline_opt,
                 name.replace("_", "-"),
                 extra,
                 desc,
                 width = width);
    }
}

/// Process command line options. Emits messages as appropriate. If compilation
/// should continue, returns a getopts::Matches object parsed from args,
/// otherwise returns `None`.
///
/// The compiler's handling of options is a little complicated as it ties into
/// our stability story, and it's even *more* complicated by historical
/// accidents. The current intention of each compiler option is to have one of
/// three modes:
///
/// 1. An option is stable and can be used everywhere.
/// 2. An option is unstable, but was historically allowed on the stable
///    channel.
/// 3. An option is unstable, and can only be used on nightly.
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
pub fn handle_options(args: &[String]) -> Option<getopts::Matches> {
    // Throw away the first argument, the name of the binary
    let args = &args[1..];

    if args.is_empty() {
        // user did not write `-v` nor `-Z unstable-options`, so do not
        // include that extra information.
        usage(false, false);
        return None;
    }

    // Parse with *all* options defined in the compiler, we don't worry about
    // option stability here we just want to parse as much as possible.
    let mut options = getopts::Options::new();
    for option in config::rustc_optgroups() {
        (option.apply)(&mut options);
    }
    let matches = options.parse(args).unwrap_or_else(|f|
        early_error(ErrorOutputType::default(), &f.to_string()));

    // For all options we just parsed, we check a few aspects:
    //
    // * If the option is stable, we're all good
    // * If the option wasn't passed, we're all good
    // * If `-Z unstable-options` wasn't passed (and we're not a -Z option
    //   ourselves), then we require the `-Z unstable-options` flag to unlock
    //   this option that was passed.
    // * If we're a nightly compiler, then unstable options are now unlocked, so
    //   we're good to go.
    // * Otherwise, if we're a truly unstable option then we generate an error
    //   (unstable option being used on stable)
    // * If we're a historically stable-but-should-be-unstable option then we
    //   emit a warning that we're going to turn this into an error soon.
    nightly_options::check_nightly_options(&matches, &config::rustc_optgroups());

    if matches.opt_present("h") || matches.opt_present("help") {
        // Only show unstable options in --help if we *really* accept unstable
        // options, which catches the case where we got `-Z unstable-options` on
        // the stable channel of Rust which was accidentally allowed
        // historically.
        usage(matches.opt_present("verbose"),
              nightly_options::is_unstable_enabled(&matches));
        return None;
    }

    // Handle the special case of -Wall.
    let wall = matches.opt_strs("W");
    if wall.iter().any(|x| *x == "all") {
        print_wall_help();
        return None;
    }

    // Don't handle -W help here, because we might first load plugins.
    let r = matches.opt_strs("Z");
    if r.iter().any(|x| *x == "help") {
        describe_debug_flags();
        return None;
    }

    let cg_flags = matches.opt_strs("C");

    if cg_flags.iter().any(|x| *x == "help") {
        describe_codegen_flags();
        return None;
    }

    if cg_flags.iter().any(|x| *x == "no-stack-check") {
        early_warn(ErrorOutputType::default(),
                   "the --no-stack-check flag is deprecated and does nothing");
    }

    if cg_flags.iter().any(|x| *x == "passes=list") {
        get_codegen_sysroot("llvm")().print_passes();
        return None;
    }

    if matches.opt_present("version") {
        version("rustc", &matches);
        return None;
    }

    Some(matches)
}

fn parse_crate_attrs<'a>(sess: &'a Session, input: &Input) -> PResult<'a, Vec<ast::Attribute>> {
    match *input {
        Input::File(ref ifile) => {
            parse::parse_crate_attrs_from_file(ifile, &sess.parse_sess)
        }
        Input::Str { ref name, ref input } => {
            parse::parse_crate_attrs_from_source_str(name.clone(),
                                                     input.clone(),
                                                     &sess.parse_sess)
        }
    }
}

/// Gets a list of extra command-line flags provided by the user, as strings.
///
/// This function is used during ICEs to show more information useful for
/// debugging, since some ICEs only happens with non-default compiler flags
/// (and the users don't always report them).
fn extra_compiler_flags() -> Option<(Vec<String>, bool)> {
    let args = env::args_os().map(|arg| arg.to_string_lossy().to_string()).collect::<Vec<_>>();

    // Avoid printing help because of empty args. This can suggest the compiler
    // itself is not the program root (consider RLS).
    if args.len() < 2 {
        return None;
    }

    let matches = if let Some(matches) = handle_options(&args) {
        matches
    } else {
        return None;
    };

    let mut result = Vec::new();
    let mut excluded_cargo_defaults = false;
    for flag in ICE_REPORT_COMPILER_FLAGS {
        let prefix = if flag.len() == 1 { "-" } else { "--" };

        for content in &matches.opt_strs(flag) {
            // Split always returns the first element
            let name = if let Some(first) = content.split('=').next() {
                first
            } else {
                &content
            };

            let content = if ICE_REPORT_COMPILER_FLAGS_STRIP_VALUE.contains(&name) {
                name
            } else {
                content
            };

            if !ICE_REPORT_COMPILER_FLAGS_EXCLUDE.contains(&name) {
                result.push(format!("{}{} {}", prefix, flag, content));
            } else {
                excluded_cargo_defaults = true;
            }
        }
    }

    if !result.is_empty() {
        Some((result, excluded_cargo_defaults))
    } else {
        None
    }
}

/// Runs a procedure which will detect panics in the compiler and print nicer
/// error messages rather than just failing the test.
///
/// The diagnostic emitter yielded to the procedure should be used for reporting
/// errors of the compiler.
pub fn report_ices_to_stderr_if_any<F: FnOnce() -> R, R>(f: F) -> Result<R, ErrorReported> {
    catch_unwind(panic::AssertUnwindSafe(f)).map_err(|value| {
        if value.is::<errors::FatalErrorMarker>() {
            ErrorReported
        } else {
            // Thread panicked without emitting a fatal diagnostic
            eprintln!("");

            let emitter =
                Box::new(errors::emitter::EmitterWriter::stderr(errors::ColorConfig::Auto,
                                                                None,
                                                                false,
                                                                false));
            let handler = errors::Handler::with_emitter(true, None, emitter);

            // a .span_bug or .bug call has already printed what
            // it wants to print.
            if !value.is::<errors::ExplicitBug>() {
                handler.emit(&MultiSpan::new(),
                             "unexpected panic",
                             errors::Level::Bug);
            }

            let mut xs: Vec<Cow<'static, str>> = vec![
                "the compiler unexpectedly panicked. this is a bug.".into(),
                format!("we would appreciate a bug report: {}", BUG_REPORT_URL).into(),
                format!("rustc {} running on {}",
                        option_env!("CFG_VERSION").unwrap_or("unknown_version"),
                        config::host_triple()).into(),
            ];

            if let Some((flags, excluded_cargo_defaults)) = extra_compiler_flags() {
                xs.push(format!("compiler flags: {}", flags.join(" ")).into());

                if excluded_cargo_defaults {
                    xs.push("some of the compiler flags provided by cargo are hidden".into());
                }
            }

            for note in &xs {
                handler.emit(&MultiSpan::new(),
                             note,
                             errors::Level::Note);
            }

            panic::resume_unwind(Box::new(errors::FatalErrorMarker));
        }
    })
}

/// This allows tools to enable rust logging without having to magically match rustc's
/// log crate version
pub fn init_rustc_env_logger() {
    env_logger::init_from_env("RUSTC_LOG");
}

pub fn main() {
    init_rustc_env_logger();
    let result = report_ices_to_stderr_if_any(|| {
        let args = env::args_os().enumerate()
            .map(|(i, arg)| arg.into_string().unwrap_or_else(|arg| {
                early_error(ErrorOutputType::default(),
                            &format!("Argument {} is not valid Unicode: {:?}", i, arg))
            }))
            .collect::<Vec<_>>();
        run_compiler(&args, &mut DefaultCallbacks, None, None)
    }).and_then(|result| result);
    process::exit(match result {
        Ok(_) => EXIT_SUCCESS,
        Err(_) => EXIT_FAILURE,
    });
}
