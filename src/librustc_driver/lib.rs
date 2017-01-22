// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![crate_name = "rustc_driver"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(box_syntax)]
#![feature(libc)]
#![feature(quote)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(set_stdio)]
#![feature(staged_api)]

extern crate arena;
extern crate getopts;
extern crate graphviz;
extern crate libc;
extern crate rustc;
extern crate rustc_back;
extern crate rustc_borrowck;
extern crate rustc_const_eval;
extern crate rustc_data_structures;
extern crate rustc_errors as errors;
extern crate rustc_passes;
extern crate rustc_lint;
extern crate rustc_plugin;
extern crate rustc_privacy;
extern crate rustc_incremental;
extern crate rustc_metadata;
extern crate rustc_mir;
extern crate rustc_resolve;
extern crate rustc_save_analysis;
extern crate rustc_trans;
extern crate rustc_typeck;
extern crate serialize;
extern crate rustc_llvm as llvm;
#[macro_use]
extern crate log;
extern crate syntax;
extern crate syntax_ext;
extern crate syntax_pos;

use driver::CompileController;
use pretty::{PpMode, UserIdentifiedItem};

use rustc_resolve as resolve;
use rustc_save_analysis as save;
use rustc_trans::back::link;
use rustc_trans::back::write::{create_target_machine, RELOC_MODEL_ARGS, CODE_GEN_MODEL_ARGS};
use rustc::dep_graph::DepGraph;
use rustc::session::{self, config, Session, build_session, CompileResult};
use rustc::session::config::{Input, PrintRequest, OutputType, ErrorOutputType};
use rustc::session::config::nightly_options;
use rustc::session::{early_error, early_warn};
use rustc::lint::Lint;
use rustc::lint;
use rustc_metadata::locator;
use rustc_metadata::cstore::CStore;
use rustc::util::common::time;

use serialize::json::ToJson;

use std::any::Any;
use std::cmp::max;
use std::cmp::Ordering::Equal;
use std::default::Default;
use std::env;
use std::io::{self, Read, Write};
use std::iter::repeat;
use std::path::PathBuf;
use std::process;
use std::rc::Rc;
use std::str;
use std::sync::{Arc, Mutex};
use std::thread;

use syntax::ast;
use syntax::codemap::{CodeMap, FileLoader, RealFileLoader};
use syntax::feature_gate::{GatedCfg, UnstableFeatures};
use syntax::parse::{self, PResult};
use syntax_pos::{DUMMY_SP, MultiSpan};

#[cfg(test)]
pub mod test;

pub mod driver;
pub mod pretty;
pub mod target_features;
mod derive_registrar;

const BUG_REPORT_URL: &'static str = "https://github.com/rust-lang/rust/blob/master/CONTRIBUTING.\
                                      md#bug-reports";

#[inline]
fn abort_msg(err_count: usize) -> String {
    match err_count {
        0 => "aborting with no errors (maybe a bug?)".to_owned(),
        1 => "aborting due to previous error".to_owned(),
        e => format!("aborting due to {} previous errors", e),
    }
}

pub fn abort_on_err<T>(result: Result<T, usize>, sess: &Session) -> T {
    match result {
        Err(err_count) => {
            sess.fatal(&abort_msg(err_count));
        }
        Ok(x) => x,
    }
}

pub fn run<F>(run_compiler: F) -> isize
    where F: FnOnce() -> (CompileResult, Option<Session>) + Send + 'static
{
    monitor(move || {
        let (result, session) = run_compiler();
        if let Err(err_count) = result {
            if err_count > 0 {
                match session {
                    Some(sess) => sess.fatal(&abort_msg(err_count)),
                    None => {
                        let emitter =
                            errors::emitter::EmitterWriter::stderr(errors::ColorConfig::Auto, None);
                        let handler = errors::Handler::with_emitter(true, false, Box::new(emitter));
                        handler.emit(&MultiSpan::new(),
                                     &abort_msg(err_count),
                                     errors::Level::Fatal);
                        exit_on_err();
                    }
                }
            }
        }
    });
    0
}

// Parse args and run the compiler. This is the primary entry point for rustc.
// See comments on CompilerCalls below for details about the callbacks argument.
// The FileLoader provides a way to load files from sources other than the file system.
pub fn run_compiler<'a>(args: &[String],
                        callbacks: &mut CompilerCalls<'a>,
                        file_loader: Option<Box<FileLoader + 'static>>,
                        emitter_dest: Option<Box<Write + Send>>)
                        -> (CompileResult, Option<Session>)
{
    macro_rules! do_or_return {($expr: expr, $sess: expr) => {
        match $expr {
            Compilation::Stop => return (Ok(()), $sess),
            Compilation::Continue => {}
        }
    }}

    let matches = match handle_options(args) {
        Some(matches) => matches,
        None => return (Ok(()), None),
    };

    let (sopts, cfg) = config::build_session_options_and_crate_config(&matches);

    if sopts.debugging_opts.debug_llvm {
        unsafe { llvm::LLVMRustSetDebug(1); }
    }

    let descriptions = diagnostics_registry();

    do_or_return!(callbacks.early_callback(&matches,
                                           &sopts,
                                           &cfg,
                                           &descriptions,
                                           sopts.error_format),
                                           None);

    let (odir, ofile) = make_output(&matches);
    let (input, input_file_path) = match make_input(&matches.free) {
        Some((input, input_file_path)) => callbacks.some_input(input, input_file_path),
        None => match callbacks.no_input(&matches, &sopts, &cfg, &odir, &ofile, &descriptions) {
            Some((input, input_file_path)) => (input, input_file_path),
            None => return (Ok(()), None),
        },
    };

    let dep_graph = DepGraph::new(sopts.build_dep_graph());
    let cstore = Rc::new(CStore::new(&dep_graph));

    let loader = file_loader.unwrap_or(box RealFileLoader);
    let codemap = Rc::new(CodeMap::with_file_loader(loader));
    let mut sess = session::build_session_with_codemap(
        sopts, &dep_graph, input_file_path, descriptions, cstore.clone(), codemap, emitter_dest,
    );
    rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));

    let mut cfg = config::build_configuration(&sess, cfg);
    target_features::add_configuration(&mut cfg, &sess);
    sess.parse_sess.config = cfg;

    do_or_return!(callbacks.late_callback(&matches, &sess, &input, &odir, &ofile), Some(sess));

    let plugins = sess.opts.debugging_opts.extra_plugins.clone();
    let control = callbacks.build_controller(&sess, &matches);
    (driver::compile_input(&sess, &cstore, &input, &odir, &ofile, Some(plugins), &control),
     Some(sess))
}

// Extract output directory and file from matches.
fn make_output(matches: &getopts::Matches) -> (Option<PathBuf>, Option<PathBuf>) {
    let odir = matches.opt_str("out-dir").map(|o| PathBuf::from(&o));
    let ofile = matches.opt_str("o").map(|o| PathBuf::from(&o));
    (odir, ofile)
}

// Extract input (string or file and optional path) from matches.
fn make_input(free_matches: &[String]) -> Option<(Input, Option<PathBuf>)> {
    if free_matches.len() == 1 {
        let ifile = &free_matches[0][..];
        if ifile == "-" {
            let mut src = String::new();
            io::stdin().read_to_string(&mut src).unwrap();
            Some((Input::Str { name: driver::anon_src(), input: src },
                  None))
        } else {
            Some((Input::File(PathBuf::from(ifile)),
                  Some(PathBuf::from(ifile))))
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
    if pretty.is_none() && sess.unstable_options() {
        matches.opt_str("unpretty").map(|a| {
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

// A trait for customising the compilation process. Offers a number of hooks for
// executing custom code or customising input.
pub trait CompilerCalls<'a> {
    // Hook for a callback early in the process of handling arguments. This will
    // be called straight after options have been parsed but before anything
    // else (e.g., selecting input and output).
    fn early_callback(&mut self,
                      _: &getopts::Matches,
                      _: &config::Options,
                      _: &ast::CrateConfig,
                      _: &errors::registry::Registry,
                      _: ErrorOutputType)
                      -> Compilation {
        Compilation::Continue
    }

    // Hook for a callback late in the process of handling arguments. This will
    // be called just before actual compilation starts (and before build_controller
    // is called), after all arguments etc. have been completely handled.
    fn late_callback(&mut self,
                     _: &getopts::Matches,
                     _: &Session,
                     _: &Input,
                     _: &Option<PathBuf>,
                     _: &Option<PathBuf>)
                     -> Compilation {
        Compilation::Continue
    }

    // Called after we extract the input from the arguments. Gives the implementer
    // an opportunity to change the inputs or to add some custom input handling.
    // The default behaviour is to simply pass through the inputs.
    fn some_input(&mut self,
                  input: Input,
                  input_path: Option<PathBuf>)
                  -> (Input, Option<PathBuf>) {
        (input, input_path)
    }

    // Called after we extract the input from the arguments if there is no valid
    // input. Gives the implementer an opportunity to supply alternate input (by
    // returning a Some value) or to add custom behaviour for this error such as
    // emitting error messages. Returning None will cause compilation to stop
    // at this point.
    fn no_input(&mut self,
                _: &getopts::Matches,
                _: &config::Options,
                _: &ast::CrateConfig,
                _: &Option<PathBuf>,
                _: &Option<PathBuf>,
                _: &errors::registry::Registry)
                -> Option<(Input, Option<PathBuf>)> {
        None
    }

    // Create a CompilController struct for controlling the behaviour of
    // compilation.
    fn build_controller(&mut self, &Session, &getopts::Matches) -> CompileController<'a>;
}

// CompilerCalls instance for a regular rustc build.
#[derive(Copy, Clone)]
pub struct RustcDefaultCalls;

fn handle_explain(code: &str,
                  descriptions: &errors::registry::Registry,
                  output: ErrorOutputType) {
    let normalised = if code.starts_with("E") {
        code.to_string()
    } else {
        format!("E{0:0>4}", code)
    };
    match descriptions.find_description(&normalised) {
        Some(ref description) => {
            // Slice off the leading newline and print.
            print!("{}", &(&description[1..]).split("\n").map(|x| {
                format!("{}\n", if x.starts_with("```") {
                    "```"
                } else {
                    x
                })
            }).collect::<String>());
        }
        None => {
            early_error(output, &format!("no extended information for {}", code));
        }
    }
}

impl<'a> CompilerCalls<'a> for RustcDefaultCalls {
    fn early_callback(&mut self,
                      matches: &getopts::Matches,
                      _: &config::Options,
                      _: &ast::CrateConfig,
                      descriptions: &errors::registry::Registry,
                      output: ErrorOutputType)
                      -> Compilation {
        if let Some(ref code) = matches.opt_str("explain") {
            handle_explain(code, descriptions, output);
            return Compilation::Stop;
        }

        Compilation::Continue
    }

    fn no_input(&mut self,
                matches: &getopts::Matches,
                sopts: &config::Options,
                cfg: &ast::CrateConfig,
                odir: &Option<PathBuf>,
                ofile: &Option<PathBuf>,
                descriptions: &errors::registry::Registry)
                -> Option<(Input, Option<PathBuf>)> {
        match matches.free.len() {
            0 => {
                if sopts.describe_lints {
                    let mut ls = lint::LintStore::new();
                    rustc_lint::register_builtins(&mut ls, None);
                    describe_lints(&ls, false);
                    return None;
                }
                let dep_graph = DepGraph::new(sopts.build_dep_graph());
                let cstore = Rc::new(CStore::new(&dep_graph));
                let mut sess = build_session(sopts.clone(),
                    &dep_graph,
                    None,
                    descriptions.clone(),
                    cstore.clone());
                rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));
                let mut cfg = config::build_configuration(&sess, cfg.clone());
                target_features::add_configuration(&mut cfg, &sess);
                sess.parse_sess.config = cfg;
                let should_stop =
                    RustcDefaultCalls::print_crate_info(&sess, None, odir, ofile);

                if should_stop == Compilation::Stop {
                    return None;
                }
                early_error(sopts.error_format, "no input filename given");
            }
            1 => panic!("make_input should have provided valid inputs"),
            _ => early_error(sopts.error_format, "multiple input filenames provided"),
        }
    }

    fn late_callback(&mut self,
                     matches: &getopts::Matches,
                     sess: &Session,
                     input: &Input,
                     odir: &Option<PathBuf>,
                     ofile: &Option<PathBuf>)
                     -> Compilation {
        RustcDefaultCalls::print_crate_info(sess, Some(input), odir, ofile)
            .and_then(|| RustcDefaultCalls::list_metadata(sess, matches, input))
    }

    fn build_controller(&mut self,
                        sess: &Session,
                        matches: &getopts::Matches)
                        -> CompileController<'a> {
        let mut control = CompileController::basic();

        if let Some((ppm, opt_uii)) = parse_pretty(sess, matches) {
            if ppm.needs_ast_map(&opt_uii) {
                control.after_hir_lowering.stop = Compilation::Stop;

                control.after_parse.callback = box move |state| {
                    state.krate = Some(pretty::fold_crate(state.krate.take().unwrap(), ppm));
                };
                control.after_hir_lowering.callback = box move |state| {
                    pretty::print_after_hir_lowering(state.session,
                                                     state.ast_map.unwrap(),
                                                     state.analysis.unwrap(),
                                                     state.resolutions.unwrap(),
                                                     state.input,
                                                     &state.expanded_crate.take().unwrap(),
                                                     state.crate_name.unwrap(),
                                                     ppm,
                                                     state.arena.unwrap(),
                                                     state.arenas.unwrap(),
                                                     opt_uii.clone(),
                                                     state.out_file);
                };
            } else {
                control.after_parse.stop = Compilation::Stop;

                control.after_parse.callback = box move |state| {
                    let krate = pretty::fold_crate(state.krate.take().unwrap(), ppm);
                    pretty::print_after_parsing(state.session,
                                                state.input,
                                                &krate,
                                                ppm,
                                                state.out_file);
                };
            }

            return control;
        }

        if sess.opts.debugging_opts.parse_only ||
           sess.opts.debugging_opts.show_span.is_some() ||
           sess.opts.debugging_opts.ast_json_noexpand {
            control.after_parse.stop = Compilation::Stop;
        }

        if sess.opts.debugging_opts.no_analysis ||
           sess.opts.debugging_opts.ast_json {
            control.after_hir_lowering.stop = Compilation::Stop;
        }

        if !sess.opts.output_types.keys().any(|&i| i == OutputType::Exe ||
                                                   i == OutputType::Metadata) {
            control.after_llvm.stop = Compilation::Stop;
        }

        if save_analysis(sess) {
            control.after_analysis.callback = box |state| {
                time(state.session.time_passes(), "save analysis", || {
                    save::process_crate(state.tcx.unwrap(),
                                        state.expanded_crate.unwrap(),
                                        state.analysis.unwrap(),
                                        state.crate_name.unwrap(),
                                        state.out_dir,
                                        save_analysis_format(state.session))
                });
            };
            control.after_analysis.run_callback_on_error = true;
            control.make_glob_map = resolve::MakeGlobMap::Yes;
        }

        control
    }
}

fn save_analysis(sess: &Session) -> bool {
    sess.opts.debugging_opts.save_analysis ||
    sess.opts.debugging_opts.save_analysis_csv ||
    sess.opts.debugging_opts.save_analysis_api
}

fn save_analysis_format(sess: &Session) -> save::Format {
    if sess.opts.debugging_opts.save_analysis {
        save::Format::Json
    } else if sess.opts.debugging_opts.save_analysis_csv {
        save::Format::Csv
    } else if sess.opts.debugging_opts.save_analysis_api {
        save::Format::JsonApi
    } else {
        unreachable!();
    }
}

impl RustcDefaultCalls {
    pub fn list_metadata(sess: &Session, matches: &getopts::Matches, input: &Input) -> Compilation {
        let r = matches.opt_strs("Z");
        if r.contains(&("ls".to_string())) {
            match input {
                &Input::File(ref ifile) => {
                    let path = &(*ifile);
                    let mut v = Vec::new();
                    locator::list_file_metadata(&sess.target.target, path, &mut v).unwrap();
                    println!("{}", String::from_utf8(v).unwrap());
                }
                &Input::Str { .. } => {
                    early_error(ErrorOutputType::default(), "cannot list metadata for stdin");
                }
            }
            return Compilation::Stop;
        }

        return Compilation::Continue;
    }


    fn print_crate_info(sess: &Session,
                        input: Option<&Input>,
                        odir: &Option<PathBuf>,
                        ofile: &Option<PathBuf>)
                        -> Compilation {
        if sess.opts.prints.is_empty() {
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
                PrintRequest::TargetList => {
                    let mut targets = rustc_back::target::get_targets().collect::<Vec<String>>();
                    targets.sort();
                    println!("{}", targets.join("\n"));
                },
                PrintRequest::Sysroot => println!("{}", sess.sysroot().display()),
                PrintRequest::TargetSpec => println!("{}", sess.target.target.to_json().pretty()),
                PrintRequest::FileNames |
                PrintRequest::CrateName => {
                    let input = match input {
                        Some(input) => input,
                        None => early_error(ErrorOutputType::default(), "no input file provided"),
                    };
                    let attrs = attrs.as_ref().unwrap();
                    let t_outputs = driver::build_output_filenames(input, odir, ofile, attrs, sess);
                    let id = link::find_crate_name(Some(sess), attrs, input);
                    if *req == PrintRequest::CrateName {
                        println!("{}", id);
                        continue;
                    }
                    let crate_types = driver::collect_crate_types(sess, attrs);
                    for &style in &crate_types {
                        let fname = link::filename_for_input(sess, style, &id, &t_outputs);
                        println!("{}",
                                 fname.file_name()
                                      .unwrap()
                                      .to_string_lossy());
                    }
                }
                PrintRequest::Cfg => {
                    let allow_unstable_cfg = UnstableFeatures::from_environment()
                        .is_nightly_build();

                    let mut cfgs = Vec::new();
                    for &(name, ref value) in sess.parse_sess.config.iter() {
                        let gated_cfg = GatedCfg::gate(&ast::MetaItem {
                            name: name,
                            node: ast::MetaItemKind::Word,
                            span: DUMMY_SP,
                        });
                        if !allow_unstable_cfg && gated_cfg.is_some() {
                            continue;
                        }

                        cfgs.push(if let &Some(ref value) = value {
                            format!("{}=\"{}\"", name, value)
                        } else {
                            format!("{}", name)
                        });
                    }

                    cfgs.sort();
                    for cfg in cfgs {
                        println!("{}", cfg);
                    }
                }
                PrintRequest::TargetCPUs => {
                    let tm = create_target_machine(sess);
                    unsafe { llvm::LLVMRustPrintTargetCPUs(tm); }
                }
                PrintRequest::TargetFeatures => {
                    let tm = create_target_machine(sess);
                    unsafe { llvm::LLVMRustPrintTargetFeatures(tm); }
                }
                PrintRequest::RelocationModels => {
                    println!("Available relocation models:");
                    for &(name, _) in RELOC_MODEL_ARGS.iter() {
                        println!("    {}", name);
                    }
                    println!("");
                }
                PrintRequest::CodeModels => {
                    println!("Available code models:");
                    for &(name, _) in CODE_GEN_MODEL_ARGS.iter(){
                        println!("    {}", name);
                    }
                    println!("");
                }
            }
        }
        return Compilation::Stop;
    }
}

/// Returns a version string such as "0.12.0-dev".
pub fn release_str() -> Option<&'static str> {
    option_env!("CFG_RELEASE")
}

/// Returns the full SHA1 hash of HEAD of the Git repo from which rustc was built.
pub fn commit_hash_str() -> Option<&'static str> {
    option_env!("CFG_VER_HASH")
}

/// Returns the "commit date" of HEAD of the Git repo from which rustc was built as a static string.
pub fn commit_date_str() -> Option<&'static str> {
    option_env!("CFG_VER_DATE")
}

/// Prints version information
pub fn version(binary: &str, matches: &getopts::Matches) {
    let verbose = matches.opt_present("verbose");

    println!("{} {}",
             binary,
             option_env!("CFG_VERSION").unwrap_or("unknown version"));
    if verbose {
        fn unw(x: Option<&str>) -> &str {
            x.unwrap_or("unknown")
        }
        println!("binary: {}", binary);
        println!("commit-hash: {}", unw(commit_hash_str()));
        println!("commit-date: {}", unw(commit_date_str()));
        println!("host: {}", config::host_triple());
        println!("release: {}", unw(release_str()));
        unsafe {
            println!("LLVM version: {}.{}",
                     llvm::LLVMRustVersionMajor(), llvm::LLVMRustVersionMinor());
        }
    }
}

fn usage(verbose: bool, include_unstable_options: bool) {
    let groups = if verbose {
        config::rustc_optgroups()
    } else {
        config::rustc_short_optgroups()
    };
    let groups: Vec<_> = groups.into_iter()
                               .filter(|x| include_unstable_options || x.is_stable())
                               .map(|x| x.opt_group)
                               .collect();
    let message = format!("Usage: rustc [OPTIONS] INPUT");
    let extra_help = if verbose {
        ""
    } else {
        "\n    --help -v           Print the full set of options rustc accepts"
    };
    println!("{}\nAdditional help:
    -C help             Print codegen options
    -W help             \
              Print 'lint' options and default settings
    -Z help             Print internal \
              options for debugging rustc{}\n",
             getopts::usage(&message, &groups),
             extra_help);
}

fn describe_lints(lint_store: &lint::LintStore, loaded_plugins: bool) {
    println!("
Available lint options:
    -W <foo>           Warn about <foo>
    -A <foo>           \
              Allow <foo>
    -D <foo>           Deny <foo>
    -F <foo>           Forbid <foo> \
              (deny, and deny all overrides)

");

    fn sort_lints(lints: Vec<(&'static Lint, bool)>) -> Vec<&'static Lint> {
        let mut lints: Vec<_> = lints.into_iter().map(|(x, _)| x).collect();
        lints.sort_by(|x: &&Lint, y: &&Lint| {
            match x.default_level.cmp(&y.default_level) {
                // The sort doesn't case-fold but it's doubtful we care.
                Equal => x.name.cmp(y.name),
                r => r,
            }
        });
        lints
    }

    fn sort_lint_groups(lints: Vec<(&'static str, Vec<lint::LintId>, bool)>)
                        -> Vec<(&'static str, Vec<lint::LintId>)> {
        let mut lints: Vec<_> = lints.into_iter().map(|(x, y, _)| (x, y)).collect();
        lints.sort_by(|&(x, _): &(&'static str, Vec<lint::LintId>),
                       &(y, _): &(&'static str, Vec<lint::LintId>)| {
            x.cmp(y)
        });
        lints
    }

    let (plugin, builtin): (Vec<_>, _) = lint_store.get_lints()
                                                   .iter()
                                                   .cloned()
                                                   .partition(|&(_, p)| p);
    let plugin = sort_lints(plugin);
    let builtin = sort_lints(builtin);

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
        let mut s = repeat(" ")
                        .take(max_name_len - x.chars().count())
                        .collect::<String>();
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
                     padded(&name[..]),
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
        let mut s = repeat(" ")
                        .take(max_name_len - x.chars().count())
                        .collect::<String>();
        s.push_str(x);
        s
    };

    println!("Lint groups provided by rustc:\n");
    println!("    {}  {}", padded("name"), "sub-lints");
    println!("    {}  {}", padded("----"), "---------");
    println!("    {}  {}", padded("warnings"), "all built-in lints");

    let print_lint_groups = |lints: Vec<(&'static str, Vec<lint::LintId>)>| {
        for (name, to) in lints {
            let name = name.to_lowercase().replace("_", "-");
            let desc = to.into_iter()
                         .map(|x| x.to_string().replace("_", "-"))
                         .collect::<Vec<String>>()
                         .join(", ");
            println!("    {}  {}", padded(&name[..]), desc);
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
    println!("\nAvailable debug options:\n");
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
/// otherwise returns None.
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
    let all_groups: Vec<getopts::OptGroup> = config::rustc_optgroups()
                                                 .into_iter()
                                                 .map(|x| x.opt_group)
                                                 .collect();
    let matches = match getopts::getopts(&args[..], &all_groups) {
        Ok(m) => m,
        Err(f) => early_error(ErrorOutputType::default(), &f.to_string()),
    };

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

    if cg_flags.contains(&"passes=list".to_string()) {
        unsafe {
            ::llvm::LLVMRustPrintPasses();
        }
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
            parse::parse_crate_attrs_from_source_str(name.clone(), input.clone(), &sess.parse_sess)
        }
    }
}

/// Runs `f` in a suitable thread for running `rustc`; returns a
/// `Result` with either the return value of `f` or -- if a panic
/// occurs -- the panic value.
pub fn in_rustc_thread<F, R>(f: F) -> Result<R, Box<Any + Send>>
    where F: FnOnce() -> R + Send + 'static,
          R: Send + 'static,
{
    // Temporarily have stack size set to 16MB to deal with nom-using crates failing
    const STACK_SIZE: usize = 16 * 1024 * 1024; // 16MB

    let mut cfg = thread::Builder::new().name("rustc".to_string());

    // FIXME: Hacks on hacks. If the env is trying to override the stack size
    // then *don't* set it explicitly.
    if env::var_os("RUST_MIN_STACK").is_none() {
        cfg = cfg.stack_size(STACK_SIZE);
    }

    let thread = cfg.spawn(f);
    thread.unwrap().join()
}

/// Run a procedure which will detect panics in the compiler and print nicer
/// error messages rather than just failing the test.
///
/// The diagnostic emitter yielded to the procedure should be used for reporting
/// errors of the compiler.
pub fn monitor<F: FnOnce() + Send + 'static>(f: F) {
    struct Sink(Arc<Mutex<Vec<u8>>>);
    impl Write for Sink {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            Write::write(&mut *self.0.lock().unwrap(), data)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let data = Arc::new(Mutex::new(Vec::new()));
    let err = Sink(data.clone());

    let result = in_rustc_thread(move || {
        io::set_panic(Some(box err));
        f()
    });

    if let Err(value) = result {
        // Thread panicked without emitting a fatal diagnostic
        if !value.is::<errors::FatalError>() {
            let emitter =
                Box::new(errors::emitter::EmitterWriter::stderr(errors::ColorConfig::Auto, None));
            let handler = errors::Handler::with_emitter(true, false, emitter);

            // a .span_bug or .bug call has already printed what
            // it wants to print.
            if !value.is::<errors::ExplicitBug>() {
                handler.emit(&MultiSpan::new(),
                             "unexpected panic",
                             errors::Level::Bug);
            }

            let xs = ["the compiler unexpectedly panicked. this is a bug.".to_string(),
                      format!("we would appreciate a bug report: {}", BUG_REPORT_URL)];
            for note in &xs {
                handler.emit(&MultiSpan::new(),
                             &note[..],
                             errors::Level::Note);
            }
            if match env::var_os("RUST_BACKTRACE") {
                Some(val) => &val != "0",
                None => false,
            } {
                handler.emit(&MultiSpan::new(),
                             "run with `RUST_BACKTRACE=1` for a backtrace",
                             errors::Level::Note);
            }

            writeln!(io::stderr(), "{}", str::from_utf8(&data.lock().unwrap()).unwrap()).unwrap();
        }

        exit_on_err();
    }
}

fn exit_on_err() -> ! {
    // Panic so the process returns a failure code, but don't pollute the
    // output with some unnecessary panic messages, we've already
    // printed everything that we needed to.
    io::set_panic(Some(box io::sink()));
    panic!();
}

pub fn diagnostics_registry() -> errors::registry::Registry {
    use errors::registry::Registry;

    let mut all_errors = Vec::new();
    all_errors.extend_from_slice(&rustc::DIAGNOSTICS);
    all_errors.extend_from_slice(&rustc_typeck::DIAGNOSTICS);
    all_errors.extend_from_slice(&rustc_borrowck::DIAGNOSTICS);
    all_errors.extend_from_slice(&rustc_resolve::DIAGNOSTICS);
    all_errors.extend_from_slice(&rustc_privacy::DIAGNOSTICS);
    all_errors.extend_from_slice(&rustc_trans::DIAGNOSTICS);
    all_errors.extend_from_slice(&rustc_const_eval::DIAGNOSTICS);
    all_errors.extend_from_slice(&rustc_metadata::DIAGNOSTICS);

    Registry::new(&all_errors)
}

pub fn main() {
    let result = run(|| run_compiler(&env::args().collect::<Vec<_>>(),
                                     &mut RustcDefaultCalls,
                                     None,
                                     None));
    process::exit(result as i32);
}
