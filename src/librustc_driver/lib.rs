// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
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
#![experimental]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://doc.rust-lang.org/nightly/")]

#![feature(default_type_params, globs, macro_rules, phase, quote)]
#![feature(slicing_syntax, unsafe_destructor)]
#![feature(rustc_diagnostic_macros)]
#![feature(unboxed_closures)]

extern crate arena;
extern crate flate;
extern crate getopts;
extern crate graphviz;
extern crate libc;
extern crate rustc;
extern crate rustc_back;
extern crate rustc_borrowck;
extern crate rustc_resolve;
extern crate rustc_trans;
extern crate rustc_typeck;
#[phase(plugin, link)] extern crate log;
#[phase(plugin, link)] extern crate syntax;
extern crate serialize;
extern crate "rustc_llvm" as llvm;

pub use syntax::diagnostic;

use rustc_trans::back::link;
use rustc::session::{config, Session, build_session};
use rustc::session::config::{Input, PrintRequest};
use rustc::lint::Lint;
use rustc::lint;
use rustc::metadata;
use rustc::DIAGNOSTICS;

use std::any::AnyRefExt;
use std::io;
use std::iter::repeat;
use std::os;
use std::thread;

use rustc::session::early_error;

use syntax::ast;
use syntax::parse;
use syntax::diagnostic::Emitter;
use syntax::diagnostics;

#[cfg(test)]
pub mod test;

pub mod driver;
pub mod pretty;

pub fn run(args: Vec<String>) -> int {
    monitor(move |:| run_compiler(args.as_slice()));
    0
}

static BUG_REPORT_URL: &'static str =
    "http://doc.rust-lang.org/complement-bugreport.html";

fn run_compiler(args: &[String]) {
    let matches = match handle_options(args.to_vec()) {
        Some(matches) => matches,
        None => return
    };

    let descriptions = diagnostics::registry::Registry::new(&DIAGNOSTICS);
    match matches.opt_str("explain") {
        Some(ref code) => {
            match descriptions.find_description(code[]) {
                Some(ref description) => {
                    println!("{}", description);
                }
                None => {
                    early_error(format!("no extended information for {}", code)[]);
                }
            }
            return;
        },
        None => ()
    }

    let sopts = config::build_session_options(&matches);
    let odir = matches.opt_str("out-dir").map(|o| Path::new(o));
    let ofile = matches.opt_str("o").map(|o| Path::new(o));
    let (input, input_file_path) = match matches.free.len() {
        0u => {
            if sopts.describe_lints {
                let mut ls = lint::LintStore::new();
                ls.register_builtin(None);
                describe_lints(&ls, false);
                return;
            }
            let sess = build_session(sopts, None, descriptions);
            if print_crate_info(&sess, None, &odir, &ofile) {
                return;
            }
            early_error("no input filename given");
        }
        1u => {
            let ifile = matches.free[0][];
            if ifile == "-" {
                let contents = io::stdin().read_to_end().unwrap();
                let src = String::from_utf8(contents).unwrap();
                (Input::Str(src), None)
            } else {
                (Input::File(Path::new(ifile)), Some(Path::new(ifile)))
            }
        }
        _ => early_error("multiple input filenames provided")
    };

    let sess = build_session(sopts, input_file_path, descriptions);
    let cfg = config::build_configuration(&sess);
    if print_crate_info(&sess, Some(&input), &odir, &ofile) {
        return
    }

    let pretty = matches.opt_default("pretty", "normal").map(|a| {
        // stable pretty-print variants only
        pretty::parse_pretty(&sess, a.as_slice(), false)
    });
    let pretty = if pretty.is_none() &&
        sess.debugging_opt(config::UNSTABLE_OPTIONS) {
            matches.opt_str("xpretty").map(|a| {
                // extended with unstable pretty-print variants
                pretty::parse_pretty(&sess, a.as_slice(), true)
            })
        } else {
            pretty
        };

    match pretty.into_iter().next() {
        Some((ppm, opt_uii)) => {
            pretty::pretty_print_input(sess, cfg, &input, ppm, opt_uii, ofile);
            return;
        }
        None => {/* continue */ }
    }

    let r = matches.opt_strs("Z");
    if r.contains(&("ls".to_string())) {
        match input {
            Input::File(ref ifile) => {
                let mut stdout = io::stdout();
                list_metadata(&sess, &(*ifile), &mut stdout).unwrap();
            }
            Input::Str(_) => {
                early_error("can not list metadata for stdin");
            }
        }
        return;
    }

    driver::compile_input(sess, cfg, &input, &odir, &ofile, None);
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

/// Prints version information and returns None on success or an error
/// message on panic.
pub fn version(binary: &str, matches: &getopts::Matches) {
    let verbose = matches.opt_present("verbose");

    println!("{} {}", binary, option_env!("CFG_VERSION").unwrap_or("unknown version"));
    if verbose {
        fn unw(x: Option<&str>) -> &str { x.unwrap_or("unknown") }
        println!("binary: {}", binary);
        println!("commit-hash: {}", unw(commit_hash_str()));
        println!("commit-date: {}", unw(commit_date_str()));
        println!("host: {}", config::host_triple());
        println!("release: {}", unw(release_str()));
    }
}

fn usage(verbose: bool, include_unstable_options: bool) {
    let groups = if verbose {
        config::rustc_optgroups()
    } else {
        config::rustc_short_optgroups()
    };
    let groups : Vec<_> = groups.into_iter()
        .filter(|x| include_unstable_options || x.is_stable())
        .map(|x|x.opt_group)
        .collect();
    let message = format!("Usage: rustc [OPTIONS] INPUT");
    let extra_help = if verbose {
        ""
    } else {
        "\n    --help -v           Print the full set of options rustc accepts"
    };
    println!("{}\n\
Additional help:
    -C help             Print codegen options
    -W help             Print 'lint' options and default settings
    -Z help             Print internal options for debugging rustc{}\n",
              getopts::usage(message.as_slice(), groups.as_slice()),
              extra_help);
}

fn describe_lints(lint_store: &lint::LintStore, loaded_plugins: bool) {
    println!("
Available lint options:
    -W <foo>           Warn about <foo>
    -A <foo>           Allow <foo>
    -D <foo>           Deny <foo>
    -F <foo>           Forbid <foo> (deny, and deny all overrides)

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

    let (plugin, builtin) = lint_store.get_lints().partitioned(|&(_, p)| p);
    let plugin = sort_lints(plugin);
    let builtin = sort_lints(builtin);

    let (plugin_groups, builtin_groups) = lint_store.get_lint_groups().partitioned(|&(_, _, p)| p);
    let plugin_groups = sort_lint_groups(plugin_groups);
    let builtin_groups = sort_lint_groups(builtin_groups);

    let max_name_len = plugin.iter().chain(builtin.iter())
        .map(|&s| s.name.width(true))
        .max().unwrap_or(0);
    let padded = |x: &str| {
        let mut s = repeat(" ").take(max_name_len - x.chars().count())
                               .collect::<String>();
        s.push_str(x);
        s
    };

    println!("Lint checks provided by rustc:\n");
    println!("    {}  {:7.7}  {}", padded("name"), "default", "meaning");
    println!("    {}  {:7.7}  {}", padded("----"), "-------", "-------");

    let print_lints = |lints: Vec<&Lint>| {
        for lint in lints.into_iter() {
            let name = lint.name_lower().replace("_", "-");
            println!("    {}  {:7.7}  {}",
                     padded(name[]), lint.default_level.as_str(), lint.desc);
        }
        println!("\n");
    };

    print_lints(builtin);



    let max_name_len = plugin_groups.iter().chain(builtin_groups.iter())
        .map(|&(s, _)| s.width(true))
        .max().unwrap_or(0);
    let padded = |x: &str| {
        let mut s = repeat(" ").take(max_name_len - x.chars().count())
                               .collect::<String>();
        s.push_str(x);
        s
    };

    println!("Lint groups provided by rustc:\n");
    println!("    {}  {}", padded("name"), "sub-lints");
    println!("    {}  {}", padded("----"), "---------");

    let print_lint_groups = |lints: Vec<(&'static str, Vec<lint::LintId>)>| {
        for (name, to) in lints.into_iter() {
            let name = name.chars().map(|x| x.to_lowercase())
                           .collect::<String>().replace("_", "-");
            let desc = to.into_iter().map(|x| x.as_str().replace("_", "-"))
                         .collect::<Vec<String>>().connect(", ");
            println!("    {}  {}",
                     padded(name[]), desc);
        }
        println!("\n");
    };

    print_lint_groups(builtin_groups);

    match (loaded_plugins, plugin.len(), plugin_groups.len()) {
        (false, 0, _) | (false, _, 0) => {
            println!("Compiler plugins can provide additional lints and lint groups. To see a \
                      listing of these, re-run `rustc -W help` with a crate filename.");
        }
        (false, _, _) => panic!("didn't load lint plugins but got them anyway!"),
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
    let r = config::debugging_opts_map();
    for tuple in r.iter() {
        match *tuple {
            (ref name, ref desc, _) => {
                println!("    -Z {:>20} -- {}", *name, *desc);
            }
        }
    }
}

fn describe_codegen_flags() {
    println!("\nAvailable codegen options:\n");
    for &(name, _, opt_type_desc, desc) in config::CG_OPTIONS.iter() {
        let (width, extra) = match opt_type_desc {
            Some(..) => (21, "=val"),
            None => (25, "")
        };
        println!("    -C {:>width$}{} -- {}", name.replace("_", "-"),
                 extra, desc, width=width);
    }
}

/// Process command line options. Emits messages as appropriate. If compilation
/// should continue, returns a getopts::Matches object parsed from args, otherwise
/// returns None.
pub fn handle_options(mut args: Vec<String>) -> Option<getopts::Matches> {
    // Throw away the first argument, the name of the binary
    let _binary = args.remove(0).unwrap();

    if args.is_empty() {
        // user did not write `-v` nor `-Z unstable-options`, so do not
        // include that extra information.
        usage(false, false);
        return None;
    }

    let matches =
        match getopts::getopts(args[], config::optgroups()[]) {
            Ok(m) => m,
            Err(f_stable_attempt) => {
                // redo option parsing, including unstable options this time,
                // in anticipation that the mishandled option was one of the
                // unstable ones.
                let all_groups : Vec<getopts::OptGroup>
                    = config::rustc_optgroups().into_iter().map(|x|x.opt_group).collect();
                match getopts::getopts(args.as_slice(), all_groups.as_slice()) {
                    Ok(m_unstable) => {
                        let r = m_unstable.opt_strs("Z");
                        let include_unstable_options = r.iter().any(|x| *x == "unstable-options");
                        if include_unstable_options {
                            m_unstable
                        } else {
                            early_error(f_stable_attempt.to_string().as_slice());
                        }
                    }
                    Err(_) => {
                        // ignore the error from the unstable attempt; just
                        // pass the error we got from the first try.
                        early_error(f_stable_attempt.to_string().as_slice());
                    }
                }
            }
        };

    let r = matches.opt_strs("Z");
    let include_unstable_options = r.iter().any(|x| *x == "unstable-options");

    if matches.opt_present("h") || matches.opt_present("help") {
        usage(matches.opt_present("verbose"), include_unstable_options);
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

    if cg_flags.contains(&"passes=list".to_string()) {
        unsafe { ::llvm::LLVMRustPrintPasses(); }
        return None;
    }

    if matches.opt_present("version") {
        version("rustc", &matches);
        return None;
    }

    Some(matches)
}

fn print_crate_info(sess: &Session,
                    input: Option<&Input>,
                    odir: &Option<Path>,
                    ofile: &Option<Path>)
                    -> bool {
    if sess.opts.prints.len() == 0 { return false }

    let attrs = input.map(|input| parse_crate_attrs(sess, input));
    for req in sess.opts.prints.iter() {
        match *req {
            PrintRequest::Sysroot => println!("{}", sess.sysroot().display()),
            PrintRequest::FileNames |
            PrintRequest::CrateName => {
                let input = match input {
                    Some(input) => input,
                    None => early_error("no input file provided"),
                };
                let attrs = attrs.as_ref().unwrap().as_slice();
                let t_outputs = driver::build_output_filenames(input,
                                                               odir,
                                                               ofile,
                                                               attrs,
                                                               sess);
                let id = link::find_crate_name(Some(sess), attrs.as_slice(),
                                               input);
                if *req == PrintRequest::CrateName {
                    println!("{}", id);
                    continue
                }
                let crate_types = driver::collect_crate_types(sess, attrs);
                let metadata = driver::collect_crate_metadata(sess, attrs);
                *sess.crate_metadata.borrow_mut() = metadata;
                for &style in crate_types.iter() {
                    let fname = link::filename_for_input(sess, style,
                                                         id.as_slice(),
                                                         &t_outputs.with_extension(""));
                    println!("{}", fname.filename_display());
                }
            }
        }
    }
    return true;
}

fn parse_crate_attrs(sess: &Session, input: &Input) ->
                     Vec<ast::Attribute> {
    let result = match *input {
        Input::File(ref ifile) => {
            parse::parse_crate_attrs_from_file(ifile,
                                               Vec::new(),
                                               &sess.parse_sess)
        }
        Input::Str(ref src) => {
            parse::parse_crate_attrs_from_source_str(
                driver::anon_src().to_string(),
                src.to_string(),
                Vec::new(),
                &sess.parse_sess)
        }
    };
    result.into_iter().collect()
}

pub fn list_metadata(sess: &Session, path: &Path,
                     out: &mut io::Writer) -> io::IoResult<()> {
    metadata::loader::list_file_metadata(sess.target.target.options.is_like_osx, path, out)
}

/// Run a procedure which will detect panics in the compiler and print nicer
/// error messages rather than just failing the test.
///
/// The diagnostic emitter yielded to the procedure should be used for reporting
/// errors of the compiler.
pub fn monitor<F:FnOnce()+Send>(f: F) {
    static STACK_SIZE: uint = 8 * 1024 * 1024; // 8MB

    let (tx, rx) = channel();
    let w = io::ChanWriter::new(tx);
    let mut r = io::ChanReader::new(rx);

    let mut cfg = thread::Builder::new().name("rustc".to_string());

    // FIXME: Hacks on hacks. If the env is trying to override the stack size
    // then *don't* set it explicitly.
    if os::getenv("RUST_MIN_STACK").is_none() {
        cfg = cfg.stack_size(STACK_SIZE);
    }

    match cfg.spawn(move || { std::io::stdio::set_stderr(box w); f() }).join() {
        Ok(()) => { /* fallthrough */ }
        Err(value) => {
            // Task panicked without emitting a fatal diagnostic
            if !value.is::<diagnostic::FatalError>() {
                let mut emitter = diagnostic::EmitterWriter::stderr(diagnostic::Auto, None);

                // a .span_bug or .bug call has already printed what
                // it wants to print.
                if !value.is::<diagnostic::ExplicitBug>() {
                    emitter.emit(
                        None,
                        "unexpected panic",
                        None,
                        diagnostic::Bug);
                }

                let xs = [
                    "the compiler unexpectedly panicked. this is a bug.".to_string(),
                    format!("we would appreciate a bug report: {}",
                            BUG_REPORT_URL),
                    "run with `RUST_BACKTRACE=1` for a backtrace".to_string(),
                ];
                for note in xs.iter() {
                    emitter.emit(None, note[], None, diagnostic::Note)
                }

                match r.read_to_string() {
                    Ok(s) => println!("{}", s),
                    Err(e) => {
                        emitter.emit(None,
                                     format!("failed to read internal \
                                              stderr: {}", e)[],
                                     None,
                                     diagnostic::Error)
                    }
                }
            }

            // Panic so the process returns a failure code, but don't pollute the
            // output with some unnecessary panic messages, we've already
            // printed everything that we needed to.
            io::stdio::set_stderr(box io::util::NullWriter);
            panic!();
        }
    }
}

pub fn main() {
    let args = std::os::args();
    let result = run(args);
    std::os::set_exit_status(result);
}
