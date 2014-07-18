// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use syntax::diagnostic;

use back::link;
use driver::driver::{Input, FileInput, StrInput};
use driver::session::{Session, build_session};
use lint::Lint;
use lint;
use metadata;

use std::any::AnyRefExt;
use std::io;
use std::os;
use std::task::TaskBuilder;

use syntax::ast;
use syntax::parse;
use syntax::diagnostic::Emitter;
use syntax::diagnostics;

use getopts;


pub mod driver;
pub mod session;
pub mod config;


pub fn main_args(args: &[String]) -> int {
    let owned_args = args.to_vec();
    monitor(proc() run_compiler(owned_args.as_slice()));
    0
}

static BUG_REPORT_URL: &'static str =
    "http://doc.rust-lang.org/complement-bugreport.html";

fn run_compiler(args: &[String]) {
    let matches = match handle_options(Vec::from_slice(args)) {
        Some(matches) => matches,
        None => return
    };

    let descriptions = diagnostics::registry::Registry::new(super::DIAGNOSTICS);
    match matches.opt_str("explain") {
        Some(ref code) => {
            match descriptions.find_description(code.as_slice()) {
                Some(ref description) => {
                    println!("{}", description);
                }
                None => {
                    early_error(format!("no extended information for {}", code).as_slice());
                }
            }
            return;
        },
        None => ()
    }

    let sopts = config::build_session_options(&matches);
    let (input, input_file_path) = match matches.free.len() {
        0u => {
            if sopts.describe_lints {
                let mut ls = lint::LintStore::new();
                ls.register_builtin(None);
                describe_lints(&ls, false);
                return;
            }
            early_error("no input filename given");
        }
        1u => {
            let ifile = matches.free.get(0).as_slice();
            if ifile == "-" {
                let contents = io::stdin().read_to_end().unwrap();
                let src = String::from_utf8(contents).unwrap();
                (StrInput(src), None)
            } else {
                (FileInput(Path::new(ifile)), Some(Path::new(ifile)))
            }
        }
        _ => early_error("multiple input filenames provided")
    };

    let sess = build_session(sopts, input_file_path, descriptions);
    let cfg = config::build_configuration(&sess);
    let odir = matches.opt_str("out-dir").map(|o| Path::new(o));
    let ofile = matches.opt_str("o").map(|o| Path::new(o));

    let pretty = matches.opt_default("pretty", "normal").map(|a| {
        parse_pretty(&sess, a.as_slice())
    });
    match pretty {
        Some::<PpMode>(ppm) => {
            driver::pretty_print_input(sess, cfg, &input, ppm, ofile);
            return;
        }
        None::<PpMode> => {/* continue */ }
    }

    let r = matches.opt_strs("Z");
    if r.contains(&("ls".to_string())) {
        match input {
            FileInput(ref ifile) => {
                let mut stdout = io::stdout();
                list_metadata(&sess, &(*ifile), &mut stdout).unwrap();
            }
            StrInput(_) => {
                early_error("can not list metadata for stdin");
            }
        }
        return;
    }

    if print_crate_info(&sess, &input, &odir, &ofile) {
        return;
    }

    driver::compile_input(sess, cfg, &input, &odir, &ofile);
}

/// Prints version information and returns None on success or an error
/// message on failure.
pub fn version(binary: &str, matches: &getopts::Matches) -> Option<String> {
    let verbose = match matches.opt_str("version").as_ref().map(|s| s.as_slice()) {
        None => false,
        Some("verbose") => true,
        Some(s) => return Some(format!("Unrecognized argument: {}", s))
    };

    println!("{} {}", binary, env!("CFG_VERSION"));
    if verbose {
        println!("binary: {}", binary);
        println!("commit-hash: {}", option_env!("CFG_VER_HASH").unwrap_or("unknown"));
        println!("commit-date: {}", option_env!("CFG_VER_DATE").unwrap_or("unknown"));
        println!("host: {}", driver::host_triple());
        println!("release: {}", env!("CFG_RELEASE"));
    }
    None
}

fn usage() {
    let message = format!("Usage: rustc [OPTIONS] INPUT");
    println!("{}\n\
Additional help:
    -C help             Print codegen options
    -W help             Print 'lint' options and default settings
    -Z help             Print internal options for debugging rustc\n",
              getopts::usage(message.as_slice(),
                             config::optgroups().as_slice()));
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
        let mut lints: Vec<_> = lints.move_iter().map(|(x, _)| x).collect();
        lints.sort_by(|x: &&Lint, y: &&Lint| {
            match x.default_level.cmp(&y.default_level) {
                // The sort doesn't case-fold but it's doubtful we care.
                Equal => x.name.cmp(&y.name),
                r => r,
            }
        });
        lints
    }

    let (plugin, builtin) = lint_store.get_lints().partitioned(|&(_, p)| p);
    let plugin = sort_lints(plugin);
    let builtin = sort_lints(builtin);

    // FIXME (#7043): We should use the width in character cells rather than
    // the number of codepoints.
    let max_name_len = plugin.iter().chain(builtin.iter())
        .map(|&s| s.name.char_len())
        .max().unwrap_or(0);
    let padded = |x: &str| {
        " ".repeat(max_name_len - x.char_len()).append(x)
    };

    println!("Lint checks provided by rustc:\n");
    println!("    {}  {:7.7s}  {}", padded("name"), "default", "meaning");
    println!("    {}  {:7.7s}  {}", padded("----"), "-------", "-------");

    let print_lints = |lints: Vec<&Lint>| {
        for lint in lints.move_iter() {
            let name = lint.name_lower().replace("_", "-");
            println!("    {}  {:7.7s}  {}",
                     padded(name.as_slice()), lint.default_level.as_str(), lint.desc);
        }
        println!("\n");
    };

    print_lints(builtin);

    match (loaded_plugins, plugin.len()) {
        (false, 0) => {
            println!("Compiler plugins can provide additional lints. To see a listing of these, \
                      re-run `rustc -W help` with a crate filename.");
        }
        (false, _) => fail!("didn't load lint plugins but got them anyway!"),
        (true, 0) => println!("This crate does not load any lint plugins."),
        (true, _) => {
            println!("Lint checks provided by plugins loaded by this crate:\n");
            print_lints(plugin);
        }
    }
}

fn describe_debug_flags() {
    println!("\nAvailable debug options:\n");
    let r = config::debugging_opts_map();
    for tuple in r.iter() {
        match *tuple {
            (ref name, ref desc, _) => {
                println!("    -Z {:>20s} -- {}", *name, *desc);
            }
        }
    }
}

fn describe_codegen_flags() {
    println!("\nAvailable codegen options:\n");
    let mut cg = config::basic_codegen_options();
    for &(name, parser, desc) in config::CG_OPTIONS.iter() {
        // we invoke the parser function on `None` to see if this option needs
        // an argument or not.
        let (width, extra) = if parser(&mut cg, None) {
            (25, "")
        } else {
            (21, "=val")
        };
        println!("    -C {:>width$s}{} -- {}", name.replace("_", "-"),
                 extra, desc, width=width);
    }
}

/// Process command line options. Emits messages as appropriate. If compilation
/// should continue, returns a getopts::Matches object parsed from args, otherwise
/// returns None.
pub fn handle_options(mut args: Vec<String>) -> Option<getopts::Matches> {
    // Throw away the first argument, the name of the binary
    let _binary = args.shift().unwrap();

    if args.is_empty() {
        usage();
        return None;
    }

    let matches =
        match getopts::getopts(args.as_slice(), config::optgroups().as_slice()) {
            Ok(m) => m,
            Err(f) => {
                early_error(f.to_string().as_slice());
            }
        };

    if matches.opt_present("h") || matches.opt_present("help") {
        usage();
        return None;
    }

    // Don't handle -W help here, because we might first load plugins.

    let r = matches.opt_strs("Z");
    if r.iter().any(|x| x.as_slice() == "help") {
        describe_debug_flags();
        return None;
    }

    let cg_flags = matches.opt_strs("C");
    if cg_flags.iter().any(|x| x.as_slice() == "help") {
        describe_codegen_flags();
        return None;
    }

    if cg_flags.contains(&"passes=list".to_string()) {
        unsafe { ::llvm::LLVMRustPrintPasses(); }
        return None;
    }

    if matches.opt_present("version") {
        match version("rustc", &matches) {
            Some(err) => early_error(err.as_slice()),
            None => return None
        }
    }

    Some(matches)
}

fn print_crate_info(sess: &Session,
                    input: &Input,
                    odir: &Option<Path>,
                    ofile: &Option<Path>)
                    -> bool {
    let (crate_name, crate_file_name) = sess.opts.print_metas;
    // these nasty nested conditions are to avoid doing extra work
    if crate_name || crate_file_name {
        let attrs = parse_crate_attrs(sess, input);
        let t_outputs = driver::build_output_filenames(input,
                                                       odir,
                                                       ofile,
                                                       attrs.as_slice(),
                                                       sess);
        let id = link::find_crate_name(Some(sess), attrs.as_slice(), input);

        if crate_name {
            println!("{}", id);
        }
        if crate_file_name {
            let crate_types = driver::collect_crate_types(sess, attrs.as_slice());
            let metadata = driver::collect_crate_metadata(sess, attrs.as_slice());
            *sess.crate_metadata.borrow_mut() = metadata;
            for &style in crate_types.iter() {
                let fname = link::filename_for_input(sess, style, id.as_slice(),
                                                     &t_outputs.with_extension(""));
                println!("{}", fname.filename_display());
            }
        }

        true
    } else {
        false
    }
}

pub enum PpMode {
    PpmNormal,
    PpmExpanded,
    PpmTyped,
    PpmIdentified,
    PpmExpandedIdentified,
    PpmFlowGraph(ast::NodeId),
}

pub fn parse_pretty(sess: &Session, name: &str) -> PpMode {
    let mut split = name.splitn('=', 1);
    let first = split.next().unwrap();
    let opt_second = split.next();
    match (opt_second, first) {
        (None, "normal")       => PpmNormal,
        (None, "expanded")     => PpmExpanded,
        (None, "typed")        => PpmTyped,
        (None, "expanded,identified") => PpmExpandedIdentified,
        (None, "identified")   => PpmIdentified,
        (arg, "flowgraph") => {
             match arg.and_then(from_str) {
                 Some(id) => PpmFlowGraph(id),
                 None => {
                     sess.fatal(format!("`pretty flowgraph=<nodeid>` needs \
                                         an integer <nodeid>; got {}",
                                        arg.unwrap_or("nothing")).as_slice())
                 }
             }
        }
        _ => {
            sess.fatal(format!(
                "argument to `pretty` must be one of `normal`, \
                 `expanded`, `flowgraph=<nodeid>`, `typed`, `identified`, \
                 or `expanded,identified`; got {}", name).as_slice());
        }
    }
}

fn parse_crate_attrs(sess: &Session, input: &Input) ->
                     Vec<ast::Attribute> {
    let result = match *input {
        FileInput(ref ifile) => {
            parse::parse_crate_attrs_from_file(ifile,
                                               Vec::new(),
                                               &sess.parse_sess)
        }
        StrInput(ref src) => {
            parse::parse_crate_attrs_from_source_str(
                driver::anon_src().to_string(),
                src.to_string(),
                Vec::new(),
                &sess.parse_sess)
        }
    };
    result.move_iter().collect()
}

pub fn early_error(msg: &str) -> ! {
    let mut emitter = diagnostic::EmitterWriter::stderr(diagnostic::Auto, None);
    emitter.emit(None, msg, None, diagnostic::Fatal);
    fail!(diagnostic::FatalError);
}

pub fn early_warn(msg: &str) {
    let mut emitter = diagnostic::EmitterWriter::stderr(diagnostic::Auto, None);
    emitter.emit(None, msg, None, diagnostic::Warning);
}

pub fn list_metadata(sess: &Session, path: &Path,
                     out: &mut io::Writer) -> io::IoResult<()> {
    metadata::loader::list_file_metadata(sess.targ_cfg.os, path, out)
}

/// Run a procedure which will detect failures in the compiler and print nicer
/// error messages rather than just failing the test.
///
/// The diagnostic emitter yielded to the procedure should be used for reporting
/// errors of the compiler.
fn monitor(f: proc():Send) {
    // FIXME: This is a hack for newsched since it doesn't support split stacks.
    // rustc needs a lot of stack! When optimizations are disabled, it needs
    // even *more* stack than usual as well.
    #[cfg(rtopt)]
    static STACK_SIZE: uint = 6000000;  // 6MB
    #[cfg(not(rtopt))]
    static STACK_SIZE: uint = 20000000; // 20MB

    let (tx, rx) = channel();
    let w = io::ChanWriter::new(tx);
    let mut r = io::ChanReader::new(rx);

    let mut task = TaskBuilder::new().named("rustc").stderr(box w);

    // FIXME: Hacks on hacks. If the env is trying to override the stack size
    // then *don't* set it explicitly.
    if os::getenv("RUST_MIN_STACK").is_none() {
        task = task.stack_size(STACK_SIZE);
    }

    match task.try(f) {
        Ok(()) => { /* fallthrough */ }
        Err(value) => {
            // Task failed without emitting a fatal diagnostic
            if !value.is::<diagnostic::FatalError>() {
                let mut emitter = diagnostic::EmitterWriter::stderr(diagnostic::Auto, None);

                // a .span_bug or .bug call has already printed what
                // it wants to print.
                if !value.is::<diagnostic::ExplicitBug>() {
                    emitter.emit(
                        None,
                        "unexpected failure",
                        None,
                        diagnostic::Bug);
                }

                let xs = [
                    "the compiler hit an unexpected failure path. this is a bug.".to_string(),
                    format!("we would appreciate a bug report: {}",
                            BUG_REPORT_URL),
                    "run with `RUST_BACKTRACE=1` for a backtrace".to_string(),
                ];
                for note in xs.iter() {
                    emitter.emit(None, note.as_slice(), None, diagnostic::Note)
                }

                match r.read_to_string() {
                    Ok(s) => println!("{}", s),
                    Err(e) => {
                        emitter.emit(None,
                                     format!("failed to read internal \
                                              stderr: {}",
                                             e).as_slice(),
                                     None,
                                     diagnostic::Error)
                    }
                }
            }

            // Fail so the process returns a failure code, but don't pollute the
            // output with some unnecessary failure messages, we've already
            // printed everything that we needed to.
            io::stdio::set_stderr(box io::util::NullWriter);
            fail!();
        }
    }
}
