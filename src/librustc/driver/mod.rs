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
use middle::lint;
use metadata;

use std::any::AnyRefExt;
use std::cmp;
use std::io;
use std::os;
use std::str;
use std::task::TaskBuilder;

use syntax::ast;
use syntax::parse;
use syntax::diagnostic::Emitter;

use getopts;


pub mod driver;
pub mod session;
pub mod config;


pub fn main_args(args: &[~str]) -> int {
    let owned_args = args.to_owned();
    monitor(proc() run_compiler(owned_args));
    0
}

static BUG_REPORT_URL: &'static str =
    "http://static.rust-lang.org/doc/master/complement-bugreport.html";

fn run_compiler(args: &[~str]) {
    let matches = match handle_options(Vec::from_slice(args)) {
        Some(matches) => matches,
        None => return
    };

    let (input, input_file_path) = match matches.free.len() {
        0u => early_error("no input filename given"),
        1u => {
            let ifile = matches.free.get(0).as_slice();
            if ifile == "-" {
                let contents = io::stdin().read_to_end().unwrap();
                let src = str::from_utf8(contents.as_slice()).unwrap()
                                                             .to_strbuf();
                (StrInput(src), None)
            } else {
                (FileInput(Path::new(ifile)), Some(Path::new(ifile)))
            }
        }
        _ => early_error("multiple input filenames provided")
    };

    let sopts = config::build_session_options(&matches);
    let sess = build_session(sopts, input_file_path);
    let cfg = config::build_configuration(&sess);
    let odir = matches.opt_str("out-dir").map(|o| Path::new(o));
    let ofile = matches.opt_str("o").map(|o| Path::new(o));

    let pretty = matches.opt_default("pretty", "normal").map(|a| {
        parse_pretty(&sess, a)
    });
    match pretty {
        Some::<PpMode>(ppm) => {
            driver::pretty_print_input(sess, cfg, &input, ppm, ofile);
            return;
        }
        None::<PpMode> => {/* continue */ }
    }

    let r = matches.opt_strs("Z");
    if r.contains(&("ls".to_owned())) {
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

pub fn version(command: &str) {
    let vers = match option_env!("CFG_VERSION") {
        Some(vers) => vers,
        None => "unknown version"
    };
    println!("{} {}", command, vers);
    println!("host: {}", driver::host_triple());
}

fn usage() {
    let message = format!("Usage: rustc [OPTIONS] INPUT");
    println!("{}\n\
Additional help:
    -C help             Print codegen options
    -W help             Print 'lint' options and default settings
    -Z help             Print internal options for debugging rustc\n",
              getopts::usage(message, config::optgroups().as_slice()));
}

fn describe_warnings() {
    println!("
Available lint options:
    -W <foo>           Warn about <foo>
    -A <foo>           Allow <foo>
    -D <foo>           Deny <foo>
    -F <foo>           Forbid <foo> (deny, and deny all overrides)
");

    let lint_dict = lint::get_lint_dict();
    let mut lint_dict = lint_dict.move_iter()
                                 .map(|(k, v)| (v, k))
                                 .collect::<Vec<(lint::LintSpec, &'static str)> >();
    lint_dict.as_mut_slice().sort();

    let mut max_key = 0;
    for &(_, name) in lint_dict.iter() {
        max_key = cmp::max(name.len(), max_key);
    }
    fn padded(max: uint, s: &str) -> ~str {
        " ".repeat(max - s.len()) + s
    }
    println!("\nAvailable lint checks:\n");
    println!("    {}  {:7.7s}  {}",
             padded(max_key, "name"), "default", "meaning");
    println!("    {}  {:7.7s}  {}\n",
             padded(max_key, "----"), "-------", "-------");
    for (spec, name) in lint_dict.move_iter() {
        let name = name.replace("_", "-");
        println!("    {}  {:7.7s}  {}",
                 padded(max_key, name),
                 lint::level_to_str(spec.default),
                 spec.desc);
    }
    println!("");
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

/// Process command line options. Emits messages as appropirate.If compilation
/// should continue, returns a getopts::Matches object parsed from args, otherwise
/// returns None.
pub fn handle_options(mut args: Vec<~str>) -> Option<getopts::Matches> {
    // Throw away the first argument, the name of the binary
    let _binary = args.shift().unwrap();

    if args.is_empty() { usage(); return None; }

    let matches =
        match getopts::getopts(args.as_slice(), config::optgroups().as_slice()) {
            Ok(m) => m,
            Err(f) => {
                early_error(f.to_err_msg());
            }
        };

    if matches.opt_present("h") || matches.opt_present("help") {
        usage();
        return None;
    }

    let lint_flags = matches.opt_strs("W").move_iter().collect::<Vec<_>>().append(
                                    matches.opt_strs("warn").as_slice());
    if lint_flags.iter().any(|x| x == &"help".to_owned()) {
        describe_warnings();
        return None;
    }

    let r = matches.opt_strs("Z");
    if r.iter().any(|x| x == &"help".to_owned()) {
        describe_debug_flags();
        return None;
    }

    let cg_flags = matches.opt_strs("C");
    if cg_flags.iter().any(|x| x == &"help".to_owned()) {
        describe_codegen_flags();
        return None;
    }

    if cg_flags.contains(&"passes=list".to_owned()) {
        unsafe { ::lib::llvm::llvm::LLVMRustPrintPasses(); }
        return None;
    }

    if matches.opt_present("v") || matches.opt_present("version") {
        version("rustc");
        return None;
    }

    Some(matches)
}

fn print_crate_info(sess: &Session,
                    input: &Input,
                    odir: &Option<Path>,
                    ofile: &Option<Path>)
                    -> bool {
    let (crate_id, crate_name, crate_file_name) = sess.opts.print_metas;
    // these nasty nested conditions are to avoid doing extra work
    if crate_id || crate_name || crate_file_name {
        let attrs = parse_crate_attrs(sess, input);
        let t_outputs = driver::build_output_filenames(input,
                                                       odir,
                                                       ofile,
                                                       attrs.as_slice(),
                                                       sess);
        let id = link::find_crate_id(attrs.as_slice(),
                                     t_outputs.out_filestem.as_slice());

        if crate_id {
            println!("{}", id.to_str());
        }
        if crate_name {
            println!("{}", id.name);
        }
        if crate_file_name {
            let crate_types = driver::collect_crate_types(sess, attrs.as_slice());
            for &style in crate_types.iter() {
                let fname = link::filename_for_input(sess, style, &id,
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
        (Some(s), "flowgraph") => {
             match from_str(s) {
                 Some(id) => PpmFlowGraph(id),
                 None => sess.fatal(format!("`pretty flowgraph=<nodeid>` needs \
                                             an integer <nodeid>; got {}", s))
             }
        }
        _ => {
            sess.fatal(format!(
                "argument to `pretty` must be one of `normal`, \
                 `expanded`, `flowgraph=<nodeid>`, `typed`, `identified`, \
                 or `expanded,identified`; got {}", name));
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
                driver::anon_src().to_strbuf(),
                src.to_strbuf(),
                Vec::new(),
                &sess.parse_sess)
        }
    };
    result.move_iter().collect()
}

pub fn early_error(msg: &str) -> ! {
    let mut emitter = diagnostic::EmitterWriter::stderr(diagnostic::Auto);
    emitter.emit(None, msg, diagnostic::Fatal);
    fail!(diagnostic::FatalError);
}

pub fn list_metadata(sess: &Session, path: &Path,
                     out: &mut io::Writer) -> io::IoResult<()> {
    metadata::loader::list_file_metadata(
        config::cfg_os_to_meta_os(sess.targ_cfg.os), path, out)
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

    let mut task_builder = TaskBuilder::new().named("rustc");

    // FIXME: Hacks on hacks. If the env is trying to override the stack size
    // then *don't* set it explicitly.
    if os::getenv("RUST_MIN_STACK").is_none() {
        task_builder.opts.stack_size = Some(STACK_SIZE);
    }

    let (tx, rx) = channel();
    let w = io::ChanWriter::new(tx);
    let mut r = io::ChanReader::new(rx);

    match task_builder.try(proc() {
        io::stdio::set_stderr(box w);
        f()
    }) {
        Ok(()) => { /* fallthrough */ }
        Err(value) => {
            // Task failed without emitting a fatal diagnostic
            if !value.is::<diagnostic::FatalError>() {
                let mut emitter = diagnostic::EmitterWriter::stderr(diagnostic::Auto);

                // a .span_bug or .bug call has already printed what
                // it wants to print.
                if !value.is::<diagnostic::ExplicitBug>() {
                    emitter.emit(
                        None,
                        "unexpected failure",
                        diagnostic::Bug);
                }

                let xs = [
                    "the compiler hit an unexpected failure path. this is a bug.".to_owned(),
                    "we would appreciate a bug report: " + BUG_REPORT_URL,
                    "run with `RUST_BACKTRACE=1` for a backtrace".to_owned(),
                ];
                for note in xs.iter() {
                    emitter.emit(None, *note, diagnostic::Note)
                }

                match r.read_to_str() {
                    Ok(s) => println!("{}", s),
                    Err(e) => emitter.emit(None,
                                           format!("failed to read internal stderr: {}", e),
                                           diagnostic::Error),
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
