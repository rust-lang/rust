// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

The Rust compiler.

# Note

This API is completely unstable and subject to change.

*/

#![crate_id = "rustc#0.11-pre"]
#![comment = "The Rust compiler"]
#![license = "MIT/ASL2"]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://static.rust-lang.org/doc/master")]

#![allow(deprecated)]
#![feature(macro_rules, globs, struct_variant, managed_boxes, quote,
           default_type_params, phase)]

extern crate flate;
extern crate arena;
extern crate syntax;
extern crate serialize;
extern crate sync;
extern crate getopts;
extern crate collections;
extern crate time;
extern crate libc;

#[phase(syntax, link)]
extern crate log;

use back::link;
use driver::session;
use middle::lint;

use d = driver::driver;

use std::any::AnyRefExt;
use std::cmp;
use std::io;
use std::os;
use std::str;
use std::task;
use syntax::ast;
use syntax::diagnostic::Emitter;
use syntax::diagnostic;
use syntax::parse;

pub mod middle {
    pub mod trans;
    pub mod ty;
    pub mod ty_fold;
    pub mod subst;
    pub mod resolve;
    pub mod resolve_lifetime;
    pub mod typeck;
    pub mod check_loop;
    pub mod check_match;
    pub mod check_const;
    pub mod check_static;
    pub mod lint;
    pub mod borrowck;
    pub mod dataflow;
    pub mod mem_categorization;
    pub mod liveness;
    pub mod kind;
    pub mod freevars;
    pub mod pat_util;
    pub mod region;
    pub mod const_eval;
    pub mod astencode;
    pub mod lang_items;
    pub mod privacy;
    pub mod moves;
    pub mod entry;
    pub mod effect;
    pub mod reachable;
    pub mod graph;
    pub mod cfg;
    pub mod dead;
}

pub mod front {
    pub mod config;
    pub mod test;
    pub mod std_inject;
    pub mod assign_node_ids_and_map;
    pub mod feature_gate;
    pub mod show_span;
}

pub mod back {
    pub mod abi;
    pub mod archive;
    pub mod arm;
    pub mod link;
    pub mod lto;
    pub mod mips;
    pub mod rpath;
    pub mod svh;
    pub mod target_strs;
    pub mod x86;
    pub mod x86_64;
}

pub mod metadata;

pub mod driver;

pub mod util {
    pub mod common;
    pub mod ppaux;
    pub mod sha2;
    pub mod nodemap;
}

pub mod lib {
    pub mod llvm;
    pub mod llvmdeps;
}

static BUG_REPORT_URL: &'static str =
    "http://static.rust-lang.org/doc/master/complement-bugreport.html";

pub fn version(argv0: &str) {
    let vers = match option_env!("CFG_VERSION") {
        Some(vers) => vers,
        None => "unknown version"
    };
    println!("{} {}", argv0, vers);
    println!("host: {}", d::host_triple());
}

pub fn usage(argv0: &str) {
    let message = format!("Usage: {} [OPTIONS] INPUT", argv0);
    println!("{}\n\
Additional help:
    -C help             Print codegen options
    -W help             Print 'lint' options and default settings
    -Z help             Print internal options for debugging rustc\n",
              getopts::usage(message, d::optgroups().as_slice()));
}

pub fn describe_warnings() {
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

pub fn describe_debug_flags() {
    println!("\nAvailable debug options:\n");
    let r = session::debugging_opts_map();
    for tuple in r.iter() {
        match *tuple {
            (ref name, ref desc, _) => {
                println!("    -Z {:>20s} -- {}", *name, *desc);
            }
        }
    }
}

pub fn describe_codegen_flags() {
    println!("\nAvailable codegen options:\n");
    let mut cg = session::basic_codegen_options();
    for &(name, parser, desc) in session::CG_OPTIONS.iter() {
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

pub fn run_compiler(args: &[~str]) {
    let mut args = args.to_owned();
    let binary = args.shift().unwrap();

    if args.is_empty() { usage(binary); return; }

    let matches =
        &match getopts::getopts(args, d::optgroups().as_slice()) {
          Ok(m) => m,
          Err(f) => {
            d::early_error(f.to_err_msg());
          }
        };

    if matches.opt_present("h") || matches.opt_present("help") {
        usage(binary);
        return;
    }

    let lint_flags = matches.opt_strs("W").move_iter().collect::<Vec<_>>().append(
                                    matches.opt_strs("warn").as_slice());
    if lint_flags.iter().any(|x| x == &~"help") {
        describe_warnings();
        return;
    }

    let r = matches.opt_strs("Z");
    if r.iter().any(|x| x == &~"help") {
        describe_debug_flags();
        return;
    }

    let cg_flags = matches.opt_strs("C");
    if cg_flags.iter().any(|x| x == &~"help") {
        describe_codegen_flags();
        return;
    }

    if cg_flags.contains(&~"passes=list") {
        unsafe { lib::llvm::llvm::LLVMRustPrintPasses(); }
        return;
    }

    if matches.opt_present("v") || matches.opt_present("version") {
        version(binary);
        return;
    }
    let (input, input_file_path) = match matches.free.len() {
      0u => d::early_error("no input filename given"),
      1u => {
        let ifile = matches.free.get(0).as_slice();
        if ifile == "-" {
            let contents = io::stdin().read_to_end().unwrap();
            let src = str::from_utf8(contents.as_slice()).unwrap().to_owned();
            (d::StrInput(src), None)
        } else {
            (d::FileInput(Path::new(ifile)), Some(Path::new(ifile)))
        }
      }
      _ => d::early_error("multiple input filenames provided")
    };

    let sopts = d::build_session_options(matches);
    let sess = d::build_session(sopts, input_file_path);
    let odir = matches.opt_str("out-dir").map(|o| Path::new(o));
    let ofile = matches.opt_str("o").map(|o| Path::new(o));
    let cfg = d::build_configuration(&sess);
    let pretty = matches.opt_default("pretty", "normal").map(|a| {
        d::parse_pretty(&sess, a)
    });
    match pretty {
        Some::<d::PpMode>(ppm) => {
            d::pretty_print_input(sess, cfg, &input, ppm);
            return;
        }
        None::<d::PpMode> => {/* continue */ }
    }
    let ls = matches.opt_present("ls");
    if ls {
        match input {
            d::FileInput(ref ifile) => {
                let mut stdout = io::stdout();
                d::list_metadata(&sess, &(*ifile), &mut stdout).unwrap();
            }
            d::StrInput(_) => {
                d::early_error("can not list metadata for stdin");
            }
        }
        return;
    }
    let (crate_id, crate_name, crate_file_name) = sess.opts.print_metas;
    // these nasty nested conditions are to avoid doing extra work
    if crate_id || crate_name || crate_file_name {
        let attrs = parse_crate_attrs(&sess, &input);
        let t_outputs = d::build_output_filenames(&input, &odir, &ofile,
                                                  attrs.as_slice(), &sess);
        let id = link::find_crate_id(attrs.as_slice(), t_outputs.out_filestem);

        if crate_id {
            println!("{}", id.to_str());
        }
        if crate_name {
            println!("{}", id.name);
        }
        if crate_file_name {
            let crate_types = session::collect_crate_types(&sess,
                                                           attrs.as_slice());
            for &style in crate_types.iter() {
                let fname = link::filename_for_input(&sess, style, &id,
                                                     &t_outputs.with_extension(""));
                println!("{}", fname.filename_display());
            }
        }

        return;
    }

    d::compile_input(sess, cfg, &input, &odir, &ofile);
}

fn parse_crate_attrs(sess: &session::Session, input: &d::Input) ->
                     Vec<ast::Attribute> {
    let result = match *input {
        d::FileInput(ref ifile) => {
            parse::parse_crate_attrs_from_file(ifile,
                                               Vec::new(),
                                               &sess.parse_sess)
        }
        d::StrInput(ref src) => {
            parse::parse_crate_attrs_from_source_str(d::anon_src(),
                                                     (*src).clone(),
                                                     Vec::new(),
                                                     &sess.parse_sess)
        }
    };
    result.move_iter().collect()
}

/// Run a procedure which will detect failures in the compiler and print nicer
/// error messages rather than just failing the test.
///
/// The diagnostic emitter yielded to the procedure should be used for reporting
/// errors of the compiler.
pub fn monitor(f: proc():Send) {
    // FIXME: This is a hack for newsched since it doesn't support split stacks.
    // rustc needs a lot of stack! When optimizations are disabled, it needs
    // even *more* stack than usual as well.
    #[cfg(rtopt)]
    static STACK_SIZE: uint = 6000000;  // 6MB
    #[cfg(not(rtopt))]
    static STACK_SIZE: uint = 20000000; // 20MB

    let mut task_builder = task::task().named("rustc");

    // FIXME: Hacks on hacks. If the env is trying to override the stack size
    // then *don't* set it explicitly.
    if os::getenv("RUST_MIN_STACK").is_none() {
        task_builder.opts.stack_size = Some(STACK_SIZE);
    }

    let (tx, rx) = channel();
    let w = io::ChanWriter::new(tx);
    let mut r = io::ChanReader::new(rx);

    match task_builder.try(proc() {
        io::stdio::set_stderr(~w);
        f()
    }) {
        Ok(()) => { /* fallthrough */ }
        Err(value) => {
            // Task failed without emitting a fatal diagnostic
            if !value.is::<diagnostic::FatalError>() {
                let mut emitter = diagnostic::EmitterWriter::stderr();

                // a .span_bug or .bug call has already printed what
                // it wants to print.
                if !value.is::<diagnostic::ExplicitBug>() {
                    emitter.emit(
                        None,
                        "unexpected failure",
                        diagnostic::Bug);
                }

                let xs = [
                    ~"the compiler hit an unexpected failure path. this is a bug.",
                    "we would appreciate a bug report: " + BUG_REPORT_URL,
                    ~"run with `RUST_BACKTRACE=1` for a backtrace",
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
            io::stdio::set_stderr(~io::util::NullWriter);
            fail!();
        }
    }
}

pub fn main() {
    std::os::set_exit_status(main_args(std::os::args()));
}

pub fn main_args(args: &[~str]) -> int {
    let owned_args = args.to_owned();
    monitor(proc() run_compiler(owned_args));
    0
}
