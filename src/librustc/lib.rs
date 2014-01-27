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

#[crate_id = "rustc#0.10-pre"];
#[comment = "The Rust compiler"];
#[license = "MIT/ASL2"];
#[crate_type = "dylib"];
#[crate_type = "rlib"];
#[doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://static.rust-lang.org/doc/master")];

#[feature(macro_rules, globs, struct_variant, managed_boxes)];

extern mod extra;
extern mod flate;
extern mod syntax;

use back::link;
use driver::session;
use middle::lint;

use d = driver::driver;

use std::io;
use std::num;
use std::os;
use std::str;
use std::task;
use std::vec;
use extra::getopts::groups;
use extra::getopts;
use syntax::ast;
use syntax::attr;
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
}

pub mod back {
    pub mod archive;
    pub mod link;
    pub mod abi;
    pub mod arm;
    pub mod mips;
    pub mod x86;
    pub mod x86_64;
    pub mod rpath;
    pub mod target_strs;
    pub mod lto;
}

pub mod metadata;

pub mod driver;

pub mod util {
    pub mod common;
    pub mod ppaux;
    pub mod sha2;
}

pub mod lib {
    pub mod llvm;
    pub mod llvmdeps;
}

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
    -W help             Print 'lint' options and default settings
    -Z help             Print internal options for debugging rustc\n",
              groups::usage(message, d::optgroups()));
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
                                 .collect::<~[(lint::LintSpec, &'static str)]>();
    lint_dict.sort();

    let mut max_key = 0;
    for &(_, name) in lint_dict.iter() {
        max_key = num::max(name.len(), max_key);
    }
    fn padded(max: uint, s: &str) -> ~str {
        " ".repeat(max - s.len()) + s
    }
    println!("{}", "\nAvailable lint checks:\n"); // FIXME: #9970
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
    println!("{}", "\nAvailable debug options:\n"); // FIXME: #9970
    let r = session::debugging_opts_map();
    for tuple in r.iter() {
        match *tuple {
            (ref name, ref desc, _) => {
                println!("    -Z {:>20s} -- {}", *name, *desc);
            }
        }
    }
}

pub fn run_compiler(args: &[~str], demitter: @diagnostic::Emitter) {
    let mut args = args.to_owned();
    let binary = args.shift().unwrap();

    if args.is_empty() { usage(binary); return; }

    let matches =
        &match getopts::groups::getopts(args, d::optgroups()) {
          Ok(m) => m,
          Err(f) => {
            d::early_error(demitter, f.to_err_msg());
          }
        };

    if matches.opt_present("h") || matches.opt_present("help") {
        usage(binary);
        return;
    }

    let lint_flags = vec::append(matches.opt_strs("W"),
                                 matches.opt_strs("warn"));
    if lint_flags.iter().any(|x| x == &~"help") {
        describe_warnings();
        return;
    }

    let r = matches.opt_strs("Z");
    if r.iter().any(|x| x == &~"help") {
        describe_debug_flags();
        return;
    }

    if matches.opt_str("passes") == Some(~"list") {
        unsafe { lib::llvm::llvm::LLVMRustPrintPasses(); }
        return;
    }

    if matches.opt_present("v") || matches.opt_present("version") {
        version(binary);
        return;
    }
    let (input, input_file_path) = match matches.free.len() {
      0u => d::early_error(demitter, "no input filename given"),
      1u => {
        let ifile = matches.free[0].as_slice();
        if "-" == ifile {
            let src = str::from_utf8_owned(io::stdin().read_to_end()).unwrap();
            (d::StrInput(src.to_managed()), None)
        } else {
            (d::FileInput(Path::new(ifile)), Some(Path::new(ifile)))
        }
      }
      _ => d::early_error(demitter, "multiple input filenames provided")
    };

    let sopts = d::build_session_options(binary, matches, demitter);
    let sess = d::build_session(sopts, input_file_path, demitter);
    let odir = matches.opt_str("out-dir").map(|o| Path::new(o));
    let ofile = matches.opt_str("o").map(|o| Path::new(o));
    let cfg = d::build_configuration(sess);
    let pretty = matches.opt_default("pretty", "normal").map(|a| {
        d::parse_pretty(sess, a)
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
            d::list_metadata(sess, &(*ifile),
                                  &mut stdout as &mut io::Writer);
          }
          d::StrInput(_) => {
            d::early_error(demitter, "can not list metadata for stdin");
          }
        }
        return;
    }
    let (crate_id, crate_name, crate_file_name) = sopts.print_metas;
    // these nasty nested conditions are to avoid doing extra work
    if crate_id || crate_name || crate_file_name {
        let attrs = parse_crate_attrs(sess, &input);
        let t_outputs = d::build_output_filenames(&input, &odir, &ofile,
                                                  attrs, sess);
        if crate_id || crate_name {
            let crateid = match attr::find_crateid(attrs) {
                Some(crateid) => crateid,
                None => {
                    sess.fatal("No crate_id and --crate-id or \
                                --crate-name requested")
                }
            };
            if crate_id {
                println!("{}", crateid.to_str());
            }
            if crate_name {
                println!("{}", crateid.name);
            }
        }

        if crate_file_name {
            let lm = link::build_link_meta(sess, attrs, &t_outputs.obj_filename,
                                           &mut ::util::sha2::Sha256::new());
            let outputs = session::collect_outputs(&sess, attrs);
            for &style in outputs.iter() {
                let fname = link::filename_for_input(&sess, style, &lm,
                                                     &t_outputs.out_filename);
                println!("{}", fname.filename_display());
            }
        }

        return;
    }

    d::compile_input(sess, cfg, &input, &odir, &ofile);
}

fn parse_crate_attrs(sess: session::Session,
                     input: &d::Input) -> ~[ast::Attribute] {
    match *input {
        d::FileInput(ref ifile) => {
            parse::parse_crate_attrs_from_file(ifile, ~[], sess.parse_sess)
        }
        d::StrInput(src) => {
            parse::parse_crate_attrs_from_source_str(
                d::anon_src(), src, ~[], sess.parse_sess)
        }
    }
}

/// Run a procedure which will detect failures in the compiler and print nicer
/// error messages rather than just failing the test.
///
/// The diagnostic emitter yielded to the procedure should be used for reporting
/// errors of the compiler.
pub fn monitor(f: proc(@diagnostic::Emitter)) {
    // FIXME: This is a hack for newsched since it doesn't support split stacks.
    // rustc needs a lot of stack! When optimizations are disabled, it needs
    // even *more* stack than usual as well.
    #[cfg(rtopt)]
    static STACK_SIZE: uint = 6000000;  // 6MB
    #[cfg(not(rtopt))]
    static STACK_SIZE: uint = 20000000; // 20MB

    let mut task_builder = task::task();
    task_builder.name("rustc");

    // FIXME: Hacks on hacks. If the env is trying to override the stack size
    // then *don't* set it explicitly.
    if os::getenv("RUST_MIN_STACK").is_none() {
        task_builder.opts.stack_size = Some(STACK_SIZE);
    }

    let (p, c) = Chan::new();
    let w = io::ChanWriter::new(c);
    let mut r = io::PortReader::new(p);

    match task_builder.try(proc() {
        io::stdio::set_stderr(~w as ~io::Writer);
        f(@diagnostic::DefaultEmitter)
    }) {
        Ok(()) => { /* fallthrough */ }
        Err(value) => {
            // Task failed without emitting a fatal diagnostic
            if !value.is::<diagnostic::FatalError>() {
                diagnostic::DefaultEmitter.emit(
                    None,
                    diagnostic::ice_msg("unexpected failure"),
                    diagnostic::Error);

                let xs = [
                    ~"the compiler hit an unexpected failure path. \
                     this is a bug",
                ];
                for note in xs.iter() {
                    diagnostic::DefaultEmitter.emit(None,
                                                    *note,
                                                    diagnostic::Note)
                }

                println!("{}", r.read_to_str());
            }

            // Fail so the process returns a failure code, but don't pollute the
            // output with some unnecessary failure messages, we've already
            // printed everything that we needed to.
            io::stdio::set_stderr(~io::util::NullWriter as ~io::Writer);
            fail!();
        }
    }
}

pub fn main() {
    std::os::set_exit_status(main_args(std::os::args()));
}

pub fn main_args(args: &[~str]) -> int {
    let owned_args = args.to_owned();
    monitor(proc(demitter) run_compiler(owned_args, demitter));
    0
}
