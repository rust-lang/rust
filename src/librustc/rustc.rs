// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name = "rustc",
       vers = "0.8-pre",
       uuid = "0ce89b41-2f92-459e-bbc1-8f5fe32f16cf",
       url = "https://github.com/mozilla/rust/tree/master/src/rustc")];

#[comment = "The Rust compiler"];
#[license = "MIT/ASL2"];
#[crate_type = "lib"];

extern mod extra;
extern mod syntax;

use driver::driver::{host_triple, optgroups, early_error};
use driver::driver::{str_input, file_input, build_session_options};
use driver::driver::{build_session, build_configuration, parse_pretty};
use driver::driver::{pp_mode, pretty_print_input, list_metadata};
use driver::driver::{compile_input};
use driver::session;
use middle::lint;

use std::io;
use std::num;
use std::os;
use std::result;
use std::str;
use std::task;
use std::vec;
use extra::getopts::{groups, opt_present};
use extra::getopts;
use syntax::codemap;
use syntax::diagnostic;

pub mod middle {
    pub mod trans;
    pub mod ty;
    pub mod subst;
    pub mod resolve;
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
}

pub mod front {
    pub mod config;
    pub mod test;
    pub mod std_inject;
}

pub mod back {
    pub mod link;
    pub mod abi;
    pub mod upcall;
    pub mod arm;
    pub mod mips;
    pub mod x86;
    pub mod x86_64;
    pub mod rpath;
    pub mod target_strs;
    pub mod passes;
}

pub mod metadata;

pub mod driver;

pub mod util {
    pub mod common;
    pub mod ppaux;
}

pub mod lib {
    pub mod llvm;
}

// A curious inner module that allows ::std::foo to be available in here for
// macros.
/*
mod std {
    pub use std::clone;
    pub use std::cmp;
    pub use std::os;
    pub use std::str;
    pub use std::sys;
    pub use std::to_bytes;
    pub use std::unstable;
    pub use extra::serialize;
}
*/

#[cfg(stage0)]
pub fn version(argv0: &str) {
    let mut vers = ~"unknown version";
    let env_vers = env!("CFG_VERSION");
    if env_vers.len() != 0 { vers = env_vers.to_owned(); }
    printfln!("%s %s", argv0, vers);
    printfln!("host: %s", host_triple());
}

#[cfg(not(stage0))]
pub fn version(argv0: &str) {
    let vers = match option_env!("CFG_VERSION") {
        Some(vers) => vers,
        None => "unknown version"
    };
    printfln!("%s %s", argv0, vers);
    printfln!("host: %s", host_triple());
}

pub fn usage(argv0: &str) {
    let message = fmt!("Usage: %s [OPTIONS] INPUT", argv0);
    printfln!("%s\
Additional help:
    -W help             Print 'lint' options and default settings
    -Z help             Print internal options for debugging rustc\n",
              groups::usage(message, optgroups()));
}

pub fn describe_warnings() {
    use extra::sort::Sort;
    println("
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
    lint_dict.qsort();

    let mut max_key = 0;
    for &(_, name) in lint_dict.iter() {
        max_key = num::max(name.len(), max_key);
    }
    fn padded(max: uint, s: &str) -> ~str {
        str::from_bytes(vec::from_elem(max - s.len(), ' ' as u8)) + s
    }
    println("\nAvailable lint checks:\n");
    printfln!("    %s  %7.7s  %s",
              padded(max_key, "name"), "default", "meaning");
    printfln!("    %s  %7.7s  %s\n",
              padded(max_key, "----"), "-------", "-------");
    for (spec, name) in lint_dict.move_iter() {
        let name = name.replace("_", "-");
        printfln!("    %s  %7.7s  %s",
                  padded(max_key, name),
                  lint::level_to_str(spec.default),
                  spec.desc);
    }
    io::println("");
}

pub fn describe_debug_flags() {
    println("\nAvailable debug options:\n");
    let r = session::debugging_opts_map();
    for tuple in r.iter() {
        match *tuple {
            (ref name, ref desc, _) => {
                printfln!("    -Z %-20s -- %s", *name, *desc);
            }
        }
    }
}

pub fn run_compiler(args: &~[~str], demitter: diagnostic::Emitter) {
    // Don't display log spew by default. Can override with RUST_LOG.
    ::std::logging::console_off();

    let mut args = (*args).clone();
    let binary = args.shift().to_managed();

    if args.is_empty() { usage(binary); return; }

    let matches =
        &match getopts::groups::getopts(args, optgroups()) {
          Ok(m) => m,
          Err(f) => {
            early_error(demitter, getopts::fail_str(f));
          }
        };

    if opt_present(matches, "h") || opt_present(matches, "help") {
        usage(binary);
        return;
    }

    // Display the available lint options if "-W help" or only "-W" is given.
    let lint_flags = vec::append(getopts::opt_strs(matches, "W"),
                                 getopts::opt_strs(matches, "warn"));

    let show_lint_options = lint_flags.iter().any(|x| x == &~"help") ||
        (opt_present(matches, "W") && lint_flags.is_empty());

    if show_lint_options {
        describe_warnings();
        return;
    }

    let r = getopts::opt_strs(matches, "Z");
    if r.iter().any(|x| x == &~"help") {
        describe_debug_flags();
        return;
    }

    if getopts::opt_maybe_str(matches, "passes") == Some(~"list") {
        back::passes::list_passes();
        return;
    }

    if opt_present(matches, "v") || opt_present(matches, "version") {
        version(binary);
        return;
    }
    let input = match matches.free.len() {
      0u => early_error(demitter, ~"no input filename given"),
      1u => {
        let ifile = matches.free[0].as_slice();
        if "-" == ifile {
            let src = str::from_bytes(io::stdin().read_whole_stream());
            str_input(src.to_managed())
        } else {
            file_input(Path(ifile))
        }
      }
      _ => early_error(demitter, ~"multiple input filenames provided")
    };

    let sopts = build_session_options(binary, matches, demitter);
    let sess = build_session(sopts, demitter);
    let odir = getopts::opt_maybe_str(matches, "out-dir").map_move(|o| Path(o));
    let ofile = getopts::opt_maybe_str(matches, "o").map_move(|o| Path(o));
    let cfg = build_configuration(sess, binary, &input);
    let pretty = do getopts::opt_default(matches, "pretty", "normal").map_move |a| {
        parse_pretty(sess, a)
    };
    match pretty {
      Some::<pp_mode>(ppm) => {
        pretty_print_input(sess, cfg, &input, ppm);
        return;
      }
      None::<pp_mode> => {/* continue */ }
    }
    let ls = opt_present(matches, "ls");
    if ls {
        match input {
          file_input(ref ifile) => {
            list_metadata(sess, &(*ifile), io::stdout());
          }
          str_input(_) => {
            early_error(demitter, ~"can not list metadata for stdin");
          }
        }
        return;
    }

    compile_input(sess, cfg, &input, &odir, &ofile);
}

#[deriving(Eq)]
pub enum monitor_msg {
    fatal,
    done,
}

/*
This is a sanity check that any failure of the compiler is performed
through the diagnostic module and reported properly - we shouldn't be calling
plain-old-fail on any execution path that might be taken. Since we have
console logging off by default, hitting a plain fail statement would make the
compiler silently exit, which would be terrible.

This method wraps the compiler in a subtask and injects a function into the
diagnostic emitter which records when we hit a fatal error. If the task
fails without recording a fatal error then we've encountered a compiler
bug and need to present an error.
*/
pub fn monitor(f: ~fn(diagnostic::Emitter)) {
    use std::comm::*;

    // XXX: This is a hack for newsched since it doesn't support split stacks.
    // rustc needs a lot of stack!
    static STACK_SIZE: uint = 6000000;

    let (p, ch) = stream();
    let ch = SharedChan::new(ch);
    let ch_capture = ch.clone();
    let mut task_builder = task::task();
    task_builder.supervised();
    task_builder.opts.stack_size = Some(STACK_SIZE);
    match do task_builder.try {
        let ch = ch_capture.clone();
        let ch_capture = ch.clone();
        // The 'diagnostics emitter'. Every error, warning, etc. should
        // go through this function.
        let demitter: @fn(Option<(@codemap::CodeMap, codemap::span)>,
                          &str,
                          diagnostic::level) =
                          |cmsp, msg, lvl| {
            if lvl == diagnostic::fatal {
                ch_capture.send(fatal);
            }
            diagnostic::emit(cmsp, msg, lvl);
        };

        struct finally {
            ch: SharedChan<monitor_msg>,
        }

        impl Drop for finally {
            fn drop(&self) { self.ch.send(done); }
        }

        let _finally = finally { ch: ch };

        f(demitter);

        // Due reasons explain in #7732, if there was a jit execution context it
        // must be consumed and passed along to our parent task.
        back::link::jit::consume_engine()
    } {
        result::Ok(_) => { /* fallthrough */ }
        result::Err(_) => {
            // Task failed without emitting a fatal diagnostic
            if p.recv() == done {
                diagnostic::emit(
                    None,
                    diagnostic::ice_msg("unexpected failure"),
                    diagnostic::error);

                let xs = [
                    ~"the compiler hit an unexpected failure path. \
                     this is a bug",
                    ~"try running with RUST_LOG=rustc=1,::rt::backtrace \
                     to get further details and report the results \
                     to github.com/mozilla/rust/issues"
                ];
                for note in xs.iter() {
                    diagnostic::emit(None, *note, diagnostic::note)
                }
            }
            // Fail so the process returns a failure code
            fail!();
        }
    }
}

pub fn main() {
    let args = os::args();
    do monitor |demitter| {
        run_compiler(&args, demitter);
    }
}
