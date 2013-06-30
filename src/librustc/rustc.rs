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
       vers = "0.7-pre",
       uuid = "0ce89b41-2f92-459e-bbc1-8f5fe32f16cf",
       url = "https://github.com/mozilla/rust/tree/master/src/rustc")];

#[comment = "The Rust compiler"];
#[license = "MIT/ASL2"];
#[crate_type = "lib"];

#[allow(non_implicitly_copyable_typarams)];
#[allow(non_camel_case_types)];
#[deny(deprecated_pattern)];

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
use std::os;
use std::result;
use std::str;
use std::task;
use std::uint;
use std::vec;
use extra::getopts::{groups, opt_present};
use extra::getopts;
use syntax::codemap;
use syntax::diagnostic;

pub mod middle {
    #[path = "trans/mod.rs"]
    pub mod trans;
    pub mod ty;
    pub mod subst;
    pub mod resolve;
    #[path = "typeck/mod.rs"]
    pub mod typeck;
    pub mod check_loop;
    pub mod check_match;
    pub mod check_const;
    pub mod lint;
    #[path = "borrowck/mod.rs"]
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

#[path = "metadata/mod.rs"]
pub mod metadata;

#[path = "driver/mod.rs"]
pub mod driver;

pub mod util {
    pub mod common;
    pub mod ppaux;
    pub mod enum_set;
}

pub mod lib {
    pub mod llvm;
}

// A curious inner module that allows ::std::foo to be available in here for
// macros.
/*
mod std {
    pub use std::cmp;
    pub use std::os;
    pub use std::str;
    pub use std::sys;
    pub use std::to_bytes;
    pub use std::unstable;
    pub use extra::serialize;
}
*/

pub fn version(argv0: &str) {
    let mut vers = ~"unknown version";
    let env_vers = env!("CFG_VERSION");
    if env_vers.len() != 0 { vers = env_vers.to_owned(); }
    io::println(fmt!("%s %s", argv0, vers));
    io::println(fmt!("host: %s", host_triple()));
}

pub fn usage(argv0: &str) {
    let message = fmt!("Usage: %s [OPTIONS] INPUT", argv0);
    io::println(fmt!("%s\
Additional help:
    -W help             Print 'lint' options and default settings
    -Z help             Print internal options for debugging rustc\n",
                     groups::usage(message, optgroups())));
}

pub fn describe_warnings() {
    io::println(fmt!("
Available lint options:
    -W <foo>           Warn about <foo>
    -A <foo>           Allow <foo>
    -D <foo>           Deny <foo>
    -F <foo>           Forbid <foo> (deny, and deny all overrides)
"));

    let lint_dict = lint::get_lint_dict();
    let mut max_key = 0;
    for lint_dict.each_key |k| { max_key = uint::max(k.len(), max_key); }
    fn padded(max: uint, s: &str) -> ~str {
        str::from_bytes(vec::from_elem(max - s.len(), ' ' as u8)) + s
    }
    io::println(fmt!("\nAvailable lint checks:\n"));
    io::println(fmt!("    %s  %7.7s  %s",
                     padded(max_key, "name"), "default", "meaning"));
    io::println(fmt!("    %s  %7.7s  %s\n",
                     padded(max_key, "----"), "-------", "-------"));
    for lint_dict.iter().advance |(k, v)| {
        let k = k.replace("_", "-");
        io::println(fmt!("    %s  %7.7s  %s",
                         padded(max_key, k),
                         match v.default {
                             lint::allow => ~"allow",
                             lint::warn => ~"warn",
                             lint::deny => ~"deny",
                             lint::forbid => ~"forbid"
                         },
                         v.desc));
    }
    io::println("");
}

pub fn describe_debug_flags() {
    io::println(fmt!("\nAvailable debug options:\n"));
    let r = session::debugging_opts_map();
    for r.iter().advance |pair| {
        let (name, desc, _) = /*bad*/copy *pair;
        io::println(fmt!("    -Z %-20s -- %s", name, desc));
    }
}

pub fn run_compiler(args: &~[~str], demitter: diagnostic::Emitter) {
    // Don't display log spew by default. Can override with RUST_LOG.
    ::std::logging::console_off();

    let mut args = /*bad*/copy *args;
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

    let show_lint_options = lint_flags.iter().any_(|x| x == &~"help") ||
        (opt_present(matches, "W") && lint_flags.is_empty());

    if show_lint_options {
        describe_warnings();
        return;
    }

    let r = getopts::opt_strs(matches, "Z");
    if r.iter().any_(|x| x == &~"help") {
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
    let odir = getopts::opt_maybe_str(matches, "out-dir");
    let odir = odir.map(|o| Path(*o));
    let ofile = getopts::opt_maybe_str(matches, "o");
    let ofile = ofile.map(|o| Path(*o));
    let cfg = build_configuration(sess, binary, &input);
    let pretty = getopts::opt_default(matches, "pretty", "normal").map(
                    |a| parse_pretty(sess, *a));
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
    let (p, ch) = stream();
    let ch = SharedChan::new(ch);
    let ch_capture = ch.clone();
    match do task::try || {
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

        f(demitter)
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
                for xs.iter().advance |note| {
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
