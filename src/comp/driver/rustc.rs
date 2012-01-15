use std;
use rustc;

// -*- rust -*-
import core::{option, str, vec, result};
import result::{ok, err};
import std::{io, getopts};
import io::writer_util;
import option::{some, none};
import getopts::{opt_present};
import rustc::driver::driver::*;
import rustc::syntax::codemap;
import rustc::driver::diagnostic;

fn version(argv0: str) {
    let vers = "unknown version";
    let env_vers = #env["CFG_VERSION"];
    if str::byte_len(env_vers) != 0u { vers = env_vers; }
    io::stdout().write_str(#fmt["%s %s\n", argv0, vers]);
    io::stdout().write_str(#fmt["host: %s\n", host_triple()]);
}

fn usage(argv0: str) {
    io::stdout().write_str(#fmt["usage: %s [options] <input>\n", argv0] +
                               "
options:

    -h --help          display this message
    -v --version       print version info and exit

    -o <filename>      write output to <filename>
    --out-dir <dir>    write output to compiler-chosen filename in <dir>
    --lib              compile a library crate
    --bin              compile an executable crate (default)
    --static           use or produce static libraries
    --no-core          omit the 'core' library (used and imported by default)
    --pretty [type]    pretty-print the input instead of compiling
    --ls               list the symbols defined by a crate file
    -L <path>          add a directory to the library search path
    --no-verify        suppress LLVM verification step (slight speedup)
    --parse-only       parse only; do not compile, assemble, or link
    --no-trans         run all passes except translation; no output
    -g                 produce debug info
    --opt-level <lvl>  optimize with possible levels 0-3
    -O                 equivalent to --opt-level=2
    -S                 compile only; do not assemble or link
    --no-asm-comments  do not add comments into the assembly source
    -c                 compile and assemble, but do not link
    --emit-llvm        produce an LLVM bitcode file
    --save-temps       write intermediate files in addition to normal output
    --stats            gather and report various compilation statistics
    --cfg <cfgspec>    configure the compilation environment
    --time-passes      time the individual phases of the compiler
    --time-llvm-passes time the individual phases of the LLVM backend
    --sysroot <path>   override the system root
    --target <triple>  target to compile for (default: host triple)
    --test             build test harness
    --gc               garbage collect shared data (experimental/temporary)
    --warn-unused-imports
                       warn about unnecessary imports

");
}

fn run_compiler(args: [str], demitter: diagnostic::emitter) {
    // Don't display log spew by default. Can override with RUST_LOG.
    logging::console_off();

    let args = args, binary = vec::shift(args);

    if vec::len(args) == 0u { usage(binary); ret; }

    let match =
        alt getopts::getopts(args, opts()) {
          ok(m) { m }
          err(f) {
            early_error(demitter, getopts::fail_str(f))
          }
        };
    if opt_present(match, "h") || opt_present(match, "help") {
        usage(binary);
        ret;
    }
    if opt_present(match, "v") || opt_present(match, "version") {
        version(binary);
        ret;
    }
    let ifile = alt vec::len(match.free) {
      0u { early_error(demitter, "No input filename given.") }
      1u { match.free[0] }
      _ { early_error(demitter, "Multiple input filenames provided.") }
    };

    let sopts = build_session_options(match, demitter);
    let sess = build_session(sopts, ifile, demitter);
    let odir = getopts::opt_maybe_str(match, "out-dir");
    let ofile = getopts::opt_maybe_str(match, "o");
    let cfg = build_configuration(sess, binary, ifile);
    let pretty =
        option::map(getopts::opt_default(match, "pretty",
                                         "normal"),
                    bind parse_pretty(sess, _));
    alt pretty {
      some::<pp_mode>(ppm) { pretty_print_input(sess, cfg, ifile, ppm); ret; }
      none::<pp_mode>. {/* continue */ }
    }
    let ls = opt_present(match, "ls");
    if ls {
        list_metadata(sess, ifile, io::stdout());
        ret;
    }

    compile_input(sess, cfg, ifile, odir, ofile);
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
fn monitor(f: fn~(diagnostic::emitter)) {
    tag monitor_msg {
        fatal;
        done;
    };

    let p = comm::port();
    let ch = comm::chan(p);

    alt task::try  {||

        task::unsupervise();

        // The 'diagnostics emitter'. Every error, warning, etc. should
        // go through this function.
        let demitter = fn@(cmsp: option<(codemap::codemap, codemap::span)>,
                           msg: str, lvl: diagnostic::level) {
            if lvl == diagnostic::fatal {
                comm::send(ch, fatal);
            }
            diagnostic::emit(cmsp, msg, lvl);
        };

        resource finally(ch: comm::chan<monitor_msg>) {
            comm::send(ch, done);
        }

        let _finally = finally(ch);

        f(demitter)
    } {
        result::ok(_) { /* fallthrough */ }
        result::err(_) {
            // Task failed without emitting a fatal diagnostic
            if comm::recv(p) == done {
                diagnostic::emit(
                    none,
                    diagnostic::ice_msg("unexpected failure"),
                    diagnostic::error);
                let note = "The compiler hit an unexpected failure path. \
                            This is a bug. Try running with \
                            RUST_LOG=rustc=0,::rt::backtrace \
                            to get further details and report the results \
                            to github.com/mozilla/rust/issues";
                diagnostic::emit(none, note, diagnostic::note);
            }
            // Fail so the process returns a failure code
            fail;
        }
    }
}

fn main(args: [str]) {
    monitor {|demitter|
        run_compiler(args, demitter);
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
