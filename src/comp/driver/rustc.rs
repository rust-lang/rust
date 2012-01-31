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
    io::stdout().write_str(#fmt["Usage: %s [options] <input>\n", argv0] +
                               "
Options:

    --bin              Compile an executable crate (default)
    -c                 Compile and assemble, but do not link
    --cfg <cfgspec>    Configure the compilation environment
    --emit-llvm        Produce an LLVM bitcode file
    -g                 Produce debug info
    --gc               Garbage collect shared data (experimental/temporary)
    -h --help          Display this message
    -L <path>          Add a directory to the library search path
    --lib              Compile a library crate
    --ls               List the symbols defined by a compiled library crate
    --no-asm-comments  Do not add comments into the assembly source
    --no-lint-ctypes   Suppress warnings for possibly incorrect ctype usage
    --no-trans         Run all passes except translation; no output
    --no-verify        Suppress LLVM verification step (slight speedup)
                       (see http://llvm.org/docs/Passes.html for detail)
    -O                 Equivalent to --opt-level=2
    -o <filename>      Write output to <filename>
    --opt-level <lvl>  Optimize with possible levels 0-3
    --out-dir <dir>    Write output to compiler-chosen filename in <dir>
    --parse-only       Parse only; do not compile, assemble, or link
    --pretty [type]    Pretty-print the input instead of compiling;
                       valid types are: normal (un-annotated source), 
                       expanded (crates expanded), typed (crates expanded,
                       with type annotations), or identified (fully
                       parenthesized, AST nodes and blocks with IDs)
    -S                 Compile only; do not assemble or link
    --save-temps       Write intermediate files (.bc, .opt.bc, .o)
                       in addition to normal output
    --static           Use or produce static libraries or binaries
    --stats            Print compilation statistics
    --sysroot <path>   Override the system root
    --test             Build a test harness
    --target <triple>  Target cpu-manufacturer-kernel[-os] to compile for
                       (default: host triple)
                       (see http://sources.redhat.com/autobook/autobook/
                       autobook_17.html for detail)

    --time-passes      Time the individual phases of the compiler
    --time-llvm-passes Time the individual phases of the LLVM backend
    -v --version       Print version info and exit
    --warn-unused-imports
                       Warn about unnecessary imports

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
      none::<pp_mode> {/* continue */ }
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
    enum monitor_msg {
        fatal,
        done,
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
