use std;
use rustc;

// -*- rust -*-
import core::{option, str, vec, result};
import result::{ok, err};
import std::{io, getopts};
import option::{some, none};
import getopts::{opt_present};
import rustc::driver::driver::*;

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

fn main(args: [str]) {
    let args = args, binary = vec::shift(args);
    let match =
        alt getopts::getopts(args, opts()) {
          ok(m) { m }
          err(f) {
            early_error(getopts::fail_str(f))
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
      0u { early_error("No input filename given.") }
      1u { match.free[0] }
      _ { early_error("Multiple input filenames provided.") }
    };

    let sopts = build_session_options(match);
    let sess = build_session(sopts, ifile);
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

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
