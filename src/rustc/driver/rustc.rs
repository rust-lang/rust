#[no_core];
#[allow(vecs_implicitly_copyable)];

use core(vers = "0.4");
use std(vers = "0.4");
use rustc(vers = "0.4");
use syntax(vers = "0.4");

use core::*;

// -*- rust -*-
use result::{Ok, Err};
use std::getopts;
use std::map::hashmap;
use getopts::{opt_present};
use rustc::driver::driver::*;
use syntax::codemap;
use syntax::diagnostic;
use rustc::driver::session;
use rustc::middle::lint;

fn version(argv0: ~str) {
    let mut vers = ~"unknown version";
    let env_vers = env!("CFG_VERSION");
    if str::len(env_vers) != 0u { vers = env_vers; }
    io::println(fmt!("%s %s", argv0, vers));
    io::println(fmt!("host: %s", host_triple()));
}

fn usage(argv0: ~str) {
    io::println(fmt!("Usage: %s [options] <input>\n", argv0) +
                 ~"
Options:

    --bin              Compile an executable crate (default)
    -c                 Compile and assemble, but do not link
    --cfg <cfgspec>    Configure the compilation environment
    --emit-llvm        Produce an LLVM bitcode file
    -g                 Produce debug info (experimental)
    --gc               Garbage collect shared data (experimental/temporary)
    -h --help          Display this message
    -L <path>          Add a directory to the library search path
    --lib              Compile a library crate
    --ls               List the symbols defined by a compiled library crate
    --jit              Execute using JIT (experimental)
    --no-trans         Run all passes except translation; no output
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

    -(W|A|D|F) help    Print available 'lint' checks and default settings

    -W <foo>           warn about <foo> by default
    -A <foo>           allow <foo> by default
    -D <foo>           deny <foo> by default
    -F <foo>           forbid <foo> (deny, and deny all overrides)

    -Z help            list internal options for debugging rustc

    -v --version       Print version info and exit
");
}

fn describe_warnings() {
    let lint_dict = lint::get_lint_dict();
    let mut max_key = 0;
    for lint_dict.each_key |k| { max_key = uint::max(k.len(), max_key); }
    fn padded(max: uint, s: ~str) -> ~str {
        str::from_bytes(vec::from_elem(max - s.len(), ' ' as u8)) + s
    }
    io::println(fmt!("\nAvailable lint checks:\n"));
    io::println(fmt!("    %s  %7.7s  %s",
                     padded(max_key, ~"name"), ~"default", ~"meaning"));
    io::println(fmt!("    %s  %7.7s  %s\n",
                     padded(max_key, ~"----"), ~"-------", ~"-------"));
    for lint_dict.each |k, v| {
        let k = str::replace(k, ~"_", ~"-");
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
    io::println(~"");
}

fn describe_debug_flags() {
    io::println(fmt!("\nAvailable debug options:\n"));
    for session::debugging_opts_map().each |pair| {
        let (name, desc, _) = pair;
        io::println(fmt!("    -Z%-20s -- %s", name, desc));
    }
}

fn run_compiler(args: ~[~str], demitter: diagnostic::emitter) {
    // Don't display log spew by default. Can override with RUST_LOG.
    logging::console_off();

    let mut args = args;
    let binary = vec::shift(args);

    if vec::len(args) == 0u { usage(binary); return; }

    let matches =
        match getopts::getopts(args, opts()) {
          Ok(m) => m,
          Err(f) => {
            early_error(demitter, getopts::fail_str(f))
          }
        };

    if opt_present(matches, ~"h") || opt_present(matches, ~"help") {
        usage(binary);
        return;
    }

    let lint_flags = vec::append(getopts::opt_strs(matches, ~"W"),
                                 getopts::opt_strs(matches, ~"warn"));
    if lint_flags.contains(~"help") {
        describe_warnings();
        return;
    }

    if getopts::opt_strs(matches, ~"Z").contains(~"help") {
        describe_debug_flags();
        return;
    }

    if opt_present(matches, ~"v") || opt_present(matches, ~"version") {
        version(binary);
        return;
    }
    let input = match vec::len(matches.free) {
      0u => early_error(demitter, ~"no input filename given"),
      1u => {
        let ifile = matches.free[0];
        if ifile == ~"-" {
            let src = str::from_bytes(io::stdin().read_whole_stream());
            str_input(src)
        } else {
            file_input(Path(ifile))
        }
      }
      _ => early_error(demitter, ~"multiple input filenames provided")
    };

    let sopts = build_session_options(binary, matches, demitter);
    let sess = build_session(sopts, demitter);
    let odir = getopts::opt_maybe_str(matches, ~"out-dir");
    let odir = option::map(odir, |o| Path(o));
    let ofile = getopts::opt_maybe_str(matches, ~"o");
    let ofile = option::map(ofile, |o| Path(o));
    let cfg = build_configuration(sess, binary, input);
    let pretty =
        option::map(getopts::opt_default(matches, ~"pretty",
                                         ~"normal"),
                    |a| parse_pretty(sess, a) );
    match pretty {
      Some::<pp_mode>(ppm) => {
        pretty_print_input(sess, cfg, input, ppm);
        return;
      }
      None::<pp_mode> => {/* continue */ }
    }
    let ls = opt_present(matches, ~"ls");
    if ls {
        match input {
          file_input(ifile) => {
            list_metadata(sess, &ifile, io::stdout());
          }
          str_input(_) => {
            early_error(demitter, ~"can not list metadata for stdin");
          }
        }
        return;
    }

    compile_input(sess, cfg, input, &odir, &ofile);
}

enum monitor_msg {
    fatal,
    done,
}

impl monitor_msg : cmp::Eq {
    pure fn eq(&&other: monitor_msg) -> bool {
        (self as uint) == (other as uint)
    }
    pure fn ne(&&other: monitor_msg) -> bool { !self.eq(other) }
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
fn monitor(+f: fn~(diagnostic::emitter)) {
    let p = comm::Port();
    let ch = comm::Chan(p);

    match do task::try  {

        // The 'diagnostics emitter'. Every error, warning, etc. should
        // go through this function.
        let demitter = fn@(cmsp: Option<(codemap::codemap, codemap::span)>,
                           msg: ~str, lvl: diagnostic::level) {
            if lvl == diagnostic::fatal {
                comm::send(ch, fatal);
            }
            diagnostic::emit(cmsp, msg, lvl);
        };

        struct finally {
            let ch: comm::Chan<monitor_msg>;
            drop { comm::send(self.ch, done); }
        }

        let _finally = finally { ch: ch };

        f(demitter)
    } {
        result::Ok(_) => { /* fallthrough */ }
        result::Err(_) => {
            // Task failed without emitting a fatal diagnostic
            if comm::recv(p) == done {
                diagnostic::emit(
                    None,
                    diagnostic::ice_msg(~"unexpected failure"),
                    diagnostic::error);

                for [
                    ~"the compiler hit an unexpected failure path. \
                     this is a bug",
                    ~"try running with RUST_LOG=rustc=0,::rt::backtrace \
                     to get further details and report the results \
                     to github.com/mozilla/rust/issues"
                ]/_.each |note| {
                    diagnostic::emit(None, note, diagnostic::note)
                }
            }
            // Fail so the process returns a failure code
            fail;
        }
    }
}

fn main(args: ~[~str]) {
    do monitor |demitter| {
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
