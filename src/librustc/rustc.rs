// -*- rust -*-
use result::{Ok, Err};
use io::ReaderUtil;
use std::getopts;
use std::map::HashMap;
use getopts::{opt_present};
use getopts::groups;
use syntax::codemap;
use syntax::diagnostic;
use driver::driver::{host_triple, optgroups, early_error,
                     str_input, file_input, build_session_options,
                     build_session, build_configuration, parse_pretty,
                     pp_mode, pretty_print_input, list_metadata,
                     compile_input};
use driver::session;
use middle::lint;

fn version(argv0: &str) {
    let mut vers = ~"unknown version";
    let env_vers = env!("CFG_VERSION");
    if env_vers.len() != 0 { vers = env_vers; }
    io::println(fmt!("%s %s", argv0, vers));
    io::println(fmt!("host: %s", host_triple()));
}

fn usage(argv0: &str) {
    let message = fmt!("Usage: %s [OPTIONS] INPUT", argv0);
    io::println(groups::usage(message, optgroups()) +
                ~"Additional help:
    -W help             Print 'lint' options and default settings
    -Z help             Print internal options for debugging rustc
");
}

fn describe_warnings() {
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
        let (name, desc, _) = *pair;
        io::println(fmt!("    -Z %-20s -- %s", name, desc));
    }
}

fn run_compiler(args: &~[~str], demitter: diagnostic::emitter) {
    // Don't display log spew by default. Can override with RUST_LOG.
    logging::console_off();

    let mut args = *args;
    let binary = args.shift();

    if args.is_empty() { usage(binary); return; }

    let matches =
        match getopts::groups::getopts(args, optgroups()) {
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
    if lint_flags.contains(&~"help") {
        describe_warnings();
        return;
    }

    if getopts::opt_strs(matches, ~"Z").contains(&~"help") {
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
    let odir = odir.map(|o| Path(*o));
    let ofile = getopts::opt_maybe_str(matches, ~"o");
    let ofile = ofile.map(|o| Path(*o));
    let cfg = build_configuration(sess, binary, input);
    let pretty =
        option::map(&getopts::opt_default(matches, ~"pretty",
                                         ~"normal"),
                    |a| parse_pretty(sess, *a) );
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
    #[cfg(stage0)]
    pure fn eq(other: &monitor_msg) -> bool {
        (self as uint) == ((*other) as uint)
    }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn eq(&self, other: &monitor_msg) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    #[cfg(stage0)]
    pure fn ne(other: &monitor_msg) -> bool { !self.eq(other) }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn ne(&self, other: &monitor_msg) -> bool { !(*self).eq(other) }
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
    let ch = comm::Chan(&p);

    match do task::try |move f| {

        // The 'diagnostics emitter'. Every error, warning, etc. should
        // go through this function.
        let demitter = fn@(cmsp: Option<(@codemap::CodeMap, codemap::span)>,
                           msg: &str, lvl: diagnostic::level) {
            if lvl == diagnostic::fatal {
                comm::send(ch, fatal);
            }
            diagnostic::emit(cmsp, msg, lvl);
        };

        struct finally {
            ch: comm::Chan<monitor_msg>,
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
                    ~"try running with RUST_LOG=rustc=1,::rt::backtrace \
                     to get further details and report the results \
                     to github.com/mozilla/rust/issues"
                ].each |note| {
                    diagnostic::emit(None, *note, diagnostic::note)
                }
            }
            // Fail so the process returns a failure code
            fail;
        }
    }
}

fn main() {
    let mut args = os::args();
    do monitor |move args, demitter| {
        run_compiler(&args, demitter);
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
