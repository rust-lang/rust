import std::{io, term};
import io::writer_util;
import syntax::codemap;
import codemap::span;

export emitter, emit;
export level, fatal, error, warning, note;
export handler, mk_handler;
export ice_msg;

type emitter = fn@(cmsp: option<(codemap::codemap, span)>,
                   msg: str, lvl: level);


iface handler {
    fn span_fatal(sp: span, msg: str) -> !;
    fn fatal(msg: str) -> !;
    fn span_err(sp: span, msg: str);
    fn err(msg: str);
    fn has_errors() -> bool;
    fn abort_if_errors();
    fn span_warn(sp: span, msg: str);
    fn warn(msg: str);
    fn span_note(sp: span, msg: str);
    fn note(msg: str);
    fn span_bug(sp: span, msg: str) -> !;
    fn bug(msg: str) -> !;
    fn span_unimpl(sp: span, msg: str) -> !;
    fn unimpl(msg: str) -> !;
}

type codemap_t = @{
    cm: codemap::codemap,
    mutable err_count: uint,
    emit: emitter
};

impl codemap_handler of handler for codemap_t {
    fn span_fatal(sp: span, msg: str) -> ! {
        self.emit(some((self.cm, sp)), msg, fatal);
        fail;
    }
    fn fatal(msg: str) -> ! {
        self.emit(none, msg, fatal);
        fail;
    }
    fn span_err(sp: span, msg: str) {
        self.emit(some((self.cm, sp)), msg, error);
        self.err_count += 1u;
    }
    fn err(msg: str) {
        self.emit(none, msg, error);
        self.err_count += 1u;
    }
    fn has_errors() -> bool { self.err_count > 0u }
    fn abort_if_errors() {
        if self.err_count > 0u {
            self.fatal("aborting due to previous errors");
        }
    }
    fn span_warn(sp: span, msg: str) {
        self.emit(some((self.cm, sp)), msg, warning);
    }
    fn warn(msg: str) {
        self.emit(none, msg, warning);
    }
    fn span_note(sp: span, msg: str) {
        self.emit(some((self.cm, sp)), msg, note);
    }
    fn note(msg: str) {
        self.emit(none, msg, note);
    }
    fn span_bug(sp: span, msg: str) -> ! {
        self.span_fatal(sp, ice_msg(msg));
    }
    fn bug(msg: str) -> ! {
        self.fatal(ice_msg(msg));
    }
    fn span_unimpl(sp: span, msg: str) -> ! {
        self.span_bug(sp, "unimplemented " + msg);
    }
    fn unimpl(msg: str) -> ! { self.bug("unimplemented " + msg); }
}

fn ice_msg(msg: str) -> str {
    #fmt["internal compiler error %s", msg]
}

fn mk_handler(cm: codemap::codemap,
              emitter: option<emitter>) -> handler {

    let emit = alt emitter {
      some(e) { e }
      none {
        let f = fn@(cmsp: option<(codemap::codemap, span)>,
            msg: str, t: level) {
            emit(cmsp, msg, t);
        };
        f
      }
    };

    @{
        cm: cm,
        mutable err_count: 0u,
        emit: emit
    } as handler
}

enum level {
    fatal,
    error,
    warning,
    note,
}

fn diagnosticstr(lvl: level) -> str {
    alt lvl {
      fatal { "error" }
      error { "error" }
      warning { "warning" }
      note { "note" }
    }
}

fn diagnosticcolor(lvl: level) -> u8 {
    alt lvl {
      fatal { term::color_bright_red }
      error { term::color_bright_red }
      warning { term::color_bright_yellow }
      note { term::color_bright_green }
    }
}

fn print_diagnostic(topic: str, lvl: level, msg: str) {
    if str::is_not_empty(topic) {
        io::stdout().write_str(#fmt["%s ", topic]);
    }
    if term::color_supported() {
        term::fg(io::stdout(), diagnosticcolor(lvl));
    }
    io::stdout().write_str(#fmt["%s:", diagnosticstr(lvl)]);
    if term::color_supported() {
        term::reset(io::stdout());
    }
    io::stdout().write_str(#fmt[" %s\n", msg]);
}

fn emit(cmsp: option<(codemap::codemap, span)>,
        msg: str, lvl: level) {
    alt cmsp {
      some((cm, sp)) {
        let ss = codemap::span_to_str(sp, cm);
        let lines = codemap::span_to_lines(sp, cm);
        print_diagnostic(ss, lvl, msg);
        highlight_lines(cm, sp, lines);
      }
      none {
        print_diagnostic("", lvl, msg);
      }
    }
}

fn highlight_lines(cm: codemap::codemap, sp: span,
                   lines: @codemap::file_lines) {

    // If we're not looking at a real file then we can't re-open it to
    // pull out the lines
    if lines.name == "-" { ret; }

    // FIXME: reading in the entire file is the worst possible way to
    //        get access to the necessary lines.
    let file = alt io::read_whole_file_str(lines.name) {
      result::ok(file) { file }
      result::err(e) {
        // Hard to report errors while reporting an error
        ret;
      }
    };
    let fm = codemap::get_filemap(cm, lines.name);

    // arbitrarily only print up to six lines of the error
    let max_lines = 6u;
    let elided = false;
    let display_lines = lines.lines;
    if vec::len(display_lines) > max_lines {
        display_lines = vec::slice(display_lines, 0u, max_lines);
        elided = true;
    }
    // Print the offending lines
    for line: uint in display_lines {
        io::stdout().write_str(#fmt["%s:%u ", fm.name, line + 1u]);
        let s = codemap::get_line(fm, line as int, file);
        if !str::ends_with(s, "\n") { s += "\n"; }
        io::stdout().write_str(s);
    }
    if elided {
        let last_line = display_lines[vec::len(display_lines) - 1u];
        let s = #fmt["%s:%u ", fm.name, last_line + 1u];
        let indent = str::char_len(s);
        let out = "";
        while indent > 0u { out += " "; indent -= 1u; }
        out += "...\n";
        io::stdout().write_str(out);
    }


    // If there's one line at fault we can easily point to the problem
    if vec::len(lines.lines) == 1u {
        let lo = codemap::lookup_char_pos(cm, sp.lo);
        let digits = 0u;
        let num = (lines.lines[0] + 1u) / 10u;

        // how many digits must be indent past?
        while num > 0u { num /= 10u; digits += 1u; }

        // indent past |name:## | and the 0-offset column location
        let left = str::char_len(fm.name) + digits + lo.col + 3u;
        let s = "";
        while left > 0u { str::push_char(s, ' '); left -= 1u; }

        s += "^";
        let hi = codemap::lookup_char_pos(cm, sp.hi);
        if hi.col != lo.col {
            // the ^ already takes up one space
            let width = hi.col - lo.col - 1u;
            while width > 0u { str::push_char(s, '~'); width -= 1u; }
        }
        io::stdout().write_str(s + "\n");
    }
}
