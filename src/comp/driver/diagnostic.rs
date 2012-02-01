import std::{io, term};
import io::writer_util;
import syntax::codemap;
import codemap::span;

export emitter, emit;
export level, fatal, error, warning, note;
export span_handler, handler, mk_span_handler, mk_handler;
export codemap_span_handler, codemap_handler;
export ice_msg;

type emitter = fn@(cmsp: option<(codemap::codemap, span)>,
                   msg: str, lvl: level);


iface span_handler {
    fn span_fatal(sp: span, msg: str) -> !;
    fn span_err(sp: span, msg: str);
    fn span_warn(sp: span, msg: str);
    fn span_note(sp: span, msg: str);
    fn span_bug(sp: span, msg: str) -> !;
    fn span_unimpl(sp: span, msg: str) -> !;
    fn handler() -> handler;
}

iface handler {
    fn fatal(msg: str) -> !;
    fn err(msg: str);
    fn bump_err_count();
    fn has_errors() -> bool;
    fn abort_if_errors();
    fn warn(msg: str);
    fn note(msg: str);
    fn bug(msg: str) -> !;
    fn unimpl(msg: str) -> !;
    fn emit(cmsp: option<(codemap::codemap, span)>, msg: str, lvl: level);
}

type handler_t = @{
    mutable err_count: uint,
    _emit: emitter
};

type codemap_t = @{
    handler: handler,
    cm: codemap::codemap
};

impl codemap_span_handler of span_handler for codemap_t {
    fn span_fatal(sp: span, msg: str) -> ! {
        self.handler.emit(some((self.cm, sp)), msg, fatal);
        fail;
    }
    fn span_err(sp: span, msg: str) {
        self.handler.emit(some((self.cm, sp)), msg, error);
        self.handler.bump_err_count();
    }
    fn span_warn(sp: span, msg: str) {
        self.handler.emit(some((self.cm, sp)), msg, warning);
    }
    fn span_note(sp: span, msg: str) {
        self.handler.emit(some((self.cm, sp)), msg, note);
    }
    fn span_bug(sp: span, msg: str) -> ! {
        self.span_fatal(sp, ice_msg(msg));
    }
    fn span_unimpl(sp: span, msg: str) -> ! {
        self.span_bug(sp, "unimplemented " + msg);
    }
    fn handler() -> handler {
        self.handler
    }
}

impl codemap_handler of handler for handler_t {
    fn fatal(msg: str) -> ! {
        self._emit(none, msg, fatal);
        fail;
    }
    fn err(msg: str) {
        self._emit(none, msg, error);
        self.bump_err_count();
    }
    fn bump_err_count() {
        self.err_count += 1u;
    }
    fn has_errors() -> bool { self.err_count > 0u }
    fn abort_if_errors() {
        if self.err_count > 0u {
            self.fatal("aborting due to previous errors");
        }
    }
    fn warn(msg: str) {
        self._emit(none, msg, warning);
    }
    fn note(msg: str) {
        self._emit(none, msg, note);
    }
    fn bug(msg: str) -> ! {
        self.fatal(ice_msg(msg));
    }
    fn unimpl(msg: str) -> ! { self.bug("unimplemented " + msg); }
    fn emit(cmsp: option<(codemap::codemap, span)>, msg: str, lvl: level) {
        self._emit(cmsp, msg, lvl);
    }
}

fn ice_msg(msg: str) -> str {
    #fmt["internal compiler error %s", msg]
}

fn mk_span_handler(handler: handler, cm: codemap::codemap) -> span_handler {
    @{ handler: handler, cm: cm } as span_handler
}

fn mk_handler(emitter: option<emitter>) -> handler {

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
        mutable err_count: 0u,
        _emit: emit
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

    let fm = lines.file;

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
        let s = codemap::get_line(fm, line as int);
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
