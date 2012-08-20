import std::term;
import io::WriterUtil;
import codemap::span;

export emitter, emit;
export level, fatal, error, warning, note;
export span_handler, handler, mk_span_handler, mk_handler;
export codemap_span_handler, codemap_handler;
export ice_msg;
export expect;

type emitter = fn@(cmsp: Option<(codemap::codemap, span)>,
                   msg: ~str, lvl: level);


trait span_handler {
    fn span_fatal(sp: span, msg: ~str) -> !;
    fn span_err(sp: span, msg: ~str);
    fn span_warn(sp: span, msg: ~str);
    fn span_note(sp: span, msg: ~str);
    fn span_bug(sp: span, msg: ~str) -> !;
    fn span_unimpl(sp: span, msg: ~str) -> !;
    fn handler() -> handler;
}

trait handler {
    fn fatal(msg: ~str) -> !;
    fn err(msg: ~str);
    fn bump_err_count();
    fn has_errors() -> bool;
    fn abort_if_errors();
    fn warn(msg: ~str);
    fn note(msg: ~str);
    fn bug(msg: ~str) -> !;
    fn unimpl(msg: ~str) -> !;
    fn emit(cmsp: Option<(codemap::codemap, span)>, msg: ~str, lvl: level);
}

type handler_t = @{
    mut err_count: uint,
    emit: emitter
};

type codemap_t = @{
    handler: handler,
    cm: codemap::codemap
};

impl codemap_t: span_handler {
    fn span_fatal(sp: span, msg: ~str) -> ! {
        self.handler.emit(Some((self.cm, sp)), msg, fatal);
        fail;
    }
    fn span_err(sp: span, msg: ~str) {
        self.handler.emit(Some((self.cm, sp)), msg, error);
        self.handler.bump_err_count();
    }
    fn span_warn(sp: span, msg: ~str) {
        self.handler.emit(Some((self.cm, sp)), msg, warning);
    }
    fn span_note(sp: span, msg: ~str) {
        self.handler.emit(Some((self.cm, sp)), msg, note);
    }
    fn span_bug(sp: span, msg: ~str) -> ! {
        self.span_fatal(sp, ice_msg(msg));
    }
    fn span_unimpl(sp: span, msg: ~str) -> ! {
        self.span_bug(sp, ~"unimplemented " + msg);
    }
    fn handler() -> handler {
        self.handler
    }
}

impl handler_t: handler {
    fn fatal(msg: ~str) -> ! {
        self.emit(None, msg, fatal);
        fail;
    }
    fn err(msg: ~str) {
        self.emit(None, msg, error);
        self.bump_err_count();
    }
    fn bump_err_count() {
        self.err_count += 1u;
    }
    fn has_errors() -> bool { self.err_count > 0u }
    fn abort_if_errors() {
        let s;
        match self.err_count {
          0u => return,
          1u => s = ~"aborting due to previous error",
          _  => {
            s = fmt!("aborting due to %u previous errors",
                     self.err_count);
          }
        }
        self.fatal(s);
    }
    fn warn(msg: ~str) {
        self.emit(None, msg, warning);
    }
    fn note(msg: ~str) {
        self.emit(None, msg, note);
    }
    fn bug(msg: ~str) -> ! {
        self.fatal(ice_msg(msg));
    }
    fn unimpl(msg: ~str) -> ! { self.bug(~"unimplemented " + msg); }
    fn emit(cmsp: Option<(codemap::codemap, span)>, msg: ~str, lvl: level) {
        self.emit(cmsp, msg, lvl);
    }
}

fn ice_msg(msg: ~str) -> ~str {
    fmt!("internal compiler error: %s", msg)
}

fn mk_span_handler(handler: handler, cm: codemap::codemap) -> span_handler {
    @{ handler: handler, cm: cm } as span_handler
}

fn mk_handler(emitter: Option<emitter>) -> handler {

    let emit = match emitter {
      Some(e) => e,
      None => {
        let f = fn@(cmsp: Option<(codemap::codemap, span)>,
            msg: ~str, t: level) {
            emit(cmsp, msg, t);
        };
        f
      }
    };

    @{
        mut err_count: 0u,
        emit: emit
    } as handler
}

enum level {
    fatal,
    error,
    warning,
    note,
}

fn diagnosticstr(lvl: level) -> ~str {
    match lvl {
      fatal => ~"error",
      error => ~"error",
      warning => ~"warning",
      note => ~"note"
    }
}

fn diagnosticcolor(lvl: level) -> u8 {
    match lvl {
      fatal => term::color_bright_red,
      error => term::color_bright_red,
      warning => term::color_bright_yellow,
      note => term::color_bright_green
    }
}

fn print_diagnostic(topic: ~str, lvl: level, msg: ~str) {
    let use_color = term::color_supported() &&
        io::stderr().get_type() == io::Screen;
    if str::is_not_empty(topic) {
        io::stderr().write_str(fmt!("%s ", topic));
    }
    if use_color {
        term::fg(io::stderr(), diagnosticcolor(lvl));
    }
    io::stderr().write_str(fmt!("%s:", diagnosticstr(lvl)));
    if use_color {
        term::reset(io::stderr());
    }
    io::stderr().write_str(fmt!(" %s\n", msg));
}

fn emit(cmsp: Option<(codemap::codemap, span)>,
        msg: ~str, lvl: level) {
    match cmsp {
      Some((cm, sp)) => {
        let sp = codemap::adjust_span(cm,sp);
        let ss = codemap::span_to_str(sp, cm);
        let lines = codemap::span_to_lines(sp, cm);
        print_diagnostic(ss, lvl, msg);
        highlight_lines(cm, sp, lines);
        print_macro_backtrace(cm, sp);
      }
      None => {
        print_diagnostic(~"", lvl, msg);
      }
    }
}

fn highlight_lines(cm: codemap::codemap, sp: span,
                   lines: @codemap::file_lines) {

    let fm = lines.file;

    // arbitrarily only print up to six lines of the error
    let max_lines = 6u;
    let mut elided = false;
    let mut display_lines = /* FIXME (#2543) */ copy lines.lines;
    if vec::len(display_lines) > max_lines {
        display_lines = vec::slice(display_lines, 0u, max_lines);
        elided = true;
    }
    // Print the offending lines
    for display_lines.each |line| {
        io::stderr().write_str(fmt!("%s:%u ", fm.name, line + 1u));
        let s = codemap::get_line(fm, line as int) + ~"\n";
        io::stderr().write_str(s);
    }
    if elided {
        let last_line = display_lines[vec::len(display_lines) - 1u];
        let s = fmt!("%s:%u ", fm.name, last_line + 1u);
        let mut indent = str::len(s);
        let mut out = ~"";
        while indent > 0u { out += ~" "; indent -= 1u; }
        out += ~"...\n";
        io::stderr().write_str(out);
    }


    // If there's one line at fault we can easily point to the problem
    if vec::len(lines.lines) == 1u {
        let lo = codemap::lookup_char_pos(cm, sp.lo);
        let mut digits = 0u;
        let mut num = (lines.lines[0] + 1u) / 10u;

        // how many digits must be indent past?
        while num > 0u { num /= 10u; digits += 1u; }

        // indent past |name:## | and the 0-offset column location
        let mut left = str::len(fm.name) + digits + lo.col + 3u;
        let mut s = ~"";
        while left > 0u { str::push_char(s, ' '); left -= 1u; }

        s += ~"^";
        let hi = codemap::lookup_char_pos(cm, sp.hi);
        if hi.col != lo.col {
            // the ^ already takes up one space
            let mut width = hi.col - lo.col - 1u;
            while width > 0u { str::push_char(s, '~'); width -= 1u; }
        }
        io::stderr().write_str(s + ~"\n");
    }
}

fn print_macro_backtrace(cm: codemap::codemap, sp: span) {
    do option::iter (sp.expn_info) |ei| {
        let ss = option::map_default(ei.callie.span, @~"",
                                     |span| @codemap::span_to_str(span, cm));
        print_diagnostic(*ss, note,
                         fmt!("in expansion of #%s", ei.callie.name));
        let ss = codemap::span_to_str(ei.call_site, cm);
        print_diagnostic(ss, note, ~"expansion site");
        print_macro_backtrace(cm, ei.call_site);
    }
}

fn expect<T: copy>(diag: span_handler,
                   opt: Option<T>, msg: fn() -> ~str) -> T {
    match opt {
       Some(t) => t,
       None => diag.handler().bug(msg())
    }
}
