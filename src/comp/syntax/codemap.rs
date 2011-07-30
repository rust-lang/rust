import std::ivec;
import std::uint;
import std::str;
import std::termivec;
import std::ioivec;
import std::option;
import std::option::some;
import std::option::none;

type filename = str;

type file_pos = {ch: uint, byte: uint};

/* A codemap is a thing that maps uints to file/line/column positions
 * in a crate. This to make it possible to represent the positions
 * with single-word things, rather than passing records all over the
 * compiler.
 */
type filemap =
    @{name: filename, start_pos: file_pos, mutable lines: file_pos[]};

type codemap = @{mutable files: filemap[]};

type loc = {filename: filename, line: uint, col: uint};

fn new_codemap() -> codemap { ret @{mutable files: ~[]}; }

fn new_filemap(filename: filename, start_pos_ch: uint, start_pos_byte: uint)
   -> filemap {
    ret @{name: filename,
          start_pos: {ch: start_pos_ch, byte: start_pos_byte},
          mutable lines: ~[{ch: start_pos_ch, byte: start_pos_byte}]};
}

fn next_line(file: filemap, chpos: uint, byte_pos: uint) {
    file.lines += ~[{ch: chpos, byte: byte_pos}];
}

type lookup_fn = fn(file_pos) -> uint ;

fn lookup_pos(map: codemap, pos: uint, lookup: lookup_fn) -> loc {
    let a = 0u;
    let b = ivec::len(map.files);
    while b - a > 1u {
        let m = (a + b) / 2u;
        if lookup(map.files.(m).start_pos) > pos { b = m; } else { a = m; }
    }
    let f = map.files.(a);
    a = 0u;
    b = ivec::len(f.lines);
    while b - a > 1u {
        let m = (a + b) / 2u;
        if lookup(f.lines.(m)) > pos { b = m; } else { a = m; }
    }
    ret {filename: f.name, line: a + 1u, col: pos - lookup(f.lines.(a))};
}

fn lookup_char_pos(map: codemap, pos: uint) -> loc {
    fn lookup(pos: file_pos) -> uint { ret pos.ch; }
    ret lookup_pos(map, pos, lookup);
}

fn lookup_byte_pos(map: codemap, pos: uint) -> loc {
    fn lookup(pos: file_pos) -> uint { ret pos.byte; }
    ret lookup_pos(map, pos, lookup);
}

type span = {lo: uint, hi: uint};

fn span_to_str(sp: &span, cm: &codemap) -> str {
    let lo = lookup_char_pos(cm, sp.lo);
    let hi = lookup_char_pos(cm, sp.hi);
    ret #fmt("%s:%u:%u:%u:%u", lo.filename, lo.line, lo.col, hi.line, hi.col);
}

fn emit_diagnostic(sp: &option::t[span], msg: &str, kind: &str, color: u8,
                   cm: &codemap) {
    let ss = "<input>:0:0:0:0";
    let maybe_lines: option::t[@file_lines] = none;
    alt sp {
      some(ssp) {
        ss = span_to_str(ssp, cm);
        maybe_lines = some(span_to_lines(ssp, cm));
      }
      none. { }
    }
    ioivec::stdout().write_str(ss + ": ");
    if termivec::color_supported() {
        termivec::fg(ioivec::stdout().get_buf_writer(), color);
    }
    ioivec::stdout().write_str(#fmt("%s:", kind));
    if termivec::color_supported() {
        termivec::reset(ioivec::stdout().get_buf_writer());
    }
    ioivec::stdout().write_str(#fmt(" %s\n", msg));

    maybe_highlight_lines(sp, cm, maybe_lines);
}

fn maybe_highlight_lines(sp: &option::t[span], cm: &codemap,
                         maybe_lines: option::t[@file_lines]) {

    alt maybe_lines {
      some(lines) {
        // If we're not looking at a real file then we can't re-open it to
        // pull out the lines
        if lines.name == "-" { ret; }

        // FIXME: reading in the entire file is the worst possible way to
        //        get access to the necessary lines.
        let rdr = ioivec::file_reader(lines.name);
        let file = str::unsafe_from_bytes_ivec(rdr.read_whole_stream());
        let fm = get_filemap(cm, lines.name);

        // arbitrarily only print up to six lines of the error
        let max_lines = 6u;
        let elided = false;
        let display_lines = lines.lines;
        if ivec::len(display_lines) > max_lines {
            display_lines = ivec::slice(display_lines, 0u, max_lines);
            elided = true;
        }
        // Print the offending lines
        for line: uint  in display_lines {
            ioivec::stdout().write_str(#fmt("%s:%u ", fm.name, line + 1u));
            let s = get_line(fm, line as int, file);
            if !str::ends_with(s, "\n") { s += "\n"; }
            ioivec::stdout().write_str(s);
        }
        if elided {
            let last_line = display_lines.(ivec::len(display_lines) - 1u);
            let s = #fmt("%s:%u ", fm.name, last_line + 1u);
            let indent = str::char_len(s);
            let out = "";
            while indent > 0u { out += " "; indent -= 1u; }
            out += "...\n";
            ioivec::stdout().write_str(out);
        }


        // If there's one line at fault we can easily point to the problem
        if ivec::len(lines.lines) == 1u {
            let lo = lookup_char_pos(cm, option::get(sp).lo);
            let digits = 0u;
            let num = lines.lines.(0) / 10u;

            // how many digits must be indent past?
            while num > 0u { num /= 10u; digits += 1u; }

            // indent past |name:## | and the 0-offset column location
            let left = str::char_len(fm.name) + digits + lo.col + 3u;
            let s = "";
            while left > 0u { str::push_char(s, ' '); left -= 1u; }

            s += "^";
            let hi = lookup_char_pos(cm, option::get(sp).hi);
            if hi.col != lo.col {
                // the ^ already takes up one space
                let width = hi.col - lo.col - 1u;
                while width > 0u { str::push_char(s, '~'); width -= 1u; }
            }
            ioivec::stdout().write_str(s + "\n");
        }
      }
      _ { }
    }
}

fn emit_warning(sp: &option::t[span], msg: &str, cm: &codemap) {
    emit_diagnostic(sp, msg, "warning", 11u8, cm);
}
fn emit_error(sp: &option::t[span], msg: &str, cm: &codemap) {
    emit_diagnostic(sp, msg, "error", 9u8, cm);
}
fn emit_note(sp: &option::t[span], msg: &str, cm: &codemap) {
    emit_diagnostic(sp, msg, "note", 10u8, cm);
}

type file_lines = {name: str, lines: uint[]};

fn span_to_lines(sp: span, cm: codemap::codemap) -> @file_lines {
    let lo = lookup_char_pos(cm, sp.lo);
    let hi = lookup_char_pos(cm, sp.hi);
    let lines = ~[];
    for each i: uint  in uint::range(lo.line - 1u, hi.line as uint) {
        lines += ~[i];
    }
    ret @{name: lo.filename, lines: lines};
}

fn get_line(fm: filemap, line: int, file: &str) -> str {
    let begin: uint = fm.lines.(line).byte - fm.start_pos.byte;
    let end: uint;
    if line as uint < ivec::len(fm.lines) - 1u {
        end = fm.lines.(line + 1).byte - fm.start_pos.byte;
    } else {
        // If we're not done parsing the file, we're at the limit of what's
        // parsed. If we just slice the rest of the string, we'll print out
        // the remainder of the file, which is undesirable.
        end = str::byte_len(file);
        let rest = str::slice(file, begin, end);
        let newline = str::index(rest, '\n' as u8);
        if newline != -1 { end = begin + (newline as uint); }
    }
    ret str::slice(file, begin, end);
}

fn get_filemap(cm: codemap, filename: str) -> filemap {
    for fm: filemap  in cm.files { if fm.name == filename { ret fm; } }
    //XXjdm the following triggers a mismatched type bug
    //      (or expected function, found _|_)
    fail; // ("asking for " + filename + " which we don't know about");
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
