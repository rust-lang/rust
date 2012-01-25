import core::{vec, uint, str, option, result};
import option::{some, none};

type filename = str;

type file_pos = {ch: uint, byte: uint};

/* A codemap is a thing that maps uints to file/line/column positions
 * in a crate. This to make it possible to represent the positions
 * with single-word things, rather than passing records all over the
 * compiler.
 */
type filemap =
    @{name: filename, src: @str,
      start_pos: file_pos, mutable lines: [file_pos]};

type codemap = @{mutable files: [filemap]};

type loc = {filename: filename, line: uint, col: uint};

fn new_codemap() -> codemap { ret @{mutable files: []}; }

fn new_filemap(filename: filename, src: @str,
               start_pos_ch: uint, start_pos_byte: uint)
   -> filemap {
    ret @{name: filename, src: src,
          start_pos: {ch: start_pos_ch, byte: start_pos_byte},
          mutable lines: [{ch: start_pos_ch, byte: start_pos_byte}]};
}

fn next_line(file: filemap, chpos: uint, byte_pos: uint) {
    file.lines += [{ch: chpos, byte: byte_pos}];
}

type lookup_fn = fn@(file_pos) -> uint;

fn lookup_pos(map: codemap, pos: uint, lookup: lookup_fn) -> loc {
    let len = vec::len(map.files);
    let a = 0u;
    let b = len;
    while b - a > 1u {
        let m = (a + b) / 2u;
        if lookup(map.files[m].start_pos) > pos { b = m; } else { a = m; }
    }
    if (a >= len) {
        ret { filename: "-", line: 0u, col: 0u };
    }
    let f = map.files[a];
    a = 0u;
    b = vec::len(f.lines);
    while b - a > 1u {
        let m = (a + b) / 2u;
        if lookup(f.lines[m]) > pos { b = m; } else { a = m; }
    }
    ret {filename: f.name, line: a + 1u, col: pos - lookup(f.lines[a])};
}

fn lookup_char_pos(map: codemap, pos: uint) -> loc {
    fn lookup(pos: file_pos) -> uint { ret pos.ch; }
    ret lookup_pos(map, pos, lookup);
}

fn lookup_byte_pos(map: codemap, pos: uint) -> loc {
    fn lookup(pos: file_pos) -> uint { ret pos.byte; }
    ret lookup_pos(map, pos, lookup);
}

enum opt_span {

    //hack (as opposed to option::t), to make `span` compile
    os_none,
    os_some(@span),
}
type span = {lo: uint, hi: uint, expanded_from: opt_span};

fn span_to_str(sp: span, cm: codemap) -> str {
    let cur = sp;
    let res = "";
    let prev_file = none;
    while true {
        let lo = lookup_char_pos(cm, cur.lo);
        let hi = lookup_char_pos(cm, cur.hi);
        res +=
            #fmt["%s:%u:%u: %u:%u",
                 if some(lo.filename) == prev_file {
                     "-"
                 } else { lo.filename }, lo.line, lo.col, hi.line, hi.col];
        alt cur.expanded_from {
          os_none { break; }
          os_some(new_sp) {
            cur = *new_sp;
            prev_file = some(lo.filename);
            res += "<<";
          }
        }
    }

    ret res;
}

type file_lines = {name: str, lines: [uint]};

fn span_to_lines(sp: span, cm: codemap::codemap) -> @file_lines {
    let lo = lookup_char_pos(cm, sp.lo);
    let hi = lookup_char_pos(cm, sp.hi);
    let lines = [];
    uint::range(lo.line - 1u, hi.line as uint) {|i| lines += [i]; };
    ret @{name: lo.filename, lines: lines};
}

fn get_line(fm: filemap, line: int) -> str {
    let begin: uint = fm.lines[line].byte - fm.start_pos.byte;
    let end: uint;
    if line as uint < vec::len(fm.lines) - 1u {
        end = fm.lines[line + 1].byte - fm.start_pos.byte;
    } else {
        // If we're not done parsing the file, we're at the limit of what's
        // parsed. If we just slice the rest of the string, we'll print out
        // the remainder of the file, which is undesirable.
        end = str::byte_len(*fm.src);
        let rest = str::slice(*fm.src, begin, end);
        let newline = str::index(rest, '\n' as u8);
        if newline != -1 { end = begin + (newline as uint); }
    }
    ret str::slice(*fm.src, begin, end);
}

fn get_filemap(cm: codemap, filename: str) -> filemap {
    for fm: filemap in cm.files { if fm.name == filename { ret fm; } }
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
// End:
//
