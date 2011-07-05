
import std::vec;
import std::term;
import std::io;
import std::option;
import std::option::some;
import std::option::none;

type filename = str;

/* A codemap is a thing that maps uints to file/line/column positions
 * in a crate. This to make it possible to represent the positions
 * with single-word things, rather than passing records all over the
 * compiler.
 */
type filemap = @rec(filename name, uint start_pos, mutable vec[uint] lines);

type codemap = @rec(mutable vec[filemap] files);

type loc = rec(filename filename, uint line, uint col);

fn new_codemap() -> codemap {
    let vec[filemap] files = [];
    ret @rec(mutable files=files);
}

fn new_filemap(filename filename, uint start_pos) -> filemap {
    ret @rec(name=filename, start_pos=start_pos, mutable lines=[0u]);
}

fn next_line(filemap file, uint pos) { vec::push[uint](file.lines, pos); }

fn lookup_pos(codemap map, uint pos) -> loc {
    auto a = 0u;
    auto b = vec::len[filemap](map.files);
    while (b - a > 1u) {
        auto m = (a + b) / 2u;
        if (map.files.(m).start_pos > pos) { b = m; } else { a = m; }
    }
    auto f = map.files.(a);
    a = 0u;
    b = vec::len[uint](f.lines);
    while (b - a > 1u) {
        auto m = (a + b) / 2u;
        if (f.lines.(m) > pos) { b = m; } else { a = m; }
    }
    ret rec(filename=f.name, line=a + 1u, col=pos - f.lines.(a));
}

type span = rec(uint lo, uint hi);

fn span_to_str(&span sp, &codemap cm) -> str {
    auto lo = lookup_pos(cm, sp.lo);
    auto hi = lookup_pos(cm, sp.hi);
    ret #fmt("%s:%u:%u:%u:%u", lo.filename, lo.line, lo.col, hi.line, hi.col);
}

fn emit_diagnostic(&option::t[span] sp, &str msg, &str kind, u8 color,
                   &codemap cm) {
    auto ss = "<input>:0:0:0:0";
    alt (sp) {
        case (some(?ssp)) { ss = span_to_str(ssp, cm); }
        case (none) { }
    }
    io::stdout().write_str(ss + ": ");
    if (term::color_supported()) {
        term::fg(io::stdout().get_buf_writer(), color);
    }
    io::stdout().write_str(#fmt("%s:", kind));
    if (term::color_supported()) {
        term::reset(io::stdout().get_buf_writer());
    }
    io::stdout().write_str(#fmt(" %s\n", msg));
}

fn emit_warning(&option::t[span] sp, &str msg, &codemap cm) {
    emit_diagnostic(sp, msg, "warning", 11u8, cm);
}
fn emit_error(&option::t[span] sp, &str msg, &codemap cm) {
    emit_diagnostic(sp, msg, "error", 9u8, cm);
}
fn emit_note(&option::t[span] sp, &str msg, &codemap cm) {
    emit_diagnostic(sp, msg, "note", 10u8, cm);
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
