import std::ivec;
import std::uint;
import std::str;
import std::termivec;
import std::ioivec;
import std::option;
import std::option::some;
import std::option::none;

type filename = str;

/* A codemap is a thing that maps uints to file/line/column positions
 * in a crate. This to make it possible to represent the positions
 * with single-word things, rather than passing records all over the
 * compiler.
 */
type filemap = @rec(filename name, uint start_pos, mutable uint[] lines);

type codemap = @rec(mutable filemap[] files);

type loc = rec(filename filename, uint line, uint col);

fn new_codemap() -> codemap {
    let filemap[] files = ~[];
    ret @rec(mutable files=files);
}

fn new_filemap(filename filename, uint start_pos) -> filemap {
    ret @rec(name=filename, start_pos=start_pos, mutable lines=~[0u]);
}

fn next_line(filemap file, uint pos) { file.lines += ~[pos]; }

fn lookup_pos(codemap map, uint pos) -> loc {
    auto a = 0u;
    auto b = ivec::len[filemap](map.files);
    while (b - a > 1u) {
        auto m = (a + b) / 2u;
        if (map.files.(m).start_pos > pos) { b = m; } else { a = m; }
    }
    auto f = map.files.(a);
    a = 0u;
    b = ivec::len[uint](f.lines);
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
    let option::t[@file_lines] maybe_lines = none;
    alt (sp) {
        case (some(?ssp)) {
            ss = span_to_str(ssp, cm);
            maybe_lines = some(span_to_lines(ssp, cm));
        }
        case (none) { }
    }
    ioivec::stdout().write_str(ss + ": ");
    if (termivec::color_supported()) {
        termivec::fg(ioivec::stdout().get_buf_writer(), color);
    }
    ioivec::stdout().write_str(#fmt("%s:", kind));
    if (termivec::color_supported()) {
        termivec::reset(ioivec::stdout().get_buf_writer());
    }
    ioivec::stdout().write_str(#fmt(" %s\n", msg));
    alt (maybe_lines) {
        case (some(?lines)) {
            // FIXME: reading in the entire file is the worst possible way to
            //        get access to the necessary lines.
            auto rdr = ioivec::file_reader(lines.name);
            auto file = str::unsafe_from_bytes_ivec(rdr.read_whole_stream());
            auto fm = get_filemap(cm, lines.name);

            // arbitrarily only print up to six lines of the error
            auto max_lines = 6u;
            auto elided = false;
            auto display_lines = lines.lines;
            if (ivec::len(display_lines) > max_lines) {
                display_lines = ivec::slice(display_lines, 0u, max_lines);
                elided = true;
            }
            // Print the offending lines
            for (uint line in display_lines) {
                ioivec::stdout().write_str(#fmt("%s:%u ", fm.name,
                                                line + 1u));
                auto s = get_line(fm, line as int, file);
                if (!str::ends_with(s, "\n")) {
                    s += "\n";
                }
                ioivec::stdout().write_str(s);
            }
            if (elided) {
                auto last_line = display_lines.(ivec::len(display_lines) -
                                                1u);
                auto s = #fmt("%s:%u ", fm.name, last_line + 1u);
                auto indent = str::char_len(s);
                auto out = "";
                while (indent > 0u) { out += " "; indent -= 1u; }
                out += "...\n";
                ioivec::stdout().write_str(out);
            }

            // If there's one line at fault we can easily point to the problem
            if (ivec::len(lines.lines) == 1u) {
                auto lo = codemap::lookup_pos(cm, option::get(sp).lo);
                auto lo = lookup_pos(cm, option::get(sp).lo);
                auto digits = 0u;
                auto num = lines.lines.(0) / 10u;

                // how many digits must be indent past?
                while (num > 0u) { num /= 10u; digits += 1u; }

                // indent past |name:## | and the 0-offset column location
                auto left = str::char_len(fm.name) + digits + lo.col + 3u;
                auto s = "";
                while (left > 0u) { str::push_char(s, ' '); left -= 1u; }

                s += "^";
                auto hi = lookup_pos(cm, option::get(sp).hi);
                if (hi.col != lo.col) {
                    // the ^ already takes up one space
                    auto width = hi.col - lo.col - 1u;
                    while (width > 0u) {
                        str::push_char(s, '~');
                        width -= 1u;
                    }
                }
                ioivec::stdout().write_str(s + "\n");
            }
        }
        case (_) {}
    }
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

type file_lines = rec(str name, uint[] lines);

fn span_to_lines(span sp, codemap::codemap cm) -> @file_lines {
    auto lo = codemap::lookup_pos(cm, sp.lo);
    auto hi = codemap::lookup_pos(cm, sp.hi);
    auto lines = ~[];
    for each (uint i in uint::range(lo.line - 1u, hi.line as uint)) {
        lines += ~[i];
    }
    ret @rec(name=lo.filename, lines=lines);
}

fn get_line(filemap fm, int line, &str file) -> str {
    let uint begin = fm.lines.(line) - fm.start_pos;
    let uint end;
    if ((line as uint) + 1u >= ivec::len(fm.lines)) {
        end = str::byte_len(file);
    } else {
        end = fm.lines.(line + 1) - fm.start_pos;
    }
    ret str::slice(file, begin, end);
}

fn get_filemap(codemap cm, str filename) -> filemap {
    for (filemap fm in cm.files) {
        if (fm.name == filename) {
            ret fm;
        }
    }
    //XXjdm the following triggers a mismatched type bug
    //      (or expected function, found _|_)
    fail;// ("asking for " + filename + " which we don't know about");
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
