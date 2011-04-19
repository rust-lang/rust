import std._vec;

/* A codemap is a thing that maps uints to file/line/column positions
 * in a crate. This to make it possible to represent the positions
 * with single-word things, rather than passing records all over the
 * compiler.
 */

type filemap = @rec(str name,
                    uint start_pos,
                    mutable vec[uint] lines);
type codemap = @rec(mutable vec[filemap] files);
type loc = rec(str filename, uint line, uint col);

fn new_codemap() -> codemap {
    let vec[filemap] files = vec();
    ret @rec(mutable files=files);
}

fn new_filemap(str filename, uint start_pos) -> filemap {
    ret @rec(name=filename,
             start_pos=start_pos,
             mutable lines=vec(0u));
}

fn next_line(filemap file, uint pos) {
    _vec.push[uint](file.lines, pos);
}

fn lookup_pos(codemap map, uint pos) -> loc {
    auto i = _vec.len[filemap](map.files);
    while (i > 0u) {
        i -= 1u;
        auto f = map.files.(i);
        if (f.start_pos <= pos) {
            // FIXME this can be a binary search if we need to be faster
            auto line = _vec.len[uint](f.lines);
            while (line > 0u) {
                line -= 1u;
                auto line_start = f.lines.(line);
                if (line_start <= pos) {
                    ret rec(filename=f.name,
                            line=line + 1u,
                            col=pos-line_start);
                }
            }
        }
    }
    log_err #fmt("Failed to find a location for character %u", pos);
    fail;
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
