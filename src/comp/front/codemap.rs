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
    let vec[uint] lines = vec();
    ret @rec(name=filename,
             start_pos=start_pos,
             mutable lines=lines);
}

fn next_line(filemap file, uint pos) {
    _vec.push[uint](file.lines, pos);
}

fn lookup_pos(codemap map, uint pos) -> loc {
    for (filemap f in map.files) {
        if (f.start_pos < pos) {
            auto line_num = 1u;
            auto line_start = 0u;
            // FIXME this can be a binary search if we need to be faster
            for (uint line_start_ in f.lines) {
                // FIXME duplicate code due to lack of working break
                if (line_start_ > pos) {
                    ret rec(filename=f.name,
                            line=line_num,
                            col=pos-line_start);
                }
                line_start = line_start_;
                line_num += 1u;
            }
            ret rec(filename=f.name,
                    line=line_num,
                    col=pos-line_start);
        }
    }
    log #fmt("Failed to find a location for character %u", pos);
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
