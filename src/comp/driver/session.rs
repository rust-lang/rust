import util.common.span;
import std._uint;

obj session() {
    fn span_err(span sp, str msg) {
        log #fmt("%s:%u:%u:%u:%u: error: %s",
                 sp.filename,
                 sp.lo.line, sp.lo.col,
                 sp.hi.line, sp.hi.col,
                 msg);
        fail;
    }

    fn err(str msg) {
        log #fmt("error: %s", msg);
        fail;
    }

    fn unimpl(str msg) {
        log #fmt("error: unimplemented %s", msg);
        fail;
    }
}


// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
