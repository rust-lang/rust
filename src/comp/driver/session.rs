import util.common.span;
import std._uint;

obj session() {
    fn span_err(span sp, str msg) {
        let str s =  sp.filename;
        s += ':' as u8;
        // We really need #fmt soon!
        s += _uint.to_str(sp.lo.line, 10u);
        s += ':' as u8;
        s += _uint.to_str(sp.lo.col, 10u);
        s += ':' as u8;
        s += _uint.to_str(sp.hi.line, 10u);
        s += ':' as u8;
        s += _uint.to_str(sp.hi.col, 10u);
        s += ": error: ";
        s += msg;
        log s;
        fail;
    }

    fn err(str msg) {
        let str s = "error: ";
        s += msg;
        log s;
        fail;
    }

    fn unimpl(str msg) {
        let str s = "error: unimplemented ";
        s += msg;
        log s;
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
