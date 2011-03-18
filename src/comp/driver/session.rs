import util.common.span;
import util.common.ty_mach;
import std._uint;

tag os {
    os_win32;
    os_macos;
    os_linux;
}

tag arch {
    arch_x86;
    arch_x64;
    arch_arm;
}

type cfg = rec(os os,
               arch arch,
               ty_mach int_type,
               ty_mach uint_type,
               ty_mach float_type);

obj session(cfg targ) {

    fn get_targ_cfg() -> cfg {
        ret targ;
    }

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

    fn bug(str msg) {
        log #fmt("error: internal compiler error %s", msg);
        fail;
    }

    fn span_unimpl(span sp, str msg) {
        log #fmt("%s:%u:%u:%u:%u: error: unimplemented %s",
                 sp.filename,
                 sp.lo.line, sp.lo.col,
                 sp.hi.line, sp.hi.col,
                 msg);
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
