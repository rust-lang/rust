import util.common.span;
import util.common.ty_mach;
import std._uint;
import std.map;

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

type crate_metadata = vec[u8];

obj session(cfg targ, map.hashmap[int, crate_metadata] crates) {

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

    fn span_warn(span sp, str msg) {
        log #fmt("%s:%u:%u:%u:%u: warning: %s",
                 sp.filename,
                 sp.lo.line, sp.lo.col,
                 sp.hi.line, sp.hi.col,
                 msg);
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

    fn get_external_crate(int num) -> crate_metadata {
        ret crates.get(num);
    }

    fn set_external_crate(int num, &crate_metadata metadata) {
        crates.insert(num, metadata);
    }

    fn has_external_crate(int num) -> bool {
        ret crates.contains_key(num);
    }
}


// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../../../build 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
