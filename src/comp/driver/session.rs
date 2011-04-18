import front.ast;
import front.codemap;
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

type crate_metadata = rec(str name,
                          vec[u8] data);

state obj session(ast.crate_num cnum, cfg targ,
                  map.hashmap[int, crate_metadata] crates,
                  mutable vec[@ast.meta_item] metadata,
                  codemap.codemap cm) {

    fn get_targ_cfg() -> cfg {
        ret targ;
    }

    fn get_targ_crate_num() -> ast.crate_num {
        ret cnum;
    }

    fn span_err(span sp, str msg) {
        auto lo = codemap.lookup_pos(cm, sp.lo);
        auto hi = codemap.lookup_pos(cm, sp.hi);
        log #fmt("%s:%u:%u:%u:%u: error: %s",
                 lo.filename,
                 lo.line, lo.col,
                 hi.line, hi.col,
                 msg);
        fail;
    }

    fn err(str msg) {
        log #fmt("error: %s", msg);
        fail;
    }

    fn add_metadata(vec[@ast.meta_item] data) {
        metadata = metadata + data;
    }
    fn get_metadata() -> vec[@ast.meta_item] {
        ret metadata;
    }

    fn span_warn(span sp, str msg) {
        auto lo = codemap.lookup_pos(cm, sp.lo);
        auto hi = codemap.lookup_pos(cm, sp.hi);
        log #fmt("%s:%u:%u:%u:%u: warning: %s",
                 lo.filename,
                 lo.line, lo.col,
                 hi.line, hi.col,
                 msg);
    }

    fn bug(str msg) {
        log #fmt("error: internal compiler error %s", msg);
        fail;
    }

    fn span_unimpl(span sp, str msg) {
        auto lo = codemap.lookup_pos(cm, sp.lo);
        auto hi = codemap.lookup_pos(cm, sp.hi);
        log #fmt("%s:%u:%u:%u:%u: error: unimplemented %s",
                 lo.filename,
                 lo.line, lo.col,
                 hi.line, hi.col,
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

    fn get_codemap() -> codemap.codemap {
        ret cm;
    }

    fn lookup_pos(uint pos) -> codemap.loc {
        ret codemap.lookup_pos(cm, pos);
    }
}


// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
