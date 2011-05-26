import front::ast;
import front::codemap;
import util::common::span;
import util::common::ty_mach;
import std::uint;
import std::term;
import std::io;
import std::map;

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

type config = rec(os os,
                  arch arch,
                  ty_mach int_type,
                  ty_mach uint_type,
                  ty_mach float_type);

type options = rec(bool shared,
                   uint optimize,
                   bool debuginfo,
                   bool verify,
                   bool run_typestate,
                   bool save_temps,
                   bool stats,
                   bool time_passes,
                   bool time_llvm_passes,
                   back::link::output_type output_type,
                   vec[str] library_search_paths,
                   str sysroot);

type crate_metadata = rec(str name,
                          vec[u8] data);

fn span_to_str(span sp, codemap::codemap cm) -> str {
    auto lo = codemap::lookup_pos(cm, sp.lo);
    auto hi = codemap::lookup_pos(cm, sp.hi);
    ret (#fmt("%s:%u:%u:%u:%u", lo.filename, lo.line,
              lo.col, hi.line, hi.col));
}

fn emit_diagnostic(span sp, str msg, str kind, u8 color,
                   codemap::codemap cm) {
    io::stdout().write_str(span_to_str(sp, cm) + ": ");

    if (term::color_supported()) {
        term::fg(io::stdout().get_buf_writer(), color);
    }

    io::stdout().write_str(#fmt("%s:", kind));

    if (term::color_supported()) {
        term::reset(io::stdout().get_buf_writer());
    }

    io::stdout().write_str(#fmt(" %s\n", msg));
}

state obj session(ast::crate_num cnum,
                  @config targ_cfg, @options opts,
                  map::hashmap[int, crate_metadata] crates,
                  mutable vec[@ast::meta_item] metadata,
                  codemap::codemap cm) {

    fn get_targ_cfg() -> @config {
        ret targ_cfg;
    }

    fn get_opts() -> @options {
        ret opts;
    }

    fn get_targ_crate_num() -> ast::crate_num {
        ret cnum;
    }

    fn span_err(span sp, str msg) -> ! {
        // FIXME: Use constants, but rustboot doesn't know how to export them.
        emit_diagnostic(sp, msg, "error", 9u8, cm);
        fail;
    }

    fn err(str msg) -> ! {
        log_err #fmt("error: %s", msg);
        fail;
    }

    fn add_metadata(vec[@ast::meta_item] data) {
        metadata = metadata + data;
    }
    fn get_metadata() -> vec[@ast::meta_item] {
        ret metadata;
    }

    fn span_warn(span sp, str msg) {
        // FIXME: Use constants, but rustboot doesn't know how to export them.
        emit_diagnostic(sp, msg, "warning", 11u8, cm);
    }

    fn span_note(span sp, str msg) {
        // FIXME: Use constants, but rustboot doesn't know how to export them.
        emit_diagnostic(sp, msg, "note", 10u8, cm);
    }

    fn bug(str msg) -> ! {
        log_err #fmt("error: internal compiler error %s", msg);
        fail;
    }

    fn span_unimpl(span sp, str msg) -> ! {
        // FIXME: Use constants, but rustboot doesn't know how to export them.
        emit_diagnostic(sp, "internal compiler error: unimplemented " + msg,
                        "error", 9u8, cm);
        fail;
    }
    
    fn unimpl(str msg) -> ! {
        log_err #fmt("error: unimplemented %s", msg);
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

    fn get_codemap() -> codemap::codemap {
        ret cm;
    }

    fn lookup_pos(uint pos) -> codemap::loc {
        ret codemap::lookup_pos(cm, pos);
    }

    fn span_str(span sp) -> str {
        ret span_to_str(sp, self.get_codemap());
    }
}


// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
