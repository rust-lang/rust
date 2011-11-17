
import syntax::{ast, codemap};
import syntax::ast::node_id;
import codemap::span;
import syntax::ast::ty_mach;
import std::{option};
import std::option::{some, none};
import syntax::parse::parser::parse_sess;
import util::filesearch;
import back::target_strs;

tag os { os_win32; os_macos; os_linux; }

tag arch { arch_x86; arch_x86_64; arch_arm; }

type config =
    {os: os,
     arch: arch,
     target_strs: target_strs::t,
     int_type: ty_mach,
     uint_type: ty_mach,
     float_type: ty_mach};

type options =
    // The crate config requested for the session, which may be combined
    // with additional crate configurations during the compile process
    {library: bool,
     static: bool,
     optimize: uint,
     debuginfo: bool,
     verify: bool,
     save_temps: bool,
     stats: bool,
     time_passes: bool,
     time_llvm_passes: bool,
     output_type: back::link::output_type,
     addl_lib_search_paths: [str],
     maybe_sysroot: option::t<str>,
     target_triple: str,
     cfg: ast::crate_cfg,
     test: bool,
     parse_only: bool,
     no_trans: bool,
     do_gc: bool,
     stack_growth: bool,
     no_asm_comments: bool,
     warn_unused_imports: bool};

type crate_metadata = {name: str, data: [u8]};

obj session(targ_cfg: @config,
            opts: @options,
            cstore: metadata::cstore::cstore,
            parse_sess: parse_sess,

            // For a library crate, this is always none
            mutable main_fn: option::t<node_id>,
            mutable err_count: uint,
            filesearch: filesearch::filesearch) {
    fn get_targ_cfg() -> @config { ret targ_cfg; }
    fn get_opts() -> @options { ret opts; }
    fn get_cstore() -> metadata::cstore::cstore { cstore }
    fn span_fatal(sp: span, msg: str) -> ! {
        // FIXME: Use constants, but rustboot doesn't know how to export them.
        codemap::emit_error(some(sp), msg, parse_sess.cm);
        fail;
    }
    fn fatal(msg: str) -> ! {
        codemap::emit_error(none, msg, parse_sess.cm);
        fail;
    }
    fn span_err(sp: span, msg: str) {
        codemap::emit_error(some(sp), msg, parse_sess.cm);
        err_count += 1u;
    }
    fn err(msg: str) {
        codemap::emit_error(none, msg, parse_sess.cm);
        err_count += 1u;
    }
    fn has_errors() -> bool { err_count > 0u }
    fn abort_if_errors() {
        if err_count > 0u { self.fatal("aborting due to previous errors"); }
    }
    fn span_warn(sp: span, msg: str) {
        // FIXME: Use constants, but rustboot doesn't know how to export them.
        codemap::emit_warning(some(sp), msg, parse_sess.cm);
    }
    fn warn(msg: str) { codemap::emit_warning(none, msg, parse_sess.cm); }
    fn span_note(sp: span, msg: str) {
        // FIXME: Use constants, but rustboot doesn't know how to export them.
        codemap::emit_note(some(sp), msg, parse_sess.cm);
    }
    fn note(msg: str) { codemap::emit_note(none, msg, parse_sess.cm); }
    fn span_bug(sp: span, msg: str) -> ! {
        self.span_fatal(sp, #fmt["internal compiler error %s", msg]);
    }
    fn bug(msg: str) -> ! {
        self.fatal(#fmt["internal compiler error %s", msg]);
    }
    fn span_unimpl(sp: span, msg: str) -> ! {
        self.span_bug(sp, "unimplemented " + msg);
    }
    fn unimpl(msg: str) -> ! { self.bug("unimplemented " + msg); }
    fn get_codemap() -> codemap::codemap { ret parse_sess.cm; }
    fn lookup_pos(pos: uint) -> codemap::loc {
        ret codemap::lookup_char_pos(parse_sess.cm, pos);
    }
    fn get_parse_sess() -> parse_sess { ret parse_sess; }
    fn next_node_id() -> ast::node_id {
        ret syntax::parse::parser::next_node_id(parse_sess);
    }
    fn span_str(sp: span) -> str {
        ret codemap::span_to_str(sp, self.get_codemap());
    }
    fn set_main_id(d: node_id) { main_fn = some(d); }
    fn get_main_id() -> option::t<node_id> { main_fn }
    fn filesearch() -> filesearch::filesearch { filesearch }
}
// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
