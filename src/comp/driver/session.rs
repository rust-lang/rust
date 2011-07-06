
import syntax::ast;
import syntax::codemap;
import codemap::span;
import syntax::ast::ty_mach;
import std::uint;
import std::io;
import std::map;
import std::option;
import std::option::some;
import std::option::none;
import std::str;
import syntax::parse::parser::parse_sess;

tag os { os_win32; os_macos; os_linux; }

tag arch { arch_x86; arch_x64; arch_arm; }

type config =
    rec(os os,
        arch arch,
        ty_mach int_type,
        ty_mach uint_type,
        ty_mach float_type);

type options =
    rec(bool shared,
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
        str sysroot,
        // The crate config requested for the session, which may be combined
        // with additional crate configurations during the compile process
        ast::crate_cfg cfg,
        bool test);

type crate_metadata = rec(str name, vec[u8] data);

obj session(@config targ_cfg,
            @options opts,
            map::hashmap[int, crate_metadata] crates,
            mutable vec[str] used_crate_files,
            mutable vec[str] used_libraries,
            mutable vec[str] used_link_args,
            parse_sess parse_sess,
            mutable uint err_count) {
    fn get_targ_cfg() -> @config { ret targ_cfg; }
    fn get_opts() -> @options { ret opts; }
    fn span_fatal(span sp, str msg) -> ! {
        // FIXME: Use constants, but rustboot doesn't know how to export them.
        codemap::emit_error(some(sp), msg, parse_sess.cm);
        fail;
    }
    fn fatal(str msg) -> ! {
        codemap::emit_error(none, msg, parse_sess.cm);
        fail;
    }
    fn span_err(span sp, str msg) {
        codemap::emit_error(some(sp), msg, parse_sess.cm);
        err_count += 1u;
    }
    fn err(str msg) {
        codemap::emit_error(none, msg, parse_sess.cm);
        err_count += 1u;
    }
    fn abort_if_errors() {
        if (err_count > 0u) {
            self.fatal("aborting due to previous errors");
        }
    }
    fn span_warn(span sp, str msg) {
        // FIXME: Use constants, but rustboot doesn't know how to export them.
        codemap::emit_warning(some(sp), msg, parse_sess.cm);
    }
    fn warn(str msg) {
        codemap::emit_warning(none, msg, parse_sess.cm);
    }
    fn span_note(span sp, str msg) {
        // FIXME: Use constants, but rustboot doesn't know how to export them.
        codemap::emit_note(some(sp), msg, parse_sess.cm);
    }
    fn note(str msg) {
        codemap::emit_note(none, msg, parse_sess.cm);
    }
    fn span_bug(span sp, str msg) -> ! {
        self.span_fatal(sp, #fmt("internal compiler error %s", msg));
    }
    fn bug(str msg) -> ! {
        self.fatal(#fmt("internal compiler error %s", msg));
    }
    fn span_unimpl(span sp, str msg) -> ! {
        self.span_bug(sp, "unimplemented " + msg);
    }
    fn unimpl(str msg) -> ! { self.bug("unimplemented " + msg); }
    fn get_external_crate(int num) -> crate_metadata { ret crates.get(num); }
    fn set_external_crate(int num, &crate_metadata metadata) {
        crates.insert(num, metadata);
    }
    fn has_external_crate(int num) -> bool { ret crates.contains_key(num); }
    fn add_used_link_args(&str args) {
        used_link_args += str::split(args, ' ' as u8);
    }
    fn get_used_link_args() -> vec[str] {
        ret used_link_args;
    }
    fn add_used_library(&str lib) -> bool {
        if (lib == "") {
            ret false;
        }
        // A program has a small number of libraries, so a vector is probably
        // a good data structure in here.
        for (str l in used_libraries) {
            if (l == lib) {
                ret false;
            }
        }
        used_libraries += [lib];
        ret true;
    }
    fn get_used_libraries() -> vec[str] {
       ret used_libraries;
    }
    fn add_used_crate_file(&str lib) {
        // A program has a small number of crates, so a vector is probably
        // a good data structure in here.
        for (str l in used_crate_files) {
            if (l == lib) {
                ret;
            }
        }
        used_crate_files += [lib];
    }
    fn get_used_crate_files() -> vec[str] {
       ret used_crate_files;
    }
    fn get_codemap() -> codemap::codemap { ret parse_sess.cm; }
    fn lookup_pos(uint pos) -> codemap::loc {
        ret codemap::lookup_pos(parse_sess.cm, pos);
    }
    fn get_parse_sess() -> parse_sess { ret parse_sess; }
    fn next_node_id() -> ast::node_id {
        ret syntax::parse::parser::next_node_id(parse_sess);
    }
    fn span_str(span sp) -> str {
        ret codemap::span_to_str(sp, self.get_codemap());
    }
}
// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
