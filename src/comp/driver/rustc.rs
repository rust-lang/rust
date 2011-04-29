// -*- rust -*-

import front.creader;
import front.parser;
import front.token;
import front.eval;
import middle.trans;
import middle.resolve;
import middle.capture;
import middle.ty;
import middle.typeck;
import middle.typestate_check;
import util.common;

import std.map.mk_hashmap;
import std.option;
import std.option.some;
import std.option.none;
import std._str;
import std._vec;
import std.io;

import std.GetOpts;
import std.GetOpts.optopt;
import std.GetOpts.optmulti;
import std.GetOpts.optflag;
import std.GetOpts.opt_present;

fn default_environment(session.session sess,
                       str argv0,
                       str input) -> eval.env {

    auto libc = "libc.so";
    alt (sess.get_targ_cfg().os) {
        case (session.os_win32) { libc = "msvcrt.dll"; }
        case (session.os_macos) { libc = "libc.dylib"; }
        case (session.os_linux) { libc = "libc.so.6"; }
    }

    ret
        vec(
            // Target bindings.
            tup("target_os", eval.val_str(std.os.target_os())),
            tup("target_arch", eval.val_str("x86")),
            tup("target_libc", eval.val_str(libc)),

            // Build bindings.
            tup("build_compiler", eval.val_str(argv0)),
            tup("build_input", eval.val_str(input))
            );
}

fn parse_input(session.session sess,
                      parser.parser p,
                      str input) -> @front.ast.crate {
    if (_str.ends_with(input, ".rc")) {
        ret parser.parse_crate_from_crate_file(p);
    } else if (_str.ends_with(input, ".rs")) {
        ret parser.parse_crate_from_source_file(p);
    }
    sess.err("unknown input file type: " + input);
    fail;
}

fn compile_input(session.session sess,
                        eval.env env,
                        str input, str output,
                        bool shared,
                        bool optimize,
                        bool verify,
                        bool save_temps,
                        trans.output_type ot,
                        vec[str] library_search_paths) {
    auto def = tup(0, 0);
    auto p = parser.new_parser(sess, env, def, input, 0u);
    auto crate = parse_input(sess, p, input);
    if (ot == trans.output_type_none) {ret;}

    crate = creader.read_crates(sess, crate, library_search_paths);
    crate = resolve.resolve_crate(sess, crate);
    capture.check_for_captures(sess, crate);

    auto ty_cx = ty.mk_ctxt(sess);
    auto typeck_result = typeck.check_crate(ty_cx, crate);
    crate = typeck_result._0;
    auto type_cache = typeck_result._1;
    crate = typestate_check.check_crate(crate);
    trans.trans_crate(sess, crate, ty_cx, type_cache, output, shared,
                      optimize, verify, save_temps, ot);
}

fn pretty_print_input(session.session sess,
                             eval.env env,
                             str input) {
    auto def = tup(0, 0);
    auto p = front.parser.new_parser(sess, env, def, input, 0u);
    auto crate = front.parser.parse_crate_from_source_file(p);
    pretty.pprust.print_file(crate.node.module, input, std.io.stdout());
}

fn usage(session.session sess, str argv0) {
    io.stdout().write_str(#fmt("usage: %s [options] <input>\n", argv0) + "
options:

    -o <filename>      write output to <filename>
    --glue             generate glue.bc file
    --shared           compile a shared-library crate
    --pretty           pretty-print the input instead of compiling
    --ls               list the symbols defined by a crate file
    -L <path>          add a directory to the library search path
    --noverify         suppress LLVM verification step (slight speedup)
    --parse-only       parse only; do not compile, assemble, or link
    -O                 optimize
    -S                 compile only; do not assemble or link
    -c                 compile and assemble, but do not link
    --save-temps       write intermediate files in addition to normal output
    -h                 display this message\n\n");
}

fn get_os() -> session.os {
    auto s = std.os.target_os();
    if (_str.eq(s, "win32")) { ret session.os_win32; }
    if (_str.eq(s, "macos")) { ret session.os_macos; }
    if (_str.eq(s, "linux")) { ret session.os_linux; }
}

fn main(vec[str] args) {

    // FIXME: don't hard-wire this.
    auto target_cfg = rec(os = get_os(),
                          arch = session.arch_x86,
                          int_type = common.ty_i32,
                          uint_type = common.ty_u32,
                          float_type = common.ty_f64 );

    auto crate_cache = common.new_int_hash[session.crate_metadata]();
    auto target_crate_num = 0;
    let vec[@front.ast.meta_item] md = vec();
    auto sess = session.session(target_crate_num, target_cfg, crate_cache,
                                md, front.codemap.new_codemap());

    auto opts = vec(optflag("h"), optflag("glue"),
                    optflag("pretty"), optflag("ls"), optflag("parse-only"),
                    optflag("O"), optflag("shared"), optmulti("L"),
                    optflag("S"), optflag("c"), optopt("o"),
                    optflag("save-temps"), optflag("noverify"));
    auto binary = _vec.shift[str](args);
    auto match;
    alt (GetOpts.getopts(args, opts)) {
        case (GetOpts.failure(?f)) { sess.err(GetOpts.fail_str(f)); fail; }
        case (GetOpts.success(?m)) { match = m; }
    }
    if (opt_present(match, "h")) {
        usage(sess, binary);
        ret;
    }

    auto pretty = opt_present(match, "pretty");
    auto ls = opt_present(match, "ls");
    auto glue = opt_present(match, "glue");
    auto shared = opt_present(match, "shared");
    auto output_file = GetOpts.opt_maybe_str(match, "o");
    auto library_search_paths = GetOpts.opt_strs(match, "L");
    auto ot = trans.output_type_bitcode;
    if (opt_present(match, "parse-only")) {
        ot = trans.output_type_none;
    } else if (opt_present(match, "S")) {
        ot = trans.output_type_assembly;
    } else if (opt_present(match, "c")) {
        ot = trans.output_type_object;
    }
    auto verify = !opt_present(match, "noverify");
    auto save_temps = opt_present(match, "save-temps");
    // FIXME: Maybe we should support -O0, -O1, -Os, etc
    auto optimize = opt_present(match, "O");
    auto n_inputs = _vec.len[str](match.free);

    if (glue) {
        if (n_inputs > 0u) {
            sess.err("No input files allowed with --glue.");
        }
        auto out = option.from_maybe[str]("glue.bc", output_file);
        middle.trans.make_common_glue(out, optimize, verify, save_temps, ot);
        ret;
    }

    if (n_inputs == 0u) {
        sess.err("No input filename given.");
    } else if (n_inputs > 1u) {
        sess.err("Multiple input filenames provided.");
    }

    auto ifile = match.free.(0);
    auto env = default_environment(sess, args.(0), ifile);
    if (pretty) {
        pretty_print_input(sess, env, ifile);
    } else if (ls) {
        front.creader.list_file_metadata(ifile, std.io.stdout());
    } else {
        alt (output_file) {
            case (none[str]) {
                let vec[str] parts = _str.split(ifile, '.' as u8);
                _vec.pop[str](parts);
                parts += vec(".bc");
                auto ofile = _str.concat(parts);
                compile_input(sess, env, ifile, ofile, shared,
                              optimize, verify, save_temps, ot,
                              library_search_paths);
            }
            case (some[str](?ofile)) {
                compile_input(sess, env, ifile, ofile, shared,
                              optimize, verify, save_temps, ot,
                              library_search_paths);
            }
        }
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
