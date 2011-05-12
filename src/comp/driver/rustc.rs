// -*- rust -*-

import front::creader;
import front::parser;
import front::token;
import front::eval;
import front::ast;
import middle::trans;
import middle::resolve;
import middle::capture;
import middle::ty;
import middle::typeck;
import middle::typestate_check;
import back::Link;
import lib::llvm;
import util::common;

import std::fs;
import std::map::mk_hashmap;
import std::option;
import std::option::some;
import std::option::none;
import std::_str;
import std::_vec;
import std::io;

import std::getopts;
import std::getopts::optopt;
import std::getopts::optmulti;
import std::getopts::optflag;
import std::getopts::opt_present;

import back::Link::output_type;

fn default_environment(session::session sess,
                       str argv0,
                       str input) -> eval::env {

    auto libc = "libc::so";
    alt (sess.get_targ_cfg().os) {
        case (session::os_win32) { libc = "msvcrt.dll"; }
        case (session::os_macos) { libc = "libc::dylib"; }
        case (session::os_linux) { libc = "libc::so.6"; }
    }

    ret
        vec(
            // Target bindings.
            tup("target_os", eval::val_str(std::os::target_os())),
            tup("target_arch", eval::val_str("x86")),
            tup("target_libc", eval::val_str(libc)),

            // Build bindings.
            tup("build_compiler", eval::val_str(argv0)),
            tup("build_input", eval::val_str(input))
            );
}

fn parse_input(session::session sess,
                      parser::parser p,
                      str input) -> @ast::crate {
    if (_str::ends_with(input, ".rc")) {
        ret parser::parse_crate_from_crate_file(p);
    } else if (_str::ends_with(input, ".rs")) {
        ret parser::parse_crate_from_source_file(p);
    }
    sess.err("unknown input file type: " + input);
    fail;
}

fn time[T](bool do_it, str what, fn()->T thunk) -> T {
    if (!do_it) { ret thunk(); }

    auto start = std::time::get_time();
    auto rv = thunk();
    auto end = std::time::get_time();

    // FIXME: Actually do timeval math.
    log_err #fmt("time: %s took %u s", what, (end.sec - start.sec) as uint);
    ret rv;
}

fn compile_input(session::session sess,
                 eval::env env,
                 str input, str output) {
    auto time_passes = sess.get_opts().time_passes;
    auto def = tup(ast::local_crate, 0);
    auto p = parser::new_parser(sess, env, def, input, 0u, 0u);
    auto crate = time(time_passes, "parsing",
                      bind parse_input(sess, p, input));
    if (sess.get_opts().output_type == Link::output_type_none) {ret;}

    crate = time(time_passes, "external crate reading",
                 bind creader::read_crates(sess, crate));
    auto res = time(time_passes, "resolution",
                    bind resolve::resolve_crate(sess, crate));
    crate = res._0;
    auto def_map = res._1;
    time[()](time_passes, "capture checking",
             bind capture::check_for_captures(sess, crate, def_map));

    auto ty_cx = ty::mk_ctxt(sess, def_map);
    auto typeck_result =
        time[typeck::typecheck_result](time_passes, "typechecking",
                                      bind typeck::check_crate(ty_cx, crate));
    crate = typeck_result._0;
    auto type_cache = typeck_result._1;

    if (sess.get_opts().run_typestate) {
        crate = time(time_passes, "typestate checking",
                     bind typestate_check::check_crate(crate, def_map));
    }

    auto llmod = time[llvm::ModuleRef](time_passes, "translation",
        bind trans::trans_crate(sess, crate, ty_cx, type_cache, output));

    time[()](time_passes, "LLVM passes",
             bind Link::Write::run_passes(sess, llmod, output));
}

fn pretty_print_input(session::session sess,
                             eval::env env,
                             str input) {
    auto def = tup(ast::local_crate, 0);
    auto p = front::parser::new_parser(sess, env, def, input, 0u, 0u);
    auto crate = front::parser::parse_crate_from_source_file(p);
    pretty::pprust::print_file(sess, crate.node.module, input,
                               std::io::stdout());
}

fn version(str argv0) {
    auto vers = "unknown version";
    auto env_vers = #env("CFG_VERSION");
    if (_str::byte_len(env_vers) != 0u) {
        vers = env_vers;
    }
    io::stdout().write_str(#fmt("%s %s\n", argv0, vers));
}

fn usage(str argv0) {
    io::stdout().write_str(#fmt("usage: %s [options] <input>\n", argv0) + "
options:

    -h --help          display this message
    -v --version       print version info and exit

    -o <filename>      write output to <filename>
    --glue             generate glue.bc file
    --shared           compile a shared-library crate
    --pretty           pretty-print the input instead of compiling
    --ls               list the symbols defined by a crate file
    -L <path>          add a directory to the library search path
    --noverify         suppress LLVM verification step (slight speedup)
    --depend           print dependencies, in makefile-rule form
    --parse-only       parse only; do not compile, assemble, or link
    -g                 produce debug info
    -O                 optimize
    -S                 compile only; do not assemble or link
    -c                 compile and assemble, but do not link
    --save-temps       write intermediate files in addition to normal output
    --stats            gather and report various compilation statistics
    --time-passes      time the individual phases of the compiler
    --time-llvm-passes time the individual phases of the LLVM backend
    --sysroot <path>   override the system root (default: rustc's directory)
    --no-typestate     don't run the typestate pass (unsafe!)\n\n");
}

fn get_os(str triple) -> session::os {
    if (_str::find(triple, "win32") > 0 ||
        _str::find(triple, "mingw32") > 0 ) {
        ret session::os_win32;
    } else if (_str::find(triple, "darwin") > 0) { ret session::os_macos; }
    else if (_str::find(triple, "linux") > 0) { ret session::os_linux; }
}

fn get_arch(str triple) -> session::arch {
    if (_str::find(triple, "i386") > 0 ||
        _str::find(triple, "i486") > 0 ||
        _str::find(triple, "i586") > 0 ||
        _str::find(triple, "i686") > 0 ||
        _str::find(triple, "i786") > 0 ) {
        ret session::arch_x86;
    } else if (_str::find(triple, "x86_64") > 0) {
        ret session::arch_x64;
    } else if (_str::find(triple, "arm") > 0 ||
        _str::find(triple, "xscale") > 0 ) {
        ret session::arch_arm;
    }
}

fn get_default_sysroot(str binary) -> str {
    auto dirname = fs::dirname(binary);
    if (_str::eq(dirname, binary)) { ret "."; }
    ret dirname;
}

fn main(vec[str] args) {

    let str triple =
        std::_str::rustrt::str_from_cstr(llvm::llvm::LLVMRustGetHostTriple());

    let @session::config target_cfg =
        @rec(os = get_os(triple),
             arch = get_arch(triple),
             int_type = common::ty_i32,
             uint_type = common::ty_u32,
             float_type = common::ty_f64);

    auto opts = vec(optflag("h"), optflag("help"),
                    optflag("v"), optflag("version"),
                    optflag("glue"),
                    optflag("pretty"), optflag("ls"), optflag("parse-only"),
                    optflag("O"), optflag("shared"), optmulti("L"),
                    optflag("S"), optflag("c"), optopt("o"), optflag("g"),
                    optflag("save-temps"), optopt("sysroot"),
                    optflag("stats"),
                    optflag("time-passes"), optflag("time-llvm-passes"),
                    optflag("no-typestate"), optflag("noverify"));
    auto binary = _vec::shift[str](args);
    auto match;
    alt (getopts::getopts(args, opts)) {
        case (getopts::failure(?f)) {
            log_err #fmt("error: %s", getopts::fail_str(f));
            fail;
        }
        case (getopts::success(?m)) { match = m; }
    }
    if (opt_present(match, "h") ||
        opt_present(match, "help")) {
        usage(binary);
        ret;
    }

    if (opt_present(match, "v") ||
        opt_present(match, "version")) {
        version(binary);
        ret;
    }

    auto pretty = opt_present(match, "pretty");
    auto ls = opt_present(match, "ls");
    auto glue = opt_present(match, "glue");
    auto shared = opt_present(match, "shared");
    auto output_file = getopts::opt_maybe_str(match, "o");
    auto library_search_paths = getopts::opt_strs(match, "L");

    auto output_type = Link::output_type_bitcode;
    if (opt_present(match, "parse-only")) {
        output_type = Link::output_type_none;
    } else if (opt_present(match, "S")) {
        output_type = Link::output_type_assembly;
    } else if (opt_present(match, "c")) {
        output_type = Link::output_type_object;
    }

    auto verify = !opt_present(match, "noverify");
    auto save_temps = opt_present(match, "save-temps");
    // FIXME: Maybe we should support -O0, -O1, -Os, etc
    auto optimize = opt_present(match, "O");
    auto debuginfo = opt_present(match, "g");
    auto stats = opt_present(match, "stats");
    auto time_passes = opt_present(match, "time-passes");
    auto time_llvm_passes = opt_present(match, "time-llvm-passes");
    auto run_typestate = !opt_present(match, "no-typestate");
    auto sysroot_opt = getopts::opt_maybe_str(match, "sysroot");

    auto sysroot;
    alt (sysroot_opt) {
        case (none[str]) { sysroot = get_default_sysroot(binary); }
        case (some[str](?s)) { sysroot = s; }
    }

    let @session::options sopts =
        @rec(shared = shared,
             optimize = optimize,
             debuginfo = debuginfo,
             verify = verify,
             run_typestate = run_typestate,
             save_temps = save_temps,
             stats = stats,
             time_passes = time_passes,
             time_llvm_passes = time_llvm_passes,
             output_type = output_type,
             library_search_paths = library_search_paths,
             sysroot = sysroot);

    auto crate_cache = common::new_int_hash[session::crate_metadata]();
    auto target_crate_num = 0;
    let vec[@ast::meta_item] md = vec();
    auto sess =
        session::session(target_crate_num, target_cfg, sopts,
                        crate_cache, md, front::codemap::new_codemap());

    auto n_inputs = _vec::len[str](match.free);

    if (glue) {
        if (n_inputs > 0u) {
            sess.err("No input files allowed with --glue.");
        }
        auto out = option::from_maybe[str]("glue.bc", output_file);
        middle::trans::make_common_glue(sess, out);
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
        front::creader::list_file_metadata(ifile, std::io::stdout());
    } else {
        alt (output_file) {
            case (none[str]) {
                let vec[str] parts = _str::split(ifile, '.' as u8);
                _vec::pop[str](parts);
                alt (output_type) {
                    case (Link::output_type_none) { parts += vec("pp"); }
                    case (Link::output_type_bitcode) { parts += vec("bc"); }
                    case (Link::output_type_assembly) { parts += vec("s"); }
                    case (Link::output_type_object) { parts += vec("o"); }
                }
                auto ofile = _str::connect(parts, ".");
                compile_input(sess, env, ifile, ofile);
            }
            case (some[str](?ofile)) {
                compile_input(sess, env, ifile, ofile);
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
