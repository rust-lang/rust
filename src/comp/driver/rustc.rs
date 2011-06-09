// -*- rust -*-

import front::creader;
import front::parser;
import front::token;
import front::eval;
import front::ast;
import middle::trans;
import middle::resolve;
import middle::ty;
import middle::typeck;
import middle::tstate::ck;
import pretty::pprust;
import pretty::ppaux;
import back::link;
import lib::llvm;
import util::common;

import std::fs;
import std::map::mk_hashmap;
import std::option;
import std::option::some;
import std::option::none;
import std::str;
import std::vec;
import std::io;
import std::run;

import std::getopts;
import std::getopts::optopt;
import std::getopts::optmulti;
import std::getopts::optflag;
import std::getopts::optflagopt;
import std::getopts::opt_present;

import back::link::output_type;

tag pp_mode {
    ppm_normal;
    ppm_typed;
    ppm_identified;
}

fn default_environment(session::session sess,
                       str argv0,
                       str input) -> eval::env {

    auto libc = alt (sess.get_targ_cfg().os) {
        case (session::os_win32) { "msvcrt.dll" }
        case (session::os_macos) { "libc.dylib" }
        case (session::os_linux) { "libc.so.6" }
        case (_) { "libc.so" }
    };

    ret [// Target bindings.
         tup("target_os", eval::val_str(std::os::target_os())),
         tup("target_arch", eval::val_str("x86")),
         tup("target_libc", eval::val_str(libc)),

         // Build bindings.
         tup("build_compiler", eval::val_str(argv0)),
         tup("build_input", eval::val_str(input))
         ];
}

fn parse_input(session::session sess,
               parser::parser p,
               str input) -> @ast::crate {
    ret if (str::ends_with(input, ".rc")) {
        parser::parse_crate_from_crate_file(p)
    } else if (str::ends_with(input, ".rs")) {
       parser::parse_crate_from_source_file(p)
    } else {
        sess.err("unknown input file type: " + input);
        fail
    };
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
    if (sess.get_opts().output_type == link::output_type_none) {ret;}

    auto def_map = time(time_passes, "resolution",
                        bind resolve::resolve_crate(sess, crate));

    auto ty_cx = ty::mk_ctxt(sess, def_map);
    time[()](time_passes, "typechecking",
             bind typeck::check_crate(ty_cx, crate));

    if (sess.get_opts().run_typestate) {
        time(time_passes, "typestate checking",
             bind middle::tstate::ck::check_crate(ty_cx, crate));
    }

    time(time_passes, "alias checking",
         bind middle::alias::check_crate(@ty_cx, def_map, crate));

    auto llmod =
        time[llvm::llvm::ModuleRef](time_passes, "translation",
                                    bind trans::trans_crate(sess, crate,
                                                            ty_cx, output));

    time[()](time_passes, "LLVM passes",
             bind link::write::run_passes(sess, llmod, output));
}

fn pretty_print_input(session::session sess, eval::env env, str input,
                      pp_mode ppm) {
    auto def = tup(ast::local_crate, 0);
    auto p = front::parser::new_parser(sess, env, def, input, 0u, 0u);
    auto crate = parse_input(sess, p, input);

    auto mode;
    alt (ppm) {
        case (ppm_typed) {
            auto def_map = resolve::resolve_crate(sess, crate);
            auto ty_cx = ty::mk_ctxt(sess, def_map);
            typeck::check_crate(ty_cx, crate);
            mode = ppaux::mo_typed(ty_cx);
        }
        case (ppm_normal) { mode = ppaux::mo_untyped; }
        case (ppm_identified) { mode = ppaux::mo_identified; }
    }

    pprust::print_file(sess, crate.node.module, input, std::io::stdout(),
                       mode);
}

fn version(str argv0) {
    auto vers = "unknown version";
    auto env_vers = #env("CFG_VERSION");
    if (str::byte_len(env_vers) != 0u) {
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
    --pretty [type]    pretty-print the input instead of compiling
    --ls               list the symbols defined by a crate file
    -L <path>          add a directory to the library search path
    --noverify         suppress LLVM verification step (slight speedup)
    --depend           print dependencies, in makefile-rule form
    --parse-only       parse only; do not compile, assemble, or link
    -g                 produce debug info
    --OptLevel=        optimize with possible levels 0-3
    -O                 equivalent to --OptLevel=2
    -S                 compile only; do not assemble or link
    -c                 compile and assemble, but do not link
    --emit-llvm        produce an LLVM bitcode file
    --save-temps       write intermediate files in addition to normal output
    --stats            gather and report various compilation statistics
    --time-passes      time the individual phases of the compiler
    --time-llvm-passes time the individual phases of the LLVM backend
    --sysroot <path>   override the system root (default: rustc's directory)
    --no-typestate     don't run the typestate pass (unsafe!)\n\n");
}

fn get_os(str triple) -> session::os {
    ret if (str::find(triple, "win32") >= 0 ||
            str::find(triple, "mingw32") >= 0 ) {
        session::os_win32
    } else if (str::find(triple, "darwin") >= 0) {
        session::os_macos
    } else if (str::find(triple, "linux") >= 0) {
        session::os_linux
    } else {
        log_err "Unknown operating system!";
        fail
    };
}

fn get_arch(str triple) -> session::arch {
    ret if (str::find(triple, "i386") >= 0 ||
            str::find(triple, "i486") >= 0 ||
            str::find(triple, "i586") >= 0 ||
            str::find(triple, "i686") >= 0 ||
            str::find(triple, "i786") >= 0 ) {
        session::arch_x86
    } else if (str::find(triple, "x86_64") >= 0) {
        session::arch_x64
    } else if (str::find(triple, "arm") >= 0 ||
        str::find(triple, "xscale") >= 0 ) {
        session::arch_arm
    }
    else {
        log_err ("Unknown architecture! " + triple);
        fail
    };
}

fn get_default_sysroot(str binary) -> str {
    auto dirname = fs::dirname(binary);
    if (str::eq(dirname, binary)) { ret "."; }
    ret dirname;
}

fn build_target_config() -> @session::config {
    let str triple =
        std::str::rustrt::str_from_cstr(llvm::llvm::LLVMRustGetHostTriple());

    let @session::config target_cfg =
        @rec(os = get_os(triple),
             arch = get_arch(triple),
             int_type = common::ty_i32,
             uint_type = common::ty_u32,
             float_type = common::ty_f64);

    ret target_cfg;
}

fn build_session_options(str binary, getopts::match match)
    -> @session::options {
    auto shared = opt_present(match, "shared");
    auto library_search_paths = getopts::opt_strs(match, "L");

    auto output_type = if (opt_present(match, "parse-only")) {
        link::output_type_none
    } else if (opt_present(match, "S")) {
        link::output_type_assembly
    } else if (opt_present(match, "c")) {
        link::output_type_object
    } else if (opt_present(match, "emit-llvm")) {
        link::output_type_bitcode
    } else {
        link::output_type_exe
    };

    auto verify = !opt_present(match, "noverify");
    auto save_temps = opt_present(match, "save-temps");
    auto debuginfo = opt_present(match, "g");
    auto stats = opt_present(match, "stats");
    auto time_passes = opt_present(match, "time-passes");
    auto time_llvm_passes = opt_present(match, "time-llvm-passes");
    auto run_typestate = !opt_present(match, "no-typestate");
    auto sysroot_opt = getopts::opt_maybe_str(match, "sysroot");

    let uint opt_level = if (opt_present(match, "O")) {
        if (opt_present(match, "OptLevel")) {
            log_err "error: -O and --OptLevel both provided";
            fail;
        }
        2u
    } else if (opt_present(match, "OptLevel")) {
        alt (getopts::opt_str(match, "OptLevel")) {
            case ("0") { 0u }
            case ("1") { 1u }
            case ("2") { 2u }
            case ("3") { 3u }
            case (_) {
                log_err "error: optimization level needs to be between 0-3";
                fail
            }
        }
    } else {
        0u
    };

    auto sysroot = alt (sysroot_opt) {
        case (none) { get_default_sysroot(binary) }
        case (some(?s)) { s }
    };

    let @session::options sopts =
        @rec(shared = shared,
             optimize = opt_level,
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

    ret sopts;
}

fn build_session(@session::options sopts) -> session::session {
    auto target_cfg = build_target_config();
    auto crate_cache = common::new_int_hash[session::crate_metadata]();
    auto target_crate_num = 0;
    let vec[@ast::meta_item] md = [];
    auto sess =
        session::session(target_crate_num, target_cfg, sopts,
                         crate_cache, md, front::codemap::new_codemap());
    ret sess;
}

fn parse_pretty(session::session sess, &str name) -> pp_mode {
    if (str::eq(name, "normal")) { ret ppm_normal; }
    else if (str::eq(name, "typed")) { ret ppm_typed; }
    else if (str::eq(name, "identified")) { ret ppm_identified; }

    sess.err("argument to `pretty` must be one of `normal`, `typed`, or " +
             "`identified`");
}

fn main(vec[str] args) {

    auto opts = [optflag("h"), optflag("help"),
                 optflag("v"), optflag("version"),
                 optflag("glue"), optflag("emit-llvm"),
                 optflagopt("pretty"),
                 optflag("ls"), optflag("parse-only"),
                 optflag("O"), optopt("OptLevel"),
                 optflag("shared"), optmulti("L"),
                 optflag("S"), optflag("c"), optopt("o"), optflag("g"),
                 optflag("save-temps"), optopt("sysroot"),
                 optflag("stats"),
                 optflag("time-passes"), optflag("time-llvm-passes"),
                 optflag("no-typestate"), optflag("noverify")];

    auto binary = vec::shift[str](args);
    auto match = alt (getopts::getopts(args, opts)) {
        case (getopts::success(?m)) { m }
        case (getopts::failure(?f)) {
            log_err #fmt("error: %s", getopts::fail_str(f));
            fail
        }
    };

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

    auto sopts = build_session_options(binary, match);
    auto sess = build_session(sopts);

    auto n_inputs = vec::len[str](match.free);

    auto output_file = getopts::opt_maybe_str(match, "o");
    auto glue = opt_present(match, "glue");
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
    let str saved_out_filename = "";
    auto env = default_environment(sess, binary, ifile);
    auto pretty = option::map[str,pp_mode](bind parse_pretty(sess, _),
        getopts::opt_default(match, "pretty", "normal"));
    auto ls = opt_present(match, "ls");

    alt (pretty) {
        case (some[pp_mode](?ppm)) {
            pretty_print_input(sess, env, ifile, ppm);
            ret;
        }
        case (none[pp_mode]) { /* continue */ }
    }

    if (ls) {
        front::creader::list_file_metadata(ifile, std::io::stdout());
        ret;
    }

    alt (output_file) {
        case (none) {
            let vec[str] parts = str::split(ifile, '.' as u8);
            vec::pop[str](parts);
            saved_out_filename = parts.(0);
            alt (sopts.output_type) {
                case (link::output_type_none) { parts += ["pp"]; }
                case (link::output_type_bitcode) { parts += ["bc"]; }
                case (link::output_type_assembly) { parts += ["s"]; }

                // Object and exe output both use the '.o' extension here
                case (link::output_type_object) { parts += ["o"]; }
                case (link::output_type_exe) { parts += ["o"]; }
            }
            auto ofile = str::connect(parts, ".");
            compile_input(sess, env, ifile, ofile);
        }
        case (some(?ofile)) {
            saved_out_filename = ofile;
            compile_input(sess, env, ifile, ofile);
        }
    }

    // If the user wants an exe generated we need to invoke
    // gcc to link the object file with some libs
    //
    // TODO: Factor this out of main.
    if (sopts.output_type == link::output_type_exe) {

        //FIXME: Should we make the 'stage3's variable here?
        let str glu = "stage3/glue.o";
        let str stage = "-Lstage3";
        let vec[str] gcc_args;
        let str prog = "gcc";
        let str exe_suffix = "";

        // The invocations of gcc share some flags across platforms
        let vec[str] common_cflags = ["-fno-strict-aliasing", "-fPIC",
                           "-Wall", "-fno-rtti", "-fno-exceptions", "-g"];
        let vec[str] common_libs = [stage, "-Lrustllvm", "-Lrt",
                           "-lrustrt", "-lrustllvm", "-lstd", "-lm"];

        alt (sess.get_targ_cfg().os) {
            case (session::os_win32) {
                exe_suffix = ".exe";
                gcc_args = common_cflags + [
                            "-march=i686", "-O2",
                            glu, "-o",
                            saved_out_filename + exe_suffix,
                            saved_out_filename + ".o"] + common_libs;
            }
            case (session::os_macos) {
                gcc_args = common_cflags + [
                           "-arch i386", "-O0", "-m32",
                           glu, "-o",
                           saved_out_filename + exe_suffix,
                           saved_out_filename + ".o"] + common_libs;
            }
            case (session::os_linux) {
                gcc_args = common_cflags + [
                           "-march=i686", "-O2", "-m32",
                           glu, "-o",
                           saved_out_filename + exe_suffix,
                           saved_out_filename + ".o"] + common_libs;
            }
        }

        // We run 'gcc' here
        run::run_program(prog, gcc_args);

        // Clean up on Darwin
        if (sess.get_targ_cfg().os == session::os_macos) {
            run::run_program("dsymutil", [saved_out_filename]);
        }

        // Remove the temporary object file if we aren't saving temps
        if (!sopts.save_temps) {
            run::run_program("rm", [saved_out_filename + ".o"]);
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
