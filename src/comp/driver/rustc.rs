

// -*- rust -*-
import metadata::creader;
import metadata::cstore;
import syntax::parse::parser;
import syntax::parse::token;
import syntax::ast;
import syntax::codemap;
import front::attr;
import middle::trans;
import middle::resolve;
import middle::freevars;
import middle::kind;
import middle::ty;
import middle::typeck;
import middle::tstate::ck;
import syntax::print::pp;
import syntax::print::pprust;
import util::ppaux;
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
import std::int;
import std::io;
import std::run;
import std::getopts;
import std::getopts::optopt;
import std::getopts::optmulti;
import std::getopts::optflag;
import std::getopts::optflagopt;
import std::getopts::opt_present;
import back::link::output_type;

tag pp_mode { ppm_normal; ppm_expanded; ppm_typed; ppm_identified; }

fn default_configuration(sess: session::session,
                         argv0: &istr, input: &istr) ->
   ast::crate_cfg {
    let libc =
        alt sess.get_targ_cfg().os {
          session::os_win32. { ~"msvcrt.dll" }
          session::os_macos. { ~"libc.dylib" }
          session::os_linux. { ~"libc.so.6" }
          _ { ~"libc.so" }
        };

    let mk = attr::mk_name_value_item_str;

    ret [ // Target bindings.
         mk(~"target_os", std::os::target_os()),
        mk(~"target_arch", ~"x86"),
         mk(~"target_libc", libc),
         // Build bindings.
         mk(~"build_compiler", argv0),
        mk(~"build_input", input)];
}

fn build_configuration(sess: session::session, argv0: &istr, input: &istr) ->
   ast::crate_cfg {
    // Combine the configuration requested by the session (command line) with
    // some default and generated configuration items
    let default_cfg = default_configuration(sess, argv0, input);
    let user_cfg = sess.get_opts().cfg;
    // If the user wants a test runner, then add the test cfg
    let gen_cfg =
        {
            if sess.get_opts().test
                && !attr::contains_name(user_cfg, ~"test") {
                [attr::mk_word_item(~"test")]
            } else { [] }
        };
    ret user_cfg + gen_cfg + default_cfg;
}

// Convert strings provided as --cfg [cfgspec] into a crate_cfg
fn parse_cfgspecs(cfgspecs: &[istr]) -> ast::crate_cfg {
    // FIXME: It would be nice to use the parser to parse all varieties of
    // meta_item here. At the moment we just support the meta_word variant.
    let words = [];
    for s: istr in cfgspecs {
        words += [attr::mk_word_item(s)];
    }
    ret words;
}

fn input_is_stdin(filename: &istr) -> bool { filename == ~"-" }

fn parse_input(sess: session::session, cfg: &ast::crate_cfg,
               input: &istr) -> @ast::crate {
    if !input_is_stdin(input) {
        parser::parse_crate_from_file(
            input, cfg, sess.get_parse_sess())
    } else { parse_input_src(sess, cfg, input).crate }
}

fn parse_input_src(sess: session::session, cfg: &ast::crate_cfg,
                   infile: &istr) -> {crate: @ast::crate, src: istr} {
    let srcbytes =
        if infile != ~"-" {
            io::file_reader(infile)
        } else { io::stdin() }.read_whole_stream();
    let src = str::unsafe_from_bytes(srcbytes);
    let crate =
        parser::parse_crate_from_source_str(
            infile,
            src, cfg,
            sess.get_parse_sess());
    ret {crate: crate, src: src};
}

fn time<@T>(do_it: bool, what: &istr, thunk: fn() -> T) -> T {
    if !do_it { ret thunk(); }
    let start = std::time::precise_time_s();
    let rv = thunk();
    let end = std::time::precise_time_s();
    log_err #ifmt["time: %s took %s s", what,
                 common::float_to_str(end - start, 3u)];
    ret rv;
}

fn compile_input(sess: session::session, cfg: ast::crate_cfg, input: &istr,
                 output: &istr) {
    let time_passes = sess.get_opts().time_passes;
    let crate =
        time(time_passes, ~"parsing", bind parse_input(sess, cfg, input));
    if sess.get_opts().parse_only { ret; }
    crate =
        time(time_passes, ~"configuration",
             bind front::config::strip_unconfigured_items(crate));
    if sess.get_opts().test {
        crate =
            time(time_passes, ~"building test harness",
                 bind front::test::modify_for_testing(crate));
    }
    crate =
        time(time_passes, ~"expansion",
             bind syntax::ext::expand::expand_crate(sess, crate));

    let ast_map =
        time(time_passes, ~"ast indexing",
             bind middle::ast_map::map_crate(*crate));
    time(time_passes, ~"external crate/lib resolution",
         bind creader::read_crates(sess, *crate));
    let {def_map: def_map, ext_map: ext_map} =
        time(time_passes, ~"resolution",
             bind resolve::resolve_crate(sess, ast_map, crate));
    let freevars =
        time(time_passes, ~"freevar finding",
             bind freevars::annotate_freevars(def_map, crate));
    let ty_cx = ty::mk_ctxt(sess, def_map, ext_map, ast_map, freevars);
    time(time_passes, ~"typechecking",
         bind typeck::check_crate(ty_cx, crate));
    time(time_passes, ~"alt checking",
         bind middle::check_alt::check_crate(ty_cx, crate));
    if sess.get_opts().run_typestate {
        time(time_passes, ~"typestate checking",
             bind middle::tstate::ck::check_crate(ty_cx, crate));
    }
    let mut_map = time(time_passes, ~"mutability checking",
                       bind middle::mut::check_crate(ty_cx, crate));
    time(time_passes, ~"alias checking",
         bind middle::alias::check_crate(ty_cx, crate));
    time(time_passes, ~"kind checking",
         bind kind::check_crate(ty_cx, crate));
    if sess.get_opts().no_trans { ret; }
    let llmod = time(time_passes, ~"translation",
                     bind trans::trans_crate(sess, crate, ty_cx,
                                             output,
                                             ast_map, mut_map));
    time(time_passes, ~"LLVM passes",
         bind link::write::run_passes(sess, llmod, output));
}

fn pretty_print_input(sess: session::session, cfg: ast::crate_cfg,
                      input: &istr, ppm: pp_mode) {
    fn ann_paren_for_expr(node: &pprust::ann_node) {
        alt node { pprust::node_expr(s, expr) { pprust::popen(s); } _ { } }
    }
    fn ann_typed_post(tcx: &ty::ctxt, node: &pprust::ann_node) {
        alt node {
          pprust::node_expr(s, expr) {
            pp::space(s.s);
            pp::word(s.s, ~"as");
            pp::space(s.s);
            pp::word(
                s.s,
                ppaux::ty_to_str(tcx, ty::expr_ty(tcx, expr)));
            pprust::pclose(s);
          }
          _ { }
        }
    }
    fn ann_identified_post(node: &pprust::ann_node) {
        alt node {
          pprust::node_item(s, item) {
            pp::space(s.s);
            pprust::synth_comment(
                s, int::to_str(item.id, 10u));
          }
          pprust::node_block(s, blk) {
            pp::space(s.s);
            pprust::synth_comment(
                s, ~"block " + int::to_str(blk.node.id, 10u));
          }
          pprust::node_expr(s, expr) {
            pp::space(s.s);
            pprust::synth_comment(
                s, int::to_str(expr.id, 10u));
            pprust::pclose(s);
          }
          _ { }
        }
    }

    // Because the pretty printer needs to make a pass over the source
    // to collect comments and literals, and we need to support reading
    // from stdin, we're going to just suck the source into a string
    // so both the parser and pretty-printer can use it.
    let crate_src = parse_input_src(sess, cfg, input);
    let crate = crate_src.crate;
    let src = crate_src.src;

    let ann;
    alt ppm {
      ppm_expanded. {
        crate = syntax::ext::expand::expand_crate(sess, crate);
        ann = pprust::no_ann();
      }
      ppm_typed. {
        crate = syntax::ext::expand::expand_crate(sess, crate);
        let amap = middle::ast_map::map_crate(*crate);
        let {def_map: def_map, ext_map: ext_map} =
            resolve::resolve_crate(sess, amap, crate);
        let freevars = freevars::annotate_freevars(def_map, crate);
        let ty_cx = ty::mk_ctxt(sess, def_map, ext_map, amap, freevars);
        typeck::check_crate(ty_cx, crate);
        ann = {pre: ann_paren_for_expr, post: bind ann_typed_post(ty_cx, _)};
      }
      ppm_identified. {
        ann = {pre: ann_paren_for_expr, post: ann_identified_post};
      }
      ppm_normal. { ann = pprust::no_ann(); }
    }
    pprust::print_crate(sess.get_codemap(), crate,
                        input,
                        io::string_reader(src),
                        io::stdout(), ann);
}

fn version(argv0: &istr) {
    let vers = ~"unknown version";
    // FIXME: Restore after istr conversion
    //let env_vers = #env["CFG_VERSION"];
    let env_vers = ~"FIXME";
    if str::byte_len(env_vers) != 0u { vers = env_vers; }
    io::stdout().write_str(
        #ifmt["%s %s\n",
                             argv0,
                             vers]);
}

fn usage(argv0: &istr) {
    io::stdout().write_str(
        #ifmt["usage: %s [options] <input>\n", argv0] +
                               ~"
options:

    -h --help          display this message
    -v --version       print version info and exit

    -o <filename>      write output to <filename>
    --glue             generate glue.bc file
    --lib              compile a library crate
    --static           use or produce static libraries
    --pretty [type]    pretty-print the input instead of compiling
    --ls               list the symbols defined by a crate file
    -L <path>          add a directory to the library search path
    --noverify         suppress LLVM verification step (slight speedup)
    --depend           print dependencies, in makefile-rule form
    --parse-only       parse only; do not compile, assemble, or link
    --no-trans         run all passes except translation; no output
    -g                 produce debug info
    --OptLevel=        optimize with possible levels 0-3
    -O                 equivalent to --OptLevel=2
    -S                 compile only; do not assemble or link
    -c                 compile and assemble, but do not link
    --emit-llvm        produce an LLVM bitcode file
    --save-temps       write intermediate files in addition to normal output
    --stats            gather and report various compilation statistics
    --cfg [cfgspec]    configure the compilation environment
    --time-passes      time the individual phases of the compiler
    --time-llvm-passes time the individual phases of the LLVM backend
    --sysroot <path>   override the system root (default: rustc's directory)
    --no-typestate     don't run the typestate pass (unsafe!)
    --test             build test harness
    --gc               garbage collect shared data (experimental/temporary)

");
}

fn get_os(triple: &istr) -> session::os {
    ret if str::find(triple, ~"win32") >= 0 ||
               str::find(triple, ~"mingw32") >= 0 {
            session::os_win32
        } else if str::find(triple, ~"darwin") >= 0 {
            session::os_macos
        } else if str::find(triple, ~"linux") >= 0 {
            session::os_linux
        } else { log_err ~"Unknown operating system!"; fail };
}

fn get_arch(triple: &istr) -> session::arch {
    ret if str::find(triple, ~"i386") >= 0 ||
        str::find(triple, ~"i486") >= 0 ||
               str::find(triple, ~"i586") >= 0 ||
               str::find(triple, ~"i686") >= 0 ||
               str::find(triple, ~"i786") >= 0 {
            session::arch_x86
        } else if str::find(triple, ~"x86_64") >= 0 {
            session::arch_x64
        } else if str::find(triple, ~"arm") >= 0 ||
                      str::find(triple, ~"xscale") >= 0 {
            session::arch_arm
        } else { log_err ~"Unknown architecture! " + triple; fail };
}

fn get_default_sysroot(binary: &istr) -> istr {
    let dirname = fs::dirname(binary);
    if str::eq(dirname, binary) { ret ~"."; }
    ret dirname;
}

fn build_target_config() -> @session::config {
    let triple: istr =
        str::str_from_cstr(llvm::llvm::LLVMRustGetHostTriple());
    let target_cfg: @session::config =
        @{os: get_os(triple),
          arch: get_arch(triple),
          int_type: ast::ty_i32,
          uint_type: ast::ty_u32,
          float_type: ast::ty_f64};
    ret target_cfg;
}

fn build_session_options(binary: &istr, match: &getopts::match,
                         binary_dir: &istr) -> @session::options {
    let library = opt_present(match, ~"lib");
    let static = opt_present(match, ~"static");

    let library_search_paths = [binary_dir + ~"/lib"];
    let lsp_vec = getopts::opt_strs(match, ~"L");
    for lsp: istr in lsp_vec {
        library_search_paths += [lsp];
    }

    let parse_only = opt_present(match, ~"parse-only");
    let no_trans = opt_present(match, ~"no-trans");

    let output_type =
        if parse_only || no_trans {
            link::output_type_none
        } else if opt_present(match, ~"S") {
            link::output_type_assembly
        } else if opt_present(match, ~"c") {
            link::output_type_object
        } else if opt_present(match, ~"emit-llvm") {
            link::output_type_bitcode
        } else { link::output_type_exe };
    let verify = !opt_present(match, ~"noverify");
    let save_temps = opt_present(match, ~"save-temps");
    let debuginfo = opt_present(match, ~"g");
    let stats = opt_present(match, ~"stats");
    let time_passes = opt_present(match, ~"time-passes");
    let time_llvm_passes = opt_present(match, ~"time-llvm-passes");
    let run_typestate = !opt_present(match, ~"no-typestate");
    let sysroot_opt = getopts::opt_maybe_str(match, ~"sysroot");
    let opt_level: uint =
        if opt_present(match, ~"O") {
            if opt_present(match, ~"OptLevel") {
                log_err "error: -O and --OptLevel both provided";
                fail;
            }
            2u
        } else if opt_present(match, ~"OptLevel") {
            alt getopts::opt_str(match, ~"OptLevel") {
              ~"0" { 0u }
              ~"1" { 1u }
              ~"2" { 2u }
              ~"3" { 3u }
              _ {
                log_err "error: optimization level needs " +
                            "to be between 0-3";
                fail
              }
            }
        } else { 0u };
    let sysroot =
        alt sysroot_opt {
          none. { get_default_sysroot(binary) }
          some(s) { s }
        };
    let cfg = parse_cfgspecs(
        getopts::opt_strs(match, ~"cfg"));
    let test = opt_present(match, ~"test");
    let do_gc = opt_present(match, ~"gc");
    let sopts: @session::options =
        @{library: library,
          static: static,
          optimize: opt_level,
          debuginfo: debuginfo,
          verify: verify,
          run_typestate: run_typestate,
          save_temps: save_temps,
          stats: stats,
          time_passes: time_passes,
          time_llvm_passes: time_llvm_passes,
          output_type: output_type,
          library_search_paths: library_search_paths,
          sysroot: sysroot,
          cfg: cfg,
          test: test,
          parse_only: parse_only,
          no_trans: no_trans,
          do_gc: do_gc};
    ret sopts;
}

fn build_session(sopts: @session::options) -> session::session {
    let target_cfg = build_target_config();
    let cstore = cstore::mk_cstore();
    ret session::session(target_cfg, sopts, cstore,
                         @{cm: codemap::new_codemap(), mutable next_id: 0},
                         none, 0u);
}

fn parse_pretty(sess: session::session, name: &istr) -> pp_mode {
    if str::eq(name, ~"normal") {
        ret ppm_normal;
    } else if str::eq(name, ~"expanded") {
        ret ppm_expanded;
    } else if str::eq(name, ~"typed") {
        ret ppm_typed;
    } else if str::eq(name, ~"identified") { ret ppm_identified; }
    sess.fatal(~"argument to `pretty` must be one of `normal`, `typed`, or "
               + ~"`identified`");
}

fn opts() -> [getopts::opt] {
    ret [optflag(~"h"), optflag(~"help"), optflag(~"v"), optflag(~"version"),
         optflag(~"glue"), optflag(~"emit-llvm"), optflagopt(~"pretty"),
         optflag(~"ls"), optflag(~"parse-only"), optflag(~"no-trans"),
         optflag(~"O"), optopt(~"OptLevel"), optmulti(~"L"),
         optflag(~"S"), optflag(~"c"), optopt(~"o"), optflag(~"g"),
         optflag(~"save-temps"), optopt(~"sysroot"), optflag(~"stats"),
         optflag(~"time-passes"), optflag(~"time-llvm-passes"),
         optflag(~"no-typestate"), optflag(~"noverify"), optmulti(~"cfg"),
         optflag(~"test"), optflag(~"lib"), optflag(~"static"),
         optflag(~"gc")];
}

fn main(args: [istr]) {
    let binary = vec::shift(args);
    let binary_dir = fs::dirname(binary);
    let match =
        alt getopts::getopts(args, opts()) {
          getopts::success(m) { m }
          getopts::failure(f) {
            log_err #ifmt["error: %s", getopts::fail_str(f)];
            fail
          }
        };
    if opt_present(match, ~"h") || opt_present(match, ~"help") {
        usage(binary);
        ret;
    }
    if opt_present(match, ~"v") || opt_present(match, ~"version") {
        version(binary);
        ret;
    }
    let sopts = build_session_options(binary, match, binary_dir);
    let sess = build_session(sopts);
    let n_inputs = vec::len::<istr>(match.free);
    let output_file = getopts::opt_maybe_str(match, ~"o");
    let glue = opt_present(match, ~"glue");
    if glue {
        if n_inputs > 0u {
            sess.fatal(~"No input files allowed with --glue.");
        }
        let out = option::from_maybe::<istr>(~"glue.bc", output_file);
        middle::trans::make_common_glue(sess, out);
        ret;
    }
    if n_inputs == 0u {
        sess.fatal(~"No input filename given.");
    } else if n_inputs > 1u {
        sess.fatal(~"Multiple input filenames provided.");
    }
    let ifile = match.free[0];
    let saved_out_filename: istr = ~"";
    let cfg = build_configuration(sess, binary,
                                  ifile);
    let pretty =
        option::map::<istr,
                      pp_mode>(bind parse_pretty(sess, _),
                               getopts::opt_default(match, ~"pretty",
                                                    ~"normal"));
    alt pretty {
      some::<pp_mode>(ppm) {
        pretty_print_input(sess, cfg, ifile, ppm);
        ret;
      }
      none::<pp_mode>. {/* continue */ }
    }
    let ls = opt_present(match, ~"ls");
    if ls { metadata::creader::list_file_metadata(ifile, io::stdout()); ret; }

    let stop_after_codegen =
        sopts.output_type != link::output_type_exe ||
            sopts.static && sopts.library;

    alt output_file {
      none. {
        // "-" as input file will cause the parser to read from stdin so we
        // have to make up a name
        // We want to toss everything after the final '.'
        let parts =
            if !input_is_stdin(ifile) {
                str::split(ifile, '.' as u8)
            } else { [~"default", ~"rs"] };
        vec::pop(parts);
        saved_out_filename = str::connect(parts, ~".");
        let suffix =
            alt sopts.output_type {
              link::output_type_none. { ~"none" }
              link::output_type_bitcode. { ~"bc" }
              link::output_type_assembly. { ~"s" }

              // Object and exe output both use the '.o' extension here
              link::output_type_object. | link::output_type_exe. {
                ~"o"
              }
            };
        let ofile = saved_out_filename + ~"." + suffix;
        compile_input(sess, cfg, ifile, ofile);
      }
      some(ofile) {
        let ofile = ofile;
        // FIXME: what about windows? This will create a foo.exe.o.
        saved_out_filename = ofile;
        let temp_filename =
            if !stop_after_codegen { ofile + ~".o" } else { ofile };
        compile_input(sess, cfg, ifile, temp_filename);
      }
    }

    // If the user wants an exe generated we need to invoke
    // gcc to link the object file with some libs
    //
    // TODO: Factor this out of main.
    if stop_after_codegen { ret; }

    let glu: istr = binary_dir + ~"/lib/glue.o";
    let main: istr = binary_dir + ~"/lib/main.o";
    let stage: istr = ~"-L" + binary_dir + ~"/lib";
    let prog: istr = ~"gcc";
    // The invocations of gcc share some flags across platforms

    let gcc_args =
        [stage,
         ~"-Lrt", ~"-lrustrt", glu,
         ~"-m32", ~"-o", saved_out_filename,
         saved_out_filename + ~".o"];
    let lib_cmd;

    let os = sess.get_targ_cfg().os;
    if os == session::os_macos {
        lib_cmd = ~"-dynamiclib";
    } else { lib_cmd = ~"-shared"; }

    // Converts a library file name into a gcc -l argument
    fn unlib(config: @session::config, filename: &istr) -> istr {
        let rmlib =
            bind fn (config: @session::config, filename: &istr) -> istr {
            if config.os == session::os_macos ||
                config.os == session::os_linux &&
                str::find(filename, ~"lib") == 0 {
                ret str::slice(filename, 3u,
                                str::byte_len(filename));
            } else { ret filename; }
        }(config, _);
        fn rmext(filename: &istr) -> istr {
            let parts = str::split(filename, '.' as u8);
            vec::pop(parts);
            ret str::connect(parts, ~".");
        }
        ret alt config.os {
              session::os_macos. { rmext(rmlib(filename)) }
              session::os_linux. { rmext(rmlib(filename)) }
              _ { rmext(filename) }
            };
    }

    let cstore = sess.get_cstore();
    for cratepath: istr in cstore::get_used_crate_files(cstore) {
        if str::ends_with(cratepath, ~".rlib") {
            gcc_args += [cratepath];
            cont;
        }
        let cratepath = cratepath;
        let dir = fs::dirname(cratepath);
        if dir != ~"" { gcc_args += [~"-L" + dir]; }
        let libarg = unlib(sess.get_targ_cfg(), fs::basename(cratepath));
        gcc_args += [~"-l" + libarg];
    }

    let ula = cstore::get_used_link_args(cstore);
    for arg: istr in ula { gcc_args += [arg]; }

    let used_libs = cstore::get_used_libraries(cstore);
    for l: istr in used_libs { gcc_args += [~"-l" + l]; }

    if sopts.library {
        gcc_args += [lib_cmd];
    } else {
        // FIXME: why do we hardcode -lm?
        gcc_args += [~"-lm", main];
    }
    // We run 'gcc' here

    let err_code = run::run_program(prog, gcc_args);
    if 0 != err_code {
        sess.err(
            #ifmt["linking with gcc failed with code %d", err_code]);
        sess.note(
            #ifmt["gcc arguments: %s",
                       str::connect(gcc_args, ~" ")]);
        sess.abort_if_errors();
    }
    // Clean up on Darwin

    if sess.get_targ_cfg().os == session::os_macos {
        run::run_program(~"dsymutil",
                         [saved_out_filename]);
    }


    // Remove the temporary object file if we aren't saving temps
    if !sopts.save_temps {
        run::run_program(~"rm",
                         [saved_out_filename + ~".o"]);
    }
}

#[cfg(test)]
mod test {

    // When the user supplies --test we should implicitly supply --cfg test
    #[test]
    fn test_switch_implies_cfg_test() {
        let match =
            alt getopts::getopts([~"--test"], opts()) {
              getopts::success(m) { m }
            };
        let sessopts = build_session_options(~"whatever", match, ~"whatever");
        let sess = build_session(sessopts);
        let cfg = build_configuration(sess, ~"whatever", ~"whatever");
        assert (attr::contains_name(cfg, ~"test"));
    }

    // When the user supplies --test and --cfg test, don't implicitly add
    // another --cfg test
    #[test]
    fn test_switch_implies_cfg_test_unless_cfg_test() {
        let match =
            alt getopts::getopts([~"--test", ~"--cfg=test"], opts()) {
              getopts::success(m) { m }
            };
        let sessopts = build_session_options(~"whatever", match, ~"whatever");
        let sess = build_session(sessopts);
        let cfg = build_configuration(sess, ~"whatever", ~"whatever");
        let test_items = attr::find_meta_items_by_name(cfg, ~"test");
        assert (vec::len(test_items) == 1u);
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
