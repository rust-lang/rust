

// -*- rust -*-
import metadata::{creader, cstore};
import syntax::parse::{parser};
import syntax::{ast, codemap};
import front::attr;
import middle::{trans, resolve, freevars, kind, ty, typeck, fn_usage};
import syntax::print::{pp, pprust};
import util::{ppaux, filesearch};
import back::link;
import std::{option, str, vec, int, io, getopts, result};
import std::option::{some, none};
import std::getopts::{optopt, optmulti, optflag, optflagopt, opt_present};
import back::{x86, x86_64};

tag pp_mode { ppm_normal; ppm_expanded; ppm_typed; ppm_identified; }

fn default_configuration(sess: session::session, argv0: str, input: str) ->
   ast::crate_cfg {
    let libc =
        alt sess.get_targ_cfg().os {
          session::os_win32. { "msvcrt.dll" }
          session::os_macos. { "libc.dylib" }
          session::os_linux. { "libc.so.6" }
          _ { "libc.so" }
        };

    let mk = attr::mk_name_value_item_str;

    let arch = alt sess.get_targ_cfg().arch {
      session::arch_x86. { "x86" }
      session::arch_x86_64. { "x86_64" }
      session::arch_arm. { "arm" }
    };

    ret [ // Target bindings.
         mk("target_os", std::os::target_os()),
         mk("target_arch", arch),
         mk("target_libc", libc),
         // Build bindings.
         mk("build_compiler", argv0),
         mk("build_input", input)];
}

fn build_configuration(sess: session::session, argv0: str, input: str) ->
   ast::crate_cfg {
    // Combine the configuration requested by the session (command line) with
    // some default and generated configuration items
    let default_cfg = default_configuration(sess, argv0, input);
    let user_cfg = sess.get_opts().cfg;
    // If the user wants a test runner, then add the test cfg
    let gen_cfg =
        {
            if sess.get_opts().test && !attr::contains_name(user_cfg, "test")
               {
                [attr::mk_word_item("test")]
            } else { [] }
        };
    ret user_cfg + gen_cfg + default_cfg;
}

// Convert strings provided as --cfg [cfgspec] into a crate_cfg
fn parse_cfgspecs(cfgspecs: [str]) -> ast::crate_cfg {
    // FIXME: It would be nice to use the parser to parse all varieties of
    // meta_item here. At the moment we just support the meta_word variant.
    let words = [];
    for s: str in cfgspecs { words += [attr::mk_word_item(s)]; }
    ret words;
}

fn input_is_stdin(filename: str) -> bool { filename == "-" }

fn parse_input(sess: session::session, cfg: ast::crate_cfg, input: str) ->
   @ast::crate {
    if !input_is_stdin(input) {
        parser::parse_crate_from_file(input, cfg, sess.get_parse_sess())
    } else { parse_input_src(sess, cfg, input).crate }
}

fn parse_input_src(sess: session::session, cfg: ast::crate_cfg, infile: str)
   -> {crate: @ast::crate, src: str} {
    let srcbytes = if infile != "-" {
        alt io::file_reader(infile) {
          result::ok(reader) { reader }
          result::err(e) {
            sess.fatal(e)
          }
        }
    } else {
        io::stdin()
    }.read_whole_stream();
    let src = str::unsafe_from_bytes(srcbytes);
    let crate =
        parser::parse_crate_from_source_str(infile, src, cfg,
                                            sess.get_parse_sess());
    ret {crate: crate, src: src};
}

fn time<T>(do_it: bool, what: str, thunk: fn@() -> T) -> T {
    if !do_it { ret thunk(); }
    let start = std::time::precise_time_s();
    let rv = thunk();
    let end = std::time::precise_time_s();
    log_err #fmt["time: %s took %s s", what,
                 std::float::to_str(end - start, 3u)];
    ret rv;
}

fn compile_input(sess: session::session, cfg: ast::crate_cfg, input: str,
                 output: str) {
    let time_passes = sess.get_opts().time_passes;
    let crate =
        time(time_passes, "parsing", bind parse_input(sess, cfg, input));
    if sess.get_opts().parse_only { ret; }
    crate =
        time(time_passes, "configuration",
             bind front::config::strip_unconfigured_items(crate));
    if sess.get_opts().test {
        crate =
            time(time_passes, "building test harness",
                 bind front::test::modify_for_testing(sess, crate));
    }
    crate =
        time(time_passes, "expansion",
             bind syntax::ext::expand::expand_crate(sess, crate));

    let ast_map =
        time(time_passes, "ast indexing",
             bind middle::ast_map::map_crate(*crate));
    time(time_passes, "external crate/lib resolution",
         bind creader::read_crates(sess, *crate));
    let {def_map: def_map, ext_map: ext_map} =
        time(time_passes, "resolution",
             bind resolve::resolve_crate(sess, ast_map, crate));
    let freevars =
        time(time_passes, "freevar finding",
             bind freevars::annotate_freevars(def_map, crate));
    let ty_cx = ty::mk_ctxt(sess, def_map, ext_map, ast_map, freevars);
    time(time_passes, "typechecking", bind typeck::check_crate(ty_cx, crate));
    time(time_passes, "function usage",
         bind fn_usage::check_crate_fn_usage(ty_cx, crate));
    time(time_passes, "alt checking",
         bind middle::check_alt::check_crate(ty_cx, crate));
    if sess.get_opts().run_typestate {
        time(time_passes, "typestate checking",
             bind middle::tstate::ck::check_crate(ty_cx, crate));
    }
    let mut_map =
        time(time_passes, "mutability checking",
             bind middle::mut::check_crate(ty_cx, crate));
    let copy_map =
        time(time_passes, "alias checking",
             bind middle::alias::check_crate(ty_cx, crate));
    time(time_passes, "kind checking", bind kind::check_crate(ty_cx, crate));
    time(time_passes, "const checking",
         bind middle::check_const::check_crate(ty_cx, crate));
    if sess.get_opts().no_trans { ret; }
    let llmod =
        time(time_passes, "translation",
             bind trans::trans_crate(sess, crate, ty_cx, output, ast_map,
                                     mut_map, copy_map));
    time(time_passes, "LLVM passes",
         bind link::write::run_passes(sess, llmod, output));
}

fn pretty_print_input(sess: session::session, cfg: ast::crate_cfg, input: str,
                      ppm: pp_mode) {
    fn ann_paren_for_expr(node: pprust::ann_node) {
        alt node { pprust::node_expr(s, expr) { pprust::popen(s); } _ { } }
    }
    fn ann_typed_post(tcx: ty::ctxt, node: pprust::ann_node) {
        alt node {
          pprust::node_expr(s, expr) {
            pp::space(s.s);
            pp::word(s.s, "as");
            pp::space(s.s);
            pp::word(s.s, ppaux::ty_to_str(tcx, ty::expr_ty(tcx, expr)));
            pprust::pclose(s);
          }
          _ { }
        }
    }
    fn ann_identified_post(node: pprust::ann_node) {
        alt node {
          pprust::node_item(s, item) {
            pp::space(s.s);
            pprust::synth_comment(s, int::to_str(item.id, 10u));
          }
          pprust::node_block(s, blk) {
            pp::space(s.s);
            pprust::synth_comment(s,
                                  "block " + int::to_str(blk.node.id, 10u));
          }
          pprust::node_expr(s, expr) {
            pp::space(s.s);
            pprust::synth_comment(s, int::to_str(expr.id, 10u));
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
    pprust::print_crate(sess.get_codemap(), crate, input,
                        io::string_reader(src), io::stdout(), ann);
}

fn version(argv0: str) {
    let vers = "unknown version";
    let env_vers = #env["CFG_VERSION"];
    if str::byte_len(env_vers) != 0u { vers = env_vers; }
    io::stdout().write_str(#fmt["%s %s\n", argv0, vers]);
    io::stdout().write_str(#fmt["host: %s\n", host_triple()]);
}

fn usage(argv0: str) {
    io::stdout().write_str(#fmt["usage: %s [options] <input>\n", argv0] +
                               "
options:

    -h --help          display this message
    -v --version       print version info and exit

    -o <filename>      write output to <filename>
    --lib              compile a library crate
    --static           use or produce static libraries
    --pretty [type]    pretty-print the input instead of compiling
    --ls               list the symbols defined by a crate file
    -L <path>          add a directory to the library search path
    --noverify         suppress LLVM verification step (slight speedup)
    --parse-only       parse only; do not compile, assemble, or link
    --no-trans         run all passes except translation; no output
    -g                 produce debug info
    --opt-level <lvl>  optimize with possible levels 0-3
    -O                 equivalent to --opt-level=2
    -S                 compile only; do not assemble or link
    --no-asm-comments  do not add comments into the assembly source
    -c                 compile and assemble, but do not link
    --emit-llvm        produce an LLVM bitcode file
    --save-temps       write intermediate files in addition to normal output
    --stats            gather and report various compilation statistics
    --cfg <cfgspec>    configure the compilation environment
    --time-passes      time the individual phases of the compiler
    --time-llvm-passes time the individual phases of the LLVM backend
    --sysroot <path>   override the system root
    --target <triple>  target to compile for (default: host triple)
    --no-typestate     don't run the typestate pass (unsafe!)
    --test             build test harness
    --gc               garbage collect shared data (experimental/temporary)
    --stack-growth     perform stack checks (experimental)

");
}

fn get_os(triple: str) -> session::os {
    ret if str::find(triple, "win32") >= 0 ||
               str::find(triple, "mingw32") >= 0 {
            session::os_win32
        } else if str::find(triple, "darwin") >= 0 {
            session::os_macos
        } else if str::find(triple, "linux") >= 0 {
            session::os_linux
        } else { early_error("Unknown operating system!") };
}

fn get_arch(triple: str) -> session::arch {
    ret if str::find(triple, "i386") >= 0 || str::find(triple, "i486") >= 0 ||
               str::find(triple, "i586") >= 0 ||
               str::find(triple, "i686") >= 0 ||
               str::find(triple, "i786") >= 0 {
            session::arch_x86
        } else if str::find(triple, "x86_64") >= 0 {
            session::arch_x86_64
        } else if str::find(triple, "arm") >= 0 ||
                      str::find(triple, "xscale") >= 0 {
            session::arch_arm
        } else { early_error("Unknown architecture! " + triple) };
}

fn build_target_config(sopts: @session::options) -> @session::config {
    let os = get_os(sopts.target_triple);
    let arch = get_arch(sopts.target_triple);
    let (int_type, uint_type, float_type) = alt arch {
      session::arch_x86. {(ast::ty_i32, ast::ty_u32, ast::ty_f64)}
      session::arch_x86_64. {(ast::ty_i64, ast::ty_u64, ast::ty_f64)}
      session::arch_arm. {(ast::ty_i32, ast::ty_u32, ast::ty_f64)}
    };
    let target_strs = alt arch {
      session::arch_x86. {x86::get_target_strs(os)}
      session::arch_x86_64. {x86_64::get_target_strs(os)}
      session::arch_arm. {x86::get_target_strs(os)}
    };
    let target_cfg: @session::config =
        @{os: os, arch: arch, target_strs: target_strs, int_type: int_type,
          uint_type: uint_type, float_type: float_type};
    ret target_cfg;
}

fn host_triple() -> str {
    // Get the host triple out of the build environment. This ensures that our
    // idea of the host triple is the same as for the set of libraries we've
    // actually built.  We can't just take LLVM's host triple because they
    // normalize all ix86 architectures to i386.
    // FIXME: Instead of grabbing the host triple we really should be
    // grabbing (at compile time) the target triple that this rustc is
    // built with and calling that (at runtime) the host triple.
    let ht = #env("CFG_HOST_TRIPLE");
    ret ht != "" ? ht : fail "rustc built without CFG_HOST_TRIPLE";
}

fn build_session_options(match: getopts::match)
   -> @session::options {
    let library = opt_present(match, "lib");
    let static = opt_present(match, "static");

    let parse_only = opt_present(match, "parse-only");
    let no_trans = opt_present(match, "no-trans");

    let output_type =
        if parse_only || no_trans {
            link::output_type_none
        } else if opt_present(match, "S") && opt_present(match, "emit-llvm") {
            link::output_type_llvm_assembly
        } else if opt_present(match, "S") {
            link::output_type_assembly
        } else if opt_present(match, "c") {
            link::output_type_object
        } else if opt_present(match, "emit-llvm") {
            link::output_type_bitcode
        } else { link::output_type_exe };
    let verify = !opt_present(match, "noverify");
    let save_temps = opt_present(match, "save-temps");
    let debuginfo = opt_present(match, "g");
    let stats = opt_present(match, "stats");
    let time_passes = opt_present(match, "time-passes");
    let time_llvm_passes = opt_present(match, "time-llvm-passes");
    let run_typestate = !opt_present(match, "no-typestate");
    let sysroot_opt = getopts::opt_maybe_str(match, "sysroot");
    let target_opt = getopts::opt_maybe_str(match, "target");
    let no_asm_comments = getopts::opt_present(match, "no-asm-comments");
    alt output_type {
      // unless we're emitting huamn-readable assembly, omit comments.
      link::output_type_llvm_assembly. | link::output_type_assembly. {}
      _ { no_asm_comments = true; }
    }
    let opt_level: uint =
        if opt_present(match, "O") {
            if opt_present(match, "opt-level") {
                early_error("-O and --opt-level both provided");
            }
            2u
        } else if opt_present(match, "opt-level") {
            alt getopts::opt_str(match, "opt-level") {
              "0" { 0u }
              "1" { 1u }
              "2" { 2u }
              "3" { 3u }
              _ {
                early_error("optimization level needs " +
                            "to be between 0-3")
              }
            }
        } else { 0u };
    let target =
        alt target_opt {
            none. { host_triple() }
            some(s) { s }
        };

    let addl_lib_search_paths = getopts::opt_strs(match, "L");
    let cfg = parse_cfgspecs(getopts::opt_strs(match, "cfg"));
    let test = opt_present(match, "test");
    let do_gc = opt_present(match, "gc");
    let stack_growth = opt_present(match, "stack-growth");
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
          addl_lib_search_paths: addl_lib_search_paths,
          maybe_sysroot: sysroot_opt,
          target_triple: target,
          cfg: cfg,
          test: test,
          parse_only: parse_only,
          no_trans: no_trans,
          do_gc: do_gc,
          stack_growth: stack_growth,
          no_asm_comments: no_asm_comments};
    ret sopts;
}

fn build_session(sopts: @session::options) -> session::session {
    let target_cfg = build_target_config(sopts);
    let cstore = cstore::mk_cstore();
    let filesearch = filesearch::mk_filesearch(
        sopts.maybe_sysroot,
        sopts.target_triple,
        sopts.addl_lib_search_paths);
    ret session::session(target_cfg, sopts, cstore,
                         @{cm: codemap::new_codemap(), mutable next_id: 0},
                         none, 0u, filesearch);
}

fn parse_pretty(sess: session::session, &&name: str) -> pp_mode {
    if str::eq(name, "normal") {
        ret ppm_normal;
    } else if str::eq(name, "expanded") {
        ret ppm_expanded;
    } else if str::eq(name, "typed") {
        ret ppm_typed;
    } else if str::eq(name, "identified") { ret ppm_identified; }
    sess.fatal("argument to `pretty` must be one of `normal`, `typed`, or " +
                   "`identified`");
}

fn opts() -> [getopts::opt] {
    ret [optflag("h"), optflag("help"), optflag("v"), optflag("version"),
         optflag("emit-llvm"), optflagopt("pretty"),
         optflag("ls"), optflag("parse-only"), optflag("no-trans"),
         optflag("O"), optopt("opt-level"), optmulti("L"), optflag("S"),
         optflag("c"), optopt("o"), optflag("g"), optflag("save-temps"),
         optopt("sysroot"), optopt("target"), optflag("stats"),
         optflag("time-passes"), optflag("time-llvm-passes"),
         optflag("no-typestate"), optflag("noverify"),
         optmulti("cfg"), optflag("test"),
         optflag("lib"), optflag("static"), optflag("gc"),
         optflag("stack-growth"), optflag("check-unsafe"),
         optflag("no-asm-comments")];
}

fn build_output_filenames(ifile: str, ofile: option::t<str>,
                          sopts: @session::options)
        -> @{out_filename: str, obj_filename:str} {
    let obj_filename = "";
    let saved_out_filename: str = "";
    let stop_after_codegen =
        sopts.output_type != link::output_type_exe ||
            sopts.static && sopts.library;
    alt ofile {
      none. {
        // "-" as input file will cause the parser to read from stdin so we
        // have to make up a name
        // We want to toss everything after the final '.'
        let parts =
            if !input_is_stdin(ifile) {
                str::split(ifile, '.' as u8)
            } else { ["default", "rs"] };
        vec::pop(parts);
        let base_filename = str::connect(parts, ".");
        let suffix =
            alt sopts.output_type {
              link::output_type_none. { "none" }
              link::output_type_bitcode. { "bc" }
              link::output_type_assembly. { "s" }
              link::output_type_llvm_assembly. { "ll" }
              // Object and exe output both use the '.o' extension here
              link::output_type_object. | link::output_type_exe. {
                "o"
              }
            };
        obj_filename = base_filename + "." + suffix;

        if sopts.library {
            saved_out_filename = std::os::dylib_filename(base_filename);
        } else {
            saved_out_filename = base_filename;
        }
      }
      some(out_file) {
        // FIXME: what about windows? This will create a foo.exe.o.
        saved_out_filename = out_file;
        obj_filename =
            if stop_after_codegen { out_file } else { out_file + ".o" };
      }
    }
    ret @{out_filename: saved_out_filename, obj_filename: obj_filename};
}

fn early_error(msg: str) -> ! {
    codemap::print_diagnostic("", codemap::error, msg);
    fail;
}

fn main(args: [str]) {
    let args = args, binary = vec::shift(args);
    let match =
        alt getopts::getopts(args, opts()) {
          getopts::success(m) { m }
          getopts::failure(f) {
            early_error(getopts::fail_str(f))
          }
        };
    if opt_present(match, "h") || opt_present(match, "help") {
        usage(binary);
        ret;
    }
    if opt_present(match, "v") || opt_present(match, "version") {
        version(binary);
        ret;
    }
    let ifile = alt vec::len(match.free) {
      0u { early_error("No input filename given.") }
      1u { match.free[0] }
      _ { early_error("Multiple input filenames provided.") }
    };

    let sopts = build_session_options(match);
    let sess = build_session(sopts);
    let ofile = getopts::opt_maybe_str(match, "o");
    let outputs = build_output_filenames(ifile, ofile, sopts);
    let cfg = build_configuration(sess, binary, ifile);
    let pretty =
        option::map::<str,
                      pp_mode>(bind parse_pretty(sess, _),
                               getopts::opt_default(match, "pretty",
                                                    "normal"));
    alt pretty {
      some::<pp_mode>(ppm) { pretty_print_input(sess, cfg, ifile, ppm); ret; }
      none::<pp_mode>. {/* continue */ }
    }
    let ls = opt_present(match, "ls");
    if ls {
        metadata::creader::list_file_metadata(sess, ifile, io::stdout());
        ret;
    }

    let stop_after_codegen =
        sopts.output_type != link::output_type_exe ||
            sopts.static && sopts.library;

    let temp_filename = outputs.obj_filename;

    compile_input(sess, cfg, ifile, temp_filename);

    if stop_after_codegen { ret; }

    link::link_binary(sess, temp_filename, outputs.out_filename);
}

#[cfg(test)]
mod test {

    // When the user supplies --test we should implicitly supply --cfg test
    #[test]
    fn test_switch_implies_cfg_test() {
        let match =
            alt getopts::getopts(["--test"], opts()) {
              getopts::success(m) { m }
            };
        let sessopts = build_session_options(match);
        let sess = build_session(sessopts);
        let cfg = build_configuration(sess, "whatever", "whatever");
        assert (attr::contains_name(cfg, "test"));
    }

    // When the user supplies --test and --cfg test, don't implicitly add
    // another --cfg test
    #[test]
    fn test_switch_implies_cfg_test_unless_cfg_test() {
        let match =
            alt getopts::getopts(["--test", "--cfg=test"], opts()) {
              getopts::success(m) { m }
            };
        let sessopts = build_session_options(match);
        let sess = build_session(sessopts);
        let cfg = build_configuration(sess, "whatever", "whatever");
        let test_items = attr::find_meta_items_by_name(cfg, "test");
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
