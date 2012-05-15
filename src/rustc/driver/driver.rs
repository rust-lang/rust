// -*- rust -*-
import metadata::{creader, cstore};
import session::session;
import syntax::parse;
import syntax::{ast, codemap};
import syntax::attr;
import middle::{trans, resolve, freevars, kind, ty, typeck,
                last_use, lint};
import syntax::print::{pp, pprust};
import util::{ppaux, filesearch};
import back::link;
import result::{ok, err};
import std::getopts;
import io::{reader_util, writer_util};
import getopts::{optopt, optmulti, optflag, optflagopt, opt_present};
import back::{x86, x86_64};

enum pp_mode {ppm_normal, ppm_expanded, ppm_typed, ppm_identified,
              ppm_expanded_identified }

#[doc = "
The name used for source code that doesn't originate in a file
(e.g. source from stdin or a string)
"]
fn anon_src() -> str { "<anon>" }

fn source_name(input: input) -> str {
    alt input {
      file_input(ifile) { ifile }
      str_input(_) { anon_src() }
    }
}

fn default_configuration(sess: session, argv0: str, input: input) ->
   ast::crate_cfg {
    let libc = alt sess.targ_cfg.os {
      session::os_win32 { "msvcrt.dll" }
      session::os_macos { "libc.dylib" }
      session::os_linux { "libc.so.6" }
      session::os_freebsd { "libc.so.7" }
      // _ { "libc.so" }
    };

    let mk = attr::mk_name_value_item_str;

    let arch = alt sess.targ_cfg.arch {
      session::arch_x86 { "x86" }
      session::arch_x86_64 { "x86_64" }
      session::arch_arm { "arm" }
    };

    ret [ // Target bindings.
         mk("target_os", os::sysname()),
         mk("target_arch", arch),
         mk("target_libc", libc),
         // Build bindings.
         mk("build_compiler", argv0),
         mk("build_input", source_name(input))];
}

fn build_configuration(sess: session, argv0: str, input: input) ->
   ast::crate_cfg {
    // Combine the configuration requested by the session (command line) with
    // some default and generated configuration items
    let default_cfg = default_configuration(sess, argv0, input);
    let user_cfg = sess.opts.cfg;
    // If the user wants a test runner, then add the test cfg
    let gen_cfg =
        {
            if sess.opts.test && !attr::contains_name(user_cfg, "test")
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
    let mut words = [];
    for cfgspecs.each {|s| words += [attr::mk_word_item(s)]; }
    ret words;
}

enum input {
    #[doc = "Load source from file"]
    file_input(str),
    #[doc = "The string is the source"]
    str_input(str)
}

fn parse_input(sess: session, cfg: ast::crate_cfg, input: input)
    -> @ast::crate {
    alt input {
      file_input(file) {
        parse::parse_crate_from_file(file, cfg, sess.parse_sess)
      }
      str_input(src) {
        // FIXME: Don't really want to box the source string
        parse::parse_crate_from_source_str(
            anon_src(), @src, cfg, sess.parse_sess)
      }
    }
}

fn time<T>(do_it: bool, what: str, thunk: fn@() -> T) -> T {
    if !do_it { ret thunk(); }
    let start = std::time::precise_time_s();
    let rv = thunk();
    let end = std::time::precise_time_s();
    io::stdout().write_str(#fmt("time: %3.3f s\t%s\n",
                                end - start, what));
    ret rv;
}

enum compile_upto {
    cu_parse,
    cu_expand,
    cu_typeck,
    cu_no_trans,
    cu_everything,
}

fn compile_upto(sess: session, cfg: ast::crate_cfg,
                input: input, upto: compile_upto,
                outputs: option<output_filenames>)
    -> {crate: @ast::crate, tcx: option<ty::ctxt>} {
    let time_passes = sess.opts.time_passes;
    let mut crate = time(time_passes, "parsing",
                         bind parse_input(sess, cfg, input));
    if upto == cu_parse { ret {crate: crate, tcx: none}; }

    sess.building_library = session::building_library(
        sess.opts.crate_type, crate, sess.opts.test);

    crate =
        time(time_passes, "configuration",
             bind front::config::strip_unconfigured_items(crate));
    crate =
        time(time_passes, "maybe building test harness",
             bind front::test::modify_for_testing(sess, crate));
    crate =
        time(time_passes, "expansion",
             bind syntax::ext::expand::expand_crate(
                 sess.parse_sess, sess.opts.cfg, crate));

    if upto == cu_expand { ret {crate: crate, tcx: none}; }

    crate =
        time(time_passes, "intrinsic injection",
             bind front::intrinsic_inject::inject_intrinsic(sess, crate));

    crate =
        time(time_passes, "core injection",
             bind front::core_inject::maybe_inject_libcore_ref(sess, crate));

    let ast_map =
        time(time_passes, "ast indexing",
             bind middle::ast_map::map_crate(sess, *crate));
    time(time_passes, "external crate/lib resolution",
         bind creader::read_crates(sess, *crate));
    let {def_map, exp_map, impl_map} =
        time(time_passes, "resolution",
             bind resolve::resolve_crate(sess, ast_map, crate));
    let freevars =
        time(time_passes, "freevar finding",
             bind freevars::annotate_freevars(def_map, crate));
    let region_map =
        time(time_passes, "region resolution",
             bind middle::region::resolve_crate(sess, def_map, crate));
    let ty_cx = ty::mk_ctxt(sess, def_map, ast_map, freevars, region_map);
    let (method_map, vtable_map) =
        time(time_passes, "typechecking",
             bind typeck::check_crate(ty_cx, impl_map, crate));
    time(time_passes, "const checking",
         bind middle::check_const::check_crate(sess, crate, ast_map, def_map,
                                               method_map, ty_cx));

    if upto == cu_typeck { ret {crate: crate, tcx: some(ty_cx)}; }

    time(time_passes, "block-use checking",
         bind middle::block_use::check_crate(ty_cx, crate));
    time(time_passes, "loop checking",
         bind middle::check_loop::check_crate(ty_cx, crate));
    time(time_passes, "alt checking",
         bind middle::check_alt::check_crate(ty_cx, crate));
    time(time_passes, "self checking",
         bind middle::check_self::check_crate(ty_cx, crate));
    time(time_passes, "typestate checking",
         bind middle::tstate::ck::check_crate(ty_cx, crate));
    let (root_map, mutbl_map) = time(
        time_passes, "borrow checking",
        bind middle::borrowck::check_crate(ty_cx, method_map, crate));
    time(time_passes, "region checking",
         bind middle::regionck::check_crate(ty_cx, crate));
    let (copy_map, ref_map) =
        time(time_passes, "alias checking",
             bind middle::alias::check_crate(ty_cx, crate));
    let (last_uses, spill_map) = time(time_passes, "last use finding",
        bind last_use::find_last_uses(crate, def_map, ref_map, ty_cx));
    time(time_passes, "kind checking",
         bind kind::check_crate(ty_cx, method_map, last_uses, crate));

    lint::check_crate(ty_cx, crate, sess.opts.lint_opts, time_passes);

    if upto == cu_no_trans { ret {crate: crate, tcx: some(ty_cx)}; }
    let outputs = option::get(outputs);

    let maps = {mutbl_map: mutbl_map, root_map: root_map,
                copy_map: copy_map, last_uses: last_uses,
                impl_map: impl_map, method_map: method_map,
                vtable_map: vtable_map, spill_map: spill_map};

    let (llmod, link_meta) =
        time(time_passes, "translation",
             bind trans::base::trans_crate(
                 sess, crate, ty_cx, outputs.obj_filename,
                 exp_map, maps));
    time(time_passes, "LLVM passes",
         bind link::write::run_passes(sess, llmod, outputs.obj_filename));

    let stop_after_codegen =
        sess.opts.output_type != link::output_type_exe ||
            sess.opts.static && sess.building_library;

    if stop_after_codegen { ret {crate: crate, tcx: some(ty_cx)}; }

    time(time_passes, "linking",
         bind link::link_binary(sess, outputs.obj_filename,
                                outputs.out_filename, link_meta));
    ret {crate: crate, tcx: some(ty_cx)};
}

fn compile_input(sess: session, cfg: ast::crate_cfg, input: input,
                 outdir: option<str>, output: option<str>) {

    let upto = if sess.opts.parse_only { cu_parse }
               else if sess.opts.no_trans { cu_no_trans }
               else { cu_everything };
    let outputs = build_output_filenames(input, outdir, output, sess);
    compile_upto(sess, cfg, input, upto, some(outputs));
}

fn pretty_print_input(sess: session, cfg: ast::crate_cfg, input: input,
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
    let upto = alt ppm {
      ppm_expanded | ppm_expanded_identified { cu_expand }
      ppm_typed { cu_typeck }
      _ { cu_parse }
    };
    let {crate, tcx} = compile_upto(sess, cfg, input, upto, none);

    let mut ann: pprust::pp_ann = pprust::no_ann();
    alt ppm {
      ppm_typed {
        ann = {pre: ann_paren_for_expr,
               post: bind ann_typed_post(option::get(tcx), _)};
      }
      ppm_identified | ppm_expanded_identified {
        ann = {pre: ann_paren_for_expr, post: ann_identified_post};
      }
      ppm_expanded | ppm_normal {}
    }
    let src = codemap::get_filemap(sess.codemap, source_name(input)).src;
    io::with_str_reader(*src) { |rdr|
        pprust::print_crate(sess.codemap, sess.span_diagnostic, crate,
                            source_name(input),
                            rdr, io::stdout(), ann);
    }
}

fn get_os(triple: str) -> option<session::os> {
    ret if str::contains(triple, "win32") ||
               str::contains(triple, "mingw32") {
            some(session::os_win32)
        } else if str::contains(triple, "darwin") {
            some(session::os_macos)
        } else if str::contains(triple, "linux") {
            some(session::os_linux)
        } else if str::contains(triple, "freebsd") {
            some(session::os_freebsd)
        } else { none };
}

fn get_arch(triple: str) -> option<session::arch> {
    ret if str::contains(triple, "i386") || str::contains(triple, "i486") ||
               str::contains(triple, "i586") ||
               str::contains(triple, "i686") ||
               str::contains(triple, "i786") {
            some(session::arch_x86)
        } else if str::contains(triple, "x86_64") {
            some(session::arch_x86_64)
        } else if str::contains(triple, "arm") ||
                      str::contains(triple, "xscale") {
            some(session::arch_arm)
        } else { none };
}

fn build_target_config(sopts: @session::options,
                       demitter: diagnostic::emitter) -> @session::config {
    let os = alt get_os(sopts.target_triple) {
      some(os) { os }
      none { early_error(demitter, "unknown operating system") }
    };
    let arch = alt get_arch(sopts.target_triple) {
      some(arch) { arch }
      none { early_error(demitter,
                          "unknown architecture: " + sopts.target_triple) }
    };
    let (int_type, uint_type, float_type) = alt arch {
      session::arch_x86 {(ast::ty_i32, ast::ty_u32, ast::ty_f64)}
      session::arch_x86_64 {(ast::ty_i64, ast::ty_u64, ast::ty_f64)}
      session::arch_arm {(ast::ty_i32, ast::ty_u32, ast::ty_f64)}
    };
    let target_strs = alt arch {
      session::arch_x86 {x86::get_target_strs(os)}
      session::arch_x86_64 {x86_64::get_target_strs(os)}
      session::arch_arm {x86::get_target_strs(os)}
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
    ret if ht != "" {
            ht
        } else {
            fail "rustc built without CFG_HOST_TRIPLE"
        };
}

fn build_session_options(match: getopts::match,
                         demitter: diagnostic::emitter) -> @session::options {
    let crate_type = if opt_present(match, "lib") {
        session::lib_crate
    } else if opt_present(match, "bin") {
        session::bin_crate
    } else {
        session::unknown_crate
    };
    let static = opt_present(match, "static");

    let parse_only = opt_present(match, "parse-only");
    let no_trans = opt_present(match, "no-trans");

    let lint_flags = (getopts::opt_strs(match, "W")
                      + getopts::opt_strs(match, "warn"));
    let lint_dict = lint::get_lint_dict();
    let lint_opts = vec::map(lint_flags) {|flag|
        alt lint::lookup_lint(lint_dict, flag) {
          none { early_error(demitter, #fmt("unknown warning: %s", flag)) }
          some(x) { x }
        }
    };

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
    let verify = !opt_present(match, "no-verify");
    let save_temps = opt_present(match, "save-temps");
    let extra_debuginfo = opt_present(match, "xg");
    let debuginfo = opt_present(match, "g") || extra_debuginfo;
    let stats = opt_present(match, "stats");
    let time_passes = opt_present(match, "time-passes");
    let time_llvm_passes = opt_present(match, "time-llvm-passes");
    let count_llvm_insns = opt_present(match, "count-llvm-insns");
    let sysroot_opt = getopts::opt_maybe_str(match, "sysroot");
    let target_opt = getopts::opt_maybe_str(match, "target");
    let mut no_asm_comments = getopts::opt_present(match, "no-asm-comments");
    let debug_rustc = getopts::opt_present(match, "debug-rustc");
    let borrowck = alt getopts::opt_maybe_str(match, "borrowck") {
      none { 0u }
      some("warn") { 1u }
      some("err") { 2u }
      some(_) {
        early_error(demitter, "borrowck may be warn or err")
      }
    };
    alt output_type {
      // unless we're emitting huamn-readable assembly, omit comments.
      link::output_type_llvm_assembly | link::output_type_assembly {}
      _ { no_asm_comments = true; }
    }
    let opt_level: uint =
        if opt_present(match, "O") {
            if opt_present(match, "opt-level") {
                early_error(demitter, "-O and --opt-level both provided");
            }
            2u
        } else if opt_present(match, "opt-level") {
            alt getopts::opt_str(match, "opt-level") {
              "0" { 0u }
              "1" { 1u }
              "2" { 2u }
              "3" { 3u }
              _ {
                early_error(demitter, "optimization level needs " +
                            "to be between 0-3")
              }
            }
        } else { 0u };
    let target =
        alt target_opt {
            none { host_triple() }
            some(s) { s }
        };

    let addl_lib_search_paths = getopts::opt_strs(match, "L");
    let cfg = parse_cfgspecs(getopts::opt_strs(match, "cfg"));
    let test = opt_present(match, "test");
    let sopts: @session::options =
        @{crate_type: crate_type,
          static: static,
          optimize: opt_level,
          debuginfo: debuginfo,
          extra_debuginfo: extra_debuginfo,
          verify: verify,
          lint_opts: lint_opts,
          save_temps: save_temps,
          stats: stats,
          time_passes: time_passes,
          count_llvm_insns: count_llvm_insns,
          time_llvm_passes: time_llvm_passes,
          output_type: output_type,
          addl_lib_search_paths: addl_lib_search_paths,
          maybe_sysroot: sysroot_opt,
          target_triple: target,
          cfg: cfg,
          test: test,
          parse_only: parse_only,
          no_trans: no_trans,
          no_asm_comments: no_asm_comments,
          debug_rustc: debug_rustc,
          borrowck: borrowck};
    ret sopts;
}

fn build_session(sopts: @session::options,
                 demitter: diagnostic::emitter) -> session {
    let codemap = codemap::new_codemap();
    let diagnostic_handler =
        diagnostic::mk_handler(some(demitter));
    let span_diagnostic_handler =
        diagnostic::mk_span_handler(diagnostic_handler, codemap);
    build_session_(sopts, codemap, demitter, span_diagnostic_handler)
}

fn build_session_(
    sopts: @session::options,
    codemap: codemap::codemap,
    demitter: diagnostic::emitter,
    span_diagnostic_handler: diagnostic::span_handler
) -> session {
    let target_cfg = build_target_config(sopts, demitter);
    let cstore = cstore::mk_cstore();
    let filesearch = filesearch::mk_filesearch(
        sopts.maybe_sysroot,
        sopts.target_triple,
        sopts.addl_lib_search_paths);
    @{targ_cfg: target_cfg,
      opts: sopts,
      cstore: cstore,
      parse_sess: @{
          cm: codemap,
          mut next_id: 1,
          span_diagnostic: span_diagnostic_handler,
          mut chpos: 0u,
          mut byte_pos: 0u
      },
      codemap: codemap,
      // For a library crate, this is always none
      mut main_fn: none,
      span_diagnostic: span_diagnostic_handler,
      filesearch: filesearch,
      mut building_library: false,
      working_dir: os::getcwd()}
}

fn parse_pretty(sess: session, &&name: str) -> pp_mode {
    if str::eq(name, "normal") {
        ret ppm_normal;
    } else if str::eq(name, "expanded") {
        ret ppm_expanded;
    } else if str::eq(name, "typed") {
        ret ppm_typed;
    } else if str::eq(name, "expanded,identified") {
        ret ppm_expanded_identified;
    } else if str::eq(name, "identified") {
        ret ppm_identified;
    }
    sess.fatal("argument to `pretty` must be one of `normal`, `typed`, or " +
                   "`identified`");
}

fn opts() -> [getopts::opt] {
    ret [optflag("h"), optflag("help"), optflag("v"), optflag("version"),
         optflag("emit-llvm"), optflagopt("pretty"),
         optflag("ls"), optflag("parse-only"), optflag("no-trans"),
         optflag("O"), optopt("opt-level"), optmulti("L"), optflag("S"),
         optopt("o"), optopt("out-dir"), optflag("xg"),
         optflag("c"), optflag("g"), optflag("save-temps"),
         optopt("sysroot"), optopt("target"), optflag("stats"),
         optflag("time-passes"), optflag("time-llvm-passes"),
         optflag("count-llvm-insns"),
         optflag("no-verify"),

         optmulti("W"), optmulti("warn"),

         optmulti("cfg"), optflag("test"),
         optflag("lib"), optflag("bin"), optflag("static"), optflag("gc"),
         optflag("no-asm-comments"),
         optflag("debug-rustc"),
         optopt("borrowck")];
}

type output_filenames = @{out_filename: str, obj_filename:str};

fn build_output_filenames(input: input,
                          odir: option<str>,
                          ofile: option<str>,
                          sess: session)
        -> output_filenames {
    let mut obj_path = "";
    let mut out_path: str = "";
    let sopts = sess.opts;
    let stop_after_codegen =
        sopts.output_type != link::output_type_exe ||
            sopts.static && sess.building_library;


    let obj_suffix =
        alt sopts.output_type {
          link::output_type_none { "none" }
          link::output_type_bitcode { "bc" }
          link::output_type_assembly { "s" }
          link::output_type_llvm_assembly { "ll" }
          // Object and exe output both use the '.o' extension here
          link::output_type_object | link::output_type_exe {
            "o"
          }
        };

    alt ofile {
      none {
        // "-" as input file will cause the parser to read from stdin so we
        // have to make up a name
        // We want to toss everything after the final '.'
        let dirname = alt odir {
          some(d) { d }
          none {
            alt input {
              str_input(_) {
                os::getcwd()
              }
              file_input(ifile) {
                path::dirname(ifile)
              }
            }
          }
        };

        let base_filename = alt input {
          file_input(ifile) {
            let (path, _) = path::splitext(ifile);
            path::basename(path)
          }
          str_input(_) {
            "rust_out"
          }
        };
        let base_path = path::connect(dirname, base_filename);


        if sess.building_library {
            let basename = path::basename(base_path);
            let dylibname = os::dll_filename(basename);
            out_path = path::connect(dirname, dylibname);
            obj_path = path::connect(dirname, basename + "." + obj_suffix);
        } else {
            out_path = base_path;
            obj_path = base_path + "." + obj_suffix;
        }
      }

      some(out_file) {
        out_path = out_file;
        obj_path = if stop_after_codegen {
            out_file
        } else {
            let (base, _) = path::splitext(out_file);
            let modified = base + "." + obj_suffix;
            modified
        };

        if sess.building_library {
            // FIXME: We might want to warn here; we're actually not going to
            // respect the user's choice of library name when it comes time to
            // link, we'll be linking to lib<basename>-<hash>-<version>.so no
            // matter what.
        }

        if odir != none {
            sess.warn("ignoring --out-dir flag due to -o flag.");
        }
      }
    }
    ret @{out_filename: out_path,
          obj_filename: obj_path};
}

fn early_error(emitter: diagnostic::emitter, msg: str) -> ! {
    emitter(none, msg, diagnostic::fatal);
    fail;
}

fn list_metadata(sess: session, path: str, out: io::writer) {
    metadata::creader::list_file_metadata(sess, path, out);
}

#[cfg(test)]
mod test {

    // When the user supplies --test we should implicitly supply --cfg test
    #[test]
    fn test_switch_implies_cfg_test() {
        let match =
            alt getopts::getopts(["--test"], opts()) {
              ok(m) { m }
              err(f) { fail "test_switch_implies_cfg_test: " +
                       getopts::fail_str(f); }
            };
        let sessopts = build_session_options(match, diagnostic::emit);
        let sess = build_session(sessopts, diagnostic::emit);
        let cfg = build_configuration(sess, "whatever", str_input(""));
        assert (attr::contains_name(cfg, "test"));
    }

    // When the user supplies --test and --cfg test, don't implicitly add
    // another --cfg test
    #[test]
    fn test_switch_implies_cfg_test_unless_cfg_test() {
        let match =
            alt getopts::getopts(["--test", "--cfg=test"], opts()) {
              ok(m) { m }
              err(f) { fail "test_switch_implies_cfg_test_unless_cfg_test: " +
                       getopts::fail_str(f); }
            };
        let sessopts = build_session_options(match, diagnostic::emit);
        let sess = build_session(sessopts, diagnostic::emit);
        let cfg = build_configuration(sess, "whatever", str_input(""));
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
// End:
