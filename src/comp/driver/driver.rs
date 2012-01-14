
// -*- rust -*-
import metadata::{creader, cstore};
import session::session;
import syntax::parse::{parser};
import syntax::{ast, codemap};
import front::attr;
import middle::{trans, resolve, freevars, kind, ty, typeck, fn_usage,
                last_use};
import syntax::print::{pp, pprust};
import util::{ppaux, filesearch};
import back::link;
import core::{option, str, int, result};
import result::{ok, err};
import std::{fs, io, getopts};
import io::reader_util;
import option::{some, none};
import getopts::{optopt, optmulti, optflag, optflagopt, opt_present};
import back::{x86, x86_64};

tag pp_mode { ppm_normal; ppm_expanded; ppm_typed; ppm_identified; }

fn default_configuration(sess: session::session, argv0: str, input: str) ->
   ast::crate_cfg {
    let libc =
        alt sess.targ_cfg.os {
          session::os_win32. { "msvcrt.dll" }
          session::os_macos. { "libc.dylib" }
          session::os_linux. { "libc.so.6" }
          session::os_freebsd. { "libc.so.7" }
          _ { "libc.so" }
        };

    let mk = attr::mk_name_value_item_str;

    let arch = alt sess.targ_cfg.arch {
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
    let words = [];
    for s: str in cfgspecs { words += [attr::mk_word_item(s)]; }
    ret words;
}

fn input_is_stdin(filename: str) -> bool { filename == "-" }

fn parse_input(sess: session::session, cfg: ast::crate_cfg, input: str) ->
   @ast::crate {
    if !input_is_stdin(input) {
        parser::parse_crate_from_file(input, cfg, sess.parse_sess)
    } else { parse_input_src(sess, cfg, input).crate }
}

fn parse_input_src(sess: session::session, cfg: ast::crate_cfg, infile: str)
   -> {crate: @ast::crate, src: str} {
    let src_stream = if infile != "-" {
        alt io::file_reader(infile) {
          result::ok(reader) { reader }
          result::err(e) {
            sess.fatal(e)
          }
        }
    } else {
        io::stdin()
    };
    let srcbytes = src_stream.read_whole_stream();
    let src = str::unsafe_from_bytes(srcbytes);
    let crate =
        parser::parse_crate_from_source_str(infile, src, cfg,
                                            sess.parse_sess);
    ret {crate: crate, src: src};
}

fn time<T>(do_it: bool, what: str, thunk: fn@() -> T) -> T {
    if !do_it { ret thunk(); }
    let start = std::time::precise_time_s();
    let rv = thunk();
    let end = std::time::precise_time_s();
    #error("time: %s took %s s", what,
           float::to_str(end - start, 3u));
    ret rv;
}

fn inject_libcore_reference(sess: session::session,
                            crate: @ast::crate) -> @ast::crate {

    fn spanned<T: copy>(x: T) -> @ast::spanned<T> {
        ret @{node: x,
              span: {lo: 0u, hi: 0u,
                     expanded_from: codemap::os_none}};
    }

    let n1 = sess.next_node_id();
    let n2 = sess.next_node_id();

    let vi1 = spanned(ast::view_item_use("core", [], n1));
    let vi2 = spanned(ast::view_item_import_glob(@["core"], n2));

    let vis = [vi1, vi2] + crate.node.module.view_items;

    ret @{node: {module: { view_items: vis with crate.node.module }
                 with crate.node} with *crate }
}


fn compile_input(sess: session::session, cfg: ast::crate_cfg, input: str,
                 outdir: option::t<str>, output: option::t<str>) {

    let time_passes = sess.opts.time_passes;
    let crate =
        time(time_passes, "parsing", bind parse_input(sess, cfg, input));
    if sess.opts.parse_only { ret; }

    sess.building_library =
        session::building_library(sess.opts.crate_type, crate);

    crate =
        time(time_passes, "configuration",
             bind front::config::strip_unconfigured_items(crate));
    crate =
        time(time_passes, "maybe building test harness",
             bind front::test::modify_for_testing(sess, crate));
    crate =
        time(time_passes, "expansion",
             bind syntax::ext::expand::expand_crate(sess, crate));

    if sess.opts.libcore {
        crate = inject_libcore_reference(sess, crate);
    }

    let ast_map =
        time(time_passes, "ast indexing",
             bind middle::ast_map::map_crate(*crate));
    time(time_passes, "external crate/lib resolution",
         bind creader::read_crates(sess, *crate));
    let {def_map, exp_map, impl_map} =
        time(time_passes, "resolution",
             bind resolve::resolve_crate(sess, ast_map, crate));
    let freevars =
        time(time_passes, "freevar finding",
             bind freevars::annotate_freevars(def_map, crate));
    time(time_passes, "const checking",
         bind middle::check_const::check_crate(sess, crate));
    let ty_cx = ty::mk_ctxt(sess, def_map, ast_map, freevars);
    let (method_map, dict_map) =
        time(time_passes, "typechecking",
             bind typeck::check_crate(ty_cx, impl_map, crate));
    time(time_passes, "block-use checking",
         bind middle::block_use::check_crate(ty_cx, crate));
    time(time_passes, "function usage",
         bind fn_usage::check_crate_fn_usage(ty_cx, crate));
    time(time_passes, "alt checking",
         bind middle::check_alt::check_crate(ty_cx, crate));
    time(time_passes, "typestate checking",
         bind middle::tstate::ck::check_crate(ty_cx, crate));
    let mut_map =
        time(time_passes, "mutability checking",
             bind middle::mut::check_crate(ty_cx, crate));
    let (copy_map, ref_map) =
        time(time_passes, "alias checking",
             bind middle::alias::check_crate(ty_cx, crate));
    let last_uses = time(time_passes, "last use finding",
        bind last_use::find_last_uses(crate, def_map, ref_map, ty_cx));
    time(time_passes, "kind checking",
         bind kind::check_crate(ty_cx, method_map, last_uses, crate));
    if sess.opts.no_trans { ret; }

    let outputs = build_output_filenames(input, outdir, output, sess);

    let (llmod, link_meta) =
        time(time_passes, "translation",
             bind trans::trans_crate(sess, crate, ty_cx,
                                     outputs.obj_filename, exp_map, ast_map,
                                     mut_map, copy_map, last_uses,
                                     method_map, dict_map));
    time(time_passes, "LLVM passes",
         bind link::write::run_passes(sess, llmod, outputs.obj_filename));

    let stop_after_codegen =
        sess.opts.output_type != link::output_type_exe ||
            sess.opts.static && sess.building_library;

    if stop_after_codegen { ret; }

    time(time_passes, "Linking",
         bind link::link_binary(sess, outputs.obj_filename,
                                outputs.out_filename, link_meta));
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
        let {def_map, impl_map, _} =
            resolve::resolve_crate(sess, amap, crate);
        let freevars = freevars::annotate_freevars(def_map, crate);
        let ty_cx = ty::mk_ctxt(sess, def_map, amap, freevars);
        typeck::check_crate(ty_cx, impl_map, crate);
        ann = {pre: ann_paren_for_expr, post: bind ann_typed_post(ty_cx, _)};
      }
      ppm_identified. {
        ann = {pre: ann_paren_for_expr, post: ann_identified_post};
      }
      ppm_normal. { ann = pprust::no_ann(); }
    }
    pprust::print_crate(sess.codemap, crate, input,
                        io::string_reader(src), io::stdout(), ann);
}

fn get_os(triple: str) -> session::os {
    ret if str::find(triple, "win32") >= 0 ||
               str::find(triple, "mingw32") >= 0 {
            session::os_win32
        } else if str::find(triple, "darwin") >= 0 {
            session::os_macos
        } else if str::find(triple, "linux") >= 0 {
            session::os_linux
        } else if str::find(triple, "freebsd") >= 0 {
            session::os_freebsd
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
    let libcore = !opt_present(match, "no-core");
    let verify = !opt_present(match, "no-verify");
    let save_temps = opt_present(match, "save-temps");
    let extra_debuginfo = opt_present(match, "xg");
    let debuginfo = opt_present(match, "g") || extra_debuginfo;
    let stats = opt_present(match, "stats");
    let time_passes = opt_present(match, "time-passes");
    let time_llvm_passes = opt_present(match, "time-llvm-passes");
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
    let warn_unused_imports = opt_present(match, "warn-unused-imports");
    let sopts: @session::options =
        @{crate_type: crate_type,
          static: static,
          libcore: libcore,
          optimize: opt_level,
          debuginfo: debuginfo,
          extra_debuginfo: extra_debuginfo,
          verify: verify,
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
          no_asm_comments: no_asm_comments,
          warn_unused_imports: warn_unused_imports};
    ret sopts;
}

fn build_session(sopts: @session::options, input: str) -> session::session {
    let target_cfg = build_target_config(sopts);
    let cstore = cstore::mk_cstore();
    let filesearch = filesearch::mk_filesearch(
        sopts.maybe_sysroot,
        sopts.target_triple,
        sopts.addl_lib_search_paths);
    let codemap = codemap::new_codemap();
    let diagnostic_handler = diagnostic::mk_codemap_handler(codemap);
    @{targ_cfg: target_cfg,
      opts: sopts,
      cstore: cstore,
      parse_sess: @{
          cm: codemap,
          mutable next_id: 1,
          diagnostic: diagnostic_handler
      },
      codemap: codemap,
      // For a library crate, this is always none
      mutable main_fn: none,
      diagnostic: diagnostic_handler,
      filesearch: filesearch,
      mutable building_library: false,
      working_dir: fs::dirname(input)}
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
         optopt("o"), optopt("out-dir"), optflag("xg"),
         optflag("c"), optflag("g"), optflag("save-temps"),
         optopt("sysroot"), optopt("target"), optflag("stats"),
         optflag("time-passes"), optflag("time-llvm-passes"),
         optflag("no-verify"),
         optmulti("cfg"), optflag("test"),
         optflag("no-core"),
         optflag("lib"), optflag("bin"), optflag("static"), optflag("gc"),
         optflag("no-asm-comments"),
         optflag("warn-unused-imports")];
}

fn build_output_filenames(ifile: str,
                          odir: option::t<str>,
                          ofile: option::t<str>,
                          sess: session::session)
        -> @{out_filename: str, obj_filename:str} {
    let obj_path = "";
    let out_path: str = "";
    let sopts = sess.opts;
    let stop_after_codegen =
        sopts.output_type != link::output_type_exe ||
            sopts.static && sess.building_library;


    let obj_suffix =
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

    alt ofile {
      none. {
        // "-" as input file will cause the parser to read from stdin so we
        // have to make up a name
        // We want to toss everything after the final '.'
        let dirname = alt odir {
          some(d) { d }
          none. {
            if input_is_stdin(ifile) {
                std::os::getcwd()
            } else {
                fs::dirname(ifile)
            }
          }
        };

        let (base_path, _) = if !input_is_stdin(ifile) {
            fs::splitext(ifile)
        } else {
            (fs::connect(dirname, "rust_out"), "")
        };


        if sess.building_library {
            let basename = fs::basename(base_path);
            let dylibname = std::os::dylib_filename(basename);
            out_path = fs::connect(dirname, dylibname);
            obj_path = fs::connect(dirname, basename + "." + obj_path);
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
            let (base, _) = fs::splitext(out_file);
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
            sess.warn("Ignoring --out-dir flag due to -o flag.");
        }
      }
    }
    ret @{out_filename: out_path,
          obj_filename: obj_path};
}

fn early_error(msg: str) -> ! {
    diagnostic::emit_error(none, msg);
    fail;
}

fn list_metadata(sess: session::session, path: str, out: io::writer) {
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
            };
        let sessopts = build_session_options(match);
        let sess = build_session(sessopts, "");
        let cfg = build_configuration(sess, "whatever", "whatever");
        assert (attr::contains_name(cfg, "test"));
    }

    // When the user supplies --test and --cfg test, don't implicitly add
    // another --cfg test
    #[test]
    fn test_switch_implies_cfg_test_unless_cfg_test() {
        let match =
            alt getopts::getopts(["--test", "--cfg=test"], opts()) {
              ok(m) { m }
            };
        let sessopts = build_session_options(match);
        let sess = build_session(sessopts, "");
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
// End:
