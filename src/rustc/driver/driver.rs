// -*- rust -*-
import metadata::{creader, cstore, filesearch};
import session::{session, session_};
import syntax::parse;
import syntax::{ast, codemap};
import syntax::attr;
import middle::{trans, freevars, kind, ty, typeck, lint};
import syntax::print::{pp, pprust};
import util::ppaux;
import back::link;
import result::{ok, err};
import std::getopts;
import io::{reader_util, writer_util};
import getopts::{optopt, optmulti, optflag, optflagopt, opt_present};
import back::{x86, x86_64};
import std::map::hashmap;
import lib::llvm::llvm;

enum pp_mode {ppm_normal, ppm_expanded, ppm_typed, ppm_identified,
              ppm_expanded_identified }

/**
 * The name used for source code that doesn't originate in a file
 * (e.g. source from stdin or a string)
 */
fn anon_src() -> ~str { ~"<anon>" }

fn source_name(input: input) -> ~str {
    alt input {
      file_input(ifile) { ifile }
      str_input(_) { anon_src() }
    }
}

fn default_configuration(sess: session, argv0: ~str, input: input) ->
   ast::crate_cfg {
    let libc = alt sess.targ_cfg.os {
      session::os_win32 { ~"msvcrt.dll" }
      session::os_macos { ~"libc.dylib" }
      session::os_linux { ~"libc.so.6" }
      session::os_freebsd { ~"libc.so.7" }
      // _ { "libc.so" }
    };

    let mk = attr::mk_name_value_item_str;

    let arch = alt sess.targ_cfg.arch {
      session::arch_x86 { ~"x86" }
      session::arch_x86_64 { ~"x86_64" }
      session::arch_arm { ~"arm" }
    };

    ret ~[ // Target bindings.
         attr::mk_word_item(@os::family()),
         mk(@~"target_os", os::sysname()),
         mk(@~"target_family", os::family()),
         mk(@~"target_arch", arch),
         mk(@~"target_libc", libc),
         // Build bindings.
         mk(@~"build_compiler", argv0),
         mk(@~"build_input", source_name(input))];
}

fn build_configuration(sess: session, argv0: ~str, input: input) ->
   ast::crate_cfg {
    // Combine the configuration requested by the session (command line) with
    // some default and generated configuration items
    let default_cfg = default_configuration(sess, argv0, input);
    let user_cfg = sess.opts.cfg;
    // If the user wants a test runner, then add the test cfg
    let gen_cfg =
        {
            if sess.opts.test && !attr::contains_name(user_cfg, ~"test") {
                ~[attr::mk_word_item(@~"test")]
            } else {
                ~[attr::mk_word_item(@~"notest")]
            }
        };
    ret vec::append(vec::append(user_cfg, gen_cfg), default_cfg);
}

// Convert strings provided as --cfg [cfgspec] into a crate_cfg
fn parse_cfgspecs(cfgspecs: ~[~str]) -> ast::crate_cfg {
    // FIXME (#2399): It would be nice to use the parser to parse all
    // varieties of meta_item here. At the moment we just support the
    // meta_word variant.
    let mut words = ~[];
    for cfgspecs.each |s| { vec::push(words, attr::mk_word_item(@s)); }
    ret words;
}

enum input {
    /// Load source from file
    file_input(~str),
    /// The string is the source
    str_input(~str)
}

fn parse_input(sess: session, cfg: ast::crate_cfg, input: input)
    -> @ast::crate {
    alt input {
      file_input(file) {
        parse::parse_crate_from_file(file, cfg, sess.parse_sess)
      }
      str_input(src) {
        // FIXME (#2319): Don't really want to box the source string
        parse::parse_crate_from_source_str(
            anon_src(), @src, cfg, sess.parse_sess)
      }
    }
}

fn time<T>(do_it: bool, what: ~str, thunk: fn() -> T) -> T {
    if !do_it { ret thunk(); }
    let start = std::time::precise_time_s();
    let rv = thunk();
    let end = std::time::precise_time_s();
    io::stdout().write_str(fmt!{"time: %3.3f s\t%s\n",
                                end - start, what});
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
    let time_passes = sess.time_passes();
    let mut crate = time(time_passes, ~"parsing",
                         ||parse_input(sess, cfg, input) );
    if upto == cu_parse { ret {crate: crate, tcx: none}; }

    sess.building_library = session::building_library(
        sess.opts.crate_type, crate, sess.opts.test);

    crate = time(time_passes, ~"configuration", ||
        front::config::strip_unconfigured_items(crate));

    crate = time(time_passes, ~"maybe building test harness", ||
        front::test::modify_for_testing(sess, crate));

    crate = time(time_passes, ~"expansion", ||
        syntax::ext::expand::expand_crate(sess.parse_sess, sess.opts.cfg,
                                          crate));

    if upto == cu_expand { ret {crate: crate, tcx: none}; }

    crate = time(time_passes, ~"intrinsic injection", ||
        front::intrinsic_inject::inject_intrinsic(sess, crate));

    crate = time(time_passes, ~"core injection", ||
        front::core_inject::maybe_inject_libcore_ref(sess, crate));

    time(time_passes, ~"building lint settings table", ||
        lint::build_settings_crate(sess, crate));

    let ast_map = time(time_passes, ~"ast indexing", ||
            syntax::ast_map::map_crate(sess.diagnostic(), *crate));

    time(time_passes, ~"external crate/lib resolution", ||
        creader::read_crates(sess.diagnostic(), *crate, sess.cstore,
                             sess.filesearch,
                             session::sess_os_to_meta_os(sess.targ_cfg.os),
                             sess.opts.static));

    let lang_items = time(time_passes, ~"language item collection", ||
         middle::lang_items::collect_language_items(crate, sess));

    let { def_map: def_map,
          exp_map: exp_map,
          impl_map: impl_map,
          trait_map: trait_map } =
        time(time_passes, ~"resolution", ||
             middle::resolve3::resolve_crate(sess, lang_items, crate));

    let freevars = time(time_passes, ~"freevar finding", ||
        freevars::annotate_freevars(def_map, crate));

    let region_map = time(time_passes, ~"region resolution", ||
        middle::region::resolve_crate(sess, def_map, crate));

    let rp_set = time(time_passes, ~"region parameterization inference", ||
        middle::region::determine_rp_in_crate(sess, ast_map, def_map, crate));

    let ty_cx = ty::mk_ctxt(sess, def_map, ast_map, freevars,
                            region_map, rp_set);

    let (method_map, vtable_map) = time(time_passes, ~"typechecking", ||
                                        typeck::check_crate(ty_cx,
                                                            impl_map,
                                                            trait_map,
                                                            crate));
    // These next two const passes can probably be merged
    time(time_passes, ~"const marking", ||
        middle::const_eval::process_crate(crate, def_map, ty_cx));

    time(time_passes, ~"const checking", ||
        middle::check_const::check_crate(sess, crate, ast_map, def_map,
                                         method_map, ty_cx));

    if upto == cu_typeck { ret {crate: crate, tcx: some(ty_cx)}; }

    time(time_passes, ~"block-use checking", ||
        middle::block_use::check_crate(ty_cx, crate));

    time(time_passes, ~"loop checking", ||
        middle::check_loop::check_crate(ty_cx, crate));

    time(time_passes, ~"alt checking", ||
        middle::check_alt::check_crate(ty_cx, crate));

    let last_use_map = time(time_passes, ~"liveness checking", ||
        middle::liveness::check_crate(ty_cx, method_map, crate));

    let (root_map, mutbl_map) = time(time_passes, ~"borrow checking", ||
        middle::borrowck::check_crate(ty_cx, method_map,
                                      last_use_map, crate));

    time(time_passes, ~"kind checking", ||
        kind::check_crate(ty_cx, method_map, last_use_map, crate));

    time(time_passes, ~"lint checking", || lint::check_crate(ty_cx, crate));

    if upto == cu_no_trans { ret {crate: crate, tcx: some(ty_cx)}; }
    let outputs = option::get(outputs);

    let maps = {mutbl_map: mutbl_map, root_map: root_map,
                last_use_map: last_use_map,
                impl_map: impl_map, method_map: method_map,
                vtable_map: vtable_map};

    let (llmod, link_meta) = time(time_passes, ~"translation", ||
        trans::base::trans_crate(sess, crate, ty_cx, outputs.obj_filename,
                                 exp_map, maps));

    time(time_passes, ~"LLVM passes", ||
        link::write::run_passes(sess, llmod, outputs.obj_filename));

    let stop_after_codegen =
        sess.opts.output_type != link::output_type_exe ||
            sess.opts.static && sess.building_library;

    if stop_after_codegen { ret {crate: crate, tcx: some(ty_cx)}; }

    time(time_passes, ~"linking", ||
         link::link_binary(sess, outputs.obj_filename,
                           outputs.out_filename, link_meta));

    ret {crate: crate, tcx: some(ty_cx)};
}

fn compile_input(sess: session, cfg: ast::crate_cfg, input: input,
                 outdir: option<~str>, output: option<~str>) {

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
            pp::word(s.s, ~"as");
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
                                  ~"block " + int::to_str(blk.node.id, 10u));
          }
          pprust::node_expr(s, expr) {
            pp::space(s.s);
            pprust::synth_comment(s, int::to_str(expr.id, 10u));
            pprust::pclose(s);
          }
          pprust::node_pat(s, pat) {
            pp::space(s.s);
            pprust::synth_comment(s, ~"pat " + int::to_str(pat.id, 10u));
          }
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

    let ann = alt ppm {
      ppm_typed {
        {pre: ann_paren_for_expr,
         post: |a| ann_typed_post(option::get(tcx), a) }
      }
      ppm_identified | ppm_expanded_identified {
        {pre: ann_paren_for_expr, post: ann_identified_post}
      }
      ppm_expanded | ppm_normal { pprust::no_ann() }
    };
    let is_expanded = upto != cu_parse;
    let src = codemap::get_filemap(sess.codemap, source_name(input)).src;
    do io::with_str_reader(*src) |rdr| {
        pprust::print_crate(sess.codemap, sess.parse_sess.interner,
                            sess.span_diagnostic, crate,
                            source_name(input),
                            rdr, io::stdout(), ann, is_expanded);
    }
}

fn get_os(triple: ~str) -> option<session::os> {
    ret if str::contains(triple, ~"win32") ||
               str::contains(triple, ~"mingw32") {
            some(session::os_win32)
        } else if str::contains(triple, ~"darwin") {
            some(session::os_macos)
        } else if str::contains(triple, ~"linux") {
            some(session::os_linux)
        } else if str::contains(triple, ~"freebsd") {
            some(session::os_freebsd)
        } else { none };
}

fn get_arch(triple: ~str) -> option<session::arch> {
    ret if str::contains(triple, ~"i386") || str::contains(triple, ~"i486") ||
               str::contains(triple, ~"i586") ||
               str::contains(triple, ~"i686") ||
               str::contains(triple, ~"i786") {
            some(session::arch_x86)
        } else if str::contains(triple, ~"x86_64") {
            some(session::arch_x86_64)
        } else if str::contains(triple, ~"arm") ||
                      str::contains(triple, ~"xscale") {
            some(session::arch_arm)
        } else { none };
}

fn build_target_config(sopts: @session::options,
                       demitter: diagnostic::emitter) -> @session::config {
    let os = alt get_os(sopts.target_triple) {
      some(os) { os }
      none { early_error(demitter, ~"unknown operating system") }
    };
    let arch = alt get_arch(sopts.target_triple) {
      some(arch) { arch }
      none { early_error(demitter,
                          ~"unknown architecture: " + sopts.target_triple) }
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

fn host_triple() -> ~str {
    // Get the host triple out of the build environment. This ensures that our
    // idea of the host triple is the same as for the set of libraries we've
    // actually built.  We can't just take LLVM's host triple because they
    // normalize all ix86 architectures to i386.

    // FIXME (#2400): Instead of grabbing the host triple we really should
    // be grabbing (at compile time) the target triple that this rustc is
    // built with and calling that (at runtime) the host triple.
    let ht = env!{"CFG_HOST_TRIPLE"};
    ret if ht != ~"" {
            ht
        } else {
            fail ~"rustc built without CFG_HOST_TRIPLE"
        };
}

fn build_session_options(matches: getopts::matches,
                         demitter: diagnostic::emitter) -> @session::options {
    let crate_type = if opt_present(matches, ~"lib") {
        session::lib_crate
    } else if opt_present(matches, ~"bin") {
        session::bin_crate
    } else {
        session::unknown_crate
    };
    let static = opt_present(matches, ~"static");

    let parse_only = opt_present(matches, ~"parse-only");
    let no_trans = opt_present(matches, ~"no-trans");

    let lint_levels = [lint::allow, lint::warn,
                       lint::deny, lint::forbid];
    let mut lint_opts = ~[];
    let lint_dict = lint::get_lint_dict();
    for lint_levels.each |level| {
        let level_name = lint::level_to_str(level);
        let level_short = level_name.substr(0,1).to_upper();
        let flags = vec::append(getopts::opt_strs(matches, level_short),
                                getopts::opt_strs(matches, level_name));
        for flags.each |lint_name| {
            let lint_name = str::replace(lint_name, ~"-", ~"_");
            alt lint_dict.find(lint_name) {
              none {
                early_error(demitter, fmt!{"unknown %s flag: %s",
                                           level_name, lint_name});
              }
              some(lint) {
                vec::push(lint_opts, (lint.lint, level));
              }
            }
        }
    }

    let mut debugging_opts = 0u;
    let debug_flags = getopts::opt_strs(matches, ~"Z");
    let debug_map = session::debugging_opts_map();
    for debug_flags.each |debug_flag| {
        let mut this_bit = 0u;
        for debug_map.each |pair| {
            let (name, _, bit) = pair;
            if name == debug_flag { this_bit = bit; break; }
        }
        if this_bit == 0u {
            early_error(demitter, fmt!{"unknown debug flag: %s", debug_flag})
        }
        debugging_opts |= this_bit;
    }
    if debugging_opts & session::debug_llvm != 0 {
        llvm::LLVMSetDebug(1);
    }

    let output_type =
        if parse_only || no_trans {
            link::output_type_none
        } else if opt_present(matches, ~"S") &&
                  opt_present(matches, ~"emit-llvm") {
            link::output_type_llvm_assembly
        } else if opt_present(matches, ~"S") {
            link::output_type_assembly
        } else if opt_present(matches, ~"c") {
            link::output_type_object
        } else if opt_present(matches, ~"emit-llvm") {
            link::output_type_bitcode
        } else { link::output_type_exe };
    let extra_debuginfo = opt_present(matches, ~"xg");
    let debuginfo = opt_present(matches, ~"g") || extra_debuginfo;
    let sysroot_opt = getopts::opt_maybe_str(matches, ~"sysroot");
    let target_opt = getopts::opt_maybe_str(matches, ~"target");
    let save_temps = getopts::opt_present(matches, ~"save-temps");
    alt output_type {
      // unless we're emitting huamn-readable assembly, omit comments.
      link::output_type_llvm_assembly | link::output_type_assembly {}
      _ { debugging_opts |= session::no_asm_comments; }
    }
    let opt_level: uint =
        if opt_present(matches, ~"O") {
            if opt_present(matches, ~"opt-level") {
                early_error(demitter, ~"-O and --opt-level both provided");
            }
            2u
        } else if opt_present(matches, ~"opt-level") {
            alt getopts::opt_str(matches, ~"opt-level") {
              ~"0" { 0u }
              ~"1" { 1u }
              ~"2" { 2u }
              ~"3" { 3u }
              _ {
                early_error(demitter, ~"optimization level needs " +
                            ~"to be between 0-3")
              }
            }
        } else { 0u };
    let target =
        alt target_opt {
            none { host_triple() }
            some(s) { s }
        };

    let addl_lib_search_paths = getopts::opt_strs(matches, ~"L");
    let cfg = parse_cfgspecs(getopts::opt_strs(matches, ~"cfg"));
    let test = opt_present(matches, ~"test");
    let sopts: @session::options =
        @{crate_type: crate_type,
          static: static,
          optimize: opt_level,
          debuginfo: debuginfo,
          extra_debuginfo: extra_debuginfo,
          lint_opts: lint_opts,
          save_temps: save_temps,
          output_type: output_type,
          addl_lib_search_paths: addl_lib_search_paths,
          maybe_sysroot: sysroot_opt,
          target_triple: target,
          cfg: cfg,
          test: test,
          parse_only: parse_only,
          no_trans: no_trans,
          debugging_opts: debugging_opts};
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

fn build_session_(sopts: @session::options,
                  cm: codemap::codemap,
                  demitter: diagnostic::emitter,
                  span_diagnostic_handler: diagnostic::span_handler)
               -> session {

    let target_cfg = build_target_config(sopts, demitter);
    let cstore = cstore::mk_cstore();
    let filesearch = filesearch::mk_filesearch(
        sopts.maybe_sysroot,
        sopts.target_triple,
        sopts.addl_lib_search_paths);
    let lint_settings = lint::mk_lint_settings();
    session_(@{targ_cfg: target_cfg,
               opts: sopts,
               cstore: cstore,
               parse_sess:
          parse::new_parse_sess_special_handler(span_diagnostic_handler, cm),
               codemap: cm,
               // For a library crate, this is always none
               mut main_fn: none,
               span_diagnostic: span_diagnostic_handler,
               filesearch: filesearch,
               mut building_library: false,
               working_dir: os::getcwd(),
               lint_settings: lint_settings})
}

fn parse_pretty(sess: session, &&name: ~str) -> pp_mode {
    if str::eq(name, ~"normal") {
        ret ppm_normal;
    } else if str::eq(name, ~"expanded") {
        ret ppm_expanded;
    } else if str::eq(name, ~"typed") {
        ret ppm_typed;
    } else if str::eq(name, ~"expanded,identified") {
        ret ppm_expanded_identified;
    } else if str::eq(name, ~"identified") {
        ret ppm_identified;
    }
    sess.fatal(~"argument to `pretty` must be one of `normal`, `typed`, or " +
                   ~"`identified`");
}

fn opts() -> ~[getopts::opt] {
    ret ~[optflag(~"h"), optflag(~"help"), optflag(~"v"), optflag(~"version"),
          optflag(~"emit-llvm"), optflagopt(~"pretty"),
          optflag(~"ls"), optflag(~"parse-only"), optflag(~"no-trans"),
          optflag(~"O"), optopt(~"opt-level"), optmulti(~"L"), optflag(~"S"),
          optopt(~"o"), optopt(~"out-dir"), optflag(~"xg"),
          optflag(~"c"), optflag(~"g"), optflag(~"save-temps"),
          optopt(~"sysroot"), optopt(~"target"),

          optmulti(~"W"), optmulti(~"warn"),
          optmulti(~"A"), optmulti(~"allow"),
          optmulti(~"D"), optmulti(~"deny"),
          optmulti(~"F"), optmulti(~"forbid"),

          optmulti(~"Z"),

          optmulti(~"cfg"), optflag(~"test"),
          optflag(~"lib"), optflag(~"bin"),
          optflag(~"static"), optflag(~"gc")];
}

type output_filenames = @{out_filename: ~str, obj_filename:~str};

fn build_output_filenames(input: input,
                          odir: option<~str>,
                          ofile: option<~str>,
                          sess: session)
        -> output_filenames {
    let obj_path;
    let out_path;
    let sopts = sess.opts;
    let stop_after_codegen =
        sopts.output_type != link::output_type_exe ||
            sopts.static && sess.building_library;


    let obj_suffix =
        alt sopts.output_type {
          link::output_type_none { ~"none" }
          link::output_type_bitcode { ~"bc" }
          link::output_type_assembly { ~"s" }
          link::output_type_llvm_assembly { ~"ll" }
          // Object and exe output both use the '.o' extension here
          link::output_type_object | link::output_type_exe {
            ~"o"
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
            ~"rust_out"
          }
        };
        let base_path = path::connect(dirname, base_filename);


        if sess.building_library {
            let basename = path::basename(base_path);
            let dylibname = os::dll_filename(basename);
            out_path = path::connect(dirname, dylibname);
            obj_path = path::connect(dirname, basename + ~"." + obj_suffix);
        } else {
            out_path = base_path;
            obj_path = base_path + ~"." + obj_suffix;
        }
      }

      some(out_file) {
        out_path = out_file;
        obj_path = if stop_after_codegen {
            out_file
        } else {
            let (base, _) = path::splitext(out_file);
            let modified = base + ~"." + obj_suffix;
            modified
        };

        if sess.building_library {
            // FIXME (#2401): We might want to warn here; we're actually not
            // going to respect the user's choice of library name when it
            // comes time to link, we'll be linking to
            // lib<basename>-<hash>-<version>.so no matter what.
        }

        if odir != none {
            sess.warn(~"ignoring --out-dir flag due to -o flag.");
        }
      }
    }
    ret @{out_filename: out_path,
          obj_filename: obj_path};
}

fn early_error(emitter: diagnostic::emitter, msg: ~str) -> ! {
    emitter(none, msg, diagnostic::fatal);
    fail;
}

fn list_metadata(sess: session, path: ~str, out: io::writer) {
    metadata::loader::list_file_metadata(
        session::sess_os_to_meta_os(sess.targ_cfg.os), path, out);
}

#[cfg(test)]
mod test {

    // When the user supplies --test we should implicitly supply --cfg test
    #[test]
    fn test_switch_implies_cfg_test() {
        let matches =
            alt getopts::getopts(~[~"--test"], opts()) {
              ok(m) { m }
              err(f) { fail ~"test_switch_implies_cfg_test: " +
                       getopts::fail_str(f); }
            };
        let sessopts = build_session_options(matches, diagnostic::emit);
        let sess = build_session(sessopts, diagnostic::emit);
        let cfg = build_configuration(sess, ~"whatever", str_input(~""));
        assert (attr::contains_name(cfg, ~"test"));
    }

    // When the user supplies --test and --cfg test, don't implicitly add
    // another --cfg test
    #[test]
    fn test_switch_implies_cfg_test_unless_cfg_test() {
        let matches =
            alt getopts::getopts(~[~"--test", ~"--cfg=test"], opts()) {
              ok(m) { m }
              err(f) {
                fail ~"test_switch_implies_cfg_test_unless_cfg_test: " +
                    getopts::fail_str(f);
              }
            };
        let sessopts = build_session_options(matches, diagnostic::emit);
        let sess = build_session(sessopts, diagnostic::emit);
        let cfg = build_configuration(sess, ~"whatever", str_input(~""));
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
// End:
