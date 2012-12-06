// -*- rust -*-
// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use metadata::{creader, cstore, filesearch};
use session::{Session, Session_, OptLevel, No, Less, Default, Aggressive};
use syntax::parse;
use syntax::{ast, codemap};
use syntax::attr;
use middle::{trans, freevars, kind, ty, typeck, lint};
use syntax::print::{pp, pprust};
use util::ppaux;
use back::link;
use result::{Ok, Err};
use std::getopts;
use std::getopts::{opt_present};
use std::getopts::groups;
use std::getopts::groups::{optopt, optmulti, optflag, optflagopt, getopts};
use io::WriterUtil;
use back::{x86, x86_64};
use std::map::HashMap;
use lib::llvm::llvm;

enum pp_mode {ppm_normal, ppm_expanded, ppm_typed, ppm_identified,
              ppm_expanded_identified }

/**
 * The name used for source code that doesn't originate in a file
 * (e.g. source from stdin or a string)
 */
fn anon_src() -> ~str { ~"<anon>" }

fn source_name(input: input) -> ~str {
    match input {
      file_input(ref ifile) => (*ifile).to_str(),
      str_input(_) => anon_src()
    }
}

fn default_configuration(sess: Session, argv0: ~str, input: input) ->
   ast::crate_cfg {
    let libc = match sess.targ_cfg.os {
      session::os_win32 => ~"msvcrt.dll",
      session::os_macos => ~"libc.dylib",
      session::os_linux => ~"libc.so.6",
      session::os_freebsd => ~"libc.so.7"
      // _ { "libc.so" }
    };

    let mk = attr::mk_name_value_item_str;

    let (arch,wordsz) = match sess.targ_cfg.arch {
      session::arch_x86 => (~"x86",~"32"),
      session::arch_x86_64 => (~"x86_64",~"64"),
      session::arch_arm => (~"arm",~"32")
    };

    return ~[ // Target bindings.
         attr::mk_word_item(os::family()),
         mk(~"target_os", os::sysname()),
         mk(~"target_family", os::family()),
         mk(~"target_arch", arch),
         mk(~"target_word_size", wordsz),
         mk(~"target_libc", libc),
         // Build bindings.
         mk(~"build_compiler", argv0),
         mk(~"build_input", source_name(input))];
}

fn append_configuration(cfg: ast::crate_cfg, name: ~str) -> ast::crate_cfg {
    if attr::contains_name(cfg, name) {
        return cfg;
    } else {
        return vec::append_one(cfg, attr::mk_word_item(name));
    }
}

fn build_configuration(sess: Session, argv0: ~str, input: input) ->
   ast::crate_cfg {
    // Combine the configuration requested by the session (command line) with
    // some default and generated configuration items
    let default_cfg = default_configuration(sess, argv0, input);
    let user_cfg = sess.opts.cfg;
    // If the user wants a test runner, then add the test cfg
    let user_cfg = append_configuration(
        user_cfg,
        if sess.opts.test { ~"test" } else { ~"notest" });
    // If the user requested GC, then add the GC cfg
    let user_cfg = append_configuration(
        user_cfg,
        if sess.opts.gc { ~"gc" } else { ~"nogc" });
    return vec::append(user_cfg, default_cfg);
}

// Convert strings provided as --cfg [cfgspec] into a crate_cfg
fn parse_cfgspecs(cfgspecs: ~[~str]) -> ast::crate_cfg {
    // FIXME (#2399): It would be nice to use the parser to parse all
    // varieties of meta_item here. At the moment we just support the
    // meta_word variant.
    let mut words = ~[];
    for cfgspecs.each |s| {
        words.push(attr::mk_word_item(*s));
    }
    return words;
}

enum input {
    /// Load source from file
    file_input(Path),
    /// The string is the source
    str_input(~str)
}

fn parse_input(sess: Session, cfg: ast::crate_cfg, input: input)
    -> @ast::crate {
    match input {
      file_input(ref file) => {
        parse::parse_crate_from_file(&(*file), cfg, sess.parse_sess)
      }
      str_input(ref src) => {
        // FIXME (#2319): Don't really want to box the source string
        parse::parse_crate_from_source_str(
            anon_src(), @(*src), cfg, sess.parse_sess)
      }
    }
}

fn time<T>(do_it: bool, what: ~str, thunk: fn() -> T) -> T {
    if !do_it { return thunk(); }
    let start = std::time::precise_time_s();
    let rv = thunk();
    let end = std::time::precise_time_s();
    io::stdout().write_str(fmt!("time: %3.3f s\t%s\n",
                                end - start, what));
    move rv
}

enum compile_upto {
    cu_parse,
    cu_expand,
    cu_typeck,
    cu_no_trans,
    cu_everything,
}

impl compile_upto : cmp::Eq {
    pure fn eq(&self, other: &compile_upto) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &compile_upto) -> bool { !(*self).eq(other) }
}

fn compile_upto(sess: Session, cfg: ast::crate_cfg,
                input: input, upto: compile_upto,
                outputs: Option<output_filenames>)
    -> {crate: @ast::crate, tcx: Option<ty::ctxt>} {
    let time_passes = sess.time_passes();
    let mut crate = time(time_passes, ~"parsing",
                         ||parse_input(sess, cfg, input) );
    if upto == cu_parse { return {crate: crate, tcx: None}; }

    sess.building_library = session::building_library(
        sess.opts.crate_type, crate, sess.opts.test);

    crate = time(time_passes, ~"configuration", ||
        front::config::strip_unconfigured_items(crate));

    crate = time(time_passes, ~"maybe building test harness", ||
        front::test::modify_for_testing(sess, crate));

    crate = time(time_passes, ~"expansion", ||
        syntax::ext::expand::expand_crate(sess.parse_sess, cfg,
                                          crate));

    if upto == cu_expand { return {crate: crate, tcx: None}; }

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
                             sess.opts.static,
                             sess.parse_sess.interner));

    let lang_items = time(time_passes, ~"language item collection", ||
         middle::lang_items::collect_language_items(crate, sess));

    let { def_map: def_map,
          exp_map2: exp_map2,
          trait_map: trait_map } =
        time(time_passes, ~"resolution", ||
             middle::resolve::resolve_crate(sess, lang_items, crate));

    let freevars = time(time_passes, ~"freevar finding", ||
        freevars::annotate_freevars(def_map, crate));

    let region_map = time(time_passes, ~"region resolution", ||
        middle::region::resolve_crate(sess, def_map, crate));

    let rp_set = time(time_passes, ~"region parameterization inference", ||
        middle::region::determine_rp_in_crate(sess, ast_map, def_map, crate));


    let outputs = outputs.get();

    let (llmod, link_meta) = {

        let ty_cx = ty::mk_ctxt(sess, def_map, ast_map, freevars,
                                region_map, rp_set, move lang_items, crate);

        let (method_map, vtable_map) =
            time(time_passes, ~"typechecking", ||
                 typeck::check_crate(ty_cx,
                                     trait_map,
                                     crate));

        // These next two const passes can probably be merged
        time(time_passes, ~"const marking", ||
             middle::const_eval::process_crate(crate, def_map, ty_cx));

        time(time_passes, ~"const checking", ||
             middle::check_const::check_crate(sess, crate, ast_map, def_map,
                                              method_map, ty_cx));

        if upto == cu_typeck { return {crate: crate, tcx: Some(ty_cx)}; }

        time(time_passes, ~"privacy checking", ||
             middle::privacy::check_crate(ty_cx, &method_map, crate));

        time(time_passes, ~"loop checking", ||
             middle::check_loop::check_crate(ty_cx, crate));

        time(time_passes, ~"alt checking", ||
             middle::check_alt::check_crate(ty_cx, method_map, crate));

        time(time_passes, ~"mode computation", ||
             middle::mode::compute_modes(ty_cx, method_map, crate));

        let last_use_map =
            time(time_passes, ~"liveness checking", ||
                 middle::liveness::check_crate(ty_cx, method_map, crate));

        let (root_map, mutbl_map) =
            time(time_passes, ~"borrow checking", ||
                 middle::borrowck::check_crate(ty_cx, method_map,
                                               last_use_map, crate));

        time(time_passes, ~"kind checking", ||
             kind::check_crate(ty_cx, method_map, last_use_map, crate));

        time(time_passes, ~"lint checking", ||
             lint::check_crate(ty_cx, crate));

        if upto == cu_no_trans { return {crate: crate, tcx: Some(ty_cx)}; }

        let maps = {mutbl_map: mutbl_map,
                    root_map: root_map,
                    last_use_map: last_use_map,
                    method_map: method_map,
                    vtable_map: vtable_map};

        time(time_passes, ~"translation", ||
             trans::base::trans_crate(sess, crate, ty_cx,
                                      &outputs.obj_filename,
                                      exp_map2, maps))

    };


    time(time_passes, ~"LLVM passes", ||
        link::write::run_passes(sess, llmod,
                                &outputs.obj_filename));

    let stop_after_codegen =
        sess.opts.output_type != link::output_type_exe ||
        (sess.opts.static && sess.building_library)    ||
        sess.opts.jit;

    if stop_after_codegen { return {crate: crate, tcx: None}; }

    time(time_passes, ~"linking", ||
         link::link_binary(sess,
                           &outputs.obj_filename,
                           &outputs.out_filename, link_meta));

    return {crate: crate, tcx: None};
}

fn compile_input(sess: Session, cfg: ast::crate_cfg, input: input,
                 outdir: &Option<Path>, output: &Option<Path>) {

    let upto = if sess.opts.parse_only { cu_parse }
               else if sess.opts.no_trans { cu_no_trans }
               else { cu_everything };
    let outputs = build_output_filenames(input, outdir, output, sess);
    compile_upto(sess, cfg, input, upto, Some(outputs));
}

fn pretty_print_input(sess: Session, cfg: ast::crate_cfg, input: input,
                      ppm: pp_mode) {
    fn ann_paren_for_expr(node: pprust::ann_node) {
        match node {
          pprust::node_expr(s, _) => pprust::popen(s),
          _ => ()
        }
    }
    fn ann_typed_post(tcx: ty::ctxt, node: pprust::ann_node) {
        match node {
          pprust::node_expr(s, expr) => {
            pp::space(s.s);
            pp::word(s.s, ~"as");
            pp::space(s.s);
            pp::word(s.s, ppaux::ty_to_str(tcx, ty::expr_ty(tcx, expr)));
            pprust::pclose(s);
          }
          _ => ()
        }
    }
    fn ann_identified_post(node: pprust::ann_node) {
        match node {
          pprust::node_item(s, item) => {
            pp::space(s.s);
            pprust::synth_comment(s, int::to_str(item.id, 10u));
          }
          pprust::node_block(s, ref blk) => {
            pp::space(s.s);
            pprust::synth_comment(
                s, ~"block " + int::to_str((*blk).node.id, 10u));
          }
          pprust::node_expr(s, expr) => {
            pp::space(s.s);
            pprust::synth_comment(s, int::to_str(expr.id, 10u));
            pprust::pclose(s);
          }
          pprust::node_pat(s, pat) => {
            pp::space(s.s);
            pprust::synth_comment(s, ~"pat " + int::to_str(pat.id, 10u));
          }
        }
    }

    // Because the pretty printer needs to make a pass over the source
    // to collect comments and literals, and we need to support reading
    // from stdin, we're going to just suck the source into a string
    // so both the parser and pretty-printer can use it.
    let upto = match ppm {
      ppm_expanded | ppm_expanded_identified => cu_expand,
      ppm_typed => cu_typeck,
      _ => cu_parse
    };
    let {crate, tcx} = compile_upto(sess, cfg, input, upto, None);

    let ann = match ppm {
      ppm_typed => {
        {pre: ann_paren_for_expr,
         post: |a| ann_typed_post(tcx.get(), a) }
      }
      ppm_identified | ppm_expanded_identified => {
        {pre: ann_paren_for_expr, post: ann_identified_post}
      }
      ppm_expanded | ppm_normal => pprust::no_ann()
    };
    let is_expanded = upto != cu_parse;
    let src = sess.codemap.get_filemap(source_name(input)).src;
    do io::with_str_reader(*src) |rdr| {
        pprust::print_crate(sess.codemap, sess.parse_sess.interner,
                            sess.span_diagnostic, crate,
                            source_name(input),
                            rdr, io::stdout(), ann, is_expanded);
    }
}

fn get_os(triple: ~str) -> Option<session::os> {
    if str::contains(triple, ~"win32") ||
               str::contains(triple, ~"mingw32") {
            Some(session::os_win32)
        } else if str::contains(triple, ~"darwin") {
            Some(session::os_macos)
        } else if str::contains(triple, ~"linux") {
            Some(session::os_linux)
        } else if str::contains(triple, ~"freebsd") {
            Some(session::os_freebsd)
        } else { None }
}

fn get_arch(triple: ~str) -> Option<session::arch> {
    if str::contains(triple, ~"i386") ||
        str::contains(triple, ~"i486") ||
               str::contains(triple, ~"i586") ||
               str::contains(triple, ~"i686") ||
               str::contains(triple, ~"i786") {
            Some(session::arch_x86)
        } else if str::contains(triple, ~"x86_64") {
            Some(session::arch_x86_64)
        } else if str::contains(triple, ~"arm") ||
                      str::contains(triple, ~"xscale") {
            Some(session::arch_arm)
        } else { None }
}

fn build_target_config(sopts: @session::options,
                       demitter: diagnostic::emitter) -> @session::config {
    let os = match get_os(sopts.target_triple) {
      Some(os) => os,
      None => early_error(demitter, ~"unknown operating system")
    };
    let arch = match get_arch(sopts.target_triple) {
      Some(arch) => arch,
      None => early_error(demitter,
                          ~"unknown architecture: " + sopts.target_triple)
    };
    let (int_type, uint_type, float_type) = match arch {
      session::arch_x86 => (ast::ty_i32, ast::ty_u32, ast::ty_f64),
      session::arch_x86_64 => (ast::ty_i64, ast::ty_u64, ast::ty_f64),
      session::arch_arm => (ast::ty_i32, ast::ty_u32, ast::ty_f64)
    };
    let target_strs = match arch {
      session::arch_x86 => x86::get_target_strs(os),
      session::arch_x86_64 => x86_64::get_target_strs(os),
      session::arch_arm => x86::get_target_strs(os)
    };
    let target_cfg: @session::config =
        @{os: os, arch: arch, target_strs: target_strs, int_type: int_type,
          uint_type: uint_type, float_type: float_type};
    return target_cfg;
}

fn host_triple() -> ~str {
    // Get the host triple out of the build environment. This ensures that our
    // idea of the host triple is the same as for the set of libraries we've
    // actually built.  We can't just take LLVM's host triple because they
    // normalize all ix86 architectures to i386.

    // FIXME (#2400): Instead of grabbing the host triple we really should
    // be grabbing (at compile time) the target triple that this rustc is
    // built with and calling that (at runtime) the host triple.
    let ht = env!("CFG_HOST_TRIPLE");
    return if ht != ~"" {
            ht
        } else {
            fail ~"rustc built without CFG_HOST_TRIPLE"
        };
}

fn build_session_options(binary: ~str,
                         matches: &getopts::Matches,
                         demitter: diagnostic::emitter) -> @session::options {
    let crate_type = if opt_present(matches, ~"lib") {
        session::lib_crate
    } else if opt_present(matches, ~"bin") {
        session::bin_crate
    } else {
        session::unknown_crate
    };
    let static = opt_present(matches, ~"static");
    let gc = opt_present(matches, ~"gc");

    let parse_only = opt_present(matches, ~"parse-only");
    let no_trans = opt_present(matches, ~"no-trans");

    let lint_levels = [lint::allow, lint::warn,
                       lint::deny, lint::forbid];
    let mut lint_opts = ~[];
    let lint_dict = lint::get_lint_dict();
    for lint_levels.each |level| {
        let level_name = lint::level_to_str(*level);
        let level_short = level_name.substr(0,1).to_upper();
        let flags = vec::append(getopts::opt_strs(matches, level_short),
                                getopts::opt_strs(matches, level_name));
        for flags.each |lint_name| {
            let lint_name = str::replace(*lint_name, ~"-", ~"_");
            match lint_dict.find(lint_name) {
              None => {
                early_error(demitter, fmt!("unknown %s flag: %s",
                                           level_name, lint_name));
              }
              Some(lint) => {
                lint_opts.push((lint.lint, *level));
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
            let (name, _, bit) = *pair;
            if name == *debug_flag { this_bit = bit; break; }
        }
        if this_bit == 0u {
            early_error(demitter, fmt!("unknown debug flag: %s", *debug_flag))
        }
        debugging_opts |= this_bit;
    }
    if debugging_opts & session::debug_llvm != 0 {
        llvm::LLVMSetDebug(1);
    }

    let jit = opt_present(matches, ~"jit");
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
    let sysroot_opt = sysroot_opt.map(|m| Path(*m));
    let target_opt = getopts::opt_maybe_str(matches, ~"target");
    let save_temps = getopts::opt_present(matches, ~"save-temps");
    match output_type {
      // unless we're emitting huamn-readable assembly, omit comments.
      link::output_type_llvm_assembly | link::output_type_assembly => (),
      _ => debugging_opts |= session::no_asm_comments
    }
    let opt_level = {
        if (debugging_opts & session::no_opt) != 0 {
            No
        } else if opt_present(matches, ~"O") {
            if opt_present(matches, ~"opt-level") {
                early_error(demitter, ~"-O and --opt-level both provided");
            }
            Default
        } else if opt_present(matches, ~"opt-level") {
            match getopts::opt_str(matches, ~"opt-level") {
              ~"0" => No,
              ~"1" => Less,
              ~"2" => Default,
              ~"3" => Aggressive,
              _ => {
                early_error(demitter, ~"optimization level needs " +
                            ~"to be between 0-3")
              }
            }
        } else { No }
    };
    let target =
        match target_opt {
            None => host_triple(),
            Some(ref s) => (*s)
        };

    let addl_lib_search_paths =
        getopts::opt_strs(matches, ~"L")
        .map(|s| Path(*s));
    let cfg = parse_cfgspecs(getopts::opt_strs(matches, ~"cfg"));
    let test = opt_present(matches, ~"test");
    let sopts: @session::options =
        @{crate_type: crate_type,
          static: static,
          gc: gc,
          optimize: opt_level,
          debuginfo: debuginfo,
          extra_debuginfo: extra_debuginfo,
          lint_opts: lint_opts,
          save_temps: save_temps,
          jit: jit,
          output_type: output_type,
          addl_lib_search_paths: addl_lib_search_paths,
          maybe_sysroot: sysroot_opt,
          target_triple: target,
          cfg: cfg,
          binary: binary,
          test: test,
          parse_only: parse_only,
          no_trans: no_trans,
          debugging_opts: debugging_opts};
    return sopts;
}

fn build_session(sopts: @session::options,
                 demitter: diagnostic::emitter) -> Session {
    let codemap = @codemap::CodeMap::new();
    let diagnostic_handler =
        diagnostic::mk_handler(Some(demitter));
    let span_diagnostic_handler =
        diagnostic::mk_span_handler(diagnostic_handler, codemap);
    build_session_(sopts, codemap, demitter, span_diagnostic_handler)
}

fn build_session_(sopts: @session::options,
                  cm: @codemap::CodeMap,
                  demitter: diagnostic::emitter,
                  span_diagnostic_handler: diagnostic::span_handler)
               -> Session {
    let target_cfg = build_target_config(sopts, demitter);
    let p_s = parse::new_parse_sess_special_handler(span_diagnostic_handler,
                                                    cm);
    let cstore = cstore::mk_cstore(p_s.interner);
    let filesearch = filesearch::mk_filesearch(
        sopts.maybe_sysroot,
        sopts.target_triple,
        sopts.addl_lib_search_paths);
    let lint_settings = lint::mk_lint_settings();
    Session_(@{targ_cfg: target_cfg,
               opts: sopts,
               cstore: cstore,
               parse_sess: p_s,
               codemap: cm,
               // For a library crate, this is always none
               mut main_fn: None,
               span_diagnostic: span_diagnostic_handler,
               filesearch: filesearch,
               mut building_library: false,
               working_dir: os::getcwd(),
               lint_settings: lint_settings})
}

fn parse_pretty(sess: Session, &&name: ~str) -> pp_mode {
    match name {
      ~"normal" => ppm_normal,
      ~"expanded" => ppm_expanded,
      ~"typed" => ppm_typed,
      ~"expanded,identified" => ppm_expanded_identified,
      ~"identified" => ppm_identified,
      _ => {
        sess.fatal(~"argument to `pretty` must be one of `normal`, \
                     `expanded`, `typed`, `identified`, \
                     or `expanded,identified`");
      }
    }
}

// rustc command line options
fn optgroups() -> ~[getopts::groups::OptGroup] {
 ~[
  optflag(~"",  ~"bin", ~"Compile an executable crate (default)"),
  optflag(~"c", ~"",    ~"Compile and assemble, but do not link"),
  optmulti(~"", ~"cfg", ~"Configure the compilation
                          environment", ~"SPEC"),
  optflag(~"",  ~"emit-llvm",
                        ~"Produce an LLVM bitcode file"),
  optflag(~"g", ~"",    ~"Produce debug info (experimental)"),
  optflag(~"",  ~"gc",  ~"Garbage collect shared data (experimental)"),
  optflag(~"h", ~"help",~"Display this message"),
  optmulti(~"L", ~"",   ~"Add a directory to the library search path",
                              ~"PATH"),
  optflag(~"",  ~"lib", ~"Compile a library crate"),
  optflag(~"",  ~"ls",  ~"List the symbols defined by a library crate"),
  optflag(~"",  ~"jit", ~"Execute using JIT (experimental)"),
  optflag(~"", ~"no-trans",
                        ~"Run all passes except translation; no output"),
  optflag(~"O", ~"",    ~"Equivalent to --opt-level=2"),
  optopt(~"o", ~"",     ~"Write output to <filename>", ~"FILENAME"),
  optopt(~"", ~"opt-level",
                        ~"Optimize with possible levels 0-3", ~"LEVEL"),
  optopt( ~"",  ~"out-dir",
                        ~"Write output to compiler-chosen filename
                          in <dir>", ~"DIR"),
  optflag(~"", ~"parse-only",
                        ~"Parse only; do not compile, assemble, or link"),
  optflagopt(~"", ~"pretty",
                        ~"Pretty-print the input instead of compiling;
                          valid types are: normal (un-annotated source),
                          expanded (crates expanded),
                          typed (crates expanded, with type annotations),
                          or identified (fully parenthesized,
                          AST nodes and blocks with IDs)", ~"TYPE"),
  optflag(~"S", ~"",    ~"Compile only; do not assemble or link"),
  optflag(~"", ~"xg",   ~"Extra debugging info (experimental)"),
  optflag(~"", ~"save-temps",
                        ~"Write intermediate files (.bc, .opt.bc, .o)
                          in addition to normal output"),
  optflag(~"", ~"static",
                        ~"Use or produce static libraries or binaries
                         (experimental)"),
  optopt(~"", ~"sysroot",
                        ~"Override the system root", ~"PATH"),
  optflag(~"", ~"test", ~"Build a test harness"),
  optopt(~"", ~"target",
                        ~"Target triple cpu-manufacturer-kernel[-os]
                          to compile for (see
         http://sources.redhat.com/autobook/autobook/autobook_17.html
                          for detail)", ~"TRIPLE"),
  optmulti(~"W", ~"warn",
                        ~"Set lint warnings", ~"OPT"),
  optmulti(~"A", ~"allow",
                        ~"Set lint allowed", ~"OPT"),
  optmulti(~"D", ~"deny",
                        ~"Set lint denied", ~"OPT"),
  optmulti(~"F", ~"forbid",
                        ~"Set lint forbidden", ~"OPT"),
  optmulti(~"Z", ~"",   ~"Set internal debugging options", "FLAG"),
  optflag( ~"v", ~"version",
                        ~"Print version info and exit"),
 ]
}

type output_filenames = @{out_filename:Path, obj_filename:Path};

fn build_output_filenames(input: input,
                          odir: &Option<Path>,
                          ofile: &Option<Path>,
                          sess: Session)
        -> output_filenames {
    let obj_path;
    let out_path;
    let sopts = sess.opts;
    let stop_after_codegen =
        sopts.output_type != link::output_type_exe ||
            sopts.static && sess.building_library;


    let obj_suffix =
        match sopts.output_type {
          link::output_type_none => ~"none",
          link::output_type_bitcode => ~"bc",
          link::output_type_assembly => ~"s",
          link::output_type_llvm_assembly => ~"ll",
          // Object and exe output both use the '.o' extension here
          link::output_type_object | link::output_type_exe => ~"o"
        };

    match *ofile {
      None => {
        // "-" as input file will cause the parser to read from stdin so we
        // have to make up a name
        // We want to toss everything after the final '.'
        let dirpath = match *odir {
          Some(ref d) => (*d),
          None => match input {
            str_input(_) => os::getcwd(),
            file_input(ref ifile) => (*ifile).dir_path()
          }
        };

        let stem = match input {
          file_input(ref ifile) => (*ifile).filestem().get(),
          str_input(_) => ~"rust_out"
        };

        if sess.building_library {
            out_path = dirpath.push(os::dll_filename(stem));
            obj_path = dirpath.push(stem).with_filetype(obj_suffix);
        } else {
            out_path = dirpath.push(stem);
            obj_path = dirpath.push(stem).with_filetype(obj_suffix);
        }
      }

      Some(ref out_file) => {
        out_path = (*out_file);
        obj_path = if stop_after_codegen {
            (*out_file)
        } else {
            (*out_file).with_filetype(obj_suffix)
        };

        if sess.building_library {
            // FIXME (#2401): We might want to warn here; we're actually not
            // going to respect the user's choice of library name when it
            // comes time to link, we'll be linking to
            // lib<basename>-<hash>-<version>.so no matter what.
        }

        if *odir != None {
            sess.warn(~"ignoring --out-dir flag due to -o flag.");
        }
      }
    }
    return @{out_filename: out_path,
             obj_filename: obj_path};
}

fn early_error(emitter: diagnostic::emitter, msg: ~str) -> ! {
    emitter(None, msg, diagnostic::fatal);
    fail;
}

fn list_metadata(sess: Session, path: &Path, out: io::Writer) {
    metadata::loader::list_file_metadata(
        sess.parse_sess.interner,
        session::sess_os_to_meta_os(sess.targ_cfg.os), path, out);
}

#[cfg(test)]
mod test {
    #[legacy_exports];

    // When the user supplies --test we should implicitly supply --cfg test
    #[test]
    fn test_switch_implies_cfg_test() {
        let matches =
            &match getopts(~[~"--test"], optgroups()) {
              Ok(m) => m,
              Err(f) => fail ~"test_switch_implies_cfg_test: " +
                             getopts::fail_str(f)
            };
        let sessopts = build_session_options(
            ~"rustc", matches, diagnostic::emit);
        let sess = build_session(sessopts, diagnostic::emit);
        let cfg = build_configuration(sess, ~"whatever", str_input(~""));
        assert (attr::contains_name(cfg, ~"test"));
    }

    // When the user supplies --test and --cfg test, don't implicitly add
    // another --cfg test
    #[test]
    fn test_switch_implies_cfg_test_unless_cfg_test() {
        let matches =
            &match getopts(~[~"--test", ~"--cfg=test"], optgroups()) {
              Ok(m) => m,
              Err(f) => {
                fail ~"test_switch_implies_cfg_test_unless_cfg_test: " +
                    getopts::fail_str(f);
              }
            };
        let sessopts = build_session_options(
            ~"rustc", matches, diagnostic::emit);
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
