// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use back::link;
use back::{arm, x86, x86_64, mips};
use driver::session::{Aggressive};
use driver::session::{Session, Session_, No, Less, Default};
use driver::session;
use front;
use lib::llvm::llvm;
use lib::llvm::{ContextRef, ModuleRef};
use metadata::common::LinkMeta;
use metadata::{creader, cstore, filesearch};
use metadata;
use middle::{trans, freevars, kind, ty, typeck, lint, astencode, reachable};
use middle;
use util::common::time;
use util::ppaux;

use std::hashmap::{HashMap,HashSet};
use std::io;
use std::os;
use std::vec;
use extra::getopts::groups::{optopt, optmulti, optflag, optflagopt};
use extra::getopts::{opt_present};
use extra::getopts;
use syntax::ast;
use syntax::abi;
use syntax::attr;
use syntax::attr::{AttrMetaMethods};
use syntax::codemap;
use syntax::diagnostic;
use syntax::parse;
use syntax::parse::token;
use syntax::print::{pp, pprust};
use syntax;

pub enum pp_mode {
    ppm_normal,
    ppm_expanded,
    ppm_typed,
    ppm_identified,
    ppm_expanded_identified
}

/**
 * The name used for source code that doesn't originate in a file
 * (e.g. source from stdin or a string)
 */
pub fn anon_src() -> @str { @"<anon>" }

pub fn source_name(input: &input) -> @str {
    match *input {
      file_input(ref ifile) => ifile.to_str().to_managed(),
      str_input(_) => anon_src()
    }
}

pub fn default_configuration(sess: Session) ->
   ast::CrateConfig {
    let tos = match sess.targ_cfg.os {
        session::os_win32 =>   @"win32",
        session::os_macos =>   @"macos",
        session::os_linux =>   @"linux",
        session::os_android => @"android",
        session::os_freebsd => @"freebsd"
    };

    // ARM is bi-endian, however using NDK seems to default
    // to little-endian unless a flag is provided.
    let (end,arch,wordsz) = match sess.targ_cfg.arch {
        abi::X86 =>    (@"little", @"x86",    @"32"),
        abi::X86_64 => (@"little", @"x86_64", @"64"),
        abi::Arm =>    (@"little", @"arm",    @"32"),
        abi::Mips =>   (@"big",    @"mips",   @"32")
    };

    let mk = attr::mk_name_value_item_str;
    return ~[ // Target bindings.
         attr::mk_word_item(os::FAMILY.to_managed()),
         mk(@"target_os", tos),
         mk(@"target_family", os::FAMILY.to_managed()),
         mk(@"target_arch", arch),
         mk(@"target_endian", end),
         mk(@"target_word_size", wordsz),
    ];
}

pub fn append_configuration(cfg: &mut ast::CrateConfig, name: @str) {
    if !cfg.iter().any(|mi| mi.name() == name) {
        cfg.push(attr::mk_word_item(name))
    }
}

pub fn build_configuration(sess: Session) ->
   ast::CrateConfig {
    // Combine the configuration requested by the session (command line) with
    // some default and generated configuration items
    let default_cfg = default_configuration(sess);
    let mut user_cfg = sess.opts.cfg.clone();
    // If the user wants a test runner, then add the test cfg
    if sess.opts.test { append_configuration(&mut user_cfg, @"test") }
    // If the user requested GC, then add the GC cfg
    append_configuration(&mut user_cfg, if sess.opts.gc { @"gc" } else { @"nogc" });
    return vec::append(user_cfg, default_cfg);
}

// Convert strings provided as --cfg [cfgspec] into a crate_cfg
fn parse_cfgspecs(cfgspecs: ~[~str],
                  demitter: diagnostic::Emitter) -> ast::CrateConfig {
    do cfgspecs.move_iter().map |s| {
        let sess = parse::new_parse_sess(Some(demitter));
        parse::parse_meta_from_source_str(@"cfgspec", s.to_managed(), ~[], sess)
    }.collect::<ast::CrateConfig>()
}

pub enum input {
    /// Load source from file
    file_input(Path),
    /// The string is the source
    // FIXME (#2319): Don't really want to box the source string
    str_input(@str)
}

pub fn phase_1_parse_input(sess: Session, cfg: ast::CrateConfig, input: &input)
    -> @ast::Crate {
    time(sess.time_passes(), ~"parsing", || {
        match *input {
            file_input(ref file) => {
                parse::parse_crate_from_file(&(*file), cfg.clone(), sess.parse_sess)
            }
            str_input(src) => {
                parse::parse_crate_from_source_str(
                    anon_src(), src, cfg.clone(), sess.parse_sess)
            }
        }
    })
}

// For continuing compilation after a parsed crate has been
// modified

/// Run the "early phases" of the compiler: initial `cfg` processing,
/// syntax expansion, secondary `cfg` expansion, synthesis of a test
/// harness if one is to be provided and injection of a dependency on the
/// standard library and prelude.
pub fn phase_2_configure_and_expand(sess: Session,
                                    cfg: ast::CrateConfig,
                                    mut crate: @ast::Crate) -> @ast::Crate {
    let time_passes = sess.time_passes();

    *sess.building_library = session::building_library(sess.opts.crate_type,
                                                       crate, sess.opts.test);


    // strip before expansion to allow macros to depend on
    // configuration variables e.g/ in
    //
    //   #[macro_escape] #[cfg(foo)]
    //   mod bar { macro_rules! baz!(() => {{}}) }
    //
    // baz! should not use this definition unless foo is enabled.
    crate = time(time_passes, ~"std macros injection", ||
                 syntax::ext::expand::inject_std_macros(sess.parse_sess,
                                                        cfg.clone(),
                                                        crate));

    crate = time(time_passes, ~"configuration 1", ||
                 front::config::strip_unconfigured_items(crate));

    crate = time(time_passes, ~"expansion", ||
                 syntax::ext::expand::expand_crate(sess.parse_sess, cfg.clone(),
                                                   crate));

    // strip again, in case expansion added anything with a #[cfg].
    crate = time(time_passes, ~"configuration 2", ||
                 front::config::strip_unconfigured_items(crate));

    crate = time(time_passes, ~"maybe building test harness", ||
                 front::test::modify_for_testing(sess, crate));

    crate = time(time_passes, ~"std injection", ||
                 front::std_inject::maybe_inject_libstd_ref(sess, crate));

    return crate;
}

pub struct CrateAnalysis {
    exp_map2: middle::resolve::ExportMap2,
    ty_cx: ty::ctxt,
    maps: astencode::Maps,
    reachable: @mut HashSet<ast::NodeId>
}

/// Run the resolution, typechecking, region checking and other
/// miscellaneous analysis passes on the crate. Return various
/// structures carrying the results of the analysis.
pub fn phase_3_run_analysis_passes(sess: Session,
                                   crate: @ast::Crate) -> CrateAnalysis {

    let time_passes = sess.time_passes();
    let ast_map = time(time_passes, ~"ast indexing", ||
                       syntax::ast_map::map_crate(sess.diagnostic(), crate));

    time(time_passes, ~"external crate/lib resolution", ||
         creader::read_crates(sess.diagnostic(), crate, sess.cstore,
                              sess.filesearch,
                              session::sess_os_to_meta_os(sess.targ_cfg.os),
                              sess.opts.is_static,
                              token::get_ident_interner()));

    let lang_items = time(time_passes, ~"language item collection", ||
                          middle::lang_items::collect_language_items(crate, sess));

    let middle::resolve::CrateMap {
        def_map: def_map,
        exp_map2: exp_map2,
        trait_map: trait_map
    } =
        time(time_passes, ~"resolution", ||
             middle::resolve::resolve_crate(sess, lang_items, crate));

    time(time_passes, ~"looking for entry point",
         || middle::entry::find_entry_point(sess, crate, ast_map));

    let freevars = time(time_passes, ~"freevar finding", ||
                        freevars::annotate_freevars(def_map, crate));

    let region_map = time(time_passes, ~"region resolution", ||
                          middle::region::resolve_crate(sess, def_map, crate));

    let rp_set = time(time_passes, ~"region parameterization inference", ||
                      middle::region::determine_rp_in_crate(sess, ast_map, def_map, crate));

    let ty_cx = ty::mk_ctxt(sess, def_map, ast_map, freevars,
                            region_map, rp_set, lang_items);

    // passes are timed inside typeck
    let (method_map, vtable_map) = typeck::check_crate(
        ty_cx, trait_map, crate);

    // These next two const passes can probably be merged
    time(time_passes, ~"const marking", ||
         middle::const_eval::process_crate(crate, ty_cx));

    time(time_passes, ~"const checking", ||
         middle::check_const::check_crate(sess, crate, ast_map, def_map,
                                          method_map, ty_cx));

    time(time_passes, ~"privacy checking", ||
         middle::privacy::check_crate(ty_cx, &method_map, crate));

    time(time_passes, ~"effect checking", ||
         middle::effect::check_crate(ty_cx, method_map, crate));

    time(time_passes, ~"loop checking", ||
         middle::check_loop::check_crate(ty_cx, crate));

    time(time_passes, ~"stack checking", ||
         middle::stack_check::stack_check_crate(ty_cx, crate));

    let middle::moves::MoveMaps {moves_map, moved_variables_set,
                                 capture_map} =
        time(time_passes, ~"compute moves", ||
             middle::moves::compute_moves(ty_cx, method_map, crate));

    time(time_passes, ~"match checking", ||
         middle::check_match::check_crate(ty_cx, method_map,
                                          moves_map, crate));

    time(time_passes, ~"liveness checking", ||
         middle::liveness::check_crate(ty_cx, method_map,
                                       capture_map, crate));

    let (root_map, write_guard_map) =
        time(time_passes, ~"borrow checking", ||
             middle::borrowck::check_crate(ty_cx, method_map,
                                           moves_map, moved_variables_set,
                                           capture_map, crate));

    time(time_passes, ~"kind checking", ||
         kind::check_crate(ty_cx, method_map, crate));

    let reachable_map =
        time(time_passes, ~"reachability checking", ||
             reachable::find_reachable(ty_cx, method_map, crate));

    time(time_passes, ~"lint checking", ||
         lint::check_crate(ty_cx, crate));

    CrateAnalysis {
        exp_map2: exp_map2,
        ty_cx: ty_cx,
        maps: astencode::Maps {
            root_map: root_map,
            method_map: method_map,
            vtable_map: vtable_map,
            write_guard_map: write_guard_map,
            capture_map: capture_map
        },
        reachable: reachable_map
    }
}

pub struct CrateTranslation {
    context: ContextRef,
    module: ModuleRef,
    link: LinkMeta
}

/// Run the translation phase to LLVM, after which the AST and analysis can
/// be discarded.
pub fn phase_4_translate_to_llvm(sess: Session,
                                 crate: @ast::Crate,
                                 analysis: &CrateAnalysis,
                                 outputs: &OutputFilenames) -> CrateTranslation {
    time(sess.time_passes(), ~"translation", ||
         trans::base::trans_crate(sess, crate, analysis,
                                  &outputs.obj_filename))
}

/// Run LLVM itself, producing a bitcode file, assembly file or object file
/// as a side effect.
pub fn phase_5_run_llvm_passes(sess: Session,
                               trans: &CrateTranslation,
                               outputs: &OutputFilenames) {

    // On Windows, LLVM integrated assembler emits bad stack unwind tables when
    // segmented stacks are enabled.  However, unwind info directives in assembly
    // output are OK, so we generate assembly first and then run it through
    // an external assembler.
    // Same for Android.
    if (sess.targ_cfg.os == session::os_android ||
        sess.targ_cfg.os == session::os_win32) &&
        (sess.opts.output_type == link::output_type_object ||
         sess.opts.output_type == link::output_type_exe) {
        let output_type = link::output_type_assembly;
        let asm_filename = outputs.obj_filename.with_filetype("s");

        time(sess.time_passes(), ~"LLVM passes", ||
            link::write::run_passes(sess,
                                    trans.context,
                                    trans.module,
                                    output_type,
                                    &asm_filename));

        link::write::run_assembler(sess, &asm_filename, &outputs.obj_filename);

        // Remove assembly source unless --save-temps was specified
        if !sess.opts.save_temps {
            os::remove_file(&asm_filename);
        }
    } else {
        time(sess.time_passes(), ~"LLVM passes", ||
            link::write::run_passes(sess,
                                    trans.context,
                                    trans.module,
                                    sess.opts.output_type,
                                    &outputs.obj_filename));
    }
}

/// Run the linker on any artifacts that resulted from the LLVM run.
/// This should produce either a finished executable or library.
pub fn phase_6_link_output(sess: Session,
                           trans: &CrateTranslation,
                           outputs: &OutputFilenames) {
    time(sess.time_passes(), ~"linking", ||
         link::link_binary(sess,
                           &outputs.obj_filename,
                           &outputs.out_filename,
                           trans.link));
}

pub fn stop_after_phase_3(sess: Session) -> bool {
   if sess.opts.no_trans {
        debug!("invoked with --no-trans, returning early from compile_input");
        return true;
    }
    return false;
}

pub fn stop_after_phase_1(sess: Session) -> bool {
    if sess.opts.parse_only {
        debug!("invoked with --parse-only, returning early from compile_input");
        return true;
    }
    return false;
}

pub fn stop_after_phase_5(sess: Session) -> bool {
    if sess.opts.output_type != link::output_type_exe {
        debug!("not building executable, returning early from compile_input");
        return true;
    }

    if sess.opts.is_static && *sess.building_library {
        debug!("building static library, returning early from compile_input");
        return true;
    }

    if sess.opts.jit {
        debug!("running JIT, returning early from compile_input");
        return true;
    }
    return false;
}

#[fixed_stack_segment]
pub fn compile_input(sess: Session, cfg: ast::CrateConfig, input: &input,
                     outdir: &Option<Path>, output: &Option<Path>) {
    // We need nested scopes here, because the intermediate results can keep
    // large chunks of memory alive and we want to free them as soon as
    // possible to keep the peak memory usage low
    let (outputs, trans) = {
        let expanded_crate = {
            let crate = phase_1_parse_input(sess, cfg.clone(), input);
            if stop_after_phase_1(sess) { return; }
            phase_2_configure_and_expand(sess, cfg, crate)
        };
        let analysis = phase_3_run_analysis_passes(sess, expanded_crate);
        if stop_after_phase_3(sess) { return; }
        let outputs = build_output_filenames(input, outdir, output, [], sess);
        let trans = phase_4_translate_to_llvm(sess, expanded_crate,
                                              &analysis, outputs);
        (outputs, trans)
    };
    phase_5_run_llvm_passes(sess, &trans, outputs);
    if stop_after_phase_5(sess) { return; }
    phase_6_link_output(sess, &trans, outputs);
}

pub fn pretty_print_input(sess: Session, cfg: ast::CrateConfig, input: &input,
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
            pp::word(s.s, "as");
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
            pprust::synth_comment(s, item.id.to_str());
          }
          pprust::node_block(s, ref blk) => {
            pp::space(s.s);
            pprust::synth_comment(
                s, ~"block " + blk.id.to_str());
          }
          pprust::node_expr(s, expr) => {
            pp::space(s.s);
            pprust::synth_comment(s, expr.id.to_str());
            pprust::pclose(s);
          }
          pprust::node_pat(s, pat) => {
            pp::space(s.s);
            pprust::synth_comment(s, ~"pat " + pat.id.to_str());
          }
        }
    }

    let crate = phase_1_parse_input(sess, cfg.clone(), input);

    let (crate, is_expanded) = match ppm {
        ppm_expanded | ppm_expanded_identified | ppm_typed => {
            (phase_2_configure_and_expand(sess, cfg, crate), true)
        }
        _ => (crate, false)
    };

    let annotation = match ppm {
        ppm_identified | ppm_expanded_identified => {
            pprust::pp_ann {
                pre: ann_paren_for_expr,
                post: ann_identified_post
            }
        }
        ppm_typed => {
            let analysis = phase_3_run_analysis_passes(sess, crate);
            pprust::pp_ann {
                pre: ann_paren_for_expr,
                post: |a| ann_typed_post(analysis.ty_cx, a)
            }
        }
        _ => pprust::no_ann()
    };

    let src = sess.codemap.get_filemap(source_name(input)).src;
    do io::with_str_reader(src) |rdr| {
        pprust::print_crate(sess.codemap, token::get_ident_interner(),
                            sess.span_diagnostic, crate,
                            source_name(input),
                            rdr, io::stdout(),
                            annotation, is_expanded);
    }
}

pub fn get_os(triple: &str) -> Option<session::os> {
    for &(name, os) in os_names.iter() {
        if triple.contains(name) { return Some(os) }
    }
    None
}
static os_names : &'static [(&'static str, session::os)] = &'static [
    ("mingw32", session::os_win32),
    ("win32",   session::os_win32),
    ("darwin",  session::os_macos),
    ("android", session::os_android),
    ("linux",   session::os_linux),
    ("freebsd", session::os_freebsd)];

pub fn get_arch(triple: &str) -> Option<abi::Architecture> {
    for &(arch, abi) in architecture_abis.iter() {
        if triple.contains(arch) { return Some(abi) }
    }
    None
}
static architecture_abis : &'static [(&'static str, abi::Architecture)] = &'static [
    ("i386",   abi::X86),
    ("i486",   abi::X86),
    ("i586",   abi::X86),
    ("i686",   abi::X86),
    ("i786",   abi::X86),

    ("x86_64", abi::X86_64),

    ("arm",    abi::Arm),
    ("xscale", abi::Arm),

    ("mips",   abi::Mips)];

pub fn build_target_config(sopts: @session::options,
                           demitter: diagnostic::Emitter)
                        -> @session::config {
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
      abi::X86 => (ast::ty_i32, ast::ty_u32, ast::ty_f64),
      abi::X86_64 => (ast::ty_i64, ast::ty_u64, ast::ty_f64),
      abi::Arm => (ast::ty_i32, ast::ty_u32, ast::ty_f64),
      abi::Mips => (ast::ty_i32, ast::ty_u32, ast::ty_f64)
    };
    let target_triple = sopts.target_triple.clone();
    let target_strs = match arch {
      abi::X86 => x86::get_target_strs(target_triple, os),
      abi::X86_64 => x86_64::get_target_strs(target_triple, os),
      abi::Arm => arm::get_target_strs(target_triple, os),
      abi::Mips => mips::get_target_strs(target_triple, os)
    };
    let target_cfg = @session::config {
        os: os,
        arch: arch,
        target_strs: target_strs,
        int_type: int_type,
        uint_type: uint_type,
        float_type: float_type
    };
    return target_cfg;
}

pub fn host_triple() -> ~str {
    // Get the host triple out of the build environment. This ensures that our
    // idea of the host triple is the same as for the set of libraries we've
    // actually built.  We can't just take LLVM's host triple because they
    // normalize all ix86 architectures to i386.
    //
    // Instead of grabbing the host triple (for the current host), we grab (at
    // compile time) the target triple that this rustc is built with and
    // calling that (at runtime) the host triple.
    (env!("CFG_COMPILER_TRIPLE")).to_owned()
}

pub fn build_session_options(binary: @str,
                             matches: &getopts::Matches,
                             demitter: diagnostic::Emitter)
                          -> @session::options {
    let crate_type = if opt_present(matches, "lib") {
        session::lib_crate
    } else if opt_present(matches, "bin") {
        session::bin_crate
    } else {
        session::unknown_crate
    };
    let parse_only = opt_present(matches, "parse-only");
    let no_trans = opt_present(matches, "no-trans");

    let lint_levels = [lint::allow, lint::warn,
                       lint::deny, lint::forbid];
    let mut lint_opts = ~[];
    let lint_dict = lint::get_lint_dict();
    for level in lint_levels.iter() {
        let level_name = lint::level_to_str(*level);

        // FIXME: #4318 Instead of to_ascii and to_str_ascii, could use
        // to_ascii_move and to_str_move to not do a unnecessary copy.
        let level_short = level_name.slice_chars(0, 1);
        let level_short = level_short.to_ascii().to_upper().to_str_ascii();
        let flags = vec::append(getopts::opt_strs(matches, level_short),
                                getopts::opt_strs(matches, level_name));
        for lint_name in flags.iter() {
            let lint_name = lint_name.replace("-", "_");
            match lint_dict.find_equiv(&lint_name) {
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
    let debug_flags = getopts::opt_strs(matches, "Z");
    let debug_map = session::debugging_opts_map();
    for debug_flag in debug_flags.iter() {
        let mut this_bit = 0u;
        for tuple in debug_map.iter() {
            let (name, bit) = match *tuple { (ref a, _, b) => (a, b) };
            if name == debug_flag { this_bit = bit; break; }
        }
        if this_bit == 0u {
            early_error(demitter, fmt!("unknown debug flag: %s", *debug_flag))
        }
        debugging_opts |= this_bit;
    }

    if debugging_opts & session::debug_llvm != 0 {
        set_llvm_debug();

        fn set_llvm_debug() {
            #[fixed_stack_segment]; #[inline(never)];
            unsafe { llvm::LLVMSetDebug(1); }
        }
    }

    let output_type =
        if parse_only || no_trans {
            link::output_type_none
        } else if opt_present(matches, "S") &&
                  opt_present(matches, "emit-llvm") {
            link::output_type_llvm_assembly
        } else if opt_present(matches, "S") {
            link::output_type_assembly
        } else if opt_present(matches, "c") {
            link::output_type_object
        } else if opt_present(matches, "emit-llvm") {
            link::output_type_bitcode
        } else { link::output_type_exe };
    let sysroot_opt = getopts::opt_maybe_str(matches, "sysroot").map_move(|m| @Path(m));
    let target = getopts::opt_maybe_str(matches, "target").unwrap_or_default(host_triple());
    let target_cpu = getopts::opt_maybe_str(matches, "target-cpu").unwrap_or_default(~"generic");
    let target_feature = getopts::opt_maybe_str(matches, "target-feature").unwrap_or_default(~"");
    let save_temps = getopts::opt_present(matches, "save-temps");
    let opt_level = {
        if (debugging_opts & session::no_opt) != 0 {
            No
        } else if opt_present(matches, "O") {
            if opt_present(matches, "opt-level") {
                early_error(demitter, ~"-O and --opt-level both provided");
            }
            Default
        } else if opt_present(matches, "opt-level") {
            match getopts::opt_str(matches, "opt-level") {
              ~"0" => No,
              ~"1" => Less,
              ~"2" => Default,
              ~"3" => Aggressive,
              _ => {
                early_error(demitter, ~"optimization level needs to be between 0-3")
              }
            }
        } else { No }
    };
    let gc = debugging_opts & session::gc != 0;
    let jit = debugging_opts & session::jit != 0;
    let extra_debuginfo = debugging_opts & session::extra_debug_info != 0;
    let debuginfo = debugging_opts & session::debug_info != 0 ||
        extra_debuginfo;

    // If debugging info is generated, do not collapse monomorphized function instances.
    // Functions with equivalent llvm code still need separate debugging descriptions because names
    // might differ.
    if debuginfo {
        debugging_opts |= session::no_monomorphic_collapse;
    }

    let statik = debugging_opts & session::statik != 0;

    let addl_lib_search_paths = getopts::opt_strs(matches, "L").map(|s| Path(*s));
    let linker = getopts::opt_maybe_str(matches, "linker");
    let linker_args = getopts::opt_strs(matches, "link-args").flat_map( |a| {
        a.split_iter(' ').map(|arg| arg.to_owned()).collect()
    });

    let cfg = parse_cfgspecs(getopts::opt_strs(matches, "cfg"), demitter);
    let test = opt_present(matches, "test");
    let android_cross_path = getopts::opt_maybe_str(
        matches, "android-cross-path");

    let custom_passes = match getopts::opt_maybe_str(matches, "passes") {
        None => ~[],
        Some(s) => {
            s.split_iter(|c: char| c == ' ' || c == ',').map(|s| {
                s.trim().to_owned()
            }).collect()
        }
    };

    let sopts = @session::options {
        crate_type: crate_type,
        is_static: statik,
        gc: gc,
        optimize: opt_level,
        custom_passes: custom_passes,
        debuginfo: debuginfo,
        extra_debuginfo: extra_debuginfo,
        lint_opts: lint_opts,
        save_temps: save_temps,
        jit: jit,
        output_type: output_type,
        addl_lib_search_paths: @mut addl_lib_search_paths,
        linker: linker,
        linker_args: linker_args,
        maybe_sysroot: sysroot_opt,
        target_triple: target,
        target_cpu: target_cpu,
        target_feature: target_feature,
        cfg: cfg,
        binary: binary,
        test: test,
        parse_only: parse_only,
        no_trans: no_trans,
        debugging_opts: debugging_opts,
        android_cross_path: android_cross_path
    };
    return sopts;
}

pub fn build_session(sopts: @session::options,
                     demitter: diagnostic::Emitter) -> Session {
    let codemap = @codemap::CodeMap::new();
    let diagnostic_handler =
        diagnostic::mk_handler(Some(demitter));
    let span_diagnostic_handler =
        diagnostic::mk_span_handler(diagnostic_handler, codemap);
    build_session_(sopts, codemap, demitter, span_diagnostic_handler)
}

pub fn build_session_(sopts: @session::options,
                      cm: @codemap::CodeMap,
                      demitter: diagnostic::Emitter,
                      span_diagnostic_handler: @mut diagnostic::span_handler)
                   -> Session {
    let target_cfg = build_target_config(sopts, demitter);
    let p_s = parse::new_parse_sess_special_handler(span_diagnostic_handler,
                                                    cm);
    let cstore = @mut cstore::mk_cstore(token::get_ident_interner());
    let filesearch = filesearch::mk_filesearch(
        &sopts.maybe_sysroot,
        sopts.target_triple,
        sopts.addl_lib_search_paths);
    @Session_ {
        targ_cfg: target_cfg,
        opts: sopts,
        cstore: cstore,
        parse_sess: p_s,
        codemap: cm,
        // For a library crate, this is always none
        entry_fn: @mut None,
        entry_type: @mut None,
        span_diagnostic: span_diagnostic_handler,
        filesearch: filesearch,
        building_library: @mut false,
        working_dir: os::getcwd(),
        lints: @mut HashMap::new(),
    }
}

pub fn parse_pretty(sess: Session, name: &str) -> pp_mode {
    match name {
      &"normal" => ppm_normal,
      &"expanded" => ppm_expanded,
      &"typed" => ppm_typed,
      &"expanded,identified" => ppm_expanded_identified,
      &"identified" => ppm_identified,
      _ => {
        sess.fatal("argument to `pretty` must be one of `normal`, \
                    `expanded`, `typed`, `identified`, \
                    or `expanded,identified`");
      }
    }
}

// rustc command line options
pub fn optgroups() -> ~[getopts::groups::OptGroup] {
 ~[
  optflag("",  "bin", "Compile an executable crate (default)"),
  optflag("c", "",    "Compile and assemble, but do not link"),
  optmulti("", "cfg", "Configure the compilation
                          environment", "SPEC"),
  optflag("",  "emit-llvm",
                        "Produce an LLVM assembly file if used with -S option;
                         produce an LLVM bitcode file otherwise"),
  optflag("h", "help","Display this message"),
  optmulti("L", "",   "Add a directory to the library search path",
                              "PATH"),
  optflag("",  "lib", "Compile a library crate"),
  optopt("", "linker", "Program to use for linking instead of the default.", "LINKER"),
  optmulti("",  "link-args", "FLAGS is a space-separated list of flags
                            passed to the linker", "FLAGS"),
  optflag("",  "ls",  "List the symbols defined by a library crate"),
  optflag("", "no-trans",
                        "Run all passes except translation; no output"),
  optflag("O", "",    "Equivalent to --opt-level=2"),
  optopt("o", "",     "Write output to <filename>", "FILENAME"),
  optopt("", "opt-level",
                        "Optimize with possible levels 0-3", "LEVEL"),
  optopt("", "passes", "Comma or space separated list of pass names to use. \
                        Appends to the default list of passes to run for the \
                        specified current optimization level. A value of \
                        \"list\" will list all of the available passes", "NAMES"),
  optopt( "",  "out-dir",
                        "Write output to compiler-chosen filename
                          in <dir>", "DIR"),
  optflag("", "parse-only",
                        "Parse only; do not compile, assemble, or link"),
  optflagopt("", "pretty",
                        "Pretty-print the input instead of compiling;
                          valid types are: normal (un-annotated source),
                          expanded (crates expanded),
                          typed (crates expanded, with type annotations),
                          or identified (fully parenthesized,
                          AST nodes and blocks with IDs)", "TYPE"),
  optflag("S", "",    "Compile only; do not assemble or link"),
  optflag("", "save-temps",
                        "Write intermediate files (.bc, .opt.bc, .o)
                          in addition to normal output"),
  optopt("", "sysroot",
                        "Override the system root", "PATH"),
  optflag("", "test", "Build a test harness"),
  optopt("", "target",
                        "Target triple cpu-manufacturer-kernel[-os]
                          to compile for (see chapter 3.4 of http://www.sourceware.org/autobook/
                          for details)", "TRIPLE"),
  optopt("", "target-cpu",
                        "Select target processor (llc -mcpu=help
                          for details)", "CPU"),
  optopt("", "target-feature",
                        "Target specific attributes (llc -mattr=help
                          for details)", "FEATURE"),
  optopt("", "android-cross-path",
         "The path to the Android NDK", "PATH"),
  optflagopt("W", "warn",
                        "Set lint warnings", "OPT"),
  optmulti("A", "allow",
                        "Set lint allowed", "OPT"),
  optmulti("D", "deny",
                        "Set lint denied", "OPT"),
  optmulti("F", "forbid",
                        "Set lint forbidden", "OPT"),
  optmulti("Z", "",   "Set internal debugging options", "FLAG"),
  optflag( "v", "version",
                        "Print version info and exit"),
 ]
}

pub struct OutputFilenames {
    out_filename: Path,
    obj_filename: Path
}

pub fn build_output_filenames(input: &input,
                              odir: &Option<Path>,
                              ofile: &Option<Path>,
                              attrs: &[ast::Attribute],
                              sess: Session)
                           -> ~OutputFilenames {
    let obj_path;
    let out_path;
    let sopts = sess.opts;
    let stop_after_codegen =
        sopts.output_type != link::output_type_exe ||
            sopts.is_static && *sess.building_library;

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
              Some(ref d) => (*d).clone(),
              None => match *input {
                  str_input(_) => os::getcwd(),
                  file_input(ref ifile) => (*ifile).dir_path()
              }
          };

          let mut stem = match *input {
              file_input(ref ifile) => (*ifile).filestem().unwrap().to_managed(),
              str_input(_) => @"rust_out"
          };

          // If a linkage name meta is present, we use it as the link name
          let linkage_metas = attr::find_linkage_metas(attrs);
          if !linkage_metas.is_empty() {
              // But if a linkage meta is present, that overrides
              let maybe_name = linkage_metas.iter().find(|m| "name" == m.name());
              match maybe_name.chain(|m| m.value_str()) {
                  Some(s) => stem = s,
                  _ => ()
              }
              // If the name is missing, we just default to the filename
              // version
          }

          if *sess.building_library {
              out_path = dirpath.push(os::dll_filename(stem));
              obj_path = dirpath.push(stem).with_filetype(obj_suffix);
          } else {
              out_path = dirpath.push(stem);
              obj_path = dirpath.push(stem).with_filetype(obj_suffix);
          }
      }

      Some(ref out_file) => {
        out_path = (*out_file).clone();
        obj_path = if stop_after_codegen {
            (*out_file).clone()
        } else {
            (*out_file).with_filetype(obj_suffix)
        };

        if *sess.building_library {
            sess.warn("ignoring specified output filename for library.");
        }

        if *odir != None {
            sess.warn("ignoring --out-dir flag due to -o flag.");
        }
      }
    }

    ~OutputFilenames {
        out_filename: out_path,
        obj_filename: obj_path
    }
}

pub fn early_error(emitter: diagnostic::Emitter, msg: ~str) -> ! {
    emitter(None, msg, diagnostic::fatal);
    fail!();
}

pub fn list_metadata(sess: Session, path: &Path, out: @io::Writer) {
    metadata::loader::list_file_metadata(
        token::get_ident_interner(),
        session::sess_os_to_meta_os(sess.targ_cfg.os), path, out);
}

#[cfg(test)]
mod test {

    use driver::driver::{build_configuration, build_session};
    use driver::driver::{build_session_options, optgroups};

    use extra::getopts::groups::getopts;
    use extra::getopts;
    use syntax::attr;
    use syntax::diagnostic;

    // When the user supplies --test we should implicitly supply --cfg test
    #[test]
    fn test_switch_implies_cfg_test() {
        let matches =
            &match getopts([~"--test"], optgroups()) {
              Ok(m) => m,
              Err(f) => fail!("test_switch_implies_cfg_test: %s", getopts::fail_str(f))
            };
        let sessopts = build_session_options(
            @"rustc", matches, diagnostic::emit);
        let sess = build_session(sessopts, diagnostic::emit);
        let cfg = build_configuration(sess);
        assert!((attr::contains_name(cfg, "test")));
    }

    // When the user supplies --test and --cfg test, don't implicitly add
    // another --cfg test
    #[test]
    fn test_switch_implies_cfg_test_unless_cfg_test() {
        let matches =
            &match getopts([~"--test", ~"--cfg=test"], optgroups()) {
              Ok(m) => m,
              Err(f) => {
                fail!("test_switch_implies_cfg_test_unless_cfg_test: %s", getopts::fail_str(f));
              }
            };
        let sessopts = build_session_options(
            @"rustc", matches, diagnostic::emit);
        let sess = build_session(sessopts, diagnostic::emit);
        let cfg = build_configuration(sess);
        let mut test_items = cfg.iter().filter(|m| "test" == m.name());
        assert!(test_items.next().is_some());
        assert!(test_items.next().is_none());
    }
}
