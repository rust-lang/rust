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
use driver::session::{Aggressive, OutputExecutable};
use driver::session::{Session, Session_, No, Less, Default};
use driver::session;
use front;
use lib::llvm::llvm;
use lib::llvm::{ContextRef, ModuleRef};
use metadata::common::LinkMeta;
use metadata::{creader, filesearch};
use metadata::cstore::CStore;
use metadata::creader::Loader;
use metadata;
use middle::{trans, freevars, kind, ty, typeck, lint, astencode, reachable};
use middle;
use util::common::time;
use util::ppaux;

use std::cell::{Cell, RefCell};
use std::hashmap::{HashMap,HashSet};
use std::io;
use std::io::fs;
use std::io::MemReader;
use std::os;
use std::vec;
use extra::getopts::groups::{optopt, optmulti, optflag, optflagopt};
use extra::getopts;
use syntax::ast;
use syntax::abi;
use syntax::attr;
use syntax::attr::{AttrMetaMethods};
use syntax::codemap;
use syntax::diagnostic;
use syntax::ext::base::CrateLoader;
use syntax::parse;
use syntax::parse::token::InternedString;
use syntax::parse::token;
use syntax::print::{pp, pprust};
use syntax;

pub enum PpMode {
    PpmNormal,
    PpmExpanded,
    PpmTyped,
    PpmIdentified,
    PpmExpandedIdentified
}

/**
 * The name used for source code that doesn't originate in a file
 * (e.g. source from stdin or a string)
 */
pub fn anon_src() -> ~str {
    "<anon>".to_str()
}

pub fn source_name(input: &Input) -> ~str {
    match *input {
      // FIXME (#9639): This needs to handle non-utf8 paths
      FileInput(ref ifile) => ifile.as_str().unwrap().to_str(),
      StrInput(_) => anon_src()
    }
}

pub fn default_configuration(sess: Session) ->
   ast::CrateConfig {
    let tos = match sess.targ_cfg.os {
        abi::OsWin32 =>   InternedString::new("win32"),
        abi::OsMacos =>   InternedString::new("macos"),
        abi::OsLinux =>   InternedString::new("linux"),
        abi::OsAndroid => InternedString::new("android"),
        abi::OsFreebsd => InternedString::new("freebsd"),
    };

    // ARM is bi-endian, however using NDK seems to default
    // to little-endian unless a flag is provided.
    let (end,arch,wordsz) = match sess.targ_cfg.arch {
        abi::X86 =>    ("little", "x86",    "32"),
        abi::X86_64 => ("little", "x86_64", "64"),
        abi::Arm =>    ("little", "arm",    "32"),
        abi::Mips =>   ("big",    "mips",   "32")
    };

    let fam = match sess.targ_cfg.os {
        abi::OsWin32 => InternedString::new("windows"),
        _ => InternedString::new("unix")
    };

    let mk = attr::mk_name_value_item_str;
    return ~[ // Target bindings.
         attr::mk_word_item(fam.clone()),
         mk(InternedString::new("target_os"), tos),
         mk(InternedString::new("target_family"), fam),
         mk(InternedString::new("target_arch"), InternedString::new(arch)),
         mk(InternedString::new("target_endian"), InternedString::new(end)),
         mk(InternedString::new("target_word_size"),
            InternedString::new(wordsz)),
    ];
}

pub fn append_configuration(cfg: &mut ast::CrateConfig,
                            name: InternedString) {
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
    if sess.opts.test {
        append_configuration(&mut user_cfg, InternedString::new("test"))
    }
    // If the user requested GC, then add the GC cfg
    append_configuration(&mut user_cfg, if sess.opts.gc {
        InternedString::new("gc")
    } else {
        InternedString::new("nogc")
    });
    return vec::append(user_cfg, default_cfg);
}

// Convert strings provided as --cfg [cfgspec] into a crate_cfg
fn parse_cfgspecs(cfgspecs: ~[~str], demitter: @diagnostic::Emitter)
                  -> ast::CrateConfig {
    cfgspecs.move_iter().map(|s| {
        let sess = parse::new_parse_sess(Some(demitter));
        parse::parse_meta_from_source_str("cfgspec".to_str(), s, ~[], sess)
    }).collect::<ast::CrateConfig>()
}

pub enum Input {
    /// Load source from file
    FileInput(Path),
    /// The string is the source
    StrInput(~str)
}

pub fn phase_1_parse_input(sess: Session, cfg: ast::CrateConfig, input: &Input)
    -> ast::Crate {
    time(sess.time_passes(), "parsing", (), |_| {
        match *input {
            FileInput(ref file) => {
                parse::parse_crate_from_file(&(*file), cfg.clone(), sess.parse_sess)
            }
            StrInput(ref src) => {
                parse::parse_crate_from_source_str(anon_src(),
                                                   (*src).clone(),
                                                   cfg.clone(),
                                                   sess.parse_sess)
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
                                    loader: &mut CrateLoader,
                                    mut crate: ast::Crate)
                                    -> (ast::Crate, syntax::ast_map::Map) {
    let time_passes = sess.time_passes();

    sess.building_library.set(session::building_library(sess.opts, &crate));
    sess.outputs.set(session::collect_outputs(&sess, crate.attrs));

    time(time_passes, "gated feature checking", (), |_|
         front::feature_gate::check_crate(sess, &crate));

    crate = time(time_passes, "crate injection", crate, |crate|
                 front::std_inject::maybe_inject_crates_ref(sess, crate));

    // strip before expansion to allow macros to depend on
    // configuration variables e.g/ in
    //
    //   #[macro_escape] #[cfg(foo)]
    //   mod bar { macro_rules! baz!(() => {{}}) }
    //
    // baz! should not use this definition unless foo is enabled.

    crate = time(time_passes, "configuration 1", crate, |crate|
                 front::config::strip_unconfigured_items(crate));

    crate = time(time_passes, "expansion", crate, |crate| {
        syntax::ext::expand::expand_crate(sess.parse_sess,
                                          loader,
                                          cfg.clone(),
                                          crate)
    });
    // dump the syntax-time crates
    sess.cstore.reset();

    // strip again, in case expansion added anything with a #[cfg].
    crate = time(time_passes, "configuration 2", crate, |crate|
                 front::config::strip_unconfigured_items(crate));

    crate = time(time_passes, "maybe building test harness", crate, |crate|
                 front::test::modify_for_testing(sess, crate));

    crate = time(time_passes, "prelude injection", crate, |crate|
                 front::std_inject::maybe_inject_prelude(sess, crate));

    time(time_passes, "assinging node ids and indexing ast", crate, |crate|
         front::assign_node_ids_and_map::assign_node_ids_and_map(sess, crate))
}

pub struct CrateAnalysis {
    exp_map2: middle::resolve::ExportMap2,
    exported_items: middle::privacy::ExportedItems,
    public_items: middle::privacy::PublicItems,
    ty_cx: ty::ctxt,
    maps: astencode::Maps,
    reachable: @RefCell<HashSet<ast::NodeId>>
}

/// Run the resolution, typechecking, region checking and other
/// miscellaneous analysis passes on the crate. Return various
/// structures carrying the results of the analysis.
pub fn phase_3_run_analysis_passes(sess: Session,
                                   crate: &ast::Crate,
                                   ast_map: syntax::ast_map::Map) -> CrateAnalysis {

    let time_passes = sess.time_passes();

    time(time_passes, "external crate/lib resolution", (), |_|
         creader::read_crates(sess, crate,
                              session::sess_os_to_meta_os(sess.targ_cfg.os),
                              token::get_ident_interner()));

    let lang_items = time(time_passes, "language item collection", (), |_|
                          middle::lang_items::collect_language_items(crate, sess));

    let middle::resolve::CrateMap {
        def_map: def_map,
        exp_map2: exp_map2,
        trait_map: trait_map,
        external_exports: external_exports,
        last_private_map: last_private_map
    } =
        time(time_passes, "resolution", (), |_|
             middle::resolve::resolve_crate(sess, lang_items, crate));

    let named_region_map = time(time_passes, "lifetime resolution", (),
                                |_| middle::resolve_lifetime::crate(sess, crate));

    time(time_passes, "looking for entry point", (),
         |_| middle::entry::find_entry_point(sess, crate, ast_map));

    sess.macro_registrar_fn.with_mut(|r| *r =
        time(time_passes, "looking for macro registrar", (), |_|
            syntax::ext::registrar::find_macro_registrar(
                sess.span_diagnostic, crate)));

    let freevars = time(time_passes, "freevar finding", (), |_|
                        freevars::annotate_freevars(def_map, crate));

    let region_map = time(time_passes, "region resolution", (), |_|
                          middle::region::resolve_crate(sess, crate));

    let ty_cx = ty::mk_ctxt(sess, def_map, named_region_map, ast_map, freevars,
                            region_map, lang_items);

    // passes are timed inside typeck
    let (method_map, vtable_map) = typeck::check_crate(ty_cx, trait_map, crate);

    // These next two const passes can probably be merged
    time(time_passes, "const marking", (), |_|
         middle::const_eval::process_crate(crate, ty_cx));

    time(time_passes, "const checking", (), |_|
         middle::check_const::check_crate(sess, crate, ast_map, def_map,
                                          method_map, ty_cx));

    let maps = (external_exports, last_private_map);
    let (exported_items, public_items) =
            time(time_passes, "privacy checking", maps, |(a, b)|
                 middle::privacy::check_crate(ty_cx, &method_map, &exp_map2,
                                              a, b, crate));

    time(time_passes, "effect checking", (), |_|
         middle::effect::check_crate(ty_cx, method_map, crate));

    time(time_passes, "loop checking", (), |_|
         middle::check_loop::check_crate(ty_cx, crate));

    let middle::moves::MoveMaps {moves_map, moved_variables_set,
                                 capture_map} =
        time(time_passes, "compute moves", (), |_|
             middle::moves::compute_moves(ty_cx, method_map, crate));

    time(time_passes, "match checking", (), |_|
         middle::check_match::check_crate(ty_cx, method_map,
                                          moves_map, crate));

    time(time_passes, "liveness checking", (), |_|
         middle::liveness::check_crate(ty_cx, method_map,
                                       capture_map, crate));

    let root_map =
        time(time_passes, "borrow checking", (), |_|
             middle::borrowck::check_crate(ty_cx, method_map,
                                           moves_map, moved_variables_set,
                                           capture_map, crate));

    time(time_passes, "kind checking", (), |_|
         kind::check_crate(ty_cx, method_map, crate));

    let reachable_map =
        time(time_passes, "reachability checking", (), |_|
             reachable::find_reachable(ty_cx, method_map, &exported_items));

    {
        let reachable_map = reachable_map.borrow();
        time(time_passes, "death checking", (), |_| {
             middle::dead::check_crate(ty_cx,
                                       method_map,
                                       &exported_items,
                                       reachable_map.get(),
                                       crate)
        });
    }

    time(time_passes, "lint checking", (), |_|
         lint::check_crate(ty_cx, method_map, &exported_items, crate));

    CrateAnalysis {
        exp_map2: exp_map2,
        ty_cx: ty_cx,
        exported_items: exported_items,
        public_items: public_items,
        maps: astencode::Maps {
            root_map: root_map,
            method_map: method_map,
            vtable_map: vtable_map,
            capture_map: capture_map
        },
        reachable: reachable_map
    }
}

pub struct CrateTranslation {
    context: ContextRef,
    module: ModuleRef,
    metadata_module: ModuleRef,
    link: LinkMeta,
    metadata: ~[u8],
    reachable: ~[~str],
}

/// Run the translation phase to LLVM, after which the AST and analysis can
/// be discarded.
pub fn phase_4_translate_to_llvm(sess: Session,
                                 crate: ast::Crate,
                                 analysis: &CrateAnalysis,
                                 outputs: &OutputFilenames) -> CrateTranslation {
    time(sess.time_passes(), "translation", crate, |crate|
         trans::base::trans_crate(sess, crate, analysis,
                                  &outputs.obj_filename))
}

/// Run LLVM itself, producing a bitcode file, assembly file or object file
/// as a side effect.
pub fn phase_5_run_llvm_passes(sess: Session,
                               trans: &CrateTranslation,
                               outputs: &OutputFilenames) {

    if sess.no_integrated_as() {
        let output_type = link::OutputTypeAssembly;
        let asm_filename = outputs.obj_filename.with_extension("s");

        time(sess.time_passes(), "LLVM passes", (), |_|
            link::write::run_passes(sess,
                                    trans,
                                    output_type,
                                    &asm_filename));

        link::write::run_assembler(sess, &asm_filename, &outputs.obj_filename);

        // Remove assembly source, unless --save-temps was specified
        if !sess.opts.save_temps {
            fs::unlink(&asm_filename).unwrap();
        }
    } else {
        time(sess.time_passes(), "LLVM passes", (), |_|
            link::write::run_passes(sess,
                                    trans,
                                    sess.opts.output_type,
                                    &outputs.obj_filename));
    }
}

/// Run the linker on any artifacts that resulted from the LLVM run.
/// This should produce either a finished executable or library.
pub fn phase_6_link_output(sess: Session,
                           trans: &CrateTranslation,
                           outputs: &OutputFilenames) {
    time(sess.time_passes(), "linking", (), |_|
         link::link_binary(sess,
                           trans,
                           &outputs.obj_filename,
                           &outputs.out_filename,
                           &trans.link));
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

pub fn stop_after_phase_2(sess: Session) -> bool {
    if sess.opts.no_analysis {
        debug!("invoked with --no-analysis, returning early from compile_input");
        return true;
    }
    return false;
}

pub fn stop_after_phase_5(sess: Session) -> bool {
    if sess.opts.output_type != link::OutputTypeExe {
        debug!("not building executable, returning early from compile_input");
        return true;
    }
    return false;
}

fn write_out_deps(sess: Session, input: &Input, outputs: &OutputFilenames,
                  crate: &ast::Crate) -> io::IoResult<()>
{
    let lm = link::build_link_meta(sess, crate.attrs, &outputs.obj_filename,
                                   &mut ::util::sha2::Sha256::new());

    let sess_outputs = sess.outputs.borrow();
    let out_filenames = sess_outputs.get().iter()
        .map(|&output| link::filename_for_input(&sess, output, &lm,
                                                &outputs.out_filename))
        .to_owned_vec();

    // Write out dependency rules to the dep-info file if requested with
    // --dep-info
    let deps_filename = match sess.opts.write_dependency_info {
        // Use filename from --dep-file argument if given
        (true, Some(ref filename)) => filename.clone(),
        // Use default filename: crate source filename with extension replaced
        // by ".d"
        (true, None) => match *input {
            FileInput(ref input_path) => {
                let filestem = input_path.filestem().expect("input file must \
                                                             have stem");
                let filename = out_filenames[0].dir_path().join(filestem);
                filename.with_extension("d")
            },
            StrInput(..) => {
                sess.warn("can not write --dep-info without a filename \
                           when compiling stdin.");
                return Ok(());
            },
        },
        _ => return Ok(()),
    };

    // Build a list of files used to compile the output and
    // write Makefile-compatible dependency rules
    let files: ~[~str] = {
        let files = sess.codemap.files.borrow();
        files.get()
             .iter()
             .filter_map(|fmap| {
                 if fmap.is_real_file() {
                     Some(fmap.name.clone())
                 } else {
                     None
                 }
             })
             .collect()
    };
    let mut file = if_ok!(io::File::create(&deps_filename));
    for path in out_filenames.iter() {
        if_ok!(write!(&mut file as &mut Writer,
                      "{}: {}\n\n", path.display(), files.connect(" ")));
    }
    Ok(())
}

pub fn compile_input(sess: Session, cfg: ast::CrateConfig, input: &Input,
                     outdir: &Option<Path>, output: &Option<Path>) {
    // We need nested scopes here, because the intermediate results can keep
    // large chunks of memory alive and we want to free them as soon as
    // possible to keep the peak memory usage low
    let (outputs, trans) = {
        let (expanded_crate, ast_map) = {
            let crate = phase_1_parse_input(sess, cfg.clone(), input);
            if stop_after_phase_1(sess) { return; }
            let loader = &mut Loader::new(sess);
            phase_2_configure_and_expand(sess, cfg, loader, crate)
        };
        let outputs = build_output_filenames(input, outdir, output,
                                             expanded_crate.attrs, sess);

        write_out_deps(sess, input, outputs, &expanded_crate).unwrap();

        if stop_after_phase_2(sess) { return; }

        let analysis = phase_3_run_analysis_passes(sess, &expanded_crate, ast_map);
        if stop_after_phase_3(sess) { return; }
        let trans = phase_4_translate_to_llvm(sess, expanded_crate,
                                              &analysis, outputs);
        (outputs, trans)
    };
    phase_5_run_llvm_passes(sess, &trans, outputs);
    if stop_after_phase_5(sess) { return; }
    phase_6_link_output(sess, &trans, outputs);
}

struct IdentifiedAnnotation {
    contents: (),
}

impl pprust::PpAnn for IdentifiedAnnotation {
    fn pre(&self, node: pprust::AnnNode) -> io::IoResult<()> {
        match node {
            pprust::NodeExpr(s, _) => pprust::popen(s),
            _ => Ok(())
        }
    }
    fn post(&self, node: pprust::AnnNode) -> io::IoResult<()> {
        match node {
            pprust::NodeItem(s, item) => {
                if_ok!(pp::space(&mut s.s));
                if_ok!(pprust::synth_comment(s, item.id.to_str()));
            }
            pprust::NodeBlock(s, blk) => {
                if_ok!(pp::space(&mut s.s));
                if_ok!(pprust::synth_comment(s, ~"block " + blk.id.to_str()));
            }
            pprust::NodeExpr(s, expr) => {
                if_ok!(pp::space(&mut s.s));
                if_ok!(pprust::synth_comment(s, expr.id.to_str()));
                if_ok!(pprust::pclose(s));
            }
            pprust::NodePat(s, pat) => {
                if_ok!(pp::space(&mut s.s));
                if_ok!(pprust::synth_comment(s, ~"pat " + pat.id.to_str()));
            }
        }
        Ok(())
    }
}

struct TypedAnnotation {
    analysis: CrateAnalysis,
}

impl pprust::PpAnn for TypedAnnotation {
    fn pre(&self, node: pprust::AnnNode) -> io::IoResult<()> {
        match node {
            pprust::NodeExpr(s, _) => pprust::popen(s),
            _ => Ok(())
        }
    }
    fn post(&self, node: pprust::AnnNode) -> io::IoResult<()> {
        let tcx = self.analysis.ty_cx;
        match node {
            pprust::NodeExpr(s, expr) => {
                if_ok!(pp::space(&mut s.s));
                if_ok!(pp::word(&mut s.s, "as"));
                if_ok!(pp::space(&mut s.s));
                if_ok!(pp::word(&mut s.s,
                                ppaux::ty_to_str(tcx, ty::expr_ty(tcx, expr))));
                if_ok!(pprust::pclose(s));
            }
            _ => ()
        }
        Ok(())
    }
}

pub fn pretty_print_input(sess: Session,
                          cfg: ast::CrateConfig,
                          input: &Input,
                          ppm: PpMode) {
    let crate = phase_1_parse_input(sess, cfg.clone(), input);

    let (crate, ast_map, is_expanded) = match ppm {
        PpmExpanded | PpmExpandedIdentified | PpmTyped => {
            let loader = &mut Loader::new(sess);
            let (crate, ast_map) = phase_2_configure_and_expand(sess, cfg, loader, crate);
            (crate, Some(ast_map), true)
        }
        _ => (crate, None, false)
    };

    let annotation = match ppm {
        PpmIdentified | PpmExpandedIdentified => {
            @IdentifiedAnnotation {
                contents: (),
            } as @pprust::PpAnn
        }
        PpmTyped => {
            let ast_map = ast_map.expect("--pretty=typed missing ast_map");
            let analysis = phase_3_run_analysis_passes(sess, &crate, ast_map);
            @TypedAnnotation {
                analysis: analysis
            } as @pprust::PpAnn
        }
        _ => @pprust::NoAnn as @pprust::PpAnn,
    };

    let src = &sess.codemap.get_filemap(source_name(input)).src;
    let mut rdr = MemReader::new(src.as_bytes().to_owned());
    let stdout = io::stdout();
    pprust::print_crate(sess.codemap,
                        token::get_ident_interner(),
                        sess.span_diagnostic,
                        &crate,
                        source_name(input),
                        &mut rdr,
                        ~stdout as ~io::Writer,
                        annotation,
                        is_expanded).unwrap();
}

pub fn get_os(triple: &str) -> Option<abi::Os> {
    for &(name, os) in os_names.iter() {
        if triple.contains(name) { return Some(os) }
    }
    None
}
static os_names : &'static [(&'static str, abi::Os)] = &'static [
    ("mingw32", abi::OsWin32),
    ("win32",   abi::OsWin32),
    ("darwin",  abi::OsMacos),
    ("android", abi::OsAndroid),
    ("linux",   abi::OsLinux),
    ("freebsd", abi::OsFreebsd)];

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
    ("thumb",  abi::Arm),

    ("mips",   abi::Mips)];

pub fn build_target_config(sopts: @session::Options,
                           demitter: @diagnostic::Emitter)
                           -> @session::Config {
    let os = match get_os(sopts.target_triple) {
      Some(os) => os,
      None => early_error(demitter, "unknown operating system")
    };
    let arch = match get_arch(sopts.target_triple) {
      Some(arch) => arch,
      None => early_error(demitter,
                          "unknown architecture: " + sopts.target_triple)
    };
    let (int_type, uint_type) = match arch {
      abi::X86 => (ast::TyI32, ast::TyU32),
      abi::X86_64 => (ast::TyI64, ast::TyU64),
      abi::Arm => (ast::TyI32, ast::TyU32),
      abi::Mips => (ast::TyI32, ast::TyU32)
    };
    let target_triple = sopts.target_triple.clone();
    let target_strs = match arch {
      abi::X86 => x86::get_target_strs(target_triple, os),
      abi::X86_64 => x86_64::get_target_strs(target_triple, os),
      abi::Arm => arm::get_target_strs(target_triple, os),
      abi::Mips => mips::get_target_strs(target_triple, os)
    };
    let target_cfg = @session::Config {
        os: os,
        arch: arch,
        target_strs: target_strs,
        int_type: int_type,
        uint_type: uint_type,
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
    (env!("CFG_COMPILER")).to_owned()
}

pub fn build_session_options(binary: ~str,
                             matches: &getopts::Matches,
                             demitter: @diagnostic::Emitter)
                             -> @session::Options {
    let mut outputs = ~[];
    if matches.opt_present("lib") {
        outputs.push(session::default_lib_output());
    }
    if matches.opt_present("rlib") {
        outputs.push(session::OutputRlib)
    }
    if matches.opt_present("staticlib") {
        outputs.push(session::OutputStaticlib)
    }
    if matches.opt_present("dylib") {
        outputs.push(session::OutputDylib)
    }
    if matches.opt_present("bin") {
        outputs.push(session::OutputExecutable)
    }

    let parse_only = matches.opt_present("parse-only");
    let no_trans = matches.opt_present("no-trans");
    let no_analysis = matches.opt_present("no-analysis");
    let no_rpath = matches.opt_present("no-rpath");

    let lint_levels = [lint::allow, lint::warn,
                       lint::deny, lint::forbid];
    let mut lint_opts = ~[];
    let lint_dict = lint::get_lint_dict();
    for level in lint_levels.iter() {
        let level_name = lint::level_to_str(*level);

        let level_short = level_name.slice_chars(0, 1);
        let level_short = level_short.to_ascii().to_upper().into_str();
        let flags = vec::append(matches.opt_strs(level_short),
                                matches.opt_strs(level_name));
        for lint_name in flags.iter() {
            let lint_name = lint_name.replace("-", "_");
            match lint_dict.find_equiv(&lint_name) {
              None => {
                early_error(demitter, format!("unknown {} flag: {}",
                                           level_name, lint_name));
              }
              Some(lint) => {
                lint_opts.push((lint.lint, *level));
              }
            }
        }
    }

    let mut debugging_opts = 0;
    let debug_flags = matches.opt_strs("Z");
    let debug_map = session::debugging_opts_map();
    for debug_flag in debug_flags.iter() {
        let mut this_bit = 0;
        for tuple in debug_map.iter() {
            let (name, bit) = match *tuple { (ref a, _, b) => (a, b) };
            if *name == *debug_flag { this_bit = bit; break; }
        }
        if this_bit == 0 {
            early_error(demitter, format!("unknown debug flag: {}", *debug_flag))
        }
        debugging_opts |= this_bit;
    }

    if debugging_opts & session::DEBUG_LLVM != 0 {
        unsafe { llvm::LLVMSetDebug(1); }
    }

    let output_type =
        if parse_only || no_trans {
            link::OutputTypeNone
        } else if matches.opt_present("S") &&
                  matches.opt_present("emit-llvm") {
            link::OutputTypeLlvmAssembly
        } else if matches.opt_present("S") {
            link::OutputTypeAssembly
        } else if matches.opt_present("c") {
            link::OutputTypeObject
        } else if matches.opt_present("emit-llvm") {
            link::OutputTypeBitcode
        } else { link::OutputTypeExe };
    let sysroot_opt = matches.opt_str("sysroot").map(|m| @Path::new(m));
    let target = matches.opt_str("target").unwrap_or(host_triple());
    let target_cpu = matches.opt_str("target-cpu").unwrap_or(~"generic");
    let target_feature = matches.opt_str("target-feature").unwrap_or(~"");
    let save_temps = matches.opt_present("save-temps");
    let opt_level = {
        if (debugging_opts & session::NO_OPT) != 0 {
            No
        } else if matches.opt_present("O") {
            if matches.opt_present("opt-level") {
                early_error(demitter, "-O and --opt-level both provided");
            }
            Default
        } else if matches.opt_present("opt-level") {
            match matches.opt_str("opt-level").unwrap() {
              ~"0" => No,
              ~"1" => Less,
              ~"2" => Default,
              ~"3" => Aggressive,
              _ => {
                early_error(demitter, "optimization level needs to be between 0-3")
              }
            }
        } else { No }
    };
    let gc = debugging_opts & session::GC != 0;
    let extra_debuginfo = debugging_opts & session::EXTRA_DEBUG_INFO != 0;
    let debuginfo = debugging_opts & session::DEBUG_INFO != 0 ||
        extra_debuginfo;

    let addl_lib_search_paths = matches.opt_strs("L").map(|s| {
        Path::new(s.as_slice())
    }).move_iter().collect();
    let ar = matches.opt_str("ar");
    let linker = matches.opt_str("linker");
    let linker_args = matches.opt_strs("link-args").flat_map( |a| {
        a.split(' ').filter_map(|arg| {
            if arg.is_empty() {
                None
            } else {
                Some(arg.to_owned())
            }
        }).collect()
    });

    let cfg = parse_cfgspecs(matches.opt_strs("cfg"), demitter);
    let test = matches.opt_present("test");
    let android_cross_path = matches.opt_str("android-cross-path");
    let write_dependency_info = (matches.opt_present("dep-info"),
                                 matches.opt_str("dep-info").map(|p| Path::new(p)));

    let custom_passes = match matches.opt_str("passes") {
        None => ~[],
        Some(s) => {
            s.split(|c: char| c == ' ' || c == ',').map(|s| {
                s.trim().to_owned()
            }).collect()
        }
    };
    let llvm_args = match matches.opt_str("llvm-args") {
        None => ~[],
        Some(s) => {
            s.split(|c: char| c == ' ' || c == ',').map(|s| {
                s.trim().to_owned()
            }).collect()
        }
    };
    let print_metas = (matches.opt_present("crate-id"),
                       matches.opt_present("crate-name"),
                       matches.opt_present("crate-file-name"));

    let sopts = @session::Options {
        outputs: outputs,
        gc: gc,
        optimize: opt_level,
        custom_passes: custom_passes,
        llvm_args: llvm_args,
        debuginfo: debuginfo,
        extra_debuginfo: extra_debuginfo,
        lint_opts: lint_opts,
        save_temps: save_temps,
        output_type: output_type,
        addl_lib_search_paths: @RefCell::new(addl_lib_search_paths),
        ar: ar,
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
        no_analysis: no_analysis,
        no_rpath: no_rpath,
        debugging_opts: debugging_opts,
        android_cross_path: android_cross_path,
        write_dependency_info: write_dependency_info,
        print_metas: print_metas,
    };
    return sopts;
}

pub fn build_session(sopts: @session::Options,
                     local_crate_source_file: Option<Path>,
                     demitter: @diagnostic::Emitter)
                     -> Session {
    let codemap = @codemap::CodeMap::new();
    let diagnostic_handler =
        diagnostic::mk_handler(Some(demitter));
    let span_diagnostic_handler =
        diagnostic::mk_span_handler(diagnostic_handler, codemap);

    build_session_(sopts, local_crate_source_file, codemap, demitter, span_diagnostic_handler)
}

pub fn build_session_(sopts: @session::Options,
                      local_crate_source_file: Option<Path>,
                      codemap: @codemap::CodeMap,
                      demitter: @diagnostic::Emitter,
                      span_diagnostic_handler: @diagnostic::SpanHandler)
                      -> Session {
    let target_cfg = build_target_config(sopts, demitter);
    let p_s = parse::new_parse_sess_special_handler(span_diagnostic_handler, codemap);
    let cstore = @CStore::new(token::get_ident_interner());
    let filesearch = @filesearch::FileSearch::new(
        &sopts.maybe_sysroot,
        sopts.target_triple,
        sopts.addl_lib_search_paths);

    // Make the path absolute, if necessary
    let local_crate_source_file = local_crate_source_file.map(|path|
        if path.is_absolute() {
            path.clone()
        } else {
            os::getcwd().join(path.clone())
        }
    );

    @Session_ {
        targ_cfg: target_cfg,
        opts: sopts,
        cstore: cstore,
        parse_sess: p_s,
        codemap: codemap,
        // For a library crate, this is always none
        entry_fn: RefCell::new(None),
        entry_type: Cell::new(None),
        macro_registrar_fn: RefCell::new(None),
        span_diagnostic: span_diagnostic_handler,
        filesearch: filesearch,
        building_library: Cell::new(false),
        local_crate_source_file: local_crate_source_file,
        working_dir: os::getcwd(),
        lints: RefCell::new(HashMap::new()),
        node_id: Cell::new(1),
        outputs: @RefCell::new(~[]),
    }
}

pub fn parse_pretty(sess: Session, name: &str) -> PpMode {
    match name {
      &"normal" => PpmNormal,
      &"expanded" => PpmExpanded,
      &"typed" => PpmTyped,
      &"expanded,identified" => PpmExpandedIdentified,
      &"identified" => PpmIdentified,
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
  optflag("c", "",    "Compile and assemble, but do not link"),
  optmulti("", "cfg", "Configure the compilation
                          environment", "SPEC"),
  optflag("",  "emit-llvm",
                        "Produce an LLVM assembly file if used with -S option;
                         produce an LLVM bitcode file otherwise"),
  optflag("h", "help","Display this message"),
  optmulti("L", "",   "Add a directory to the library search path",
                              "PATH"),
  optflag("",  "bin", "Compile an executable crate (default)"),
  optflag("",  "lib", "Compile a rust library crate using the compiler's default"),
  optflag("",  "rlib", "Compile a rust library crate as an rlib file"),
  optflag("",  "staticlib", "Compile a static library crate"),
  optflag("",  "dylib", "Compile a dynamic library crate"),
  optopt("", "linker", "Program to use for linking instead of the default.", "LINKER"),
  optopt("", "ar", "Program to use for managing archives instead of the default.", "AR"),
  optflag("", "crate-id", "Output the crate id and exit"),
  optflag("", "crate-name", "Output the crate name and exit"),
  optflag("", "crate-file-name", "Output the file(s) that would be written if compilation \
          continued and exit"),
  optmulti("",  "link-args", "FLAGS is a space-separated list of flags
                            passed to the linker", "FLAGS"),
  optflag("",  "ls",  "List the symbols defined by a library crate"),
  optflag("", "no-trans",
                        "Run all passes except translation; no output"),
  optflag("", "no-analysis",
                        "Parse and expand the output, but run no analysis or produce \
                        output"),
  optflag("O", "",    "Equivalent to --opt-level=2"),
  optopt("o", "",     "Write output to <filename>", "FILENAME"),
  optopt("", "opt-level",
                        "Optimize with possible levels 0-3", "LEVEL"),
  optopt("", "passes", "Comma or space separated list of pass names to use. \
                        Appends to the default list of passes to run for the \
                        specified current optimization level. A value of \
                        \"list\" will list all of the available passes", "NAMES"),
  optopt("", "llvm-args", "A list of arguments to pass to llvm, comma \
                           separated", "ARGS"),
  optflag("", "no-rpath", "Disables setting the rpath in libs/exes"),
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
  optflagopt("", "dep-info",
                        "Output dependency info to <filename> after compiling", "FILENAME"),
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
  optmulti("W", "warn",
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

pub fn build_output_filenames(input: &Input,
                              odir: &Option<Path>,
                              ofile: &Option<Path>,
                              attrs: &[ast::Attribute],
                              sess: Session)
                           -> ~OutputFilenames {
    let obj_path;
    let out_path;
    let sopts = sess.opts;
    let stop_after_codegen = sopts.output_type != link::OutputTypeExe;

    let obj_suffix = match sopts.output_type {
        link::OutputTypeNone => ~"none",
        link::OutputTypeBitcode => ~"bc",
        link::OutputTypeAssembly => ~"s",
        link::OutputTypeLlvmAssembly => ~"ll",
        // Object and exe output both use the '.o' extension here
        link::OutputTypeObject | link::OutputTypeExe => ~"o"
    };

    match *ofile {
      None => {
          // "-" as input file will cause the parser to read from stdin so we
          // have to make up a name
          // We want to toss everything after the final '.'
          let dirpath = match *odir {
              Some(ref d) => (*d).clone(),
              None => match *input {
                  StrInput(_) => os::getcwd(),
                  FileInput(ref ifile) => (*ifile).dir_path()
              }
          };

          let mut stem = match *input {
              // FIXME (#9639): This needs to handle non-utf8 paths
              FileInput(ref ifile) => {
                  (*ifile).filestem_str().unwrap().to_str()
              }
              StrInput(_) => ~"rust_out"
          };

          // If a crateid is present, we use it as the link name
          let crateid = attr::find_crateid(attrs);
          match crateid {
              None => {}
              Some(crateid) => stem = crateid.name.to_str(),
          }

          if sess.building_library.get() {
              out_path = dirpath.join(os::dll_filename(stem));
              obj_path = {
                  let mut p = dirpath.join(stem);
                  p.set_extension(obj_suffix);
                  p
              };
          } else {
              out_path = dirpath.join(stem);
              obj_path = out_path.with_extension(obj_suffix);
          }
      }

      Some(ref out_file) => {
        out_path = out_file.clone();
        obj_path = if stop_after_codegen {
            out_file.clone()
        } else {
            out_file.with_extension(obj_suffix)
        };

        if sess.building_library.get() {
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

pub fn early_error(emitter: &diagnostic::Emitter, msg: &str) -> ! {
    emitter.emit(None, msg, diagnostic::Fatal);
    fail!(diagnostic::FatalError);
}

pub fn list_metadata(sess: Session, path: &Path,
                     out: &mut io::Writer) -> io::IoResult<()> {
    metadata::loader::list_file_metadata(
        token::get_ident_interner(),
        session::sess_os_to_meta_os(sess.targ_cfg.os), path, out)
}

#[cfg(test)]
mod test {

    use driver::driver::{build_configuration, build_session};
    use driver::driver::{build_session_options, optgroups};

    use extra::getopts::groups::getopts;
    use syntax::attr;
    use syntax::attr::AttrMetaMethods;
    use syntax::diagnostic;

    // When the user supplies --test we should implicitly supply --cfg test
    #[test]
    fn test_switch_implies_cfg_test() {
        let matches =
            &match getopts([~"--test"], optgroups()) {
              Ok(m) => m,
              Err(f) => fail!("test_switch_implies_cfg_test: {}", f.to_err_msg())
            };
        let sessopts = build_session_options(~"rustc", matches, @diagnostic::DefaultEmitter);
        let sess = build_session(sessopts, None, @diagnostic::DefaultEmitter);
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
                fail!("test_switch_implies_cfg_test_unless_cfg_test: {}",
                       f.to_err_msg());
              }
            };
        let sessopts = build_session_options(~"rustc", matches, @diagnostic::DefaultEmitter);
        let sess = build_session(sessopts, None, @diagnostic::DefaultEmitter);
        let cfg = build_configuration(sess);
        let mut test_items = cfg.iter().filter(|m| m.name().equiv(&("test")));
        assert!(test_items.next().is_some());
        assert!(test_items.next().is_none());
    }
}
