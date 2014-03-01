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
use driver::session::{Aggressive, CrateTypeExecutable};
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

use serialize::{json, Encodable};

use std::cell::{Cell, RefCell};
use std::io;
use std::io::fs;
use std::io::MemReader;
use std::os;
use std::vec;
use collections::{HashMap, HashSet};
use getopts::{optopt, optmulti, optflag, optflagopt};
use getopts;
use syntax::ast;
use syntax::abi;
use syntax::attr;
use syntax::attr::{AttrMetaMethods};
use syntax::codemap;
use syntax::diagnostic;
use syntax::diagnostic::Emitter;
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
fn parse_cfgspecs(cfgspecs: ~[~str])
                  -> ast::CrateConfig {
    cfgspecs.move_iter().map(|s| {
        let sess = parse::new_parse_sess();
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
    let krate = time(sess.time_passes(), "parsing", (), |_| {
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
    });

    if sess.opts.debugging_opts & session::AST_JSON_NOEXPAND != 0 {
        let mut stdout = io::stdout();
        let mut json = json::PrettyEncoder::new(&mut stdout);
        krate.encode(&mut json);
    }

    krate
}

// For continuing compilation after a parsed crate has been
// modified

/// Run the "early phases" of the compiler: initial `cfg` processing,
/// syntax expansion, secondary `cfg` expansion, synthesis of a test
/// harness if one is to be provided and injection of a dependency on the
/// standard library and prelude.
pub fn phase_2_configure_and_expand(sess: Session,
                                    loader: &mut CrateLoader,
                                    mut krate: ast::Crate)
                                    -> (ast::Crate, syntax::ast_map::Map) {
    let time_passes = sess.time_passes();

    sess.building_library.set(session::building_library(sess.opts, &krate));
    sess.crate_types.set(session::collect_crate_types(&sess, krate.attrs));

    time(time_passes, "gated feature checking", (), |_|
         front::feature_gate::check_crate(sess, &krate));

    krate = time(time_passes, "crate injection", krate, |krate|
                 front::std_inject::maybe_inject_crates_ref(sess, krate));

    // strip before expansion to allow macros to depend on
    // configuration variables e.g/ in
    //
    //   #[macro_escape] #[cfg(foo)]
    //   mod bar { macro_rules! baz!(() => {{}}) }
    //
    // baz! should not use this definition unless foo is enabled.

    krate = time(time_passes, "configuration 1", krate, |krate|
                 front::config::strip_unconfigured_items(krate));

    krate = time(time_passes, "expansion", krate, |krate| {
        syntax::ext::expand::expand_crate(sess.parse_sess,
                                          loader,
                                          krate)
    });
    // dump the syntax-time crates
    sess.cstore.reset();

    // strip again, in case expansion added anything with a #[cfg].
    krate = time(time_passes, "configuration 2", krate, |krate|
                 front::config::strip_unconfigured_items(krate));

    krate = time(time_passes, "maybe building test harness", krate, |krate|
                 front::test::modify_for_testing(sess, krate));

    krate = time(time_passes, "prelude injection", krate, |krate|
                 front::std_inject::maybe_inject_prelude(sess, krate));

    let (krate, map) = time(time_passes, "assinging node ids and indexing ast", krate, |krate|
         front::assign_node_ids_and_map::assign_node_ids_and_map(sess, krate));

    if sess.opts.debugging_opts & session::AST_JSON != 0 {
        let mut stdout = io::stdout();
        let mut json = json::PrettyEncoder::new(&mut stdout);
        krate.encode(&mut json);
    }

    (krate, map)
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
                                   krate: &ast::Crate,
                                   ast_map: syntax::ast_map::Map) -> CrateAnalysis {

    let time_passes = sess.time_passes();

    time(time_passes, "external crate/lib resolution", (), |_|
         creader::read_crates(sess, krate,
                              session::sess_os_to_meta_os(sess.targ_cfg.os),
                              token::get_ident_interner()));

    let lang_items = time(time_passes, "language item collection", (), |_|
                          middle::lang_items::collect_language_items(krate, sess));

    let middle::resolve::CrateMap {
        def_map: def_map,
        exp_map2: exp_map2,
        trait_map: trait_map,
        external_exports: external_exports,
        last_private_map: last_private_map
    } =
        time(time_passes, "resolution", (), |_|
             middle::resolve::resolve_crate(sess, lang_items, krate));

    let named_region_map = time(time_passes, "lifetime resolution", (),
                                |_| middle::resolve_lifetime::krate(sess, krate));

    time(time_passes, "looking for entry point", (),
         |_| middle::entry::find_entry_point(sess, krate, &ast_map));

    sess.macro_registrar_fn.with_mut(|r| *r =
        time(time_passes, "looking for macro registrar", (), |_|
            syntax::ext::registrar::find_macro_registrar(
                sess.span_diagnostic, krate)));

    let freevars = time(time_passes, "freevar finding", (), |_|
                        freevars::annotate_freevars(def_map, krate));

    let region_map = time(time_passes, "region resolution", (), |_|
                          middle::region::resolve_crate(sess, krate));

    let ty_cx = ty::mk_ctxt(sess, def_map, named_region_map, ast_map,
                            freevars, region_map, lang_items);

    // passes are timed inside typeck
    let (method_map, vtable_map) = typeck::check_crate(ty_cx, trait_map, krate);

    time(time_passes, "check static items", (), |_|
         middle::check_static::check_crate(ty_cx, krate));

    // These next two const passes can probably be merged
    time(time_passes, "const marking", (), |_|
         middle::const_eval::process_crate(krate, ty_cx));

    time(time_passes, "const checking", (), |_|
         middle::check_const::check_crate(sess, krate, def_map,
                                          method_map, ty_cx));

    let maps = (external_exports, last_private_map);
    let (exported_items, public_items) =
            time(time_passes, "privacy checking", maps, |(a, b)|
                 middle::privacy::check_crate(ty_cx, &method_map, &exp_map2,
                                              a, b, krate));

    time(time_passes, "effect checking", (), |_|
         middle::effect::check_crate(ty_cx, method_map, krate));

    time(time_passes, "loop checking", (), |_|
         middle::check_loop::check_crate(ty_cx, krate));

    let middle::moves::MoveMaps {moves_map, moved_variables_set,
                                 capture_map} =
        time(time_passes, "compute moves", (), |_|
             middle::moves::compute_moves(ty_cx, method_map, krate));

    time(time_passes, "match checking", (), |_|
         middle::check_match::check_crate(ty_cx, method_map,
                                          moves_map, krate));

    time(time_passes, "liveness checking", (), |_|
         middle::liveness::check_crate(ty_cx, method_map,
                                       capture_map, krate));

    let root_map =
        time(time_passes, "borrow checking", (), |_|
             middle::borrowck::check_crate(ty_cx, method_map,
                                           moves_map, moved_variables_set,
                                           capture_map, krate));

    time(time_passes, "kind checking", (), |_|
         kind::check_crate(ty_cx, method_map, krate));

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
                                       krate)
        });
    }

    time(time_passes, "lint checking", (), |_|
         lint::check_crate(ty_cx, method_map, &exported_items, krate));

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
                                 krate: ast::Crate,
                                 analysis: &CrateAnalysis,
                                 outputs: &OutputFilenames) -> CrateTranslation {
    time(sess.time_passes(), "translation", krate, |krate|
         trans::base::trans_crate(sess, krate, analysis, outputs))
}

/// Run LLVM itself, producing a bitcode file, assembly file or object file
/// as a side effect.
pub fn phase_5_run_llvm_passes(sess: Session,
                               trans: &CrateTranslation,
                               outputs: &OutputFilenames) {
    if sess.opts.cg.no_integrated_as {
        let output_type = link::OutputTypeAssembly;

        time(sess.time_passes(), "LLVM passes", (), |_|
            link::write::run_passes(sess, trans, [output_type], outputs));

        link::write::run_assembler(sess, outputs);

        // Remove assembly source, unless --save-temps was specified
        if !sess.opts.cg.save_temps {
            fs::unlink(&outputs.temp_path(link::OutputTypeAssembly)).unwrap();
        }
    } else {
        time(sess.time_passes(), "LLVM passes", (), |_|
            link::write::run_passes(sess,
                                    trans,
                                    sess.opts.output_types,
                                    outputs));
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
                           outputs,
                           &trans.link.crateid));
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
    return sess.opts.debugging_opts & session::AST_JSON_NOEXPAND != 0;
}

pub fn stop_after_phase_2(sess: Session) -> bool {
    if sess.opts.no_analysis {
        debug!("invoked with --no-analysis, returning early from compile_input");
        return true;
    }
    return sess.opts.debugging_opts & session::AST_JSON != 0;
}

pub fn stop_after_phase_5(sess: Session) -> bool {
    if !sess.opts.output_types.iter().any(|&i| i == link::OutputTypeExe) {
        debug!("not building executable, returning early from compile_input");
        return true;
    }
    return false;
}

fn write_out_deps(sess: Session,
                  input: &Input,
                  outputs: &OutputFilenames,
                  krate: &ast::Crate) -> io::IoResult<()> {
    let id = link::find_crate_id(krate.attrs, outputs);

    let mut out_filenames = ~[];
    for output_type in sess.opts.output_types.iter() {
        let file = outputs.path(*output_type);
        match *output_type {
            link::OutputTypeExe => {
                let crate_types = sess.crate_types.borrow();
                for output in crate_types.get().iter() {
                    let p = link::filename_for_input(&sess, *output, &id, &file);
                    out_filenames.push(p);
                }
            }
            _ => { out_filenames.push(file); }
        }
    }

    // Write out dependency rules to the dep-info file if requested with
    // --dep-info
    let deps_filename = match sess.opts.write_dependency_info {
        // Use filename from --dep-file argument if given
        (true, Some(ref filename)) => filename.clone(),
        // Use default filename: crate source filename with extension replaced
        // by ".d"
        (true, None) => match *input {
            FileInput(..) => outputs.with_extension("d"),
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
    let mut file = try!(io::File::create(&deps_filename));
    for path in out_filenames.iter() {
        try!(write!(&mut file as &mut Writer,
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
            let krate = phase_1_parse_input(sess, cfg, input);
            if sess.show_span() {
                front::show_span::run(sess, &krate);
                return;
            }
            if stop_after_phase_1(sess) { return; }
            let loader = &mut Loader::new(sess);
            phase_2_configure_and_expand(sess, loader, krate)
        };
        let outputs = build_output_filenames(input, outdir, output,
                                             expanded_crate.attrs, sess);

        write_out_deps(sess, input, &outputs, &expanded_crate).unwrap();

        if stop_after_phase_2(sess) { return; }

        let analysis = phase_3_run_analysis_passes(sess, &expanded_crate, ast_map);
        if stop_after_phase_3(sess) { return; }
        let trans = phase_4_translate_to_llvm(sess, expanded_crate,
                                              &analysis, &outputs);
        (outputs, trans)
    };
    phase_5_run_llvm_passes(sess, &trans, &outputs);
    if stop_after_phase_5(sess) { return; }
    phase_6_link_output(sess, &trans, &outputs);
}

struct IdentifiedAnnotation;

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
                try!(pp::space(&mut s.s));
                try!(pprust::synth_comment(s, item.id.to_str()));
            }
            pprust::NodeBlock(s, blk) => {
                try!(pp::space(&mut s.s));
                try!(pprust::synth_comment(s, ~"block " + blk.id.to_str()));
            }
            pprust::NodeExpr(s, expr) => {
                try!(pp::space(&mut s.s));
                try!(pprust::synth_comment(s, expr.id.to_str()));
                try!(pprust::pclose(s));
            }
            pprust::NodePat(s, pat) => {
                try!(pp::space(&mut s.s));
                try!(pprust::synth_comment(s, ~"pat " + pat.id.to_str()));
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
                try!(pp::space(&mut s.s));
                try!(pp::word(&mut s.s, "as"));
                try!(pp::space(&mut s.s));
                try!(pp::word(&mut s.s,
                                ppaux::ty_to_str(tcx, ty::expr_ty(tcx, expr))));
                try!(pprust::pclose(s));
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
    let krate = phase_1_parse_input(sess, cfg, input);

    let (krate, ast_map, is_expanded) = match ppm {
        PpmExpanded | PpmExpandedIdentified | PpmTyped => {
            let loader = &mut Loader::new(sess);
            let (krate, ast_map) = phase_2_configure_and_expand(sess, loader, krate);
            (krate, Some(ast_map), true)
        }
        _ => (krate, None, false)
    };

    let annotation = match ppm {
        PpmIdentified | PpmExpandedIdentified => {
            ~IdentifiedAnnotation as ~pprust::PpAnn
        }
        PpmTyped => {
            let ast_map = ast_map.expect("--pretty=typed missing ast_map");
            let analysis = phase_3_run_analysis_passes(sess, &krate, ast_map);
            ~TypedAnnotation {
                analysis: analysis
            } as ~pprust::PpAnn:
        }
        _ => ~pprust::NoAnn as ~pprust::PpAnn:,
    };

    let src = &sess.codemap.get_filemap(source_name(input)).src;
    let mut rdr = MemReader::new(src.as_bytes().to_owned());
    let stdout = io::stdout();
    pprust::print_crate(sess.codemap,
                        sess.span_diagnostic,
                        &krate,
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

pub fn build_target_config(sopts: @session::Options)
                           -> @session::Config {
    let os = match get_os(sopts.target_triple) {
      Some(os) => os,
      None => early_error("unknown operating system")
    };
    let arch = match get_arch(sopts.target_triple) {
      Some(arch) => arch,
      None => early_error("unknown architecture: " + sopts.target_triple)
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

pub fn build_session_options(matches: &getopts::Matches)
                             -> @session::Options {
    let crate_types = matches.opt_strs("crate-type").flat_map(|s| {
        s.split(',').map(|part| {
            match part {
                "lib"       => session::default_lib_output(),
                "rlib"      => session::CrateTypeRlib,
                "staticlib" => session::CrateTypeStaticlib,
                "dylib"     => session::CrateTypeDylib,
                "bin"       => session::CrateTypeExecutable,
                _ => early_error(format!("unknown crate type: `{}`", part))
            }
        }).collect()
    });

    let parse_only = matches.opt_present("parse-only");
    let no_trans = matches.opt_present("no-trans");
    let no_analysis = matches.opt_present("no-analysis");

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
                early_error(format!("unknown {} flag: {}",
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
            early_error(format!("unknown debug flag: {}", *debug_flag))
        }
        debugging_opts |= this_bit;
    }

    if debugging_opts & session::DEBUG_LLVM != 0 {
        unsafe { llvm::LLVMSetDebug(1); }
    }

    let mut output_types = if parse_only || no_trans {
        ~[]
    } else {
        matches.opt_strs("emit").flat_map(|s| {
            s.split(',').map(|part| {
                match part.as_slice() {
                    "asm"  => link::OutputTypeAssembly,
                    "ir"   => link::OutputTypeLlvmAssembly,
                    "bc"   => link::OutputTypeBitcode,
                    "obj"  => link::OutputTypeObject,
                    "link" => link::OutputTypeExe,
                    _ => early_error(format!("unknown emission type: `{}`", part))
                }
            }).collect()
        })
    };
    output_types.sort();
    output_types.dedup();
    if output_types.len() == 0 {
        output_types.push(link::OutputTypeExe);
    }

    let sysroot_opt = matches.opt_str("sysroot").map(|m| @Path::new(m));
    let target = matches.opt_str("target").unwrap_or(host_triple());
    let opt_level = {
        if (debugging_opts & session::NO_OPT) != 0 {
            No
        } else if matches.opt_present("O") {
            if matches.opt_present("opt-level") {
                early_error("-O and --opt-level both provided");
            }
            Default
        } else if matches.opt_present("opt-level") {
            match matches.opt_str("opt-level").unwrap() {
              ~"0" => No,
              ~"1" => Less,
              ~"2" => Default,
              ~"3" => Aggressive,
              _ => {
                early_error("optimization level needs to be between 0-3")
              }
            }
        } else { No }
    };
    let gc = debugging_opts & session::GC != 0;
    let debuginfo = matches.opt_present("g") || matches.opt_present("debuginfo");

    let addl_lib_search_paths = matches.opt_strs("L").map(|s| {
        Path::new(s.as_slice())
    }).move_iter().collect();

    let cfg = parse_cfgspecs(matches.opt_strs("cfg"));
    let test = matches.opt_present("test");
    let write_dependency_info = (matches.opt_present("dep-info"),
                                 matches.opt_str("dep-info").map(|p| Path::new(p)));

    let print_metas = (matches.opt_present("crate-id"),
                       matches.opt_present("crate-name"),
                       matches.opt_present("crate-file-name"));
    let cg = build_codegen_options(matches);

    let sopts = @session::Options {
        crate_types: crate_types,
        gc: gc,
        optimize: opt_level,
        debuginfo: debuginfo,
        lint_opts: lint_opts,
        output_types: output_types,
        addl_lib_search_paths: @RefCell::new(addl_lib_search_paths),
        maybe_sysroot: sysroot_opt,
        target_triple: target,
        cfg: cfg,
        test: test,
        parse_only: parse_only,
        no_trans: no_trans,
        no_analysis: no_analysis,
        debugging_opts: debugging_opts,
        write_dependency_info: write_dependency_info,
        print_metas: print_metas,
        cg: cg,
    };
    return sopts;
}

pub fn build_codegen_options(matches: &getopts::Matches)
        -> session::CodegenOptions
{
    let mut cg = session::basic_codegen_options();
    for option in matches.opt_strs("C").move_iter() {
        let mut iter = option.splitn('=', 1);
        let key = iter.next().unwrap();
        let value = iter.next();
        let option_to_lookup = key.replace("-", "_");
        let mut found = false;
        for &(candidate, setter, _) in session::CG_OPTIONS.iter() {
            if option_to_lookup.as_slice() != candidate { continue }
            if !setter(&mut cg, value) {
                match value {
                    Some(..) => early_error(format!("codegen option `{}` takes \
                                                     no value", key)),
                    None => early_error(format!("codegen option `{0}` requires \
                                                 a value (-C {0}=<value>)",
                                                key))
                }
            }
            found = true;
            break;
        }
        if !found {
            early_error(format!("unknown codegen option: `{}`", key));
        }
    }
    return cg;
}

pub fn build_session(sopts: @session::Options,
                     local_crate_source_file: Option<Path>)
                     -> Session {
    let codemap = @codemap::CodeMap::new();
    let diagnostic_handler =
        diagnostic::default_handler();
    let span_diagnostic_handler =
        diagnostic::mk_span_handler(diagnostic_handler, codemap);

    build_session_(sopts, local_crate_source_file, codemap, span_diagnostic_handler)
}

pub fn build_session_(sopts: @session::Options,
                      local_crate_source_file: Option<Path>,
                      codemap: @codemap::CodeMap,
                      span_diagnostic_handler: @diagnostic::SpanHandler)
                      -> Session {
    let target_cfg = build_target_config(sopts);
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
        crate_types: @RefCell::new(~[]),
        features: front::feature_gate::Features::new()
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
pub fn optgroups() -> ~[getopts::OptGroup] {
 ~[
  optflag("h", "help", "Display this message"),
  optmulti("", "cfg", "Configure the compilation environment", "SPEC"),
  optmulti("L", "",   "Add a directory to the library search path", "PATH"),
  optmulti("", "crate-type", "Comma separated list of types of crates for the \
                              compiler to emit",
           "[bin|lib|rlib|dylib|staticlib]"),
  optmulti("", "emit", "Comma separated list of types of output for the compiler
                        to emit",
           "[asm|bc|ir|obj|link]"),
  optflag("", "crate-id", "Output the crate id and exit"),
  optflag("", "crate-name", "Output the crate name and exit"),
  optflag("", "crate-file-name", "Output the file(s) that would be written if compilation \
          continued and exit"),
  optflag("",  "ls",  "List the symbols defined by a library crate"),
  optflag("g",  "debuginfo",  "Emit DWARF debug info to the objects created"),
  optflag("", "no-trans",
                        "Run all passes except translation; no output"),
  optflag("", "no-analysis",
                        "Parse and expand the output, but run no analysis or produce \
                        output"),
  optflag("O", "",    "Equivalent to --opt-level=2"),
  optopt("o", "",     "Write output to <filename>", "FILENAME"),
  optopt("", "opt-level",
                        "Optimize with possible levels 0-3", "LEVEL"),
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
  optflagopt("", "dep-info",
                        "Output dependency info to <filename> after compiling", "FILENAME"),
  optopt("", "sysroot",
                        "Override the system root", "PATH"),
  optflag("", "test", "Build a test harness"),
  optopt("", "target",
                        "Target triple cpu-manufacturer-kernel[-os]
                          to compile for (see chapter 3.4 of http://www.sourceware.org/autobook/
                          for details)", "TRIPLE"),
  optmulti("W", "warn",
                        "Set lint warnings", "OPT"),
  optmulti("A", "allow",
                        "Set lint allowed", "OPT"),
  optmulti("D", "deny",
                        "Set lint denied", "OPT"),
  optmulti("F", "forbid",
                        "Set lint forbidden", "OPT"),
  optmulti("C", "codegen",
                        "Set a codegen option", "OPT[=VALUE]"),
  optmulti("Z", "",   "Set internal debugging options", "FLAG"),
  optflag( "v", "version",
                        "Print version info and exit"),
 ]
}

pub struct OutputFilenames {
    out_directory: Path,
    out_filestem: ~str,
    single_output_file: Option<Path>,
}

impl OutputFilenames {
    pub fn path(&self, flavor: link::OutputType) -> Path {
        match self.single_output_file {
            Some(ref path) => return path.clone(),
            None => {}
        }
        self.temp_path(flavor)
    }

    pub fn temp_path(&self, flavor: link::OutputType) -> Path {
        let base = self.out_directory.join(self.out_filestem.as_slice());
        match flavor {
            link::OutputTypeBitcode => base.with_extension("bc"),
            link::OutputTypeAssembly => base.with_extension("s"),
            link::OutputTypeLlvmAssembly => base.with_extension("ll"),
            link::OutputTypeObject => base.with_extension("o"),
            link::OutputTypeExe => base,
        }
    }

    pub fn with_extension(&self, extension: &str) -> Path {
        let stem = self.out_filestem.as_slice();
        self.out_directory.join(stem).with_extension(extension)
    }
}

pub fn build_output_filenames(input: &Input,
                              odir: &Option<Path>,
                              ofile: &Option<Path>,
                              attrs: &[ast::Attribute],
                              sess: Session)
                           -> OutputFilenames {
    match *ofile {
        None => {
            // "-" as input file will cause the parser to read from stdin so we
            // have to make up a name
            // We want to toss everything after the final '.'
            let dirpath = match *odir {
                Some(ref d) => d.clone(),
                None => Path::new(".")
            };

            let mut stem = match *input {
                // FIXME (#9639): This needs to handle non-utf8 paths
                FileInput(ref ifile) => ifile.filestem_str().unwrap().to_str(),
                StrInput(_) => ~"rust_out"
            };

            // If a crateid is present, we use it as the link name
            let crateid = attr::find_crateid(attrs);
            match crateid {
                None => {}
                Some(crateid) => stem = crateid.name.to_str(),
            }
            OutputFilenames {
                out_directory: dirpath,
                out_filestem: stem,
                single_output_file: None,
            }
        }

        Some(ref out_file) => {
            let ofile = if sess.opts.output_types.len() > 1 {
                sess.warn("ignoring specified output filename because multiple \
                           outputs were requested");
                None
            } else {
                Some(out_file.clone())
            };
            if *odir != None {
                sess.warn("ignoring --out-dir flag due to -o flag.");
            }
            OutputFilenames {
                out_directory: out_file.dir_path(),
                out_filestem: out_file.filestem_str().unwrap().to_str(),
                single_output_file: ofile,
            }
        }
    }
}

pub fn early_error(msg: &str) -> ! {
    let mut emitter = diagnostic::EmitterWriter::stderr();
    emitter.emit(None, msg, diagnostic::Fatal);
    fail!(diagnostic::FatalError);
}

pub fn list_metadata(sess: Session, path: &Path,
                     out: &mut io::Writer) -> io::IoResult<()> {
    metadata::loader::list_file_metadata(
        session::sess_os_to_meta_os(sess.targ_cfg.os), path, out)
}

#[cfg(test)]
mod test {

    use driver::driver::{build_configuration, build_session};
    use driver::driver::{build_session_options, optgroups};

    use getopts::getopts;
    use syntax::attr;
    use syntax::attr::AttrMetaMethods;

    // When the user supplies --test we should implicitly supply --cfg test
    #[test]
    fn test_switch_implies_cfg_test() {
        let matches =
            &match getopts([~"--test"], optgroups()) {
              Ok(m) => m,
              Err(f) => fail!("test_switch_implies_cfg_test: {}", f.to_err_msg())
            };
        let sessopts = build_session_options(matches);
        let sess = build_session(sessopts, None);
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
        let sessopts = build_session_options(matches);
        let sess = build_session(sessopts, None);
        let cfg = build_configuration(sess);
        let mut test_items = cfg.iter().filter(|m| m.name().equiv(&("test")));
        assert!(test_items.next().is_some());
        assert!(test_items.next().is_none());
    }
}
