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
use driver::session::Session;
use driver::{config, PpMode};
use driver::{PpmFlowGraph, PpmExpanded, PpmExpandedIdentified, PpmTyped};
use driver::{PpmIdentified};
use front;
use lib::llvm::{ContextRef, ModuleRef};
use metadata::common::LinkMeta;
use metadata::creader;
use middle::cfg;
use middle::cfg::graphviz::LabelledCFG;
use middle::{trans, freevars, kind, ty, typeck, lint, reachable};
use middle::dependency_format;
use middle;
use plugin::load::Plugins;
use plugin::registry::Registry;
use plugin;
use util::common::time;
use util::ppaux;
use util::nodemap::{NodeSet};

use dot = graphviz;

use serialize::{json, Encodable};

use std::io;
use std::io::fs;
use std::io::MemReader;
use syntax::ast;
use syntax::attr;
use syntax::attr::{AttrMetaMethods};
use syntax::crateid::CrateId;
use syntax::parse;
use syntax::parse::token;
use syntax::print::{pp, pprust};
use syntax;

pub fn host_triple() -> &'static str {
    // Get the host triple out of the build environment. This ensures that our
    // idea of the host triple is the same as for the set of libraries we've
    // actually built.  We can't just take LLVM's host triple because they
    // normalize all ix86 architectures to i386.
    //
    // Instead of grabbing the host triple (for the current host), we grab (at
    // compile time) the target triple that this rustc is built with and
    // calling that (at runtime) the host triple.
    (option_env!("CFG_COMPILER_HOST_TRIPLE")).
        expect("CFG_COMPILER_HOST_TRIPLE")
}

pub fn compile_input(sess: Session,
                     cfg: ast::CrateConfig,
                     input: &Input,
                     outdir: &Option<Path>,
                     output: &Option<Path>) {
    // We need nested scopes here, because the intermediate results can keep
    // large chunks of memory alive and we want to free them as soon as
    // possible to keep the peak memory usage low
    let (outputs, trans, sess) = {
        let (outputs, expanded_crate, ast_map) = {
            let krate = phase_1_parse_input(&sess, cfg, input);
            if stop_after_phase_1(&sess) { return; }
            let outputs = build_output_filenames(input,
                                                 outdir,
                                                 output,
                                                 krate.attrs.as_slice(),
                                                 &sess);
            let id = link::find_crate_id(krate.attrs.as_slice(),
                                         outputs.out_filestem.as_slice());
            let (expanded_crate, ast_map) =
                phase_2_configure_and_expand(&sess, krate, &id);
            (outputs, expanded_crate, ast_map)
        };
        write_out_deps(&sess, input, &outputs, &expanded_crate);

        if stop_after_phase_2(&sess) { return; }

        let analysis = phase_3_run_analysis_passes(sess, &expanded_crate, ast_map);
        phase_save_analysis(&analysis.ty_cx.sess, &expanded_crate, &analysis, outdir);
        if stop_after_phase_3(&analysis.ty_cx.sess) { return; }
        let (tcx, trans) = phase_4_translate_to_llvm(expanded_crate,
                                                     analysis, &outputs);

        // Discard interned strings as they are no longer required.
        token::get_ident_interner().clear();

        (outputs, trans, tcx.sess)
    };
    phase_5_run_llvm_passes(&sess, &trans, &outputs);
    if stop_after_phase_5(&sess) { return; }
    phase_6_link_output(&sess, &trans, &outputs);
}

/**
 * The name used for source code that doesn't originate in a file
 * (e.g. source from stdin or a string)
 */
pub fn anon_src() -> String {
    "<anon>".to_string()
}

pub fn source_name(input: &Input) -> String {
    match *input {
        // FIXME (#9639): This needs to handle non-utf8 paths
        FileInput(ref ifile) => ifile.as_str().unwrap().to_string(),
        StrInput(_) => anon_src()
    }
}

pub enum Input {
    /// Load source from file
    FileInput(Path),
    /// The string is the source
    StrInput(String)
}

impl Input {
    fn filestem(&self) -> String {
        match *self {
            FileInput(ref ifile) => ifile.filestem_str().unwrap().to_string(),
            StrInput(_) => "rust_out".to_string(),
        }
    }
}


pub fn phase_1_parse_input(sess: &Session, cfg: ast::CrateConfig, input: &Input)
    -> ast::Crate {
    let krate = time(sess.time_passes(), "parsing", (), |_| {
        match *input {
            FileInput(ref file) => {
                parse::parse_crate_from_file(&(*file), cfg.clone(), &sess.parse_sess)
            }
            StrInput(ref src) => {
                parse::parse_crate_from_source_str(anon_src().to_string(),
                                                   src.to_string(),
                                                   cfg.clone(),
                                                   &sess.parse_sess)
            }
        }
    });

    if sess.opts.debugging_opts & config::AST_JSON_NOEXPAND != 0 {
        let mut stdout = io::BufferedWriter::new(io::stdout());
        let mut json = json::PrettyEncoder::new(&mut stdout);
        // unwrapping so IoError isn't ignored
        krate.encode(&mut json).unwrap();
    }

    if sess.show_span() {
        front::show_span::run(sess, &krate);
    }

    krate
}

// For continuing compilation after a parsed crate has been
// modified

/// Run the "early phases" of the compiler: initial `cfg` processing,
/// syntax expansion, secondary `cfg` expansion, synthesis of a test
/// harness if one is to be provided and injection of a dependency on the
/// standard library and prelude.
pub fn phase_2_configure_and_expand(sess: &Session,
                                    mut krate: ast::Crate,
                                    crate_id: &CrateId)
                                    -> (ast::Crate, syntax::ast_map::Map) {
    let time_passes = sess.time_passes();

    *sess.crate_types.borrow_mut() = collect_crate_types(sess, krate.attrs.as_slice());

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

    let Plugins { macros, registrars }
        = time(time_passes, "plugin loading", (), |_|
               plugin::load::load_plugins(sess, &krate));

    let mut registry = Registry::new(&krate);

    time(time_passes, "plugin registration", (), |_| {
        for &registrar in registrars.iter() {
            registrar(&mut registry);
        }
    });

    let Registry { syntax_exts, .. } = registry;

    krate = time(time_passes, "expansion", (krate, macros, syntax_exts),
        |(krate, macros, syntax_exts)| {
            // Windows dlls do not have rpaths, so they don't know how to find their
            // dependencies. It's up to us to tell the system where to find all the
            // dependent dlls. Note that this uses cfg!(windows) as opposed to
            // targ_cfg because syntax extensions are always loaded for the host
            // compiler, not for the target.
            if cfg!(windows) {
                sess.host_filesearch().add_dylib_search_paths();
            }
            let cfg = syntax::ext::expand::ExpansionConfig {
                deriving_hash_type_parameter: sess.features.default_type_params.get(),
                crate_id: crate_id.clone(),
            };
            syntax::ext::expand::expand_crate(&sess.parse_sess,
                                              cfg,
                                              macros,
                                              syntax_exts,
                                              krate)
        }
    );

    // strip again, in case expansion added anything with a #[cfg].
    krate = time(time_passes, "configuration 2", krate, |krate|
                 front::config::strip_unconfigured_items(krate));

    krate = time(time_passes, "maybe building test harness", krate, |krate|
                 front::test::modify_for_testing(sess, krate));

    krate = time(time_passes, "prelude injection", krate, |krate|
                 front::std_inject::maybe_inject_prelude(sess, krate));

    let (krate, map) = time(time_passes, "assigning node ids and indexing ast", krate, |krate|
         front::assign_node_ids_and_map::assign_node_ids_and_map(sess, krate));

    if sess.opts.debugging_opts & config::AST_JSON != 0 {
        let mut stdout = io::BufferedWriter::new(io::stdout());
        let mut json = json::PrettyEncoder::new(&mut stdout);
        // unwrapping so IoError isn't ignored
        krate.encode(&mut json).unwrap();
    }

    (krate, map)
}

pub struct CrateAnalysis {
    pub exp_map2: middle::resolve::ExportMap2,
    pub exported_items: middle::privacy::ExportedItems,
    pub public_items: middle::privacy::PublicItems,
    pub ty_cx: ty::ctxt,
    pub reachable: NodeSet,
}

/// Run the resolution, typechecking, region checking and other
/// miscellaneous analysis passes on the crate. Return various
/// structures carrying the results of the analysis.
pub fn phase_3_run_analysis_passes(sess: Session,
                                   krate: &ast::Crate,
                                   ast_map: syntax::ast_map::Map) -> CrateAnalysis {

    let time_passes = sess.time_passes();

    time(time_passes, "external crate/lib resolution", (), |_|
         creader::read_crates(&sess, krate));

    let lang_items = time(time_passes, "language item collection", (), |_|
                          middle::lang_items::collect_language_items(krate, &sess));

    let middle::resolve::CrateMap {
        def_map: def_map,
        exp_map2: exp_map2,
        trait_map: trait_map,
        external_exports: external_exports,
        last_private_map: last_private_map
    } =
        time(time_passes, "resolution", (), |_|
             middle::resolve::resolve_crate(&sess, &lang_items, krate));

    // Discard MTWT tables that aren't required past resolution.
    syntax::ext::mtwt::clear_tables();

    let named_region_map = time(time_passes, "lifetime resolution", (),
                                |_| middle::resolve_lifetime::krate(&sess, krate));

    time(time_passes, "looking for entry point", (),
         |_| middle::entry::find_entry_point(&sess, krate, &ast_map));

    sess.plugin_registrar_fn.set(
        time(time_passes, "looking for plugin registrar", (), |_|
            plugin::build::find_plugin_registrar(
                sess.diagnostic(), krate)));

    let freevars = time(time_passes, "freevar finding", (), |_|
                        freevars::annotate_freevars(&def_map, krate));

    let region_map = time(time_passes, "region resolution", (), |_|
                          middle::region::resolve_crate(&sess, krate));

    time(time_passes, "loop checking", (), |_|
         middle::check_loop::check_crate(&sess, krate));

    let ty_cx = ty::mk_ctxt(sess, def_map, named_region_map, ast_map,
                            freevars, region_map, lang_items);

    // passes are timed inside typeck
    typeck::check_crate(&ty_cx, trait_map, krate);

    time(time_passes, "check static items", (), |_|
         middle::check_static::check_crate(&ty_cx, krate));

    // These next two const passes can probably be merged
    time(time_passes, "const marking", (), |_|
         middle::const_eval::process_crate(krate, &ty_cx));

    time(time_passes, "const checking", (), |_|
         middle::check_const::check_crate(krate, &ty_cx));

    let maps = (external_exports, last_private_map);
    let (exported_items, public_items) =
            time(time_passes, "privacy checking", maps, |(a, b)|
                 middle::privacy::check_crate(&ty_cx, &exp_map2, a, b, krate));

    time(time_passes, "effect checking", (), |_|
         middle::effect::check_crate(&ty_cx, krate));

    time(time_passes, "match checking", (), |_|
         middle::check_match::check_crate(&ty_cx, krate));

    time(time_passes, "liveness checking", (), |_|
         middle::liveness::check_crate(&ty_cx, krate));

    time(time_passes, "borrow checking", (), |_|
         middle::borrowck::check_crate(&ty_cx, krate));

    time(time_passes, "kind checking", (), |_|
         kind::check_crate(&ty_cx, krate));

    let reachable_map =
        time(time_passes, "reachability checking", (), |_|
             reachable::find_reachable(&ty_cx, &exported_items));

    time(time_passes, "death checking", (), |_| {
        middle::dead::check_crate(&ty_cx,
                                  &exported_items,
                                  &reachable_map,
                                  krate)
    });

    time(time_passes, "lint checking", (), |_|
         lint::check_crate(&ty_cx, &exported_items, krate));

    CrateAnalysis {
        exp_map2: exp_map2,
        ty_cx: ty_cx,
        exported_items: exported_items,
        public_items: public_items,
        reachable: reachable_map,
    }
}

pub fn phase_save_analysis(sess: &Session,
                           krate: &ast::Crate,
                           analysis: &CrateAnalysis,
                           odir: &Option<Path>) {
    if (sess.opts.debugging_opts & config::SAVE_ANALYSIS) == 0 {
        return;
    }
    time(sess.time_passes(), "save analysis", krate, |krate|
         middle::save::process_crate(sess, krate, analysis, odir));
}

pub struct CrateTranslation {
    pub context: ContextRef,
    pub module: ModuleRef,
    pub metadata_module: ModuleRef,
    pub link: LinkMeta,
    pub metadata: Vec<u8>,
    pub reachable: Vec<String>,
    pub crate_formats: dependency_format::Dependencies,
    pub no_builtins: bool,
}

/// Run the translation phase to LLVM, after which the AST and analysis can
/// be discarded.
pub fn phase_4_translate_to_llvm(krate: ast::Crate,
                                 analysis: CrateAnalysis,
                                 outputs: &OutputFilenames) -> (ty::ctxt, CrateTranslation) {
    let time_passes = analysis.ty_cx.sess.time_passes();

    time(time_passes, "resolving dependency formats", (), |_|
         dependency_format::calculate(&analysis.ty_cx));

    // Option dance to work around the lack of stack once closures.
    time(time_passes, "translation", (krate, analysis), |(krate, analysis)|
         trans::base::trans_crate(krate, analysis, outputs))
}

/// Run LLVM itself, producing a bitcode file, assembly file or object file
/// as a side effect.
pub fn phase_5_run_llvm_passes(sess: &Session,
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
                                    sess.opts.output_types.as_slice(),
                                    outputs));
    }
}

/// Run the linker on any artifacts that resulted from the LLVM run.
/// This should produce either a finished executable or library.
pub fn phase_6_link_output(sess: &Session,
                           trans: &CrateTranslation,
                           outputs: &OutputFilenames) {
    time(sess.time_passes(), "linking", (), |_|
         link::link_binary(sess,
                           trans,
                           outputs,
                           &trans.link.crateid));
}

pub fn stop_after_phase_3(sess: &Session) -> bool {
   if sess.opts.no_trans {
        debug!("invoked with --no-trans, returning early from compile_input");
        return true;
    }
    return false;
}

pub fn stop_after_phase_1(sess: &Session) -> bool {
    if sess.opts.parse_only {
        debug!("invoked with --parse-only, returning early from compile_input");
        return true;
    }
    if sess.show_span() {
        return true;
    }
    return sess.opts.debugging_opts & config::AST_JSON_NOEXPAND != 0;
}

pub fn stop_after_phase_2(sess: &Session) -> bool {
    if sess.opts.no_analysis {
        debug!("invoked with --no-analysis, returning early from compile_input");
        return true;
    }
    return sess.opts.debugging_opts & config::AST_JSON != 0;
}

pub fn stop_after_phase_5(sess: &Session) -> bool {
    if !sess.opts.output_types.iter().any(|&i| i == link::OutputTypeExe) {
        debug!("not building executable, returning early from compile_input");
        return true;
    }
    return false;
}

fn write_out_deps(sess: &Session,
                  input: &Input,
                  outputs: &OutputFilenames,
                  krate: &ast::Crate) {
    let id = link::find_crate_id(krate.attrs.as_slice(),
                                 outputs.out_filestem.as_slice());

    let mut out_filenames = Vec::new();
    for output_type in sess.opts.output_types.iter() {
        let file = outputs.path(*output_type);
        match *output_type {
            link::OutputTypeExe => {
                for output in sess.crate_types.borrow().iter() {
                    let p = link::filename_for_input(sess, *output, &id, &file);
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
                return
            },
        },
        _ => return,
    };

    let result = (|| {
        // Build a list of files used to compile the output and
        // write Makefile-compatible dependency rules
        let files: Vec<String> = sess.codemap().files.borrow()
                                   .iter().filter(|fmap| fmap.is_real_file())
                                   .map(|fmap| fmap.name.to_string())
                                   .collect();
        let mut file = try!(io::File::create(&deps_filename));
        for path in out_filenames.iter() {
            try!(write!(&mut file as &mut Writer,
                          "{}: {}\n\n", path.display(), files.connect(" ")));
        }
        Ok(())
    })();

    match result {
        Ok(()) => {}
        Err(e) => {
            sess.fatal(format!("error writing dependencies to `{}`: {}",
                               deps_filename.display(), e).as_slice());
        }
    }
}

struct IdentifiedAnnotation;

impl pprust::PpAnn for IdentifiedAnnotation {
    fn pre(&self,
           s: &mut pprust::State,
           node: pprust::AnnNode) -> io::IoResult<()> {
        match node {
            pprust::NodeExpr(_) => s.popen(),
            _ => Ok(())
        }
    }
    fn post(&self,
            s: &mut pprust::State,
            node: pprust::AnnNode) -> io::IoResult<()> {
        match node {
            pprust::NodeItem(item) => {
                try!(pp::space(&mut s.s));
                s.synth_comment(item.id.to_str())
            }
            pprust::NodeBlock(blk) => {
                try!(pp::space(&mut s.s));
                s.synth_comment((format!("block {}", blk.id)).to_string())
            }
            pprust::NodeExpr(expr) => {
                try!(pp::space(&mut s.s));
                try!(s.synth_comment(expr.id.to_str()));
                s.pclose()
            }
            pprust::NodePat(pat) => {
                try!(pp::space(&mut s.s));
                s.synth_comment((format!("pat {}", pat.id)).to_string())
            }
        }
    }
}

struct TypedAnnotation {
    analysis: CrateAnalysis,
}

impl pprust::PpAnn for TypedAnnotation {
    fn pre(&self,
           s: &mut pprust::State,
           node: pprust::AnnNode) -> io::IoResult<()> {
        match node {
            pprust::NodeExpr(_) => s.popen(),
            _ => Ok(())
        }
    }
    fn post(&self,
            s: &mut pprust::State,
            node: pprust::AnnNode) -> io::IoResult<()> {
        let tcx = &self.analysis.ty_cx;
        match node {
            pprust::NodeExpr(expr) => {
                try!(pp::space(&mut s.s));
                try!(pp::word(&mut s.s, "as"));
                try!(pp::space(&mut s.s));
                try!(pp::word(&mut s.s,
                              ppaux::ty_to_str(
                                  tcx,
                                  ty::expr_ty(tcx, expr)).as_slice()));
                s.pclose()
            }
            _ => Ok(())
        }
    }
}

pub fn pretty_print_input(sess: Session,
                          cfg: ast::CrateConfig,
                          input: &Input,
                          ppm: PpMode,
                          ofile: Option<Path>) {
    let krate = phase_1_parse_input(&sess, cfg, input);
    let id = link::find_crate_id(krate.attrs.as_slice(),
                                 input.filestem().as_slice());

    let (krate, ast_map, is_expanded) = match ppm {
        PpmExpanded | PpmExpandedIdentified | PpmTyped | PpmFlowGraph(_) => {
            let (krate, ast_map) = phase_2_configure_and_expand(&sess,
                                                                krate,
                                                                &id);
            (krate, Some(ast_map), true)
        }
        _ => (krate, None, false)
    };

    let src_name = source_name(input);
    let src = Vec::from_slice(sess.codemap()
                                  .get_filemap(src_name.as_slice())
                                  .src
                                  .as_bytes());
    let mut rdr = MemReader::new(src);

    let out = match ofile {
        None => box io::stdout() as Box<Writer>,
        Some(p) => {
            let r = io::File::create(&p);
            match r {
                Ok(w) => box w as Box<Writer>,
                Err(e) => fail!("print-print failed to open {} due to {}",
                                p.display(), e),
            }
        }
    };
    match ppm {
        PpmIdentified | PpmExpandedIdentified => {
            pprust::print_crate(sess.codemap(),
                                sess.diagnostic(),
                                &krate,
                                src_name.to_string(),
                                &mut rdr,
                                out,
                                &IdentifiedAnnotation,
                                is_expanded)
        }
        PpmTyped => {
            let ast_map = ast_map.expect("--pretty=typed missing ast_map");
            let analysis = phase_3_run_analysis_passes(sess, &krate, ast_map);
            let annotation = TypedAnnotation {
                analysis: analysis
            };
            pprust::print_crate(annotation.analysis.ty_cx.sess.codemap(),
                                annotation.analysis.ty_cx.sess.diagnostic(),
                                &krate,
                                src_name.to_string(),
                                &mut rdr,
                                out,
                                &annotation,
                                is_expanded)
        }
        PpmFlowGraph(nodeid) => {
            let ast_map = ast_map.expect("--pretty flowgraph missing ast_map");
            let node = ast_map.find(nodeid).unwrap_or_else(|| {
                sess.fatal(format!("--pretty flowgraph couldn't find id: {}",
                                   nodeid).as_slice())
            });
            let block = match node {
                syntax::ast_map::NodeBlock(block) => block,
                _ => {
                    let message = format!("--pretty=flowgraph needs block, got {:?}",
                                          node);

                    // point to what was found, if there's an
                    // accessible span.
                    match ast_map.opt_span(nodeid) {
                        Some(sp) => sess.span_fatal(sp, message.as_slice()),
                        None => sess.fatal(message.as_slice())
                    }
                }
            };
            let analysis = phase_3_run_analysis_passes(sess, &krate, ast_map);
            print_flowgraph(analysis, block, out)
        }
        _ => {
            pprust::print_crate(sess.codemap(),
                                sess.diagnostic(),
                                &krate,
                                src_name.to_string(),
                                &mut rdr,
                                out,
                                &pprust::NoAnn,
                                is_expanded)
        }
    }.unwrap()

}

fn print_flowgraph<W:io::Writer>(analysis: CrateAnalysis,
                                 block: ast::P<ast::Block>,
                                 mut out: W) -> io::IoResult<()> {
    let ty_cx = &analysis.ty_cx;
    let cfg = cfg::CFG::new(ty_cx, &*block);
    let lcfg = LabelledCFG { ast_map: &ty_cx.map,
                             cfg: &cfg,
                             name: format!("block{}", block.id).to_string(), };
    debug!("cfg: {:?}", cfg);
    let r = dot::render(&lcfg, &mut out);
    return expand_err_details(r);

    fn expand_err_details(r: io::IoResult<()>) -> io::IoResult<()> {
        r.map_err(|ioerr| {
            let orig_detail = ioerr.detail.clone();
            let m = "graphviz::render failed";
            io::IoError {
                detail: Some(match orig_detail {
                    None => m.into_string(),
                    Some(d) => format!("{}: {}", m, d)
                }),
                ..ioerr
            }
        })
    }
}

pub fn collect_crate_types(session: &Session,
                           attrs: &[ast::Attribute]) -> Vec<config::CrateType> {
    // Unconditionally collect crate types from attributes to make them used
    let attr_types: Vec<config::CrateType> = attrs.iter().filter_map(|a| {
        if a.check_name("crate_type") {
            match a.value_str() {
                Some(ref n) if n.equiv(&("rlib")) => {
                    Some(config::CrateTypeRlib)
                }
                Some(ref n) if n.equiv(&("dylib")) => {
                    Some(config::CrateTypeDylib)
                }
                Some(ref n) if n.equiv(&("lib")) => {
                    Some(config::default_lib_output())
                }
                Some(ref n) if n.equiv(&("staticlib")) => {
                    Some(config::CrateTypeStaticlib)
                }
                Some(ref n) if n.equiv(&("bin")) => Some(config::CrateTypeExecutable),
                Some(_) => {
                    session.add_lint(lint::UnknownCrateType,
                                     ast::CRATE_NODE_ID,
                                     a.span,
                                     "invalid `crate_type` \
                                      value".to_string());
                    None
                }
                _ => {
                    session.add_lint(lint::UnknownCrateType,
                                     ast::CRATE_NODE_ID,
                                     a.span,
                                     "`crate_type` requires a \
                                      value".to_string());
                    None
                }
            }
        } else {
            None
        }
    }).collect();

    // If we're generating a test executable, then ignore all other output
    // styles at all other locations
    if session.opts.test {
        return vec!(config::CrateTypeExecutable)
    }

    // Only check command line flags if present. If no types are specified by
    // command line, then reuse the empty `base` Vec to hold the types that
    // will be found in crate attributes.
    let mut base = session.opts.crate_types.clone();
    if base.len() > 0 {
        return base
    } else {
        base.extend(attr_types.move_iter());
        if base.len() == 0 {
            base.push(config::CrateTypeExecutable);
        }
        base.as_mut_slice().sort();
        base.dedup();
        return base;
    }
}

pub struct OutputFilenames {
    pub out_directory: Path,
    pub out_filestem: String,
    pub single_output_file: Option<Path>,
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
                              sess: &Session)
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

            let mut stem = input.filestem();

            // If a crateid is present, we use it as the link name
            let crateid = attr::find_crateid(attrs);
            match crateid {
                None => {}
                Some(crateid) => stem = crateid.name.to_string(),
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
                out_filestem: out_file.filestem_str().unwrap().to_string(),
                single_output_file: ofile,
            }
        }
    }
}
