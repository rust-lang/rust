// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg_attr(not(feature="llvm"), allow(dead_code))]

use rustc::dep_graph::DepGraph;
use rustc::hir::{self, map as hir_map};
use rustc::hir::lowering::lower_crate;
use rustc::ich::Fingerprint;
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_mir as mir;
use rustc::session::{Session, CompileResult};
use rustc::session::CompileIncomplete;
use rustc::session::config::{self, Input, OutputFilenames, OutputType};
use rustc::session::search_paths::PathKind;
use rustc::lint;
use rustc::middle::{self, stability, reachable};
use rustc::middle::cstore::CrateStore;
use rustc::middle::privacy::AccessLevels;
use rustc::mir::transform::{MIR_CONST, MIR_VALIDATED, MIR_OPTIMIZED, Passes};
use rustc::ty::{self, TyCtxt, Resolutions, GlobalArenas};
use rustc::traits;
use rustc::util::common::{ErrorReported, time};
use rustc_allocator as allocator;
use rustc_borrowck as borrowck;
use rustc_incremental::{self, IncrementalHashesMap};
use rustc_resolve::{MakeGlobMap, Resolver};
use rustc_metadata::creader::CrateLoader;
use rustc_metadata::cstore::{self, CStore};
use rustc_trans::back::write;
use rustc_trans as trans;
use rustc_typeck as typeck;
use rustc_privacy;
use rustc_plugin::registry::Registry;
use rustc_plugin as plugin;
use rustc_passes::{ast_validation, no_asm, loops, consts, static_recursion, hir_stats};
use rustc_const_eval::{self, check_match};
use super::Compilation;

use serialize::json;

use std::any::Any;
use std::env;
use std::ffi::{OsString, OsStr};
use std::fs;
use std::io::{self, Write};
use std::iter;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::mpsc;
use syntax::{ast, diagnostics, visit};
use syntax::attr;
use syntax::ext::base::ExtCtxt;
use syntax::parse::{self, PResult};
use syntax::symbol::Symbol;
use syntax::util::node_count::NodeCounter;
use syntax;
use syntax_ext;
use arena::DroplessArena;

use derive_registrar;

use profile;

pub fn compile_input(sess: &Session,
                     cstore: &CStore,
                     input: &Input,
                     outdir: &Option<PathBuf>,
                     output: &Option<PathBuf>,
                     addl_plugins: Option<Vec<String>>,
                     control: &CompileController) -> CompileResult {
    use rustc_trans::back::write::OngoingCrateTranslation;
    macro_rules! controller_entry_point {
        ($point: ident, $tsess: expr, $make_state: expr, $phase_result: expr) => {{
            let state = &mut $make_state;
            let phase_result: &CompileResult = &$phase_result;
            if phase_result.is_ok() || control.$point.run_callback_on_error {
                (control.$point.callback)(state);
            }

            if control.$point.stop == Compilation::Stop {
                // FIXME: shouldn't this return Err(CompileIncomplete::Stopped)
                // if there are no errors?
                return $tsess.compile_status();
            }
        }}
    }

    if cfg!(not(feature="llvm")) {
        use rustc::session::config::CrateType;
        if !sess.opts.debugging_opts.no_trans && sess.opts.output_types.should_trans() {
            sess.err("LLVM is not supported by this rustc. Please use -Z no-trans to compile")
        }

        if sess.opts.crate_types.iter().all(|&t|{
            t != CrateType::CrateTypeRlib && t != CrateType::CrateTypeExecutable
        }) && !sess.opts.crate_types.is_empty() {
            sess.err(
                "LLVM is not supported by this rustc, so non rlib libraries are not supported"
            );
        }

        sess.abort_if_errors();
    }

    if sess.profile_queries() {
        profile::begin();
    }

    // We need nested scopes here, because the intermediate results can keep
    // large chunks of memory alive and we want to free them as soon as
    // possible to keep the peak memory usage low
    let (outputs, trans, dep_graph): (OutputFilenames, OngoingCrateTranslation, DepGraph) = {
        let krate = match phase_1_parse_input(control, sess, input) {
            Ok(krate) => krate,
            Err(mut parse_error) => {
                parse_error.emit();
                return Err(CompileIncomplete::Errored(ErrorReported));
            }
        };

        let (krate, registry) = {
            let mut compile_state = CompileState::state_after_parse(input,
                                                                    sess,
                                                                    outdir,
                                                                    output,
                                                                    krate,
                                                                    &cstore);
            controller_entry_point!(after_parse,
                                    sess,
                                    compile_state,
                                    Ok(()));

            (compile_state.krate.unwrap(), compile_state.registry)
        };

        let outputs = build_output_filenames(input, outdir, output, &krate.attrs, sess);
        let crate_name =
            ::rustc_trans_utils::link::find_crate_name(Some(sess), &krate.attrs, input);
        let ExpansionResult { expanded_crate, defs, analysis, resolutions, mut hir_forest } = {
            phase_2_configure_and_expand(
                sess,
                &cstore,
                krate,
                registry,
                &crate_name,
                addl_plugins,
                control.make_glob_map,
                |expanded_crate| {
                    let mut state = CompileState::state_after_expand(
                        input, sess, outdir, output, &cstore, expanded_crate, &crate_name,
                    );
                    controller_entry_point!(after_expand, sess, state, Ok(()));
                    Ok(())
                }
            )?
        };

        write_out_deps(sess, &outputs, &crate_name);
        if sess.opts.output_types.contains_key(&OutputType::DepInfo) &&
            sess.opts.output_types.keys().count() == 1 {
            return Ok(())
        }

        let arena = DroplessArena::new();
        let arenas = GlobalArenas::new();

        // Construct the HIR map
        let hir_map = time(sess.time_passes(),
                           "indexing hir",
                           || hir_map::map_crate(&mut hir_forest, &defs));

        {
            let _ignore = hir_map.dep_graph.in_ignore();
            controller_entry_point!(after_hir_lowering,
                                    sess,
                                    CompileState::state_after_hir_lowering(input,
                                                                  sess,
                                                                  outdir,
                                                                  output,
                                                                  &arena,
                                                                  &arenas,
                                                                  &cstore,
                                                                  &hir_map,
                                                                  &analysis,
                                                                  &resolutions,
                                                                  &expanded_crate,
                                                                  &hir_map.krate(),
                                                                  &outputs,
                                                                  &crate_name),
                                    Ok(()));
        }

        time(sess.time_passes(), "attribute checking", || {
            hir::check_attr::check_crate(sess, &expanded_crate);
        });

        let opt_crate = if control.keep_ast {
            Some(&expanded_crate)
        } else {
            drop(expanded_crate);
            None
        };

        phase_3_run_analysis_passes(sess,
                                    cstore,
                                    hir_map,
                                    analysis,
                                    resolutions,
                                    &arena,
                                    &arenas,
                                    &crate_name,
                                    &outputs,
                                    |tcx, analysis, incremental_hashes_map, rx, result| {
            {
                // Eventually, we will want to track plugins.
                let _ignore = tcx.dep_graph.in_ignore();

                let mut state = CompileState::state_after_analysis(input,
                                                                   sess,
                                                                   outdir,
                                                                   output,
                                                                   opt_crate,
                                                                   tcx.hir.krate(),
                                                                   &analysis,
                                                                   tcx,
                                                                   &crate_name);
                (control.after_analysis.callback)(&mut state);

                if control.after_analysis.stop == Compilation::Stop {
                    return result.and_then(|_| Err(CompileIncomplete::Stopped));
                }
            }

            result?;

            if log_enabled!(::log::LogLevel::Info) {
                println!("Pre-trans");
                tcx.print_debug_stats();
            }

            let trans = phase_4_translate_to_llvm(tcx,
                                                  incremental_hashes_map,
                                                  rx);

            if log_enabled!(::log::LogLevel::Info) {
                println!("Post-trans");
                tcx.print_debug_stats();
            }

            if tcx.sess.opts.output_types.contains_key(&OutputType::Mir) {
                if let Err(e) = mir::transform::dump_mir::emit_mir(tcx, &outputs) {
                    sess.err(&format!("could not emit MIR: {}", e));
                    sess.abort_if_errors();
                }
            }

            Ok((outputs.clone(), trans, tcx.dep_graph.clone()))
        })??
    };

    if cfg!(not(feature="llvm")) {
        let (_, _) = (outputs, trans);
        sess.fatal("LLVM is not supported by this rustc");
    }

    #[cfg(feature="llvm")]
    {
        if sess.opts.debugging_opts.print_type_sizes {
            sess.code_stats.borrow().print_type_sizes();
        }

        let (phase5_result, trans) = phase_5_run_llvm_passes(sess, &dep_graph, trans);

        controller_entry_point!(after_llvm,
                                sess,
                                CompileState::state_after_llvm(input, sess, outdir, output, &trans),
                                phase5_result);
        phase5_result?;

        phase_6_link_output(sess, &trans, &outputs);

        // Now that we won't touch anything in the incremental compilation directory
        // any more, we can finalize it (which involves renaming it)
        rustc_incremental::finalize_session_directory(sess, trans.link.crate_hash);

        if sess.opts.debugging_opts.perf_stats {
            sess.print_perf_stats();
        }

        controller_entry_point!(
            compilation_done,
            sess,
            CompileState::state_when_compilation_done(input, sess, outdir, output),
            Ok(())
        );

        Ok(())
    }
}

fn keep_hygiene_data(sess: &Session) -> bool {
    sess.opts.debugging_opts.keep_hygiene_data
}


/// The name used for source code that doesn't originate in a file
/// (e.g. source from stdin or a string)
pub fn anon_src() -> String {
    "<anon>".to_string()
}

pub fn source_name(input: &Input) -> String {
    match *input {
        // FIXME (#9639): This needs to handle non-utf8 paths
        Input::File(ref ifile) => ifile.to_str().unwrap().to_string(),
        Input::Str { ref name, .. } => name.clone(),
    }
}

/// CompileController is used to customize compilation, it allows compilation to
/// be stopped and/or to call arbitrary code at various points in compilation.
/// It also allows for various flags to be set to influence what information gets
/// collected during compilation.
///
/// This is a somewhat higher level controller than a Session - the Session
/// controls what happens in each phase, whereas the CompileController controls
/// whether a phase is run at all and whether other code (from outside the
/// compiler) is run between phases.
///
/// Note that if compilation is set to stop and a callback is provided for a
/// given entry point, the callback is called before compilation is stopped.
///
/// Expect more entry points to be added in the future.
pub struct CompileController<'a> {
    pub after_parse: PhaseController<'a>,
    pub after_expand: PhaseController<'a>,
    pub after_hir_lowering: PhaseController<'a>,
    pub after_analysis: PhaseController<'a>,
    pub after_llvm: PhaseController<'a>,
    pub compilation_done: PhaseController<'a>,

    // FIXME we probably want to group the below options together and offer a
    // better API, rather than this ad-hoc approach.
    pub make_glob_map: MakeGlobMap,
    // Whether the compiler should keep the ast beyond parsing.
    pub keep_ast: bool,
    // -Zcontinue-parse-after-error
    pub continue_parse_after_error: bool,
}

impl<'a> CompileController<'a> {
    pub fn basic() -> CompileController<'a> {
        CompileController {
            after_parse: PhaseController::basic(),
            after_expand: PhaseController::basic(),
            after_hir_lowering: PhaseController::basic(),
            after_analysis: PhaseController::basic(),
            after_llvm: PhaseController::basic(),
            compilation_done: PhaseController::basic(),
            make_glob_map: MakeGlobMap::No,
            keep_ast: false,
            continue_parse_after_error: false,
        }
    }
}

pub struct PhaseController<'a> {
    pub stop: Compilation,
    // If true then the compiler will try to run the callback even if the phase
    // ends with an error. Note that this is not always possible.
    pub run_callback_on_error: bool,
    pub callback: Box<Fn(&mut CompileState) + 'a>,
}

impl<'a> PhaseController<'a> {
    pub fn basic() -> PhaseController<'a> {
        PhaseController {
            stop: Compilation::Continue,
            run_callback_on_error: false,
            callback: box |_| {},
        }
    }
}

/// State that is passed to a callback. What state is available depends on when
/// during compilation the callback is made. See the various constructor methods
/// (`state_*`) in the impl to see which data is provided for any given entry point.
pub struct CompileState<'a, 'tcx: 'a> {
    pub input: &'a Input,
    pub session: &'tcx Session,
    pub krate: Option<ast::Crate>,
    pub registry: Option<Registry<'a>>,
    pub cstore: Option<&'tcx CStore>,
    pub crate_name: Option<&'a str>,
    pub output_filenames: Option<&'a OutputFilenames>,
    pub out_dir: Option<&'a Path>,
    pub out_file: Option<&'a Path>,
    pub arena: Option<&'tcx DroplessArena>,
    pub arenas: Option<&'tcx GlobalArenas<'tcx>>,
    pub expanded_crate: Option<&'a ast::Crate>,
    pub hir_crate: Option<&'a hir::Crate>,
    pub hir_map: Option<&'a hir_map::Map<'tcx>>,
    pub resolutions: Option<&'a Resolutions>,
    pub analysis: Option<&'a ty::CrateAnalysis>,
    pub tcx: Option<TyCtxt<'a, 'tcx, 'tcx>>,
    pub trans: Option<&'a trans::CrateTranslation>,
}

impl<'a, 'tcx> CompileState<'a, 'tcx> {
    fn empty(input: &'a Input,
             session: &'tcx Session,
             out_dir: &'a Option<PathBuf>)
             -> Self {
        CompileState {
            input,
            session,
            out_dir: out_dir.as_ref().map(|s| &**s),
            out_file: None,
            arena: None,
            arenas: None,
            krate: None,
            registry: None,
            cstore: None,
            crate_name: None,
            output_filenames: None,
            expanded_crate: None,
            hir_crate: None,
            hir_map: None,
            resolutions: None,
            analysis: None,
            tcx: None,
            trans: None,
        }
    }

    fn state_after_parse(input: &'a Input,
                         session: &'tcx Session,
                         out_dir: &'a Option<PathBuf>,
                         out_file: &'a Option<PathBuf>,
                         krate: ast::Crate,
                         cstore: &'tcx CStore)
                         -> Self {
        CompileState {
            // Initialize the registry before moving `krate`
            registry: Some(Registry::new(&session, krate.span)),
            krate: Some(krate),
            cstore: Some(cstore),
            out_file: out_file.as_ref().map(|s| &**s),
            ..CompileState::empty(input, session, out_dir)
        }
    }

    fn state_after_expand(input: &'a Input,
                          session: &'tcx Session,
                          out_dir: &'a Option<PathBuf>,
                          out_file: &'a Option<PathBuf>,
                          cstore: &'tcx CStore,
                          expanded_crate: &'a ast::Crate,
                          crate_name: &'a str)
                          -> Self {
        CompileState {
            crate_name: Some(crate_name),
            cstore: Some(cstore),
            expanded_crate: Some(expanded_crate),
            out_file: out_file.as_ref().map(|s| &**s),
            ..CompileState::empty(input, session, out_dir)
        }
    }

    fn state_after_hir_lowering(input: &'a Input,
                                session: &'tcx Session,
                                out_dir: &'a Option<PathBuf>,
                                out_file: &'a Option<PathBuf>,
                                arena: &'tcx DroplessArena,
                                arenas: &'tcx GlobalArenas<'tcx>,
                                cstore: &'tcx CStore,
                                hir_map: &'a hir_map::Map<'tcx>,
                                analysis: &'a ty::CrateAnalysis,
                                resolutions: &'a Resolutions,
                                krate: &'a ast::Crate,
                                hir_crate: &'a hir::Crate,
                                output_filenames: &'a OutputFilenames,
                                crate_name: &'a str)
                                -> Self {
        CompileState {
            crate_name: Some(crate_name),
            arena: Some(arena),
            arenas: Some(arenas),
            cstore: Some(cstore),
            hir_map: Some(hir_map),
            analysis: Some(analysis),
            resolutions: Some(resolutions),
            expanded_crate: Some(krate),
            hir_crate: Some(hir_crate),
            output_filenames: Some(output_filenames),
            out_file: out_file.as_ref().map(|s| &**s),
            ..CompileState::empty(input, session, out_dir)
        }
    }

    fn state_after_analysis(input: &'a Input,
                            session: &'tcx Session,
                            out_dir: &'a Option<PathBuf>,
                            out_file: &'a Option<PathBuf>,
                            krate: Option<&'a ast::Crate>,
                            hir_crate: &'a hir::Crate,
                            analysis: &'a ty::CrateAnalysis,
                            tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            crate_name: &'a str)
                            -> Self {
        CompileState {
            analysis: Some(analysis),
            tcx: Some(tcx),
            expanded_crate: krate,
            hir_crate: Some(hir_crate),
            crate_name: Some(crate_name),
            out_file: out_file.as_ref().map(|s| &**s),
            ..CompileState::empty(input, session, out_dir)
        }
    }

    fn state_after_llvm(input: &'a Input,
                        session: &'tcx Session,
                        out_dir: &'a Option<PathBuf>,
                        out_file: &'a Option<PathBuf>,
                        trans: &'a trans::CrateTranslation)
                        -> Self {
        CompileState {
            trans: Some(trans),
            out_file: out_file.as_ref().map(|s| &**s),
            ..CompileState::empty(input, session, out_dir)
        }
    }

    fn state_when_compilation_done(input: &'a Input,
                                   session: &'tcx Session,
                                   out_dir: &'a Option<PathBuf>,
                                   out_file: &'a Option<PathBuf>)
                                   -> Self {
        CompileState {
            out_file: out_file.as_ref().map(|s| &**s),
            ..CompileState::empty(input, session, out_dir)
        }
    }
}

pub fn phase_1_parse_input<'a>(control: &CompileController,
                               sess: &'a Session,
                               input: &Input)
                               -> PResult<'a, ast::Crate> {
    sess.diagnostic().set_continue_after_error(control.continue_parse_after_error);

    if sess.profile_queries() {
        profile::begin();
    }

    let krate = time(sess.time_passes(), "parsing", || {
        match *input {
            Input::File(ref file) => {
                parse::parse_crate_from_file(file, &sess.parse_sess)
            }
            Input::Str { ref input, ref name } => {
                parse::parse_crate_from_source_str(name.clone(), input.clone(), &sess.parse_sess)
            }
        }
    })?;

    sess.diagnostic().set_continue_after_error(true);

    if sess.opts.debugging_opts.ast_json_noexpand {
        println!("{}", json::as_json(&krate));
    }

    if sess.opts.debugging_opts.input_stats {
        println!("Lines of code:             {}", sess.codemap().count_lines());
        println!("Pre-expansion node count:  {}", count_nodes(&krate));
    }

    if let Some(ref s) = sess.opts.debugging_opts.show_span {
        syntax::show_span::run(sess.diagnostic(), s, &krate);
    }

    if sess.opts.debugging_opts.hir_stats {
        hir_stats::print_ast_stats(&krate, "PRE EXPANSION AST STATS");
    }

    Ok(krate)
}

fn count_nodes(krate: &ast::Crate) -> usize {
    let mut counter = NodeCounter::new();
    visit::walk_crate(&mut counter, krate);
    counter.count
}

// For continuing compilation after a parsed crate has been
// modified

pub struct ExpansionResult {
    pub expanded_crate: ast::Crate,
    pub defs: hir_map::Definitions,
    pub analysis: ty::CrateAnalysis,
    pub resolutions: Resolutions,
    pub hir_forest: hir_map::Forest,
}

/// Run the "early phases" of the compiler: initial `cfg` processing,
/// loading compiler plugins (including those from `addl_plugins`),
/// syntax expansion, secondary `cfg` expansion, synthesis of a test
/// harness if one is to be provided, injection of a dependency on the
/// standard library and prelude, and name resolution.
///
/// Returns `None` if we're aborting after handling -W help.
pub fn phase_2_configure_and_expand<F>(sess: &Session,
                                       cstore: &CStore,
                                       krate: ast::Crate,
                                       registry: Option<Registry>,
                                       crate_name: &str,
                                       addl_plugins: Option<Vec<String>>,
                                       make_glob_map: MakeGlobMap,
                                       after_expand: F)
                                       -> Result<ExpansionResult, CompileIncomplete>
    where F: FnOnce(&ast::Crate) -> CompileResult,
{
    let time_passes = sess.time_passes();

    let (mut krate, features) = syntax::config::features(krate, &sess.parse_sess, sess.opts.test);
    // these need to be set "early" so that expansion sees `quote` if enabled.
    *sess.features.borrow_mut() = features;

    *sess.crate_types.borrow_mut() = collect_crate_types(sess, &krate.attrs);

    let disambiguator = Symbol::intern(&compute_crate_disambiguator(sess));
    *sess.crate_disambiguator.borrow_mut() = Some(disambiguator);
    rustc_incremental::prepare_session_directory(
        sess,
        &crate_name,
        &disambiguator.as_str(),
    );
    let dep_graph = DepGraph::new(sess.opts.build_dep_graph());

    time(time_passes, "recursion limit", || {
        middle::recursion_limit::update_limits(sess, &krate);
    });

    krate = time(time_passes, "crate injection", || {
        let alt_std_name = sess.opts.alt_std_name.clone();
        syntax::std_inject::maybe_inject_crates_ref(krate, alt_std_name)
    });

    let mut addl_plugins = Some(addl_plugins);
    let registrars = time(time_passes, "plugin loading", || {
        plugin::load::load_plugins(sess,
                                   &cstore,
                                   &krate,
                                   crate_name,
                                   addl_plugins.take().unwrap())
    });

    let mut registry = registry.unwrap_or(Registry::new(sess, krate.span));

    time(time_passes, "plugin registration", || {
        if sess.features.borrow().rustc_diagnostic_macros {
            registry.register_macro("__diagnostic_used",
                                    diagnostics::plugin::expand_diagnostic_used);
            registry.register_macro("__register_diagnostic",
                                    diagnostics::plugin::expand_register_diagnostic);
            registry.register_macro("__build_diagnostic_array",
                                    diagnostics::plugin::expand_build_diagnostic_array);
        }

        for registrar in registrars {
            registry.args_hidden = Some(registrar.args);
            (registrar.fun)(&mut registry);
        }
    });

    let whitelisted_legacy_custom_derives = registry.take_whitelisted_custom_derives();
    let Registry { syntax_exts, early_lint_passes, late_lint_passes, lint_groups,
                   llvm_passes, attributes, .. } = registry;

    sess.track_errors(|| {
        let mut ls = sess.lint_store.borrow_mut();
        for pass in early_lint_passes {
            ls.register_early_pass(Some(sess), true, pass);
        }
        for pass in late_lint_passes {
            ls.register_late_pass(Some(sess), true, pass);
        }

        for (name, to) in lint_groups {
            ls.register_group(Some(sess), true, name, to);
        }

        *sess.plugin_llvm_passes.borrow_mut() = llvm_passes;
        *sess.plugin_attributes.borrow_mut() = attributes.clone();
    })?;

    // Lint plugins are registered; now we can process command line flags.
    if sess.opts.describe_lints {
        super::describe_lints(&sess.lint_store.borrow(), true);
        return Err(CompileIncomplete::Stopped);
    }

    // Currently, we ignore the name resolution data structures for the purposes of dependency
    // tracking. Instead we will run name resolution and include its output in the hash of each
    // item, much like we do for macro expansion. In other words, the hash reflects not just
    // its contents but the results of name resolution on those contents. Hopefully we'll push
    // this back at some point.
    let _ignore = dep_graph.in_ignore();
    let mut crate_loader = CrateLoader::new(sess, &cstore, crate_name);
    let resolver_arenas = Resolver::arenas();
    let mut resolver = Resolver::new(sess,
                                     cstore,
                                     &krate,
                                     crate_name,
                                     make_glob_map,
                                     &mut crate_loader,
                                     &resolver_arenas);
    resolver.whitelisted_legacy_custom_derives = whitelisted_legacy_custom_derives;
    syntax_ext::register_builtins(&mut resolver, syntax_exts, sess.features.borrow().quote);

    krate = time(time_passes, "expansion", || {
        // Windows dlls do not have rpaths, so they don't know how to find their
        // dependencies. It's up to us to tell the system where to find all the
        // dependent dlls. Note that this uses cfg!(windows) as opposed to
        // targ_cfg because syntax extensions are always loaded for the host
        // compiler, not for the target.
        //
        // This is somewhat of an inherently racy operation, however, as
        // multiple threads calling this function could possibly continue
        // extending PATH far beyond what it should. To solve this for now we
        // just don't add any new elements to PATH which are already there
        // within PATH. This is basically a targeted fix at #17360 for rustdoc
        // which runs rustc in parallel but has been seen (#33844) to cause
        // problems with PATH becoming too long.
        let mut old_path = OsString::new();
        if cfg!(windows) {
            old_path = env::var_os("PATH").unwrap_or(old_path);
            let mut new_path = sess.host_filesearch(PathKind::All)
                                   .get_dylib_search_paths();
            for path in env::split_paths(&old_path) {
                if !new_path.contains(&path) {
                    new_path.push(path);
                }
            }
            env::set_var("PATH",
                &env::join_paths(new_path.iter()
                                         .filter(|p| env::join_paths(iter::once(p)).is_ok()))
                     .unwrap());
        }
        let features = sess.features.borrow();
        let cfg = syntax::ext::expand::ExpansionConfig {
            features: Some(&features),
            recursion_limit: sess.recursion_limit.get(),
            trace_mac: sess.opts.debugging_opts.trace_macros,
            should_test: sess.opts.test,
            ..syntax::ext::expand::ExpansionConfig::default(crate_name.to_string())
        };

        let mut ecx = ExtCtxt::new(&sess.parse_sess, cfg, &mut resolver);
        let err_count = ecx.parse_sess.span_diagnostic.err_count();

        let krate = ecx.monotonic_expander().expand_crate(krate);

        ecx.check_unused_macros();

        let mut missing_fragment_specifiers: Vec<_> =
            ecx.parse_sess.missing_fragment_specifiers.borrow().iter().cloned().collect();
        missing_fragment_specifiers.sort();
        for span in missing_fragment_specifiers {
            let lint = lint::builtin::MISSING_FRAGMENT_SPECIFIER;
            let msg = "missing fragment specifier";
            sess.buffer_lint(lint, ast::CRATE_NODE_ID, span, msg);
        }
        if ecx.parse_sess.span_diagnostic.err_count() - ecx.resolve_err_count > err_count {
            ecx.parse_sess.span_diagnostic.abort_if_errors();
        }
        if cfg!(windows) {
            env::set_var("PATH", &old_path);
        }
        krate
    });

    krate = time(time_passes, "maybe building test harness", || {
        syntax::test::modify_for_testing(&sess.parse_sess,
                                         &mut resolver,
                                         sess.opts.test,
                                         krate,
                                         sess.diagnostic())
    });

    // If we're in rustdoc we're always compiling as an rlib, but that'll trip a
    // bunch of checks in the `modify` function below. For now just skip this
    // step entirely if we're rustdoc as it's not too useful anyway.
    if !sess.opts.actually_rustdoc {
        krate = time(time_passes, "maybe creating a macro crate", || {
            let crate_types = sess.crate_types.borrow();
            let num_crate_types = crate_types.len();
            let is_proc_macro_crate = crate_types.contains(&config::CrateTypeProcMacro);
            let is_test_crate = sess.opts.test;
            syntax_ext::proc_macro_registrar::modify(&sess.parse_sess,
                                                     &mut resolver,
                                                     krate,
                                                     is_proc_macro_crate,
                                                     is_test_crate,
                                                     num_crate_types,
                                                     sess.diagnostic())
        });
    }

    krate = time(time_passes, "creating allocators", || {
        allocator::expand::modify(&sess.parse_sess,
                                  &mut resolver,
                                  krate,
                                  sess.diagnostic())
    });

    after_expand(&krate)?;

    if sess.opts.debugging_opts.input_stats {
        println!("Post-expansion node count: {}", count_nodes(&krate));
    }

    if sess.opts.debugging_opts.hir_stats {
        hir_stats::print_ast_stats(&krate, "POST EXPANSION AST STATS");
    }

    if sess.opts.debugging_opts.ast_json {
        println!("{}", json::as_json(&krate));
    }

    time(time_passes,
         "checking for inline asm in case the target doesn't support it",
         || no_asm::check_crate(sess, &krate));

    time(time_passes,
         "AST validation",
         || ast_validation::check_crate(sess, &krate));

    time(time_passes, "name resolution", || -> CompileResult {
        resolver.resolve_crate(&krate);
        Ok(())
    })?;

    if resolver.found_unresolved_macro {
        sess.parse_sess.span_diagnostic.abort_if_errors();
    }

    // Needs to go *after* expansion to be able to check the results of macro expansion.
    time(time_passes, "complete gated feature checking", || {
        sess.track_errors(|| {
            syntax::feature_gate::check_crate(&krate,
                                              &sess.parse_sess,
                                              &sess.features.borrow(),
                                              &attributes,
                                              sess.opts.unstable_features);
        })
    })?;

    // Lower ast -> hir.
    let hir_forest = time(time_passes, "lowering ast -> hir", || {
        let hir_crate = lower_crate(sess, cstore, &dep_graph, &krate, &mut resolver);

        if sess.opts.debugging_opts.hir_stats {
            hir_stats::print_hir_stats(&hir_crate);
        }

        hir_map::Forest::new(hir_crate, &dep_graph)
    });

    time(time_passes,
         "early lint checks",
         || lint::check_ast_crate(sess, &krate));

    // Discard hygiene data, which isn't required after lowering to HIR.
    if !keep_hygiene_data(sess) {
        syntax::ext::hygiene::clear_markings();
    }

    Ok(ExpansionResult {
        expanded_crate: krate,
        defs: resolver.definitions,
        analysis: ty::CrateAnalysis {
            access_levels: Rc::new(AccessLevels::default()),
            name: crate_name.to_string(),
            glob_map: if resolver.make_glob_map { Some(resolver.glob_map) } else { None },
        },
        resolutions: Resolutions {
            freevars: resolver.freevars,
            export_map: resolver.export_map,
            trait_map: resolver.trait_map,
            maybe_unused_trait_imports: resolver.maybe_unused_trait_imports,
            maybe_unused_extern_crates: resolver.maybe_unused_extern_crates,
        },
        hir_forest,
    })
}

/// Run the resolution, typechecking, region checking and other
/// miscellaneous analysis passes on the crate. Return various
/// structures carrying the results of the analysis.
pub fn phase_3_run_analysis_passes<'tcx, F, R>(sess: &'tcx Session,
                                               cstore: &'tcx CrateStore,
                                               hir_map: hir_map::Map<'tcx>,
                                               mut analysis: ty::CrateAnalysis,
                                               resolutions: Resolutions,
                                               arena: &'tcx DroplessArena,
                                               arenas: &'tcx GlobalArenas<'tcx>,
                                               name: &str,
                                               output_filenames: &OutputFilenames,
                                               f: F)
                                               -> Result<R, CompileIncomplete>
    where F: for<'a> FnOnce(TyCtxt<'a, 'tcx, 'tcx>,
                            ty::CrateAnalysis,
                            IncrementalHashesMap,
                            mpsc::Receiver<Box<Any + Send>>,
                            CompileResult) -> R
{
    macro_rules! try_with_f {
        ($e: expr, ($($t:tt)*)) => {
            match $e {
                Ok(x) => x,
                Err(x) => {
                    f($($t)*, Err(x));
                    return Err(x);
                }
            }
        }
    }

    let time_passes = sess.time_passes();

    let named_region_map = time(time_passes,
                                "lifetime resolution",
                                || middle::resolve_lifetime::krate(sess, cstore, &hir_map))?;

    time(time_passes,
         "looking for entry point",
         || middle::entry::find_entry_point(sess, &hir_map));

    sess.plugin_registrar_fn.set(time(time_passes, "looking for plugin registrar", || {
        plugin::build::find_plugin_registrar(sess.diagnostic(), &hir_map)
    }));
    sess.derive_registrar_fn.set(derive_registrar::find(&hir_map));

    time(time_passes,
         "loop checking",
         || loops::check_crate(sess, &hir_map));

    time(time_passes,
              "static item recursion checking",
              || static_recursion::check_crate(sess, &hir_map))?;

    let mut local_providers = ty::maps::Providers::default();
    borrowck::provide(&mut local_providers);
    mir::provide(&mut local_providers);
    reachable::provide(&mut local_providers);
    rustc_privacy::provide(&mut local_providers);
    trans::provide_local(&mut local_providers);
    typeck::provide(&mut local_providers);
    ty::provide(&mut local_providers);
    traits::provide(&mut local_providers);
    reachable::provide(&mut local_providers);
    rustc_const_eval::provide(&mut local_providers);
    middle::region::provide(&mut local_providers);
    cstore::provide_local(&mut local_providers);
    lint::provide(&mut local_providers);

    let mut extern_providers = ty::maps::Providers::default();
    cstore::provide(&mut extern_providers);
    trans::provide_extern(&mut extern_providers);
    ty::provide_extern(&mut extern_providers);
    traits::provide_extern(&mut extern_providers);
    // FIXME(eddyb) get rid of this once we replace const_eval with miri.
    rustc_const_eval::provide(&mut extern_providers);

    // Setup the MIR passes that we want to run.
    let mut passes = Passes::new();
    passes.push_hook(mir::transform::dump_mir::DumpMir);

    // Remove all `EndRegion` statements that are not involved in borrows.
    passes.push_pass(MIR_CONST, mir::transform::clean_end_regions::CleanEndRegions);

    // What we need to do constant evaluation.
    passes.push_pass(MIR_CONST, mir::transform::simplify::SimplifyCfg::new("initial"));
    passes.push_pass(MIR_CONST, mir::transform::type_check::TypeckMir);
    passes.push_pass(MIR_CONST, mir::transform::rustc_peek::SanityCheck);

    // We compute "constant qualifications" between MIR_CONST and MIR_VALIDATED.

    // What we need to run borrowck etc.

    passes.push_pass(MIR_VALIDATED, mir::transform::qualify_consts::QualifyAndPromoteConstants);

    // FIXME: ariel points SimplifyBranches should run after
    // mir-borrowck; otherwise code within `if false { ... }` would
    // not be checked.
    passes.push_pass(MIR_VALIDATED,
                     mir::transform::simplify_branches::SimplifyBranches::new("initial"));
    passes.push_pass(MIR_VALIDATED, mir::transform::simplify::SimplifyCfg::new("qualify-consts"));
    passes.push_pass(MIR_VALIDATED, mir::transform::nll::NLL);

    // borrowck runs between MIR_VALIDATED and MIR_OPTIMIZED.

    // These next passes must be executed together
    passes.push_pass(MIR_OPTIMIZED, mir::transform::no_landing_pads::NoLandingPads);
    passes.push_pass(MIR_OPTIMIZED, mir::transform::add_call_guards::CriticalCallEdges);
    passes.push_pass(MIR_OPTIMIZED, mir::transform::elaborate_drops::ElaborateDrops);
    passes.push_pass(MIR_OPTIMIZED, mir::transform::no_landing_pads::NoLandingPads);
    // AddValidation needs to run after ElaborateDrops and before EraseRegions, and it needs
    // an AllCallEdges pass right before it.
    passes.push_pass(MIR_OPTIMIZED, mir::transform::add_call_guards::AllCallEdges);
    passes.push_pass(MIR_OPTIMIZED, mir::transform::add_validation::AddValidation);
    passes.push_pass(MIR_OPTIMIZED, mir::transform::simplify::SimplifyCfg::new("elaborate-drops"));
    // No lifetime analysis based on borrowing can be done from here on out.

    // From here on out, regions are gone.
    passes.push_pass(MIR_OPTIMIZED, mir::transform::erase_regions::EraseRegions);

    // Optimizations begin.
    passes.push_pass(MIR_OPTIMIZED, mir::transform::inline::Inline);
    passes.push_pass(MIR_OPTIMIZED, mir::transform::instcombine::InstCombine);
    passes.push_pass(MIR_OPTIMIZED, mir::transform::deaggregator::Deaggregator);
    passes.push_pass(MIR_OPTIMIZED, mir::transform::copy_prop::CopyPropagation);
    passes.push_pass(MIR_OPTIMIZED, mir::transform::simplify::SimplifyLocals);

    passes.push_pass(MIR_OPTIMIZED, mir::transform::generator::StateTransform);
    passes.push_pass(MIR_OPTIMIZED, mir::transform::add_call_guards::CriticalCallEdges);
    passes.push_pass(MIR_OPTIMIZED, mir::transform::dump_mir::Marker("PreTrans"));

    let (tx, rx) = mpsc::channel();

    TyCtxt::create_and_enter(sess,
                             cstore,
                             local_providers,
                             extern_providers,
                             Rc::new(passes),
                             arenas,
                             arena,
                             resolutions,
                             named_region_map,
                             hir_map,
                             name,
                             tx,
                             output_filenames,
                             |tcx| {
        let incremental_hashes_map =
            time(time_passes,
                 "compute_incremental_hashes_map",
                 || rustc_incremental::compute_incremental_hashes_map(tcx));

        time(time_passes,
             "load_dep_graph",
             || rustc_incremental::load_dep_graph(tcx, &incremental_hashes_map));

        time(time_passes,
             "stability checking",
             || stability::check_unstable_api_usage(tcx));

        // passes are timed inside typeck
        try_with_f!(typeck::check_crate(tcx),
                    (tcx, analysis, incremental_hashes_map, rx));

        time(time_passes,
             "const checking",
             || consts::check_crate(tcx));

        analysis.access_levels =
            time(time_passes, "privacy checking", || rustc_privacy::check_crate(tcx));

        time(time_passes,
             "intrinsic checking",
             || middle::intrinsicck::check_crate(tcx));

        time(time_passes,
             "effect checking",
             || middle::effect::check_crate(tcx));

        time(time_passes,
             "match checking",
             || check_match::check_crate(tcx));

        // this must run before MIR dump, because
        // "not all control paths return a value" is reported here.
        //
        // maybe move the check to a MIR pass?
        time(time_passes,
             "liveness checking",
             || middle::liveness::check_crate(tcx));

        time(time_passes,
             "borrow checking",
             || borrowck::check_crate(tcx));

        time(time_passes,
             "MIR borrow checking",
             || for def_id in tcx.body_owners() { tcx.mir_borrowck(def_id) });

        // Avoid overwhelming user with errors if type checking failed.
        // I'm not sure how helpful this is, to be honest, but it avoids
        // a
        // lot of annoying errors in the compile-fail tests (basically,
        // lint warnings and so on -- kindck used to do this abort, but
        // kindck is gone now). -nmatsakis
        if sess.err_count() > 0 {
            return Ok(f(tcx, analysis, incremental_hashes_map, rx, sess.compile_status()));
        }

        time(time_passes, "death checking", || middle::dead::check_crate(tcx));

        time(time_passes, "unused lib feature checking", || {
            stability::check_unused_or_stable_features(tcx)
        });

        time(time_passes, "lint checking", || lint::check_crate(tcx));

        return Ok(f(tcx, analysis, incremental_hashes_map, rx, tcx.sess.compile_status()));
    })
}

/// Run the translation phase to LLVM, after which the AST and analysis can
/// be discarded.
pub fn phase_4_translate_to_llvm<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                           incremental_hashes_map: IncrementalHashesMap,
                                           rx: mpsc::Receiver<Box<Any + Send>>)
                                           -> write::OngoingCrateTranslation {
    let time_passes = tcx.sess.time_passes();

    time(time_passes,
         "resolving dependency formats",
         || ::rustc::middle::dependency_format::calculate(tcx));

    let translation =
        time(time_passes, "translation", move || {
            trans::trans_crate(tcx, incremental_hashes_map, rx)
        });

    if tcx.sess.profile_queries() {
        profile::dump("profile_queries".to_string())
    }

    translation
}

/// Run LLVM itself, producing a bitcode file, assembly file or object file
/// as a side effect.
#[cfg(feature="llvm")]
pub fn phase_5_run_llvm_passes(sess: &Session,
                               dep_graph: &DepGraph,
                               trans: write::OngoingCrateTranslation)
                               -> (CompileResult, trans::CrateTranslation) {
    let trans = trans.join(sess, dep_graph);

    if sess.opts.debugging_opts.incremental_info {
        write::dump_incremental_data(&trans);
    }

    time(sess.time_passes(),
         "serialize work products",
         move || rustc_incremental::save_work_products(sess, dep_graph));

    (sess.compile_status(), trans)
}

/// Run the linker on any artifacts that resulted from the LLVM run.
/// This should produce either a finished executable or library.
#[cfg(feature="llvm")]
pub fn phase_6_link_output(sess: &Session,
                           trans: &trans::CrateTranslation,
                           outputs: &OutputFilenames) {
    time(sess.time_passes(), "linking", || {
        ::rustc_trans::back::link::link_binary(sess,
                                               trans,
                                               outputs,
                                               &trans.crate_name.as_str())
    });
}

fn escape_dep_filename(filename: &str) -> String {
    // Apparently clang and gcc *only* escape spaces:
    // http://llvm.org/klaus/clang/commit/9d50634cfc268ecc9a7250226dd5ca0e945240d4
    filename.replace(" ", "\\ ")
}

fn write_out_deps(sess: &Session, outputs: &OutputFilenames, crate_name: &str) {
    let mut out_filenames = Vec::new();
    for output_type in sess.opts.output_types.keys() {
        let file = outputs.path(*output_type);
        match *output_type {
            OutputType::Exe => {
                for output in sess.crate_types.borrow().iter() {
                    let p = ::rustc_trans_utils::link::filename_for_input(
                        sess,
                        *output,
                        crate_name,
                        outputs
                    );
                    out_filenames.push(p);
                }
            }
            _ => {
                out_filenames.push(file);
            }
        }
    }

    // Write out dependency rules to the dep-info file if requested
    if !sess.opts.output_types.contains_key(&OutputType::DepInfo) {
        return;
    }
    let deps_filename = outputs.path(OutputType::DepInfo);

    let result =
        (|| -> io::Result<()> {
            // Build a list of files used to compile the output and
            // write Makefile-compatible dependency rules
            let files: Vec<String> = sess.codemap()
                                         .files()
                                         .iter()
                                         .filter(|fmap| fmap.is_real_file())
                                         .filter(|fmap| !fmap.is_imported())
                                         .map(|fmap| escape_dep_filename(&fmap.name))
                                         .collect();
            let mut file = fs::File::create(&deps_filename)?;
            for path in &out_filenames {
                write!(file, "{}: {}\n\n", path.display(), files.join(" "))?;
            }

            // Emit a fake target for each input file to the compilation. This
            // prevents `make` from spitting out an error if a file is later
            // deleted. For more info see #28735
            for path in files {
                writeln!(file, "{}:", path)?;
            }
            Ok(())
        })();

    match result {
        Ok(()) => {}
        Err(e) => {
            sess.fatal(&format!("error writing dependencies to `{}`: {}",
                                deps_filename.display(),
                                e));
        }
    }
}

pub fn collect_crate_types(session: &Session, attrs: &[ast::Attribute]) -> Vec<config::CrateType> {
    // Unconditionally collect crate types from attributes to make them used
    let attr_types: Vec<config::CrateType> =
        attrs.iter()
             .filter_map(|a| {
                 if a.check_name("crate_type") {
                     match a.value_str() {
                         Some(ref n) if *n == "rlib" => {
                             Some(config::CrateTypeRlib)
                         }
                         Some(ref n) if *n == "dylib" => {
                             Some(config::CrateTypeDylib)
                         }
                         Some(ref n) if *n == "cdylib" => {
                             Some(config::CrateTypeCdylib)
                         }
                         Some(ref n) if *n == "lib" => {
                             Some(config::default_lib_output())
                         }
                         Some(ref n) if *n == "staticlib" => {
                             Some(config::CrateTypeStaticlib)
                         }
                         Some(ref n) if *n == "proc-macro" => {
                             Some(config::CrateTypeProcMacro)
                         }
                         Some(ref n) if *n == "bin" => Some(config::CrateTypeExecutable),
                         Some(_) => {
                             session.buffer_lint(lint::builtin::UNKNOWN_CRATE_TYPES,
                                                 ast::CRATE_NODE_ID,
                                                 a.span,
                                                 "invalid `crate_type` value");
                             None
                         }
                         _ => {
                             session.struct_span_err(a.span, "`crate_type` requires a value")
                                 .note("for example: `#![crate_type=\"lib\"]`")
                                 .emit();
                             None
                         }
                     }
                 } else {
                     None
                 }
             })
             .collect();

    // If we're generating a test executable, then ignore all other output
    // styles at all other locations
    if session.opts.test {
        return vec![config::CrateTypeExecutable];
    }

    // Only check command line flags if present. If no types are specified by
    // command line, then reuse the empty `base` Vec to hold the types that
    // will be found in crate attributes.
    let mut base = session.opts.crate_types.clone();
    if base.is_empty() {
        base.extend(attr_types);
        if base.is_empty() {
            base.push(::rustc_trans_utils::link::default_output_for_target(session));
        }
        base.sort();
        base.dedup();
    }

    base.into_iter()
        .filter(|crate_type| {
            let res = !::rustc_trans_utils::link::invalid_output_for_target(session, *crate_type);

            if !res {
                session.warn(&format!("dropping unsupported crate type `{}` for target `{}`",
                                      *crate_type,
                                      session.opts.target_triple));
            }

            res
        })
        .collect()
}

pub fn compute_crate_disambiguator(session: &Session) -> String {
    use std::hash::Hasher;

    // The crate_disambiguator is a 128 bit hash. The disambiguator is fed
    // into various other hashes quite a bit (symbol hashes, incr. comp. hashes,
    // debuginfo type IDs, etc), so we don't want it to be too wide. 128 bits
    // should still be safe enough to avoid collisions in practice.
    // FIXME(mw): It seems that the crate_disambiguator is used everywhere as
    //            a hex-string instead of raw bytes. We should really use the
    //            smaller representation.
    let mut hasher = StableHasher::<Fingerprint>::new();

    let mut metadata = session.opts.cg.metadata.clone();
    // We don't want the crate_disambiguator to dependent on the order
    // -C metadata arguments, so sort them:
    metadata.sort();
    // Every distinct -C metadata value is only incorporated once:
    metadata.dedup();

    hasher.write(b"metadata");
    for s in &metadata {
        // Also incorporate the length of a metadata string, so that we generate
        // different values for `-Cmetadata=ab -Cmetadata=c` and
        // `-Cmetadata=a -Cmetadata=bc`
        hasher.write_usize(s.len());
        hasher.write(s.as_bytes());
    }

    // If this is an executable, add a special suffix, so that we don't get
    // symbol conflicts when linking against a library of the same name.
    let is_exe = session.crate_types.borrow().contains(&config::CrateTypeExecutable);

    format!("{}{}", hasher.finish().to_hex(), if is_exe { "-exe" } else {""})
}

pub fn build_output_filenames(input: &Input,
                              odir: &Option<PathBuf>,
                              ofile: &Option<PathBuf>,
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
                None => PathBuf::new(),
            };

            // If a crate name is present, we use it as the link name
            let stem = sess.opts
                           .crate_name
                           .clone()
                           .or_else(|| attr::find_crate_name(attrs).map(|n| n.to_string()))
                           .unwrap_or(input.filestem());

            OutputFilenames {
                out_directory: dirpath,
                out_filestem: stem,
                single_output_file: None,
                extra: sess.opts.cg.extra_filename.clone(),
                outputs: sess.opts.output_types.clone(),
            }
        }

        Some(ref out_file) => {
            let unnamed_output_types = sess.opts
                                           .output_types
                                           .values()
                                           .filter(|a| a.is_none())
                                           .count();
            let ofile = if unnamed_output_types > 1 {
                sess.warn("due to multiple output types requested, the explicitly specified \
                           output file name will be adapted for each output type");
                None
            } else {
                Some(out_file.clone())
            };
            if *odir != None {
                sess.warn("ignoring --out-dir flag due to -o flag.");
            }

            let cur_dir = Path::new("");

            OutputFilenames {
                out_directory: out_file.parent().unwrap_or(cur_dir).to_path_buf(),
                out_filestem: out_file.file_stem()
                                      .unwrap_or(OsStr::new(""))
                                      .to_str()
                                      .unwrap()
                                      .to_string(),
                single_output_file: ofile,
                extra: sess.opts.cg.extra_filename.clone(),
                outputs: sess.opts.output_types.clone(),
            }
        }
    }
}
