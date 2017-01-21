// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::{self, map as hir_map};
use rustc::hir::lowering::lower_crate;
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_mir as mir;
use rustc::session::{Session, CompileResult, compile_result_from_err_count};
use rustc::session::config::{self, Input, OutputFilenames, OutputType,
                             OutputTypes};
use rustc::session::search_paths::PathKind;
use rustc::lint;
use rustc::middle::{self, dependency_format, stability, reachable};
use rustc::middle::privacy::AccessLevels;
use rustc::ty::{self, TyCtxt, Resolutions, GlobalArenas};
use rustc::util::common::time;
use rustc::util::nodemap::{NodeSet, NodeMap};
use rustc_borrowck as borrowck;
use rustc_incremental::{self, IncrementalHashesMap};
use rustc_incremental::ich::Fingerprint;
use rustc_resolve::{MakeGlobMap, Resolver};
use rustc_metadata::creader::CrateLoader;
use rustc_metadata::cstore::CStore;
use rustc_trans::back::{link, write};
use rustc_trans as trans;
use rustc_typeck as typeck;
use rustc_privacy;
use rustc_plugin::registry::Registry;
use rustc_plugin as plugin;
use rustc_passes::{ast_validation, no_asm, loops, consts, rvalues,
                   static_recursion, hir_stats, mir_stats};
use rustc_const_eval::check_match;
use super::Compilation;

use serialize::json;

use std::env;
use std::mem;
use std::ffi::{OsString, OsStr};
use std::fs;
use std::io::{self, Write};
use std::iter;
use std::path::{Path, PathBuf};
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

pub fn compile_input(sess: &Session,
                     cstore: &CStore,
                     input: &Input,
                     outdir: &Option<PathBuf>,
                     output: &Option<PathBuf>,
                     addl_plugins: Option<Vec<String>>,
                     control: &CompileController) -> CompileResult {
    macro_rules! controller_entry_point {
        ($point: ident, $tsess: expr, $make_state: expr, $phase_result: expr) => {{
            let state = &mut $make_state;
            let phase_result: &CompileResult = &$phase_result;
            if phase_result.is_ok() || control.$point.run_callback_on_error {
                (control.$point.callback)(state);
            }

            if control.$point.stop == Compilation::Stop {
                return compile_result_from_err_count($tsess.err_count());
            }
        }}
    }

    // We need nested scopes here, because the intermediate results can keep
    // large chunks of memory alive and we want to free them as soon as
    // possible to keep the peak memory usage low
    let (outputs, trans) = {
        let krate = match phase_1_parse_input(sess, input) {
            Ok(krate) => krate,
            Err(mut parse_error) => {
                parse_error.emit();
                return Err(1);
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
        let crate_name = link::find_crate_name(Some(sess), &krate.attrs, input);
        let ExpansionResult { expanded_crate, defs, analysis, resolutions, mut hir_forest } = {
            phase_2_configure_and_expand(
                sess, &cstore, krate, registry, &crate_name, addl_plugins, control.make_glob_map,
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

        let arena = DroplessArena::new();
        let arenas = GlobalArenas::new();

        // Construct the HIR map
        let hir_map = time(sess.time_passes(),
                           "indexing hir",
                           || hir_map::map_crate(&mut hir_forest, defs));

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
                                                                  &crate_name),
                                    Ok(()));
        }

        time(sess.time_passes(), "attribute checking", || {
            hir::check_attr::check_crate(sess, &expanded_crate);
        });

        let opt_crate = if keep_ast(sess) {
            Some(&expanded_crate)
        } else {
            drop(expanded_crate);
            None
        };

        phase_3_run_analysis_passes(sess,
                                    hir_map,
                                    analysis,
                                    resolutions,
                                    &arena,
                                    &arenas,
                                    &crate_name,
                                    |tcx, analysis, incremental_hashes_map, result| {
            {
                // Eventually, we will want to track plugins.
                let _ignore = tcx.dep_graph.in_ignore();

                let mut state = CompileState::state_after_analysis(input,
                                                                   sess,
                                                                   outdir,
                                                                   output,
                                                                   opt_crate,
                                                                   tcx.map.krate(),
                                                                   &analysis,
                                                                   tcx,
                                                                   &crate_name);
                (control.after_analysis.callback)(&mut state);

                if control.after_analysis.stop == Compilation::Stop {
                    return result.and_then(|_| Err(0usize));
                }
            }

            result?;

            if log_enabled!(::log::INFO) {
                println!("Pre-trans");
                tcx.print_debug_stats();
            }
            let trans = phase_4_translate_to_llvm(tcx, analysis, &incremental_hashes_map);

            if log_enabled!(::log::INFO) {
                println!("Post-trans");
                tcx.print_debug_stats();
            }

            Ok((outputs, trans))
        })??
    };

    if sess.opts.debugging_opts.print_type_sizes {
        sess.code_stats.borrow().print_type_sizes();
    }

    let phase5_result = phase_5_run_llvm_passes(sess, &trans, &outputs);

    controller_entry_point!(after_llvm,
                            sess,
                            CompileState::state_after_llvm(input, sess, outdir, output, &trans),
                            phase5_result);
    phase5_result?;

    write::cleanup_llvm(&trans);

    phase_6_link_output(sess, &trans, &outputs);

    // Now that we won't touch anything in the incremental compilation directory
    // any more, we can finalize it (which involves renaming it)
    rustc_incremental::finalize_session_directory(sess, trans.link.crate_hash);

    if sess.opts.debugging_opts.perf_stats {
        sess.print_perf_stats();
    }

    controller_entry_point!(compilation_done,
                            sess,
                            CompileState::state_when_compilation_done(input, sess, outdir, output),
                            Ok(()));

    Ok(())
}

fn keep_hygiene_data(sess: &Session) -> bool {
    sess.opts.debugging_opts.keep_hygiene_data
}

fn keep_ast(sess: &Session) -> bool {
    sess.opts.debugging_opts.keep_ast ||
    sess.opts.debugging_opts.save_analysis ||
    sess.opts.debugging_opts.save_analysis_csv ||
    sess.opts.debugging_opts.save_analysis_api
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

/// CompileController is used to customise compilation, it allows compilation to
/// be stopped and/or to call arbitrary code at various points in compilation.
/// It also allows for various flags to be set to influence what information gets
/// collected during compilation.
///
/// This is a somewhat higher level controller than a Session - the Session
/// controls what happens in each phase, whereas the CompileController controls
/// whether a phase is run at all and whether other code (from outside the
/// the compiler) is run between phases.
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

    pub make_glob_map: MakeGlobMap,
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
    pub cstore: Option<&'a CStore>,
    pub crate_name: Option<&'a str>,
    pub output_filenames: Option<&'a OutputFilenames>,
    pub out_dir: Option<&'a Path>,
    pub out_file: Option<&'a Path>,
    pub arena: Option<&'tcx DroplessArena>,
    pub arenas: Option<&'tcx GlobalArenas<'tcx>>,
    pub expanded_crate: Option<&'a ast::Crate>,
    pub hir_crate: Option<&'a hir::Crate>,
    pub ast_map: Option<&'a hir_map::Map<'tcx>>,
    pub resolutions: Option<&'a Resolutions>,
    pub analysis: Option<&'a ty::CrateAnalysis<'tcx>>,
    pub tcx: Option<TyCtxt<'a, 'tcx, 'tcx>>,
    pub trans: Option<&'a trans::CrateTranslation>,
}

impl<'a, 'tcx> CompileState<'a, 'tcx> {
    fn empty(input: &'a Input,
             session: &'tcx Session,
             out_dir: &'a Option<PathBuf>)
             -> Self {
        CompileState {
            input: input,
            session: session,
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
            ast_map: None,
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
                         cstore: &'a CStore)
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
                          cstore: &'a CStore,
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
                                cstore: &'a CStore,
                                hir_map: &'a hir_map::Map<'tcx>,
                                analysis: &'a ty::CrateAnalysis<'static>,
                                resolutions: &'a Resolutions,
                                krate: &'a ast::Crate,
                                hir_crate: &'a hir::Crate,
                                crate_name: &'a str)
                                -> Self {
        CompileState {
            crate_name: Some(crate_name),
            arena: Some(arena),
            arenas: Some(arenas),
            cstore: Some(cstore),
            ast_map: Some(hir_map),
            analysis: Some(analysis),
            resolutions: Some(resolutions),
            expanded_crate: Some(krate),
            hir_crate: Some(hir_crate),
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
                            analysis: &'a ty::CrateAnalysis<'tcx>,
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

pub fn phase_1_parse_input<'a>(sess: &'a Session, input: &Input) -> PResult<'a, ast::Crate> {
    let continue_after_error = sess.opts.debugging_opts.continue_parse_after_error;
    sess.diagnostic().set_continue_after_error(continue_after_error);

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
    pub analysis: ty::CrateAnalysis<'static>,
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
                                       -> Result<ExpansionResult, usize>
    where F: FnOnce(&ast::Crate) -> CompileResult,
{
    let time_passes = sess.time_passes();

    let (mut krate, features) = syntax::config::features(krate, &sess.parse_sess, sess.opts.test);
    // these need to be set "early" so that expansion sees `quote` if enabled.
    *sess.features.borrow_mut() = features;

    *sess.crate_types.borrow_mut() = collect_crate_types(sess, &krate.attrs);
    *sess.crate_disambiguator.borrow_mut() = Symbol::intern(&compute_crate_disambiguator(sess));

    time(time_passes, "recursion limit", || {
        middle::recursion_limit::update_limits(sess, &krate);
    });

    krate = time(time_passes, "crate injection", || {
        let alt_std_name = sess.opts.alt_std_name.clone();
        syntax::std_inject::maybe_inject_crates_ref(&sess.parse_sess, krate, alt_std_name)
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
                   llvm_passes, attributes, mir_passes, .. } = registry;

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
        sess.mir_passes.borrow_mut().extend(mir_passes);
        *sess.plugin_attributes.borrow_mut() = attributes.clone();
    })?;

    // Lint plugins are registered; now we can process command line flags.
    if sess.opts.describe_lints {
        super::describe_lints(&sess.lint_store.borrow(), true);
        return Err(0);
    }
    sess.track_errors(|| sess.lint_store.borrow_mut().process_command_line(sess))?;

    // Currently, we ignore the name resolution data structures for the purposes of dependency
    // tracking. Instead we will run name resolution and include its output in the hash of each
    // item, much like we do for macro expansion. In other words, the hash reflects not just
    // its contents but the results of name resolution on those contents. Hopefully we'll push
    // this back at some point.
    let _ignore = sess.dep_graph.in_ignore();
    let mut crate_loader = CrateLoader::new(sess, &cstore, crate_name);
    crate_loader.preprocess(&krate);
    let resolver_arenas = Resolver::arenas();
    let mut resolver =
        Resolver::new(sess, &krate, make_glob_map, &mut crate_loader, &resolver_arenas);
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

        if ecx.parse_sess.span_diagnostic.err_count() - ecx.resolve_err_count > err_count {
            ecx.parse_sess.span_diagnostic.abort_if_errors();
        }
        if cfg!(windows) {
            env::set_var("PATH", &old_path);
        }
        krate
    });

    krate.exported_macros = mem::replace(&mut resolver.exported_macros, Vec::new());

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

    time(sess.time_passes(),
         "early lint checks",
         || lint::check_ast_crate(sess, &krate));

    time(sess.time_passes(),
         "AST validation",
         || ast_validation::check_crate(sess, &krate));

    time(sess.time_passes(), "name resolution", || -> CompileResult {
        // Since import resolution will eventually happen in expansion,
        // don't perform `after_expand` until after import resolution.
        after_expand(&krate)?;

        resolver.resolve_crate(&krate);
        Ok(())
    })?;

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
    let hir_forest = time(sess.time_passes(), "lowering ast -> hir", || {
        let hir_crate = lower_crate(sess, &krate, &mut resolver);

        if sess.opts.debugging_opts.hir_stats {
            hir_stats::print_hir_stats(&hir_crate);
        }

        hir_map::Forest::new(hir_crate, &sess.dep_graph)
    });

    // Discard hygiene data, which isn't required past lowering to HIR.
    if !keep_hygiene_data(sess) {
        syntax::ext::hygiene::reset_hygiene_data();
    }

    Ok(ExpansionResult {
        expanded_crate: krate,
        defs: resolver.definitions,
        analysis: ty::CrateAnalysis {
            export_map: resolver.export_map,
            access_levels: AccessLevels::default(),
            reachable: NodeSet(),
            name: crate_name.to_string(),
            glob_map: if resolver.make_glob_map { Some(resolver.glob_map) } else { None },
            hir_ty_to_ty: NodeMap(),
        },
        resolutions: Resolutions {
            freevars: resolver.freevars,
            trait_map: resolver.trait_map,
            maybe_unused_trait_imports: resolver.maybe_unused_trait_imports,
        },
        hir_forest: hir_forest
    })
}

/// Run the resolution, typechecking, region checking and other
/// miscellaneous analysis passes on the crate. Return various
/// structures carrying the results of the analysis.
pub fn phase_3_run_analysis_passes<'tcx, F, R>(sess: &'tcx Session,
                                               hir_map: hir_map::Map<'tcx>,
                                               mut analysis: ty::CrateAnalysis<'tcx>,
                                               resolutions: Resolutions,
                                               arena: &'tcx DroplessArena,
                                               arenas: &'tcx GlobalArenas<'tcx>,
                                               name: &str,
                                               f: F)
                                               -> Result<R, usize>
    where F: for<'a> FnOnce(TyCtxt<'a, 'tcx, 'tcx>,
                            ty::CrateAnalysis<'tcx>,
                            IncrementalHashesMap,
                            CompileResult) -> R
{
    macro_rules! try_with_f {
        ($e: expr, ($t: expr, $a: expr, $h: expr)) => {
            match $e {
                Ok(x) => x,
                Err(x) => {
                    f($t, $a, $h, Err(x));
                    return Err(x);
                }
            }
        }
    }

    let time_passes = sess.time_passes();

    let lang_items = time(time_passes, "language item collection", || {
        sess.track_errors(|| {
            middle::lang_items::collect_language_items(&sess, &hir_map)
        })
    })?;

    let named_region_map = time(time_passes,
                                "lifetime resolution",
                                || middle::resolve_lifetime::krate(sess, &hir_map))?;

    time(time_passes,
         "looking for entry point",
         || middle::entry::find_entry_point(sess, &hir_map));

    sess.plugin_registrar_fn.set(time(time_passes, "looking for plugin registrar", || {
        plugin::build::find_plugin_registrar(sess.diagnostic(), &hir_map)
    }));
    sess.derive_registrar_fn.set(derive_registrar::find(&hir_map));

    let region_map = time(time_passes,
                          "region resolution",
                          || middle::region::resolve_crate(sess, &hir_map));

    time(time_passes,
         "loop checking",
         || loops::check_crate(sess, &hir_map));

    time(time_passes,
              "static item recursion checking",
              || static_recursion::check_crate(sess, &hir_map))?;

    let index = stability::Index::new(&hir_map);

    TyCtxt::create_and_enter(sess,
                             arenas,
                             arena,
                             resolutions,
                             named_region_map,
                             hir_map,
                             region_map,
                             lang_items,
                             index,
                             name,
                             |tcx| {
        let incremental_hashes_map =
            time(time_passes,
                 "compute_incremental_hashes_map",
                 || rustc_incremental::compute_incremental_hashes_map(tcx));
        time(time_passes,
             "load_dep_graph",
             || rustc_incremental::load_dep_graph(tcx, &incremental_hashes_map));

        time(time_passes, "stability index", || {
            tcx.stability.borrow_mut().build(tcx)
        });

        time(time_passes,
             "stability checking",
             || stability::check_unstable_api_usage(tcx));

        // passes are timed inside typeck
        analysis.hir_ty_to_ty =
            try_with_f!(typeck::check_crate(tcx), (tcx, analysis, incremental_hashes_map));

        time(time_passes,
             "const checking",
             || consts::check_crate(tcx));

        analysis.access_levels =
            time(time_passes, "privacy checking", || {
                rustc_privacy::check_crate(tcx, &analysis.export_map)
            });

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
             "rvalue checking",
             || rvalues::check_crate(tcx));

        time(time_passes,
             "MIR dump",
             || mir::mir_map::build_mir_for_crate(tcx));

        if sess.opts.debugging_opts.mir_stats {
            mir_stats::print_mir_stats(tcx, "PRE CLEANUP MIR STATS");
        }

        time(time_passes, "MIR cleanup and validation", || {
            let mut passes = sess.mir_passes.borrow_mut();
            // Push all the built-in validation passes.
            // NB: if you’re adding an *optimisation* it ought to go to another set of passes
            // in stage 4 below.
            passes.push_hook(box mir::transform::dump_mir::DumpMir);
            passes.push_pass(box mir::transform::simplify::SimplifyCfg::new("initial"));
            passes.push_pass(
                box mir::transform::qualify_consts::QualifyAndPromoteConstants::default());
            passes.push_pass(box mir::transform::type_check::TypeckMir);
            passes.push_pass(
                box mir::transform::simplify_branches::SimplifyBranches::new("initial"));
            passes.push_pass(box mir::transform::simplify::SimplifyCfg::new("qualify-consts"));
            // And run everything.
            passes.run_passes(tcx);
        });

        time(time_passes,
             "borrow checking",
             || borrowck::check_crate(tcx));

        // Avoid overwhelming user with errors if type checking failed.
        // I'm not sure how helpful this is, to be honest, but it avoids
        // a
        // lot of annoying errors in the compile-fail tests (basically,
        // lint warnings and so on -- kindck used to do this abort, but
        // kindck is gone now). -nmatsakis
        if sess.err_count() > 0 {
            return Ok(f(tcx, analysis, incremental_hashes_map, Err(sess.err_count())));
        }

        analysis.reachable =
            time(time_passes,
                 "reachability checking",
                 || reachable::find_reachable(tcx, &analysis.access_levels));

        time(time_passes, "death checking", || {
            middle::dead::check_crate(tcx, &analysis.access_levels);
        });

        time(time_passes, "unused lib feature checking", || {
            stability::check_unused_or_stable_features(tcx, &analysis.access_levels)
        });

        time(time_passes,
             "lint checking",
             || lint::check_crate(tcx, &analysis.access_levels));

        // The above three passes generate errors w/o aborting
        if sess.err_count() > 0 {
            return Ok(f(tcx, analysis, incremental_hashes_map, Err(sess.err_count())));
        }

        Ok(f(tcx, analysis, incremental_hashes_map, Ok(())))
    })
}

/// Run the translation phase to LLVM, after which the AST and analysis can
pub fn phase_4_translate_to_llvm<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                           analysis: ty::CrateAnalysis,
                                           incremental_hashes_map: &IncrementalHashesMap)
                                           -> trans::CrateTranslation {
    let time_passes = tcx.sess.time_passes();

    time(time_passes,
         "resolving dependency formats",
         || dependency_format::calculate(&tcx.sess));

    if tcx.sess.opts.debugging_opts.mir_stats {
        mir_stats::print_mir_stats(tcx, "PRE OPTIMISATION MIR STATS");
    }

    // Run the passes that transform the MIR into a more suitable form for translation to LLVM
    // code.
    time(time_passes, "MIR optimisations", || {
        let mut passes = ::rustc::mir::transform::Passes::new();
        passes.push_hook(box mir::transform::dump_mir::DumpMir);
        passes.push_pass(box mir::transform::no_landing_pads::NoLandingPads);
        passes.push_pass(box mir::transform::simplify::SimplifyCfg::new("no-landing-pads"));

        // From here on out, regions are gone.
        passes.push_pass(box mir::transform::erase_regions::EraseRegions);

        passes.push_pass(box mir::transform::add_call_guards::AddCallGuards);
        passes.push_pass(box borrowck::ElaborateDrops);
        passes.push_pass(box mir::transform::no_landing_pads::NoLandingPads);
        passes.push_pass(box mir::transform::simplify::SimplifyCfg::new("elaborate-drops"));

        // No lifetime analysis based on borrowing can be done from here on out.
        passes.push_pass(box mir::transform::instcombine::InstCombine::new());
        passes.push_pass(box mir::transform::deaggregator::Deaggregator);
        passes.push_pass(box mir::transform::copy_prop::CopyPropagation);

        passes.push_pass(box mir::transform::simplify::SimplifyLocals);
        passes.push_pass(box mir::transform::add_call_guards::AddCallGuards);
        passes.push_pass(box mir::transform::dump_mir::Marker("PreTrans"));

        passes.run_passes(tcx);
    });

    if tcx.sess.opts.debugging_opts.mir_stats {
        mir_stats::print_mir_stats(tcx, "POST OPTIMISATION MIR STATS");
    }

    let translation =
        time(time_passes,
             "translation",
             move || trans::trans_crate(tcx, analysis, &incremental_hashes_map));

    time(time_passes,
         "assert dep graph",
         || rustc_incremental::assert_dep_graph(tcx));

    time(time_passes,
         "serialize dep graph",
         || rustc_incremental::save_dep_graph(tcx,
                                              &incremental_hashes_map,
                                              translation.link.crate_hash));
    translation
}

/// Run LLVM itself, producing a bitcode file, assembly file or object file
/// as a side effect.
pub fn phase_5_run_llvm_passes(sess: &Session,
                               trans: &trans::CrateTranslation,
                               outputs: &OutputFilenames) -> CompileResult {
    if sess.opts.cg.no_integrated_as ||
        (sess.target.target.options.no_integrated_as &&
         (outputs.outputs.contains_key(&OutputType::Object) ||
          outputs.outputs.contains_key(&OutputType::Exe)))
    {
        let output_types = OutputTypes::new(&[(OutputType::Assembly, None)]);
        time(sess.time_passes(),
             "LLVM passes",
             || write::run_passes(sess, trans, &output_types, outputs));

        write::run_assembler(sess, outputs);

        // HACK the linker expects the object file to be named foo.0.o but
        // `run_assembler` produces an object named just foo.o. Rename it if we
        // are going to build an executable
        if sess.opts.output_types.contains_key(&OutputType::Exe) {
            let f = outputs.path(OutputType::Object);
            fs::copy(&f,
                     f.with_file_name(format!("{}.0.o",
                                              f.file_stem().unwrap().to_string_lossy()))).unwrap();
            fs::remove_file(f).unwrap();
        }

        // Remove assembly source, unless --save-temps was specified
        if !sess.opts.cg.save_temps {
            fs::remove_file(&outputs.temp_path(OutputType::Assembly, None)).unwrap();
        }
    } else {
        time(sess.time_passes(),
             "LLVM passes",
             || write::run_passes(sess, trans, &sess.opts.output_types, outputs));
    }

    time(sess.time_passes(),
         "serialize work products",
         move || rustc_incremental::save_work_products(sess));

    if sess.err_count() > 0 {
        Err(sess.err_count())
    } else {
        Ok(())
    }
}

/// Run the linker on any artifacts that resulted from the LLVM run.
/// This should produce either a finished executable or library.
pub fn phase_6_link_output(sess: &Session,
                           trans: &trans::CrateTranslation,
                           outputs: &OutputFilenames) {
    time(sess.time_passes(),
         "linking",
         || link::link_binary(sess, trans, outputs, &trans.link.crate_name.as_str()));
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
                    let p = link::filename_for_input(sess, *output, crate_name, outputs);
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
                                         .files
                                         .borrow()
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
                             session.add_lint(lint::builtin::UNKNOWN_CRATE_TYPES,
                                              ast::CRATE_NODE_ID,
                                              a.span,
                                              "invalid `crate_type` value".to_string());
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
            base.push(link::default_output_for_target(session));
        }
        base.sort();
        base.dedup();
    }

    base.into_iter()
        .filter(|crate_type| {
            let res = !link::invalid_output_for_target(session, *crate_type);

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
            let ofile = if unnamed_output_types > 1 &&
                            sess.opts.output_types.contains_key(&OutputType::Exe) {
                sess.warn("ignoring specified output filename for 'link' output because multiple \
                           outputs were requested");
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
