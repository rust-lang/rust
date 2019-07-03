use crate::interface::{Compiler, Result};
use crate::util;
use crate::proc_macro_decls;

use log::{debug, info, warn, log_enabled};
use rustc::dep_graph::DepGraph;
use rustc::hir;
use rustc::hir::lowering::lower_crate;
use rustc::hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc::lint;
use rustc::middle::{self, reachable, resolve_lifetime, stability};
use rustc::middle::privacy::AccessLevels;
use rustc::ty::{self, AllArenas, Resolutions, TyCtxt, GlobalCtxt};
use rustc::ty::steal::Steal;
use rustc::traits;
use rustc::util::common::{time, ErrorReported};
use rustc::util::profiling::ProfileCategory;
use rustc::session::{CompileResult, CrateDisambiguator, Session};
use rustc::session::config::{self, CrateType, Input, OutputFilenames, OutputType};
use rustc::session::search_paths::PathKind;
use rustc_allocator as allocator;
use rustc_borrowck as borrowck;
use rustc_codegen_ssa::back::link::emit_metadata;
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc_codegen_utils::link::filename_for_metadata;
use rustc_data_structures::{box_region_allow_access, declare_box_region_type, parallel};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_data_structures::sync::{Lrc, ParallelIterator, par_iter};
use rustc_incremental;
use rustc_incremental::DepGraphFuture;
use rustc_metadata::creader::CrateLoader;
use rustc_metadata::cstore::{self, CStore};
use rustc_mir as mir;
use rustc_passes::{self, ast_validation, hir_stats, loops, rvalue_promotion, layout_test};
use rustc_plugin as plugin;
use rustc_plugin::registry::Registry;
use rustc_privacy;
use rustc_resolve::{Resolver, ResolverArenas};
use rustc_traits;
use rustc_typeck as typeck;
use syntax::{self, ast, attr, diagnostics, visit};
use syntax::early_buffered_lints::BufferedEarlyLint;
use syntax::ext::base::{NamedSyntaxExtension, ExtCtxt};
use syntax::mut_visit::MutVisitor;
use syntax::parse::{self, PResult};
use syntax::util::node_count::NodeCounter;
use syntax::util::lev_distance::find_best_match_for_name;
use syntax::symbol::Symbol;
use syntax::feature_gate::AttributeType;
use syntax_pos::{FileName, edition::Edition, hygiene};
use syntax_ext;

use serialize::json;
use tempfile::Builder as TempFileBuilder;

use std::any::Any;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Write};
use std::iter;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::cell::RefCell;
use std::rc::Rc;
use std::mem;
use std::ops::Generator;

pub fn parse<'a>(sess: &'a Session, input: &Input) -> PResult<'a, ast::Crate> {
    sess.diagnostic()
        .set_continue_after_error(sess.opts.debugging_opts.continue_parse_after_error);
    sess.profiler(|p| p.start_activity("parsing"));
    let krate = time(sess, "parsing", || match *input {
        Input::File(ref file) => parse::parse_crate_from_file(file, &sess.parse_sess),
        Input::Str {
            ref input,
            ref name,
        } => parse::parse_crate_from_source_str(name.clone(), input.clone(), &sess.parse_sess),
    })?;
    sess.profiler(|p| p.end_activity("parsing"));

    sess.diagnostic().set_continue_after_error(true);

    if sess.opts.debugging_opts.ast_json_noexpand {
        println!("{}", json::as_json(&krate));
    }

    if sess.opts.debugging_opts.input_stats {
        println!(
            "Lines of code:             {}",
            sess.source_map().count_lines()
        );
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

declare_box_region_type!(
    pub BoxedResolver,
    for(),
    (&mut Resolver<'_>) -> (Result<ast::Crate>, ExpansionResult)
);

/// Runs the "early phases" of the compiler: initial `cfg` processing,
/// loading compiler plugins (including those from `addl_plugins`),
/// syntax expansion, secondary `cfg` expansion, synthesis of a test
/// harness if one is to be provided, injection of a dependency on the
/// standard library and prelude, and name resolution.
///
/// Returns `None` if we're aborting after handling -W help.
pub fn configure_and_expand(
    sess: Lrc<Session>,
    cstore: Lrc<CStore>,
    krate: ast::Crate,
    crate_name: &str,
    plugin_info: PluginInfo,
) -> Result<(ast::Crate, BoxedResolver)> {
    // Currently, we ignore the name resolution data structures for the purposes of dependency
    // tracking. Instead we will run name resolution and include its output in the hash of each
    // item, much like we do for macro expansion. In other words, the hash reflects not just
    // its contents but the results of name resolution on those contents. Hopefully we'll push
    // this back at some point.
    let crate_name = crate_name.to_string();
    let (result, resolver) = BoxedResolver::new(static move || {
        let sess = &*sess;
        let mut crate_loader = CrateLoader::new(sess, &*cstore, &crate_name);
        let resolver_arenas = Resolver::arenas();
        let res = configure_and_expand_inner(
            sess,
            &*cstore,
            krate,
            &crate_name,
            &resolver_arenas,
            &mut crate_loader,
            plugin_info,
        );
        let mut resolver = match res {
            Err(v) => {
                yield BoxedResolver::initial_yield(Err(v));
                panic!()
            }
            Ok((krate, resolver)) => {
                yield BoxedResolver::initial_yield(Ok(krate));
                resolver
            }
        };
        box_region_allow_access!(for(), (&mut Resolver<'_>), (&mut resolver));
        ExpansionResult::from_owned_resolver(resolver)
    });
    result.map(|k| (k, resolver))
}

pub struct ExpansionResult {
    pub defs: Steal<hir::map::Definitions>,
    pub resolutions: Steal<Resolutions>,
}

impl ExpansionResult {
    fn from_owned_resolver(
        resolver: Resolver<'_>,
    ) -> Self {
        ExpansionResult {
            defs: Steal::new(resolver.definitions),
            resolutions: Steal::new(Resolutions {
                export_map: resolver.export_map,
                trait_map: resolver.trait_map,
                glob_map: resolver.glob_map,
                maybe_unused_trait_imports: resolver.maybe_unused_trait_imports,
                maybe_unused_extern_crates: resolver.maybe_unused_extern_crates,
                extern_prelude: resolver.extern_prelude.iter().map(|(ident, entry)| {
                    (ident.name, entry.introduced_by_item)
                }).collect(),
            }),
        }
    }

    pub fn from_resolver_ref(
        resolver: &Resolver<'_>,
    ) -> Self {
        ExpansionResult {
            defs: Steal::new(resolver.definitions.clone()),
            resolutions: Steal::new(Resolutions {
                export_map: resolver.export_map.clone(),
                trait_map: resolver.trait_map.clone(),
                glob_map: resolver.glob_map.clone(),
                maybe_unused_trait_imports: resolver.maybe_unused_trait_imports.clone(),
                maybe_unused_extern_crates: resolver.maybe_unused_extern_crates.clone(),
                extern_prelude: resolver.extern_prelude.iter().map(|(ident, entry)| {
                    (ident.name, entry.introduced_by_item)
                }).collect(),
            }),
        }
    }
}

impl BoxedResolver {
    pub fn to_expansion_result(
        mut resolver: Rc<Option<RefCell<BoxedResolver>>>,
    ) -> ExpansionResult {
        if let Some(resolver) = Rc::get_mut(&mut resolver) {
            mem::replace(resolver, None).unwrap().into_inner().complete()
        } else {
            let resolver = &*resolver;
            resolver.as_ref().unwrap().borrow_mut().access(|resolver| {
                ExpansionResult::from_resolver_ref(resolver)
            })
        }
    }
}

pub struct PluginInfo {
    syntax_exts: Vec<NamedSyntaxExtension>,
    attributes: Vec<(Symbol, AttributeType)>,
}

pub fn register_plugins<'a>(
    compiler: &Compiler,
    sess: &'a Session,
    cstore: &'a CStore,
    mut krate: ast::Crate,
    crate_name: &str,
) -> Result<(ast::Crate, PluginInfo)> {
    krate = time(sess, "attributes injection", || {
        syntax::attr::inject(krate, &sess.parse_sess, &sess.opts.debugging_opts.crate_attr)
    });

    let (mut krate, features) = syntax::config::features(
        krate,
        &sess.parse_sess,
        sess.edition(),
        &sess.opts.debugging_opts.allow_features,
    );
    // these need to be set "early" so that expansion sees `quote` if enabled.
    sess.init_features(features);

    let crate_types = util::collect_crate_types(sess, &krate.attrs);
    sess.crate_types.set(crate_types);

    let disambiguator = util::compute_crate_disambiguator(sess);
    sess.crate_disambiguator.set(disambiguator);
    rustc_incremental::prepare_session_directory(sess, &crate_name, disambiguator);

    if sess.opts.incremental.is_some() {
        time(sess, "garbage collect incremental cache directory", || {
            if let Err(e) = rustc_incremental::garbage_collect_session_directories(sess) {
                warn!(
                    "Error while trying to garbage collect incremental \
                     compilation cache directory: {}",
                    e
                );
            }
        });
    }

    // If necessary, compute the dependency graph (in the background).
    compiler.dep_graph_future().ok();

    time(sess, "recursion limit", || {
        middle::recursion_limit::update_limits(sess, &krate);
    });

    krate = time(sess, "crate injection", || {
        let alt_std_name = sess.opts.alt_std_name.as_ref().map(|s| &**s);
        syntax::std_inject::maybe_inject_crates_ref(krate, alt_std_name, sess.edition())
    });

    let registrars = time(sess, "plugin loading", || {
        plugin::load::load_plugins(
            sess,
            &cstore,
            &krate,
            crate_name,
            Some(sess.opts.debugging_opts.extra_plugins.clone()),
        )
    });

    let mut registry = Registry::new(sess, krate.span);

    time(sess, "plugin registration", || {
        if sess.features_untracked().rustc_diagnostic_macros {
            registry.register_macro(
                "__diagnostic_used",
                diagnostics::plugin::expand_diagnostic_used,
            );
            registry.register_macro(
                "__register_diagnostic",
                diagnostics::plugin::expand_register_diagnostic,
            );
            registry.register_macro(
                "__build_diagnostic_array",
                diagnostics::plugin::expand_build_diagnostic_array,
            );
        }

        for registrar in registrars {
            registry.args_hidden = Some(registrar.args);
            (registrar.fun)(&mut registry);
        }
    });

    let Registry {
        syntax_exts,
        early_lint_passes,
        late_lint_passes,
        lint_groups,
        llvm_passes,
        attributes,
        ..
    } = registry;

    let mut ls = sess.lint_store.borrow_mut();
    for pass in early_lint_passes {
        ls.register_early_pass(Some(sess), true, false, pass);
    }
    for pass in late_lint_passes {
        ls.register_late_pass(Some(sess), true, false, false, pass);
    }

    for (name, (to, deprecated_name)) in lint_groups {
        ls.register_group(Some(sess), true, name, deprecated_name, to);
    }

    *sess.plugin_llvm_passes.borrow_mut() = llvm_passes;
    *sess.plugin_attributes.borrow_mut() = attributes.clone();

    Ok((krate, PluginInfo {
        syntax_exts,
        attributes,
    }))
}

fn configure_and_expand_inner<'a>(
    sess: &'a Session,
    cstore: &'a CStore,
    mut krate: ast::Crate,
    crate_name: &str,
    resolver_arenas: &'a ResolverArenas<'a>,
    crate_loader: &'a mut CrateLoader<'a>,
    plugin_info: PluginInfo,
) -> Result<(ast::Crate, Resolver<'a>)> {
    let attributes = plugin_info.attributes;
    time(sess, "pre ast expansion lint checks", || {
        lint::check_ast_crate(
            sess,
            &krate,
            true,
            rustc_lint::BuiltinCombinedPreExpansionLintPass::new());
    });

    let mut resolver = Resolver::new(
        sess,
        cstore,
        &krate,
        crate_name,
        crate_loader,
        &resolver_arenas,
    );
    syntax_ext::register_builtins(&mut resolver, plugin_info.syntax_exts, sess.edition());

    // Expand all macros
    sess.profiler(|p| p.start_activity("macro expansion"));
    krate = time(sess, "expansion", || {
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
            let mut new_path = sess.host_filesearch(PathKind::All).search_path_dirs();
            for path in env::split_paths(&old_path) {
                if !new_path.contains(&path) {
                    new_path.push(path);
                }
            }
            env::set_var(
                "PATH",
                &env::join_paths(
                    new_path
                        .iter()
                        .filter(|p| env::join_paths(iter::once(p)).is_ok()),
                ).unwrap(),
            );
        }

        // Create the config for macro expansion
        let features = sess.features_untracked();
        let cfg = syntax::ext::expand::ExpansionConfig {
            features: Some(&features),
            recursion_limit: *sess.recursion_limit.get(),
            trace_mac: sess.opts.debugging_opts.trace_macros,
            should_test: sess.opts.test,
            ..syntax::ext::expand::ExpansionConfig::default(crate_name.to_string())
        };

        let mut ecx = ExtCtxt::new(&sess.parse_sess, cfg, &mut resolver);

        // Expand macros now!
        let krate = time(sess, "expand crate", || {
            ecx.monotonic_expander().expand_crate(krate)
        });

        // The rest is error reporting

        time(sess, "check unused macros", || {
            ecx.check_unused_macros();
        });

        let mut missing_fragment_specifiers: Vec<_> = ecx.parse_sess
            .missing_fragment_specifiers
            .borrow()
            .iter()
            .cloned()
            .collect();
        missing_fragment_specifiers.sort();

        for span in missing_fragment_specifiers {
            let lint = lint::builtin::MISSING_FRAGMENT_SPECIFIER;
            let msg = "missing fragment specifier";
            sess.buffer_lint(lint, ast::CRATE_NODE_ID, span, msg);
        }
        if cfg!(windows) {
            env::set_var("PATH", &old_path);
        }
        krate
    });
    sess.profiler(|p| p.end_activity("macro expansion"));

    time(sess, "maybe building test harness", || {
        syntax::test::modify_for_testing(
            &sess.parse_sess,
            &mut resolver,
            sess.opts.test,
            &mut krate,
            sess.diagnostic(),
            &sess.features_untracked(),
        )
    });

    // If we're actually rustdoc then there's no need to actually compile
    // anything, so switch everything to just looping
    if sess.opts.actually_rustdoc {
        util::ReplaceBodyWithLoop::new(sess).visit_crate(&mut krate);
    }

    let (has_proc_macro_decls, has_global_allocator) = time(sess, "AST validation", || {
        ast_validation::check_crate(sess, &krate)
    });

    // If we're in rustdoc we're always compiling as an rlib, but that'll trip a
    // bunch of checks in the `modify` function below. For now just skip this
    // step entirely if we're rustdoc as it's not too useful anyway.
    if !sess.opts.actually_rustdoc {
        krate = time(sess, "maybe creating a macro crate", || {
            let crate_types = sess.crate_types.borrow();
            let num_crate_types = crate_types.len();
            let is_proc_macro_crate = crate_types.contains(&config::CrateType::ProcMacro);
            let is_test_crate = sess.opts.test;
            syntax_ext::proc_macro_decls::modify(
                &sess.parse_sess,
                &mut resolver,
                krate,
                is_proc_macro_crate,
                has_proc_macro_decls,
                is_test_crate,
                num_crate_types,
                sess.diagnostic(),
            )
        });
    }

    if has_global_allocator {
        // Expand global allocators, which are treated as an in-tree proc macro
        time(sess, "creating allocators", || {
            allocator::expand::modify(
                &sess.parse_sess,
                &mut resolver,
                &mut krate,
                crate_name.to_string(),
                sess.diagnostic(),
            )
        });
    }

    // Done with macro expansion!

    if sess.opts.debugging_opts.input_stats {
        println!("Post-expansion node count: {}", count_nodes(&krate));
    }

    if sess.opts.debugging_opts.hir_stats {
        hir_stats::print_ast_stats(&krate, "POST EXPANSION AST STATS");
    }

    if sess.opts.debugging_opts.ast_json {
        println!("{}", json::as_json(&krate));
    }

    time(sess, "name resolution", || {
        resolver.resolve_crate(&krate);
    });

    // Needs to go *after* expansion to be able to check the results of macro expansion.
    time(sess, "complete gated feature checking", || {
        syntax::feature_gate::check_crate(
            &krate,
            &sess.parse_sess,
            &sess.features_untracked(),
            &attributes,
            sess.opts.unstable_features,
        );
    });

    // Add all buffered lints from the `ParseSess` to the `Session`.
    sess.parse_sess.buffered_lints.with_lock(|buffered_lints| {
        info!("{} parse sess buffered_lints", buffered_lints.len());
        for BufferedEarlyLint{id, span, msg, lint_id} in buffered_lints.drain(..) {
            let lint = lint::Lint::from_parser_lint_id(lint_id);
            sess.buffer_lint(lint, id, span, &msg);
        }
    });

    Ok((krate, resolver))
}

pub fn lower_to_hir(
    sess: &Session,
    cstore: &CStore,
    resolver: &mut Resolver<'_>,
    dep_graph: &DepGraph,
    krate: &ast::Crate,
) -> Result<hir::map::Forest> {
    // Lower ast -> hir
    let hir_forest = time(sess, "lowering ast -> hir", || {
        let hir_crate = lower_crate(sess, cstore, &dep_graph, &krate, resolver);

        if sess.opts.debugging_opts.hir_stats {
            hir_stats::print_hir_stats(&hir_crate);
        }

        hir::map::Forest::new(hir_crate, &dep_graph)
    });

    time(sess, "early lint checks", || {
        lint::check_ast_crate(sess, &krate, false, rustc_lint::BuiltinCombinedEarlyLintPass::new())
    });

    // Discard hygiene data, which isn't required after lowering to HIR.
    if !sess.opts.debugging_opts.keep_hygiene_data {
        syntax::ext::hygiene::clear_markings();
    }

    Ok(hir_forest)
}

// Returns all the paths that correspond to generated files.
fn generated_output_paths(
    sess: &Session,
    outputs: &OutputFilenames,
    exact_name: bool,
    crate_name: &str,
) -> Vec<PathBuf> {
    let mut out_filenames = Vec::new();
    for output_type in sess.opts.output_types.keys() {
        let file = outputs.path(*output_type);
        match *output_type {
            // If the filename has been overridden using `-o`, it will not be modified
            // by appending `.rlib`, `.exe`, etc., so we can skip this transformation.
            OutputType::Exe if !exact_name => for crate_type in sess.crate_types.borrow().iter() {
                let p = ::rustc_codegen_utils::link::filename_for_input(
                    sess,
                    *crate_type,
                    crate_name,
                    outputs,
                );
                out_filenames.push(p);
            },
            OutputType::DepInfo if sess.opts.debugging_opts.dep_info_omit_d_target => {
                // Don't add the dep-info output when omitting it from dep-info targets
            }
            _ => {
                out_filenames.push(file);
            }
        }
    }
    out_filenames
}

// Runs `f` on every output file path and returns the first non-None result, or None if `f`
// returns None for every file path.
fn check_output<F, T>(output_paths: &[PathBuf], f: F) -> Option<T>
where
    F: Fn(&PathBuf) -> Option<T>,
{
    for output_path in output_paths {
        if let Some(result) = f(output_path) {
            return Some(result);
        }
    }
    None
}

fn output_contains_path(output_paths: &[PathBuf], input_path: &PathBuf) -> bool {
    let input_path = input_path.canonicalize().ok();
    if input_path.is_none() {
        return false;
    }
    let check = |output_path: &PathBuf| {
        if output_path.canonicalize().ok() == input_path {
            Some(())
        } else {
            None
        }
    };
    check_output(output_paths, check).is_some()
}

fn output_conflicts_with_dir(output_paths: &[PathBuf]) -> Option<PathBuf> {
    let check = |output_path: &PathBuf| {
        if output_path.is_dir() {
            Some(output_path.clone())
        } else {
            None
        }
    };
    check_output(output_paths, check)
}

fn escape_dep_filename(filename: &FileName) -> String {
    // Apparently clang and gcc *only* escape spaces:
    // http://llvm.org/klaus/clang/commit/9d50634cfc268ecc9a7250226dd5ca0e945240d4
    filename.to_string().replace(" ", "\\ ")
}

fn write_out_deps(sess: &Session, outputs: &OutputFilenames, out_filenames: &[PathBuf]) {
    // Write out dependency rules to the dep-info file if requested
    if !sess.opts.output_types.contains_key(&OutputType::DepInfo) {
        return;
    }
    let deps_filename = outputs.path(OutputType::DepInfo);

    let result = (|| -> io::Result<()> {
        // Build a list of files used to compile the output and
        // write Makefile-compatible dependency rules
        let files: Vec<String> = sess.source_map()
            .files()
            .iter()
            .filter(|fmap| fmap.is_real_file())
            .filter(|fmap| !fmap.is_imported())
            .map(|fmap| escape_dep_filename(&fmap.name))
            .collect();
        let mut file = fs::File::create(&deps_filename)?;
        for path in out_filenames {
            writeln!(file, "{}: {}\n", path.display(), files.join(" "))?;
        }

        // Emit a fake target for each input file to the compilation. This
        // prevents `make` from spitting out an error if a file is later
        // deleted. For more info see #28735
        for path in files {
            writeln!(file, "{}:", path)?;
        }
        Ok(())
    })();

    if let Err(e) = result {
        sess.fatal(&format!(
            "error writing dependencies to `{}`: {}",
            deps_filename.display(),
            e
        ));
    }
}

pub fn prepare_outputs(
    sess: &Session,
    compiler: &Compiler,
    krate: &ast::Crate,
    crate_name: &str
) -> Result<OutputFilenames> {
    // FIXME: rustdoc passes &[] instead of &krate.attrs here
    let outputs = util::build_output_filenames(
        &compiler.input,
        &compiler.output_dir,
        &compiler.output_file,
        &krate.attrs,
        sess
    );

    let output_paths = generated_output_paths(
        sess,
        &outputs,
        compiler.output_file.is_some(),
        &crate_name,
    );

    // Ensure the source file isn't accidentally overwritten during compilation.
    if let Some(ref input_path) = compiler.input_path {
        if sess.opts.will_create_output_file() {
            if output_contains_path(&output_paths, input_path) {
                sess.err(&format!(
                    "the input file \"{}\" would be overwritten by the generated \
                        executable",
                    input_path.display()
                ));
                return Err(ErrorReported);
            }
            if let Some(dir_path) = output_conflicts_with_dir(&output_paths) {
                sess.err(&format!(
                    "the generated executable for the input file \"{}\" conflicts with the \
                        existing directory \"{}\"",
                    input_path.display(),
                    dir_path.display()
                ));
                return Err(ErrorReported);
            }
        }
    }

    write_out_deps(sess, &outputs, &output_paths);

    let only_dep_info = sess.opts.output_types.contains_key(&OutputType::DepInfo)
        && sess.opts.output_types.len() == 1;

    if !only_dep_info {
        if let Some(ref dir) = compiler.output_dir {
            if fs::create_dir_all(dir).is_err() {
                sess.err("failed to find or create the directory specified by --out-dir");
                return Err(ErrorReported);
            }
        }
    }

    Ok(outputs)
}

pub fn default_provide(providers: &mut ty::query::Providers<'_>) {
    providers.analysis = analysis;
    proc_macro_decls::provide(providers);
    plugin::build::provide(providers);
    hir::provide(providers);
    borrowck::provide(providers);
    mir::provide(providers);
    reachable::provide(providers);
    resolve_lifetime::provide(providers);
    rustc_privacy::provide(providers);
    typeck::provide(providers);
    ty::provide(providers);
    traits::provide(providers);
    stability::provide(providers);
    middle::intrinsicck::provide(providers);
    middle::liveness::provide(providers);
    reachable::provide(providers);
    rustc_passes::provide(providers);
    rustc_traits::provide(providers);
    middle::region::provide(providers);
    middle::entry::provide(providers);
    cstore::provide(providers);
    lint::provide(providers);
    rustc_lint::provide(providers);
}

pub fn default_provide_extern(providers: &mut ty::query::Providers<'_>) {
    cstore::provide_extern(providers);
}

declare_box_region_type!(
    pub BoxedGlobalCtxt,
    for('tcx),
    (&'tcx GlobalCtxt<'tcx>) -> ((), ())
);

impl BoxedGlobalCtxt {
    pub fn enter<F, R>(&mut self, f: F) -> R
    where
        F: for<'tcx> FnOnce(TyCtxt<'tcx>) -> R,
    {
        self.access(|gcx| ty::tls::enter_global(gcx, |tcx| f(tcx)))
    }
}

pub fn create_global_ctxt(
    compiler: &Compiler,
    mut hir_forest: hir::map::Forest,
    defs: hir::map::Definitions,
    resolutions: Resolutions,
    outputs: OutputFilenames,
    tx: mpsc::Sender<Box<dyn Any + Send>>,
    crate_name: &str,
) -> BoxedGlobalCtxt {
    let sess = compiler.session().clone();
    let cstore = compiler.cstore.clone();
    let codegen_backend = compiler.codegen_backend().clone();
    let crate_name = crate_name.to_string();

    let ((), result) = BoxedGlobalCtxt::new(static move || {
        let sess = &*sess;
        let cstore = &*cstore;

        let global_ctxt: Option<GlobalCtxt<'_>>;
        let arenas = AllArenas::new();

        // Construct the HIR map
        let hir_map = time(sess, "indexing hir", || {
            hir::map::map_crate(sess, cstore, &mut hir_forest, &defs)
        });

        let query_result_on_disk_cache = time(sess, "load query result cache", || {
            rustc_incremental::load_query_result_cache(sess)
        });

        let mut local_providers = ty::query::Providers::default();
        default_provide(&mut local_providers);
        codegen_backend.provide(&mut local_providers);

        let mut extern_providers = local_providers;
        default_provide_extern(&mut extern_providers);
        codegen_backend.provide_extern(&mut extern_providers);

        let gcx = TyCtxt::create_global_ctxt(
            sess,
            cstore,
            local_providers,
            extern_providers,
            &arenas,
            resolutions,
            hir_map,
            query_result_on_disk_cache,
            &crate_name,
            tx,
            &outputs
        );

        global_ctxt = Some(gcx);
        let gcx = global_ctxt.as_ref().unwrap();

        ty::tls::enter_global(gcx, |tcx| {
            // Do some initialization of the DepGraph that can only be done with the
            // tcx available.
            time(tcx.sess, "dep graph tcx init", || rustc_incremental::dep_graph_tcx_init(tcx));
        });

        yield BoxedGlobalCtxt::initial_yield(());
        box_region_allow_access!(for('tcx), (&'tcx GlobalCtxt<'tcx>), (gcx));

        if sess.opts.debugging_opts.query_stats {
            gcx.queries.print_stats();
        }
    });

    result
}

/// Runs the resolution, type-checking, region checking and other
/// miscellaneous analysis passes on the crate.
fn analysis(tcx: TyCtxt<'_>, cnum: CrateNum) -> Result<()> {
    assert_eq!(cnum, LOCAL_CRATE);

    let sess = tcx.sess;
    let mut entry_point = None;

    time(sess, "misc checking 1", || {
        parallel!({
            entry_point = time(sess, "looking for entry point", || {
                middle::entry::find_entry_point(tcx)
            });

            time(sess, "looking for plugin registrar", || {
                plugin::build::find_plugin_registrar(tcx)
            });

            time(sess, "looking for derive registrar", || {
                proc_macro_decls::find(tcx)
            });
        }, {
            par_iter(&tcx.hir().krate().modules).for_each(|(&module, _)| {
                tcx.ensure().check_mod_loops(tcx.hir().local_def_id(module));
                tcx.ensure().check_mod_attrs(tcx.hir().local_def_id(module));
                tcx.ensure().check_mod_unstable_api_usage(tcx.hir().local_def_id(module));
            });
        });
    });

    // passes are timed inside typeck
    typeck::check_crate(tcx)?;

    time(sess, "misc checking 2", || {
        parallel!({
            time(sess, "rvalue promotion + match checking", || {
                tcx.par_body_owners(|def_id| {
                    tcx.ensure().const_is_rvalue_promotable_to_static(def_id);
                    tcx.ensure().check_match(def_id);
                });
            });
        }, {
            time(sess, "liveness checking + intrinsic checking", || {
                par_iter(&tcx.hir().krate().modules).for_each(|(&module, _)| {
                    // this must run before MIR dump, because
                    // "not all control paths return a value" is reported here.
                    //
                    // maybe move the check to a MIR pass?
                    tcx.ensure().check_mod_liveness(tcx.hir().local_def_id(module));

                    tcx.ensure().check_mod_intrinsics(tcx.hir().local_def_id(module));
                });
            });
        });
    });

    time(sess, "borrow checking", || {
        if tcx.use_ast_borrowck() {
            borrowck::check_crate(tcx);
        }
    });

    time(sess, "MIR borrow checking", || {
        tcx.par_body_owners(|def_id| tcx.ensure().mir_borrowck(def_id));
    });

    time(sess, "dumping chalk-like clauses", || {
        rustc_traits::lowering::dump_program_clauses(tcx);
    });

    time(sess, "MIR effect checking", || {
        for def_id in tcx.body_owners() {
            mir::transform::check_unsafety::check_unsafety(tcx, def_id)
        }
    });

    time(sess, "layout testing", || layout_test::test_layout(tcx));

    // Avoid overwhelming user with errors if borrow checking failed.
    // I'm not sure how helpful this is, to be honest, but it avoids a
    // lot of annoying errors in the compile-fail tests (basically,
    // lint warnings and so on -- kindck used to do this abort, but
    // kindck is gone now). -nmatsakis
    if sess.has_errors() {
        return Err(ErrorReported);
    }

    time(sess, "misc checking 3", || {
        parallel!({
            time(sess, "privacy access levels", || {
                tcx.ensure().privacy_access_levels(LOCAL_CRATE);
            });
            parallel!({
                time(sess, "private in public", || {
                    tcx.ensure().check_private_in_public(LOCAL_CRATE);
                });
            }, {
                time(sess, "death checking", || middle::dead::check_crate(tcx));
            },  {
                time(sess, "unused lib feature checking", || {
                    stability::check_unused_or_stable_features(tcx)
                });
            }, {
                time(sess, "lint checking", || {
                    lint::check_crate(tcx, || rustc_lint::BuiltinCombinedLateLintPass::new());
                });
            });
        }, {
            time(sess, "privacy checking modules", || {
                par_iter(&tcx.hir().krate().modules).for_each(|(&module, _)| {
                    tcx.ensure().check_mod_privacy(tcx.hir().local_def_id(module));
                });
            });
        });
    });

    Ok(())
}

fn encode_and_write_metadata(
    tcx: TyCtxt<'_>,
    outputs: &OutputFilenames,
) -> (middle::cstore::EncodedMetadata, bool) {
    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    enum MetadataKind {
        None,
        Uncompressed,
        Compressed
    }

    let metadata_kind = tcx.sess.crate_types.borrow().iter().map(|ty| {
        match *ty {
            CrateType::Executable |
            CrateType::Staticlib |
            CrateType::Cdylib => MetadataKind::None,

            CrateType::Rlib => MetadataKind::Uncompressed,

            CrateType::Dylib |
            CrateType::ProcMacro => MetadataKind::Compressed,
        }
    }).max().unwrap_or(MetadataKind::None);

    let metadata = match metadata_kind {
        MetadataKind::None => middle::cstore::EncodedMetadata::new(),
        MetadataKind::Uncompressed |
        MetadataKind::Compressed => tcx.encode_metadata(),
    };

    let need_metadata_file = tcx.sess.opts.output_types.contains_key(&OutputType::Metadata);
    if need_metadata_file {
        let crate_name = &tcx.crate_name(LOCAL_CRATE).as_str();
        let out_filename = filename_for_metadata(tcx.sess, crate_name, outputs);
        // To avoid races with another rustc process scanning the output directory,
        // we need to write the file somewhere else and atomically move it to its
        // final destination, with an `fs::rename` call. In order for the rename to
        // always succeed, the temporary file needs to be on the same filesystem,
        // which is why we create it inside the output directory specifically.
        let metadata_tmpdir = TempFileBuilder::new()
            .prefix("rmeta")
            .tempdir_in(out_filename.parent().unwrap())
            .unwrap_or_else(|err| {
                tcx.sess.fatal(&format!("couldn't create a temp dir: {}", err))
            });
        let metadata_filename = emit_metadata(tcx.sess, &metadata, &metadata_tmpdir);
        if let Err(e) = fs::rename(&metadata_filename, &out_filename) {
            tcx.sess.fatal(&format!("failed to write {}: {}", out_filename.display(), e));
        }
        if tcx.sess.opts.debugging_opts.emit_artifact_notifications {
            tcx.sess.parse_sess.span_diagnostic
                .emit_artifact_notification(&out_filename, "metadata");
        }
    }

    let need_metadata_module = metadata_kind == MetadataKind::Compressed;

    (metadata, need_metadata_module)
}

/// Runs the codegen backend, after which the AST and analysis can
/// be discarded.
pub fn start_codegen<'tcx>(
    codegen_backend: &dyn CodegenBackend,
    tcx: TyCtxt<'tcx>,
    rx: mpsc::Receiver<Box<dyn Any + Send>>,
    outputs: &OutputFilenames,
) -> Box<dyn Any> {
    if log_enabled!(::log::Level::Info) {
        println!("Pre-codegen");
        tcx.print_debug_stats();
    }

    time(tcx.sess, "resolving dependency formats", || {
        middle::dependency_format::calculate(tcx)
    });

    let (metadata, need_metadata_module) = time(tcx.sess, "metadata encoding and writing", || {
        encode_and_write_metadata(tcx, outputs)
    });

    tcx.sess.profiler(|p| p.start_activity("codegen crate"));
    let codegen = time(tcx.sess, "codegen", move || {
        codegen_backend.codegen_crate(tcx, metadata, need_metadata_module, rx)
    });
    tcx.sess.profiler(|p| p.end_activity("codegen crate"));

    if log_enabled!(::log::Level::Info) {
        println!("Post-codegen");
        tcx.print_debug_stats();
    }

    if tcx.sess.opts.output_types.contains_key(&OutputType::Mir) {
        if let Err(e) = mir::transform::dump_mir::emit_mir(tcx, outputs) {
            tcx.sess.err(&format!("could not emit MIR: {}", e));
            tcx.sess.abort_if_errors();
        }
    }

    codegen
}
