use std::any::Any;
use std::ffi::OsString;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::{env, fs, iter};

use rustc_ast::{self as ast, visit};
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_data_structures::parallel;
use rustc_data_structures::steal::Steal;
use rustc_data_structures::sync::{AppendOnlyIndexVec, FreezeLock, Lrc, OnceLock, WorkerLocal};
use rustc_expand::base::{ExtCtxt, LintStoreExpand};
use rustc_feature::Features;
use rustc_fs_util::try_canonicalize;
use rustc_hir::def_id::{StableCrateId, StableCrateIdMap, LOCAL_CRATE};
use rustc_hir::definitions::Definitions;
use rustc_incremental::setup_dep_graph;
use rustc_lint::{unerased_lint_store, BufferedEarlyLint, EarlyCheckNode, LintStore};
use rustc_metadata::creader::CStore;
use rustc_middle::arena::Arena;
use rustc_middle::ty::{self, GlobalCtxt, RegisteredTools, TyCtxt};
use rustc_middle::util::Providers;
use rustc_parse::{
    new_parser_from_file, new_parser_from_source_str, unwrap_or_emit_fatal, validate_attr,
};
use rustc_passes::{abi_test, hir_stats, layout_test};
use rustc_resolve::Resolver;
use rustc_session::code_stats::VTableSizeInfo;
use rustc_session::config::{CrateType, Input, OutFileName, OutputFilenames, OutputType};
use rustc_session::cstore::Untracked;
use rustc_session::output::{collect_crate_types, filename_for_input, find_crate_name};
use rustc_session::search_paths::PathKind;
use rustc_session::{Limit, Session};
use rustc_span::symbol::{sym, Symbol};
use rustc_span::FileName;
use rustc_target::spec::PanicStrategy;
use rustc_trait_selection::traits;
use tracing::{info, instrument};

use crate::interface::{Compiler, Result};
use crate::{errors, proc_macro_decls, util};

pub(crate) fn parse<'a>(sess: &'a Session) -> Result<ast::Crate> {
    let krate = sess
        .time("parse_crate", || {
            let mut parser = unwrap_or_emit_fatal(match &sess.io.input {
                Input::File(file) => new_parser_from_file(&sess.psess, file, None),
                Input::Str { input, name } => {
                    new_parser_from_source_str(&sess.psess, name.clone(), input.clone())
                }
            });
            parser.parse_crate_mod()
        })
        .map_err(|parse_error| parse_error.emit())?;

    if sess.opts.unstable_opts.input_stats {
        eprintln!("Lines of code:             {}", sess.source_map().count_lines());
        eprintln!("Pre-expansion node count:  {}", count_nodes(&krate));
    }

    if let Some(ref s) = sess.opts.unstable_opts.show_span {
        rustc_ast_passes::show_span::run(sess.dcx(), s, &krate);
    }

    if sess.opts.unstable_opts.hir_stats {
        hir_stats::print_ast_stats(&krate, "PRE EXPANSION AST STATS", "ast-stats-1");
    }

    Ok(krate)
}

fn count_nodes(krate: &ast::Crate) -> usize {
    let mut counter = rustc_ast_passes::node_count::NodeCounter::new();
    visit::walk_crate(&mut counter, krate);
    counter.count
}

fn pre_expansion_lint<'a>(
    sess: &Session,
    features: &Features,
    lint_store: &LintStore,
    registered_tools: &RegisteredTools,
    check_node: impl EarlyCheckNode<'a>,
    node_name: Symbol,
) {
    sess.prof.generic_activity_with_arg("pre_AST_expansion_lint_checks", node_name.as_str()).run(
        || {
            rustc_lint::check_ast_node(
                sess,
                features,
                true,
                lint_store,
                registered_tools,
                None,
                rustc_lint::BuiltinCombinedPreExpansionLintPass::new(),
                check_node,
            );
        },
    );
}

// Cannot implement directly for `LintStore` due to trait coherence.
struct LintStoreExpandImpl<'a>(&'a LintStore);

impl LintStoreExpand for LintStoreExpandImpl<'_> {
    fn pre_expansion_lint(
        &self,
        sess: &Session,
        features: &Features,
        registered_tools: &RegisteredTools,
        node_id: ast::NodeId,
        attrs: &[ast::Attribute],
        items: &[rustc_ast::ptr::P<ast::Item>],
        name: Symbol,
    ) {
        pre_expansion_lint(sess, features, self.0, registered_tools, (node_id, attrs, items), name);
    }
}

/// Runs the "early phases" of the compiler: initial `cfg` processing,
/// syntax expansion, secondary `cfg` expansion, synthesis of a test
/// harness if one is to be provided, injection of a dependency on the
/// standard library and prelude, and name resolution.
#[instrument(level = "trace", skip(krate, resolver))]
fn configure_and_expand(
    mut krate: ast::Crate,
    pre_configured_attrs: &[ast::Attribute],
    resolver: &mut Resolver<'_, '_>,
) -> ast::Crate {
    let tcx = resolver.tcx();
    let sess = tcx.sess;
    let features = tcx.features();
    let lint_store = unerased_lint_store(tcx.sess);
    let crate_name = tcx.crate_name(LOCAL_CRATE);
    let lint_check_node = (&krate, pre_configured_attrs);
    pre_expansion_lint(
        sess,
        features,
        lint_store,
        tcx.registered_tools(()),
        lint_check_node,
        crate_name,
    );
    rustc_builtin_macros::register_builtin_macros(resolver);

    let num_standard_library_imports = sess.time("crate_injection", || {
        rustc_builtin_macros::standard_library_imports::inject(
            &mut krate,
            pre_configured_attrs,
            resolver,
            sess,
            features,
        )
    });

    util::check_attr_crate_type(sess, pre_configured_attrs, resolver.lint_buffer());

    // Expand all macros
    krate = sess.time("macro_expand_crate", || {
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
            let mut new_path = Vec::from_iter(
                sess.host_filesearch(PathKind::All).search_paths().map(|p| p.dir.clone()),
            );
            for path in env::split_paths(&old_path) {
                if !new_path.contains(&path) {
                    new_path.push(path);
                }
            }
            env::set_var(
                "PATH",
                &env::join_paths(
                    new_path.iter().filter(|p| env::join_paths(iter::once(p)).is_ok()),
                )
                .unwrap(),
            );
        }

        // Create the config for macro expansion
        let recursion_limit = get_recursion_limit(pre_configured_attrs, sess);
        let cfg = rustc_expand::expand::ExpansionConfig {
            crate_name: crate_name.to_string(),
            features,
            recursion_limit,
            trace_mac: sess.opts.unstable_opts.trace_macros,
            should_test: sess.is_test_crate(),
            span_debug: sess.opts.unstable_opts.span_debug,
            proc_macro_backtrace: sess.opts.unstable_opts.proc_macro_backtrace,
        };

        let lint_store = LintStoreExpandImpl(lint_store);
        let mut ecx = ExtCtxt::new(sess, cfg, resolver, Some(&lint_store));
        ecx.num_standard_library_imports = num_standard_library_imports;
        // Expand macros now!
        let krate = sess.time("expand_crate", || ecx.monotonic_expander().expand_crate(krate));

        // The rest is error reporting

        sess.psess.buffered_lints.with_lock(|buffered_lints: &mut Vec<BufferedEarlyLint>| {
            buffered_lints.append(&mut ecx.buffered_early_lint);
        });

        sess.time("check_unused_macros", || {
            ecx.check_unused_macros();
        });

        // If we hit a recursion limit, exit early to avoid later passes getting overwhelmed
        // with a large AST
        if ecx.reduced_recursion_limit.is_some() {
            sess.dcx().abort_if_errors();
            unreachable!();
        }

        if cfg!(windows) {
            env::set_var("PATH", &old_path);
        }

        krate
    });

    sess.time("maybe_building_test_harness", || {
        rustc_builtin_macros::test_harness::inject(&mut krate, sess, features, resolver)
    });

    let has_proc_macro_decls = sess.time("AST_validation", || {
        rustc_ast_passes::ast_validation::check_crate(
            sess,
            features,
            &krate,
            resolver.lint_buffer(),
        )
    });

    let crate_types = tcx.crate_types();
    let is_executable_crate = crate_types.contains(&CrateType::Executable);
    let is_proc_macro_crate = crate_types.contains(&CrateType::ProcMacro);

    if crate_types.len() > 1 {
        if is_executable_crate {
            sess.dcx().emit_err(errors::MixedBinCrate);
        }
        if is_proc_macro_crate {
            sess.dcx().emit_err(errors::MixedProcMacroCrate);
        }
    }

    if is_proc_macro_crate && sess.panic_strategy() == PanicStrategy::Abort {
        sess.dcx().emit_warn(errors::ProcMacroCratePanicAbort);
    }

    sess.time("maybe_create_a_macro_crate", || {
        let is_test_crate = sess.is_test_crate();
        rustc_builtin_macros::proc_macro_harness::inject(
            &mut krate,
            sess,
            features,
            resolver,
            is_proc_macro_crate,
            has_proc_macro_decls,
            is_test_crate,
            sess.dcx(),
        )
    });

    // Done with macro expansion!

    resolver.resolve_crate(&krate);

    krate
}

fn early_lint_checks(tcx: TyCtxt<'_>, (): ()) {
    let sess = tcx.sess;
    let (resolver, krate) = &*tcx.resolver_for_lowering().borrow();
    let mut lint_buffer = resolver.lint_buffer.steal();

    if sess.opts.unstable_opts.input_stats {
        eprintln!("Post-expansion node count: {}", count_nodes(krate));
    }

    if sess.opts.unstable_opts.hir_stats {
        hir_stats::print_ast_stats(krate, "POST EXPANSION AST STATS", "ast-stats-2");
    }

    // Needs to go *after* expansion to be able to check the results of macro expansion.
    sess.time("complete_gated_feature_checking", || {
        rustc_ast_passes::feature_gate::check_crate(krate, sess, tcx.features());
    });

    // Add all buffered lints from the `ParseSess` to the `Session`.
    sess.psess.buffered_lints.with_lock(|buffered_lints| {
        info!("{} parse sess buffered_lints", buffered_lints.len());
        for early_lint in buffered_lints.drain(..) {
            lint_buffer.add_early_lint(early_lint);
        }
    });

    // Gate identifiers containing invalid Unicode codepoints that were recovered during lexing.
    sess.psess.bad_unicode_identifiers.with_lock(|identifiers| {
        for (ident, mut spans) in identifiers.drain(..) {
            spans.sort();
            if ident == sym::ferris {
                let first_span = spans[0];
                sess.dcx().emit_err(errors::FerrisIdentifier { spans, first_span });
            } else {
                sess.dcx().emit_err(errors::EmojiIdentifier { spans, ident });
            }
        }
    });

    let lint_store = unerased_lint_store(tcx.sess);
    rustc_lint::check_ast_node(
        sess,
        tcx.features(),
        false,
        lint_store,
        tcx.registered_tools(()),
        Some(lint_buffer),
        rustc_lint::BuiltinCombinedEarlyLintPass::new(),
        (&**krate, &*krate.attrs),
    )
}

// Returns all the paths that correspond to generated files.
fn generated_output_paths(
    tcx: TyCtxt<'_>,
    outputs: &OutputFilenames,
    exact_name: bool,
    crate_name: Symbol,
) -> Vec<PathBuf> {
    let sess = tcx.sess;
    let mut out_filenames = Vec::new();
    for output_type in sess.opts.output_types.keys() {
        let out_filename = outputs.path(*output_type);
        let file = out_filename.as_path().to_path_buf();
        match *output_type {
            // If the filename has been overridden using `-o`, it will not be modified
            // by appending `.rlib`, `.exe`, etc., so we can skip this transformation.
            OutputType::Exe if !exact_name => {
                for crate_type in tcx.crate_types().iter() {
                    let p = filename_for_input(sess, *crate_type, crate_name, outputs);
                    out_filenames.push(p.as_path().to_path_buf());
                }
            }
            OutputType::DepInfo if sess.opts.unstable_opts.dep_info_omit_d_target => {
                // Don't add the dep-info output when omitting it from dep-info targets
            }
            OutputType::DepInfo if out_filename.is_stdout() => {
                // Don't add the dep-info output when it goes to stdout
            }
            _ => {
                out_filenames.push(file);
            }
        }
    }
    out_filenames
}

fn output_contains_path(output_paths: &[PathBuf], input_path: &Path) -> bool {
    let input_path = try_canonicalize(input_path).ok();
    if input_path.is_none() {
        return false;
    }
    output_paths.iter().any(|output_path| try_canonicalize(output_path).ok() == input_path)
}

fn output_conflicts_with_dir(output_paths: &[PathBuf]) -> Option<&PathBuf> {
    output_paths.iter().find(|output_path| output_path.is_dir())
}

fn escape_dep_filename(filename: &str) -> String {
    // Apparently clang and gcc *only* escape spaces:
    // https://llvm.org/klaus/clang/commit/9d50634cfc268ecc9a7250226dd5ca0e945240d4
    filename.replace(' ', "\\ ")
}

// Makefile comments only need escaping newlines and `\`.
// The result can be unescaped by anything that can unescape `escape_default` and friends.
fn escape_dep_env(symbol: Symbol) -> String {
    let s = symbol.as_str();
    let mut escaped = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\n' => escaped.push_str(r"\n"),
            '\r' => escaped.push_str(r"\r"),
            '\\' => escaped.push_str(r"\\"),
            _ => escaped.push(c),
        }
    }
    escaped
}

fn write_out_deps(tcx: TyCtxt<'_>, outputs: &OutputFilenames, out_filenames: &[PathBuf]) {
    // Write out dependency rules to the dep-info file if requested
    let sess = tcx.sess;
    if !sess.opts.output_types.contains_key(&OutputType::DepInfo) {
        return;
    }
    let deps_output = outputs.path(OutputType::DepInfo);
    let deps_filename = deps_output.as_path();

    let result: io::Result<()> = try {
        // Build a list of files used to compile the output and
        // write Makefile-compatible dependency rules
        let mut files: Vec<String> = sess
            .source_map()
            .files()
            .iter()
            .filter(|fmap| fmap.is_real_file())
            .filter(|fmap| !fmap.is_imported())
            .map(|fmap| escape_dep_filename(&fmap.name.prefer_local().to_string()))
            .collect();

        // Account for explicitly marked-to-track files
        // (e.g. accessed in proc macros).
        let file_depinfo = sess.psess.file_depinfo.borrow();

        let normalize_path = |path: PathBuf| {
            let file = FileName::from(path);
            escape_dep_filename(&file.prefer_local().to_string())
        };

        // The entries will be used to declare dependencies between files in a
        // Makefile-like output, so the iteration order does not matter.
        #[allow(rustc::potential_query_instability)]
        let extra_tracked_files =
            file_depinfo.iter().map(|path_sym| normalize_path(PathBuf::from(path_sym.as_str())));
        files.extend(extra_tracked_files);

        // We also need to track used PGO profile files
        if let Some(ref profile_instr) = sess.opts.cg.profile_use {
            files.push(normalize_path(profile_instr.as_path().to_path_buf()));
        }
        if let Some(ref profile_sample) = sess.opts.unstable_opts.profile_sample_use {
            files.push(normalize_path(profile_sample.as_path().to_path_buf()));
        }

        // Debugger visualizer files
        for debugger_visualizer in tcx.debugger_visualizers(LOCAL_CRATE) {
            files.push(normalize_path(debugger_visualizer.path.clone().unwrap()));
        }

        if sess.binary_dep_depinfo() {
            if let Some(ref backend) = sess.opts.unstable_opts.codegen_backend {
                if backend.contains('.') {
                    // If the backend name contain a `.`, it is the path to an external dynamic
                    // library. If not, it is not a path.
                    files.push(backend.to_string());
                }
            }

            for &cnum in tcx.crates(()) {
                let source = tcx.used_crate_source(cnum);
                if let Some((path, _)) = &source.dylib {
                    files.push(escape_dep_filename(&path.display().to_string()));
                }
                if let Some((path, _)) = &source.rlib {
                    files.push(escape_dep_filename(&path.display().to_string()));
                }
                if let Some((path, _)) = &source.rmeta {
                    files.push(escape_dep_filename(&path.display().to_string()));
                }
            }
        }

        let write_deps_to_file = |file: &mut dyn Write| -> io::Result<()> {
            for path in out_filenames {
                writeln!(file, "{}: {}\n", path.display(), files.join(" "))?;
            }

            // Emit a fake target for each input file to the compilation. This
            // prevents `make` from spitting out an error if a file is later
            // deleted. For more info see #28735
            for path in files {
                writeln!(file, "{path}:")?;
            }

            // Emit special comments with information about accessed environment variables.
            let env_depinfo = sess.psess.env_depinfo.borrow();
            if !env_depinfo.is_empty() {
                // We will soon sort, so the initial order does not matter.
                #[allow(rustc::potential_query_instability)]
                let mut envs: Vec<_> = env_depinfo
                    .iter()
                    .map(|(k, v)| (escape_dep_env(*k), v.map(escape_dep_env)))
                    .collect();
                envs.sort_unstable();
                writeln!(file)?;
                for (k, v) in envs {
                    write!(file, "# env-dep:{k}")?;
                    if let Some(v) = v {
                        write!(file, "={v}")?;
                    }
                    writeln!(file)?;
                }
            }

            Ok(())
        };

        match deps_output {
            OutFileName::Stdout => {
                let mut file = BufWriter::new(io::stdout());
                write_deps_to_file(&mut file)?;
            }
            OutFileName::Real(ref path) => {
                let mut file = BufWriter::new(fs::File::create(path)?);
                write_deps_to_file(&mut file)?;
            }
        }
    };

    match result {
        Ok(_) => {
            if sess.opts.json_artifact_notifications {
                sess.dcx().emit_artifact_notification(deps_filename, "dep-info");
            }
        }
        Err(error) => {
            sess.dcx().emit_fatal(errors::ErrorWritingDependencies { path: deps_filename, error });
        }
    }
}

fn resolver_for_lowering_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    (): (),
) -> (&'tcx Steal<(ty::ResolverAstLowering, Lrc<ast::Crate>)>, &'tcx ty::ResolverGlobalCtxt) {
    let arenas = Resolver::arenas();
    let _ = tcx.registered_tools(()); // Uses `crate_for_resolver`.
    let (krate, pre_configured_attrs) = tcx.crate_for_resolver(()).steal();
    let mut resolver = Resolver::new(
        tcx,
        &pre_configured_attrs,
        krate.spans.inner_span,
        krate.spans.inject_use_span,
        &arenas,
    );
    let krate = configure_and_expand(krate, &pre_configured_attrs, &mut resolver);

    // Make sure we don't mutate the cstore from here on.
    tcx.untracked().cstore.freeze();

    let ty::ResolverOutputs {
        global_ctxt: untracked_resolutions,
        ast_lowering: untracked_resolver_for_lowering,
    } = resolver.into_outputs();

    let resolutions = tcx.arena.alloc(untracked_resolutions);
    (tcx.arena.alloc(Steal::new((untracked_resolver_for_lowering, Lrc::new(krate)))), resolutions)
}

pub fn write_dep_info(tcx: TyCtxt<'_>) {
    // Make sure name resolution and macro expansion is run for
    // the side-effect of providing a complete set of all
    // accessed files and env vars.
    let _ = tcx.resolver_for_lowering();

    let sess = tcx.sess;
    let _timer = sess.timer("write_dep_info");
    let crate_name = tcx.crate_name(LOCAL_CRATE);

    let outputs = tcx.output_filenames(());
    let output_paths =
        generated_output_paths(tcx, &outputs, sess.io.output_file.is_some(), crate_name);

    // Ensure the source file isn't accidentally overwritten during compilation.
    if let Some(input_path) = sess.io.input.opt_path() {
        if sess.opts.will_create_output_file() {
            if output_contains_path(&output_paths, input_path) {
                sess.dcx().emit_fatal(errors::InputFileWouldBeOverWritten { path: input_path });
            }
            if let Some(dir_path) = output_conflicts_with_dir(&output_paths) {
                sess.dcx().emit_fatal(errors::GeneratedFileConflictsWithDirectory {
                    input_path,
                    dir_path,
                });
            }
        }
    }

    if let Some(ref dir) = sess.io.temps_dir {
        if fs::create_dir_all(dir).is_err() {
            sess.dcx().emit_fatal(errors::TempsDirError);
        }
    }

    write_out_deps(tcx, &outputs, &output_paths);

    let only_dep_info = sess.opts.output_types.contains_key(&OutputType::DepInfo)
        && sess.opts.output_types.len() == 1;

    if !only_dep_info {
        if let Some(ref dir) = sess.io.output_dir {
            if fs::create_dir_all(dir).is_err() {
                sess.dcx().emit_fatal(errors::OutDirError);
            }
        }
    }
}

pub static DEFAULT_QUERY_PROVIDERS: LazyLock<Providers> = LazyLock::new(|| {
    let providers = &mut Providers::default();
    providers.analysis = analysis;
    providers.hir_crate = rustc_ast_lowering::lower_to_hir;
    providers.resolver_for_lowering_raw = resolver_for_lowering_raw;
    providers.stripped_cfg_items =
        |tcx, _| tcx.arena.alloc_from_iter(tcx.resolutions(()).stripped_cfg_items.steal());
    providers.resolutions = |tcx, ()| tcx.resolver_for_lowering_raw(()).1;
    providers.early_lint_checks = early_lint_checks;
    proc_macro_decls::provide(providers);
    rustc_const_eval::provide(providers);
    rustc_middle::hir::provide(providers);
    rustc_borrowck::provide(providers);
    rustc_mir_build::provide(providers);
    rustc_mir_transform::provide(providers);
    rustc_monomorphize::provide(providers);
    rustc_privacy::provide(providers);
    rustc_resolve::provide(providers);
    rustc_hir_analysis::provide(providers);
    rustc_hir_typeck::provide(providers);
    ty::provide(providers);
    traits::provide(providers);
    rustc_passes::provide(providers);
    rustc_traits::provide(providers);
    rustc_ty_utils::provide(providers);
    rustc_metadata::provide(providers);
    rustc_lint::provide(providers);
    rustc_symbol_mangling::provide(providers);
    rustc_codegen_ssa::provide(providers);
    *providers
});

pub(crate) fn create_global_ctxt<'tcx>(
    compiler: &'tcx Compiler,
    mut krate: rustc_ast::Crate,
    gcx_cell: &'tcx OnceLock<GlobalCtxt<'tcx>>,
    arena: &'tcx WorkerLocal<Arena<'tcx>>,
    hir_arena: &'tcx WorkerLocal<rustc_hir::Arena<'tcx>>,
) -> Result<&'tcx GlobalCtxt<'tcx>> {
    let sess = &compiler.sess;

    rustc_builtin_macros::cmdline_attrs::inject(
        &mut krate,
        &sess.psess,
        &sess.opts.unstable_opts.crate_attr,
    );

    let pre_configured_attrs = rustc_expand::config::pre_configure_attrs(sess, &krate.attrs);

    // parse `#[crate_name]` even if `--crate-name` was passed, to make sure it matches.
    let crate_name = find_crate_name(sess, &pre_configured_attrs);
    let crate_types = collect_crate_types(sess, &pre_configured_attrs);
    let stable_crate_id = StableCrateId::new(
        crate_name,
        crate_types.contains(&CrateType::Executable),
        sess.opts.cg.metadata.clone(),
        sess.cfg_version,
    );
    let outputs = util::build_output_filenames(&pre_configured_attrs, sess);
    let dep_graph = setup_dep_graph(sess)?;

    let cstore =
        FreezeLock::new(Box::new(CStore::new(compiler.codegen_backend.metadata_loader())) as _);
    let definitions = FreezeLock::new(Definitions::new(stable_crate_id));

    let stable_crate_ids = FreezeLock::new(StableCrateIdMap::default());
    let untracked =
        Untracked { cstore, source_span: AppendOnlyIndexVec::new(), definitions, stable_crate_ids };

    // We're constructing the HIR here; we don't care what we will
    // read, since we haven't even constructed the *input* to
    // incr. comp. yet.
    dep_graph.assert_ignored();

    let query_result_on_disk_cache = rustc_incremental::load_query_result_cache(sess);

    let codegen_backend = &compiler.codegen_backend;
    let mut providers = *DEFAULT_QUERY_PROVIDERS;
    codegen_backend.provide(&mut providers);

    if let Some(callback) = compiler.override_queries {
        callback(sess, &mut providers);
    }

    let incremental = dep_graph.is_fully_enabled();

    sess.time("setup_global_ctxt", || {
        let qcx = gcx_cell.get_or_init(move || {
            TyCtxt::create_global_ctxt(
                sess,
                crate_types,
                stable_crate_id,
                arena,
                hir_arena,
                untracked,
                dep_graph,
                rustc_query_impl::query_callbacks(arena),
                rustc_query_impl::query_system(
                    providers.queries,
                    providers.extern_queries,
                    query_result_on_disk_cache,
                    incremental,
                ),
                providers.hooks,
                compiler.current_gcx.clone(),
            )
        });

        qcx.enter(|tcx| {
            let feed = tcx.create_crate_num(stable_crate_id).unwrap();
            assert_eq!(feed.key(), LOCAL_CRATE);
            feed.crate_name(crate_name);

            let feed = tcx.feed_unit_query();
            feed.features_query(tcx.arena.alloc(rustc_expand::config::features(
                sess,
                &pre_configured_attrs,
                crate_name,
            )));
            feed.crate_for_resolver(tcx.arena.alloc(Steal::new((krate, pre_configured_attrs))));
            feed.output_filenames(Arc::new(outputs));
        });
        Ok(qcx)
    })
}

/// Runs all analyses that we guarantee to run, even if errors were reported in earlier analyses.
/// This function never fails.
fn run_required_analyses(tcx: TyCtxt<'_>) {
    if tcx.sess.opts.unstable_opts.hir_stats {
        rustc_passes::hir_stats::print_hir_stats(tcx);
    }
    #[cfg(debug_assertions)]
    rustc_passes::hir_id_validator::check_crate(tcx);
    let sess = tcx.sess;
    sess.time("misc_checking_1", || {
        parallel!(
            {
                sess.time("looking_for_entry_point", || tcx.ensure().entry_fn(()));

                sess.time("looking_for_derive_registrar", || {
                    tcx.ensure().proc_macro_decls_static(())
                });

                CStore::from_tcx(tcx).report_unused_deps(tcx);
            },
            {
                tcx.hir().par_for_each_module(|module| {
                    tcx.ensure().check_mod_loops(module);
                    tcx.ensure().check_mod_attrs(module);
                    tcx.ensure().check_mod_naked_functions(module);
                    tcx.ensure().check_mod_unstable_api_usage(module);
                    tcx.ensure().check_mod_const_bodies(module);
                });
            },
            {
                sess.time("unused_lib_feature_checking", || {
                    rustc_passes::stability::check_unused_or_stable_features(tcx)
                });
            },
            {
                // We force these queries to run,
                // since they might not otherwise get called.
                // This marks the corresponding crate-level attributes
                // as used, and ensures that their values are valid.
                tcx.ensure().limits(());
                tcx.ensure().stability_index(());
            }
        );
    });

    rustc_hir_analysis::check_crate(tcx);
    sess.time("MIR_coroutine_by_move_body", || {
        tcx.hir().par_body_owners(|def_id| {
            if tcx.needs_coroutine_by_move_body_def_id(def_id.to_def_id()) {
                tcx.ensure_with_value().coroutine_by_move_body_def_id(def_id);
            }
        });
    });
    // Freeze definitions as we don't add new ones at this point.
    // We need to wait until now since we synthesize a by-move body
    // This improves performance by allowing lock-free access to them.
    tcx.untracked().definitions.freeze();

    sess.time("MIR_borrow_checking", || {
        tcx.hir().par_body_owners(|def_id| {
            // Run unsafety check because it's responsible for stealing and
            // deallocating THIR.
            tcx.ensure().check_unsafety(def_id);
            tcx.ensure().mir_borrowck(def_id)
        });
    });
    sess.time("MIR_effect_checking", || {
        for def_id in tcx.hir().body_owners() {
            tcx.ensure().has_ffi_unwind_calls(def_id);

            // If we need to codegen, ensure that we emit all errors from
            // `mir_drops_elaborated_and_const_checked` now, to avoid discovering
            // them later during codegen.
            if tcx.sess.opts.output_types.should_codegen()
                || tcx.hir().body_const_context(def_id).is_some()
            {
                tcx.ensure().mir_drops_elaborated_and_const_checked(def_id);
                tcx.ensure().unused_generic_params(ty::InstanceKind::Item(def_id.to_def_id()));
            }
        }
    });
    tcx.hir().par_body_owners(|def_id| {
        if tcx.is_coroutine(def_id.to_def_id()) {
            tcx.ensure().mir_coroutine_witnesses(def_id);
            tcx.ensure().check_coroutine_obligations(
                tcx.typeck_root_def_id(def_id.to_def_id()).expect_local(),
            );
        }
    });

    sess.time("layout_testing", || layout_test::test_layout(tcx));
    sess.time("abi_testing", || abi_test::test_abi(tcx));

    // If `-Zvalidate-mir` is set, we also want to compute the final MIR for each item
    // (either its `mir_for_ctfe` or `optimized_mir`) since that helps uncover any bugs
    // in MIR optimizations that may only be reachable through codegen, or other codepaths
    // that requires the optimized/ctfe MIR, such as polymorphization, coroutine bodies,
    // or evaluating consts.
    if tcx.sess.opts.unstable_opts.validate_mir {
        sess.time("ensuring_final_MIR_is_computable", || {
            tcx.hir().par_body_owners(|def_id| {
                tcx.instance_mir(ty::InstanceKind::Item(def_id.into()));
            });
        });
    }
}

/// Runs the type-checking, region checking and other miscellaneous analysis
/// passes on the crate.
fn analysis(tcx: TyCtxt<'_>, (): ()) -> Result<()> {
    run_required_analyses(tcx);

    let sess = tcx.sess;

    // Avoid overwhelming user with errors if borrow checking failed.
    // I'm not sure how helpful this is, to be honest, but it avoids a
    // lot of annoying errors in the ui tests (basically,
    // lint warnings and so on -- kindck used to do this abort, but
    // kindck is gone now). -nmatsakis
    //
    // But we exclude lint errors from this, because lint errors are typically
    // less serious and we're more likely to want to continue (#87337).
    if let Some(guar) = sess.dcx().has_errors_excluding_lint_errors() {
        return Err(guar);
    }

    sess.time("misc_checking_3", || {
        parallel!(
            {
                tcx.ensure().effective_visibilities(());

                parallel!(
                    {
                        tcx.ensure().check_private_in_public(());
                    },
                    {
                        tcx.hir()
                            .par_for_each_module(|module| tcx.ensure().check_mod_deathness(module));
                    },
                    {
                        sess.time("lint_checking", || {
                            rustc_lint::check_crate(tcx);
                        });
                    },
                    {
                        tcx.ensure().clashing_extern_declarations(());
                    }
                );
            },
            {
                sess.time("privacy_checking_modules", || {
                    tcx.hir().par_for_each_module(|module| {
                        tcx.ensure().check_mod_privacy(module);
                    });
                });
            }
        );

        // This check has to be run after all lints are done processing. We don't
        // define a lint filter, as all lint checks should have finished at this point.
        sess.time("check_lint_expectations", || tcx.ensure().check_expectations(None));

        // This query is only invoked normally if a diagnostic is emitted that needs any
        // diagnostic item. If the crate compiles without checking any diagnostic items,
        // we will fail to emit overlap diagnostics. Thus we invoke it here unconditionally.
        let _ = tcx.all_diagnostic_items(());
    });

    if sess.opts.unstable_opts.print_vtable_sizes {
        let traits = tcx.traits(LOCAL_CRATE);

        for &tr in traits {
            if !tcx.is_object_safe(tr) {
                continue;
            }

            let name = ty::print::with_no_trimmed_paths!(tcx.def_path_str(tr));

            let mut first_dsa = true;

            // Number of vtable entries, if we didn't have upcasting
            let mut entries_ignoring_upcasting = 0;
            // Number of vtable entries needed solely for upcasting
            let mut entries_for_upcasting = 0;

            let trait_ref = ty::Binder::dummy(ty::TraitRef::identity(tcx, tr));

            // A slightly edited version of the code in
            // `rustc_trait_selection::traits::vtable::vtable_entries`, that works without self
            // type and just counts number of entries.
            //
            // Note that this is technically wrong, for traits which have associated types in
            // supertraits:
            //
            //   trait A: AsRef<Self::T> + AsRef<()> { type T; }
            //
            // Without self type we can't normalize `Self::T`, so we can't know if `AsRef<Self::T>`
            // and `AsRef<()>` are the same trait, thus we assume that those are different, and
            // potentially over-estimate how many vtable entries there are.
            //
            // Similarly this is wrong for traits that have methods with possibly-impossible bounds.
            // For example:
            //
            //   trait B<T> { fn f(&self) where T: Copy; }
            //
            // Here `dyn B<u8>` will have 4 entries, while `dyn B<String>` will only have 3.
            // However, since we don't know `T`, we can't know if `T: Copy` holds or not,
            // thus we lean on the bigger side and say it has 4 entries.
            traits::vtable::prepare_vtable_segments(tcx, trait_ref, |segment| {
                match segment {
                    traits::vtable::VtblSegment::MetadataDSA => {
                        // If this is the first dsa, it would be included either way,
                        // otherwise it's needed for upcasting
                        if std::mem::take(&mut first_dsa) {
                            entries_ignoring_upcasting += 3;
                        } else {
                            entries_for_upcasting += 3;
                        }
                    }

                    traits::vtable::VtblSegment::TraitOwnEntries { trait_ref, emit_vptr } => {
                        // Lookup the shape of vtable for the trait.
                        let own_existential_entries =
                            tcx.own_existential_vtable_entries(trait_ref.def_id());

                        // The original code here ignores the method if its predicates are
                        // impossible. We can't really do that as, for example, all not trivial
                        // bounds on generic parameters are impossible (since we don't know the
                        // parameters...), see the comment above.
                        entries_ignoring_upcasting += own_existential_entries.len();

                        if emit_vptr {
                            entries_for_upcasting += 1;
                        }
                    }
                }

                std::ops::ControlFlow::Continue::<std::convert::Infallible>(())
            });

            sess.code_stats.record_vtable_size(
                tr,
                &name,
                VTableSizeInfo {
                    trait_name: name.clone(),
                    entries: entries_ignoring_upcasting + entries_for_upcasting,
                    entries_ignoring_upcasting,
                    entries_for_upcasting,
                    upcasting_cost_percent: entries_for_upcasting as f64
                        / entries_ignoring_upcasting as f64
                        * 100.,
                },
            )
        }
    }

    Ok(())
}

/// Check for the `#[rustc_error]` annotation, which forces an error in codegen. This is used
/// to write UI tests that actually test that compilation succeeds without reporting
/// an error.
fn check_for_rustc_errors_attr(tcx: TyCtxt<'_>) {
    let Some((def_id, _)) = tcx.entry_fn(()) else { return };
    for attr in tcx.get_attrs(def_id, sym::rustc_error) {
        match attr.meta_item_list() {
            // Check if there is a `#[rustc_error(delayed_bug_from_inside_query)]`.
            Some(list)
                if list.iter().any(|list_item| {
                    matches!(
                        list_item.ident().map(|i| i.name),
                        Some(sym::delayed_bug_from_inside_query)
                    )
                }) =>
            {
                tcx.ensure().trigger_delayed_bug(def_id);
            }

            // Bare `#[rustc_error]`.
            None => {
                tcx.dcx().emit_fatal(errors::RustcErrorFatal { span: tcx.def_span(def_id) });
            }

            // Some other attribute.
            Some(_) => {
                tcx.dcx().emit_warn(errors::RustcErrorUnexpectedAnnotation {
                    span: tcx.def_span(def_id),
                });
            }
        }
    }
}

/// Runs the codegen backend, after which the AST and analysis can
/// be discarded.
pub(crate) fn start_codegen<'tcx>(
    codegen_backend: &dyn CodegenBackend,
    tcx: TyCtxt<'tcx>,
) -> Result<Box<dyn Any>> {
    // Don't do code generation if there were any errors. Likewise if
    // there were any delayed bugs, because codegen will likely cause
    // more ICEs, obscuring the original problem.
    if let Some(guar) = tcx.sess.dcx().has_errors_or_delayed_bugs() {
        return Err(guar);
    }

    // Hook for UI tests.
    check_for_rustc_errors_attr(tcx);

    info!("Pre-codegen\n{:?}", tcx.debug_stats());

    let (metadata, need_metadata_module) = rustc_metadata::fs::encode_and_write_metadata(tcx);

    let codegen = tcx.sess.time("codegen_crate", move || {
        codegen_backend.codegen_crate(tcx, metadata, need_metadata_module)
    });

    // Don't run this test assertions when not doing codegen. Compiletest tries to build
    // build-fail tests in check mode first and expects it to not give an error in that case.
    if tcx.sess.opts.output_types.should_codegen() {
        rustc_symbol_mangling::test::report_symbol_names(tcx);
    }

    info!("Post-codegen\n{:?}", tcx.debug_stats());

    if tcx.sess.opts.output_types.contains_key(&OutputType::Mir) {
        if let Err(error) = rustc_mir_transform::dump_mir::emit_mir(tcx) {
            tcx.dcx().emit_fatal(errors::CantEmitMIR { error });
        }
    }

    Ok(codegen)
}

fn get_recursion_limit(krate_attrs: &[ast::Attribute], sess: &Session) -> Limit {
    if let Some(attr) = krate_attrs
        .iter()
        .find(|attr| attr.has_name(sym::recursion_limit) && attr.value_str().is_none())
    {
        // This is here mainly to check for using a macro, such as
        // #![recursion_limit = foo!()]. That is not supported since that
        // would require expanding this while in the middle of expansion,
        // which needs to know the limit before expanding. Otherwise,
        // validation would normally be caught in AstValidator (via
        // `check_builtin_attribute`), but by the time that runs the macro
        // is expanded, and it doesn't give an error.
        validate_attr::emit_fatal_malformed_builtin_attribute(
            &sess.psess,
            attr,
            sym::recursion_limit,
        );
    }
    rustc_middle::middle::limits::get_recursion_limit(krate_attrs, sess)
}
