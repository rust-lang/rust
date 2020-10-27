use crate::interface::{Compiler, Result};
use crate::proc_macro_decls;
use crate::util;

use rustc_ast::mut_visit::{self, MutVisitor};
use rustc_ast::ptr::P;
use rustc_ast::{self as ast, token, visit};
use rustc_codegen_ssa::back::link::emit_metadata;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_data_structures::sync::{par_iter, Lrc, OnceCell, ParallelIterator, WorkerLocal};
use rustc_data_structures::temp_dir::MaybeTempDir;
use rustc_data_structures::{box_region_allow_access, declare_box_region_type, parallel};
use rustc_errors::{ErrorReported, PResult};
use rustc_expand::base::ExtCtxt;
use rustc_hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc_hir::definitions::Definitions;
use rustc_hir::Crate;
use rustc_lint::LintStore;
use rustc_middle::arena::Arena;
use rustc_middle::dep_graph::DepGraph;
use rustc_middle::middle;
use rustc_middle::middle::cstore::{CrateStore, MetadataLoader, MetadataLoaderDyn};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::steal::Steal;
use rustc_middle::ty::{self, GlobalCtxt, ResolverOutputs, TyCtxt};
use rustc_mir as mir;
use rustc_mir_build as mir_build;
use rustc_parse::{parse_crate_from_file, parse_crate_from_source_str};
use rustc_passes::{self, hir_stats, layout_test};
use rustc_plugin_impl as plugin;
use rustc_resolve::{Resolver, ResolverArenas};
use rustc_session::config::{CrateType, Input, OutputFilenames, OutputType, PpMode, PpSourceMode};
use rustc_session::output::{filename_for_input, filename_for_metadata};
use rustc_session::search_paths::PathKind;
use rustc_session::Session;
use rustc_span::symbol::Symbol;
use rustc_span::{FileName, RealFileName};
use rustc_trait_selection::traits;
use rustc_typeck as typeck;
use smallvec::SmallVec;
use tracing::{info, warn};

use rustc_serialize::json;
use tempfile::Builder as TempFileBuilder;

use std::any::Any;
use std::cell::RefCell;
use std::ffi::OsString;
use std::io::{self, BufWriter, Write};
use std::lazy::SyncLazy;
use std::path::PathBuf;
use std::rc::Rc;
use std::{env, fs, iter, mem};

/// Remove alls `LazyTokenStreams` from an AST struct
/// Normally, this is done during AST lowering. However,
/// printing the AST JSON requires us to serialize
/// the entire AST, and we don't want to serialize
/// a `LazyTokenStream`.
struct TokenStripper;
impl mut_visit::MutVisitor for TokenStripper {
    fn flat_map_item(&mut self, mut i: P<ast::Item>) -> SmallVec<[P<ast::Item>; 1]> {
        i.tokens = None;
        mut_visit::noop_flat_map_item(i, self)
    }
    fn flat_map_foreign_item(
        &mut self,
        mut i: P<ast::ForeignItem>,
    ) -> SmallVec<[P<ast::ForeignItem>; 1]> {
        i.tokens = None;
        mut_visit::noop_flat_map_foreign_item(i, self)
    }
    fn visit_block(&mut self, b: &mut P<ast::Block>) {
        b.tokens = None;
        mut_visit::noop_visit_block(b, self);
    }
    fn flat_map_stmt(&mut self, mut stmt: ast::Stmt) -> SmallVec<[ast::Stmt; 1]> {
        stmt.tokens = None;
        mut_visit::noop_flat_map_stmt(stmt, self)
    }
    fn visit_pat(&mut self, p: &mut P<ast::Pat>) {
        p.tokens = None;
        mut_visit::noop_visit_pat(p, self);
    }
    fn visit_ty(&mut self, ty: &mut P<ast::Ty>) {
        ty.tokens = None;
        mut_visit::noop_visit_ty(ty, self);
    }
    fn visit_attribute(&mut self, attr: &mut ast::Attribute) {
        attr.tokens = None;
        if let ast::AttrKind::Normal(ast::AttrItem { tokens, .. }) = &mut attr.kind {
            *tokens = None;
        }
        mut_visit::noop_visit_attribute(attr, self);
    }

    fn visit_interpolated(&mut self, nt: &mut token::Nonterminal) {
        if let token::Nonterminal::NtMeta(meta) = nt {
            meta.tokens = None;
        }
        // Handles all of the other cases
        mut_visit::noop_visit_interpolated(nt, self);
    }

    fn visit_path(&mut self, p: &mut ast::Path) {
        p.tokens = None;
        mut_visit::noop_visit_path(p, self);
    }
    fn visit_vis(&mut self, vis: &mut ast::Visibility) {
        vis.tokens = None;
        mut_visit::noop_visit_vis(vis, self);
    }
    fn visit_expr(&mut self, e: &mut P<ast::Expr>) {
        e.tokens = None;
        mut_visit::noop_visit_expr(e, self);
    }
    fn visit_mac(&mut self, _mac: &mut ast::MacCall) {}
}

pub fn parse<'a>(sess: &'a Session, input: &Input) -> PResult<'a, ast::Crate> {
    let krate = sess.time("parse_crate", || match input {
        Input::File(file) => parse_crate_from_file(file, &sess.parse_sess),
        Input::Str { input, name } => {
            parse_crate_from_source_str(name.clone(), input.clone(), &sess.parse_sess)
        }
    })?;

    if sess.opts.debugging_opts.ast_json_noexpand {
        // Set any `token` fields to `None` before
        // we display the AST.
        let mut krate = krate.clone();
        TokenStripper.visit_crate(&mut krate);
        println!("{}", json::as_json(&krate));
    }

    if sess.opts.debugging_opts.input_stats {
        println!("Lines of code:             {}", sess.source_map().count_lines());
        println!("Pre-expansion node count:  {}", count_nodes(&krate));
    }

    if let Some(ref s) = sess.opts.debugging_opts.show_span {
        rustc_ast_passes::show_span::run(sess.diagnostic(), s, &krate);
    }

    if sess.opts.debugging_opts.hir_stats {
        hir_stats::print_ast_stats(&krate, "PRE EXPANSION AST STATS");
    }

    Ok(krate)
}

fn count_nodes(krate: &ast::Crate) -> usize {
    let mut counter = rustc_ast_passes::node_count::NodeCounter::new();
    visit::walk_crate(&mut counter, krate);
    counter.count
}

declare_box_region_type!(
    pub BoxedResolver,
    for(),
    (&mut Resolver<'_>) -> (Result<ast::Crate>, ResolverOutputs)
);

/// Runs the "early phases" of the compiler: initial `cfg` processing, loading compiler plugins,
/// syntax expansion, secondary `cfg` expansion, synthesis of a test
/// harness if one is to be provided, injection of a dependency on the
/// standard library and prelude, and name resolution.
///
/// Returns `None` if we're aborting after handling -W help.
pub fn configure_and_expand(
    sess: Lrc<Session>,
    lint_store: Lrc<LintStore>,
    metadata_loader: Box<MetadataLoaderDyn>,
    krate: ast::Crate,
    crate_name: &str,
) -> Result<(ast::Crate, BoxedResolver)> {
    tracing::trace!("configure_and_expand");
    // Currently, we ignore the name resolution data structures for the purposes of dependency
    // tracking. Instead we will run name resolution and include its output in the hash of each
    // item, much like we do for macro expansion. In other words, the hash reflects not just
    // its contents but the results of name resolution on those contents. Hopefully we'll push
    // this back at some point.
    let crate_name = crate_name.to_string();
    let (result, resolver) = BoxedResolver::new(static move |mut action| {
        let _ = action;
        let sess = &*sess;
        let resolver_arenas = Resolver::arenas();
        let res = configure_and_expand_inner(
            sess,
            &lint_store,
            krate,
            &crate_name,
            &resolver_arenas,
            &*metadata_loader,
        );
        let mut resolver = match res {
            Err(v) => {
                yield BoxedResolver::initial_yield(Err(v));
                panic!()
            }
            Ok((krate, resolver)) => {
                action = yield BoxedResolver::initial_yield(Ok(krate));
                resolver
            }
        };
        box_region_allow_access!(for(), (&mut Resolver<'_>), (&mut resolver), action);
        resolver.into_outputs()
    });
    result.map(|k| (k, resolver))
}

impl BoxedResolver {
    pub fn to_resolver_outputs(resolver: Rc<RefCell<BoxedResolver>>) -> ResolverOutputs {
        match Rc::try_unwrap(resolver) {
            Ok(resolver) => resolver.into_inner().complete(),
            Err(resolver) => resolver.borrow_mut().access(|resolver| resolver.clone_outputs()),
        }
    }
}

pub fn register_plugins<'a>(
    sess: &'a Session,
    metadata_loader: &'a dyn MetadataLoader,
    register_lints: impl Fn(&Session, &mut LintStore),
    mut krate: ast::Crate,
    crate_name: &str,
) -> Result<(ast::Crate, Lrc<LintStore>)> {
    krate = sess.time("attributes_injection", || {
        rustc_builtin_macros::cmdline_attrs::inject(
            krate,
            &sess.parse_sess,
            &sess.opts.debugging_opts.crate_attr,
        )
    });

    let (krate, features) = rustc_expand::config::features(sess, krate);
    // these need to be set "early" so that expansion sees `quote` if enabled.
    sess.init_features(features);

    let crate_types = util::collect_crate_types(sess, &krate.attrs);
    sess.init_crate_types(crate_types);

    let disambiguator = util::compute_crate_disambiguator(sess);
    sess.crate_disambiguator.set(disambiguator).expect("not yet initialized");
    rustc_incremental::prepare_session_directory(sess, &crate_name, disambiguator);

    if sess.opts.incremental.is_some() {
        sess.time("incr_comp_garbage_collect_session_directories", || {
            if let Err(e) = rustc_incremental::garbage_collect_session_directories(sess) {
                warn!(
                    "Error while trying to garbage collect incremental \
                     compilation cache directory: {}",
                    e
                );
            }
        });
    }

    sess.time("recursion_limit", || {
        middle::limits::update_limits(sess, &krate);
    });

    let mut lint_store = rustc_lint::new_lint_store(
        sess.opts.debugging_opts.no_interleave_lints,
        sess.unstable_options(),
    );
    register_lints(&sess, &mut lint_store);

    let registrars =
        sess.time("plugin_loading", || plugin::load::load_plugins(sess, metadata_loader, &krate));
    sess.time("plugin_registration", || {
        let mut registry = plugin::Registry { lint_store: &mut lint_store };
        for registrar in registrars {
            registrar(&mut registry);
        }
    });

    Ok((krate, Lrc::new(lint_store)))
}

fn pre_expansion_lint(sess: &Session, lint_store: &LintStore, krate: &ast::Crate) {
    sess.time("pre_AST_expansion_lint_checks", || {
        rustc_lint::check_ast_crate(
            sess,
            lint_store,
            &krate,
            true,
            None,
            rustc_lint::BuiltinCombinedPreExpansionLintPass::new(),
        );
    });
}

fn configure_and_expand_inner<'a>(
    sess: &'a Session,
    lint_store: &'a LintStore,
    mut krate: ast::Crate,
    crate_name: &str,
    resolver_arenas: &'a ResolverArenas<'a>,
    metadata_loader: &'a MetadataLoaderDyn,
) -> Result<(ast::Crate, Resolver<'a>)> {
    tracing::trace!("configure_and_expand_inner");
    pre_expansion_lint(sess, lint_store, &krate);

    let mut resolver = Resolver::new(sess, &krate, crate_name, metadata_loader, &resolver_arenas);
    rustc_builtin_macros::register_builtin_macros(&mut resolver, sess.edition());

    krate = sess.time("crate_injection", || {
        let alt_std_name = sess.opts.alt_std_name.as_ref().map(|s| Symbol::intern(s));
        let (krate, name) = rustc_builtin_macros::standard_library_imports::inject(
            krate,
            &mut resolver,
            &sess,
            alt_std_name,
        );
        if let Some(name) = name {
            sess.parse_sess.injected_crate_name.set(name).expect("not yet initialized");
        }
        krate
    });

    util::check_attr_crate_type(&sess, &krate.attrs, &mut resolver.lint_buffer());

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
            let mut new_path = sess.host_filesearch(PathKind::All).search_path_dirs();
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
        let features = sess.features_untracked();
        let cfg = rustc_expand::expand::ExpansionConfig {
            features: Some(&features),
            recursion_limit: sess.recursion_limit(),
            trace_mac: sess.opts.debugging_opts.trace_macros,
            should_test: sess.opts.test,
            span_debug: sess.opts.debugging_opts.span_debug,
            proc_macro_backtrace: sess.opts.debugging_opts.proc_macro_backtrace,
            ..rustc_expand::expand::ExpansionConfig::default(crate_name.to_string())
        };

        let extern_mod_loaded = |k: &ast::Crate| pre_expansion_lint(sess, lint_store, k);
        let mut ecx = ExtCtxt::new(&sess, cfg, &mut resolver, Some(&extern_mod_loaded));

        // Expand macros now!
        let krate = sess.time("expand_crate", || ecx.monotonic_expander().expand_crate(krate));

        // The rest is error reporting

        sess.time("check_unused_macros", || {
            ecx.check_unused_macros();
        });

        if cfg!(windows) {
            env::set_var("PATH", &old_path);
        }

        let recursion_limit_hit = ecx.reduced_recursion_limit.is_some();
        if recursion_limit_hit {
            // If we hit a recursion limit, exit early to avoid later passes getting overwhelmed
            // with a large AST
            Err(ErrorReported)
        } else {
            Ok(krate)
        }
    })?;

    sess.time("maybe_building_test_harness", || {
        rustc_builtin_macros::test_harness::inject(&sess, &mut resolver, &mut krate)
    });

    if let Some(PpMode::PpmSource(PpSourceMode::PpmEveryBodyLoops)) = sess.opts.pretty {
        tracing::debug!("replacing bodies with loop {{}}");
        util::ReplaceBodyWithLoop::new(&mut resolver).visit_crate(&mut krate);
    }

    let has_proc_macro_decls = sess.time("AST_validation", || {
        rustc_ast_passes::ast_validation::check_crate(sess, &krate, &mut resolver.lint_buffer())
    });

    let crate_types = sess.crate_types();
    let is_proc_macro_crate = crate_types.contains(&CrateType::ProcMacro);

    // For backwards compatibility, we don't try to run proc macro injection
    // if rustdoc is run on a proc macro crate without '--crate-type proc-macro' being
    // specified. This should only affect users who manually invoke 'rustdoc', as
    // 'cargo doc' will automatically pass the proper '--crate-type' flags.
    // However, we do emit a warning, to let such users know that they should
    // start passing '--crate-type proc-macro'
    if has_proc_macro_decls && sess.opts.actually_rustdoc && !is_proc_macro_crate {
        let mut msg = sess.diagnostic().struct_warn(
            &"Trying to document proc macro crate \
            without passing '--crate-type proc-macro to rustdoc",
        );

        msg.warn("The generated documentation may be incorrect");
        msg.emit()
    } else {
        krate = sess.time("maybe_create_a_macro_crate", || {
            let num_crate_types = crate_types.len();
            let is_test_crate = sess.opts.test;
            rustc_builtin_macros::proc_macro_harness::inject(
                &sess,
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

    // Done with macro expansion!

    if sess.opts.debugging_opts.input_stats {
        println!("Post-expansion node count: {}", count_nodes(&krate));
    }

    if sess.opts.debugging_opts.hir_stats {
        hir_stats::print_ast_stats(&krate, "POST EXPANSION AST STATS");
    }

    if sess.opts.debugging_opts.ast_json {
        // Set any `token` fields to `None` before
        // we display the AST.
        let mut krate = krate.clone();
        TokenStripper.visit_crate(&mut krate);
        println!("{}", json::as_json(&krate));
    }

    resolver.resolve_crate(&krate);

    // Needs to go *after* expansion to be able to check the results of macro expansion.
    sess.time("complete_gated_feature_checking", || {
        rustc_ast_passes::feature_gate::check_crate(&krate, sess);
    });

    // Add all buffered lints from the `ParseSess` to the `Session`.
    sess.parse_sess.buffered_lints.with_lock(|buffered_lints| {
        info!("{} parse sess buffered_lints", buffered_lints.len());
        for early_lint in buffered_lints.drain(..) {
            resolver.lint_buffer().add_early_lint(early_lint);
        }
    });

    Ok((krate, resolver))
}

pub fn lower_to_hir<'res, 'tcx>(
    sess: &'tcx Session,
    lint_store: &LintStore,
    resolver: &'res mut Resolver<'_>,
    dep_graph: &'res DepGraph,
    krate: &'res ast::Crate,
    arena: &'tcx rustc_ast_lowering::Arena<'tcx>,
) -> Crate<'tcx> {
    // We're constructing the HIR here; we don't care what we will
    // read, since we haven't even constructed the *input* to
    // incr. comp. yet.
    dep_graph.assert_ignored();

    // Lower AST to HIR.
    let hir_crate = rustc_ast_lowering::lower_crate(
        sess,
        &krate,
        resolver,
        rustc_parse::nt_to_tokenstream,
        arena,
    );

    if sess.opts.debugging_opts.hir_stats {
        hir_stats::print_hir_stats(&hir_crate);
    }

    sess.time("early_lint_checks", || {
        rustc_lint::check_ast_crate(
            sess,
            lint_store,
            &krate,
            false,
            Some(std::mem::take(resolver.lint_buffer())),
            rustc_lint::BuiltinCombinedEarlyLintPass::new(),
        )
    });

    // Discard hygiene data, which isn't required after lowering to HIR.
    if !sess.opts.debugging_opts.keep_hygiene_data {
        rustc_span::hygiene::clear_syntax_context_map();
    }

    hir_crate
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
            OutputType::Exe if !exact_name => {
                for crate_type in sess.crate_types().iter() {
                    let p = filename_for_input(sess, *crate_type, crate_name, outputs);
                    out_filenames.push(p);
                }
            }
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
        if output_path.canonicalize().ok() == input_path { Some(()) } else { None }
    };
    check_output(output_paths, check).is_some()
}

fn output_conflicts_with_dir(output_paths: &[PathBuf]) -> Option<PathBuf> {
    let check = |output_path: &PathBuf| output_path.is_dir().then(|| output_path.clone());
    check_output(output_paths, check)
}

fn escape_dep_filename(filename: &FileName) -> String {
    // Apparently clang and gcc *only* escape spaces:
    // http://llvm.org/klaus/clang/commit/9d50634cfc268ecc9a7250226dd5ca0e945240d4
    filename.to_string().replace(" ", "\\ ")
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

fn write_out_deps(
    sess: &Session,
    boxed_resolver: &Steal<Rc<RefCell<BoxedResolver>>>,
    outputs: &OutputFilenames,
    out_filenames: &[PathBuf],
) {
    // Write out dependency rules to the dep-info file if requested
    if !sess.opts.output_types.contains_key(&OutputType::DepInfo) {
        return;
    }
    let deps_filename = outputs.path(OutputType::DepInfo);

    let result = (|| -> io::Result<()> {
        // Build a list of files used to compile the output and
        // write Makefile-compatible dependency rules
        let mut files: Vec<String> = sess
            .source_map()
            .files()
            .iter()
            .filter(|fmap| fmap.is_real_file())
            .filter(|fmap| !fmap.is_imported())
            .map(|fmap| escape_dep_filename(&fmap.unmapped_path.as_ref().unwrap_or(&fmap.name)))
            .collect();

        if let Some(ref backend) = sess.opts.debugging_opts.codegen_backend {
            files.push(backend.to_string());
        }

        if sess.binary_dep_depinfo() {
            boxed_resolver.borrow().borrow_mut().access(|resolver| {
                for cnum in resolver.cstore().crates_untracked() {
                    let source = resolver.cstore().crate_source_untracked(cnum);
                    if let Some((path, _)) = source.dylib {
                        let file_name = FileName::Real(RealFileName::Named(path));
                        files.push(escape_dep_filename(&file_name));
                    }
                    if let Some((path, _)) = source.rlib {
                        let file_name = FileName::Real(RealFileName::Named(path));
                        files.push(escape_dep_filename(&file_name));
                    }
                    if let Some((path, _)) = source.rmeta {
                        let file_name = FileName::Real(RealFileName::Named(path));
                        files.push(escape_dep_filename(&file_name));
                    }
                }
            });
        }

        let mut file = BufWriter::new(fs::File::create(&deps_filename)?);
        for path in out_filenames {
            writeln!(file, "{}: {}\n", path.display(), files.join(" "))?;
        }

        // Emit a fake target for each input file to the compilation. This
        // prevents `make` from spitting out an error if a file is later
        // deleted. For more info see #28735
        for path in files {
            writeln!(file, "{}:", path)?;
        }

        // Emit special comments with information about accessed environment variables.
        let env_depinfo = sess.parse_sess.env_depinfo.borrow();
        if !env_depinfo.is_empty() {
            let mut envs: Vec<_> = env_depinfo
                .iter()
                .map(|(k, v)| (escape_dep_env(*k), v.map(escape_dep_env)))
                .collect();
            envs.sort_unstable();
            writeln!(file)?;
            for (k, v) in envs {
                write!(file, "# env-dep:{}", k)?;
                if let Some(v) = v {
                    write!(file, "={}", v)?;
                }
                writeln!(file)?;
            }
        }

        Ok(())
    })();

    match result {
        Ok(_) => {
            if sess.opts.json_artifact_notifications {
                sess.parse_sess
                    .span_diagnostic
                    .emit_artifact_notification(&deps_filename, "dep-info");
            }
        }
        Err(e) => sess.fatal(&format!(
            "error writing dependencies to `{}`: {}",
            deps_filename.display(),
            e
        )),
    }
}

pub fn prepare_outputs(
    sess: &Session,
    compiler: &Compiler,
    krate: &ast::Crate,
    boxed_resolver: &Steal<Rc<RefCell<BoxedResolver>>>,
    crate_name: &str,
) -> Result<OutputFilenames> {
    let _timer = sess.timer("prepare_outputs");

    // FIXME: rustdoc passes &[] instead of &krate.attrs here
    let outputs = util::build_output_filenames(
        &compiler.input,
        &compiler.output_dir,
        &compiler.output_file,
        &krate.attrs,
        sess,
    );

    let output_paths =
        generated_output_paths(sess, &outputs, compiler.output_file.is_some(), &crate_name);

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

    write_out_deps(sess, boxed_resolver, &outputs, &output_paths);

    let only_dep_info = sess.opts.output_types.contains_key(&OutputType::DepInfo)
        && sess.opts.output_types.len() == 1;

    if !only_dep_info {
        if let Some(ref dir) = compiler.output_dir {
            if fs::create_dir_all(dir).is_err() {
                sess.err("failed to find or create the directory specified by `--out-dir`");
                return Err(ErrorReported);
            }
        }
    }

    Ok(outputs)
}

pub static DEFAULT_QUERY_PROVIDERS: SyncLazy<Providers> = SyncLazy::new(|| {
    let providers = &mut Providers::default();
    providers.analysis = analysis;
    proc_macro_decls::provide(providers);
    plugin::build::provide(providers);
    rustc_middle::hir::provide(providers);
    mir::provide(providers);
    mir_build::provide(providers);
    rustc_privacy::provide(providers);
    typeck::provide(providers);
    ty::provide(providers);
    traits::provide(providers);
    rustc_passes::provide(providers);
    rustc_resolve::provide(providers);
    rustc_traits::provide(providers);
    rustc_ty::provide(providers);
    rustc_metadata::provide(providers);
    rustc_lint::provide(providers);
    rustc_symbol_mangling::provide(providers);
    rustc_codegen_ssa::provide(providers);
    *providers
});

pub static DEFAULT_EXTERN_QUERY_PROVIDERS: SyncLazy<Providers> = SyncLazy::new(|| {
    let mut extern_providers = *DEFAULT_QUERY_PROVIDERS;
    rustc_metadata::provide_extern(&mut extern_providers);
    rustc_codegen_ssa::provide_extern(&mut extern_providers);
    extern_providers
});

pub struct QueryContext<'tcx>(&'tcx GlobalCtxt<'tcx>);

impl<'tcx> QueryContext<'tcx> {
    pub fn enter<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(TyCtxt<'tcx>) -> R,
    {
        let icx = ty::tls::ImplicitCtxt::new(self.0);
        ty::tls::enter_context(&icx, |_| f(icx.tcx))
    }

    pub fn print_stats(&mut self) {
        self.enter(ty::query::print_stats)
    }
}

pub fn create_global_ctxt<'tcx>(
    compiler: &'tcx Compiler,
    lint_store: Lrc<LintStore>,
    krate: &'tcx Crate<'tcx>,
    dep_graph: DepGraph,
    mut resolver_outputs: ResolverOutputs,
    outputs: OutputFilenames,
    crate_name: &str,
    global_ctxt: &'tcx OnceCell<GlobalCtxt<'tcx>>,
    arena: &'tcx WorkerLocal<Arena<'tcx>>,
) -> QueryContext<'tcx> {
    let sess = &compiler.session();
    let defs: &'tcx Definitions = arena.alloc(mem::replace(
        &mut resolver_outputs.definitions,
        Definitions::new(crate_name, sess.local_crate_disambiguator()),
    ));

    let query_result_on_disk_cache = rustc_incremental::load_query_result_cache(sess);

    let codegen_backend = compiler.codegen_backend();
    let mut local_providers = *DEFAULT_QUERY_PROVIDERS;
    codegen_backend.provide(&mut local_providers);

    let mut extern_providers = *DEFAULT_EXTERN_QUERY_PROVIDERS;
    codegen_backend.provide(&mut extern_providers);
    codegen_backend.provide_extern(&mut extern_providers);

    if let Some(callback) = compiler.override_queries {
        callback(sess, &mut local_providers, &mut extern_providers);
    }

    let gcx = sess.time("setup_global_ctxt", || {
        global_ctxt.get_or_init(|| {
            TyCtxt::create_global_ctxt(
                sess,
                lint_store,
                local_providers,
                extern_providers,
                arena,
                resolver_outputs,
                krate,
                defs,
                dep_graph,
                query_result_on_disk_cache,
                &crate_name,
                &outputs,
            )
        })
    });

    // Do some initialization of the DepGraph that can only be done with the tcx available.
    let icx = ty::tls::ImplicitCtxt::new(&gcx);
    ty::tls::enter_context(&icx, |_| {
        icx.tcx.sess.time("dep_graph_tcx_init", || rustc_incremental::dep_graph_tcx_init(icx.tcx));
    });

    QueryContext(gcx)
}

/// Runs the resolution, type-checking, region checking and other
/// miscellaneous analysis passes on the crate.
fn analysis(tcx: TyCtxt<'_>, cnum: CrateNum) -> Result<()> {
    assert_eq!(cnum, LOCAL_CRATE);

    rustc_passes::hir_id_validator::check_crate(tcx);

    let sess = tcx.sess;
    let mut entry_point = None;

    sess.time("misc_checking_1", || {
        parallel!(
            {
                entry_point = sess
                    .time("looking_for_entry_point", || rustc_passes::entry::find_entry_point(tcx));

                sess.time("looking_for_plugin_registrar", || {
                    plugin::build::find_plugin_registrar(tcx)
                });

                sess.time("looking_for_derive_registrar", || proc_macro_decls::find(tcx));
            },
            {
                par_iter(&tcx.hir().krate().modules).for_each(|(&module, _)| {
                    let local_def_id = tcx.hir().local_def_id(module);
                    tcx.ensure().check_mod_loops(local_def_id);
                    tcx.ensure().check_mod_attrs(local_def_id);
                    tcx.ensure().check_mod_unstable_api_usage(local_def_id);
                    tcx.ensure().check_mod_const_bodies(local_def_id);
                });
            }
        );
    });

    // passes are timed inside typeck
    typeck::check_crate(tcx)?;

    sess.time("misc_checking_2", || {
        parallel!(
            {
                sess.time("match_checking", || {
                    tcx.par_body_owners(|def_id| {
                        tcx.ensure().check_match(def_id.to_def_id());
                    });
                });
            },
            {
                sess.time("liveness_and_intrinsic_checking", || {
                    par_iter(&tcx.hir().krate().modules).for_each(|(&module, _)| {
                        // this must run before MIR dump, because
                        // "not all control paths return a value" is reported here.
                        //
                        // maybe move the check to a MIR pass?
                        let local_def_id = tcx.hir().local_def_id(module);

                        tcx.ensure().check_mod_liveness(local_def_id);
                        tcx.ensure().check_mod_intrinsics(local_def_id);
                    });
                });
            }
        );
    });

    sess.time("MIR_borrow_checking", || {
        tcx.par_body_owners(|def_id| tcx.ensure().mir_borrowck(def_id));
    });

    sess.time("MIR_effect_checking", || {
        for def_id in tcx.body_owners() {
            mir::transform::check_unsafety::check_unsafety(tcx, def_id);

            if tcx.hir().body_const_context(def_id).is_some() {
                tcx.ensure()
                    .mir_drops_elaborated_and_const_checked(ty::WithOptConstParam::unknown(def_id));
            }
        }
    });

    sess.time("layout_testing", || layout_test::test_layout(tcx));

    // Avoid overwhelming user with errors if borrow checking failed.
    // I'm not sure how helpful this is, to be honest, but it avoids a
    // lot of annoying errors in the compile-fail tests (basically,
    // lint warnings and so on -- kindck used to do this abort, but
    // kindck is gone now). -nmatsakis
    if sess.has_errors() {
        return Err(ErrorReported);
    }

    sess.time("misc_checking_3", || {
        parallel!(
            {
                tcx.ensure().privacy_access_levels(LOCAL_CRATE);

                parallel!(
                    {
                        tcx.ensure().check_private_in_public(LOCAL_CRATE);
                    },
                    {
                        sess.time("death_checking", || rustc_passes::dead::check_crate(tcx));
                    },
                    {
                        sess.time("unused_lib_feature_checking", || {
                            rustc_passes::stability::check_unused_or_stable_features(tcx)
                        });
                    },
                    {
                        sess.time("lint_checking", || {
                            rustc_lint::check_crate(tcx, || {
                                rustc_lint::BuiltinCombinedLateLintPass::new()
                            });
                        });
                    }
                );
            },
            {
                sess.time("privacy_checking_modules", || {
                    par_iter(&tcx.hir().krate().modules).for_each(|(&module, _)| {
                        tcx.ensure().check_mod_privacy(tcx.hir().local_def_id(module));
                    });
                });
            }
        );
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
        Compressed,
    }

    let metadata_kind = tcx
        .sess
        .crate_types()
        .iter()
        .map(|ty| match *ty {
            CrateType::Executable | CrateType::Staticlib | CrateType::Cdylib => MetadataKind::None,

            CrateType::Rlib => MetadataKind::Uncompressed,

            CrateType::Dylib | CrateType::ProcMacro => MetadataKind::Compressed,
        })
        .max()
        .unwrap_or(MetadataKind::None);

    let metadata = match metadata_kind {
        MetadataKind::None => middle::cstore::EncodedMetadata::new(),
        MetadataKind::Uncompressed | MetadataKind::Compressed => tcx.encode_metadata(),
    };

    let _prof_timer = tcx.sess.prof.generic_activity("write_crate_metadata");

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
            .unwrap_or_else(|err| tcx.sess.fatal(&format!("couldn't create a temp dir: {}", err)));
        let metadata_tmpdir = MaybeTempDir::new(metadata_tmpdir, tcx.sess.opts.cg.save_temps);
        let metadata_filename = emit_metadata(tcx.sess, &metadata, &metadata_tmpdir);
        if let Err(e) = fs::rename(&metadata_filename, &out_filename) {
            tcx.sess.fatal(&format!("failed to write {}: {}", out_filename.display(), e));
        }
        if tcx.sess.opts.json_artifact_notifications {
            tcx.sess
                .parse_sess
                .span_diagnostic
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
    outputs: &OutputFilenames,
) -> Box<dyn Any> {
    info!("Pre-codegen\n{:?}", tcx.debug_stats());

    let (metadata, need_metadata_module) = encode_and_write_metadata(tcx, outputs);

    let codegen = tcx.sess.time("codegen_crate", move || {
        codegen_backend.codegen_crate(tcx, metadata, need_metadata_module)
    });

    info!("Post-codegen\n{:?}", tcx.debug_stats());

    if tcx.sess.opts.output_types.contains_key(&OutputType::Mir) {
        if let Err(e) = mir::transform::dump_mir::emit_mir(tcx, outputs) {
            tcx.sess.err(&format!("could not emit MIR: {}", e));
            tcx.sess.abort_if_errors();
        }
    }

    codegen
}
