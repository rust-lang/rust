use std::any::Any;
use std::ffi::{OsStr, OsString};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, OnceLock};
use std::{env, fs, iter};

use rustc_ast as ast;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_data_structures::jobserver::Proxy;
use rustc_data_structures::parallel;
use rustc_data_structures::steal::Steal;
use rustc_data_structures::sync::{AppendOnlyIndexVec, FreezeLock, WorkerLocal};
use rustc_expand::base::{ExtCtxt, LintStoreExpand};
use rustc_feature::Features;
use rustc_fs_util::try_canonicalize;
use rustc_hir::def_id::{LOCAL_CRATE, StableCrateId, StableCrateIdMap};
use rustc_hir::definitions::Definitions;
use rustc_incremental::setup_dep_graph;
use rustc_lint::{BufferedEarlyLint, EarlyCheckNode, LintStore, unerased_lint_store};
use rustc_metadata::creader::CStore;
use rustc_middle::arena::Arena;
use rustc_middle::dep_graph::DepsType;
use rustc_middle::ty::{self, CurrentGcx, GlobalCtxt, RegisteredTools, TyCtxt};
use rustc_middle::util::Providers;
use rustc_parse::{
    new_parser_from_file, new_parser_from_source_str, unwrap_or_emit_fatal, validate_attr,
};
use rustc_passes::{abi_test, input_stats, layout_test};
use rustc_resolve::Resolver;
use rustc_session::config::{CrateType, Input, OutFileName, OutputFilenames, OutputType};
use rustc_session::cstore::Untracked;
use rustc_session::output::{collect_crate_types, filename_for_input};
use rustc_session::parse::feature_err;
use rustc_session::search_paths::PathKind;
use rustc_session::{Limit, Session};
use rustc_span::{
    DUMMY_SP, ErrorGuaranteed, FileName, SourceFileHash, SourceFileHashAlgorithm, Span, Symbol, sym,
};
use rustc_target::spec::PanicStrategy;
use rustc_trait_selection::traits;
use tracing::{info, instrument};

use crate::interface::Compiler;
use crate::{errors, limits, proc_macro_decls, util};

pub fn parse<'a>(sess: &'a Session) -> ast::Crate {
    let mut krate = sess
        .time("parse_crate", || {
            let mut parser = unwrap_or_emit_fatal(match &sess.io.input {
                Input::File(file) => new_parser_from_file(&sess.psess, file, None),
                Input::Str { input, name } => {
                    new_parser_from_source_str(&sess.psess, name.clone(), input.clone())
                }
            });
            parser.parse_crate_mod()
        })
        .unwrap_or_else(|parse_error| {
            let guar: ErrorGuaranteed = parse_error.emit();
            guar.raise_fatal();
        });

    rustc_builtin_macros::cmdline_attrs::inject(
        &mut krate,
        &sess.psess,
        &sess.opts.unstable_opts.crate_attr,
    );

    krate
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
                None,
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
                sess.host_filesearch().search_paths(PathKind::All).map(|p| p.dir.clone()),
            );
            for path in env::split_paths(&old_path) {
                if !new_path.contains(&path) {
                    new_path.push(path);
                }
            }
            unsafe {
                env::set_var(
                    "PATH",
                    &env::join_paths(
                        new_path.iter().filter(|p| env::join_paths(iter::once(p)).is_ok()),
                    )
                    .unwrap(),
                );
            }
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
            unsafe {
                env::set_var("PATH", &old_path);
            }
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
            tcx.is_sdylib_interface_build(),
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
    if crate_types.contains(&CrateType::Sdylib) && !tcx.features().export_stable() {
        feature_err(sess, sym::export_stable, DUMMY_SP, "`sdylib` crate type is unstable").emit();
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

    CStore::from_tcx(tcx).report_incompatible_target_modifiers(tcx, &krate);
    CStore::from_tcx(tcx).report_incompatible_async_drop_feature(tcx, &krate);
    krate
}

fn early_lint_checks(tcx: TyCtxt<'_>, (): ()) {
    let sess = tcx.sess;
    let (resolver, krate) = &*tcx.resolver_for_lowering().borrow();
    let mut lint_buffer = resolver.lint_buffer.steal();

    if sess.opts.unstable_opts.input_stats {
        input_stats::print_ast_stats(krate, "POST EXPANSION AST STATS", "ast-stats");
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
                enum FerrisFix {
                    SnakeCase,
                    ScreamingSnakeCase,
                    PascalCase,
                }

                impl FerrisFix {
                    const fn as_str(self) -> &'static str {
                        match self {
                            FerrisFix::SnakeCase => "ferris",
                            FerrisFix::ScreamingSnakeCase => "FERRIS",
                            FerrisFix::PascalCase => "Ferris",
                        }
                    }
                }

                let first_span = spans[0];
                let prev_source = sess.psess.source_map().span_to_prev_source(first_span);
                let ferris_fix = prev_source
                    .map_or(FerrisFix::SnakeCase, |source| {
                        let mut source_before_ferris = source.trim_end().split_whitespace().rev();
                        match source_before_ferris.next() {
                            Some("struct" | "trait" | "mod" | "union" | "type" | "enum") => {
                                FerrisFix::PascalCase
                            }
                            Some("const" | "static") => FerrisFix::ScreamingSnakeCase,
                            Some("mut") if source_before_ferris.next() == Some("static") => {
                                FerrisFix::ScreamingSnakeCase
                            }
                            _ => FerrisFix::SnakeCase,
                        }
                    })
                    .as_str();

                sess.dcx().emit_err(errors::FerrisIdentifier { spans, first_span, ferris_fix });
            } else {
                sess.dcx().emit_err(errors::EmojiIdentifier { spans, ident });
            }
        }
    });

    let lint_store = unerased_lint_store(tcx.sess);
    rustc_lint::check_ast_node(
        sess,
        Some(tcx),
        tcx.features(),
        false,
        lint_store,
        tcx.registered_tools(()),
        Some(lint_buffer),
        rustc_lint::BuiltinCombinedEarlyLintPass::new(),
        (&**krate, &*krate.attrs),
    )
}

fn env_var_os<'tcx>(tcx: TyCtxt<'tcx>, key: &'tcx OsStr) -> Option<&'tcx OsStr> {
    let value = env::var_os(key);

    let value_tcx = value.as_ref().map(|value| {
        let encoded_bytes = tcx.arena.alloc_slice(value.as_encoded_bytes());
        debug_assert_eq!(value.as_encoded_bytes(), encoded_bytes);
        // SAFETY: The bytes came from `as_encoded_bytes`, and we assume that
        // `alloc_slice` is implemented correctly, and passes the same bytes
        // back (debug asserted above).
        unsafe { OsStr::from_encoded_bytes_unchecked(encoded_bytes) }
    });

    // Also add the variable to Cargo's dependency tracking
    //
    // NOTE: This only works for passes run before `write_dep_info`. See that
    // for extension points for configuring environment variables to be
    // properly change-tracked.
    tcx.sess.psess.env_depinfo.borrow_mut().insert((
        Symbol::intern(&key.to_string_lossy()),
        value.as_ref().and_then(|value| value.to_str()).map(|value| Symbol::intern(&value)),
    ));

    value_tcx
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
        let mut files: Vec<(String, u64, Option<SourceFileHash>)> = sess
            .source_map()
            .files()
            .iter()
            .filter(|fmap| fmap.is_real_file())
            .filter(|fmap| !fmap.is_imported())
            .map(|fmap| {
                (
                    escape_dep_filename(&fmap.name.prefer_local().to_string()),
                    fmap.source_len.0 as u64,
                    fmap.checksum_hash,
                )
            })
            .collect();

        let checksum_hash_algo = sess.opts.unstable_opts.checksum_hash_algorithm;

        // Account for explicitly marked-to-track files
        // (e.g. accessed in proc macros).
        let file_depinfo = sess.psess.file_depinfo.borrow();

        let normalize_path = |path: PathBuf| {
            let file = FileName::from(path);
            escape_dep_filename(&file.prefer_local().to_string())
        };

        // The entries will be used to declare dependencies between files in a
        // Makefile-like output, so the iteration order does not matter.
        fn hash_iter_files<P: AsRef<Path>>(
            it: impl Iterator<Item = P>,
            checksum_hash_algo: Option<SourceFileHashAlgorithm>,
        ) -> impl Iterator<Item = (P, u64, Option<SourceFileHash>)> {
            it.map(move |path| {
                match checksum_hash_algo.and_then(|algo| {
                    fs::File::open(path.as_ref())
                        .and_then(|mut file| {
                            SourceFileHash::new(algo, &mut file).map(|h| (file, h))
                        })
                        .and_then(|(file, h)| file.metadata().map(|m| (m.len(), h)))
                        .map_err(|e| {
                            tracing::error!(
                                "failed to compute checksum, omitting it from dep-info {} {e}",
                                path.as_ref().display()
                            )
                        })
                        .ok()
                }) {
                    Some((file_len, checksum)) => (path, file_len, Some(checksum)),
                    None => (path, 0, None),
                }
            })
        }

        let extra_tracked_files = hash_iter_files(
            file_depinfo.iter().map(|path_sym| normalize_path(PathBuf::from(path_sym.as_str()))),
            checksum_hash_algo,
        );
        files.extend(extra_tracked_files);

        // We also need to track used PGO profile files
        if let Some(ref profile_instr) = sess.opts.cg.profile_use {
            files.extend(hash_iter_files(
                iter::once(normalize_path(profile_instr.as_path().to_path_buf())),
                checksum_hash_algo,
            ));
        }
        if let Some(ref profile_sample) = sess.opts.unstable_opts.profile_sample_use {
            files.extend(hash_iter_files(
                iter::once(normalize_path(profile_sample.as_path().to_path_buf())),
                checksum_hash_algo,
            ));
        }

        // Debugger visualizer files
        for debugger_visualizer in tcx.debugger_visualizers(LOCAL_CRATE) {
            files.extend(hash_iter_files(
                iter::once(normalize_path(debugger_visualizer.path.clone().unwrap())),
                checksum_hash_algo,
            ));
        }

        if sess.binary_dep_depinfo() {
            if let Some(ref backend) = sess.opts.unstable_opts.codegen_backend {
                if backend.contains('.') {
                    // If the backend name contain a `.`, it is the path to an external dynamic
                    // library. If not, it is not a path.
                    files.extend(hash_iter_files(
                        iter::once(backend.to_string()),
                        checksum_hash_algo,
                    ));
                }
            }

            for &cnum in tcx.crates(()) {
                let source = tcx.used_crate_source(cnum);
                if let Some((path, _)) = &source.dylib {
                    files.extend(hash_iter_files(
                        iter::once(escape_dep_filename(&path.display().to_string())),
                        checksum_hash_algo,
                    ));
                }
                if let Some((path, _)) = &source.rlib {
                    files.extend(hash_iter_files(
                        iter::once(escape_dep_filename(&path.display().to_string())),
                        checksum_hash_algo,
                    ));
                }
                if let Some((path, _)) = &source.rmeta {
                    files.extend(hash_iter_files(
                        iter::once(escape_dep_filename(&path.display().to_string())),
                        checksum_hash_algo,
                    ));
                }
            }
        }

        let write_deps_to_file = |file: &mut dyn Write| -> io::Result<()> {
            for path in out_filenames {
                writeln!(
                    file,
                    "{}: {}\n",
                    path.display(),
                    files
                        .iter()
                        .map(|(path, _file_len, _checksum_hash_algo)| path.as_str())
                        .intersperse(" ")
                        .collect::<String>()
                )?;
            }

            // Emit a fake target for each input file to the compilation. This
            // prevents `make` from spitting out an error if a file is later
            // deleted. For more info see #28735
            for (path, _file_len, _checksum_hash_algo) in &files {
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

            // If caller requested this information, add special comments about source file checksums.
            // These are not necessarily the same checksums as was used in the debug files.
            if sess.opts.unstable_opts.checksum_hash_algorithm().is_some() {
                files
                    .iter()
                    .filter_map(|(path, file_len, hash_algo)| {
                        hash_algo.map(|hash_algo| (path, file_len, hash_algo))
                    })
                    .try_for_each(|(path, file_len, checksum_hash)| {
                        writeln!(file, "# checksum:{checksum_hash} file_len:{file_len} {path}")
                    })?;
            }

            Ok(())
        };

        match deps_output {
            OutFileName::Stdout => {
                let mut file = BufWriter::new(io::stdout());
                write_deps_to_file(&mut file)?;
            }
            OutFileName::Real(ref path) => {
                let mut file = fs::File::create_buffered(path)?;
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
) -> (&'tcx Steal<(ty::ResolverAstLowering, Arc<ast::Crate>)>, &'tcx ty::ResolverGlobalCtxt) {
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
    (tcx.arena.alloc(Steal::new((untracked_resolver_for_lowering, Arc::new(krate)))), resolutions)
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

pub fn write_interface<'tcx>(tcx: TyCtxt<'tcx>) {
    if !tcx.crate_types().contains(&rustc_session::config::CrateType::Sdylib) {
        return;
    }
    let _timer = tcx.sess.timer("write_interface");
    let (_, krate) = &*tcx.resolver_for_lowering().borrow();

    let krate = rustc_ast_pretty::pprust::print_crate_as_interface(
        krate,
        tcx.sess.psess.edition,
        &tcx.sess.psess.attr_id_generator,
    );
    let export_output = tcx.output_filenames(()).interface_path();
    let mut file = fs::File::create_buffered(export_output).unwrap();
    if let Err(err) = write!(file, "{}", krate) {
        tcx.dcx().fatal(format!("error writing interface file: {}", err));
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
    providers.env_var_os = env_var_os;
    limits::provide(providers);
    proc_macro_decls::provide(providers);
    rustc_const_eval::provide(providers);
    rustc_middle::hir::provide(providers);
    rustc_borrowck::provide(providers);
    rustc_incremental::provide(providers);
    rustc_mir_build::provide(providers);
    rustc_mir_transform::provide(providers);
    rustc_monomorphize::provide(providers);
    rustc_privacy::provide(providers);
    rustc_query_impl::provide(providers);
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

pub fn create_and_enter_global_ctxt<T, F: for<'tcx> FnOnce(TyCtxt<'tcx>) -> T>(
    compiler: &Compiler,
    krate: rustc_ast::Crate,
    f: F,
) -> T {
    let sess = &compiler.sess;

    let pre_configured_attrs = rustc_expand::config::pre_configure_attrs(sess, &krate.attrs);

    let crate_name = get_crate_name(sess, &pre_configured_attrs);
    let crate_types = collect_crate_types(sess, &pre_configured_attrs);
    let stable_crate_id = StableCrateId::new(
        crate_name,
        crate_types.contains(&CrateType::Executable),
        sess.opts.cg.metadata.clone(),
        sess.cfg_version,
    );

    let outputs = util::build_output_filenames(&pre_configured_attrs, sess);

    let dep_type = DepsType { dep_names: rustc_query_impl::dep_kind_names() };
    let dep_graph = setup_dep_graph(sess, crate_name, &dep_type);

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

    let gcx_cell = OnceLock::new();
    let arena = WorkerLocal::new(|_| Arena::default());
    let hir_arena = WorkerLocal::new(|_| rustc_hir::Arena::default());

    // This closure is necessary to force rustc to perform the correct lifetime
    // subtyping for GlobalCtxt::enter to be allowed.
    let inner: Box<
        dyn for<'tcx> FnOnce(
            &'tcx Session,
            CurrentGcx,
            Arc<Proxy>,
            &'tcx OnceLock<GlobalCtxt<'tcx>>,
            &'tcx WorkerLocal<Arena<'tcx>>,
            &'tcx WorkerLocal<rustc_hir::Arena<'tcx>>,
            F,
        ) -> T,
    > = Box::new(move |sess, current_gcx, jobserver_proxy, gcx_cell, arena, hir_arena, f| {
        TyCtxt::create_global_ctxt(
            gcx_cell,
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
            current_gcx,
            jobserver_proxy,
            |tcx| {
                let feed = tcx.create_crate_num(stable_crate_id).unwrap();
                assert_eq!(feed.key(), LOCAL_CRATE);
                feed.crate_name(crate_name);

                let feed = tcx.feed_unit_query();
                feed.features_query(tcx.arena.alloc(rustc_expand::config::features(
                    tcx.sess,
                    &pre_configured_attrs,
                    crate_name,
                )));
                feed.crate_for_resolver(tcx.arena.alloc(Steal::new((krate, pre_configured_attrs))));
                feed.output_filenames(Arc::new(outputs));

                let res = f(tcx);
                // FIXME maybe run finish even when a fatal error occurred? or at least tcx.alloc_self_profile_query_strings()?
                tcx.finish();
                res
            },
        )
    });

    inner(
        &compiler.sess,
        compiler.current_gcx.clone(),
        Arc::clone(&compiler.jobserver_proxy),
        &gcx_cell,
        &arena,
        &hir_arena,
        f,
    )
}

/// Runs all analyses that we guarantee to run, even if errors were reported in earlier analyses.
/// This function never fails.
fn run_required_analyses(tcx: TyCtxt<'_>) {
    if tcx.sess.opts.unstable_opts.input_stats {
        rustc_passes::input_stats::print_hir_stats(tcx);
    }
    // When using rustdoc's "jump to def" feature, it enters this code and `check_crate`
    // is not defined. So we need to cfg it out.
    #[cfg(all(not(doc), debug_assertions))]
    rustc_passes::hir_id_validator::check_crate(tcx);

    // Prefetch this to prevent multiple threads from blocking on it later.
    // This is needed since the `hir_id_validator::check_crate` call above is not guaranteed
    // to use `hir_crate`.
    tcx.ensure_done().hir_crate(());

    let sess = tcx.sess;
    sess.time("misc_checking_1", || {
        parallel!(
            {
                sess.time("looking_for_entry_point", || tcx.ensure_ok().entry_fn(()));

                sess.time("looking_for_derive_registrar", || {
                    tcx.ensure_ok().proc_macro_decls_static(())
                });

                CStore::from_tcx(tcx).report_unused_deps(tcx);
            },
            {
                tcx.ensure_ok().exportable_items(LOCAL_CRATE);
                tcx.ensure_ok().stable_order_of_exportable_impls(LOCAL_CRATE);
                tcx.par_hir_for_each_module(|module| {
                    tcx.ensure_ok().check_mod_loops(module);
                    tcx.ensure_ok().check_mod_attrs(module);
                    tcx.ensure_ok().check_mod_unstable_api_usage(module);
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
                tcx.ensure_ok().limits(());
                tcx.ensure_ok().stability_index(());
            }
        );
    });

    rustc_hir_analysis::check_crate(tcx);
    // Freeze definitions as we don't add new ones at this point.
    // We need to wait until now since we synthesize a by-move body
    // for all coroutine-closures.
    //
    // This improves performance by allowing lock-free access to them.
    tcx.untracked().definitions.freeze();

    sess.time("MIR_borrow_checking", || {
        tcx.par_hir_body_owners(|def_id| {
            if !tcx.is_typeck_child(def_id.to_def_id()) {
                // Child unsafety and borrowck happens together with the parent
                tcx.ensure_ok().check_unsafety(def_id);
                tcx.ensure_ok().mir_borrowck(def_id)
            }
            tcx.ensure_ok().has_ffi_unwind_calls(def_id);

            // If we need to codegen, ensure that we emit all errors from
            // `mir_drops_elaborated_and_const_checked` now, to avoid discovering
            // them later during codegen.
            if tcx.sess.opts.output_types.should_codegen()
                || tcx.hir_body_const_context(def_id).is_some()
            {
                tcx.ensure_ok().mir_drops_elaborated_and_const_checked(def_id);
            }
            if tcx.is_coroutine(def_id.to_def_id()) {
                tcx.ensure_ok().mir_coroutine_witnesses(def_id);
                let _ = tcx.ensure_ok().check_coroutine_obligations(
                    tcx.typeck_root_def_id(def_id.to_def_id()).expect_local(),
                );
                if !tcx.is_async_drop_in_place_coroutine(def_id.to_def_id()) {
                    // Eagerly check the unsubstituted layout for cycles.
                    tcx.ensure_ok().layout_of(
                        ty::TypingEnv::post_analysis(tcx, def_id.to_def_id())
                            .as_query_input(tcx.type_of(def_id).instantiate_identity()),
                    );
                }
            }
        });
    });

    sess.time("layout_testing", || layout_test::test_layout(tcx));
    sess.time("abi_testing", || abi_test::test_abi(tcx));

    // If `-Zvalidate-mir` is set, we also want to compute the final MIR for each item
    // (either its `mir_for_ctfe` or `optimized_mir`) since that helps uncover any bugs
    // in MIR optimizations that may only be reachable through codegen, or other codepaths
    // that requires the optimized/ctfe MIR, coroutine bodies, or evaluating consts.
    if tcx.sess.opts.unstable_opts.validate_mir {
        sess.time("ensuring_final_MIR_is_computable", || {
            tcx.par_hir_body_owners(|def_id| {
                tcx.instance_mir(ty::InstanceKind::Item(def_id.into()));
            });
        });
    }
}

/// Runs the type-checking, region checking and other miscellaneous analysis
/// passes on the crate.
fn analysis(tcx: TyCtxt<'_>, (): ()) {
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
        guar.raise_fatal();
    }

    sess.time("misc_checking_3", || {
        parallel!(
            {
                tcx.ensure_ok().effective_visibilities(());

                parallel!(
                    {
                        tcx.ensure_ok().check_private_in_public(());
                    },
                    {
                        tcx.par_hir_for_each_module(|module| {
                            tcx.ensure_ok().check_mod_deathness(module)
                        });
                    },
                    {
                        sess.time("lint_checking", || {
                            rustc_lint::check_crate(tcx);
                        });
                    },
                    {
                        tcx.ensure_ok().clashing_extern_declarations(());
                    }
                );
            },
            {
                sess.time("privacy_checking_modules", || {
                    tcx.par_hir_for_each_module(|module| {
                        tcx.ensure_ok().check_mod_privacy(module);
                    });
                });
            }
        );

        // This check has to be run after all lints are done processing. We don't
        // define a lint filter, as all lint checks should have finished at this point.
        sess.time("check_lint_expectations", || tcx.ensure_ok().check_expectations(None));

        // This query is only invoked normally if a diagnostic is emitted that needs any
        // diagnostic item. If the crate compiles without checking any diagnostic items,
        // we will fail to emit overlap diagnostics. Thus we invoke it here unconditionally.
        let _ = tcx.all_diagnostic_items(());
    });
}

/// Runs the codegen backend, after which the AST and analysis can
/// be discarded.
pub(crate) fn start_codegen<'tcx>(
    codegen_backend: &dyn CodegenBackend,
    tcx: TyCtxt<'tcx>,
) -> Box<dyn Any> {
    // Hook for tests.
    if let Some((def_id, _)) = tcx.entry_fn(())
        && tcx.has_attr(def_id, sym::rustc_delayed_bug_from_inside_query)
    {
        tcx.ensure_ok().trigger_delayed_bug(def_id);
    }

    // Don't run this test assertions when not doing codegen. Compiletest tries to build
    // build-fail tests in check mode first and expects it to not give an error in that case.
    if tcx.sess.opts.output_types.should_codegen() {
        rustc_symbol_mangling::test::report_symbol_names(tcx);
    }

    // Don't do code generation if there were any errors. Likewise if
    // there were any delayed bugs, because codegen will likely cause
    // more ICEs, obscuring the original problem.
    if let Some(guar) = tcx.sess.dcx().has_errors_or_delayed_bugs() {
        guar.raise_fatal();
    }

    info!("Pre-codegen\n{:?}", tcx.debug_stats());

    let (metadata, need_metadata_module) = rustc_metadata::fs::encode_and_write_metadata(tcx);

    let codegen = tcx.sess.time("codegen_crate", move || {
        codegen_backend.codegen_crate(tcx, metadata, need_metadata_module)
    });

    info!("Post-codegen\n{:?}", tcx.debug_stats());

    // This must run after monomorphization so that all generic types
    // have been instantiated.
    if tcx.sess.opts.unstable_opts.print_type_sizes {
        tcx.sess.code_stats.print_type_sizes();
    }

    codegen
}

/// Compute and validate the crate name.
pub fn get_crate_name(sess: &Session, krate_attrs: &[ast::Attribute]) -> Symbol {
    // We validate *all* occurrences of `#![crate_name]`, pick the first find and
    // if a crate name was passed on the command line via `--crate-name` we enforce
    // that they match.
    // We perform the validation step here instead of later to ensure it gets run
    // in all code paths that require the crate name very early on, namely before
    // macro expansion.

    let attr_crate_name =
        validate_and_find_value_str_builtin_attr(sym::crate_name, sess, krate_attrs);

    let validate = |name, span| {
        rustc_session::output::validate_crate_name(sess, name, span);
        name
    };

    if let Some(crate_name) = &sess.opts.crate_name {
        let crate_name = Symbol::intern(crate_name);
        if let Some((attr_crate_name, span)) = attr_crate_name
            && attr_crate_name != crate_name
        {
            sess.dcx().emit_err(errors::CrateNameDoesNotMatch {
                span,
                crate_name,
                attr_crate_name,
            });
        }
        return validate(crate_name, None);
    }

    if let Some((crate_name, span)) = attr_crate_name {
        return validate(crate_name, Some(span));
    }

    if let Input::File(ref path) = sess.io.input
        && let Some(file_stem) = path.file_stem().and_then(|s| s.to_str())
    {
        if file_stem.starts_with('-') {
            sess.dcx().emit_err(errors::CrateNameInvalid { crate_name: file_stem });
        } else {
            return validate(Symbol::intern(&file_stem.replace('-', "_")), None);
        }
    }

    sym::rust_out
}

fn get_recursion_limit(krate_attrs: &[ast::Attribute], sess: &Session) -> Limit {
    // We don't permit macro calls inside of the attribute (e.g., #![recursion_limit = `expand!()`])
    // because that would require expanding this while in the middle of expansion, which needs to
    // know the limit before expanding.
    let _ = validate_and_find_value_str_builtin_attr(sym::recursion_limit, sess, krate_attrs);
    crate::limits::get_recursion_limit(krate_attrs, sess)
}

/// Validate *all* occurrences of the given "[value-str]" built-in attribute and return the first find.
///
/// This validator is intended for built-in attributes whose value needs to be known very early
/// during compilation (namely, before macro expansion) and it mainly exists to reject macro calls
/// inside of the attributes, such as in `#![name = expand!()]`. Normal attribute validation happens
/// during semantic analysis via [`TyCtxt::check_mod_attrs`] which happens *after* macro expansion
/// when such macro calls (here: `expand`) have already been expanded and we can no longer check for
/// their presence.
///
/// [value-str]: ast::Attribute::value_str
fn validate_and_find_value_str_builtin_attr(
    name: Symbol,
    sess: &Session,
    krate_attrs: &[ast::Attribute],
) -> Option<(Symbol, Span)> {
    let mut result = None;
    // Validate *all* relevant attributes, not just the first occurrence.
    for attr in ast::attr::filter_by_name(krate_attrs, name) {
        let Some(value) = attr.value_str() else {
            validate_attr::emit_fatal_malformed_builtin_attribute(&sess.psess, attr, name)
        };
        // Choose the first occurrence as our result.
        result.get_or_insert((value, attr.span));
    }
    result
}
