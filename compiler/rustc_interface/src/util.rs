use crate::errors;
use info;
use libloading::Library;
use rustc_ast as ast;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::registry::Registry;
use rustc_parse::validate_attr;
use rustc_session as session;
use rustc_session::config::CheckCfg;
use rustc_session::config::{self, CrateType};
use rustc_session::config::{ErrorOutputType, OutputFilenames};
use rustc_session::filesearch::sysroot_candidates;
use rustc_session::lint::{self, BuiltinLintDiagnostics, LintBuffer};
use rustc_session::parse::CrateConfig;
use rustc_session::{early_error, filesearch, output, Session};
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::edition::Edition;
use rustc_span::source_map::FileLoader;
use rustc_span::symbol::{sym, Symbol};
use session::CompilerIO;
use std::env;
use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::thread;

/// Function pointer type that constructs a new CodegenBackend.
pub type MakeBackendFn = fn() -> Box<dyn CodegenBackend>;

/// Adds `target_feature = "..."` cfgs for a variety of platform
/// specific features (SSE, NEON etc.).
///
/// This is performed by checking whether a set of permitted features
/// is available on the target machine, by querying LLVM.
pub fn add_configuration(
    cfg: &mut CrateConfig,
    sess: &mut Session,
    codegen_backend: &dyn CodegenBackend,
) {
    let tf = sym::target_feature;

    let unstable_target_features = codegen_backend.target_features(sess, true);
    sess.unstable_target_features.extend(unstable_target_features.iter().cloned());

    let target_features = codegen_backend.target_features(sess, false);
    sess.target_features.extend(target_features.iter().cloned());

    cfg.extend(target_features.into_iter().map(|feat| (tf, Some(feat))));

    if sess.crt_static(None) {
        cfg.insert((tf, Some(sym::crt_dash_static)));
    }
}

pub fn create_session(
    sopts: config::Options,
    cfg: FxHashSet<(String, Option<String>)>,
    check_cfg: CheckCfg,
    locale_resources: &'static [&'static str],
    file_loader: Option<Box<dyn FileLoader + Send + Sync + 'static>>,
    io: CompilerIO,
    lint_caps: FxHashMap<lint::LintId, lint::Level>,
    make_codegen_backend: Option<
        Box<dyn FnOnce(&config::Options) -> Box<dyn CodegenBackend> + Send>,
    >,
    descriptions: Registry,
) -> (Session, Box<dyn CodegenBackend>) {
    let codegen_backend = if let Some(make_codegen_backend) = make_codegen_backend {
        make_codegen_backend(&sopts)
    } else {
        get_codegen_backend(&sopts.maybe_sysroot, sopts.unstable_opts.codegen_backend.as_deref())
    };

    // target_override is documented to be called before init(), so this is okay
    let target_override = codegen_backend.target_override(&sopts);

    let bundle = match rustc_errors::fluent_bundle(
        sopts.maybe_sysroot.clone(),
        sysroot_candidates().to_vec(),
        sopts.unstable_opts.translate_lang.clone(),
        sopts.unstable_opts.translate_additional_ftl.as_deref(),
        sopts.unstable_opts.translate_directionality_markers,
    ) {
        Ok(bundle) => bundle,
        Err(e) => {
            early_error(sopts.error_format, &format!("failed to load fluent bundle: {e}"));
        }
    };

    let mut locale_resources = Vec::from(locale_resources);
    locale_resources.push(codegen_backend.locale_resource());

    let mut sess = session::build_session(
        sopts,
        io,
        bundle,
        descriptions,
        locale_resources,
        lint_caps,
        file_loader,
        target_override,
    );

    codegen_backend.init(&sess);

    let mut cfg = config::build_configuration(&sess, config::to_crate_config(cfg));
    add_configuration(&mut cfg, &mut sess, &*codegen_backend);

    let mut check_cfg = config::to_crate_check_config(check_cfg);
    check_cfg.fill_well_known(&sess.target);

    sess.parse_sess.config = cfg;
    sess.parse_sess.check_config = check_cfg;

    (sess, codegen_backend)
}

const STACK_SIZE: usize = 8 * 1024 * 1024;

fn get_stack_size() -> Option<usize> {
    // FIXME: Hacks on hacks. If the env is trying to override the stack size
    // then *don't* set it explicitly.
    env::var_os("RUST_MIN_STACK").is_none().then_some(STACK_SIZE)
}

#[cfg(not(parallel_compiler))]
pub(crate) fn run_in_thread_pool_with_globals<F: FnOnce() -> R + Send, R: Send>(
    edition: Edition,
    _threads: usize,
    f: F,
) -> R {
    // The "thread pool" is a single spawned thread in the non-parallel
    // compiler. We run on a spawned thread instead of the main thread (a) to
    // provide control over the stack size, and (b) to increase similarity with
    // the parallel compiler, in particular to ensure there is no accidental
    // sharing of data between the main thread and the compilation thread
    // (which might cause problems for the parallel compiler).
    let mut builder = thread::Builder::new().name("rustc".to_string());
    if let Some(size) = get_stack_size() {
        builder = builder.stack_size(size);
    }

    // We build the session globals and run `f` on the spawned thread, because
    // `SessionGlobals` does not impl `Send` in the non-parallel compiler.
    thread::scope(|s| {
        // `unwrap` is ok here because `spawn_scoped` only panics if the thread
        // name contains null bytes.
        let r = builder
            .spawn_scoped(s, move || rustc_span::create_session_globals_then(edition, f))
            .unwrap()
            .join();

        match r {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        }
    })
}

#[cfg(parallel_compiler)]
pub(crate) fn run_in_thread_pool_with_globals<F: FnOnce() -> R + Send, R: Send>(
    edition: Edition,
    threads: usize,
    f: F,
) -> R {
    use rustc_data_structures::jobserver;
    use rustc_middle::ty::tls;
    use rustc_query_impl::{deadlock, QueryContext, QueryCtxt};

    let mut builder = rayon::ThreadPoolBuilder::new()
        .thread_name(|_| "rustc".to_string())
        .acquire_thread_handler(jobserver::acquire_thread)
        .release_thread_handler(jobserver::release_thread)
        .num_threads(threads)
        .deadlock_handler(|| {
            // On deadlock, creates a new thread and forwards information in thread
            // locals to it. The new thread runs the deadlock handler.
            let query_map = tls::with(|tcx| {
                QueryCtxt::from_tcx(tcx)
                    .try_collect_active_jobs()
                    .expect("active jobs shouldn't be locked in deadlock handler")
            });
            let registry = rustc_rayon_core::Registry::current();
            thread::spawn(move || deadlock(query_map, &registry));
        });
    if let Some(size) = get_stack_size() {
        builder = builder.stack_size(size);
    }

    // We create the session globals on the main thread, then create the thread
    // pool. Upon creation, each worker thread created gets a copy of the
    // session globals in TLS. This is possible because `SessionGlobals` impls
    // `Send` in the parallel compiler.
    rustc_span::create_session_globals_then(edition, || {
        rustc_span::with_session_globals(|session_globals| {
            builder
                .build_scoped(
                    // Initialize each new worker thread when created.
                    move |thread: rayon::ThreadBuilder| {
                        rustc_span::set_session_globals_then(session_globals, || thread.run())
                    },
                    // Run `f` on the first thread in the thread pool.
                    move |pool: &rayon::ThreadPool| pool.install(f),
                )
                .unwrap()
        })
    })
}

fn load_backend_from_dylib(path: &Path) -> MakeBackendFn {
    let lib = unsafe { Library::new(path) }.unwrap_or_else(|err| {
        let err = format!("couldn't load codegen backend {path:?}: {err}");
        early_error(ErrorOutputType::default(), &err);
    });

    let backend_sym = unsafe { lib.get::<MakeBackendFn>(b"__rustc_codegen_backend") }
        .unwrap_or_else(|e| {
            let err = format!("couldn't load codegen backend: {e}");
            early_error(ErrorOutputType::default(), &err);
        });

    // Intentionally leak the dynamic library. We can't ever unload it
    // since the library can make things that will live arbitrarily long.
    let backend_sym = unsafe { backend_sym.into_raw() };
    mem::forget(lib);

    *backend_sym
}

/// Get the codegen backend based on the name and specified sysroot.
///
/// A name of `None` indicates that the default backend should be used.
pub fn get_codegen_backend(
    maybe_sysroot: &Option<PathBuf>,
    backend_name: Option<&str>,
) -> Box<dyn CodegenBackend> {
    static LOAD: OnceLock<unsafe fn() -> Box<dyn CodegenBackend>> = OnceLock::new();

    let load = LOAD.get_or_init(|| {
        let default_codegen_backend = option_env!("CFG_DEFAULT_CODEGEN_BACKEND").unwrap_or("llvm");

        match backend_name.unwrap_or(default_codegen_backend) {
            filename if filename.contains('.') => load_backend_from_dylib(filename.as_ref()),
            #[cfg(feature = "llvm")]
            "llvm" => rustc_codegen_llvm::LlvmCodegenBackend::new,
            backend_name => get_codegen_sysroot(maybe_sysroot, backend_name),
        }
    });

    // SAFETY: In case of a builtin codegen backend this is safe. In case of an external codegen
    // backend we hope that the backend links against the same rustc_driver version. If this is not
    // the case, we get UB.
    unsafe { load() }
}

// This is used for rustdoc, but it uses similar machinery to codegen backend
// loading, so we leave the code here. It is potentially useful for other tools
// that want to invoke the rustc binary while linking to rustc as well.
pub fn rustc_path<'a>() -> Option<&'a Path> {
    static RUSTC_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

    const BIN_PATH: &str = env!("RUSTC_INSTALL_BINDIR");

    RUSTC_PATH.get_or_init(|| get_rustc_path_inner(BIN_PATH)).as_deref()
}

fn get_rustc_path_inner(bin_path: &str) -> Option<PathBuf> {
    sysroot_candidates().iter().find_map(|sysroot| {
        let candidate = sysroot.join(bin_path).join(if cfg!(target_os = "windows") {
            "rustc.exe"
        } else {
            "rustc"
        });
        candidate.exists().then_some(candidate)
    })
}

fn get_codegen_sysroot(maybe_sysroot: &Option<PathBuf>, backend_name: &str) -> MakeBackendFn {
    // For now we only allow this function to be called once as it'll dlopen a
    // few things, which seems to work best if we only do that once. In
    // general this assertion never trips due to the once guard in `get_codegen_backend`,
    // but there's a few manual calls to this function in this file we protect
    // against.
    static LOADED: AtomicBool = AtomicBool::new(false);
    assert!(
        !LOADED.fetch_or(true, Ordering::SeqCst),
        "cannot load the default codegen backend twice"
    );

    let target = session::config::host_triple();
    let sysroot_candidates = sysroot_candidates();

    let sysroot = maybe_sysroot
        .iter()
        .chain(sysroot_candidates.iter())
        .map(|sysroot| {
            filesearch::make_target_lib_path(sysroot, target).with_file_name("codegen-backends")
        })
        .find(|f| {
            info!("codegen backend candidate: {}", f.display());
            f.exists()
        });
    let sysroot = sysroot.unwrap_or_else(|| {
        let candidates = sysroot_candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join("\n* ");
        let err = format!(
            "failed to find a `codegen-backends` folder \
                           in the sysroot candidates:\n* {candidates}"
        );
        early_error(ErrorOutputType::default(), &err);
    });
    info!("probing {} for a codegen backend", sysroot.display());

    let d = sysroot.read_dir().unwrap_or_else(|e| {
        let err = format!(
            "failed to load default codegen backend, couldn't \
                           read `{}`: {}",
            sysroot.display(),
            e
        );
        early_error(ErrorOutputType::default(), &err);
    });

    let mut file: Option<PathBuf> = None;

    let expected_names = &[
        format!("rustc_codegen_{}-{}", backend_name, env!("CFG_RELEASE")),
        format!("rustc_codegen_{backend_name}"),
    ];
    for entry in d.filter_map(|e| e.ok()) {
        let path = entry.path();
        let Some(filename) = path.file_name().and_then(|s| s.to_str()) else { continue };
        if !(filename.starts_with(DLL_PREFIX) && filename.ends_with(DLL_SUFFIX)) {
            continue;
        }
        let name = &filename[DLL_PREFIX.len()..filename.len() - DLL_SUFFIX.len()];
        if !expected_names.iter().any(|expected| expected == name) {
            continue;
        }
        if let Some(ref prev) = file {
            let err = format!(
                "duplicate codegen backends found\n\
                               first:  {}\n\
                               second: {}\n\
            ",
                prev.display(),
                path.display()
            );
            early_error(ErrorOutputType::default(), &err);
        }
        file = Some(path.clone());
    }

    match file {
        Some(ref s) => load_backend_from_dylib(s),
        None => {
            let err = format!("unsupported builtin codegen backend `{backend_name}`");
            early_error(ErrorOutputType::default(), &err);
        }
    }
}

pub(crate) fn check_attr_crate_type(
    sess: &Session,
    attrs: &[ast::Attribute],
    lint_buffer: &mut LintBuffer,
) {
    // Unconditionally collect crate types from attributes to make them used
    for a in attrs.iter() {
        if a.has_name(sym::crate_type) {
            if let Some(n) = a.value_str() {
                if categorize_crate_type(n).is_some() {
                    return;
                }

                if let ast::MetaItemKind::NameValue(spanned) = a.meta_kind().unwrap() {
                    let span = spanned.span;
                    let lev_candidate = find_best_match_for_name(
                        &CRATE_TYPES.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
                        n,
                        None,
                    );
                    if let Some(candidate) = lev_candidate {
                        lint_buffer.buffer_lint_with_diagnostic(
                            lint::builtin::UNKNOWN_CRATE_TYPES,
                            ast::CRATE_NODE_ID,
                            span,
                            "invalid `crate_type` value",
                            BuiltinLintDiagnostics::UnknownCrateTypes(
                                span,
                                "did you mean".to_string(),
                                format!("\"{candidate}\""),
                            ),
                        );
                    } else {
                        lint_buffer.buffer_lint(
                            lint::builtin::UNKNOWN_CRATE_TYPES,
                            ast::CRATE_NODE_ID,
                            span,
                            "invalid `crate_type` value",
                        );
                    }
                }
            } else {
                // This is here mainly to check for using a macro, such as
                // #![crate_type = foo!()]. That is not supported since the
                // crate type needs to be known very early in compilation long
                // before expansion. Otherwise, validation would normally be
                // caught in AstValidator (via `check_builtin_attribute`), but
                // by the time that runs the macro is expanded, and it doesn't
                // give an error.
                validate_attr::emit_fatal_malformed_builtin_attribute(
                    &sess.parse_sess,
                    a,
                    sym::crate_type,
                );
            }
        }
    }
}

const CRATE_TYPES: &[(Symbol, CrateType)] = &[
    (sym::rlib, CrateType::Rlib),
    (sym::dylib, CrateType::Dylib),
    (sym::cdylib, CrateType::Cdylib),
    (sym::lib, config::default_lib_output()),
    (sym::staticlib, CrateType::Staticlib),
    (sym::proc_dash_macro, CrateType::ProcMacro),
    (sym::bin, CrateType::Executable),
];

fn categorize_crate_type(s: Symbol) -> Option<CrateType> {
    Some(CRATE_TYPES.iter().find(|(key, _)| *key == s)?.1)
}

pub fn collect_crate_types(session: &Session, attrs: &[ast::Attribute]) -> Vec<CrateType> {
    // Unconditionally collect crate types from attributes to make them used
    let attr_types: Vec<CrateType> = attrs
        .iter()
        .filter_map(|a| {
            if a.has_name(sym::crate_type) {
                match a.value_str() {
                    Some(s) => categorize_crate_type(s),
                    _ => None,
                }
            } else {
                None
            }
        })
        .collect();

    // If we're generating a test executable, then ignore all other output
    // styles at all other locations
    if session.opts.test {
        return vec![CrateType::Executable];
    }

    // Only check command line flags if present. If no types are specified by
    // command line, then reuse the empty `base` Vec to hold the types that
    // will be found in crate attributes.
    // JUSTIFICATION: before wrapper fn is available
    #[allow(rustc::bad_opt_access)]
    let mut base = session.opts.crate_types.clone();
    if base.is_empty() {
        base.extend(attr_types);
        if base.is_empty() {
            base.push(output::default_output_for_target(session));
        } else {
            base.sort();
            base.dedup();
        }
    }

    base.retain(|crate_type| {
        if output::invalid_output_for_target(session, *crate_type) {
            session.emit_warning(errors::UnsupportedCrateTypeForTarget {
                crate_type: *crate_type,
                target_triple: &session.opts.target_triple,
            });
            false
        } else {
            true
        }
    });

    base
}

pub fn build_output_filenames(attrs: &[ast::Attribute], sess: &Session) -> OutputFilenames {
    match sess.io.output_file {
        None => {
            // "-" as input file will cause the parser to read from stdin so we
            // have to make up a name
            // We want to toss everything after the final '.'
            let dirpath = sess.io.output_dir.clone().unwrap_or_default();

            // If a crate name is present, we use it as the link name
            let stem = sess
                .opts
                .crate_name
                .clone()
                .or_else(|| rustc_attr::find_crate_name(sess, attrs).map(|n| n.to_string()))
                .unwrap_or_else(|| sess.io.input.filestem().to_owned());

            OutputFilenames::new(
                dirpath,
                stem,
                None,
                sess.io.temps_dir.clone(),
                sess.opts.cg.extra_filename.clone(),
                sess.opts.output_types.clone(),
            )
        }

        Some(ref out_file) => {
            let unnamed_output_types =
                sess.opts.output_types.values().filter(|a| a.is_none()).count();
            let ofile = if unnamed_output_types > 1 {
                sess.emit_warning(errors::MultipleOutputTypesAdaption);
                None
            } else {
                if !sess.opts.cg.extra_filename.is_empty() {
                    sess.emit_warning(errors::IgnoringExtraFilename);
                }
                Some(out_file.clone())
            };
            if sess.io.output_dir != None {
                sess.emit_warning(errors::IgnoringOutDir);
            }

            OutputFilenames::new(
                out_file.parent().unwrap_or_else(|| Path::new("")).to_path_buf(),
                out_file.file_stem().unwrap_or_default().to_str().unwrap().to_string(),
                ofile,
                sess.io.temps_dir.clone(),
                sess.opts.cg.extra_filename.clone(),
                sess.opts.output_types.clone(),
            )
        }
    }
}

/// Returns a version string such as "1.46.0 (04488afe3 2020-08-24)" when invoked by an in-tree tool.
pub macro version_str() {
    option_env!("CFG_VERSION")
}

/// Returns the version string for `rustc` itself (which may be different from a tool version).
pub fn rustc_version_str() -> Option<&'static str> {
    version_str!()
}
