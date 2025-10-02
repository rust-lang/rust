use std::any::Any;
use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::{env, thread};

use rustc_ast as ast;
use rustc_attr_parsing::{ShouldEmit, validate_attr};
use rustc_codegen_ssa::back::archive::ArArchiveBuilderBuilder;
use rustc_codegen_ssa::back::link::link_binary;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::{CodegenResults, CrateInfo};
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::jobserver::Proxy;
use rustc_data_structures::sync;
use rustc_errors::LintBuffer;
use rustc_metadata::{DylibError, EncodedMetadata, load_symbol_from_dylib};
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::{CurrentGcx, TyCtxt};
use rustc_session::config::{
    Cfg, CrateType, OutFileName, OutputFilenames, OutputTypes, Sysroot, host_tuple,
};
use rustc_session::output::{CRATE_TYPES, categorize_crate_type};
use rustc_session::{EarlyDiagCtxt, Session, filesearch, lint};
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::edition::Edition;
use rustc_span::source_map::SourceMapInputs;
use rustc_span::{SessionGlobals, Symbol, sym};
use rustc_target::spec::Target;
use tracing::info;

use crate::errors;
use crate::passes::parse_crate_name;

/// Function pointer type that constructs a new CodegenBackend.
type MakeBackendFn = fn() -> Box<dyn CodegenBackend>;

/// Adds `target_feature = "..."` cfgs for a variety of platform
/// specific features (SSE, NEON etc.).
///
/// This is performed by checking whether a set of permitted features
/// is available on the target machine, by querying the codegen backend.
pub(crate) fn add_configuration(
    cfg: &mut Cfg,
    sess: &mut Session,
    codegen_backend: &dyn CodegenBackend,
) {
    let tf = sym::target_feature;
    let tf_cfg = codegen_backend.target_config(sess);

    sess.unstable_target_features.extend(tf_cfg.unstable_target_features.iter().copied());
    sess.target_features.extend(tf_cfg.target_features.iter().copied());

    cfg.extend(tf_cfg.target_features.into_iter().map(|feat| (tf, Some(feat))));

    if tf_cfg.has_reliable_f16 {
        cfg.insert((sym::target_has_reliable_f16, None));
    }
    if tf_cfg.has_reliable_f16_math {
        cfg.insert((sym::target_has_reliable_f16_math, None));
    }
    if tf_cfg.has_reliable_f128 {
        cfg.insert((sym::target_has_reliable_f128, None));
    }
    if tf_cfg.has_reliable_f128_math {
        cfg.insert((sym::target_has_reliable_f128_math, None));
    }

    if sess.crt_static(None) {
        cfg.insert((tf, Some(sym::crt_dash_static)));
    }
}

/// Ensures that all target features required by the ABI are present.
/// Must be called after `unstable_target_features` has been populated!
pub(crate) fn check_abi_required_features(sess: &Session) {
    let abi_feature_constraints = sess.target.abi_required_features();
    // We check this against `unstable_target_features` as that is conveniently already
    // back-translated to rustc feature names, taking into account `-Ctarget-cpu` and `-Ctarget-feature`.
    // Just double-check that the features we care about are actually on our list.
    for feature in
        abi_feature_constraints.required.iter().chain(abi_feature_constraints.incompatible.iter())
    {
        assert!(
            sess.target.rust_target_features().iter().any(|(name, ..)| feature == name),
            "target feature {feature} is required/incompatible for the current ABI but not a recognized feature for this target"
        );
    }

    for feature in abi_feature_constraints.required {
        if !sess.unstable_target_features.contains(&Symbol::intern(feature)) {
            sess.dcx().emit_warn(errors::AbiRequiredTargetFeature { feature, enabled: "enabled" });
        }
    }
    for feature in abi_feature_constraints.incompatible {
        if sess.unstable_target_features.contains(&Symbol::intern(feature)) {
            sess.dcx().emit_warn(errors::AbiRequiredTargetFeature { feature, enabled: "disabled" });
        }
    }
}

pub static STACK_SIZE: OnceLock<usize> = OnceLock::new();
pub const DEFAULT_STACK_SIZE: usize = 8 * 1024 * 1024;

fn init_stack_size(early_dcx: &EarlyDiagCtxt) -> usize {
    // Obey the environment setting or default
    *STACK_SIZE.get_or_init(|| {
        env::var_os("RUST_MIN_STACK")
            .as_ref()
            .map(|os_str| os_str.to_string_lossy())
            // if someone finds out `export RUST_MIN_STACK=640000` isn't enough stack
            // they might try to "unset" it by running `RUST_MIN_STACK=  rustc code.rs`
            // this is wrong, but std would nonetheless "do what they mean", so let's do likewise
            .filter(|s| !s.trim().is_empty())
            // rustc is a batch program, so error early on inputs which are unlikely to be intended
            // so no one thinks we parsed them setting `RUST_MIN_STACK="64 megabytes"`
            // FIXME: we could accept `RUST_MIN_STACK=64MB`, perhaps?
            .map(|s| {
                let s = s.trim();
                // FIXME(workingjubilee): add proper diagnostics when we factor out "pre-run" setup
                #[allow(rustc::untranslatable_diagnostic, rustc::diagnostic_outside_of_impl)]
                s.parse::<usize>().unwrap_or_else(|_| {
                    let mut err = early_dcx.early_struct_fatal(format!(
                        r#"`RUST_MIN_STACK` should be a number of bytes, but was "{s}""#,
                    ));
                    err.note("you can also unset `RUST_MIN_STACK` to use the default stack size");
                    err.emit()
                })
            })
            // otherwise pick a consistent default
            .unwrap_or(DEFAULT_STACK_SIZE)
    })
}

fn run_in_thread_with_globals<F: FnOnce(CurrentGcx, Arc<Proxy>) -> R + Send, R: Send>(
    thread_stack_size: usize,
    edition: Edition,
    sm_inputs: SourceMapInputs,
    extra_symbols: &[&'static str],
    f: F,
) -> R {
    // The "thread pool" is a single spawned thread in the non-parallel
    // compiler. We run on a spawned thread instead of the main thread (a) to
    // provide control over the stack size, and (b) to increase similarity with
    // the parallel compiler, in particular to ensure there is no accidental
    // sharing of data between the main thread and the compilation thread
    // (which might cause problems for the parallel compiler).
    let builder = thread::Builder::new().name("rustc".to_string()).stack_size(thread_stack_size);

    // We build the session globals and run `f` on the spawned thread, because
    // `SessionGlobals` does not impl `Send` in the non-parallel compiler.
    thread::scope(|s| {
        // `unwrap` is ok here because `spawn_scoped` only panics if the thread
        // name contains null bytes.
        let r = builder
            .spawn_scoped(s, move || {
                rustc_span::create_session_globals_then(
                    edition,
                    extra_symbols,
                    Some(sm_inputs),
                    || f(CurrentGcx::new(), Proxy::new()),
                )
            })
            .unwrap()
            .join();

        match r {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        }
    })
}

pub(crate) fn run_in_thread_pool_with_globals<
    F: FnOnce(CurrentGcx, Arc<Proxy>) -> R + Send,
    R: Send,
>(
    thread_builder_diag: &EarlyDiagCtxt,
    edition: Edition,
    threads: usize,
    extra_symbols: &[&'static str],
    sm_inputs: SourceMapInputs,
    f: F,
) -> R {
    use std::process;

    use rustc_data_structures::defer;
    use rustc_data_structures::sync::FromDyn;
    use rustc_middle::ty::tls;
    use rustc_query_impl::QueryCtxt;
    use rustc_query_system::query::{QueryContext, break_query_cycles};

    let thread_stack_size = init_stack_size(thread_builder_diag);

    let registry = sync::Registry::new(std::num::NonZero::new(threads).unwrap());

    if !sync::is_dyn_thread_safe() {
        return run_in_thread_with_globals(
            thread_stack_size,
            edition,
            sm_inputs,
            extra_symbols,
            |current_gcx, jobserver_proxy| {
                // Register the thread for use with the `WorkerLocal` type.
                registry.register();

                f(current_gcx, jobserver_proxy)
            },
        );
    }

    let current_gcx = FromDyn::from(CurrentGcx::new());
    let current_gcx2 = current_gcx.clone();

    let proxy = Proxy::new();

    let proxy_ = Arc::clone(&proxy);
    let proxy__ = Arc::clone(&proxy);
    let builder = rustc_thread_pool::ThreadPoolBuilder::new()
        .thread_name(|_| "rustc".to_string())
        .acquire_thread_handler(move || proxy_.acquire_thread())
        .release_thread_handler(move || proxy__.release_thread())
        .num_threads(threads)
        .deadlock_handler(move || {
            // On deadlock, creates a new thread and forwards information in thread
            // locals to it. The new thread runs the deadlock handler.

            let current_gcx2 = current_gcx2.clone();
            let registry = rustc_thread_pool::Registry::current();
            let session_globals = rustc_span::with_session_globals(|session_globals| {
                session_globals as *const SessionGlobals as usize
            });
            thread::Builder::new()
                .name("rustc query cycle handler".to_string())
                .spawn(move || {
                    let on_panic = defer(|| {
                        eprintln!("internal compiler error: query cycle handler thread panicked, aborting process");
                        // We need to abort here as we failed to resolve the deadlock,
                        // otherwise the compiler could just hang,
                        process::abort();
                    });

                    // Get a `GlobalCtxt` reference from `CurrentGcx` as we cannot rely on having a
                    // `TyCtxt` TLS reference here.
                    current_gcx2.access(|gcx| {
                        tls::enter_context(&tls::ImplicitCtxt::new(gcx), || {
                            tls::with(|tcx| {
                                // Accessing session globals is sound as they outlive `GlobalCtxt`.
                                // They are needed to hash query keys containing spans or symbols.
                                let query_map = rustc_span::set_session_globals_then(unsafe { &*(session_globals as *const SessionGlobals) }, || {
                                    // Ensure there was no errors collecting all active jobs.
                                    // We need the complete map to ensure we find a cycle to break.
                                    QueryCtxt::new(tcx).collect_active_jobs().expect("failed to collect active queries in deadlock handler")
                                });
                                break_query_cycles(query_map, &registry);
                            })
                        })
                    });

                    on_panic.disable();
                })
                .unwrap();
        })
        .stack_size(thread_stack_size);

    // We create the session globals on the main thread, then create the thread
    // pool. Upon creation, each worker thread created gets a copy of the
    // session globals in TLS. This is possible because `SessionGlobals` impls
    // `Send` in the parallel compiler.
    rustc_span::create_session_globals_then(edition, extra_symbols, Some(sm_inputs), || {
        rustc_span::with_session_globals(|session_globals| {
            let session_globals = FromDyn::from(session_globals);
            builder
                .build_scoped(
                    // Initialize each new worker thread when created.
                    move |thread: rustc_thread_pool::ThreadBuilder| {
                        // Register the thread for use with the `WorkerLocal` type.
                        registry.register();

                        rustc_span::set_session_globals_then(session_globals.into_inner(), || {
                            thread.run()
                        })
                    },
                    // Run `f` on the first thread in the thread pool.
                    move |pool: &rustc_thread_pool::ThreadPool| {
                        pool.install(|| f(current_gcx.into_inner(), proxy))
                    },
                )
                .unwrap()
        })
    })
}

#[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
fn load_backend_from_dylib(early_dcx: &EarlyDiagCtxt, path: &Path) -> MakeBackendFn {
    match unsafe { load_symbol_from_dylib::<MakeBackendFn>(path, "__rustc_codegen_backend") } {
        Ok(backend_sym) => backend_sym,
        Err(DylibError::DlOpen(path, err)) => {
            let err = format!("couldn't load codegen backend {path}{err}");
            early_dcx.early_fatal(err);
        }
        Err(DylibError::DlSym(_path, err)) => {
            let e = format!(
                "`__rustc_codegen_backend` symbol lookup in the codegen backend failed{err}",
            );
            early_dcx.early_fatal(e);
        }
    }
}

/// Get the codegen backend based on the name and specified sysroot.
///
/// A name of `None` indicates that the default backend should be used.
pub fn get_codegen_backend(
    early_dcx: &EarlyDiagCtxt,
    sysroot: &Sysroot,
    backend_name: Option<&str>,
    target: &Target,
) -> Box<dyn CodegenBackend> {
    static LOAD: OnceLock<unsafe fn() -> Box<dyn CodegenBackend>> = OnceLock::new();

    let load = LOAD.get_or_init(|| {
        let backend = backend_name
            .or(target.default_codegen_backend.as_deref())
            .or(option_env!("CFG_DEFAULT_CODEGEN_BACKEND"))
            .unwrap_or("dummy");

        match backend {
            filename if filename.contains('.') => {
                load_backend_from_dylib(early_dcx, filename.as_ref())
            }
            "dummy" => || Box::new(DummyCodegenBackend),
            #[cfg(feature = "llvm")]
            "llvm" => rustc_codegen_llvm::LlvmCodegenBackend::new,
            backend_name => get_codegen_sysroot(early_dcx, sysroot, backend_name),
        }
    });

    // SAFETY: In case of a builtin codegen backend this is safe. In case of an external codegen
    // backend we hope that the backend links against the same rustc_driver version. If this is not
    // the case, we get UB.
    unsafe { load() }
}

struct DummyCodegenBackend;

impl CodegenBackend for DummyCodegenBackend {
    fn locale_resource(&self) -> &'static str {
        ""
    }

    fn name(&self) -> &'static str {
        "dummy"
    }

    fn codegen_crate<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Box<dyn Any> {
        Box::new(CodegenResults {
            modules: vec![],
            allocator_module: None,
            crate_info: CrateInfo::new(tcx, String::new()),
        })
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        _sess: &Session,
        _outputs: &OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        (*ongoing_codegen.downcast().unwrap(), FxIndexMap::default())
    }

    fn link(
        &self,
        sess: &Session,
        codegen_results: CodegenResults,
        metadata: EncodedMetadata,
        outputs: &OutputFilenames,
    ) {
        // JUSTIFICATION: TyCtxt no longer available here
        #[allow(rustc::bad_opt_access)]
        if sess.opts.crate_types.iter().any(|&crate_type| crate_type != CrateType::Rlib) {
            #[allow(rustc::untranslatable_diagnostic)]
            #[allow(rustc::diagnostic_outside_of_impl)]
            sess.dcx().fatal(format!(
                "crate type {} not supported by the dummy codegen backend",
                sess.opts.crate_types[0],
            ));
        }

        link_binary(
            sess,
            &ArArchiveBuilderBuilder,
            codegen_results,
            metadata,
            outputs,
            self.name(),
        );
    }
}

// This is used for rustdoc, but it uses similar machinery to codegen backend
// loading, so we leave the code here. It is potentially useful for other tools
// that want to invoke the rustc binary while linking to rustc as well.
pub fn rustc_path<'a>(sysroot: &Sysroot) -> Option<&'a Path> {
    static RUSTC_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

    RUSTC_PATH
        .get_or_init(|| {
            let candidate = sysroot
                .default
                .join(env!("RUSTC_INSTALL_BINDIR"))
                .join(if cfg!(target_os = "windows") { "rustc.exe" } else { "rustc" });
            candidate.exists().then_some(candidate)
        })
        .as_deref()
}

#[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
fn get_codegen_sysroot(
    early_dcx: &EarlyDiagCtxt,
    sysroot: &Sysroot,
    backend_name: &str,
) -> MakeBackendFn {
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

    let target = host_tuple();

    let sysroot = sysroot
        .all_paths()
        .map(|sysroot| {
            filesearch::make_target_lib_path(sysroot, target).with_file_name("codegen-backends")
        })
        .find(|f| {
            info!("codegen backend candidate: {}", f.display());
            f.exists()
        })
        .unwrap_or_else(|| {
            let candidates = sysroot
                .all_paths()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join("\n* ");
            let err = format!(
                "failed to find a `codegen-backends` folder in the sysroot candidates:\n\
                 * {candidates}"
            );
            early_dcx.early_fatal(err);
        });

    info!("probing {} for a codegen backend", sysroot.display());

    let d = sysroot.read_dir().unwrap_or_else(|e| {
        let err = format!(
            "failed to load default codegen backend, couldn't read `{}`: {e}",
            sysroot.display(),
        );
        early_dcx.early_fatal(err);
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
            early_dcx.early_fatal(err);
        }
        file = Some(path.clone());
    }

    match file {
        Some(ref s) => load_backend_from_dylib(early_dcx, s),
        None => {
            let err = format!("unsupported builtin codegen backend `{backend_name}`");
            early_dcx.early_fatal(err);
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
                    let candidate = find_best_match_for_name(
                        &CRATE_TYPES.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
                        n,
                        None,
                    );
                    lint_buffer.buffer_lint(
                        lint::builtin::UNKNOWN_CRATE_TYPES,
                        ast::CRATE_NODE_ID,
                        span,
                        errors::UnknownCrateTypes {
                            sugg: candidate
                                .map(|cand| errors::UnknownCrateTypesSub { span, snippet: cand }),
                        },
                    );
                }
            } else {
                // This is here mainly to check for using a macro, such as
                // `#![crate_type = foo!()]`. That is not supported since the
                // crate type needs to be known very early in compilation long
                // before expansion. Otherwise, validation would normally be
                // caught during semantic analysis via `TyCtxt::check_mod_attrs`,
                // but by the time that runs the macro is expanded, and it doesn't
                // give an error.
                validate_attr::emit_fatal_malformed_builtin_attribute(
                    &sess.psess,
                    a,
                    sym::crate_type,
                );
            }
        }
    }
}

fn multiple_output_types_to_stdout(
    output_types: &OutputTypes,
    single_output_file_is_stdout: bool,
) -> bool {
    use std::io::IsTerminal;
    if std::io::stdout().is_terminal() {
        // If stdout is a tty, check if multiple text output types are
        // specified by `--emit foo=- --emit bar=-` or `-o - --emit foo,bar`
        let named_text_types = output_types
            .iter()
            .filter(|(f, o)| f.is_text_output() && *o == &Some(OutFileName::Stdout))
            .count();
        let unnamed_text_types =
            output_types.iter().filter(|(f, o)| f.is_text_output() && o.is_none()).count();
        named_text_types > 1 || unnamed_text_types > 1 && single_output_file_is_stdout
    } else {
        // Otherwise, all the output types should be checked
        let named_types =
            output_types.values().filter(|o| *o == &Some(OutFileName::Stdout)).count();
        let unnamed_types = output_types.values().filter(|o| o.is_none()).count();
        named_types > 1 || unnamed_types > 1 && single_output_file_is_stdout
    }
}

pub fn build_output_filenames(attrs: &[ast::Attribute], sess: &Session) -> OutputFilenames {
    if multiple_output_types_to_stdout(
        &sess.opts.output_types,
        sess.io.output_file == Some(OutFileName::Stdout),
    ) {
        sess.dcx().emit_fatal(errors::MultipleOutputTypesToStdout);
    }

    let crate_name =
        sess.opts.crate_name.clone().or_else(|| {
            parse_crate_name(sess, attrs, ShouldEmit::Nothing).map(|i| i.0.to_string())
        });

    match sess.io.output_file {
        None => {
            // "-" as input file will cause the parser to read from stdin so we
            // have to make up a name
            // We want to toss everything after the final '.'
            let dirpath = sess.io.output_dir.clone().unwrap_or_default();

            // If a crate name is present, we use it as the link name
            let stem = crate_name.clone().unwrap_or_else(|| sess.io.input.filestem().to_owned());

            OutputFilenames::new(
                dirpath,
                crate_name.unwrap_or_else(|| stem.replace('-', "_")),
                stem,
                None,
                sess.io.temps_dir.clone(),
                sess.opts.unstable_opts.split_dwarf_out_dir.clone(),
                sess.opts.cg.extra_filename.clone(),
                sess.opts.output_types.clone(),
            )
        }

        Some(ref out_file) => {
            let unnamed_output_types =
                sess.opts.output_types.values().filter(|a| a.is_none()).count();
            let ofile = if unnamed_output_types > 1 {
                sess.dcx().emit_warn(errors::MultipleOutputTypesAdaption);
                None
            } else {
                if !sess.opts.cg.extra_filename.is_empty() {
                    sess.dcx().emit_warn(errors::IgnoringExtraFilename);
                }
                Some(out_file.clone())
            };
            if sess.io.output_dir.is_some() {
                sess.dcx().emit_warn(errors::IgnoringOutDir);
            }

            let out_filestem =
                out_file.filestem().unwrap_or_default().to_str().unwrap().to_string();
            OutputFilenames::new(
                out_file.parent().unwrap_or_else(|| Path::new("")).to_path_buf(),
                crate_name.unwrap_or_else(|| out_filestem.replace('-', "_")),
                out_filestem,
                ofile,
                sess.io.temps_dir.clone(),
                sess.opts.unstable_opts.split_dwarf_out_dir.clone(),
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
