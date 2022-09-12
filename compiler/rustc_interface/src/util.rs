use info;
use libloading::Library;
use rustc_ast as ast;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
#[cfg(parallel_compiler)]
use rustc_data_structures::jobserver;
use rustc_data_structures::sync::Lrc;
use rustc_errors::registry::Registry;
#[cfg(parallel_compiler)]
use rustc_middle::ty::tls;
use rustc_parse::validate_attr;
#[cfg(parallel_compiler)]
use rustc_query_impl::{QueryContext, QueryCtxt};
use rustc_session as session;
use rustc_session::config::CheckCfg;
use rustc_session::config::{self, CrateType};
use rustc_session::config::{ErrorOutputType, Input, OutputFilenames};
use rustc_session::lint::{self, BuiltinLintDiagnostics, LintBuffer};
use rustc_session::parse::CrateConfig;
use rustc_session::{early_error, filesearch, output, DiagnosticOutput, Session};
use rustc_span::edition::Edition;
use rustc_span::lev_distance::find_best_match_for_name;
use rustc_span::source_map::FileLoader;
use rustc_span::symbol::{sym, Symbol};
use std::env;
use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
use std::mem;
#[cfg(not(parallel_compiler))]
use std::panic;
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
    diagnostic_output: DiagnosticOutput,
    file_loader: Option<Box<dyn FileLoader + Send + Sync + 'static>>,
    input_path: Option<PathBuf>,
    lint_caps: FxHashMap<lint::LintId, lint::Level>,
    make_codegen_backend: Option<
        Box<dyn FnOnce(&config::Options) -> Box<dyn CodegenBackend> + Send>,
    >,
    descriptions: Registry,
) -> (Lrc<Session>, Lrc<Box<dyn CodegenBackend>>) {
    let codegen_backend = if let Some(make_codegen_backend) = make_codegen_backend {
        make_codegen_backend(&sopts)
    } else {
        get_codegen_backend(
            &sopts.maybe_sysroot,
            sopts.unstable_opts.codegen_backend.as_ref().map(|name| &name[..]),
        )
    };

    // target_override is documented to be called before init(), so this is okay
    let target_override = codegen_backend.target_override(&sopts);

    let bundle = match rustc_errors::fluent_bundle(
        sopts.maybe_sysroot.clone(),
        sysroot_candidates(),
        sopts.unstable_opts.translate_lang.clone(),
        sopts.unstable_opts.translate_additional_ftl.as_deref(),
        sopts.unstable_opts.translate_directionality_markers,
    ) {
        Ok(bundle) => bundle,
        Err(e) => {
            early_error(sopts.error_format, &format!("failed to load fluent bundle: {e}"));
        }
    };

    let mut sess = session::build_session(
        sopts,
        input_path,
        bundle,
        descriptions,
        diagnostic_output,
        lint_caps,
        file_loader,
        target_override,
    );

    codegen_backend.init(&sess);

    let mut cfg = config::build_configuration(&sess, config::to_crate_config(cfg));
    add_configuration(&mut cfg, &mut sess, &*codegen_backend);

    let mut check_cfg = config::to_crate_check_config(check_cfg);
    check_cfg.fill_well_known();

    sess.parse_sess.config = cfg;
    sess.parse_sess.check_config = check_cfg;

    (Lrc::new(sess), Lrc::new(codegen_backend))
}

const STACK_SIZE: usize = 8 * 1024 * 1024;

fn get_stack_size() -> Option<usize> {
    // FIXME: Hacks on hacks. If the env is trying to override the stack size
    // then *don't* set it explicitly.
    env::var_os("RUST_MIN_STACK").is_none().then_some(STACK_SIZE)
}

/// Like a `thread::Builder::spawn` followed by a `join()`, but avoids the need
/// for `'static` bounds.
#[cfg(not(parallel_compiler))]
fn scoped_thread<F: FnOnce() -> R + Send, R: Send>(cfg: thread::Builder, f: F) -> R {
    // SAFETY: join() is called immediately, so any closure captures are still
    // alive.
    match unsafe { cfg.spawn_unchecked(f) }.unwrap().join() {
        Ok(v) => v,
        Err(e) => panic::resume_unwind(e),
    }
}

#[cfg(not(parallel_compiler))]
pub fn run_in_thread_pool_with_globals<F: FnOnce() -> R + Send, R: Send>(
    edition: Edition,
    _threads: usize,
    f: F,
) -> R {
    let mut cfg = thread::Builder::new().name("rustc".to_string());

    if let Some(size) = get_stack_size() {
        cfg = cfg.stack_size(size);
    }

    let main_handler = move || rustc_span::create_session_globals_then(edition, f);

    scoped_thread(cfg, main_handler)
}

/// Creates a new thread and forwards information in thread locals to it.
/// The new thread runs the deadlock handler.
/// Must only be called when a deadlock is about to happen.
#[cfg(parallel_compiler)]
unsafe fn handle_deadlock() {
    let registry = rustc_rayon_core::Registry::current();

    let query_map = tls::with(|tcx| {
        QueryCtxt::from_tcx(tcx)
            .try_collect_active_jobs()
            .expect("active jobs shouldn't be locked in deadlock handler")
    });
    thread::spawn(move || rustc_query_impl::deadlock(query_map, &registry));
}

#[cfg(parallel_compiler)]
pub fn run_in_thread_pool_with_globals<F: FnOnce() -> R + Send, R: Send>(
    edition: Edition,
    threads: usize,
    f: F,
) -> R {
    let mut config = rayon::ThreadPoolBuilder::new()
        .thread_name(|_| "rustc".to_string())
        .acquire_thread_handler(jobserver::acquire_thread)
        .release_thread_handler(jobserver::release_thread)
        .num_threads(threads)
        .deadlock_handler(|| unsafe { handle_deadlock() });

    if let Some(size) = get_stack_size() {
        config = config.stack_size(size);
    }

    let with_pool = move |pool: &rayon::ThreadPool| pool.install(f);

    rustc_span::create_session_globals_then(edition, || {
        rustc_span::with_session_globals(|session_globals| {
            // The main handler runs for each Rayon worker thread and sets up
            // the thread local rustc uses. `session_globals` is captured and set
            // on the new threads.
            let main_handler = move |thread: rayon::ThreadBuilder| {
                rustc_span::set_session_globals_then(session_globals, || thread.run())
            };

            config.build_scoped(main_handler, with_pool).unwrap()
        })
    })
}

fn load_backend_from_dylib(path: &Path) -> MakeBackendFn {
    let lib = unsafe { Library::new(path) }.unwrap_or_else(|err| {
        let err = format!("couldn't load codegen backend {:?}: {}", path, err);
        early_error(ErrorOutputType::default(), &err);
    });

    let backend_sym = unsafe { lib.get::<MakeBackendFn>(b"__rustc_codegen_backend") }
        .unwrap_or_else(|e| {
            let err = format!("couldn't load codegen backend: {}", e);
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

    RUSTC_PATH.get_or_init(|| get_rustc_path_inner(BIN_PATH)).as_ref().map(|v| &**v)
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

fn sysroot_candidates() -> Vec<PathBuf> {
    let target = session::config::host_triple();
    let mut sysroot_candidates = vec![filesearch::get_or_default_sysroot()];
    let path = current_dll_path().and_then(|s| s.canonicalize().ok());
    if let Some(dll) = path {
        // use `parent` twice to chop off the file name and then also the
        // directory containing the dll which should be either `lib` or `bin`.
        if let Some(path) = dll.parent().and_then(|p| p.parent()) {
            // The original `path` pointed at the `rustc_driver` crate's dll.
            // Now that dll should only be in one of two locations. The first is
            // in the compiler's libdir, for example `$sysroot/lib/*.dll`. The
            // other is the target's libdir, for example
            // `$sysroot/lib/rustlib/$target/lib/*.dll`.
            //
            // We don't know which, so let's assume that if our `path` above
            // ends in `$target` we *could* be in the target libdir, and always
            // assume that we may be in the main libdir.
            sysroot_candidates.push(path.to_owned());

            if path.ends_with(target) {
                sysroot_candidates.extend(
                    path.parent() // chop off `$target`
                        .and_then(|p| p.parent()) // chop off `rustlib`
                        .and_then(|p| p.parent()) // chop off `lib`
                        .map(|s| s.to_owned()),
                );
            }
        }
    }

    return sysroot_candidates;

    #[cfg(unix)]
    fn current_dll_path() -> Option<PathBuf> {
        use std::ffi::{CStr, OsStr};
        use std::os::unix::prelude::*;

        unsafe {
            let addr = current_dll_path as usize as *mut _;
            let mut info = mem::zeroed();
            if libc::dladdr(addr, &mut info) == 0 {
                info!("dladdr failed");
                return None;
            }
            if info.dli_fname.is_null() {
                info!("dladdr returned null pointer");
                return None;
            }
            let bytes = CStr::from_ptr(info.dli_fname).to_bytes();
            let os = OsStr::from_bytes(bytes);
            Some(PathBuf::from(os))
        }
    }

    #[cfg(windows)]
    fn current_dll_path() -> Option<PathBuf> {
        use std::ffi::OsString;
        use std::io;
        use std::os::windows::prelude::*;
        use std::ptr;

        use winapi::um::libloaderapi::{
            GetModuleFileNameW, GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
        };

        unsafe {
            let mut module = ptr::null_mut();
            let r = GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                current_dll_path as usize as *mut _,
                &mut module,
            );
            if r == 0 {
                info!("GetModuleHandleExW failed: {}", io::Error::last_os_error());
                return None;
            }
            let mut space = Vec::with_capacity(1024);
            let r = GetModuleFileNameW(module, space.as_mut_ptr(), space.capacity() as u32);
            if r == 0 {
                info!("GetModuleFileNameW failed: {}", io::Error::last_os_error());
                return None;
            }
            let r = r as usize;
            if r >= space.capacity() {
                info!("our buffer was too small? {}", io::Error::last_os_error());
                return None;
            }
            space.set_len(r);
            let os = OsString::from_wide(&space);
            Some(PathBuf::from(os))
        }
    }
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
                           in the sysroot candidates:\n* {}",
            candidates
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
        format!("rustc_codegen_{}-{}", backend_name, release_str().expect("CFG_RELEASE")),
        format!("rustc_codegen_{}", backend_name),
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
            let err = format!("unsupported builtin codegen backend `{}`", backend_name);
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
                                format!("\"{}\"", candidate),
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
        let res = !output::invalid_output_for_target(session, *crate_type);

        if !res {
            session.warn(&format!(
                "dropping unsupported crate type `{}` for target `{}`",
                *crate_type, session.opts.target_triple
            ));
        }

        res
    });

    base
}

pub fn build_output_filenames(
    input: &Input,
    odir: &Option<PathBuf>,
    ofile: &Option<PathBuf>,
    temps_dir: &Option<PathBuf>,
    attrs: &[ast::Attribute],
    sess: &Session,
) -> OutputFilenames {
    match *ofile {
        None => {
            // "-" as input file will cause the parser to read from stdin so we
            // have to make up a name
            // We want to toss everything after the final '.'
            let dirpath = (*odir).as_ref().cloned().unwrap_or_default();

            // If a crate name is present, we use it as the link name
            let stem = sess
                .opts
                .crate_name
                .clone()
                .or_else(|| rustc_attr::find_crate_name(sess, attrs).map(|n| n.to_string()))
                .unwrap_or_else(|| input.filestem().to_owned());

            OutputFilenames::new(
                dirpath,
                stem,
                None,
                temps_dir.clone(),
                sess.opts.cg.extra_filename.clone(),
                sess.opts.output_types.clone(),
            )
        }

        Some(ref out_file) => {
            let unnamed_output_types =
                sess.opts.output_types.values().filter(|a| a.is_none()).count();
            let ofile = if unnamed_output_types > 1 {
                sess.warn(
                    "due to multiple output types requested, the explicitly specified \
                     output file name will be adapted for each output type",
                );
                None
            } else {
                if !sess.opts.cg.extra_filename.is_empty() {
                    sess.warn("ignoring -C extra-filename flag due to -o flag");
                }
                Some(out_file.clone())
            };
            if *odir != None {
                sess.warn("ignoring --out-dir flag due to -o flag");
            }

            OutputFilenames::new(
                out_file.parent().unwrap_or_else(|| Path::new("")).to_path_buf(),
                out_file.file_stem().unwrap_or_default().to_str().unwrap().to_string(),
                ofile,
                temps_dir.clone(),
                sess.opts.cg.extra_filename.clone(),
                sess.opts.output_types.clone(),
            )
        }
    }
}

/// Returns a version string such as "1.46.0 (04488afe3 2020-08-24)"
pub fn version_str() -> Option<&'static str> {
    option_env!("CFG_VERSION")
}

/// Returns a version string such as "0.12.0-dev".
pub fn release_str() -> Option<&'static str> {
    option_env!("CFG_RELEASE")
}

/// Returns the full SHA1 hash of HEAD of the Git repo from which rustc was built.
pub fn commit_hash_str() -> Option<&'static str> {
    option_env!("CFG_VER_HASH")
}

/// Returns the "commit date" of HEAD of the Git repo from which rustc was built as a static string.
pub fn commit_date_str() -> Option<&'static str> {
    option_env!("CFG_VER_DATE")
}
