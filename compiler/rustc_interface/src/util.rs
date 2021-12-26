use libloading::Library;
use rustc_ast::mut_visit::{visit_clobber, MutVisitor, *};
use rustc_ast::ptr::P;
use rustc_ast::{self as ast, AttrVec, BlockCheckMode};
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
use rustc_query_impl::QueryCtxt;
use rustc_resolve::{self, Resolver};
use rustc_session as session;
use rustc_session::config::{self, CrateType};
use rustc_session::config::{ErrorOutputType, Input, OutputFilenames};
use rustc_session::lint::{self, BuiltinLintDiagnostics, LintBuffer};
use rustc_session::parse::CrateConfig;
use rustc_session::{early_error, filesearch, output, DiagnosticOutput, Session};
use rustc_span::edition::Edition;
use rustc_span::lev_distance::find_best_match_for_name;
use rustc_span::source_map::FileLoader;
use rustc_span::symbol::{sym, Symbol};
use smallvec::SmallVec;
use std::env;
use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
use std::io;
use std::lazy::SyncOnceCell;
use std::mem;
use std::ops::DerefMut;
#[cfg(not(parallel_compiler))]
use std::panic;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use tracing::info;

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

    let target_features = codegen_backend.target_features(sess);
    sess.target_features.extend(target_features.iter().cloned());

    cfg.extend(target_features.into_iter().map(|feat| (tf, Some(feat))));

    if sess.crt_static(None) {
        cfg.insert((tf, Some(sym::crt_dash_static)));
    }
}

pub fn create_session(
    sopts: config::Options,
    cfg: FxHashSet<(String, Option<String>)>,
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
            sopts.debugging_opts.codegen_backend.as_ref().map(|name| &name[..]),
        )
    };

    // target_override is documented to be called before init(), so this is okay
    let target_override = codegen_backend.target_override(&sopts);

    let mut sess = session::build_session(
        sopts,
        input_path,
        descriptions,
        diagnostic_output,
        lint_caps,
        file_loader,
        target_override,
    );

    codegen_backend.init(&sess);

    let mut cfg = config::build_configuration(&sess, config::to_crate_config(cfg));
    add_configuration(&mut cfg, &mut sess, &*codegen_backend);
    sess.parse_sess.config = cfg;

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
pub fn scoped_thread<F: FnOnce() -> R + Send, R: Send>(cfg: thread::Builder, f: F) -> R {
    // SAFETY: join() is called immediately, so any closure captures are still
    // alive.
    match unsafe { cfg.spawn_unchecked(f) }.unwrap().join() {
        Ok(v) => v,
        Err(e) => panic::resume_unwind(e),
    }
}

#[cfg(not(parallel_compiler))]
pub fn setup_callbacks_and_run_in_thread_pool_with_globals<F: FnOnce() -> R + Send, R: Send>(
    edition: Edition,
    _threads: usize,
    stderr: &Option<Arc<Mutex<Vec<u8>>>>,
    f: F,
) -> R {
    let mut cfg = thread::Builder::new().name("rustc".to_string());

    if let Some(size) = get_stack_size() {
        cfg = cfg.stack_size(size);
    }

    crate::callbacks::setup_callbacks();

    let main_handler = move || {
        rustc_span::create_session_globals_then(edition, || {
            io::set_output_capture(stderr.clone());
            f()
        })
    };

    scoped_thread(cfg, main_handler)
}

/// Creates a new thread and forwards information in thread locals to it.
/// The new thread runs the deadlock handler.
/// Must only be called when a deadlock is about to happen.
#[cfg(parallel_compiler)]
unsafe fn handle_deadlock() {
    let registry = rustc_rayon_core::Registry::current();

    let context = tls::get_tlv();
    assert!(context != 0);
    rustc_data_structures::sync::assert_sync::<tls::ImplicitCtxt<'_, '_>>();
    let icx: &tls::ImplicitCtxt<'_, '_> = &*(context as *const tls::ImplicitCtxt<'_, '_>);

    let session_globals = rustc_span::with_session_globals(|sg| sg as *const _);
    let session_globals = &*session_globals;
    thread::spawn(move || {
        tls::enter_context(icx, |_| {
            rustc_span::set_session_globals_then(session_globals, || {
                tls::with(|tcx| QueryCtxt::from_tcx(tcx).deadlock(&registry))
            })
        });
    });
}

#[cfg(parallel_compiler)]
pub fn setup_callbacks_and_run_in_thread_pool_with_globals<F: FnOnce() -> R + Send, R: Send>(
    edition: Edition,
    threads: usize,
    stderr: &Option<Arc<Mutex<Vec<u8>>>>,
    f: F,
) -> R {
    crate::callbacks::setup_callbacks();

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
                rustc_span::set_session_globals_then(session_globals, || {
                    io::set_output_capture(stderr.clone());
                    thread.run()
                })
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
    static LOAD: SyncOnceCell<unsafe fn() -> Box<dyn CodegenBackend>> = SyncOnceCell::new();

    let load = LOAD.get_or_init(|| {
        #[cfg(feature = "llvm")]
        const DEFAULT_CODEGEN_BACKEND: &str = "llvm";

        #[cfg(not(feature = "llvm"))]
        const DEFAULT_CODEGEN_BACKEND: &str = "cranelift";

        match backend_name.unwrap_or(DEFAULT_CODEGEN_BACKEND) {
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
    static RUSTC_PATH: SyncOnceCell<Option<PathBuf>> = SyncOnceCell::new();

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

pub fn get_codegen_sysroot(maybe_sysroot: &Option<PathBuf>, backend_name: &str) -> MakeBackendFn {
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
        let filename = match path.file_name().and_then(|s| s.to_str()) {
            Some(s) => s,
            None => continue,
        };
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

#[cfg(not(target_os = "linux"))]
pub fn non_durable_rename(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::rename(src, dst)
}

/// This function attempts to bypass the auto_da_alloc heuristic implemented by some filesystems
/// such as btrfs and ext4. When renaming over a file that already exists then they will "helpfully"
/// write back the source file before committing the rename in case a developer forgot some of
/// the fsyncs in the open/write/fsync(file)/rename/fsync(dir) dance for atomic file updates.
///
/// To avoid triggering this heuristic we delete the destination first, if it exists.
/// The cost of an extra syscall is much lower than getting descheduled for the sync IO.
#[cfg(target_os = "linux")]
pub fn non_durable_rename(src: &Path, dst: &Path) -> std::io::Result<()> {
    let _ = std::fs::remove_file(dst);
    std::fs::rename(src, dst)
}

/// Replaces function bodies with `loop {}` (an infinite loop). This gets rid of
/// all semantic errors in the body while still satisfying the return type,
/// except in certain cases, see below for more.
///
/// This pass is known as `everybody_loops`. Very punny.
///
/// As of March 2021, `everybody_loops` is only used for the
/// `-Z unpretty=everybody_loops` debugging option.
///
/// FIXME: Currently the `everybody_loops` transformation is not applied to:
///  * `const fn`; support could be added, but hasn't. Originally `const fn`
///    was skipped due to issue #43636 that `loop` was not supported for
///    const evaluation.
///  * `impl Trait`, due to issue #43869 that functions returning impl Trait cannot be diverging.
///    Solving this may require `!` to implement every trait, which relies on the an even more
///    ambitious form of the closed RFC #1637. See also [#34511].
///
/// [#34511]: https://github.com/rust-lang/rust/issues/34511#issuecomment-322340401
pub struct ReplaceBodyWithLoop<'a, 'b> {
    within_static_or_const: bool,
    nested_blocks: Option<Vec<ast::Block>>,
    resolver: &'a mut Resolver<'b>,
}

impl<'a, 'b> ReplaceBodyWithLoop<'a, 'b> {
    pub fn new(resolver: &'a mut Resolver<'b>) -> ReplaceBodyWithLoop<'a, 'b> {
        ReplaceBodyWithLoop { within_static_or_const: false, nested_blocks: None, resolver }
    }

    fn run<R, F: FnOnce(&mut Self) -> R>(&mut self, is_const: bool, action: F) -> R {
        let old_const = mem::replace(&mut self.within_static_or_const, is_const);
        let old_blocks = self.nested_blocks.take();
        let ret = action(self);
        self.within_static_or_const = old_const;
        self.nested_blocks = old_blocks;
        ret
    }

    fn should_ignore_fn(ret_ty: &ast::FnRetTy) -> bool {
        if let ast::FnRetTy::Ty(ref ty) = ret_ty {
            fn involves_impl_trait(ty: &ast::Ty) -> bool {
                match ty.kind {
                    ast::TyKind::ImplTrait(..) => true,
                    ast::TyKind::Slice(ref subty)
                    | ast::TyKind::Array(ref subty, _)
                    | ast::TyKind::Ptr(ast::MutTy { ty: ref subty, .. })
                    | ast::TyKind::Rptr(_, ast::MutTy { ty: ref subty, .. })
                    | ast::TyKind::Paren(ref subty) => involves_impl_trait(subty),
                    ast::TyKind::Tup(ref tys) => any_involves_impl_trait(tys.iter()),
                    ast::TyKind::Path(_, ref path) => {
                        path.segments.iter().any(|seg| match seg.args.as_deref() {
                            None => false,
                            Some(&ast::GenericArgs::AngleBracketed(ref data)) => {
                                data.args.iter().any(|arg| match arg {
                                    ast::AngleBracketedArg::Arg(arg) => match arg {
                                        ast::GenericArg::Type(ty) => involves_impl_trait(ty),
                                        ast::GenericArg::Lifetime(_)
                                        | ast::GenericArg::Const(_) => false,
                                    },
                                    ast::AngleBracketedArg::Constraint(c) => match c.kind {
                                        ast::AssocTyConstraintKind::Bound { .. } => true,
                                        ast::AssocTyConstraintKind::Equality { ref ty } => {
                                            involves_impl_trait(ty)
                                        }
                                    },
                                })
                            }
                            Some(&ast::GenericArgs::Parenthesized(ref data)) => {
                                any_involves_impl_trait(data.inputs.iter())
                                    || ReplaceBodyWithLoop::should_ignore_fn(&data.output)
                            }
                        })
                    }
                    _ => false,
                }
            }

            fn any_involves_impl_trait<'a, I: Iterator<Item = &'a P<ast::Ty>>>(mut it: I) -> bool {
                it.any(|subty| involves_impl_trait(subty))
            }

            involves_impl_trait(ty)
        } else {
            false
        }
    }

    fn is_sig_const(sig: &ast::FnSig) -> bool {
        matches!(sig.header.constness, ast::Const::Yes(_))
            || ReplaceBodyWithLoop::should_ignore_fn(&sig.decl.output)
    }
}

impl<'a> MutVisitor for ReplaceBodyWithLoop<'a, '_> {
    fn visit_item_kind(&mut self, i: &mut ast::ItemKind) {
        let is_const = match i {
            ast::ItemKind::Static(..) | ast::ItemKind::Const(..) => true,
            ast::ItemKind::Fn(box ast::Fn { ref sig, .. }) => Self::is_sig_const(sig),
            _ => false,
        };
        self.run(is_const, |s| noop_visit_item_kind(i, s))
    }

    fn flat_map_trait_item(&mut self, i: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        let is_const = match i.kind {
            ast::AssocItemKind::Const(..) => true,
            ast::AssocItemKind::Fn(box ast::Fn { ref sig, .. }) => Self::is_sig_const(sig),
            _ => false,
        };
        self.run(is_const, |s| noop_flat_map_assoc_item(i, s))
    }

    fn flat_map_impl_item(&mut self, i: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        self.flat_map_trait_item(i)
    }

    fn visit_anon_const(&mut self, c: &mut ast::AnonConst) {
        self.run(true, |s| noop_visit_anon_const(c, s))
    }

    fn visit_block(&mut self, b: &mut P<ast::Block>) {
        fn stmt_to_block(
            rules: ast::BlockCheckMode,
            s: Option<ast::Stmt>,
            resolver: &mut Resolver<'_>,
        ) -> ast::Block {
            ast::Block {
                stmts: s.into_iter().collect(),
                rules,
                id: resolver.next_node_id(),
                span: rustc_span::DUMMY_SP,
                tokens: None,
                could_be_bare_literal: false,
            }
        }

        fn block_to_stmt(b: ast::Block, resolver: &mut Resolver<'_>) -> ast::Stmt {
            let expr = P(ast::Expr {
                id: resolver.next_node_id(),
                kind: ast::ExprKind::Block(P(b), None),
                span: rustc_span::DUMMY_SP,
                attrs: AttrVec::new(),
                tokens: None,
            });

            ast::Stmt {
                id: resolver.next_node_id(),
                kind: ast::StmtKind::Expr(expr),
                span: rustc_span::DUMMY_SP,
            }
        }

        let empty_block = stmt_to_block(BlockCheckMode::Default, None, self.resolver);
        let loop_expr = P(ast::Expr {
            kind: ast::ExprKind::Loop(P(empty_block), None),
            id: self.resolver.next_node_id(),
            span: rustc_span::DUMMY_SP,
            attrs: AttrVec::new(),
            tokens: None,
        });

        let loop_stmt = ast::Stmt {
            id: self.resolver.next_node_id(),
            span: rustc_span::DUMMY_SP,
            kind: ast::StmtKind::Expr(loop_expr),
        };

        if self.within_static_or_const {
            noop_visit_block(b, self)
        } else {
            visit_clobber(b.deref_mut(), |b| {
                let mut stmts = vec![];
                for s in b.stmts {
                    let old_blocks = self.nested_blocks.replace(vec![]);

                    stmts.extend(self.flat_map_stmt(s).into_iter().filter(|s| s.is_item()));

                    // we put a Some in there earlier with that replace(), so this is valid
                    let new_blocks = self.nested_blocks.take().unwrap();
                    self.nested_blocks = old_blocks;
                    stmts.extend(new_blocks.into_iter().map(|b| block_to_stmt(b, self.resolver)));
                }

                let mut new_block = ast::Block { stmts, ..b };

                if let Some(old_blocks) = self.nested_blocks.as_mut() {
                    //push our fresh block onto the cache and yield an empty block with `loop {}`
                    if !new_block.stmts.is_empty() {
                        old_blocks.push(new_block);
                    }

                    stmt_to_block(b.rules, Some(loop_stmt), &mut self.resolver)
                } else {
                    //push `loop {}` onto the end of our fresh block and yield that
                    new_block.stmts.push(loop_stmt);

                    new_block
                }
            })
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
