use log::info;
use rustc::session::config::{Input, OutputFilenames, ErrorOutputType};
use rustc::session::{self, config, early_error, filesearch, Session, DiagnosticOutput};
use rustc::session::CrateDisambiguator;
use rustc::ty;
use rustc::lint;
use rustc_codegen_utils::codegen_backend::CodegenBackend;
#[cfg(parallel_compiler)]
use rustc_data_structures::jobserver;
use rustc_data_structures::sync::{Lock, Lrc};
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_data_structures::fx::{FxHashSet, FxHashMap};
use rustc_errors::registry::Registry;
use rustc_lint;
use rustc_metadata::dynamic_lib::DynamicLibrary;
use rustc_mir;
use rustc_passes;
use rustc_plugin;
use rustc_privacy;
use rustc_resolve;
use rustc_typeck;
use std::env;
use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
use std::io::{self, Write};
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::ops::DerefMut;
use smallvec::SmallVec;
use syntax::ptr::P;
use syntax::mut_visit::{*, MutVisitor, visit_clobber};
use syntax::ast::BlockCheckMode;
use syntax::util::lev_distance::find_best_match_for_name;
use syntax::source_map::{FileLoader, RealFileLoader, SourceMap};
use syntax::symbol::{Symbol, sym};
use syntax::{self, ast, attr};
use syntax_pos::edition::Edition;
#[cfg(not(parallel_compiler))]
use std::{thread, panic};

pub fn diagnostics_registry() -> Registry {
    let mut all_errors = Vec::new();
    all_errors.extend_from_slice(&rustc::DIAGNOSTICS);
    all_errors.extend_from_slice(&rustc_typeck::DIAGNOSTICS);
    all_errors.extend_from_slice(&rustc_resolve::DIAGNOSTICS);
    all_errors.extend_from_slice(&rustc_privacy::DIAGNOSTICS);
    // FIXME: need to figure out a way to get these back in here
    // all_errors.extend_from_slice(get_codegen_backend(sess).diagnostics());
    all_errors.extend_from_slice(&rustc_metadata::DIAGNOSTICS);
    all_errors.extend_from_slice(&rustc_passes::DIAGNOSTICS);
    all_errors.extend_from_slice(&rustc_plugin::DIAGNOSTICS);
    all_errors.extend_from_slice(&rustc_mir::DIAGNOSTICS);
    all_errors.extend_from_slice(&syntax::DIAGNOSTICS);

    Registry::new(&all_errors)
}

/// Adds `target_feature = "..."` cfgs for a variety of platform
/// specific features (SSE, NEON etc.).
///
/// This is performed by checking whether a whitelisted set of
/// features is available on the target machine, by querying LLVM.
pub fn add_configuration(
    cfg: &mut ast::CrateConfig,
    sess: &Session,
    codegen_backend: &dyn CodegenBackend,
) {
    let tf = sym::target_feature;

    cfg.extend(
        codegen_backend
            .target_features(sess)
            .into_iter()
            .map(|feat| (tf, Some(feat))),
    );

    if sess.crt_static_feature() {
        cfg.insert((tf, Some(Symbol::intern("crt-static"))));
    }
}

pub fn create_session(
    sopts: config::Options,
    cfg: FxHashSet<(String, Option<String>)>,
    diagnostic_output: DiagnosticOutput,
    file_loader: Option<Box<dyn FileLoader + Send + Sync + 'static>>,
    input_path: Option<PathBuf>,
    lint_caps: FxHashMap<lint::LintId, lint::Level>,
) -> (Lrc<Session>, Lrc<Box<dyn CodegenBackend>>, Lrc<SourceMap>) {
    let descriptions = diagnostics_registry();

    let loader = file_loader.unwrap_or(box RealFileLoader);
    let source_map = Lrc::new(SourceMap::with_file_loader(
        loader,
        sopts.file_path_mapping(),
    ));
    let mut sess = session::build_session_with_source_map(
        sopts,
        input_path,
        descriptions,
        source_map.clone(),
        diagnostic_output,
        lint_caps,
    );

    let codegen_backend = get_codegen_backend(&sess);

    rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));
    if sess.unstable_options() {
        rustc_lint::register_internals(&mut sess.lint_store.borrow_mut(), Some(&sess));
    }

    let mut cfg = config::build_configuration(&sess, config::to_crate_config(cfg));
    add_configuration(&mut cfg, &sess, &*codegen_backend);
    sess.parse_sess.config = cfg;

    (Lrc::new(sess), Lrc::new(codegen_backend), source_map)
}

// Temporarily have stack size set to 32MB to deal with various crates with long method
// chains or deep syntax trees, except when on Haiku.
// FIXME(oli-obk): get https://github.com/rust-lang/rust/pull/55617 the finish line
#[cfg(not(target_os = "haiku"))]
const STACK_SIZE: usize = 32 * 1024 * 1024;

#[cfg(target_os = "haiku")]
const STACK_SIZE: usize = 16 * 1024 * 1024;

fn get_stack_size() -> Option<usize> {
    // FIXME: Hacks on hacks. If the env is trying to override the stack size
    // then *don't* set it explicitly.
    if env::var_os("RUST_MIN_STACK").is_none() {
        Some(STACK_SIZE)
    } else {
        None
    }
}

struct Sink(Arc<Mutex<Vec<u8>>>);
impl Write for Sink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        Write::write(&mut *self.0.lock().unwrap(), data)
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

#[cfg(not(parallel_compiler))]
pub fn scoped_thread<F: FnOnce() -> R + Send, R: Send>(cfg: thread::Builder, f: F) -> R {
    struct Ptr(*mut ());
    unsafe impl Send for Ptr {}
    unsafe impl Sync for Ptr {}

    let mut f = Some(f);
    let run = Ptr(&mut f as *mut _ as *mut ());
    let mut result = None;
    let result_ptr = Ptr(&mut result as *mut _ as *mut ());

    let thread = cfg.spawn(move || {
        let run = unsafe { (*(run.0 as *mut Option<F>)).take().unwrap() };
        let result = unsafe { &mut *(result_ptr.0 as *mut Option<R>) };
        *result = Some(run());
    });

    match thread.unwrap().join() {
        Ok(()) => result.unwrap(),
        Err(p) => panic::resume_unwind(p),
    }
}

#[cfg(not(parallel_compiler))]
pub fn spawn_thread_pool<F: FnOnce() -> R + Send, R: Send>(
    edition: Edition,
    _threads: Option<usize>,
    stderr: &Option<Arc<Mutex<Vec<u8>>>>,
    f: F,
) -> R {
    let mut cfg = thread::Builder::new().name("rustc".to_string());

    if let Some(size) = get_stack_size() {
        cfg = cfg.stack_size(size);
    }

    scoped_thread(cfg, || {
        syntax::with_globals(edition, || {
            ty::tls::GCX_PTR.set(&Lock::new(0), || {
                if let Some(stderr) = stderr {
                    io::set_panic(Some(box Sink(stderr.clone())));
                }
                ty::tls::with_thread_locals(|| f())
            })
        })
    })
}

#[cfg(parallel_compiler)]
pub fn spawn_thread_pool<F: FnOnce() -> R + Send, R: Send>(
    edition: Edition,
    threads: Option<usize>,
    stderr: &Option<Arc<Mutex<Vec<u8>>>>,
    f: F,
) -> R {
    use rayon::{ThreadPool, ThreadPoolBuilder};
    use syntax;
    use syntax_pos;

    let gcx_ptr = &Lock::new(0);

    let mut config = ThreadPoolBuilder::new()
        .acquire_thread_handler(jobserver::acquire_thread)
        .release_thread_handler(jobserver::release_thread)
        .num_threads(Session::threads_from_count(threads))
        .deadlock_handler(|| unsafe { ty::query::handle_deadlock() });

    if let Some(size) = get_stack_size() {
        config = config.stack_size(size);
    }

    let with_pool = move |pool: &ThreadPool| pool.install(move || f());

    syntax::with_globals(edition, || {
        syntax::GLOBALS.with(|syntax_globals| {
            syntax_pos::GLOBALS.with(|syntax_pos_globals| {
                // The main handler runs for each Rayon worker thread and sets up
                // the thread local rustc uses. syntax_globals and syntax_pos_globals are
                // captured and set on the new threads. ty::tls::with_thread_locals sets up
                // thread local callbacks from libsyntax
                let main_handler = move |worker: &mut dyn FnMut()| {
                    syntax::GLOBALS.set(syntax_globals, || {
                        syntax_pos::GLOBALS.set(syntax_pos_globals, || {
                            if let Some(stderr) = stderr {
                                io::set_panic(Some(box Sink(stderr.clone())));
                            }
                            ty::tls::with_thread_locals(|| {
                                ty::tls::GCX_PTR.set(gcx_ptr, || worker())
                            })
                        })
                    })
                };

                ThreadPool::scoped_pool(config, main_handler, with_pool).unwrap()
            })
        })
    })
}

fn load_backend_from_dylib(path: &Path) -> fn() -> Box<dyn CodegenBackend> {
    let lib = DynamicLibrary::open(Some(path)).unwrap_or_else(|err| {
        let err = format!("couldn't load codegen backend {:?}: {:?}", path, err);
        early_error(ErrorOutputType::default(), &err);
    });
    unsafe {
        match lib.symbol("__rustc_codegen_backend") {
            Ok(f) => {
                mem::forget(lib);
                mem::transmute::<*mut u8, _>(f)
            }
            Err(e) => {
                let err = format!("couldn't load codegen backend as it \
                                   doesn't export the `__rustc_codegen_backend` \
                                   symbol: {:?}", e);
                early_error(ErrorOutputType::default(), &err);
            }
        }
    }
}

pub fn get_codegen_backend(sess: &Session) -> Box<dyn CodegenBackend> {
    static INIT: Once = Once::new();

    static mut LOAD: fn() -> Box<dyn CodegenBackend> = || unreachable!();

    INIT.call_once(|| {
        let codegen_name = sess.opts.debugging_opts.codegen_backend.as_ref()
            .unwrap_or(&sess.target.target.options.codegen_backend);
        let backend = match &codegen_name[..] {
            filename if filename.contains(".") => {
                load_backend_from_dylib(filename.as_ref())
            }
            codegen_name => get_codegen_sysroot(codegen_name),
        };

        unsafe {
            LOAD = backend;
        }
    });
    let backend = unsafe { LOAD() };
    backend.init(sess);
    backend
}

pub fn get_codegen_sysroot(backend_name: &str) -> fn() -> Box<dyn CodegenBackend> {
    // For now we only allow this function to be called once as it'll dlopen a
    // few things, which seems to work best if we only do that once. In
    // general this assertion never trips due to the once guard in `get_codegen_backend`,
    // but there's a few manual calls to this function in this file we protect
    // against.
    static LOADED: AtomicBool = AtomicBool::new(false);
    assert!(!LOADED.fetch_or(true, Ordering::SeqCst),
            "cannot load the default codegen backend twice");

    let target = session::config::host_triple();
    let mut sysroot_candidates = vec![filesearch::get_or_default_sysroot()];
    let path = current_dll_path()
        .and_then(|s| s.canonicalize().ok());
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
                sysroot_candidates.extend(path.parent() // chop off `$target`
                    .and_then(|p| p.parent())           // chop off `rustlib`
                    .and_then(|p| p.parent())           // chop off `lib`
                    .map(|s| s.to_owned()));
            }
        }
    }

    let sysroot = sysroot_candidates.iter()
        .map(|sysroot| {
            let libdir = filesearch::relative_target_lib_path(&sysroot, &target);
            sysroot.join(libdir).with_file_name(
                option_env!("CFG_CODEGEN_BACKENDS_DIR").unwrap_or("codegen-backends"))
        })
        .filter(|f| {
            info!("codegen backend candidate: {}", f.display());
            f.exists()
        })
        .next();
    let sysroot = sysroot.unwrap_or_else(|| {
        let candidates = sysroot_candidates.iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join("\n* ");
        let err = format!("failed to find a `codegen-backends` folder \
                           in the sysroot candidates:\n* {}", candidates);
        early_error(ErrorOutputType::default(), &err);
    });
    info!("probing {} for a codegen backend", sysroot.display());

    let d = sysroot.read_dir().unwrap_or_else(|e| {
        let err = format!("failed to load default codegen backend, couldn't \
                           read `{}`: {}", sysroot.display(), e);
        early_error(ErrorOutputType::default(), &err);
    });

    let mut file: Option<PathBuf> = None;

    let expected_name = format!("rustc_codegen_llvm-{}", backend_name);
    for entry in d.filter_map(|e| e.ok()) {
        let path = entry.path();
        let filename = match path.file_name().and_then(|s| s.to_str()) {
            Some(s) => s,
            None => continue,
        };
        if !(filename.starts_with(DLL_PREFIX) && filename.ends_with(DLL_SUFFIX)) {
            continue
        }
        let name = &filename[DLL_PREFIX.len() .. filename.len() - DLL_SUFFIX.len()];
        if name != expected_name {
            continue
        }
        if let Some(ref prev) = file {
            let err = format!("duplicate codegen backends found\n\
                               first:  {}\n\
                               second: {}\n\
            ", prev.display(), path.display());
            early_error(ErrorOutputType::default(), &err);
        }
        file = Some(path.clone());
    }

    match file {
        Some(ref s) => return load_backend_from_dylib(s),
        None => {
            let err = format!("failed to load default codegen backend for `{}`, \
                               no appropriate codegen dylib found in `{}`",
                              backend_name, sysroot.display());
            early_error(ErrorOutputType::default(), &err);
        }
    }

    #[cfg(unix)]
    fn current_dll_path() -> Option<PathBuf> {
        use std::ffi::{OsStr, CStr};
        use std::os::unix::prelude::*;

        unsafe {
            let addr = current_dll_path as usize as *mut _;
            let mut info = mem::zeroed();
            if libc::dladdr(addr, &mut info) == 0 {
                info!("dladdr failed");
                return None
            }
            if info.dli_fname.is_null() {
                info!("dladdr returned null pointer");
                return None
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

        extern "system" {
            fn GetModuleHandleExW(dwFlags: u32,
                                  lpModuleName: usize,
                                  phModule: *mut usize) -> i32;
            fn GetModuleFileNameW(hModule: usize,
                                  lpFilename: *mut u16,
                                  nSize: u32) -> u32;
        }

        const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x00000004;

        unsafe {
            let mut module = 0;
            let r = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                                       current_dll_path as usize,
                                       &mut module);
            if r == 0 {
                info!("GetModuleHandleExW failed: {}", io::Error::last_os_error());
                return None
            }
            let mut space = Vec::with_capacity(1024);
            let r = GetModuleFileNameW(module,
                                       space.as_mut_ptr(),
                                       space.capacity() as u32);
            if r == 0 {
                info!("GetModuleFileNameW failed: {}", io::Error::last_os_error());
                return None
            }
            let r = r as usize;
            if r >= space.capacity() {
                info!("our buffer was too small? {}",
                      io::Error::last_os_error());
                return None
            }
            space.set_len(r);
            let os = OsString::from_wide(&space);
            Some(PathBuf::from(os))
        }
    }
}

pub(crate) fn compute_crate_disambiguator(session: &Session) -> CrateDisambiguator {
    use std::hash::Hasher;

    // The crate_disambiguator is a 128 bit hash. The disambiguator is fed
    // into various other hashes quite a bit (symbol hashes, incr. comp. hashes,
    // debuginfo type IDs, etc), so we don't want it to be too wide. 128 bits
    // should still be safe enough to avoid collisions in practice.
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

    // Also incorporate crate type, so that we don't get symbol conflicts when
    // linking against a library of the same name, if this is an executable.
    let is_exe = session
        .crate_types
        .borrow()
        .contains(&config::CrateType::Executable);
    hasher.write(if is_exe { b"exe" } else { b"lib" });

    CrateDisambiguator::from(hasher.finish())
}

pub fn collect_crate_types(session: &Session, attrs: &[ast::Attribute]) -> Vec<config::CrateType> {
    // Unconditionally collect crate types from attributes to make them used
    let attr_types: Vec<config::CrateType> = attrs
        .iter()
        .filter_map(|a| {
            if a.check_name(sym::crate_type) {
                match a.value_str() {
                    Some(sym::rlib) => Some(config::CrateType::Rlib),
                    Some(sym::dylib) => Some(config::CrateType::Dylib),
                    Some(sym::cdylib) => Some(config::CrateType::Cdylib),
                    Some(sym::lib) => Some(config::default_lib_output()),
                    Some(sym::staticlib) => Some(config::CrateType::Staticlib),
                    Some(sym::proc_dash_macro) => Some(config::CrateType::ProcMacro),
                    Some(sym::bin) => Some(config::CrateType::Executable),
                    Some(n) => {
                        let crate_types = vec![
                            sym::rlib,
                            sym::dylib,
                            sym::cdylib,
                            sym::lib,
                            sym::staticlib,
                            sym::proc_dash_macro,
                            sym::bin
                        ];

                        if let ast::MetaItemKind::NameValue(spanned) = a.meta().unwrap().node {
                            let span = spanned.span;
                            let lev_candidate = find_best_match_for_name(
                                crate_types.iter(),
                                &n.as_str(),
                                None
                            );
                            if let Some(candidate) = lev_candidate {
                                session.buffer_lint_with_diagnostic(
                                    lint::builtin::UNKNOWN_CRATE_TYPES,
                                    ast::CRATE_NODE_ID,
                                    span,
                                    "invalid `crate_type` value",
                                    lint::builtin::BuiltinLintDiagnostics::
                                        UnknownCrateTypes(
                                            span,
                                            "did you mean".to_string(),
                                            format!("\"{}\"", candidate)
                                        )
                                );
                            } else {
                                session.buffer_lint(
                                    lint::builtin::UNKNOWN_CRATE_TYPES,
                                    ast::CRATE_NODE_ID,
                                    span,
                                    "invalid `crate_type` value"
                                );
                            }
                        }
                        None
                    }
                    None => None
                }
            } else {
                None
            }
        })
        .collect();

    // If we're generating a test executable, then ignore all other output
    // styles at all other locations
    if session.opts.test {
        return vec![config::CrateType::Executable];
    }

    // Only check command line flags if present. If no types are specified by
    // command line, then reuse the empty `base` Vec to hold the types that
    // will be found in crate attributes.
    let mut base = session.opts.crate_types.clone();
    if base.is_empty() {
        base.extend(attr_types);
        if base.is_empty() {
            base.push(::rustc_codegen_utils::link::default_output_for_target(
                session,
            ));
        } else {
            base.sort();
            base.dedup();
        }
    }

    base.retain(|crate_type| {
        let res = !::rustc_codegen_utils::link::invalid_output_for_target(session, *crate_type);

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
            let stem = sess.opts
                .crate_name
                .clone()
                .or_else(|| attr::find_crate_name(attrs).map(|n| n.to_string()))
                .unwrap_or_else(|| input.filestem().to_owned());

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

            OutputFilenames {
                out_directory: out_file.parent().unwrap_or_else(|| Path::new("")).to_path_buf(),
                out_filestem: out_file
                    .file_stem()
                    .unwrap_or_default()
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

// Note: Also used by librustdoc, see PR #43348. Consider moving this struct elsewhere.
//
// FIXME: Currently the `everybody_loops` transformation is not applied to:
//  * `const fn`, due to issue #43636 that `loop` is not supported for const evaluation. We are
//    waiting for miri to fix that.
//  * `impl Trait`, due to issue #43869 that functions returning impl Trait cannot be diverging.
//    Solving this may require `!` to implement every trait, which relies on the an even more
//    ambitious form of the closed RFC #1637. See also [#34511].
//
// [#34511]: https://github.com/rust-lang/rust/issues/34511#issuecomment-322340401
pub struct ReplaceBodyWithLoop<'a> {
    within_static_or_const: bool,
    nested_blocks: Option<Vec<ast::Block>>,
    sess: &'a Session,
}

impl<'a> ReplaceBodyWithLoop<'a> {
    pub fn new(sess: &'a Session) -> ReplaceBodyWithLoop<'a> {
        ReplaceBodyWithLoop {
            within_static_or_const: false,
            nested_blocks: None,
            sess
        }
    }

    fn run<R, F: FnOnce(&mut Self) -> R>(&mut self, is_const: bool, action: F) -> R {
        let old_const = mem::replace(&mut self.within_static_or_const, is_const);
        let old_blocks = self.nested_blocks.take();
        let ret = action(self);
        self.within_static_or_const = old_const;
        self.nested_blocks = old_blocks;
        ret
    }

    fn should_ignore_fn(ret_ty: &ast::FnDecl) -> bool {
        if let ast::FunctionRetTy::Ty(ref ty) = ret_ty.output {
            fn involves_impl_trait(ty: &ast::Ty) -> bool {
                match ty.node {
                    ast::TyKind::ImplTrait(..) => true,
                    ast::TyKind::Slice(ref subty) |
                    ast::TyKind::Array(ref subty, _) |
                    ast::TyKind::Ptr(ast::MutTy { ty: ref subty, .. }) |
                    ast::TyKind::Rptr(_, ast::MutTy { ty: ref subty, .. }) |
                    ast::TyKind::Paren(ref subty) => involves_impl_trait(subty),
                    ast::TyKind::Tup(ref tys) => any_involves_impl_trait(tys.iter()),
                    ast::TyKind::Path(_, ref path) => path.segments.iter().any(|seg| {
                        match seg.args.as_ref().map(|generic_arg| &**generic_arg) {
                            None => false,
                            Some(&ast::GenericArgs::AngleBracketed(ref data)) => {
                                let types = data.args.iter().filter_map(|arg| match arg {
                                    ast::GenericArg::Type(ty) => Some(ty),
                                    _ => None,
                                });
                                any_involves_impl_trait(types.into_iter()) ||
                                data.constraints.iter().any(|c| {
                                    match c.kind {
                                        ast::AssocTyConstraintKind::Bound { .. } => true,
                                        ast::AssocTyConstraintKind::Equality { ref ty } =>
                                            involves_impl_trait(ty),
                                    }
                                })
                            },
                            Some(&ast::GenericArgs::Parenthesized(ref data)) => {
                                any_involves_impl_trait(data.inputs.iter()) ||
                                any_involves_impl_trait(data.output.iter())
                            }
                        }
                    }),
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
}

impl<'a> MutVisitor for ReplaceBodyWithLoop<'a> {
    fn visit_item_kind(&mut self, i: &mut ast::ItemKind) {
        let is_const = match i {
            ast::ItemKind::Static(..) | ast::ItemKind::Const(..) => true,
            ast::ItemKind::Fn(ref decl, ref header, _, _) =>
                header.constness.node == ast::Constness::Const || Self::should_ignore_fn(decl),
            _ => false,
        };
        self.run(is_const, |s| noop_visit_item_kind(i, s))
    }

    fn flat_map_trait_item(&mut self, i: ast::TraitItem) -> SmallVec<[ast::TraitItem; 1]> {
        let is_const = match i.node {
            ast::TraitItemKind::Const(..) => true,
            ast::TraitItemKind::Method(ast::MethodSig { ref decl, ref header, .. }, _) =>
                header.constness.node == ast::Constness::Const || Self::should_ignore_fn(decl),
            _ => false,
        };
        self.run(is_const, |s| noop_flat_map_trait_item(i, s))
    }

    fn flat_map_impl_item(&mut self, i: ast::ImplItem) -> SmallVec<[ast::ImplItem; 1]> {
        let is_const = match i.node {
            ast::ImplItemKind::Const(..) => true,
            ast::ImplItemKind::Method(ast::MethodSig { ref decl, ref header, .. }, _) =>
                header.constness.node == ast::Constness::Const || Self::should_ignore_fn(decl),
            _ => false,
        };
        self.run(is_const, |s| noop_flat_map_impl_item(i, s))
    }

    fn visit_anon_const(&mut self, c: &mut ast::AnonConst) {
        self.run(true, |s| noop_visit_anon_const(c, s))
    }

    fn visit_block(&mut self, b: &mut P<ast::Block>) {
        fn stmt_to_block(rules: ast::BlockCheckMode,
                         s: Option<ast::Stmt>,
                         sess: &Session) -> ast::Block {
            ast::Block {
                stmts: s.into_iter().collect(),
                rules,
                id: sess.next_node_id(),
                span: syntax_pos::DUMMY_SP,
            }
        }

        fn block_to_stmt(b: ast::Block, sess: &Session) -> ast::Stmt {
            let expr = P(ast::Expr {
                id: sess.next_node_id(),
                node: ast::ExprKind::Block(P(b), None),
                span: syntax_pos::DUMMY_SP,
                attrs: ThinVec::new(),
            });

            ast::Stmt {
                id: sess.next_node_id(),
                node: ast::StmtKind::Expr(expr),
                span: syntax_pos::DUMMY_SP,
            }
        }

        let empty_block = stmt_to_block(BlockCheckMode::Default, None, self.sess);
        let loop_expr = P(ast::Expr {
            node: ast::ExprKind::Loop(P(empty_block), None),
            id: self.sess.next_node_id(),
            span: syntax_pos::DUMMY_SP,
                attrs: ThinVec::new(),
        });

        let loop_stmt = ast::Stmt {
            id: self.sess.next_node_id(),
            span: syntax_pos::DUMMY_SP,
            node: ast::StmtKind::Expr(loop_expr),
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
                    stmts.extend(new_blocks.into_iter().map(|b| block_to_stmt(b, &self.sess)));
                }

                let mut new_block = ast::Block {
                    stmts,
                    ..b
                };

                if let Some(old_blocks) = self.nested_blocks.as_mut() {
                    //push our fresh block onto the cache and yield an empty block with `loop {}`
                    if !new_block.stmts.is_empty() {
                        old_blocks.push(new_block);
                    }

                    stmt_to_block(b.rules, Some(loop_stmt), self.sess)
                } else {
                    //push `loop {}` onto the end of our fresh block and yield that
                    new_block.stmts.push(loop_stmt);

                    new_block
                }
            })
        }
    }

    // in general the pretty printer processes unexpanded code, so
    // we override the default `visit_mac` method which panics.
    fn visit_mac(&mut self, mac: &mut ast::Mac) {
        noop_visit_mac(mac, self)
    }
}
