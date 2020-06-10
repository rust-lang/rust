use log::info;
use rustc_ast::ast::{AttrVec, BlockCheckMode};
use rustc_ast::mut_visit::{visit_clobber, MutVisitor, *};
use rustc_ast::ptr::P;
use rustc_ast::util::lev_distance::find_best_match_for_name;
use rustc_ast::{self, ast};
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
#[cfg(parallel_compiler)]
use rustc_data_structures::jobserver;
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_data_structures::sync::{Lock, Lrc};
use rustc_errors::registry::Registry;
use rustc_metadata::dynamic_lib::DynamicLibrary;
use rustc_middle::ty;
use rustc_resolve::{self, Resolver};
use rustc_session as session;
use rustc_session::config::{self, CrateType};
use rustc_session::config::{ErrorOutputType, Input, OutputFilenames};
use rustc_session::lint::{self, BuiltinLintDiagnostics, LintBuffer};
use rustc_session::parse::CrateConfig;
use rustc_session::CrateDisambiguator;
use rustc_session::{early_error, filesearch, output, DiagnosticOutput, Session};
use rustc_span::edition::Edition;
use rustc_span::source_map::FileLoader;
use rustc_span::symbol::{sym, Symbol};
use smallvec::SmallVec;
use std::env;
use std::io::{self, Write};
use std::mem;
use std::ops::DerefMut;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Once};
#[cfg(not(parallel_compiler))]
use std::{panic, thread};

/// Adds `target_feature = "..."` cfgs for a variety of platform
/// specific features (SSE, NEON etc.).
///
/// This is performed by checking whether a whitelisted set of
/// features is available on the target machine, by querying LLVM.
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
    descriptions: Registry,
) -> (Lrc<Session>, Lrc<Box<dyn CodegenBackend>>) {
    let mut sess = session::build_session(
        sopts,
        input_path,
        descriptions,
        diagnostic_output,
        lint_caps,
        file_loader,
    );

    let codegen_backend = get_codegen_backend(&sess);

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

struct Sink(Arc<Mutex<Vec<u8>>>);
impl Write for Sink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        Write::write(&mut *self.0.lock().unwrap(), data)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
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
    _threads: usize,
    stderr: &Option<Arc<Mutex<Vec<u8>>>>,
    f: F,
) -> R {
    let mut cfg = thread::Builder::new().name("rustc".to_string());

    if let Some(size) = get_stack_size() {
        cfg = cfg.stack_size(size);
    }

    crate::callbacks::setup_callbacks();

    scoped_thread(cfg, || {
        rustc_ast::with_globals(edition, || {
            ty::tls::GCX_PTR.set(&Lock::new(0), || {
                if let Some(stderr) = stderr {
                    io::set_panic(Some(box Sink(stderr.clone())));
                }
                f()
            })
        })
    })
}

#[cfg(parallel_compiler)]
pub fn spawn_thread_pool<F: FnOnce() -> R + Send, R: Send>(
    edition: Edition,
    threads: usize,
    stderr: &Option<Arc<Mutex<Vec<u8>>>>,
    f: F,
) -> R {
    use rayon::{ThreadBuilder, ThreadPool, ThreadPoolBuilder};

    let gcx_ptr = &Lock::new(0);
    crate::callbacks::setup_callbacks();

    let mut config = ThreadPoolBuilder::new()
        .thread_name(|_| "rustc".to_string())
        .acquire_thread_handler(jobserver::acquire_thread)
        .release_thread_handler(jobserver::release_thread)
        .num_threads(threads)
        .deadlock_handler(|| unsafe { ty::query::handle_deadlock() });

    if let Some(size) = get_stack_size() {
        config = config.stack_size(size);
    }

    let with_pool = move |pool: &ThreadPool| pool.install(move || f());

    rustc_ast::with_globals(edition, || {
        rustc_ast::GLOBALS.with(|syntax_globals| {
            rustc_span::GLOBALS.with(|rustc_span_globals| {
                // The main handler runs for each Rayon worker thread and sets up
                // the thread local rustc uses. syntax_globals and rustc_span_globals are
                // captured and set on the new threads. ty::tls::with_thread_locals sets up
                // thread local callbacks from librustc_ast
                let main_handler = move |thread: ThreadBuilder| {
                    rustc_ast::GLOBALS.set(syntax_globals, || {
                        rustc_span::GLOBALS.set(rustc_span_globals, || {
                            if let Some(stderr) = stderr {
                                io::set_panic(Some(box Sink(stderr.clone())));
                            }
                            ty::tls::GCX_PTR.set(gcx_ptr, || thread.run())
                        })
                    })
                };

                config.build_scoped(main_handler, with_pool).unwrap()
            })
        })
    })
}

fn load_backend_from_dylib(path: &Path) -> fn() -> Box<dyn CodegenBackend> {
    let lib = DynamicLibrary::open(path).unwrap_or_else(|err| {
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
                let err = format!(
                    "couldn't load codegen backend as it \
                                   doesn't export the `__rustc_codegen_backend` \
                                   symbol: {:?}",
                    e
                );
                early_error(ErrorOutputType::default(), &err);
            }
        }
    }
}

pub fn get_codegen_backend(sess: &Session) -> Box<dyn CodegenBackend> {
    static INIT: Once = Once::new();

    static mut LOAD: fn() -> Box<dyn CodegenBackend> = || unreachable!();

    INIT.call_once(|| {
        let codegen_name = sess
            .opts
            .debugging_opts
            .codegen_backend
            .as_ref()
            .unwrap_or(&sess.target.target.options.codegen_backend);
        let backend = match &codegen_name[..] {
            filename if filename.contains('.') => load_backend_from_dylib(filename.as_ref()),
            codegen_name => get_builtin_codegen_backend(codegen_name),
        };

        unsafe {
            LOAD = backend;
        }
    });
    let backend = unsafe { LOAD() };
    backend.init(sess);
    backend
}

// This is used for rustdoc, but it uses similar machinery to codegen backend
// loading, so we leave the code here. It is potentially useful for other tools
// that want to invoke the rustc binary while linking to rustc as well.
pub fn rustc_path<'a>() -> Option<&'a Path> {
    static RUSTC_PATH: once_cell::sync::OnceCell<Option<PathBuf>> =
        once_cell::sync::OnceCell::new();

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

pub fn get_builtin_codegen_backend(backend_name: &str) -> fn() -> Box<dyn CodegenBackend> {
    #[cfg(feature = "llvm")]
    {
        if backend_name == "llvm" {
            return rustc_codegen_llvm::LlvmCodegenBackend::new;
        }
    }

    let err = format!("unsupported builtin codegen backend `{}`", backend_name);
    early_error(ErrorOutputType::default(), &err);
}

pub(crate) fn compute_crate_disambiguator(session: &Session) -> CrateDisambiguator {
    use std::hash::Hasher;

    // The crate_disambiguator is a 128 bit hash. The disambiguator is fed
    // into various other hashes quite a bit (symbol hashes, incr. comp. hashes,
    // debuginfo type IDs, etc), so we don't want it to be too wide. 128 bits
    // should still be safe enough to avoid collisions in practice.
    let mut hasher = StableHasher::new();

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
    let is_exe = session.crate_types().contains(&CrateType::Executable);
    hasher.write(if is_exe { b"exe" } else { b"lib" });

    CrateDisambiguator::from(hasher.finish::<Fingerprint>())
}

pub(crate) fn check_attr_crate_type(attrs: &[ast::Attribute], lint_buffer: &mut LintBuffer) {
    // Unconditionally collect crate types from attributes to make them used
    for a in attrs.iter() {
        if a.check_name(sym::crate_type) {
            if let Some(n) = a.value_str() {
                if categorize_crate_type(n).is_some() {
                    return;
                }

                if let ast::MetaItemKind::NameValue(spanned) = a.meta().unwrap().kind {
                    let span = spanned.span;
                    let lev_candidate = find_best_match_for_name(
                        CRATE_TYPES.iter().map(|(k, _)| k),
                        &n.as_str(),
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
            if a.check_name(sym::crate_type) {
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
                .or_else(|| rustc_attr::find_crate_name(attrs).map(|n| n.to_string()))
                .unwrap_or_else(|| input.filestem().to_owned());

            OutputFilenames::new(
                dirpath,
                stem,
                None,
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
                sess.opts.cg.extra_filename.clone(),
                sess.opts.output_types.clone(),
            )
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
            ast::ItemKind::Fn(_, ref sig, _, _) => Self::is_sig_const(sig),
            _ => false,
        };
        self.run(is_const, |s| noop_visit_item_kind(i, s))
    }

    fn flat_map_trait_item(&mut self, i: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        let is_const = match i.kind {
            ast::AssocItemKind::Const(..) => true,
            ast::AssocItemKind::Fn(_, ref sig, _, _) => Self::is_sig_const(sig),
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

    // in general the pretty printer processes unexpanded code, so
    // we override the default `visit_mac` method which panics.
    fn visit_mac(&mut self, mac: &mut ast::MacCall) {
        noop_visit_mac(mac, self)
    }
}
