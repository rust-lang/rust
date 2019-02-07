//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(box_syntax)]
#![cfg_attr(unix, feature(libc))]
#![feature(nll)]
#![feature(rustc_diagnostic_macros)]
#![feature(slice_sort_by_cached_key)]
#![feature(set_stdio)]
#![feature(no_debug)]
#![feature(integer_atomics)]

#![recursion_limit="256"]

extern crate arena;
pub extern crate getopts;
extern crate graphviz;
extern crate env_logger;
#[cfg(unix)]
extern crate libc;
extern crate rustc_rayon as rayon;
extern crate rustc;
extern crate rustc_allocator;
extern crate rustc_target;
extern crate rustc_borrowck;
#[macro_use]
extern crate rustc_data_structures;
extern crate rustc_errors as errors;
extern crate rustc_passes;
extern crate rustc_lint;
extern crate rustc_plugin;
extern crate rustc_privacy;
extern crate rustc_incremental;
extern crate rustc_metadata;
extern crate rustc_mir;
extern crate rustc_resolve;
extern crate rustc_save_analysis;
extern crate rustc_traits;
extern crate rustc_codegen_utils;
extern crate rustc_typeck;
extern crate scoped_tls;
extern crate serialize;
extern crate smallvec;
#[macro_use]
extern crate log;
extern crate syntax;
extern crate syntax_ext;
extern crate syntax_pos;

use driver::CompileController;
use pretty::{PpMode, UserIdentifiedItem};

use rustc_save_analysis as save;
use rustc_save_analysis::DumpHandler;
use rustc_data_structures::sync::{self, Lrc, Ordering::SeqCst};
use rustc_data_structures::OnDrop;
use rustc::session::{self, config, Session, build_session, CompileResult};
use rustc::session::CompileIncomplete;
use rustc::session::config::{Input, PrintRequest, ErrorOutputType};
use rustc::session::config::nightly_options;
use rustc::session::filesearch;
use rustc::session::{early_error, early_warn};
use rustc::lint::Lint;
use rustc::lint;
use rustc_metadata::locator;
use rustc_metadata::cstore::CStore;
use rustc_metadata::dynamic_lib::DynamicLibrary;
use rustc::util::common::{time, ErrorReported};
use rustc_codegen_utils::codegen_backend::CodegenBackend;

use serialize::json::ToJson;

use std::any::Any;
use std::borrow::Cow;
use std::cmp::max;
use std::default::Default;
use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fmt::{self, Display};
use std::io::{self, Read, Write};
use std::mem;
use std::panic;
use std::path::{PathBuf, Path};
use std::process::{self, Command, Stdio};
use std::str;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Once, ONCE_INIT};
use std::thread;

use syntax::ast;
use syntax::source_map::{SourceMap, FileLoader, RealFileLoader};
use syntax::feature_gate::{GatedCfg, UnstableFeatures};
use syntax::parse::{self, PResult};
use syntax_pos::{DUMMY_SP, MultiSpan, FileName};

#[cfg(test)]
mod test;

pub mod profile;
pub mod driver;
pub mod pretty;
mod proc_macro_decls;

pub mod target_features {
    use syntax::ast;
    use syntax::symbol::Symbol;
    use rustc::session::Session;
    use rustc_codegen_utils::codegen_backend::CodegenBackend;

    /// Add `target_feature = "..."` cfgs for a variety of platform
    /// specific features (SSE, NEON etc.).
    ///
    /// This is performed by checking whether a whitelisted set of
    /// features is available on the target machine, by querying LLVM.
    pub fn add_configuration(cfg: &mut ast::CrateConfig,
                             sess: &Session,
                             codegen_backend: &dyn CodegenBackend) {
        let tf = Symbol::intern("target_feature");

        cfg.extend(codegen_backend.target_features(sess).into_iter().map(|feat| (tf, Some(feat))));

        if sess.crt_static_feature() {
            cfg.insert((tf, Some(Symbol::intern("crt-static"))));
        }
    }
}

/// Exit status code used for successful compilation and help output.
pub const EXIT_SUCCESS: isize = 0;

/// Exit status code used for compilation failures and  invalid flags.
pub const EXIT_FAILURE: isize = 1;

const BUG_REPORT_URL: &str = "https://github.com/rust-lang/rust/blob/master/CONTRIBUTING.\
                              md#bug-reports";

const ICE_REPORT_COMPILER_FLAGS: &[&str] = &["Z", "C", "crate-type"];

const ICE_REPORT_COMPILER_FLAGS_EXCLUDE: &[&str] = &["metadata", "extra-filename"];

const ICE_REPORT_COMPILER_FLAGS_STRIP_VALUE: &[&str] = &["incremental"];

pub fn abort_on_err<T>(result: Result<T, CompileIncomplete>, sess: &Session) -> T {
    match result {
        Err(CompileIncomplete::Errored(ErrorReported)) => {
            sess.abort_if_errors();
            panic!("error reported but abort_if_errors didn't abort???");
        }
        Err(CompileIncomplete::Stopped) => {
            sess.fatal("compilation terminated");
        }
        Ok(x) => x,
    }
}

pub fn run<F>(run_compiler: F) -> isize
    where F: FnOnce() -> (CompileResult, Option<Session>) + Send + 'static
{
    let result = monitor(move || {
        syntax::with_globals(|| {
            let (result, session) = run_compiler();
            if let Err(CompileIncomplete::Errored(_)) = result {
                match session {
                    Some(sess) => {
                        sess.abort_if_errors();
                        panic!("error reported but abort_if_errors didn't abort???");
                    }
                    None => {
                        let emitter =
                            errors::emitter::EmitterWriter::stderr(
                                errors::ColorConfig::Auto,
                                None,
                                true,
                                false
                            );
                        let handler = errors::Handler::with_emitter(true, false, Box::new(emitter));
                        handler.emit(&MultiSpan::new(),
                                     "aborting due to previous error(s)",
                                     errors::Level::Fatal);
                        panic::resume_unwind(Box::new(errors::FatalErrorMarker));
                    }
                }
            }
        });
    });

    match result {
        Ok(()) => EXIT_SUCCESS,
        Err(_) => EXIT_FAILURE,
    }
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
    static INIT: Once = ONCE_INIT;

    #[allow(deprecated)]
    #[no_debug]
    static mut LOAD: fn() -> Box<dyn CodegenBackend> = || unreachable!();

    INIT.call_once(|| {
        let codegen_name = sess.opts.debugging_opts.codegen_backend.as_ref()
            .unwrap_or(&sess.target.target.options.codegen_backend);
        let backend = match &codegen_name[..] {
            "metadata_only" => {
                rustc_codegen_utils::codegen_backend::MetadataOnlyCodegenBackend::boxed
            }
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

fn get_codegen_sysroot(backend_name: &str) -> fn() -> Box<dyn CodegenBackend> {
    // For now we only allow this function to be called once as it'll dlopen a
    // few things, which seems to work best if we only do that once. In
    // general this assertion never trips due to the once guard in `get_codegen_backend`,
    // but there's a few manual calls to this function in this file we protect
    // against.
    static LOADED: AtomicBool = AtomicBool::new(false);
    assert!(!LOADED.fetch_or(true, Ordering::SeqCst),
            "cannot load the default codegen backend twice");

    // When we're compiling this library with `--test` it'll run as a binary but
    // not actually exercise much functionality. As a result most of the logic
    // here is defunkt (it assumes we're a dynamic library in a sysroot) so
    // let's just return a dummy creation function which won't be used in
    // general anyway.
    if cfg!(test) {
        return rustc_codegen_utils::codegen_backend::MetadataOnlyCodegenBackend::boxed
    }

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

// Parse args and run the compiler. This is the primary entry point for rustc.
// See comments on CompilerCalls below for details about the callbacks argument.
// The FileLoader provides a way to load files from sources other than the file system.
pub fn run_compiler<'a>(args: &[String],
                        callbacks: Box<dyn CompilerCalls<'a> + sync::Send + 'a>,
                        file_loader: Option<Box<dyn FileLoader + Send + Sync + 'static>>,
                        emitter_dest: Option<Box<dyn Write + Send>>)
                        -> (CompileResult, Option<Session>)
{
    let matches = match handle_options(args) {
        Some(matches) => matches,
        None => return (Ok(()), None),
    };

    let (sopts, cfg) = config::build_session_options_and_crate_config(&matches);

    driver::spawn_thread_pool(sopts, |sopts| {
        run_compiler_with_pool(matches, sopts, cfg, callbacks, file_loader, emitter_dest)
    })
}

fn run_compiler_with_pool<'a>(
    matches: getopts::Matches,
    sopts: config::Options,
    cfg: ast::CrateConfig,
    mut callbacks: Box<dyn CompilerCalls<'a> + sync::Send + 'a>,
    file_loader: Option<Box<dyn FileLoader + Send + Sync + 'static>>,
    emitter_dest: Option<Box<dyn Write + Send>>
) -> (CompileResult, Option<Session>) {
    macro_rules! do_or_return {($expr: expr, $sess: expr) => {
        match $expr {
            Compilation::Stop => return (Ok(()), $sess),
            Compilation::Continue => {}
        }
    }}

    let descriptions = diagnostics_registry();

    do_or_return!(callbacks.early_callback(&matches,
                                           &sopts,
                                           &cfg,
                                           &descriptions,
                                           sopts.error_format),
                                           None);

    let (odir, ofile) = make_output(&matches);
    let (input, input_file_path, input_err) = match make_input(&matches.free) {
        Some((input, input_file_path, input_err)) => {
            let (input, input_file_path) = callbacks.some_input(input, input_file_path);
            (input, input_file_path, input_err)
        },
        None => match callbacks.no_input(&matches, &sopts, &cfg, &odir, &ofile, &descriptions) {
            Some((input, input_file_path)) => (input, input_file_path, None),
            None => return (Ok(()), None),
        },
    };

    let loader = file_loader.unwrap_or(box RealFileLoader);
    let source_map = Lrc::new(SourceMap::with_file_loader(loader, sopts.file_path_mapping()));
    let mut sess = session::build_session_with_source_map(
        sopts, input_file_path.clone(), descriptions, source_map, emitter_dest,
    );

    if let Some(err) = input_err {
        // Immediately stop compilation if there was an issue reading
        // the input (for example if the input stream is not UTF-8).
        sess.err(&err.to_string());
        return (Err(CompileIncomplete::Stopped), Some(sess));
    }

    let codegen_backend = get_codegen_backend(&sess);

    rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));

    let mut cfg = config::build_configuration(&sess, cfg);
    target_features::add_configuration(&mut cfg, &sess, &*codegen_backend);
    sess.parse_sess.config = cfg;

    let result = {
        let plugins = sess.opts.debugging_opts.extra_plugins.clone();

        let cstore = CStore::new(codegen_backend.metadata_loader());

        do_or_return!(callbacks.late_callback(&*codegen_backend,
                                              &matches,
                                              &sess,
                                              &cstore,
                                              &input,
                                              &odir,
                                              &ofile), Some(sess));

        let _sess_abort_error = OnDrop(|| sess.diagnostic().print_error_count());

        let control = callbacks.build_controller(&sess, &matches);

        driver::compile_input(codegen_backend,
                              &sess,
                              &cstore,
                              &input_file_path,
                              &input,
                              &odir,
                              &ofile,
                              Some(plugins),
                              &control)
    };

    (result, Some(sess))
}

#[cfg(unix)]
pub fn set_sigpipe_handler() {
    unsafe {
        // Set the SIGPIPE signal handler, so that an EPIPE
        // will cause rustc to terminate, as expected.
        assert_ne!(libc::signal(libc::SIGPIPE, libc::SIG_DFL), libc::SIG_ERR);
    }
}

#[cfg(windows)]
pub fn set_sigpipe_handler() {}

// Extract output directory and file from matches.
fn make_output(matches: &getopts::Matches) -> (Option<PathBuf>, Option<PathBuf>) {
    let odir = matches.opt_str("out-dir").map(|o| PathBuf::from(&o));
    let ofile = matches.opt_str("o").map(|o| PathBuf::from(&o));
    (odir, ofile)
}

// Extract input (string or file and optional path) from matches.
fn make_input(free_matches: &[String]) -> Option<(Input, Option<PathBuf>, Option<io::Error>)> {
    if free_matches.len() == 1 {
        let ifile = &free_matches[0];
        if ifile == "-" {
            let mut src = String::new();
            let err = if io::stdin().read_to_string(&mut src).is_err() {
                Some(io::Error::new(io::ErrorKind::InvalidData,
                                    "couldn't read from stdin, as it did not contain valid UTF-8"))
            } else {
                None
            };
            Some((Input::Str { name: FileName::anon_source_code(&src), input: src },
                  None, err))
        } else {
            Some((Input::File(PathBuf::from(ifile)),
                  Some(PathBuf::from(ifile)), None))
        }
    } else {
        None
    }
}

fn parse_pretty(sess: &Session,
                matches: &getopts::Matches)
                -> Option<(PpMode, Option<UserIdentifiedItem>)> {
    let pretty = if sess.opts.debugging_opts.unstable_options {
        matches.opt_default("pretty", "normal").map(|a| {
            // stable pretty-print variants only
            pretty::parse_pretty(sess, &a, false)
        })
    } else {
        None
    };

    if pretty.is_none() {
        sess.opts.debugging_opts.unpretty.as_ref().map(|a| {
            // extended with unstable pretty-print variants
            pretty::parse_pretty(sess, &a, true)
        })
    } else {
        pretty
    }
}

// Whether to stop or continue compilation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Compilation {
    Stop,
    Continue,
}

impl Compilation {
    pub fn and_then<F: FnOnce() -> Compilation>(self, next: F) -> Compilation {
        match self {
            Compilation::Stop => Compilation::Stop,
            Compilation::Continue => next(),
        }
    }
}

/// A trait for customizing the compilation process. Offers a number of hooks for
/// executing custom code or customizing input.
pub trait CompilerCalls<'a> {
    /// Hook for a callback early in the process of handling arguments. This will
    /// be called straight after options have been parsed but before anything
    /// else (e.g., selecting input and output).
    fn early_callback(&mut self,
                      _: &getopts::Matches,
                      _: &config::Options,
                      _: &ast::CrateConfig,
                      _: &errors::registry::Registry,
                      _: ErrorOutputType)
                      -> Compilation {
        Compilation::Continue
    }

    /// Hook for a callback late in the process of handling arguments. This will
    /// be called just before actual compilation starts (and before build_controller
    /// is called), after all arguments etc. have been completely handled.
    fn late_callback(&mut self,
                     _: &dyn CodegenBackend,
                     _: &getopts::Matches,
                     _: &Session,
                     _: &CStore,
                     _: &Input,
                     _: &Option<PathBuf>,
                     _: &Option<PathBuf>)
                     -> Compilation {
        Compilation::Continue
    }

    /// Called after we extract the input from the arguments. Gives the implementer
    /// an opportunity to change the inputs or to add some custom input handling.
    /// The default behaviour is to simply pass through the inputs.
    fn some_input(&mut self,
                  input: Input,
                  input_path: Option<PathBuf>)
                  -> (Input, Option<PathBuf>) {
        (input, input_path)
    }

    /// Called after we extract the input from the arguments if there is no valid
    /// input. Gives the implementer an opportunity to supply alternate input (by
    /// returning a Some value) or to add custom behaviour for this error such as
    /// emitting error messages. Returning None will cause compilation to stop
    /// at this point.
    fn no_input(&mut self,
                _: &getopts::Matches,
                _: &config::Options,
                _: &ast::CrateConfig,
                _: &Option<PathBuf>,
                _: &Option<PathBuf>,
                _: &errors::registry::Registry)
                -> Option<(Input, Option<PathBuf>)> {
        None
    }

    // Create a CompilController struct for controlling the behaviour of
    // compilation.
    fn build_controller(
        self: Box<Self>,
        _: &Session,
        _: &getopts::Matches
    ) -> CompileController<'a>;
}

/// CompilerCalls instance for a regular rustc build.
#[derive(Copy, Clone)]
pub struct RustcDefaultCalls;

// FIXME remove these and use winapi 0.3 instead
// Duplicates: bootstrap/compile.rs, librustc_errors/emitter.rs
#[cfg(unix)]
fn stdout_isatty() -> bool {
    unsafe { libc::isatty(libc::STDOUT_FILENO) != 0 }
}

#[cfg(windows)]
fn stdout_isatty() -> bool {
    type DWORD = u32;
    type BOOL = i32;
    type HANDLE = *mut u8;
    type LPDWORD = *mut u32;
    const STD_OUTPUT_HANDLE: DWORD = -11i32 as DWORD;
    extern "system" {
        fn GetStdHandle(which: DWORD) -> HANDLE;
        fn GetConsoleMode(hConsoleHandle: HANDLE, lpMode: LPDWORD) -> BOOL;
    }
    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut out = 0;
        GetConsoleMode(handle, &mut out) != 0
    }
}

fn handle_explain(code: &str,
                  descriptions: &errors::registry::Registry,
                  output: ErrorOutputType) {
    let normalised = if code.starts_with("E") {
        code.to_string()
    } else {
        format!("E{0:0>4}", code)
    };
    match descriptions.find_description(&normalised) {
        Some(ref description) => {
            let mut is_in_code_block = false;
            let mut text = String::new();

            // Slice off the leading newline and print.
            for line in description[1..].lines() {
                let indent_level = line.find(|c: char| !c.is_whitespace())
                    .unwrap_or_else(|| line.len());
                let dedented_line = &line[indent_level..];
                if dedented_line.starts_with("```") {
                    is_in_code_block = !is_in_code_block;
                    text.push_str(&line[..(indent_level+3)]);
                } else if is_in_code_block && dedented_line.starts_with("# ") {
                    continue;
                } else {
                    text.push_str(line);
                }
                text.push('\n');
            }

            if stdout_isatty() {
                show_content_with_pager(&text);
            } else {
                print!("{}", text);
            }
        }
        None => {
            early_error(output, &format!("no extended information for {}", code));
        }
    }
}

fn show_content_with_pager(content: &String) {
    let pager_name = env::var_os("PAGER").unwrap_or_else(|| if cfg!(windows) {
        OsString::from("more.com")
    } else {
        OsString::from("less")
    });

    let mut fallback_to_println = false;

    match Command::new(pager_name).stdin(Stdio::piped()).spawn() {
        Ok(mut pager) => {
            if let Some(pipe) = pager.stdin.as_mut() {
                if pipe.write_all(content.as_bytes()).is_err() {
                    fallback_to_println = true;
                }
            }

            if pager.wait().is_err() {
                fallback_to_println = true;
            }
        }
        Err(_) => {
            fallback_to_println = true;
        }
    }

    // If pager fails for whatever reason, we should still print the content
    // to standard output
    if fallback_to_println {
        print!("{}", content);
    }
}

impl<'a> CompilerCalls<'a> for RustcDefaultCalls {
    fn early_callback(&mut self,
                      matches: &getopts::Matches,
                      _: &config::Options,
                      _: &ast::CrateConfig,
                      descriptions: &errors::registry::Registry,
                      output: ErrorOutputType)
                      -> Compilation {
        if let Some(ref code) = matches.opt_str("explain") {
            handle_explain(code, descriptions, output);
            return Compilation::Stop;
        }

        Compilation::Continue
    }

    fn no_input(&mut self,
                matches: &getopts::Matches,
                sopts: &config::Options,
                cfg: &ast::CrateConfig,
                odir: &Option<PathBuf>,
                ofile: &Option<PathBuf>,
                descriptions: &errors::registry::Registry)
                -> Option<(Input, Option<PathBuf>)> {
        match matches.free.len() {
            0 => {
                let mut sess = build_session(sopts.clone(),
                    None,
                    descriptions.clone());
                if sopts.describe_lints {
                    let mut ls = lint::LintStore::new();
                    rustc_lint::register_builtins(&mut ls, Some(&sess));
                    describe_lints(&sess, &ls, false);
                    return None;
                }
                rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));
                let mut cfg = config::build_configuration(&sess, cfg.clone());
                let codegen_backend = get_codegen_backend(&sess);
                target_features::add_configuration(&mut cfg, &sess, &*codegen_backend);
                sess.parse_sess.config = cfg;
                let should_stop = RustcDefaultCalls::print_crate_info(
                    &*codegen_backend,
                    &sess,
                    None,
                    odir,
                    ofile
                );

                if should_stop == Compilation::Stop {
                    return None;
                }
                early_error(sopts.error_format, "no input filename given");
            }
            1 => panic!("make_input should have provided valid inputs"),
            _ => early_error(sopts.error_format, "multiple input filenames provided"),
        }
    }

    fn late_callback(&mut self,
                     codegen_backend: &dyn CodegenBackend,
                     matches: &getopts::Matches,
                     sess: &Session,
                     cstore: &CStore,
                     input: &Input,
                     odir: &Option<PathBuf>,
                     ofile: &Option<PathBuf>)
                     -> Compilation {
        RustcDefaultCalls::print_crate_info(codegen_backend, sess, Some(input), odir, ofile)
            .and_then(|| RustcDefaultCalls::list_metadata(sess, cstore, matches, input))
    }

    fn build_controller(self: Box<Self>,
                        sess: &Session,
                        matches: &getopts::Matches)
                        -> CompileController<'a> {
        let mut control = CompileController::basic();

        control.keep_ast = sess.opts.debugging_opts.keep_ast;
        control.continue_parse_after_error = sess.opts.debugging_opts.continue_parse_after_error;

        if let Some((ppm, opt_uii)) = parse_pretty(sess, matches) {
            if ppm.needs_ast_map(&opt_uii) {
                control.after_hir_lowering.stop = Compilation::Stop;

                control.after_parse.callback = box move |state| {
                    let mut krate = state.krate.take().unwrap();
                    pretty::visit_crate(state.session, &mut krate, ppm);
                    state.krate = Some(krate);
                };
                control.after_hir_lowering.callback = box move |state| {
                    pretty::print_after_hir_lowering(state.session,
                                                     state.cstore.unwrap(),
                                                     state.hir_map.unwrap(),
                                                     state.resolutions.unwrap(),
                                                     state.input,
                                                     &state.expanded_crate.take().unwrap(),
                                                     state.crate_name.unwrap(),
                                                     ppm,
                                                     state.output_filenames.unwrap(),
                                                     opt_uii.clone(),
                                                     state.out_file);
                };
            } else {
                control.after_parse.stop = Compilation::Stop;

                control.after_parse.callback = box move |state| {
                    let mut krate = state.krate.take().unwrap();
                    pretty::visit_crate(state.session, &mut krate, ppm);
                    pretty::print_after_parsing(state.session,
                                                state.input,
                                                &krate,
                                                ppm,
                                                state.out_file);
                };
            }

            return control;
        }

        if sess.opts.debugging_opts.parse_only ||
           sess.opts.debugging_opts.show_span.is_some() ||
           sess.opts.debugging_opts.ast_json_noexpand {
            control.after_parse.stop = Compilation::Stop;
        }

        if sess.opts.debugging_opts.no_analysis ||
           sess.opts.debugging_opts.ast_json {
            control.after_hir_lowering.stop = Compilation::Stop;
        }

        if sess.opts.debugging_opts.save_analysis {
            enable_save_analysis(&mut control);
        }

        if sess.print_fuel_crate.is_some() {
            let old_callback = control.compilation_done.callback;
            control.compilation_done.callback = box move |state| {
                old_callback(state);
                let sess = state.session;
                eprintln!("Fuel used by {}: {}",
                    sess.print_fuel_crate.as_ref().unwrap(),
                    sess.print_fuel.load(SeqCst));
            }
        }
        control
    }
}

pub fn enable_save_analysis(control: &mut CompileController) {
    control.keep_ast = true;
    control.after_analysis.callback = box |state| {
        time(state.session, "save analysis", || {
            save::process_crate(state.tcx.unwrap(),
                                state.expanded_crate.unwrap(),
                                state.crate_name.unwrap(),
                                state.input,
                                None,
                                DumpHandler::new(state.out_dir,
                                                 state.crate_name.unwrap()))
        });
    };
    control.after_analysis.run_callback_on_error = true;
}

impl RustcDefaultCalls {
    pub fn list_metadata(sess: &Session,
                         cstore: &CStore,
                         matches: &getopts::Matches,
                         input: &Input)
                         -> Compilation {
        let r = matches.opt_strs("Z");
        if r.iter().any(|s| *s == "ls") {
            match input {
                &Input::File(ref ifile) => {
                    let path = &(*ifile);
                    let mut v = Vec::new();
                    locator::list_file_metadata(&sess.target.target,
                                                path,
                                                &*cstore.metadata_loader,
                                                &mut v)
                            .unwrap();
                    println!("{}", String::from_utf8(v).unwrap());
                }
                &Input::Str { .. } => {
                    early_error(ErrorOutputType::default(), "cannot list metadata for stdin");
                }
            }
            return Compilation::Stop;
        }

        Compilation::Continue
    }


    fn print_crate_info(codegen_backend: &dyn CodegenBackend,
                        sess: &Session,
                        input: Option<&Input>,
                        odir: &Option<PathBuf>,
                        ofile: &Option<PathBuf>)
                        -> Compilation {
        use rustc::session::config::PrintRequest::*;
        // PrintRequest::NativeStaticLibs is special - printed during linking
        // (empty iterator returns true)
        if sess.opts.prints.iter().all(|&p| p == PrintRequest::NativeStaticLibs) {
            return Compilation::Continue;
        }

        let attrs = match input {
            None => None,
            Some(input) => {
                let result = parse_crate_attrs(sess, input);
                match result {
                    Ok(attrs) => Some(attrs),
                    Err(mut parse_error) => {
                        parse_error.emit();
                        return Compilation::Stop;
                    }
                }
            }
        };
        for req in &sess.opts.prints {
            match *req {
                TargetList => {
                    let mut targets = rustc_target::spec::get_targets().collect::<Vec<String>>();
                    targets.sort();
                    println!("{}", targets.join("\n"));
                },
                Sysroot => println!("{}", sess.sysroot.display()),
                TargetSpec => println!("{}", sess.target.target.to_json().pretty()),
                FileNames | CrateName => {
                    let input = input.unwrap_or_else(||
                        early_error(ErrorOutputType::default(), "no input file provided"));
                    let attrs = attrs.as_ref().unwrap();
                    let t_outputs = driver::build_output_filenames(input, odir, ofile, attrs, sess);
                    let id = rustc_codegen_utils::link::find_crate_name(Some(sess), attrs, input);
                    if *req == PrintRequest::CrateName {
                        println!("{}", id);
                        continue;
                    }
                    let crate_types = driver::collect_crate_types(sess, attrs);
                    for &style in &crate_types {
                        let fname = rustc_codegen_utils::link::filename_for_input(
                            sess,
                            style,
                            &id,
                            &t_outputs
                        );
                        println!("{}", fname.file_name().unwrap().to_string_lossy());
                    }
                }
                Cfg => {
                    let allow_unstable_cfg = UnstableFeatures::from_environment()
                        .is_nightly_build();

                    let mut cfgs = sess.parse_sess.config.iter().filter_map(|&(name, ref value)| {
                        let gated_cfg = GatedCfg::gate(&ast::MetaItem {
                            ident: ast::Path::from_ident(ast::Ident::with_empty_ctxt(name)),
                            node: ast::MetaItemKind::Word,
                            span: DUMMY_SP,
                        });

                        // Note that crt-static is a specially recognized cfg
                        // directive that's printed out here as part of
                        // rust-lang/rust#37406, but in general the
                        // `target_feature` cfg is gated under
                        // rust-lang/rust#29717. For now this is just
                        // specifically allowing the crt-static cfg and that's
                        // it, this is intended to get into Cargo and then go
                        // through to build scripts.
                        let value = value.as_ref().map(|s| s.as_str());
                        let value = value.as_ref().map(|s| s.as_ref());
                        if name != "target_feature" || value != Some("crt-static") {
                            if !allow_unstable_cfg && gated_cfg.is_some() {
                                return None
                            }
                        }

                        if let Some(value) = value {
                            Some(format!("{}=\"{}\"", name, value))
                        } else {
                            Some(name.to_string())
                        }
                    }).collect::<Vec<String>>();

                    cfgs.sort();
                    for cfg in cfgs {
                        println!("{}", cfg);
                    }
                }
                RelocationModels | CodeModels | TlsModels | TargetCPUs | TargetFeatures => {
                    codegen_backend.print(*req, sess);
                }
                // Any output here interferes with Cargo's parsing of other printed output
                PrintRequest::NativeStaticLibs => {}
            }
        }
        return Compilation::Stop;
    }
}

/// Returns a version string such as "0.12.0-dev".
fn release_str() -> Option<&'static str> {
    option_env!("CFG_RELEASE")
}

/// Returns the full SHA1 hash of HEAD of the Git repo from which rustc was built.
fn commit_hash_str() -> Option<&'static str> {
    option_env!("CFG_VER_HASH")
}

/// Returns the "commit date" of HEAD of the Git repo from which rustc was built as a static string.
fn commit_date_str() -> Option<&'static str> {
    option_env!("CFG_VER_DATE")
}

/// Prints version information
pub fn version(binary: &str, matches: &getopts::Matches) {
    let verbose = matches.opt_present("verbose");

    println!("{} {}", binary, option_env!("CFG_VERSION").unwrap_or("unknown version"));

    if verbose {
        fn unw(x: Option<&str>) -> &str {
            x.unwrap_or("unknown")
        }
        println!("binary: {}", binary);
        println!("commit-hash: {}", unw(commit_hash_str()));
        println!("commit-date: {}", unw(commit_date_str()));
        println!("host: {}", config::host_triple());
        println!("release: {}", unw(release_str()));
        get_codegen_sysroot("llvm")().print_version();
    }
}

fn usage(verbose: bool, include_unstable_options: bool) {
    let groups = if verbose {
        config::rustc_optgroups()
    } else {
        config::rustc_short_optgroups()
    };
    let mut options = getopts::Options::new();
    for option in groups.iter().filter(|x| include_unstable_options || x.is_stable()) {
        (option.apply)(&mut options);
    }
    let message = "Usage: rustc [OPTIONS] INPUT";
    let nightly_help = if nightly_options::is_nightly_build() {
        "\n    -Z help             Print internal options for debugging rustc"
    } else {
        ""
    };
    let verbose_help = if verbose {
        ""
    } else {
        "\n    --help -v           Print the full set of options rustc accepts"
    };
    println!("{}\nAdditional help:
    -C help             Print codegen options
    -W help             \
              Print 'lint' options and default settings{}{}\n",
             options.usage(message),
             nightly_help,
             verbose_help);
}

fn print_wall_help() {
    println!("
The flag `-Wall` does not exist in `rustc`. Most useful lints are enabled by
default. Use `rustc -W help` to see all available lints. It's more common to put
warning settings in the crate root using `#![warn(LINT_NAME)]` instead of using
the command line flag directly.
");
}

fn describe_lints(sess: &Session, lint_store: &lint::LintStore, loaded_plugins: bool) {
    println!("
Available lint options:
    -W <foo>           Warn about <foo>
    -A <foo>           \
              Allow <foo>
    -D <foo>           Deny <foo>
    -F <foo>           Forbid <foo> \
              (deny <foo> and all attempts to override)

");

    fn sort_lints(sess: &Session, lints: Vec<(&'static Lint, bool)>) -> Vec<&'static Lint> {
        let mut lints: Vec<_> = lints.into_iter().map(|(x, _)| x).collect();
        // The sort doesn't case-fold but it's doubtful we care.
        lints.sort_by_cached_key(|x: &&Lint| (x.default_level(sess), x.name));
        lints
    }

    fn sort_lint_groups(lints: Vec<(&'static str, Vec<lint::LintId>, bool)>)
                        -> Vec<(&'static str, Vec<lint::LintId>)> {
        let mut lints: Vec<_> = lints.into_iter().map(|(x, y, _)| (x, y)).collect();
        lints.sort_by_key(|l| l.0);
        lints
    }

    let (plugin, builtin): (Vec<_>, _) = lint_store.get_lints()
                                                   .iter()
                                                   .cloned()
                                                   .partition(|&(_, p)| p);
    let plugin = sort_lints(sess, plugin);
    let builtin = sort_lints(sess, builtin);

    let (plugin_groups, builtin_groups): (Vec<_>, _) = lint_store.get_lint_groups()
                                                                 .iter()
                                                                 .cloned()
                                                                 .partition(|&(.., p)| p);
    let plugin_groups = sort_lint_groups(plugin_groups);
    let builtin_groups = sort_lint_groups(builtin_groups);

    let max_name_len = plugin.iter()
                             .chain(&builtin)
                             .map(|&s| s.name.chars().count())
                             .max()
                             .unwrap_or(0);
    let padded = |x: &str| {
        let mut s = " ".repeat(max_name_len - x.chars().count());
        s.push_str(x);
        s
    };

    println!("Lint checks provided by rustc:\n");
    println!("    {}  {:7.7}  {}", padded("name"), "default", "meaning");
    println!("    {}  {:7.7}  {}", padded("----"), "-------", "-------");

    let print_lints = |lints: Vec<&Lint>| {
        for lint in lints {
            let name = lint.name_lower().replace("_", "-");
            println!("    {}  {:7.7}  {}",
                     padded(&name),
                     lint.default_level.as_str(),
                     lint.desc);
        }
        println!("\n");
    };

    print_lints(builtin);

    let max_name_len = max("warnings".len(),
                           plugin_groups.iter()
                                        .chain(&builtin_groups)
                                        .map(|&(s, _)| s.chars().count())
                                        .max()
                                        .unwrap_or(0));

    let padded = |x: &str| {
        let mut s = " ".repeat(max_name_len - x.chars().count());
        s.push_str(x);
        s
    };

    println!("Lint groups provided by rustc:\n");
    println!("    {}  {}", padded("name"), "sub-lints");
    println!("    {}  {}", padded("----"), "---------");
    println!("    {}  {}", padded("warnings"), "all lints that are set to issue warnings");

    let print_lint_groups = |lints: Vec<(&'static str, Vec<lint::LintId>)>| {
        for (name, to) in lints {
            let name = name.to_lowercase().replace("_", "-");
            let desc = to.into_iter()
                         .map(|x| x.to_string().replace("_", "-"))
                         .collect::<Vec<String>>()
                         .join(", ");
            println!("    {}  {}", padded(&name), desc);
        }
        println!("\n");
    };

    print_lint_groups(builtin_groups);

    match (loaded_plugins, plugin.len(), plugin_groups.len()) {
        (false, 0, _) | (false, _, 0) => {
            println!("Compiler plugins can provide additional lints and lint groups. To see a \
                      listing of these, re-run `rustc -W help` with a crate filename.");
        }
        (false, ..) => panic!("didn't load lint plugins but got them anyway!"),
        (true, 0, 0) => println!("This crate does not load any lint plugins or lint groups."),
        (true, l, g) => {
            if l > 0 {
                println!("Lint checks provided by plugins loaded by this crate:\n");
                print_lints(plugin);
            }
            if g > 0 {
                println!("Lint groups provided by plugins loaded by this crate:\n");
                print_lint_groups(plugin_groups);
            }
        }
    }
}

fn describe_debug_flags() {
    println!("\nAvailable debug options:\n");
    print_flag_list("-Z", config::DB_OPTIONS);
}

fn describe_codegen_flags() {
    println!("\nAvailable codegen options:\n");
    print_flag_list("-C", config::CG_OPTIONS);
}

fn print_flag_list<T>(cmdline_opt: &str,
                      flag_list: &[(&'static str, T, Option<&'static str>, &'static str)]) {
    let max_len = flag_list.iter()
                           .map(|&(name, _, opt_type_desc, _)| {
                               let extra_len = match opt_type_desc {
                                   Some(..) => 4,
                                   None => 0,
                               };
                               name.chars().count() + extra_len
                           })
                           .max()
                           .unwrap_or(0);

    for &(name, _, opt_type_desc, desc) in flag_list {
        let (width, extra) = match opt_type_desc {
            Some(..) => (max_len - 4, "=val"),
            None => (max_len, ""),
        };
        println!("    {} {:>width$}{} -- {}",
                 cmdline_opt,
                 name.replace("_", "-"),
                 extra,
                 desc,
                 width = width);
    }
}

/// Process command line options. Emits messages as appropriate. If compilation
/// should continue, returns a getopts::Matches object parsed from args,
/// otherwise returns None.
///
/// The compiler's handling of options is a little complicated as it ties into
/// our stability story, and it's even *more* complicated by historical
/// accidents. The current intention of each compiler option is to have one of
/// three modes:
///
/// 1. An option is stable and can be used everywhere.
/// 2. An option is unstable, but was historically allowed on the stable
///    channel.
/// 3. An option is unstable, and can only be used on nightly.
///
/// Like unstable library and language features, however, unstable options have
/// always required a form of "opt in" to indicate that you're using them. This
/// provides the easy ability to scan a code base to check to see if anything
/// unstable is being used. Currently, this "opt in" is the `-Z` "zed" flag.
///
/// All options behind `-Z` are considered unstable by default. Other top-level
/// options can also be considered unstable, and they were unlocked through the
/// `-Z unstable-options` flag. Note that `-Z` remains to be the root of
/// instability in both cases, though.
///
/// So with all that in mind, the comments below have some more detail about the
/// contortions done here to get things to work out correctly.
pub fn handle_options(args: &[String]) -> Option<getopts::Matches> {
    // Throw away the first argument, the name of the binary
    let args = &args[1..];

    if args.is_empty() {
        // user did not write `-v` nor `-Z unstable-options`, so do not
        // include that extra information.
        usage(false, false);
        return None;
    }

    // Parse with *all* options defined in the compiler, we don't worry about
    // option stability here we just want to parse as much as possible.
    let mut options = getopts::Options::new();
    for option in config::rustc_optgroups() {
        (option.apply)(&mut options);
    }
    let matches = options.parse(args).unwrap_or_else(|f|
        early_error(ErrorOutputType::default(), &f.to_string()));

    // For all options we just parsed, we check a few aspects:
    //
    // * If the option is stable, we're all good
    // * If the option wasn't passed, we're all good
    // * If `-Z unstable-options` wasn't passed (and we're not a -Z option
    //   ourselves), then we require the `-Z unstable-options` flag to unlock
    //   this option that was passed.
    // * If we're a nightly compiler, then unstable options are now unlocked, so
    //   we're good to go.
    // * Otherwise, if we're a truly unstable option then we generate an error
    //   (unstable option being used on stable)
    // * If we're a historically stable-but-should-be-unstable option then we
    //   emit a warning that we're going to turn this into an error soon.
    nightly_options::check_nightly_options(&matches, &config::rustc_optgroups());

    if matches.opt_present("h") || matches.opt_present("help") {
        // Only show unstable options in --help if we *really* accept unstable
        // options, which catches the case where we got `-Z unstable-options` on
        // the stable channel of Rust which was accidentally allowed
        // historically.
        usage(matches.opt_present("verbose"),
              nightly_options::is_unstable_enabled(&matches));
        return None;
    }

    // Handle the special case of -Wall.
    let wall = matches.opt_strs("W");
    if wall.iter().any(|x| *x == "all") {
        print_wall_help();
        return None;
    }

    // Don't handle -W help here, because we might first load plugins.
    let r = matches.opt_strs("Z");
    if r.iter().any(|x| *x == "help") {
        describe_debug_flags();
        return None;
    }

    let cg_flags = matches.opt_strs("C");

    if cg_flags.iter().any(|x| *x == "help") {
        describe_codegen_flags();
        return None;
    }

    if cg_flags.iter().any(|x| *x == "no-stack-check") {
        early_warn(ErrorOutputType::default(),
                   "the --no-stack-check flag is deprecated and does nothing");
    }

    if cg_flags.iter().any(|x| *x == "passes=list") {
        get_codegen_sysroot("llvm")().print_passes();
        return None;
    }

    if matches.opt_present("version") {
        version("rustc", &matches);
        return None;
    }

    Some(matches)
}

fn parse_crate_attrs<'a>(sess: &'a Session, input: &Input) -> PResult<'a, Vec<ast::Attribute>> {
    match *input {
        Input::File(ref ifile) => {
            parse::parse_crate_attrs_from_file(ifile, &sess.parse_sess)
        }
        Input::Str { ref name, ref input } => {
            parse::parse_crate_attrs_from_source_str(name.clone(),
                                                     input.clone(),
                                                     &sess.parse_sess)
        }
    }
}

// Temporarily have stack size set to 32MB to deal with various crates with long method
// chains or deep syntax trees.
// FIXME(oli-obk): get https://github.com/rust-lang/rust/pull/55617 the finish line
const STACK_SIZE: usize = 32 * 1024 * 1024; // 32MB

/// Runs `f` in a suitable thread for running `rustc`; returns a `Result` with either the return
/// value of `f` or -- if a panic occurs -- the panic value.
///
/// This version applies the given name to the thread. This is used by rustdoc to ensure consistent
/// doctest output across platforms and executions.
pub fn in_named_rustc_thread<F, R>(name: String, f: F) -> Result<R, Box<dyn Any + Send>>
    where F: FnOnce() -> R + Send + 'static,
          R: Send + 'static,
{
    // We need a thread for soundness of thread local storage in rustc. For debugging purposes
    // we allow an escape hatch where everything runs on the main thread.
    if env::var_os("RUSTC_UNSTABLE_NO_MAIN_THREAD").is_none() {
        let mut cfg = thread::Builder::new().name(name);

        // If the env is trying to override the stack size then *don't* set it explicitly.
        // The libstd thread impl will fetch the `RUST_MIN_STACK` env var itself.
        if env::var_os("RUST_MIN_STACK").is_none() {
            cfg = cfg.stack_size(STACK_SIZE);
        }

        let thread = cfg.spawn(f);
        thread.unwrap().join()
    } else {
        let f = panic::AssertUnwindSafe(f);
        panic::catch_unwind(f)
    }
}

/// Runs `f` in a suitable thread for running `rustc`; returns a
/// `Result` with either the return value of `f` or -- if a panic
/// occurs -- the panic value.
pub fn in_rustc_thread<F, R>(f: F) -> Result<R, Box<dyn Any + Send>>
    where F: FnOnce() -> R + Send + 'static,
          R: Send + 'static,
{
    in_named_rustc_thread("rustc".to_string(), f)
}

/// Get a list of extra command-line flags provided by the user, as strings.
///
/// This function is used during ICEs to show more information useful for
/// debugging, since some ICEs only happens with non-default compiler flags
/// (and the users don't always report them).
fn extra_compiler_flags() -> Option<(Vec<String>, bool)> {
    let args = env::args_os().map(|arg| arg.to_string_lossy().to_string()).collect::<Vec<_>>();

    // Avoid printing help because of empty args. This can suggest the compiler
    // itself is not the program root (consider RLS).
    if args.len() < 2 {
        return None;
    }

    let matches = if let Some(matches) = handle_options(&args) {
        matches
    } else {
        return None;
    };

    let mut result = Vec::new();
    let mut excluded_cargo_defaults = false;
    for flag in ICE_REPORT_COMPILER_FLAGS {
        let prefix = if flag.len() == 1 { "-" } else { "--" };

        for content in &matches.opt_strs(flag) {
            // Split always returns the first element
            let name = if let Some(first) = content.split('=').next() {
                first
            } else {
                &content
            };

            let content = if ICE_REPORT_COMPILER_FLAGS_STRIP_VALUE.contains(&name) {
                name
            } else {
                content
            };

            if !ICE_REPORT_COMPILER_FLAGS_EXCLUDE.contains(&name) {
                result.push(format!("{}{} {}", prefix, flag, content));
            } else {
                excluded_cargo_defaults = true;
            }
        }
    }

    if !result.is_empty() {
        Some((result, excluded_cargo_defaults))
    } else {
        None
    }
}

#[derive(Debug)]
pub struct CompilationFailure;

impl Error for CompilationFailure {}

impl Display for CompilationFailure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "compilation had errors")
    }
}

/// Run a procedure which will detect panics in the compiler and print nicer
/// error messages rather than just failing the test.
///
/// The diagnostic emitter yielded to the procedure should be used for reporting
/// errors of the compiler.
pub fn monitor<F: FnOnce() + Send + 'static>(f: F) -> Result<(), CompilationFailure> {
    in_rustc_thread(move || {
        f()
    }).map_err(|value| {
        if value.is::<errors::FatalErrorMarker>() {
            CompilationFailure
        } else {
            // Thread panicked without emitting a fatal diagnostic
            eprintln!("");

            let emitter =
                Box::new(errors::emitter::EmitterWriter::stderr(errors::ColorConfig::Auto,
                                                                None,
                                                                false,
                                                                false));
            let handler = errors::Handler::with_emitter(true, false, emitter);

            // a .span_bug or .bug call has already printed what
            // it wants to print.
            if !value.is::<errors::ExplicitBug>() {
                handler.emit(&MultiSpan::new(),
                             "unexpected panic",
                             errors::Level::Bug);
            }

            let mut xs: Vec<Cow<'static, str>> = vec![
                "the compiler unexpectedly panicked. this is a bug.".into(),
                format!("we would appreciate a bug report: {}", BUG_REPORT_URL).into(),
                format!("rustc {} running on {}",
                        option_env!("CFG_VERSION").unwrap_or("unknown_version"),
                        config::host_triple()).into(),
            ];

            if let Some((flags, excluded_cargo_defaults)) = extra_compiler_flags() {
                xs.push(format!("compiler flags: {}", flags.join(" ")).into());

                if excluded_cargo_defaults {
                    xs.push("some of the compiler flags provided by cargo are hidden".into());
                }
            }

            for note in &xs {
                handler.emit(&MultiSpan::new(),
                             note,
                             errors::Level::Note);
            }

            panic::resume_unwind(Box::new(errors::FatalErrorMarker));
        }
    })
}

pub fn diagnostics_registry() -> errors::registry::Registry {
    use errors::registry::Registry;

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

/// This allows tools to enable rust logging without having to magically match rustc's
/// log crate version
pub fn init_rustc_env_logger() {
    env_logger::init();
}

pub fn main() {
    init_rustc_env_logger();
    let result = run(|| {
        let args = env::args_os().enumerate()
            .map(|(i, arg)| arg.into_string().unwrap_or_else(|arg| {
                early_error(ErrorOutputType::default(),
                            &format!("Argument {} is not valid Unicode: {:?}", i, arg))
            }))
            .collect::<Vec<_>>();
        run_compiler(&args,
                     Box::new(RustcDefaultCalls),
                     None,
                     None)
    });
    process::exit(result as i32);
}
