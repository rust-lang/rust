//! Handles dynamic library loading for proc macro

mod proc_macros;

use rustc_codegen_ssa::back::metadata::DefaultMetadataLoader;
use rustc_interface::util::rustc_version_str;
use rustc_metadata::locator::MetadataError;
use rustc_proc_macro::bridge;
use rustc_session::config::host_tuple;
use rustc_target::spec::{Target, TargetTuple};
use std::path::Path;
use std::{fmt, fs, io, time::SystemTime};
use temp_dir::TempDir;

use libloading::Library;
use object::Object;
use paths::{Utf8Path, Utf8PathBuf};

use crate::{
    PanicMessage, ProcMacroClientHandle, ProcMacroKind, ProcMacroSrvSpan, TrackedEnv,
    dylib::proc_macros::{ProcMacroClients, ProcMacros},
    token_stream::TokenStream,
};

pub(crate) struct Expander {
    inner: ProcMacroLibrary,
    modified_time: SystemTime,
}

impl Expander {
    pub(crate) fn new(
        temp_dir: &TempDir,
        lib: &Utf8Path,
    ) -> Result<Expander, LoadProcMacroDylibError> {
        // Some libraries for dynamic loading require canonicalized path even when it is
        // already absolute
        let lib = lib.canonicalize_utf8()?;
        let modified_time = fs::metadata(&lib).and_then(|it| it.modified())?;

        let path = ensure_file_with_lock_free_access(temp_dir, &lib)?;
        let library = ProcMacroLibrary::open(path.as_ref())?;

        Ok(Expander { inner: library, modified_time })
    }

    pub(crate) fn expand<'a, S: ProcMacroSrvSpan + 'a>(
        &self,
        macro_name: &str,
        macro_body: TokenStream<S>,
        attribute: Option<TokenStream<S>>,
        def_site: S,
        call_site: S,
        mixed_site: S,
        tracked_env: &'a mut TrackedEnv,
        callback: Option<ProcMacroClientHandle<'a>>,
    ) -> Result<TokenStream<S>, PanicMessage>
    where
        <S::Server<'a> as bridge::server::Server>::TokenStream: Default,
    {
        self.inner.proc_macros.expand(
            macro_name,
            macro_body,
            attribute,
            def_site,
            call_site,
            mixed_site,
            tracked_env,
            callback,
        )
    }

    pub(crate) fn list_macros(&self) -> impl Iterator<Item = (&str, ProcMacroKind)> {
        self.inner.proc_macros.list_macros()
    }

    pub(crate) fn modified_time(&self) -> SystemTime {
        self.modified_time
    }
}

#[derive(Debug)]
pub enum LoadProcMacroDylibError {
    Io(io::Error),
    LibLoading(libloading::Error),
    MetadataError(MetadataError<'static>),
}

impl fmt::Display for LoadProcMacroDylibError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::LibLoading(e) => e.fmt(f),
            Self::MetadataError(e) => e.fmt(f),
        }
    }
}

impl From<io::Error> for LoadProcMacroDylibError {
    fn from(e: io::Error) -> Self {
        LoadProcMacroDylibError::Io(e)
    }
}

impl From<libloading::Error> for LoadProcMacroDylibError {
    fn from(e: libloading::Error) -> Self {
        LoadProcMacroDylibError::LibLoading(e)
    }
}

impl From<MetadataError<'_>> for LoadProcMacroDylibError {
    fn from(e: MetadataError<'_>) -> Self {
        LoadProcMacroDylibError::MetadataError(match e {
            MetadataError::NotPresent(path) => MetadataError::NotPresent(path.into_owned().into()),
            MetadataError::LoadFailure(err) => MetadataError::LoadFailure(err),
            MetadataError::VersionMismatch { expected_version, found_version } => {
                MetadataError::VersionMismatch { expected_version, found_version }
            }
        })
    }
}

struct ProcMacroLibrary {
    // this contains references to the library, so make sure this drops before _lib
    proc_macros: ProcMacros,
    // Hold on to the library so it doesn't unload
    _lib: Library,
}

impl ProcMacroLibrary {
    fn open(path: &Utf8Path) -> Result<Self, LoadProcMacroDylibError> {
        let proc_macro_kinds = rustc_span::create_default_session_globals_then(|| {
            let (target, _) =
                Target::search(&TargetTuple::from_tuple(host_tuple()), Path::new(""), false)
                    .unwrap();
            rustc_metadata::locator::get_proc_macro_info(
                &target,
                path.as_ref(),
                &DefaultMetadataLoader,
                rustc_version_str().unwrap_or("unknown"),
            )
        })?;

        let file = fs::File::open(path)?;
        #[allow(clippy::undocumented_unsafe_blocks)] // FIXME
        let file = unsafe { memmap2::Mmap::map(&file) }?;
        let obj = object::File::parse(&*file)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let symbol_name =
            find_registrar_symbol(&obj).map_err(invalid_data_err)?.ok_or_else(|| {
                invalid_data_err(format!("Cannot find registrar symbol in file {path}"))
            })?;

        // SAFETY: We have verified the validity of the dylib as a proc-macro library
        let lib = unsafe { load_library(path) }.map_err(invalid_data_err)?;
        // SAFETY: We have verified the validity of the dylib as a proc-macro library
        // The 'static lifetime is a lie, it's actually the lifetime of the library but unavoidable
        // due to self-referentiality
        // But we make sure that we do not drop it before the symbol is dropped
        let proc_macros =
            unsafe { lib.get::<&'static &'static ProcMacroClients>(symbol_name.as_bytes()) };
        match proc_macros {
            Ok(proc_macros) => Ok(ProcMacroLibrary {
                proc_macros: ProcMacros::new(*proc_macros, proc_macro_kinds),
                _lib: lib,
            }),
            Err(e) => Err(e.into()),
        }
    }
}

fn invalid_data_err(e: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

fn is_derive_registrar_symbol(symbol: &str) -> bool {
    const NEW_REGISTRAR_SYMBOL: &str = "_rustc_proc_macro_decls_";
    symbol.contains(NEW_REGISTRAR_SYMBOL)
}

fn find_registrar_symbol(obj: &object::File<'_>) -> object::Result<Option<String>> {
    Ok(obj
        .exports()?
        .into_iter()
        .map(|export| export.name())
        .filter_map(|sym| String::from_utf8(sym.into()).ok())
        .find(|sym| is_derive_registrar_symbol(sym))
        .map(|sym| {
            // From MacOS docs:
            // https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/dlsym.3.html
            // Unlike other dyld API's, the symbol name passed to dlsym() must NOT be
            // prepended with an underscore.
            if cfg!(target_os = "macos") && sym.starts_with('_') {
                sym[1..].to_owned()
            } else {
                sym
            }
        }))
}

/// Copy the dylib to temp directory to prevent locking in Windows
#[cfg(windows)]
fn ensure_file_with_lock_free_access(
    temp_dir: &TempDir,
    path: &Utf8Path,
) -> io::Result<Utf8PathBuf> {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    if std::env::var("RA_DONT_COPY_PROC_MACRO_DLL").is_ok() {
        return Ok(path.to_path_buf());
    }

    let mut to = Utf8Path::from_path(temp_dir.path()).unwrap().to_owned();

    let file_name = path.file_stem().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, format!("File path is invalid: {path}"))
    })?;

    to.push({
        // Generate a unique number by abusing `HashMap`'s hasher.
        // Maybe this will also "inspire" a libs team member to finally put `rand` in libstd.
        let unique_name = RandomState::new().build_hasher().finish();
        format!("{file_name}-{unique_name}.dll")
    });
    fs::copy(path, &to)?;
    Ok(to)
}

#[cfg(unix)]
fn ensure_file_with_lock_free_access(
    _temp_dir: &TempDir,
    path: &Utf8Path,
) -> io::Result<Utf8PathBuf> {
    Ok(path.to_owned())
}

/// Loads dynamic library in platform dependent manner.
///
/// For unix, you have to use RTLD_DEEPBIND flag to escape problems described
/// [here](https://github.com/fedochet/rust-proc-macro-panic-inside-panic-expample)
/// and [here](https://github.com/rust-lang/rust/issues/60593).
///
/// Usage of RTLD_DEEPBIND
/// [here](https://github.com/fedochet/rust-proc-macro-panic-inside-panic-expample/issues/1)
///
/// It seems that on Windows that behaviour is default, so we do nothing in that case.
///
/// # Safety
///
/// The caller is responsible for ensuring that the path is valid proc-macro library
#[cfg(windows)]
unsafe fn load_library(file: &Utf8Path) -> Result<Library, libloading::Error> {
    // SAFETY: The caller is responsible for ensuring that the path is valid proc-macro library
    unsafe { Library::new(file) }
}

/// Loads dynamic library in platform dependent manner.
///
/// For unix, you have to use RTLD_DEEPBIND flag to escape problems described
/// [here](https://github.com/fedochet/rust-proc-macro-panic-inside-panic-expample)
/// and [here](https://github.com/rust-lang/rust/issues/60593).
///
/// Usage of RTLD_DEEPBIND
/// [here](https://github.com/fedochet/rust-proc-macro-panic-inside-panic-expample/issues/1)
///
/// It seems that on Windows that behaviour is default, so we do nothing in that case.
///
/// # Safety
///
/// The caller is responsible for ensuring that the path is valid proc-macro library
#[cfg(unix)]
unsafe fn load_library(file: &Utf8Path) -> Result<Library, libloading::Error> {
    // not defined by POSIX, different values on mips vs other targets
    #[cfg(target_env = "gnu")]
    use libc::RTLD_DEEPBIND;
    use libloading::os::unix::Library as UnixLibrary;
    // defined by POSIX
    use libloading::os::unix::RTLD_NOW;

    // MUSL and bionic don't have it..
    #[cfg(not(target_env = "gnu"))]
    const RTLD_DEEPBIND: std::os::raw::c_int = 0x0;

    // SAFETY: The caller is responsible for ensuring that the path is valid proc-macro library
    unsafe { UnixLibrary::open(Some(file), RTLD_NOW | RTLD_DEEPBIND).map(|lib| lib.into()) }
}
