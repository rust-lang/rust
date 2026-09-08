//! Handles dynamic library loading for proc macro

mod proc_macros;

use paths::{Utf8Path, Utf8PathBuf};
use rustc_codegen_ssa::back::metadata::DefaultMetadataLoader;
use rustc_interface::util::rustc_version_str;
use rustc_proc_macro::bridge;
use std::{fs, io, path::Path, time::SystemTime};
use stdx::tempfile::NamedTempFile;

use crate::{
    PanicMessage, ProcMacroClientHandle, ProcMacroKind, ProcMacroSrvSpan, TrackedEnv,
    dylib::proc_macros::ProcMacros, token_stream::TokenStream,
};

pub(crate) struct Expander {
    inner: ProcMacroLibrary,
    modified_time: SystemTime,
    _file: NamedTempFile,
}

impl Expander {
    pub(crate) fn new(lib: &Utf8Path) -> io::Result<Expander> {
        // Some libraries for dynamic loading require canonicalized path even when it is
        // already absolute
        let lib = lib.canonicalize_utf8()?;
        let modified_time = fs::metadata(&lib).and_then(|it| it.modified())?;

        let file = ensure_file_with_lock_free_access(lib);
        let library = ProcMacroLibrary::open(file.path())?;

        Ok(Expander { inner: library, modified_time, _file: file })
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

struct ProcMacroLibrary {
    proc_macros: ProcMacros,
}

impl ProcMacroLibrary {
    fn open(path: &Path) -> io::Result<Self> {
        let proc_macros = rustc_span::create_default_session_globals_then(|| {
            rustc_metadata::locator::get_proc_macros(
                path,
                &DefaultMetadataLoader,
                rustc_version_str().unwrap_or("unknown"),
            )
        })?;

        Ok(ProcMacroLibrary { proc_macros: ProcMacros::new(proc_macros) })
    }
}

/// Copy the dylib to temp directory to prevent locking in Windows
#[cfg(windows)]
fn ensure_file_with_lock_free_access(path: Utf8PathBuf) -> NamedTempFile {
    if std::env::var("RA_DONT_COPY_PROC_MACRO_DLL").is_ok() {
        return NamedTempFile::from_path(path.into_std_path_buf());
    }

    (|| {
        let file_name = path.file_stem().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("File path is invalid: {path}"))
        })?;

        NamedTempFile::new_from_existing(
            &format!("proc-macro-srv-{file_name}.dll"),
            path.as_std_path(),
        )
    })()
    .unwrap_or_else(|_err| NamedTempFile::from_path(path.into_std_path_buf()))
}

#[cfg(unix)]
fn ensure_file_with_lock_free_access(path: Utf8PathBuf) -> NamedTempFile {
    NamedTempFile::from_path(path.into_std_path_buf())
}
