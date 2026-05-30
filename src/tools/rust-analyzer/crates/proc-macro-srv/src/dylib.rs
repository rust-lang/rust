//! Handles dynamic library loading for proc macro

mod proc_macros;

use rustc_codegen_ssa::back::metadata::DefaultMetadataLoader;
use rustc_interface::util::rustc_version_str;
use rustc_proc_macro::bridge;
use rustc_session::config::host_tuple;
use rustc_target::spec::{Target, TargetTuple};
use std::path::Path;
use std::{fs, io, time::SystemTime};
use temp_dir::TempDir;

use paths::{Utf8Path, Utf8PathBuf};

use crate::{
    PanicMessage, ProcMacroClientHandle, ProcMacroKind, ProcMacroSrvSpan, TrackedEnv,
    dylib::proc_macros::ProcMacros, token_stream::TokenStream,
};

pub(crate) struct Expander {
    inner: ProcMacroLibrary,
    modified_time: SystemTime,
}

impl Expander {
    pub(crate) fn new(temp_dir: &TempDir, lib: &Utf8Path) -> io::Result<Expander> {
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

struct ProcMacroLibrary {
    proc_macros: ProcMacros,
}

impl ProcMacroLibrary {
    fn open(path: &Utf8Path) -> io::Result<Self> {
        let proc_macros = rustc_span::create_default_session_globals_then(|| {
            // FIXME support wasm proc-macros
            let (target, _) =
                Target::search(&TargetTuple::from_tuple(host_tuple()), Path::new(""), false)
                    .unwrap();
            rustc_metadata::locator::get_proc_macros(
                &target,
                path.as_ref(),
                &DefaultMetadataLoader,
                rustc_version_str().unwrap_or("unknown"),
            )
        })?;

        Ok(ProcMacroLibrary { proc_macros: ProcMacros::new(proc_macros) })
    }
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
