use std::error::Error;
use std::path::Path;
use std::time::Duration;

use rustc_fs_util::try_canonicalize;
use rustc_proc_macro::bridge::client::Client as ProcMacroClient;
use rustc_session::StableCrateId;
use tracing::debug;

use crate::locator::CrateError;

fn format_dlopen_err(e: &(dyn std::error::Error + 'static)) -> String {
    e.sources().map(|e| format!(": {e}")).collect()
}

fn attempt_load_dylib(path: &Path) -> Result<libloading::Library, libloading::Error> {
    #[cfg(target_os = "aix")]
    if let Some(ext) = path.extension()
        && ext.eq("a")
    {
        // On AIX, we ship all libraries as .a big_af archive
        // the expected format is lib<name>.a(libname.so) for the actual
        // dynamic library
        let library_name = path.file_stem().expect("expect a library name");
        let mut archive_member = std::ffi::OsString::from("a(");
        archive_member.push(library_name);
        archive_member.push(".so)");
        let new_path = path.with_extension(archive_member);

        // On AIX, we need RTLD_MEMBER to dlopen an archived shared
        let flags = libc::RTLD_LAZY | libc::RTLD_LOCAL | libc::RTLD_MEMBER;
        return unsafe { libloading::os::unix::Library::open(Some(&new_path), flags) }
            .map(|lib| lib.into());
    }

    unsafe { libloading::Library::new(&path) }
}

// On Windows the compiler would sometimes intermittently fail to open the
// proc-macro DLL with `Error::LoadLibraryExW`. It is suspected that something in the
// system still holds a lock on the file, so we retry a few times before calling it
// an error.
fn load_dylib(path: &Path, max_attempts: usize) -> Result<libloading::Library, String> {
    assert!(max_attempts > 0);

    let mut last_error = None;

    for attempt in 0..max_attempts {
        debug!("Attempt to load proc-macro `{}`.", path.display());
        match attempt_load_dylib(path) {
            Ok(lib) => {
                if attempt > 0 {
                    debug!(
                        "Loaded proc-macro `{}` after {} attempts.",
                        path.display(),
                        attempt + 1
                    );
                }
                return Ok(lib);
            }
            Err(err) => {
                // Only try to recover from this specific error.
                if !matches!(err, libloading::Error::LoadLibraryExW { .. }) {
                    debug!("Failed to load proc-macro `{}`. Not retrying", path.display());
                    let err = format_dlopen_err(&err);
                    // We include the path of the dylib in the error ourselves, so
                    // if it's in the error, we strip it.
                    if let Some(err) = err.strip_prefix(&format!(": {}", path.display())) {
                        return Err(err.to_string());
                    }
                    return Err(err);
                }

                last_error = Some(err);
                std::thread::sleep(Duration::from_millis(100));
                debug!("Failed to load proc-macro `{}`. Retrying.", path.display());
            }
        }
    }

    debug!("Failed to load proc-macro `{}` even after {} attempts.", path.display(), max_attempts);

    let last_error = last_error.unwrap();
    let message = if let Some(src) = last_error.source() {
        format!("{} ({src}) (retried {max_attempts} times)", format_dlopen_err(&last_error))
    } else {
        format!("{} (retried {max_attempts} times)", format_dlopen_err(&last_error))
    };
    Err(message)
}

pub enum DylibError {
    DlOpen(String, String),
    DlSym(String, String),
}

impl From<DylibError> for CrateError {
    fn from(err: DylibError) -> CrateError {
        match err {
            DylibError::DlOpen(path, err) => CrateError::DlOpen(path, err),
            DylibError::DlSym(path, err) => CrateError::DlSym(path, err),
        }
    }
}

pub unsafe fn load_symbol_from_dylib<T: Copy>(
    path: &Path,
    sym_name: &str,
) -> Result<T, DylibError> {
    // Make sure the path contains a / or the linker will search for it.
    let path = try_canonicalize(path).unwrap();
    let lib =
        load_dylib(&path, 5).map_err(|err| DylibError::DlOpen(path.display().to_string(), err))?;

    let sym = unsafe { lib.get::<T>(sym_name.as_bytes()) }
        .map_err(|err| DylibError::DlSym(path.display().to_string(), format_dlopen_err(&err)))?;

    // Intentionally leak the dynamic library. We can't ever unload it
    // since the library can make things that will live arbitrarily long.
    let sym = unsafe { sym.into_raw() };
    std::mem::forget(lib);

    Ok(*sym)
}

pub(crate) fn dlsym_proc_macros(
    path: &Path,
    stable_crate_id: StableCrateId,
) -> Result<&'static [ProcMacroClient], DylibError> {
    let sym_name = rustc_session::generate_proc_macro_decls_symbol(stable_crate_id);
    debug!("trying to dlsym proc_macros {} for symbol `{}`", path.display(), sym_name);

    unsafe {
        // FIXME(bjorn3) this depends on the unstable slice memory layout
        let result = crate::load_symbol_from_dylib::<*const &[ProcMacroClient]>(path, &sym_name);
        match result {
            Ok(result) => {
                debug!("loaded dlsym proc_macros {} for symbol `{}`", path.display(), sym_name);
                Ok(*result)
            }
            Err(err) => {
                debug!("failed to dlsym proc_macros {} for symbol `{}`", path.display(), sym_name);
                Err(err.into())
            }
        }
    }
}
