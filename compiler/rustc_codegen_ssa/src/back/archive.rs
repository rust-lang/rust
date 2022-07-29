use rustc_data_structures::temp_dir::MaybeTempDir;
use rustc_session::cstore::DllImport;
use rustc_session::Session;

use std::io;
use std::path::{Path, PathBuf};

pub(super) fn find_library(
    name: &str,
    verbatim: bool,
    search_paths: &[PathBuf],
    sess: &Session,
) -> PathBuf {
    // On Windows, static libraries sometimes show up as libfoo.a and other
    // times show up as foo.lib
    let oslibname = if verbatim {
        name.to_string()
    } else {
        format!("{}{}{}", sess.target.staticlib_prefix, name, sess.target.staticlib_suffix)
    };
    let unixlibname = format!("lib{}.a", name);

    for path in search_paths {
        debug!("looking for {} inside {:?}", name, path);
        let test = path.join(&oslibname);
        if test.exists() {
            return test;
        }
        if oslibname != unixlibname {
            let test = path.join(&unixlibname);
            if test.exists() {
                return test;
            }
        }
    }
    sess.fatal(&format!(
        "could not find native static library `{}`, \
                         perhaps an -L flag is missing?",
        name
    ));
}

pub trait ArchiveBuilder<'a> {
    fn new(sess: &'a Session, output: &Path) -> Self;

    fn add_file(&mut self, path: &Path);

    fn add_archive<F>(&mut self, archive: &Path, skip: F) -> io::Result<()>
    where
        F: FnMut(&str) -> bool + 'static;

    fn build(self) -> bool;

    fn sess(&self) -> &Session;

    /// Creates a DLL Import Library <https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-creation#creating-an-import-library>.
    /// and returns the path on disk to that import library.
    /// This functions doesn't take `self` so that it can be called from
    /// `linker_with_args`, which is specialized on `ArchiveBuilder` but
    /// doesn't take or create an instance of that type.
    fn create_dll_import_lib(
        sess: &Session,
        lib_name: &str,
        dll_imports: &[DllImport],
        tmpdir: &Path,
    ) -> PathBuf;

    /// Creates a DLL Import Library <https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-creation#creating-an-import-library>
    /// and adds it to the current compilation's set of archives.
    fn inject_dll_import_lib(
        &mut self,
        lib_name: &str,
        dll_imports: &[DllImport],
        tmpdir: &MaybeTempDir,
    ) {
        let output_path =
            Self::create_dll_import_lib(self.sess(), lib_name, dll_imports, tmpdir.as_ref());

        self.add_archive(&output_path, |_| false).unwrap_or_else(|e| {
            self.sess().fatal(&format!(
                "failed to add native library {}: {}",
                output_path.display(),
                e
            ));
        });
    }
}
