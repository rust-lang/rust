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

pub trait ArchiveBuilderBuilder {
    fn new_archive_builder<'a>(&self, sess: &'a Session) -> Box<dyn ArchiveBuilder<'a> + 'a>;

    /// Creates a DLL Import Library <https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-creation#creating-an-import-library>.
    /// and returns the path on disk to that import library.
    /// This functions doesn't take `self` so that it can be called from
    /// `linker_with_args`, which is specialized on `ArchiveBuilder` but
    /// doesn't take or create an instance of that type.
    fn create_dll_import_lib(
        &self,
        sess: &Session,
        lib_name: &str,
        dll_imports: &[DllImport],
        tmpdir: &Path,
    ) -> PathBuf;
}

pub trait ArchiveBuilder<'a> {
    fn add_file(&mut self, path: &Path);

    fn add_archive(
        &mut self,
        archive: &Path,
        skip: Box<dyn FnMut(&str) -> bool + 'static>,
    ) -> io::Result<()>;

    fn build(self: Box<Self>, output: &Path) -> bool;
}
