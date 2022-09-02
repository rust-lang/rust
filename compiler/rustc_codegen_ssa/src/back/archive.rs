use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::memmap::Mmap;
use rustc_session::cstore::DllImport;
use rustc_session::Session;
use rustc_span::symbol::Symbol;

use object::read::archive::ArchiveFile;

use std::fmt::Display;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

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

    fn extract_bundled_libs(
        &self,
        rlib: &Path,
        outdir: &Path,
        bundled_lib_file_names: &FxHashSet<Symbol>,
    ) -> Result<(), String> {
        let message = |msg: &str, e: &dyn Display| format!("{} '{}': {}", msg, &rlib.display(), e);
        let archive_map = unsafe {
            Mmap::map(File::open(rlib).map_err(|e| message("failed to open file", &e))?)
                .map_err(|e| message("failed to mmap file", &e))?
        };
        let archive = ArchiveFile::parse(&*archive_map)
            .map_err(|e| message("failed to parse archive", &e))?;

        for entry in archive.members() {
            let entry = entry.map_err(|e| message("failed to read entry", &e))?;
            let data = entry
                .data(&*archive_map)
                .map_err(|e| message("failed to get data from archive member", &e))?;
            let name = std::str::from_utf8(entry.name())
                .map_err(|e| message("failed to convert name", &e))?;
            if !bundled_lib_file_names.contains(&Symbol::intern(name)) {
                continue; // We need to extract only native libraries.
            }
            std::fs::write(&outdir.join(&name), data)
                .map_err(|e| message("failed to write file", &e))?;
        }
        Ok(())
    }
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
