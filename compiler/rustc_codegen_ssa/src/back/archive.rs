use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::memmap::Mmap;
use rustc_session::cstore::DllImport;
use rustc_session::Session;
use rustc_span::symbol::Symbol;

use super::metadata::search_for_section;

pub use ar_archive_writer::get_native_object_symbols;
use ar_archive_writer::{write_archive_to_stream, ArchiveKind, NewArchiveMember};
use object::read::archive::ArchiveFile;
use object::read::macho::FatArch;
use tempfile::Builder as TempFileBuilder;

use std::error::Error;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

// Re-exporting for rustc_codegen_llvm::back::archive
pub use crate::errors::{ArchiveBuildFailure, ExtractBundledLibsError, UnknownArchiveKind};

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
        is_direct_dependency: bool,
    ) -> PathBuf;

    fn extract_bundled_libs<'a>(
        &'a self,
        rlib: &'a Path,
        outdir: &Path,
        bundled_lib_file_names: &FxHashSet<Symbol>,
    ) -> Result<(), ExtractBundledLibsError<'_>> {
        let archive_map = unsafe {
            Mmap::map(
                File::open(rlib)
                    .map_err(|e| ExtractBundledLibsError::OpenFile { rlib, error: Box::new(e) })?,
            )
            .map_err(|e| ExtractBundledLibsError::MmapFile { rlib, error: Box::new(e) })?
        };
        let archive = ArchiveFile::parse(&*archive_map)
            .map_err(|e| ExtractBundledLibsError::ParseArchive { rlib, error: Box::new(e) })?;

        for entry in archive.members() {
            let entry = entry
                .map_err(|e| ExtractBundledLibsError::ReadEntry { rlib, error: Box::new(e) })?;
            let data = entry
                .data(&*archive_map)
                .map_err(|e| ExtractBundledLibsError::ArchiveMember { rlib, error: Box::new(e) })?;
            let name = std::str::from_utf8(entry.name())
                .map_err(|e| ExtractBundledLibsError::ConvertName { rlib, error: Box::new(e) })?;
            if !bundled_lib_file_names.contains(&Symbol::intern(name)) {
                continue; // We need to extract only native libraries.
            }
            let data = search_for_section(rlib, data, ".bundled_lib").map_err(|e| {
                ExtractBundledLibsError::ExtractSection { rlib, error: Box::<dyn Error>::from(e) }
            })?;
            std::fs::write(&outdir.join(&name), data)
                .map_err(|e| ExtractBundledLibsError::WriteFile { rlib, error: Box::new(e) })?;
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

#[must_use = "must call build() to finish building the archive"]
pub struct ArArchiveBuilder<'a> {
    sess: &'a Session,
    get_object_symbols:
        fn(buf: &[u8], f: &mut dyn FnMut(&[u8]) -> io::Result<()>) -> io::Result<bool>,

    src_archives: Vec<(PathBuf, Mmap)>,
    // Don't use an `HashMap` here, as the order is important. `lib.rmeta` needs
    // to be at the end of an archive in some cases for linkers to not get confused.
    entries: Vec<(Vec<u8>, ArchiveEntry)>,
}

#[derive(Debug)]
enum ArchiveEntry {
    FromArchive { archive_index: usize, file_range: (u64, u64) },
    File(PathBuf),
}

impl<'a> ArArchiveBuilder<'a> {
    pub fn new(
        sess: &'a Session,
        get_object_symbols: fn(
            buf: &[u8],
            f: &mut dyn FnMut(&[u8]) -> io::Result<()>,
        ) -> io::Result<bool>,
    ) -> ArArchiveBuilder<'a> {
        ArArchiveBuilder { sess, get_object_symbols, src_archives: vec![], entries: vec![] }
    }
}

fn try_filter_fat_archs(
    archs: object::read::Result<&[impl FatArch]>,
    target_arch: object::Architecture,
    archive_path: &Path,
    archive_map_data: &[u8],
) -> io::Result<Option<PathBuf>> {
    let archs = archs.map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    let desired = match archs.iter().find(|a| a.architecture() == target_arch) {
        Some(a) => a,
        None => return Ok(None),
    };

    let (mut new_f, extracted_path) = tempfile::Builder::new()
        .suffix(archive_path.file_name().unwrap())
        .tempfile()?
        .keep()
        .unwrap();

    new_f.write_all(
        desired.data(archive_map_data).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?,
    )?;

    Ok(Some(extracted_path))
}

pub fn try_extract_macho_fat_archive(
    sess: &Session,
    archive_path: &Path,
) -> io::Result<Option<PathBuf>> {
    let archive_map = unsafe { Mmap::map(File::open(&archive_path)?)? };
    let target_arch = match sess.target.arch.as_ref() {
        "aarch64" => object::Architecture::Aarch64,
        "x86_64" => object::Architecture::X86_64,
        _ => return Ok(None),
    };

    match object::macho::FatHeader::parse(&*archive_map) {
        Ok(h) if h.magic.get(object::endian::BigEndian) == object::macho::FAT_MAGIC => {
            let archs = object::macho::FatHeader::parse_arch32(&*archive_map);
            try_filter_fat_archs(archs, target_arch, archive_path, &*archive_map)
        }
        Ok(h) if h.magic.get(object::endian::BigEndian) == object::macho::FAT_MAGIC_64 => {
            let archs = object::macho::FatHeader::parse_arch64(&*archive_map);
            try_filter_fat_archs(archs, target_arch, archive_path, &*archive_map)
        }
        // Not a FatHeader at all, just return None.
        _ => Ok(None),
    }
}

impl<'a> ArchiveBuilder<'a> for ArArchiveBuilder<'a> {
    fn add_archive(
        &mut self,
        archive_path: &Path,
        mut skip: Box<dyn FnMut(&str) -> bool + 'static>,
    ) -> io::Result<()> {
        let mut archive_path = archive_path.to_path_buf();
        if self.sess.target.llvm_target.contains("-apple-macosx") {
            if let Some(new_archive_path) =
                try_extract_macho_fat_archive(&self.sess, &archive_path)?
            {
                archive_path = new_archive_path
            }
        }

        if self.src_archives.iter().any(|archive| archive.0 == archive_path) {
            return Ok(());
        }

        let archive_map = unsafe { Mmap::map(File::open(&archive_path)?)? };
        let archive = ArchiveFile::parse(&*archive_map)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        let archive_index = self.src_archives.len();

        for entry in archive.members() {
            let entry = entry.map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            let file_name = String::from_utf8(entry.name().to_vec())
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            if !skip(&file_name) {
                self.entries.push((
                    file_name.into_bytes(),
                    ArchiveEntry::FromArchive { archive_index, file_range: entry.file_range() },
                ));
            }
        }

        self.src_archives.push((archive_path, archive_map));
        Ok(())
    }

    /// Adds an arbitrary file to this archive
    fn add_file(&mut self, file: &Path) {
        self.entries.push((
            file.file_name().unwrap().to_str().unwrap().to_string().into_bytes(),
            ArchiveEntry::File(file.to_owned()),
        ));
    }

    /// Combine the provided files, rlibs, and native libraries into a single
    /// `Archive`.
    fn build(self: Box<Self>, output: &Path) -> bool {
        let sess = self.sess;
        match self.build_inner(output) {
            Ok(any_members) => any_members,
            Err(e) => sess.emit_fatal(ArchiveBuildFailure { error: e }),
        }
    }
}

impl<'a> ArArchiveBuilder<'a> {
    fn build_inner(self, output: &Path) -> io::Result<bool> {
        let archive_kind = match &*self.sess.target.archive_format {
            "gnu" => ArchiveKind::Gnu,
            "bsd" => ArchiveKind::Bsd,
            "darwin" => ArchiveKind::Darwin,
            "coff" => ArchiveKind::Coff,
            kind => {
                self.sess.emit_fatal(UnknownArchiveKind { kind });
            }
        };

        let mut entries = Vec::new();

        for (entry_name, entry) in self.entries {
            let data =
                match entry {
                    ArchiveEntry::FromArchive { archive_index, file_range } => {
                        let src_archive = &self.src_archives[archive_index];

                        let data = &src_archive.1
                            [file_range.0 as usize..file_range.0 as usize + file_range.1 as usize];

                        Box::new(data) as Box<dyn AsRef<[u8]>>
                    }
                    ArchiveEntry::File(file) => unsafe {
                        Box::new(
                            Mmap::map(File::open(file).map_err(|err| {
                                io_error_context("failed to open object file", err)
                            })?)
                            .map_err(|err| io_error_context("failed to map object file", err))?,
                        ) as Box<dyn AsRef<[u8]>>
                    },
                };

            entries.push(NewArchiveMember {
                buf: data,
                get_symbols: self.get_object_symbols,
                member_name: String::from_utf8(entry_name).unwrap(),
                mtime: 0,
                uid: 0,
                gid: 0,
                perms: 0o644,
            })
        }

        // Write to a temporary file first before atomically renaming to the final name.
        // This prevents programs (including rustc) from attempting to read a partial archive.
        // It also enables writing an archive with the same filename as a dependency on Windows as
        // required by a test.
        let mut archive_tmpfile = TempFileBuilder::new()
            .suffix(".temp-archive")
            .tempfile_in(output.parent().unwrap_or_else(|| Path::new("")))
            .map_err(|err| io_error_context("couldn't create a temp file", err))?;

        write_archive_to_stream(
            archive_tmpfile.as_file_mut(),
            &entries,
            true,
            archive_kind,
            true,
            false,
        )?;

        let any_entries = !entries.is_empty();
        drop(entries);
        // Drop src_archives to unmap all input archives, which is necessary if we want to write the
        // output archive to the same location as an input archive on Windows.
        drop(self.src_archives);

        archive_tmpfile
            .persist(output)
            .map_err(|err| io_error_context("failed to rename archive file", err.error))?;

        Ok(any_entries)
    }
}

fn io_error_context(context: &str, err: io::Error) -> io::Error {
    io::Error::new(io::ErrorKind::Other, format!("{context}: {err}"))
}
