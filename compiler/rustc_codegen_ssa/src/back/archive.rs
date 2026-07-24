use std::borrow::Cow;
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use ar_archive_writer::{
    ArchiveKind, COFFShortExport, MachineTypes, NewArchiveMember, write_archive_to_stream,
};
pub use ar_archive_writer::{DEFAULT_OBJECT_READER, ObjectReader};
use object::read::archive::{ArchiveFile, ArchiveKind as ObjectArchiveKind};
use object::read::macho::FatArch;
use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_data_structures::memmap::Mmap;
use rustc_fs_util::TempDirBuilder;
use rustc_metadata::EncodedMetadata;
use rustc_session::Session;
use rustc_span::Symbol;
use rustc_target::spec::Arch;
use tracing::trace;

use super::metadata::{create_compressed_metadata_file, search_for_section};
use super::rmeta_link::{self, RmetaLinkCache};
use super::symbol_edit::{apply_edits, collect_internal_names};
// Public for ArchiveBuilderBuilder::extract_bundled_libs
pub use crate::diagnostics::ExtractBundledLibsError;
use crate::diagnostics::{ArchiveBuildFailure, ErrorCreatingImportLibrary, UnknownArchiveKind};

/// An item to be included in an import library.
/// This is a slimmed down version of `COFFShortExport` from `ar-archive-writer`.
pub struct ImportLibraryItem {
    /// The name to be exported.
    pub name: String,
    /// The ordinal to be exported, if any.
    pub ordinal: Option<u16>,
    /// The original, decorated name if `name` is not decorated.
    pub symbol_name: Option<String>,
    /// True if this is a data export, false if it is a function export.
    pub is_data: bool,
}

impl ImportLibraryItem {
    fn into_coff_short_export(self, sess: &Session) -> COFFShortExport {
        let import_name = (sess.target.arch == Arch::Arm64EC).then(|| self.name.clone());
        COFFShortExport {
            name: self.name,
            ext_name: None,
            symbol_name: self.symbol_name,
            import_name,
            export_as: None,
            ordinal: self.ordinal.unwrap_or(0),
            noname: self.ordinal.is_some(),
            data: self.is_data,
            private: false,
            constant: false,
        }
    }
}

pub trait ArchiveBuilderBuilder {
    fn new_archive_builder<'a>(&self, sess: &'a Session) -> Box<dyn ArchiveBuilder + 'a>;

    fn create_dylib_metadata_wrapper(
        &self,
        sess: &Session,
        metadata: &EncodedMetadata,
        symbol_name: &str,
    ) -> Vec<u8> {
        create_compressed_metadata_file(sess, metadata, symbol_name)
    }

    /// Creates a DLL Import Library <https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-creation#creating-an-import-library>.
    /// and returns the path on disk to that import library.
    /// This functions doesn't take `self` so that it can be called from
    /// `linker_with_args`, which is specialized on `ArchiveBuilder` but
    /// doesn't take or create an instance of that type.
    fn create_dll_import_lib(
        &self,
        sess: &Session,
        lib_name: &str,
        items: Vec<ImportLibraryItem>,
        output_path: &Path,
    ) {
        trace!("creating import library");
        trace!("  dll_name {:#?}", lib_name);
        trace!("  output_path {}", output_path.display());
        trace!(
            "  import names: {}",
            items
                .iter()
                .map(|ImportLibraryItem { name, .. }| name.clone())
                .collect::<Vec<_>>()
                .join(", "),
        );

        // All import names are Rust identifiers and therefore cannot contain \0 characters.
        // FIXME: when support for #[link_name] is implemented, ensure that the import names
        // still don't contain any \0 characters. Also need to check that the names don't
        // contain substrings like " @" or "NONAME" that are keywords or otherwise reserved
        // in definition files.

        let mut file = match fs::File::create_new(&output_path) {
            Ok(file) => file,
            Err(error) => sess
                .dcx()
                .emit_fatal(ErrorCreatingImportLibrary { lib_name, error: error.to_string() }),
        };

        let exports =
            items.into_iter().map(|item| item.into_coff_short_export(sess)).collect::<Vec<_>>();
        let machine = match &sess.target.arch {
            Arch::X86_64 => MachineTypes::AMD64,
            Arch::X86 => MachineTypes::I386,
            Arch::AArch64 => MachineTypes::ARM64,
            Arch::Arm64EC => MachineTypes::ARM64EC,
            Arch::Arm => MachineTypes::ARMNT,
            cpu => panic!("unsupported cpu type {cpu}"),
        };

        if let Err(error) = ar_archive_writer::write_import_library(
            &mut file,
            lib_name,
            &exports,
            machine,
            !sess.target.is_like_msvc,
            // Enable compatibility with MSVC's `/WHOLEARCHIVE` flag.
            // Without this flag a duplicate symbol error would be emitted
            // when linking a rust staticlib using `/WHOLEARCHIVE`.
            // See #129020
            true,
            &[],
        ) {
            sess.dcx()
                .emit_fatal(ErrorCreatingImportLibrary { lib_name, error: error.to_string() });
        }
    }

    fn extract_bundled_libs<'a>(
        &'a self,
        rlib: &'a Path,
        outdir: &Path,
        bundled_lib_file_names: &FxIndexSet<Symbol>,
    ) -> Result<(), ExtractBundledLibsError<'a>> {
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

pub enum AddArchiveKind<'a> {
    Rlib(&'a mut RmetaLinkCache, /*skip*/ &'a dyn Fn(&str, ArchiveEntryKind) -> bool),
    Other,
}

pub struct ArchiveSymbols {
    pub exported: FxHashSet<String>,
    pub rename_suffix: Option<String>,
    pub hide: bool,
}

pub trait ArchiveBuilder {
    fn add_file(&mut self, path: &Path, kind: ArchiveEntryKind);

    fn add_archive(&mut self, archive: &Path, kind: AddArchiveKind<'_>) -> io::Result<()>;

    fn build(self: Box<Self>, output: &Path, symbols: Option<ArchiveSymbols>) -> bool;
}

fn target_archive_format_to_object_kind(format: &str) -> Option<ObjectArchiveKind> {
    match format {
        "gnu" => Some(ObjectArchiveKind::Gnu),
        "bsd" => Some(ObjectArchiveKind::Bsd),
        "darwin" => Some(ObjectArchiveKind::Bsd64),
        "coff" => Some(ObjectArchiveKind::Coff),
        "aix_big" => Some(ObjectArchiveKind::AixBig),
        _ => None,
    }
}

fn archive_kinds_compatible(actual: ObjectArchiveKind, expected: ObjectArchiveKind) -> bool {
    if actual == expected {
        return true;
    }
    matches!(
        (actual, expected),
        // An archive without long filenames or symbol table is detected as Unknown;
        // this is compatible with any target format.
        (ObjectArchiveKind::Unknown, _)
        // 64-bit symbol table variants are compatible with their 32-bit counterparts
        | (ObjectArchiveKind::Gnu64, ObjectArchiveKind::Gnu)
        | (ObjectArchiveKind::Gnu, ObjectArchiveKind::Gnu64)
        | (ObjectArchiveKind::Bsd64, ObjectArchiveKind::Bsd)
        | (ObjectArchiveKind::Bsd, ObjectArchiveKind::Bsd64)
        // GNU and COFF archives share the same magic and member header format;
        // only the symbol table layout differs.
        | (ObjectArchiveKind::Gnu, ObjectArchiveKind::Coff)
        | (ObjectArchiveKind::Coff, ObjectArchiveKind::Gnu)
        | (ObjectArchiveKind::Gnu64, ObjectArchiveKind::Coff)
    )
}

fn archive_kind_display_name(kind: ObjectArchiveKind) -> String {
    match kind {
        ObjectArchiveKind::Gnu | ObjectArchiveKind::Gnu64 => "GNU".to_string(),
        ObjectArchiveKind::Bsd => "BSD".to_string(),
        ObjectArchiveKind::Bsd64 => "Darwin".to_string(),
        ObjectArchiveKind::Coff => "COFF".to_string(),
        ObjectArchiveKind::AixBig => "AIX big".to_string(),
        _ => format!("{kind:?}"),
    }
}

pub struct ArArchiveBuilderBuilder;

impl ArchiveBuilderBuilder for ArArchiveBuilderBuilder {
    fn new_archive_builder<'a>(&self, sess: &'a Session) -> Box<dyn ArchiveBuilder + 'a> {
        Box::new(ArArchiveBuilder::new(sess, &DEFAULT_OBJECT_READER))
    }
}

#[must_use = "must call build() to finish building the archive"]
pub struct ArArchiveBuilder<'a> {
    sess: &'a Session,
    object_reader: &'static ObjectReader,

    src_archives: Vec<(PathBuf, Mmap)>,
    // Don't use an `HashMap` here, as the order is important. `lib.rmeta` needs
    // to be at the end of an archive in some cases for linkers to not get confused.
    entries: Vec<(Vec<u8>, ArchiveEntry)>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ArchiveEntryKind {
    /// Object file produced from Rust code.
    RustObj,
    /// Anything else, introduce new variants as needed.
    Other,
}

#[derive(Debug)]
enum ArchiveEntrySource {
    Archive { archive_index: usize, file_range: (u64, u64) },
    File(PathBuf),
}

#[derive(Debug)]
struct ArchiveEntry {
    source: ArchiveEntrySource,
    kind: ArchiveEntryKind,
}

impl<'a> ArArchiveBuilder<'a> {
    pub fn new(sess: &'a Session, object_reader: &'static ObjectReader) -> ArArchiveBuilder<'a> {
        ArArchiveBuilder { sess, object_reader, src_archives: vec![], entries: vec![] }
    }
}

fn try_filter_fat_archs(
    archs: &[impl FatArch],
    target_arch: object::Architecture,
    archive_path: &Path,
    archive_map_data: &[u8],
) -> io::Result<Option<PathBuf>> {
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
    let target_arch = match sess.target.arch {
        Arch::AArch64 => object::Architecture::Aarch64,
        Arch::X86_64 => object::Architecture::X86_64,
        _ => return Ok(None),
    };

    if let Ok(h) = object::read::macho::MachOFatFile32::parse(&*archive_map) {
        let archs = h.arches();
        try_filter_fat_archs(archs, target_arch, archive_path, &*archive_map)
    } else if let Ok(h) = object::read::macho::MachOFatFile64::parse(&*archive_map) {
        let archs = h.arches();
        try_filter_fat_archs(archs, target_arch, archive_path, &*archive_map)
    } else {
        // Not a FatHeader at all, just return None.
        Ok(None)
    }
}

impl<'a> ArchiveBuilder for ArArchiveBuilder<'a> {
    fn add_archive(
        &mut self,
        archive_path: &Path,
        mut ar_kind: AddArchiveKind<'_>,
    ) -> io::Result<()> {
        let mut archive_path = archive_path.to_path_buf();
        if self.sess.target.llvm_target.contains("-apple-macosx")
            && let Some(new_archive_path) = try_extract_macho_fat_archive(self.sess, &archive_path)?
        {
            archive_path = new_archive_path
        }

        if self.src_archives.iter().any(|archive| archive.0 == archive_path) {
            return Ok(());
        }

        let archive_map = unsafe { Mmap::map(File::open(&archive_path)?)? };
        let archive = ArchiveFile::parse(&*archive_map)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        let skip = match &ar_kind {
            AddArchiveKind::Rlib(_, skip) => Some(*skip),
            AddArchiveKind::Other => None,
        };
        let metadata_link = match &mut ar_kind {
            AddArchiveKind::Rlib(cache, _) => cache.get_or_insert_with(&archive_path, || {
                rmeta_link::read(&archive, &archive_map, &archive_path)
            }),
            AddArchiveKind::Other => None,
        };
        let archive_index = self.src_archives.len();

        if let Some(expected_kind) =
            target_archive_format_to_object_kind(&self.sess.target.archive_format)
        {
            let actual_kind = archive.kind();
            if !archive_kinds_compatible(actual_kind, expected_kind) {
                self.sess.dcx().emit_warn(crate::diagnostics::IncompatibleArchiveFormat {
                    path: archive_path.clone(),
                    actual: archive_kind_display_name(actual_kind),
                    expected: archive_kind_display_name(expected_kind),
                });
            }
        }

        for entry in archive.members() {
            let entry = entry.map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            let file_name = String::from_utf8(entry.name().to_vec())
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            let kind = if metadata_link
                .as_ref()
                .is_some_and(|m| m.rust_object_files.iter().any(|f| f == &file_name))
            {
                ArchiveEntryKind::RustObj
            } else {
                ArchiveEntryKind::Other
            };
            let drop = match skip {
                Some(skip) => skip(&file_name, kind),
                None => false,
            };
            if !drop {
                let source = if entry.is_thin() {
                    let member_path = archive_path.parent().unwrap().join(Path::new(&file_name));
                    ArchiveEntrySource::File(member_path)
                } else {
                    ArchiveEntrySource::Archive { archive_index, file_range: entry.file_range() }
                };
                self.entries.push((file_name.into_bytes(), ArchiveEntry { source, kind }));
            }
        }

        self.src_archives.push((archive_path, archive_map));
        Ok(())
    }

    /// Adds an arbitrary file to this archive
    fn add_file(&mut self, file: &Path, kind: ArchiveEntryKind) {
        self.entries.push((
            file.file_name().unwrap().to_str().unwrap().to_string().into_bytes(),
            ArchiveEntry { source: ArchiveEntrySource::File(file.to_owned()), kind },
        ));
    }

    /// Combine the provided files, rlibs, and native libraries into a single
    /// `Archive`.
    fn build(self: Box<Self>, output: &Path, symbols: Option<ArchiveSymbols>) -> bool {
        let sess = self.sess;
        match self.build_inner(output, symbols) {
            Ok(any_members) => any_members,
            Err(error) => {
                sess.dcx().emit_fatal(ArchiveBuildFailure { path: output.to_owned(), error })
            }
        }
    }
}

impl<'a> ArArchiveBuilder<'a> {
    fn build_inner(self, output: &Path, symbols: Option<ArchiveSymbols>) -> io::Result<bool> {
        let archive_kind = match &*self.sess.target.archive_format {
            "gnu" => ArchiveKind::Gnu,
            "bsd" => ArchiveKind::Bsd,
            "darwin" => ArchiveKind::Darwin,
            "coff" => ArchiveKind::Coff,
            "aix_big" => ArchiveKind::AixBig,
            kind => {
                self.sess.dcx().emit_fatal(UnknownArchiveKind { kind });
            }
        };

        // Collect all internally-defined symbol names across every Rust object file.
        // This set is needed because rename must also apply to *undefined* references
        // (cross-object calls within the staticlib), but we cannot use `!exported.contains(name)`
        // alone — that would also match external C symbols like `malloc` which must not be renamed.
        let rename = if let Some(sym) = &symbols
            && let Some(rename_suffix) = sym.rename_suffix.as_deref()
        {
            let mut names = FxHashSet::default();
            for (_, entry) in &self.entries {
                if entry.kind != ArchiveEntryKind::RustObj {
                    continue;
                }
                match &entry.source {
                    ArchiveEntrySource::Archive { archive_index, file_range } => {
                        let src_archive = &self.src_archives[*archive_index];
                        let start = file_range.0 as usize;
                        let end = start + file_range.1 as usize;
                        if let Some(data) = src_archive.1.get(start..end) {
                            collect_internal_names(data, &sym.exported, &mut names);
                        }
                    }
                    ArchiveEntrySource::File(file) => {
                        if let Ok(data) = fs::read(file) {
                            collect_internal_names(&data, &sym.exported, &mut names);
                        }
                    }
                }
            }
            Some((names, rename_suffix))
        } else {
            None
        };

        let mut entries = Vec::new();

        for (entry_name, entry) in self.entries {
            let data: Box<dyn AsRef<[u8]>> = match entry.source {
                ArchiveEntrySource::Archive { archive_index, file_range } => {
                    let src_archive = &self.src_archives[archive_index];
                    let archive_data = &src_archive.1;
                    let start = file_range.0 as usize;
                    let end = start + file_range.1 as usize;
                    let Some(data) = archive_data.get(start..end) else {
                        return Err(io_error_context(
                            "invalid archive member",
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "archive member at offset {start} with size {} \
                                         exceeds archive size {} in `{}`",
                                    file_range.1,
                                    archive_data.len(),
                                    src_archive.0.display(),
                                ),
                            ),
                        ));
                    };

                    if entry.kind == ArchiveEntryKind::RustObj
                        && let Some(sym) = &symbols
                    {
                        Box::new(apply_edits(data, &sym.exported, sym.hide, rename.as_ref()))
                    } else {
                        Box::new(data)
                    }
                }
                ArchiveEntrySource::File(file) => unsafe {
                    let mmap = Mmap::map(
                        File::open(file)
                            .map_err(|err| io_error_context("failed to open object file", err))?,
                    )
                    .map_err(|err| io_error_context("failed to map object file", err))?;
                    if entry.kind == ArchiveEntryKind::RustObj
                        && let Some(sym) = &symbols
                    {
                        let edited = apply_edits(&mmap, &sym.exported, sym.hide, rename.as_ref());
                        match edited {
                            Cow::Borrowed(_) => Box::new(mmap) as Box<dyn AsRef<[u8]>>,
                            Cow::Owned(v) => Box::new(v),
                        }
                    } else {
                        Box::new(mmap) as Box<dyn AsRef<[u8]>>
                    }
                },
            };

            entries.push(NewArchiveMember {
                buf: data,
                object_reader: self.object_reader,
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
        // The tempfile crate currently uses 0o600 as mode for the temporary files and directories
        // it creates. We need it to be the default mode for back compat reasons however. (See
        // #107495) To handle this we are telling tempfile to create a temporary directory instead
        // and then inside this directory create a file using File::create.
        let archive_tmpdir = TempDirBuilder::new()
            .suffix(".temp-archive")
            .tempdir_in(output.parent().unwrap_or_else(|| Path::new("")))
            .map_err(|err| {
                io_error_context("couldn't create a directory for the temp file", err)
            })?;
        let archive_tmpfile_path = archive_tmpdir.path().join("tmp.a");
        let archive_tmpfile = File::create_new(&archive_tmpfile_path)
            .map_err(|err| io_error_context("couldn't create the temp file", err))?;

        let mut archive_tmpfile = BufWriter::new(archive_tmpfile);
        write_archive_to_stream(
            &mut archive_tmpfile,
            &entries,
            archive_kind,
            false,
            /* is_ec = */ Some(self.sess.target.arch == Arch::Arm64EC),
        )?;
        archive_tmpfile.flush()?;
        drop(archive_tmpfile);

        let any_entries = !entries.is_empty();
        drop(entries);
        // Drop src_archives to unmap all input archives, which is necessary if we want to write the
        // output archive to the same location as an input archive on Windows.
        drop(self.src_archives);

        fs::rename(archive_tmpfile_path, output)
            .map_err(|err| io_error_context("failed to rename archive file", err))?;
        archive_tmpdir
            .close()
            .map_err(|err| io_error_context("failed to remove temporary directory", err))?;

        Ok(any_entries)
    }
}

fn io_error_context(context: &str, err: io::Error) -> io::Error {
    io::Error::new(io::ErrorKind::Other, format!("{context}: {err}"))
}
