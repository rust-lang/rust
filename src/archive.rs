//! Creation of ar archives like for the lib and staticlib crate type

use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{self, Read, Seek};
use std::path::{Path, PathBuf};

use rustc_codegen_ssa::back::archive::ArchiveBuilder;
use rustc_session::Session;

use object::read::archive::ArchiveFile;
use object::{Object, ObjectSymbol, ReadCache, SymbolKind};

#[derive(Debug)]
enum ArchiveEntry {
    FromArchive { archive_index: usize, file_range: (u64, u64) },
    File(PathBuf),
}

pub(crate) struct ArArchiveBuilder<'a> {
    sess: &'a Session,
    dst: PathBuf,
    use_gnu_style_archive: bool,
    no_builtin_ranlib: bool,

    src_archives: Vec<File>,
    // Don't use `HashMap` here, as the order is important. `rust.metadata.bin` must always be at
    // the end of an archive for linkers to not get confused.
    entries: Vec<(Vec<u8>, ArchiveEntry)>,
}

impl<'a> ArchiveBuilder<'a> for ArArchiveBuilder<'a> {
    fn new(sess: &'a Session, output: &Path, input: Option<&Path>) -> Self {
        let (src_archives, entries) = if let Some(input) = input {
            let read_cache = ReadCache::new(File::open(input).unwrap());
            let archive = ArchiveFile::parse(&read_cache).unwrap();
            let mut entries = Vec::new();

            for entry in archive.members() {
                let entry = entry.unwrap();
                entries.push((
                    entry.name().to_vec(),
                    ArchiveEntry::FromArchive { archive_index: 0, file_range: entry.file_range() },
                ));
            }

            (vec![read_cache.into_inner()], entries)
        } else {
            (vec![], Vec::new())
        };

        ArArchiveBuilder {
            sess,
            dst: output.to_path_buf(),
            use_gnu_style_archive: sess.target.archive_format == "gnu",
            // FIXME fix builtin ranlib on macOS
            no_builtin_ranlib: sess.target.is_like_osx,

            src_archives,
            entries,
        }
    }

    fn src_files(&mut self) -> Vec<String> {
        self.entries.iter().map(|(name, _)| String::from_utf8(name.clone()).unwrap()).collect()
    }

    fn remove_file(&mut self, name: &str) {
        let index = self
            .entries
            .iter()
            .position(|(entry_name, _)| entry_name == name.as_bytes())
            .expect("Tried to remove file not existing in src archive");
        self.entries.remove(index);
    }

    fn add_file(&mut self, file: &Path) {
        self.entries.push((
            file.file_name().unwrap().to_str().unwrap().to_string().into_bytes(),
            ArchiveEntry::File(file.to_owned()),
        ));
    }

    fn add_archive<F>(&mut self, archive_path: &Path, mut skip: F) -> std::io::Result<()>
    where
        F: FnMut(&str) -> bool + 'static,
    {
        let read_cache = ReadCache::new(std::fs::File::open(&archive_path)?);
        let archive = ArchiveFile::parse(&read_cache).unwrap();
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

        self.src_archives.push(read_cache.into_inner());
        Ok(())
    }

    fn update_symbols(&mut self) {}

    fn build(mut self) {
        enum BuilderKind {
            Bsd(ar::Builder<File>),
            Gnu(ar::GnuBuilder<File>),
        }

        let sess = self.sess;

        let mut symbol_table = BTreeMap::new();

        let mut entries = Vec::new();

        for (entry_name, entry) in self.entries {
            // FIXME only read the symbol table of the object files to avoid having to keep all
            // object files in memory at once, or read them twice.
            let data = match entry {
                ArchiveEntry::FromArchive { archive_index, file_range } => {
                    // FIXME read symbols from symtab
                    let src_read_cache = &mut self.src_archives[archive_index];

                    src_read_cache.seek(io::SeekFrom::Start(file_range.0)).unwrap();
                    let mut data = std::vec::from_elem(0, usize::try_from(file_range.1).unwrap());
                    src_read_cache.read_exact(&mut data).unwrap();

                    data
                }
                ArchiveEntry::File(file) => std::fs::read(file).unwrap_or_else(|err| {
                    sess.fatal(&format!(
                        "error while reading object file during archive building: {}",
                        err
                    ));
                }),
            };

            if !self.no_builtin_ranlib {
                match object::File::parse(&*data) {
                    Ok(object) => {
                        symbol_table.insert(
                            entry_name.to_vec(),
                            object
                                .symbols()
                                .filter_map(|symbol| {
                                    if symbol.is_undefined()
                                        || symbol.is_local()
                                        || symbol.kind() != SymbolKind::Data
                                            && symbol.kind() != SymbolKind::Text
                                            && symbol.kind() != SymbolKind::Tls
                                    {
                                        None
                                    } else {
                                        symbol.name().map(|name| name.as_bytes().to_vec()).ok()
                                    }
                                })
                                .collect::<Vec<_>>(),
                        );
                    }
                    Err(err) => {
                        let err = err.to_string();
                        if err == "Unknown file magic" {
                            // Not an object file; skip it.
                        } else {
                            sess.fatal(&format!(
                                "error parsing `{}` during archive creation: {}",
                                String::from_utf8_lossy(&entry_name),
                                err
                            ));
                        }
                    }
                }
            }

            entries.push((entry_name, data));
        }

        let mut builder = if self.use_gnu_style_archive {
            BuilderKind::Gnu(
                ar::GnuBuilder::new(
                    File::create(&self.dst).unwrap_or_else(|err| {
                        sess.fatal(&format!(
                            "error opening destination during archive building: {}",
                            err
                        ));
                    }),
                    entries.iter().map(|(name, _)| name.clone()).collect(),
                    ar::GnuSymbolTableFormat::Size32,
                    symbol_table,
                )
                .unwrap(),
            )
        } else {
            BuilderKind::Bsd(
                ar::Builder::new(
                    File::create(&self.dst).unwrap_or_else(|err| {
                        sess.fatal(&format!(
                            "error opening destination during archive building: {}",
                            err
                        ));
                    }),
                    symbol_table,
                )
                .unwrap(),
            )
        };

        // Add all files
        for (entry_name, data) in entries.into_iter() {
            let header = ar::Header::new(entry_name, data.len() as u64);
            match builder {
                BuilderKind::Bsd(ref mut builder) => builder.append(&header, &mut &*data).unwrap(),
                BuilderKind::Gnu(ref mut builder) => builder.append(&header, &mut &*data).unwrap(),
            }
        }

        // Finalize archive
        std::mem::drop(builder);

        if self.no_builtin_ranlib {
            let ranlib = crate::toolchain::get_toolchain_binary(self.sess, "ranlib");

            // Run ranlib to be able to link the archive
            let status = std::process::Command::new(ranlib)
                .arg(self.dst)
                .status()
                .expect("Couldn't run ranlib");

            if !status.success() {
                self.sess.fatal(&format!("Ranlib exited with code {:?}", status.code()));
            }
        }
    }

    fn inject_dll_import_lib(
        &mut self,
        _lib_name: &str,
        _dll_imports: &[rustc_middle::middle::cstore::DllImport],
        _tmpdir: &rustc_data_structures::temp_dir::MaybeTempDir,
    ) {
        bug!("injecting dll imports is not supported");
    }
}
