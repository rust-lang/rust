use std::fs::File;
use std::path::{Path, PathBuf};

use crate::prelude::*;

use rustc_codegen_ssa::back::archive::{find_library, ArchiveBuilder};
use rustc_codegen_ssa::{METADATA_FILENAME, RLIB_BYTECODE_EXTENSION};

struct ArchiveConfig<'a> {
    sess: &'a Session,
    dst: PathBuf,
    lib_search_paths: Vec<PathBuf>,
    use_native_ar: bool,
    use_gnu_style_archive: bool,
}

#[derive(Debug)]
enum ArchiveEntry {
    FromArchive {
        archive_index: usize,
        entry_index: usize,
    },
    File(PathBuf),
}

pub struct ArArchiveBuilder<'a> {
    config: ArchiveConfig<'a>,
    src_archives: Vec<(PathBuf, ar::Archive<File>)>,
    // Don't use `HashMap` here, as the order is important. `rust.metadata.bin` must always be at
    // the end of an archive for linkers to not get confused.
    entries: Vec<(String, ArchiveEntry)>,
    update_symbols: bool,
}

impl<'a> ArchiveBuilder<'a> for ArArchiveBuilder<'a> {
    fn new(sess: &'a Session, output: &Path, input: Option<&Path>) -> Self {
        use rustc_codegen_ssa::back::link::archive_search_paths;
        let config = ArchiveConfig {
            sess,
            dst: output.to_path_buf(),
            lib_search_paths: archive_search_paths(sess),
            use_native_ar: false,
            // FIXME test for linux and System V derivatives instead
            use_gnu_style_archive: sess.target.target.options.archive_format == "gnu",
        };

        let (src_archives, entries) = if let Some(input) = input {
            let mut archive = ar::Archive::new(File::open(input).unwrap());
            let mut entries = Vec::new();

            let mut i = 0;
            while let Some(entry) = archive.next_entry() {
                let entry = entry.unwrap();
                entries.push((
                    String::from_utf8(entry.header().identifier().to_vec()).unwrap(),
                    ArchiveEntry::FromArchive {
                        archive_index: 0,
                        entry_index: i,
                    },
                ));
                i += 1;
            }

            (vec![(input.to_owned(), archive)], entries)
        } else {
            (vec![], Vec::new())
        };

        ArArchiveBuilder {
            config,
            src_archives,
            entries,
            update_symbols: false,
        }
    }

    fn src_files(&mut self) -> Vec<String> {
        self.entries.iter().map(|(name, _)| name.clone()).collect()
    }

    fn remove_file(&mut self, name: &str) {
        let index = self
            .entries
            .iter()
            .position(|(entry_name, _)| entry_name == name)
            .expect("Tried to remove file not existing in src archive");
        self.entries.remove(index);
    }

    fn add_file(&mut self, file: &Path) {
        self.entries.push((
            file.file_name().unwrap().to_str().unwrap().to_string(),
            ArchiveEntry::File(file.to_owned()),
        ));
    }

    fn add_native_library(&mut self, name: rustc_ast::ast::Name) {
        let location = find_library(name, &self.config.lib_search_paths, self.config.sess);
        self.add_archive(location.clone(), |_| false)
            .unwrap_or_else(|e| {
                panic!(
                    "failed to add native library {}: {}",
                    location.to_string_lossy(),
                    e
                );
            });
    }

    fn add_rlib(
        &mut self,
        rlib: &Path,
        name: &str,
        lto: bool,
        skip_objects: bool,
    ) -> std::io::Result<()> {
        let obj_start = name.to_owned();

        self.add_archive(rlib.to_owned(), move |fname: &str| {
            // Ignore bytecode/metadata files, no matter the name.
            if fname.ends_with(RLIB_BYTECODE_EXTENSION) || fname == METADATA_FILENAME {
                return true;
            }

            // Don't include Rust objects if LTO is enabled
            if lto && fname.starts_with(&obj_start) && fname.ends_with(".o") {
                return true;
            }

            // Otherwise if this is *not* a rust object and we're skipping
            // objects then skip this file
            if skip_objects && (!fname.starts_with(&obj_start) || !fname.ends_with(".o")) {
                return true;
            }

            // ok, don't skip this
            return false;
        })
    }

    fn update_symbols(&mut self) {
        self.update_symbols = true;
    }

    fn build(mut self) {
        use std::process::Command;

        fn add_file_using_ar(archive: &Path, file: &Path) {
            Command::new("ar")
                .arg("r") // add or replace file
                .arg("-c") // silence created file message
                .arg(archive)
                .arg(&file)
                .status()
                .unwrap();
        }

        enum BuilderKind<'a> {
            Bsd(ar::Builder<File>),
            Gnu(ar::GnuBuilder<File>),
            NativeAr(&'a Path),
        }

        let mut builder = if self.config.use_native_ar {
            BuilderKind::NativeAr(&self.config.dst)
        } else if self.config.use_gnu_style_archive {
            BuilderKind::Gnu(ar::GnuBuilder::new(
                File::create(&self.config.dst).unwrap(),
                self.entries
                    .iter()
                    .map(|(name, _)| name.as_bytes().to_vec())
                    .collect(),
            ))
        } else {
            BuilderKind::Bsd(ar::Builder::new(File::create(&self.config.dst).unwrap()))
        };

        // Add all files
        for (entry_name, entry) in self.entries.into_iter() {
            match entry {
                ArchiveEntry::FromArchive {
                    archive_index,
                    entry_index,
                } => {
                    let (ref src_archive_path, ref mut src_archive) =
                        self.src_archives[archive_index];
                    let entry = src_archive.jump_to_entry(entry_index).unwrap();
                    let orig_header = entry.header();

                    // FIXME implement clone for `ar::Archive`.
                    let mut header =
                        ar::Header::new(orig_header.identifier().to_vec(), orig_header.size());
                    header.set_mtime(orig_header.mtime());
                    header.set_uid(orig_header.uid());
                    header.set_gid(orig_header.gid());
                    header.set_mode(orig_header.mode());

                    match builder {
                        BuilderKind::Bsd(ref mut builder) => {
                            builder.append(&header, entry).unwrap()
                        }
                        BuilderKind::Gnu(ref mut builder) => {
                            builder.append(&header, entry).unwrap()
                        }
                        BuilderKind::NativeAr(archive_file) => {
                            Command::new("ar")
                                .arg("x")
                                .arg(src_archive_path)
                                .arg(&entry_name)
                                .status()
                                .unwrap();
                            add_file_using_ar(archive_file, Path::new(&entry_name));
                            std::fs::remove_file(entry_name).unwrap();
                        }
                    }
                }
                ArchiveEntry::File(file) => match builder {
                    BuilderKind::Bsd(ref mut builder) => builder
                        .append_file(entry_name.as_bytes(), &mut File::open(file).unwrap())
                        .unwrap(),
                    BuilderKind::Gnu(ref mut builder) => builder
                        .append_file(entry_name.as_bytes(), &mut File::open(file).unwrap())
                        .unwrap(),
                    BuilderKind::NativeAr(archive_file) => add_file_using_ar(archive_file, &file),
                },
            }
        }

        // Finalize archive
        std::mem::drop(builder);

        // Run ranlib to be able to link the archive
        let status = std::process::Command::new("ranlib")
            .arg(self.config.dst)
            .status()
            .expect("Couldn't run ranlib");
        assert!(
            status.success(),
            "Ranlib exited with code {:?}",
            status.code()
        );
    }
}

impl<'a> ArArchiveBuilder<'a> {
    fn add_archive<F>(&mut self, archive_path: PathBuf, mut skip: F) -> std::io::Result<()>
    where
        F: FnMut(&str) -> bool + 'static,
    {
        let mut archive = ar::Archive::new(std::fs::File::open(&archive_path)?);
        let archive_index = self.src_archives.len();

        let mut i = 0;
        while let Some(entry) = archive.next_entry() {
            let entry = entry.unwrap();
            let file_name = String::from_utf8(entry.header().identifier().to_vec()).unwrap();
            if !skip(&file_name) {
                self.entries.push((
                    file_name,
                    ArchiveEntry::FromArchive {
                        archive_index,
                        entry_index: i,
                    },
                ));
            }
            i += 1;
        }

        self.src_archives.push((archive_path, archive));
        Ok(())
    }
}
