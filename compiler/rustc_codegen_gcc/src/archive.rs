use std::fs::File;
use std::path::{Path, PathBuf};

use crate::errors::RanlibFailure;

use rustc_codegen_ssa::back::archive::{ArchiveBuilder, ArchiveBuilderBuilder};
use rustc_session::Session;

use rustc_session::cstore::DllImport;

struct ArchiveConfig<'a> {
    sess: &'a Session,
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

pub struct ArArchiveBuilderBuilder;

impl ArchiveBuilderBuilder for ArArchiveBuilderBuilder {
    fn new_archive_builder<'a>(&self, sess: &'a Session) -> Box<dyn ArchiveBuilder<'a> + 'a> {
        let config = ArchiveConfig {
            sess,
            use_native_ar: false,
            // FIXME test for linux and System V derivatives instead
            use_gnu_style_archive: sess.target.options.archive_format == "gnu",
        };

        Box::new(ArArchiveBuilder {
            config,
            src_archives: vec![],
            entries: vec![],
        })
    }

    fn create_dll_import_lib(
        &self,
        _sess: &Session,
        _lib_name: &str,
        _dll_imports: &[DllImport],
        _tmpdir: &Path,
    ) -> PathBuf {
        unimplemented!();
    }
}

pub struct ArArchiveBuilder<'a> {
    config: ArchiveConfig<'a>,
    src_archives: Vec<(PathBuf, ar::Archive<File>)>,
    // Don't use `HashMap` here, as the order is important. `rust.metadata.bin` must always be at
    // the end of an archive for linkers to not get confused.
    entries: Vec<(String, ArchiveEntry)>,
}

impl<'a> ArchiveBuilder<'a> for ArArchiveBuilder<'a> {
    fn add_file(&mut self, file: &Path) {
        self.entries.push((
            file.file_name().unwrap().to_str().unwrap().to_string(),
            ArchiveEntry::File(file.to_owned()),
        ));
    }

    fn add_archive(
        &mut self,
        archive_path: &Path,
        mut skip: Box<dyn FnMut(&str) -> bool + 'static>,
    ) -> std::io::Result<()> {
        let mut archive = ar::Archive::new(std::fs::File::open(&archive_path)?);
        let archive_index = self.src_archives.len();

        let mut i = 0;
        while let Some(entry) = archive.next_entry() {
            let entry = entry?;
            let file_name = String::from_utf8(entry.header().identifier().to_vec())
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
            if !skip(&file_name) {
                self.entries
                    .push((file_name, ArchiveEntry::FromArchive { archive_index, entry_index: i }));
            }
            i += 1;
        }

        self.src_archives.push((archive_path.to_owned(), archive));
        Ok(())
    }

    fn build(mut self: Box<Self>, output: &Path) -> bool {
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
            BuilderKind::NativeAr(output)
        } else if self.config.use_gnu_style_archive {
            BuilderKind::Gnu(ar::GnuBuilder::new(
                File::create(output).unwrap(),
                self.entries
                    .iter()
                    .map(|(name, _)| name.as_bytes().to_vec())
                    .collect(),
            ))
        } else {
            BuilderKind::Bsd(ar::Builder::new(File::create(output).unwrap()))
        };

        let any_members = !self.entries.is_empty();

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
                    let header = entry.header().clone();

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
                ArchiveEntry::File(file) =>
                    match builder {
                        BuilderKind::Bsd(ref mut builder) => {
                            builder
                                .append_file(entry_name.as_bytes(), &mut File::open(file).expect("file for bsd builder"))
                                .unwrap()
                        },
                        BuilderKind::Gnu(ref mut builder) => {
                            builder
                                .append_file(entry_name.as_bytes(), &mut File::open(&file).expect(&format!("file {:?} for gnu builder", file)))
                                .unwrap()
                        },
                        BuilderKind::NativeAr(archive_file) => add_file_using_ar(archive_file, &file),
                    },
            }
        }

        // Finalize archive
        std::mem::drop(builder);

        // Run ranlib to be able to link the archive
        let status =
            std::process::Command::new("ranlib").arg(output).status().expect("Couldn't run ranlib");

        if !status.success() {
            self.config.sess.emit_fatal(RanlibFailure::new(status.code()));
        }

        any_members
    }
}
