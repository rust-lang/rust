use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::prelude::*;

use rustc_codegen_ssa::{METADATA_FILENAME, RLIB_BYTECODE_EXTENSION};
use rustc_codegen_ssa::back::archive::{ArchiveBuilder, find_library};

struct ArchiveConfig<'a> {
    pub sess: &'a Session,
    pub dst: PathBuf,
    pub src: Option<PathBuf>,
    pub lib_search_paths: Vec<PathBuf>,
}

pub struct ArArchiveBuilder<'a> {
    config: ArchiveConfig<'a>,
    src_archive: Option<ar::Archive<File>>,
    src_entries: HashMap<String, usize>,
    builder: ar::Builder<File>,
    update_symbols: bool,
}

impl<'a> ArchiveBuilder<'a> for ArArchiveBuilder<'a> {
    fn new(sess: &'a Session, output: &Path, input: Option<&Path>) -> Self {
        use rustc_codegen_ssa::back::link::archive_search_paths;
        let cfg = ArchiveConfig {
            sess,
            dst: output.to_path_buf(),
            src: input.map(|p| p.to_path_buf()),
            lib_search_paths: archive_search_paths(sess),
        };

        let (src_archive, src_entries) = if let Some(src) = &cfg.src {
            let mut archive = ar::Archive::new(File::open(src).unwrap());
            let mut entries = HashMap::new();

            let mut i = 0;
            while let Some(entry) = archive.next_entry() {
                let entry = entry.unwrap();
                entries.insert(
                    String::from_utf8(entry.header().identifier().to_vec()).unwrap(),
                    i,
                );
                i += 1;
            }

            (Some(archive), entries)
        } else {
            (None, HashMap::new())
        };

        let builder = ar::Builder::new(File::create(&cfg.dst).unwrap());

        ArArchiveBuilder {
            config: cfg,
            src_archive,
            src_entries,
            builder,
            update_symbols: false,
        }
    }

    fn src_files(&mut self) -> Vec<String> {
        self.src_entries.keys().cloned().collect()
    }

    fn remove_file(&mut self, name: &str) {
        let file = self.src_entries.remove(name);
        assert!(
            file.is_some(),
            "Tried to remove file not existing in src archive",
        );
    }

    fn add_file(&mut self, file: &Path) {
        self.builder.append_path(file).unwrap();
    }

    fn add_native_library(&mut self, name: &str) {
        let location = find_library(name, &self.config.lib_search_paths, self.config.sess);
        self.add_archive(&location, |_| false).unwrap_or_else(|e| {
            panic!("failed to add native library {}: {}", location.to_string_lossy(), e);
        });
    }

    fn add_rlib(&mut self, rlib: &Path, name: &str, lto: bool, skip_objects: bool) -> std::io::Result<()> {
        let obj_start = name.to_owned();

        self.add_archive(rlib, move |fname: &str| {
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
        // Add files from original archive
        if let Some(mut src_archive) = self.src_archive {
            for (_entry_name, entry_idx) in self.src_entries.into_iter() {
                let entry = src_archive.jump_to_entry(entry_idx).unwrap();
                let orig_header = entry.header();
                let mut header =
                    ar::Header::new(orig_header.identifier().to_vec(), orig_header.size());
                header.set_mtime(orig_header.mtime());
                header.set_uid(orig_header.uid());
                header.set_gid(orig_header.gid());
                header.set_mode(orig_header.mode());
                self.builder.append(&header, entry).unwrap();
            }
        }

        // Finalize archive
        std::mem::drop(self.builder);

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
    fn add_archive<F>(&mut self, archive: &Path, mut skip: F) -> std::io::Result<()>
        where F: FnMut(&str) -> bool + 'static
    {
        let mut archive = ar::Archive::new(std::fs::File::open(archive)?);
        while let Some(entry) = archive.next_entry() {
            let entry = entry?;
            let orig_header = entry.header();

            if skip(std::str::from_utf8(orig_header.identifier()).unwrap()) {
                continue;
            }

            let mut header =
                ar::Header::new(orig_header.identifier().to_vec(), orig_header.size());
            header.set_mtime(orig_header.mtime());
            header.set_uid(orig_header.uid());
            header.set_gid(orig_header.gid());
            header.set_mode(orig_header.mode());
            self.builder.append(&header, entry).unwrap();
        }
        Ok(())
    }
}
