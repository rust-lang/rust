use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

use crate::prelude::*;

pub struct ArchiveConfig<'a> {
    pub sess: &'a Session,
    pub dst: PathBuf,
    pub src: Option<PathBuf>,
    pub lib_search_paths: Vec<PathBuf>,
}

pub struct ArchiveBuilder<'a> {
    cfg: ArchiveConfig<'a>,
    src_archive: Option<ar::Archive<File>>,
    src_entries: HashMap<String, usize>,
    builder: ar::Builder<File>,
    update_symbols: bool,
}

impl<'a> ArchiveBuilder<'a> {
    pub fn new(cfg: ArchiveConfig<'a>) -> Self {
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

        ArchiveBuilder {
            cfg,
            src_archive,
            src_entries,
            builder,
            update_symbols: false,
        }
    }

    pub fn src_files(&self) -> Vec<String> {
        self.src_entries.keys().cloned().collect()
    }

    pub fn remove_file(&mut self, name: &str) {
        assert!(
            self.src_entries.remove(name).is_some(),
            "Tried to remove file not existing in src archive",
        );
    }

    pub fn update_symbols(&mut self) {
        self.update_symbols = true;
    }

    pub fn build(mut self) {
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
            .arg(self.cfg.dst)
            .status()
            .expect("Couldn't run ranlib");
        assert!(
            status.success(),
            "Ranlib exited with code {:?}",
            status.code()
        );
    }
}
