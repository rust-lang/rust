use std::fmt;

use ra_syntax::AstNode;
use ra_db::{
    SourceFileQuery,
    salsa::{Database, debug::DebugQueryTable},
};

use crate::db::RootDatabase;

pub(crate) fn status(db: &RootDatabase) -> String {
    let file_stats = {
        let mut stats = FilesStats::default();
        for entry in db.query(SourceFileQuery).entries::<Vec<_>>() {
            stats.total += 1;
            if let Some(value) = entry.value {
                stats.retained += 1;
                stats.retained_size = stats
                    .retained_size
                    .checked_add(value.syntax().memory_size_of_subtree())
                    .unwrap();
            }
        }
        stats
    };
    let n_defs = {
        let interner: &hir::HirInterner = db.as_ref();
        interner.len()
    };
    format!("{}\nn_defs {}\n", file_stats, n_defs)
}

#[derive(Default)]
struct FilesStats {
    total: usize,
    retained: usize,
    retained_size: usize,
}

impl fmt::Display for FilesStats {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let (size, suff) = human_bytes(self.retained_size);
        write!(
            fmt,
            "{} parsed_files, {} ({}{}) retained",
            self.total, self.retained, size, suff
        )
    }
}

fn human_bytes(bytes: usize) -> (usize, &'static str) {
    if bytes < 4096 {
        return (bytes, " bytes");
    }
    let kb = bytes / 1024;
    if kb < 4096 {
        return (kb, "kb");
    }
    let mb = kb / 1024;
    (mb, "mb")
}
