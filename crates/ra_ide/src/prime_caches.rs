//! rust-analyzer is lazy and doesn't not compute anything unless asked. This
//! sometimes is counter productive when, for example, the first goto definition
//! request takes longer to compute. This modules implemented prepopulating of
//! various caches, it's not really advanced at the moment.

use crate::{FileId, RootDatabase};

pub(crate) fn prime_caches(db: &RootDatabase, files: Vec<FileId>) {
    for file in files {
        let _ = crate::syntax_highlighting::highlight(db, file, None, false);
    }
}
