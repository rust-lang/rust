use ra_db::SyntaxDatabase;

use crate::{
    TextRange, FileRange,
    db::RootDatabase,
};

pub(crate) fn extend_selection(db: &RootDatabase, frange: FileRange) -> TextRange {
    let file = db.source_file(frange.file_id);
    ra_editor::extend_selection(&file, frange.range).unwrap_or(frange.range)
}
