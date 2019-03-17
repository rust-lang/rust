use ra_db::SourceDatabase;
use ra_syntax::AstNode;

use crate::{
    TextRange, FileRange,
    db::RootDatabase,
};

// FIXME: restore macro support
pub(crate) fn extend_selection(db: &RootDatabase, frange: FileRange) -> TextRange {
    let source_file = db.parse(frange.file_id);
    ra_ide_api_light::extend_selection(source_file.syntax(), frange.range).unwrap_or(frange.range)
}
