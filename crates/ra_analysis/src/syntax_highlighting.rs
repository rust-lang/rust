use ra_editor::HighlightedRange;
use ra_db::SyntaxDatabase;

use crate::{
    db::RootDatabase,
    FileId, Cancelable,
};

pub(crate) fn highlight(db: &RootDatabase, file_id: FileId) -> Cancelable<Vec<HighlightedRange>> {
    let source_file = db.source_file(file_id);
    let mut res = ra_editor::highlight(&source_file);
    for node in source_file.syntax().descendants() {}
    Ok(res)
}
