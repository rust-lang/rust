use ra_editor::HighlightedRange;
use ra_db::SyntaxDatabase;

use crate::{
    db::RootDatabase,
    FileId, Cancelable,
};

pub(crate) fn highlight(db: &RootDatabase, file_id: FileId) -> Cancelable<Vec<HighlightedRange>> {
    let file = db.source_file(file_id);
    Ok(ra_editor::highlight(&file))
}
