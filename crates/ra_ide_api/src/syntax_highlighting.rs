use ra_syntax::AstNode;
use ra_db::SourceDatabase;

use crate::{
    FileId, HighlightedRange,
    db::RootDatabase,
};

pub(crate) fn highlight(db: &RootDatabase, file_id: FileId) -> Vec<HighlightedRange> {
    let source_file = db.parse(file_id);
    ra_ide_api_light::highlight(source_file.syntax())
}
