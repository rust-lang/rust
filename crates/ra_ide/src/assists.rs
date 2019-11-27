//! FIXME: write short doc here

use ra_db::{FilePosition, FileRange};

use crate::{db::RootDatabase, SourceChange, SourceFileEdit};

pub use ra_assists::AssistId;

#[derive(Debug)]
pub struct Assist {
    pub id: AssistId,
    pub change: SourceChange,
}

pub(crate) fn assists(db: &RootDatabase, frange: FileRange) -> Vec<Assist> {
    ra_assists::assists(db, frange)
        .into_iter()
        .map(|(label, action)| {
            let file_id = frange.file_id;
            let file_edit = SourceFileEdit { file_id, edit: action.edit };
            let id = label.id;
            let change = SourceChange::source_file_edit(label.label, file_edit).with_cursor_opt(
                action.cursor_position.map(|offset| FilePosition { offset, file_id }),
            );
            Assist { id, change }
        })
        .collect()
}
