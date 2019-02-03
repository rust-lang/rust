use ra_db::{FileRange, FilePosition};

use crate::{SourceFileEdit, SourceChange, db::RootDatabase};

pub(crate) fn assists(db: &RootDatabase, frange: FileRange) -> Vec<SourceChange> {
    ra_assists::assists(db, frange)
        .into_iter()
        .map(|(label, action)| {
            let file_id = frange.file_id;
            let file_edit = SourceFileEdit {
                file_id,
                edit: action.edit,
            };
            SourceChange {
                label: label.label,
                source_file_edits: vec![file_edit],
                file_system_edits: vec![],
                cursor_position: action
                    .cursor_position
                    .map(|offset| FilePosition { offset, file_id }),
            }
        })
        .collect()
}
