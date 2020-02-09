//! FIXME: write short doc here

use ra_assists::{resolved_assists, AssistAction, AssistLabel};
use ra_db::{FilePosition, FileRange};
use ra_ide_db::RootDatabase;

use crate::{FileId, SourceChange, SourceFileEdit};

pub use ra_assists::AssistId;

#[derive(Debug)]
pub struct Assist {
    pub id: AssistId,
    pub label: String,
    pub group_label: Option<String>,
    pub source_change: SourceChange,
}

pub(crate) fn assists(db: &RootDatabase, frange: FileRange) -> Vec<Assist> {
    resolved_assists(db, frange)
        .into_iter()
        .map(|assist| {
            let file_id = frange.file_id;
            let assist_label = &assist.label;
            Assist {
                id: assist_label.id,
                label: assist_label.label.clone(),
                group_label: assist.group_label.map(|it| it.0),
                source_change: action_to_edit(assist.action, file_id, assist_label),
            }
        })
        .collect()
}

fn action_to_edit(
    action: AssistAction,
    file_id: FileId,
    assist_label: &AssistLabel,
) -> SourceChange {
    let file_edit = SourceFileEdit { file_id, edit: action.edit };
    SourceChange::source_file_edit(assist_label.label.clone(), file_edit)
        .with_cursor_opt(action.cursor_position.map(|offset| FilePosition { offset, file_id }))
}
