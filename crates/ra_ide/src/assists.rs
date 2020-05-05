//! FIXME: write short doc here

use ra_assists::{resolved_assists, AssistAction};
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
            Assist {
                id: assist.label.id,
                label: assist.label.label.clone(),
                group_label: assist.label.group.map(|it| it.0),
                source_change: action_to_edit(assist.action, file_id, assist.label.label.clone()),
            }
        })
        .collect()
}

fn action_to_edit(action: AssistAction, file_id: FileId, label: String) -> SourceChange {
    let file_id = match action.file {
        ra_assists::AssistFile::TargetFile(it) => it,
        _ => file_id,
    };
    let file_edit = SourceFileEdit { file_id, edit: action.edit };
    SourceChange::source_file_edit(label, file_edit)
        .with_cursor_opt(action.cursor_position.map(|offset| FilePosition { offset, file_id }))
}
