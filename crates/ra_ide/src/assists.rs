//! FIXME: write short doc here

use ra_db::{FilePosition, FileRange};

use crate::{db::RootDatabase, FileId, SourceChange, SourceFileEdit};

pub use ra_assists::AssistId;
use ra_assists::{AssistAction, AssistLabel};

#[derive(Debug)]
pub struct Assist {
    pub id: AssistId,
    pub change: SourceChange,
    pub label: String,
    pub alternative_changes: Vec<SourceChange>,
}

pub(crate) fn assists(db: &RootDatabase, frange: FileRange) -> Vec<Assist> {
    ra_assists::assists(db, frange)
        .into_iter()
        .map(|(assist_label, action, alternative_actions)| {
            let file_id = frange.file_id;
            Assist {
                id: assist_label.id,
                label: assist_label.label.clone(),
                change: action_to_edit(action, file_id, &assist_label),
                alternative_changes: alternative_actions
                    .into_iter()
                    .map(|action| action_to_edit(action, file_id, &assist_label))
                    .collect(),
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
    SourceChange::source_file_edit(
        action.label.unwrap_or_else(|| assist_label.label.clone()),
        file_edit,
    )
    .with_cursor_opt(action.cursor_position.map(|offset| FilePosition { offset, file_id }))
}
