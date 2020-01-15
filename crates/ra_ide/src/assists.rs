//! FIXME: write short doc here

use ra_db::{FilePosition, FileRange};

use crate::{db::RootDatabase, FileId, SourceChange, SourceFileEdit};

use either::Either;
pub use ra_assists::AssistId;
use ra_assists::{AssistAction, AssistLabel};

#[derive(Debug)]
pub struct Assist {
    pub id: AssistId,
    pub label: String,
    pub change_data: Either<SourceChange, Vec<SourceChange>>,
}

pub(crate) fn assists(db: &RootDatabase, frange: FileRange) -> Vec<Assist> {
    ra_assists::assists(db, frange)
        .into_iter()
        .map(|assist| {
            let file_id = frange.file_id;
            let assist_label = &assist.label;
            Assist {
                id: assist_label.id,
                label: assist_label.label.clone(),
                change_data: match assist.action_data {
                    Either::Left(action) => {
                        Either::Left(action_to_edit(action, file_id, assist_label))
                    }
                    Either::Right(actions) => Either::Right(
                        actions
                            .into_iter()
                            .map(|action| action_to_edit(action, file_id, assist_label))
                            .collect(),
                    ),
                },
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
