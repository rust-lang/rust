//! Utilities for LSP-related boilerplate code.
use std::{error::Error, ops::Range, sync::Arc};

use ide_db::base_db::Cancelled;
use lsp_server::Notification;

use crate::{
    from_proto,
    global_state::GlobalState,
    line_index::{LineEndings, LineIndex, OffsetEncoding},
    LspError,
};

pub(crate) fn invalid_params_error(message: String) -> LspError {
    LspError { code: lsp_server::ErrorCode::InvalidParams as i32, message }
}

pub(crate) fn is_cancelled(e: &(dyn Error + 'static)) -> bool {
    e.downcast_ref::<Cancelled>().is_some()
}

pub(crate) fn notification_is<N: lsp_types::notification::Notification>(
    notification: &Notification,
) -> bool {
    notification.method == N::METHOD
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum Progress {
    Begin,
    Report,
    End,
}

impl Progress {
    pub(crate) fn fraction(done: usize, total: usize) -> f64 {
        assert!(done <= total);
        done as f64 / total.max(1) as f64
    }
}

impl GlobalState {
    pub(crate) fn show_message(&mut self, typ: lsp_types::MessageType, message: String) {
        let message = message;
        self.send_notification::<lsp_types::notification::ShowMessage>(
            lsp_types::ShowMessageParams { typ, message },
        )
    }

    /// rust-analyzer is resilient -- if it fails, this doesn't usually affect
    /// the user experience. Part of that is that we deliberately hide panics
    /// from the user.
    ///
    /// We do however want to pester rust-analyzer developers with panics and
    /// other "you really gotta fix that" messages. The current strategy is to
    /// be noisy for "from source" builds or when profiling is enabled.
    ///
    /// It's unclear if making from source `cargo xtask install` builds more
    /// panicky is a good idea, let's see if we can keep our awesome bleeding
    /// edge users from being upset!
    pub(crate) fn poke_rust_analyzer_developer(&mut self, message: String) {
        let from_source_build = env!("REV").contains("dev");
        let profiling_enabled = std::env::var("RA_PROFILE").is_ok();
        if from_source_build || profiling_enabled {
            self.show_message(lsp_types::MessageType::ERROR, message)
        }
    }

    pub(crate) fn report_progress(
        &mut self,
        title: &str,
        state: Progress,
        message: Option<String>,
        fraction: Option<f64>,
    ) {
        if !self.config.work_done_progress() {
            return;
        }
        let percentage = fraction.map(|f| {
            assert!((0.0..=1.0).contains(&f));
            (f * 100.0) as u32
        });
        let token = lsp_types::ProgressToken::String(format!("rustAnalyzer/{}", title));
        let work_done_progress = match state {
            Progress::Begin => {
                self.send_request::<lsp_types::request::WorkDoneProgressCreate>(
                    lsp_types::WorkDoneProgressCreateParams { token: token.clone() },
                    |_, _| (),
                );

                lsp_types::WorkDoneProgress::Begin(lsp_types::WorkDoneProgressBegin {
                    title: title.into(),
                    cancellable: None,
                    message,
                    percentage,
                })
            }
            Progress::Report => {
                lsp_types::WorkDoneProgress::Report(lsp_types::WorkDoneProgressReport {
                    cancellable: None,
                    message,
                    percentage,
                })
            }
            Progress::End => {
                lsp_types::WorkDoneProgress::End(lsp_types::WorkDoneProgressEnd { message })
            }
        };
        self.send_notification::<lsp_types::notification::Progress>(lsp_types::ProgressParams {
            token,
            value: lsp_types::ProgressParamsValue::WorkDone(work_done_progress),
        });
    }
}

pub(crate) fn apply_document_changes(
    old_text: &mut String,
    content_changes: Vec<lsp_types::TextDocumentContentChangeEvent>,
) {
    let mut line_index = LineIndex {
        index: Arc::new(ide::LineIndex::new(old_text)),
        // We don't care about line endings or offset encoding here.
        endings: LineEndings::Unix,
        encoding: OffsetEncoding::Utf16,
    };

    // The changes we got must be applied sequentially, but can cross lines so we
    // have to keep our line index updated.
    // Some clients (e.g. Code) sort the ranges in reverse. As an optimization, we
    // remember the last valid line in the index and only rebuild it if needed.
    // The VFS will normalize the end of lines to `\n`.
    enum IndexValid {
        All,
        UpToLineExclusive(u32),
    }

    impl IndexValid {
        fn covers(&self, line: u32) -> bool {
            match *self {
                IndexValid::UpToLineExclusive(to) => to > line,
                _ => true,
            }
        }
    }

    let mut index_valid = IndexValid::All;
    for change in content_changes {
        match change.range {
            Some(range) => {
                if !index_valid.covers(range.end.line) {
                    line_index.index = Arc::new(ide::LineIndex::new(old_text));
                }
                index_valid = IndexValid::UpToLineExclusive(range.start.line);
                let range = from_proto::text_range(&line_index, range);
                old_text.replace_range(Range::<usize>::from(range), &change.text);
            }
            None => {
                *old_text = change.text;
                index_valid = IndexValid::UpToLineExclusive(0);
            }
        }
    }
}

/// Checks that the edits inside the completion and the additional edits do not overlap.
/// LSP explicitly forbids the additional edits to overlap both with the main edit and themselves.
pub(crate) fn all_edits_are_disjoint(
    completion: &lsp_types::CompletionItem,
    additional_edits: &[lsp_types::TextEdit],
) -> bool {
    let mut edit_ranges = Vec::new();
    match completion.text_edit.as_ref() {
        Some(lsp_types::CompletionTextEdit::Edit(edit)) => {
            edit_ranges.push(edit.range);
        }
        Some(lsp_types::CompletionTextEdit::InsertAndReplace(edit)) => {
            let replace = edit.replace;
            let insert = edit.insert;
            if replace.start != insert.start
                || insert.start > insert.end
                || insert.end > replace.end
            {
                // insert has to be a prefix of replace but it is not
                return false;
            }
            edit_ranges.push(replace);
        }
        None => {}
    }
    if let Some(additional_changes) = completion.additional_text_edits.as_ref() {
        edit_ranges.extend(additional_changes.iter().map(|edit| edit.range));
    };
    edit_ranges.extend(additional_edits.iter().map(|edit| edit.range));
    edit_ranges.sort_by_key(|range| (range.start, range.end));
    edit_ranges
        .iter()
        .zip(edit_ranges.iter().skip(1))
        .all(|(previous, next)| previous.end <= next.start)
}

#[cfg(test)]
mod tests {
    use lsp_types::{
        CompletionItem, CompletionTextEdit, InsertReplaceEdit, Position, Range,
        TextDocumentContentChangeEvent,
    };

    use super::*;

    #[test]
    fn test_apply_document_changes() {
        macro_rules! c {
            [$($sl:expr, $sc:expr; $el:expr, $ec:expr => $text:expr),+] => {
                vec![$(TextDocumentContentChangeEvent {
                    range: Some(Range {
                        start: Position { line: $sl, character: $sc },
                        end: Position { line: $el, character: $ec },
                    }),
                    range_length: None,
                    text: String::from($text),
                }),+]
            };
        }

        let mut text = String::new();
        apply_document_changes(&mut text, vec![]);
        assert_eq!(text, "");
        apply_document_changes(
            &mut text,
            vec![TextDocumentContentChangeEvent {
                range: None,
                range_length: None,
                text: String::from("the"),
            }],
        );
        assert_eq!(text, "the");
        apply_document_changes(&mut text, c![0, 3; 0, 3 => " quick"]);
        assert_eq!(text, "the quick");
        apply_document_changes(&mut text, c![0, 0; 0, 4 => "", 0, 5; 0, 5 => " foxes"]);
        assert_eq!(text, "quick foxes");
        apply_document_changes(&mut text, c![0, 11; 0, 11 => "\ndream"]);
        assert_eq!(text, "quick foxes\ndream");
        apply_document_changes(&mut text, c![1, 0; 1, 0 => "have "]);
        assert_eq!(text, "quick foxes\nhave dream");
        apply_document_changes(
            &mut text,
            c![0, 0; 0, 0 => "the ", 1, 4; 1, 4 => " quiet", 1, 16; 1, 16 => "s\n"],
        );
        assert_eq!(text, "the quick foxes\nhave quiet dreams\n");
        apply_document_changes(&mut text, c![0, 15; 0, 15 => "\n", 2, 17; 2, 17 => "\n"]);
        assert_eq!(text, "the quick foxes\n\nhave quiet dreams\n\n");
        apply_document_changes(
            &mut text,
            c![1, 0; 1, 0 => "DREAM", 2, 0; 2, 0 => "they ", 3, 0; 3, 0 => "DON'T THEY?"],
        );
        assert_eq!(text, "the quick foxes\nDREAM\nthey have quiet dreams\nDON'T THEY?\n");
        apply_document_changes(&mut text, c![0, 10; 1, 5 => "", 2, 0; 2, 12 => ""]);
        assert_eq!(text, "the quick \nthey have quiet dreams\n");

        text = String::from("❤️");
        apply_document_changes(&mut text, c![0, 0; 0, 0 => "a"]);
        assert_eq!(text, "a❤️");

        text = String::from("a\nb");
        apply_document_changes(&mut text, c![0, 1; 1, 0 => "\nțc", 0, 1; 1, 1 => "d"]);
        assert_eq!(text, "adcb");

        text = String::from("a\nb");
        apply_document_changes(&mut text, c![0, 1; 1, 0 => "ț\nc", 0, 2; 0, 2 => "c"]);
        assert_eq!(text, "ațc\ncb");
    }

    #[test]
    fn empty_completion_disjoint_tests() {
        let empty_completion =
            CompletionItem::new_simple("label".to_string(), "detail".to_string());

        let disjoint_edit_1 = lsp_types::TextEdit::new(
            Range::new(Position::new(2, 2), Position::new(3, 3)),
            "new_text".to_string(),
        );
        let disjoint_edit_2 = lsp_types::TextEdit::new(
            Range::new(Position::new(3, 3), Position::new(4, 4)),
            "new_text".to_string(),
        );

        let joint_edit = lsp_types::TextEdit::new(
            Range::new(Position::new(1, 1), Position::new(5, 5)),
            "new_text".to_string(),
        );

        assert!(
            all_edits_are_disjoint(&empty_completion, &[]),
            "Empty completion has all its edits disjoint"
        );
        assert!(
            all_edits_are_disjoint(
                &empty_completion,
                &[disjoint_edit_1.clone(), disjoint_edit_2.clone()]
            ),
            "Empty completion is disjoint to whatever disjoint extra edits added"
        );

        assert!(
            !all_edits_are_disjoint(
                &empty_completion,
                &[disjoint_edit_1, disjoint_edit_2, joint_edit]
            ),
            "Empty completion does not prevent joint extra edits from failing the validation"
        );
    }

    #[test]
    fn completion_with_joint_edits_disjoint_tests() {
        let disjoint_edit = lsp_types::TextEdit::new(
            Range::new(Position::new(1, 1), Position::new(2, 2)),
            "new_text".to_string(),
        );
        let disjoint_edit_2 = lsp_types::TextEdit::new(
            Range::new(Position::new(2, 2), Position::new(3, 3)),
            "new_text".to_string(),
        );
        let joint_edit = lsp_types::TextEdit::new(
            Range::new(Position::new(1, 1), Position::new(5, 5)),
            "new_text".to_string(),
        );

        let mut completion_with_joint_edits =
            CompletionItem::new_simple("label".to_string(), "detail".to_string());
        completion_with_joint_edits.additional_text_edits =
            Some(vec![disjoint_edit.clone(), joint_edit.clone()]);
        assert!(
            !all_edits_are_disjoint(&completion_with_joint_edits, &[]),
            "Completion with disjoint edits fails the validation even with empty extra edits"
        );

        completion_with_joint_edits.text_edit =
            Some(CompletionTextEdit::Edit(disjoint_edit.clone()));
        completion_with_joint_edits.additional_text_edits = Some(vec![joint_edit.clone()]);
        assert!(
            !all_edits_are_disjoint(&completion_with_joint_edits, &[]),
            "Completion with disjoint edits fails the validation even with empty extra edits"
        );

        completion_with_joint_edits.text_edit =
            Some(CompletionTextEdit::InsertAndReplace(InsertReplaceEdit {
                new_text: "new_text".to_string(),
                insert: disjoint_edit.range,
                replace: disjoint_edit_2.range,
            }));
        completion_with_joint_edits.additional_text_edits = Some(vec![joint_edit]);
        assert!(
            !all_edits_are_disjoint(&completion_with_joint_edits, &[]),
            "Completion with disjoint edits fails the validation even with empty extra edits"
        );
    }

    #[test]
    fn completion_with_disjoint_edits_disjoint_tests() {
        let disjoint_edit = lsp_types::TextEdit::new(
            Range::new(Position::new(1, 1), Position::new(2, 2)),
            "new_text".to_string(),
        );
        let disjoint_edit_2 = lsp_types::TextEdit::new(
            Range::new(Position::new(2, 2), Position::new(3, 3)),
            "new_text".to_string(),
        );
        let joint_edit = lsp_types::TextEdit::new(
            Range::new(Position::new(1, 1), Position::new(5, 5)),
            "new_text".to_string(),
        );

        let mut completion_with_disjoint_edits =
            CompletionItem::new_simple("label".to_string(), "detail".to_string());
        completion_with_disjoint_edits.text_edit = Some(CompletionTextEdit::Edit(disjoint_edit));
        let completion_with_disjoint_edits = completion_with_disjoint_edits;

        assert!(
            all_edits_are_disjoint(&completion_with_disjoint_edits, &[]),
            "Completion with disjoint edits is valid"
        );
        assert!(
            !all_edits_are_disjoint(&completion_with_disjoint_edits, &[joint_edit]),
            "Completion with disjoint edits and joint extra edit is invalid"
        );
        assert!(
            all_edits_are_disjoint(&completion_with_disjoint_edits, &[disjoint_edit_2]),
            "Completion with disjoint edits and joint extra edit is valid"
        );
    }
}
