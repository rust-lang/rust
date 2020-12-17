mod generated;

use hir::Semantics;
use ide_db::base_db::{fixture::WithFixture, FileId, FileRange, SourceDatabaseExt};
use ide_db::source_change::FileSystemEdit;
use ide_db::RootDatabase;
use syntax::TextRange;
use test_utils::{assert_eq_text, extract_offset, extract_range};

use crate::{handlers::Handler, Assist, AssistConfig, AssistContext, AssistKind, Assists};
use stdx::{format_to, trim_indent};

pub(crate) fn with_single_file(text: &str) -> (RootDatabase, FileId) {
    RootDatabase::with_single_file(text)
}

pub(crate) fn check_assist(assist: Handler, ra_fixture_before: &str, ra_fixture_after: &str) {
    let ra_fixture_after = trim_indent(ra_fixture_after);
    check(assist, ra_fixture_before, ExpectedResult::After(&ra_fixture_after), None);
}

// There is no way to choose what assist within a group you want to test against,
// so this is here to allow you choose.
pub(crate) fn check_assist_by_label(
    assist: Handler,
    ra_fixture_before: &str,
    ra_fixture_after: &str,
    label: &str,
) {
    let ra_fixture_after = trim_indent(ra_fixture_after);
    check(assist, ra_fixture_before, ExpectedResult::After(&ra_fixture_after), Some(label));
}

// FIXME: instead of having a separate function here, maybe use
// `extract_ranges` and mark the target as `<target> </target>` in the
// fixture?
pub(crate) fn check_assist_target(assist: Handler, ra_fixture: &str, target: &str) {
    check(assist, ra_fixture, ExpectedResult::Target(target), None);
}

pub(crate) fn check_assist_not_applicable(assist: Handler, ra_fixture: &str) {
    check(assist, ra_fixture, ExpectedResult::NotApplicable, None);
}

fn check_doc_test(assist_id: &str, before: &str, after: &str) {
    let after = trim_indent(after);
    let (db, file_id, selection) = RootDatabase::with_range_or_offset(&before);
    let before = db.file_text(file_id).to_string();
    let frange = FileRange { file_id, range: selection.into() };

    let assist = Assist::resolved(&db, &AssistConfig::default(), frange)
        .into_iter()
        .find(|assist| assist.assist.id.0 == assist_id)
        .unwrap_or_else(|| {
            panic!(
                "\n\nAssist is not applicable: {}\nAvailable assists: {}",
                assist_id,
                Assist::resolved(&db, &AssistConfig::default(), frange)
                    .into_iter()
                    .map(|assist| assist.assist.id.0)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        });

    let actual = {
        let mut actual = before;
        for source_file_edit in assist.source_change.source_file_edits {
            if source_file_edit.file_id == file_id {
                source_file_edit.edit.apply(&mut actual)
            }
        }
        actual
    };
    assert_eq_text!(&after, &actual);
}

enum ExpectedResult<'a> {
    NotApplicable,
    After(&'a str),
    Target(&'a str),
}

fn check(handler: Handler, before: &str, expected: ExpectedResult, assist_label: Option<&str>) {
    let (db, file_with_caret_id, range_or_offset) = RootDatabase::with_range_or_offset(before);
    let text_without_caret = db.file_text(file_with_caret_id).to_string();

    let frange = FileRange { file_id: file_with_caret_id, range: range_or_offset.into() };

    let sema = Semantics::new(&db);
    let config = AssistConfig::default();
    let ctx = AssistContext::new(sema, &config, frange);
    let mut acc = Assists::new_resolved(&ctx);
    handler(&mut acc, &ctx);
    let mut res = acc.finish_resolved();

    let assist = match assist_label {
        Some(label) => res.into_iter().find(|resolved| resolved.assist.label == label),
        None => res.pop(),
    };

    match (assist, expected) {
        (Some(assist), ExpectedResult::After(after)) => {
            let mut source_change = assist.source_change;
            assert!(!source_change.source_file_edits.is_empty());
            let skip_header = source_change.source_file_edits.len() == 1
                && source_change.file_system_edits.len() == 0;
            source_change.source_file_edits.sort_by_key(|it| it.file_id);

            let mut buf = String::new();
            for source_file_edit in source_change.source_file_edits {
                let mut text = db.file_text(source_file_edit.file_id).as_ref().to_owned();
                source_file_edit.edit.apply(&mut text);
                if !skip_header {
                    let sr = db.file_source_root(source_file_edit.file_id);
                    let sr = db.source_root(sr);
                    let path = sr.path_for_file(&source_file_edit.file_id).unwrap();
                    format_to!(buf, "//- {}\n", path)
                }
                buf.push_str(&text);
            }

            for file_system_edit in source_change.file_system_edits.clone() {
                match file_system_edit {
                    FileSystemEdit::CreateFile { dst, initial_contents } => {
                        let target_dst = dst.path;
                        format_to!(buf, "//- {}\n", target_dst);
                        buf.push_str(&initial_contents);
                    }
                    _ => (),
                }
            }

            assert_eq_text!(after, &buf);
        }
        (Some(assist), ExpectedResult::Target(target)) => {
            let range = assist.assist.target;
            assert_eq_text!(&text_without_caret[range], target);
        }
        (Some(_), ExpectedResult::NotApplicable) => panic!("assist should not be applicable!"),
        (None, ExpectedResult::After(_)) | (None, ExpectedResult::Target(_)) => {
            panic!("code action is not applicable")
        }
        (None, ExpectedResult::NotApplicable) => (),
    };
}

#[test]
fn assist_order_field_struct() {
    let before = "struct Foo { <|>bar: u32 }";
    let (before_cursor_pos, before) = extract_offset(before);
    let (db, file_id) = with_single_file(&before);
    let frange = FileRange { file_id, range: TextRange::empty(before_cursor_pos) };
    let assists = Assist::resolved(&db, &AssistConfig::default(), frange);
    let mut assists = assists.iter();

    assert_eq!(
        assists.next().expect("expected assist").assist.label,
        "Change visibility to pub(crate)"
    );
    assert_eq!(assists.next().expect("expected assist").assist.label, "Add `#[derive]`");
}

#[test]
fn assist_order_if_expr() {
    let before = "
    pub fn test_some_range(a: int) -> bool {
        if let 2..6 = <|>5<|> {
            true
        } else {
            false
        }
    }";
    let (range, before) = extract_range(before);
    let (db, file_id) = with_single_file(&before);
    let frange = FileRange { file_id, range };
    let assists = Assist::resolved(&db, &AssistConfig::default(), frange);
    let mut assists = assists.iter();

    assert_eq!(assists.next().expect("expected assist").assist.label, "Extract into variable");
    assert_eq!(assists.next().expect("expected assist").assist.label, "Replace with match");
}

#[test]
fn assist_filter_works() {
    let before = "
    pub fn test_some_range(a: int) -> bool {
        if let 2..6 = <|>5<|> {
            true
        } else {
            false
        }
    }";
    let (range, before) = extract_range(before);
    let (db, file_id) = with_single_file(&before);
    let frange = FileRange { file_id, range };

    {
        let mut cfg = AssistConfig::default();
        cfg.allowed = Some(vec![AssistKind::Refactor]);

        let assists = Assist::resolved(&db, &cfg, frange);
        let mut assists = assists.iter();

        assert_eq!(assists.next().expect("expected assist").assist.label, "Extract into variable");
        assert_eq!(assists.next().expect("expected assist").assist.label, "Replace with match");
    }

    {
        let mut cfg = AssistConfig::default();
        cfg.allowed = Some(vec![AssistKind::RefactorExtract]);
        let assists = Assist::resolved(&db, &cfg, frange);
        assert_eq!(assists.len(), 1);

        let mut assists = assists.iter();
        assert_eq!(assists.next().expect("expected assist").assist.label, "Extract into variable");
    }

    {
        let mut cfg = AssistConfig::default();
        cfg.allowed = Some(vec![AssistKind::QuickFix]);
        let assists = Assist::resolved(&db, &cfg, frange);
        assert!(assists.is_empty(), "All asserts but quickfixes should be filtered out");
    }
}
