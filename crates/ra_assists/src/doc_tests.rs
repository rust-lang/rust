//! Each assist definition has a special comment, which specifies docs and
//! example.
//!
//! We collect all the example and write the as tests in this module.

mod generated;

use hir::mock::MockDatabase;
use ra_db::FileRange;
use ra_syntax::TextRange;
use test_utils::{assert_eq_text, extract_offset};

fn check(assist_id: &str, before: &str, after: &str) {
    let (before_cursor_pos, before) = extract_offset(before);
    let (db, _source_root, file_id) = MockDatabase::with_single_file(&before);
    let frange = FileRange { file_id, range: TextRange::offset_len(before_cursor_pos, 0.into()) };

    let (_assist_id, action) =
        crate::assists(&db, frange).into_iter().find(|(id, _)| id.id.0 == assist_id).unwrap();

    let actual = action.edit.apply(&before);
    assert_eq_text!(after, &actual);
}
