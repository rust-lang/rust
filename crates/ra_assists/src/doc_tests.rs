//! Each assist definition has a special comment, which specifies docs and
//! example.
//!
//! We collect all the example and write the as tests in this module.

mod generated;

use ra_db::{fixture::WithFixture, FileRange};
use test_utils::{assert_eq_text, extract_range_or_offset};

use ra_ide_db::RootDatabase;

fn check(assist_id: &str, before: &str, after: &str) {
    // FIXME we cannot get the imports search functionality here yet, but still need to generate a test and a doc for an assist
    if assist_id == "auto_import" {
        return;
    }
    let (selection, before) = extract_range_or_offset(before);
    let (db, file_id) = RootDatabase::with_single_file(&before);
    let frange = FileRange { file_id, range: selection.into() };

    let assist = crate::assists(&db, frange)
        .into_iter()
        .find(|assist| assist.label.id.0 == assist_id)
        .unwrap_or_else(|| {
            panic!(
                "\n\nAssist is not applicable: {}\nAvailable assists: {}",
                assist_id,
                crate::assists(&db, frange)
                    .into_iter()
                    .map(|assist| assist.label.id.0)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        });

    let actual = assist.get_first_action().edit.apply(&before);
    assert_eq_text!(after, &actual);
}
