//! Utilities for creating `Analysis` instances for tests.
use test_fixture::ChangeFixture;
use test_utils::{RangeOrOffset, extract_annotations};

use crate::{Analysis, AnalysisHost, FileId, FilePosition, FileRange};

/// Creates analysis for a single file.
pub(crate) fn file(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> (Analysis, FileId) {
    let mut host = AnalysisHost::default();
    let change_fixture = ChangeFixture::parse(&host.db, ra_fixture);
    host.db.enable_proc_attr_macros();
    host.db.apply_change(change_fixture.change);
    (host.analysis(), change_fixture.files[0].file_id(&host.db))
}

/// Creates analysis from a multi-file fixture, returns positions marked with $0.
pub(crate) fn position(
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
) -> (Analysis, FilePosition) {
    let mut host = AnalysisHost::default();
    let change_fixture = ChangeFixture::parse(&host.db, ra_fixture);
    host.db.enable_proc_attr_macros();
    host.db.apply_change(change_fixture.change);
    let (file_id, range_or_offset) = change_fixture.file_position.expect("expected a marker ($0)");
    let offset = range_or_offset.expect_offset();
    (host.analysis(), FilePosition { file_id: file_id.file_id(&host.db), offset })
}

/// Creates analysis for a single file, returns range marked with a pair of $0.
pub(crate) fn range(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> (Analysis, FileRange) {
    let mut host = AnalysisHost::default();
    let change_fixture = ChangeFixture::parse(&host.db, ra_fixture);
    host.db.enable_proc_attr_macros();
    host.db.apply_change(change_fixture.change);
    let (file_id, range_or_offset) = change_fixture.file_position.expect("expected a marker ($0)");
    let range = range_or_offset.expect_range();
    (host.analysis(), FileRange { file_id: file_id.file_id(&host.db), range })
}

/// Creates analysis for a single file, returns range marked with a pair of $0 or a position marked with $0.
pub(crate) fn range_or_position(
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
) -> (Analysis, FileId, RangeOrOffset) {
    let mut host = AnalysisHost::default();
    let change_fixture = ChangeFixture::parse(&host.db, ra_fixture);
    host.db.enable_proc_attr_macros();
    host.db.apply_change(change_fixture.change);
    let (file_id, range_or_offset) = change_fixture.file_position.expect("expected a marker ($0)");
    (host.analysis(), file_id.file_id(&host.db), range_or_offset)
}

/// Creates analysis from a multi-file fixture, returns positions marked with $0.
pub(crate) fn annotations(
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
) -> (Analysis, FilePosition, Vec<(FileRange, String)>) {
    let mut host = AnalysisHost::default();
    let change_fixture = ChangeFixture::parse(&host.db, ra_fixture);
    host.db.enable_proc_attr_macros();
    host.db.apply_change(change_fixture.change);
    let (file_id, range_or_offset) = change_fixture.file_position.expect("expected a marker ($0)");
    let offset = range_or_offset.expect_offset();

    let db = &host.db;
    let annotations = change_fixture
        .files
        .iter()
        .flat_map(|&file_id| {
            let file_text = host.analysis().file_text(file_id.file_id(&host.db)).unwrap();
            let annotations = extract_annotations(&file_text);
            annotations
                .into_iter()
                .map(move |(range, data)| (FileRange { file_id: file_id.file_id(db), range }, data))
        })
        .collect();
    (host.analysis(), FilePosition { file_id: file_id.file_id(&host.db), offset }, annotations)
}

/// Creates analysis from a multi-file fixture with annotations without $0
pub(crate) fn annotations_without_marker(
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
) -> (Analysis, Vec<(FileRange, String)>) {
    let mut host = AnalysisHost::default();
    let change_fixture = ChangeFixture::parse(&host.db, ra_fixture);
    host.db.enable_proc_attr_macros();
    host.db.apply_change(change_fixture.change);

    let db = &host.db;
    let annotations = change_fixture
        .files
        .iter()
        .flat_map(|&file_id| {
            let file_text = host.analysis().file_text(file_id.file_id(db)).unwrap();
            let annotations = extract_annotations(&file_text);
            annotations
                .into_iter()
                .map(move |(range, data)| (FileRange { file_id: file_id.file_id(db), range }, data))
        })
        .collect();
    (host.analysis(), annotations)
}
