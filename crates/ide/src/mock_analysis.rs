//! FIXME: write short doc here

use base_db::fixture::ChangeFixture;
use test_utils::{extract_annotations, RangeOrOffset};

use crate::{Analysis, AnalysisHost, FileId, FilePosition, FileRange};

/// Creates analysis from a multi-file fixture, returns positions marked with <|>.
pub(crate) fn analysis_and_position(ra_fixture: &str) -> (Analysis, FilePosition) {
    let mut host = AnalysisHost::default();
    let change_fixture = ChangeFixture::parse(ra_fixture);
    host.db.apply_change(change_fixture.change);
    let (file_id, range_or_offset) = change_fixture.file_position.expect("expected a marker (<|>)");
    let offset = match range_or_offset {
        RangeOrOffset::Range(_) => panic!(),
        RangeOrOffset::Offset(it) => it,
    };
    (host.analysis(), FilePosition { file_id, offset })
}

/// Creates analysis for a single file.
pub(crate) fn single_file(ra_fixture: &str) -> (Analysis, FileId) {
    let mut host = AnalysisHost::default();
    let change_fixture = ChangeFixture::parse(ra_fixture);
    host.db.apply_change(change_fixture.change);
    (host.analysis(), change_fixture.files[0])
}

/// Creates analysis for a single file.
pub(crate) fn many_files(ra_fixture: &str) -> (Analysis, Vec<FileId>) {
    let mut host = AnalysisHost::default();
    let change_fixture = ChangeFixture::parse(ra_fixture);
    host.db.apply_change(change_fixture.change);
    (host.analysis(), change_fixture.files)
}

/// Creates analysis for a single file, returns range marked with a pair of <|>.
pub(crate) fn analysis_and_range(ra_fixture: &str) -> (Analysis, FileRange) {
    let mut host = AnalysisHost::default();
    let change_fixture = ChangeFixture::parse(ra_fixture);
    host.db.apply_change(change_fixture.change);
    let (file_id, range_or_offset) = change_fixture.file_position.expect("expected a marker (<|>)");
    let range = match range_or_offset {
        RangeOrOffset::Range(it) => it,
        RangeOrOffset::Offset(_) => panic!(),
    };
    (host.analysis(), FileRange { file_id, range })
}

/// Creates analysis from a multi-file fixture, returns positions marked with <|>.
pub(crate) fn analysis_and_annotations(
    ra_fixture: &str,
) -> (Analysis, FilePosition, Vec<(FileRange, String)>) {
    let mut host = AnalysisHost::default();
    let change_fixture = ChangeFixture::parse(ra_fixture);
    host.db.apply_change(change_fixture.change);
    let (file_id, range_or_offset) = change_fixture.file_position.expect("expected a marker (<|>)");
    let offset = match range_or_offset {
        RangeOrOffset::Range(_) => panic!(),
        RangeOrOffset::Offset(it) => it,
    };

    let annotations = change_fixture
        .files
        .iter()
        .flat_map(|&file_id| {
            let file_text = host.analysis().file_text(file_id).unwrap();
            let annotations = extract_annotations(&file_text);
            annotations.into_iter().map(move |(range, data)| (FileRange { file_id, range }, data))
        })
        .collect();
    (host.analysis(), FilePosition { file_id, offset }, annotations)
}
