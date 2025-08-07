#![allow(clippy::print_stderr)]

mod overly_long_real_world_cases;

use ide_db::{
    LineIndexDatabase, RootDatabase,
    assists::{AssistResolveStrategy, ExprFillDefaultMode},
    base_db::SourceDatabase,
};
use itertools::Itertools;
use stdx::trim_indent;
use test_fixture::WithFixture;
use test_utils::{MiniCore, assert_eq_text, extract_annotations};

use crate::{DiagnosticsConfig, Severity};

/// Takes a multi-file input fixture with annotated cursor positions,
/// and checks that:
///  * a diagnostic is produced
///  * the first diagnostic fix trigger range touches the input cursor position
///  * that the contents of the file containing the cursor match `after` after the diagnostic fix is applied
#[track_caller]
pub(crate) fn check_fix(
    #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_after: &str,
) {
    check_nth_fix(0, ra_fixture_before, ra_fixture_after);
}
/// Takes a multi-file input fixture with annotated cursor positions,
/// and checks that:
///  * a diagnostic is produced
///  * every diagnostic fixes trigger range touches the input cursor position
///  * that the contents of the file containing the cursor match `after` after each diagnostic fix is applied
pub(crate) fn check_fixes(
    #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
    ra_fixtures_after: Vec<&str>,
) {
    for (i, ra_fixture_after) in ra_fixtures_after.iter().enumerate() {
        check_nth_fix(i, ra_fixture_before, ra_fixture_after)
    }
}

#[track_caller]
fn check_nth_fix(
    nth: usize,
    #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_after: &str,
) {
    let mut config = DiagnosticsConfig::test_sample();
    config.expr_fill_default = ExprFillDefaultMode::Default;
    check_nth_fix_with_config(config, nth, ra_fixture_before, ra_fixture_after)
}

#[track_caller]
pub(crate) fn check_fix_with_disabled(
    #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_after: &str,
    disabled: impl Iterator<Item = String>,
) {
    let mut config = DiagnosticsConfig::test_sample();
    config.expr_fill_default = ExprFillDefaultMode::Default;
    config.disabled.extend(disabled);
    check_nth_fix_with_config(config, 0, ra_fixture_before, ra_fixture_after)
}

#[track_caller]
fn check_nth_fix_with_config(
    config: DiagnosticsConfig,
    nth: usize,
    #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_after: &str,
) {
    let after = trim_indent(ra_fixture_after);

    let (db, file_position) = RootDatabase::with_position(ra_fixture_before);
    let diagnostic = super::full_diagnostics(
        &db,
        &config,
        &AssistResolveStrategy::All,
        file_position.file_id.file_id(&db),
    )
    .pop()
    .expect("no diagnostics");
    let fix = &diagnostic
        .fixes
        .unwrap_or_else(|| panic!("{:?} diagnostic misses fixes", diagnostic.code))[nth];
    let actual = {
        let source_change = fix.source_change.as_ref().unwrap();
        let file_id = *source_change.source_file_edits.keys().next().unwrap();
        let mut actual = db.file_text(file_id).text(&db).to_string();

        for (edit, snippet_edit) in source_change.source_file_edits.values() {
            edit.apply(&mut actual);
            if let Some(snippet_edit) = snippet_edit {
                snippet_edit.apply(&mut actual);
            }
        }
        actual
    };

    assert!(
        fix.target.contains_inclusive(file_position.offset),
        "diagnostic fix range {:?} does not touch cursor position {:?}",
        fix.target,
        file_position.offset
    );
    assert_eq_text!(&after, &actual);
}

pub(crate) fn check_fixes_unordered(
    #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
    ra_fixtures_after: Vec<&str>,
) {
    for ra_fixture_after in ra_fixtures_after.iter() {
        check_has_fix(ra_fixture_before, ra_fixture_after)
    }
}

#[track_caller]
pub(crate) fn check_has_fix(
    #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_after: &str,
) {
    let after = trim_indent(ra_fixture_after);

    let (db, file_position) = RootDatabase::with_position(ra_fixture_before);
    let mut conf = DiagnosticsConfig::test_sample();
    conf.expr_fill_default = ExprFillDefaultMode::Default;
    let fix = super::full_diagnostics(
        &db,
        &conf,
        &AssistResolveStrategy::All,
        file_position.file_id.file_id(&db),
    )
    .into_iter()
    .find(|d| {
        d.fixes
            .as_ref()
            .and_then(|fixes| {
                fixes.iter().find(|fix| {
                    if !fix.target.contains_inclusive(file_position.offset) {
                        return false;
                    }
                    let actual = {
                        let source_change = fix.source_change.as_ref().unwrap();
                        let file_id = *source_change.source_file_edits.keys().next().unwrap();
                        let mut actual = db.file_text(file_id).text(&db).to_string();

                        for (edit, snippet_edit) in source_change.source_file_edits.values() {
                            edit.apply(&mut actual);
                            if let Some(snippet_edit) = snippet_edit {
                                snippet_edit.apply(&mut actual);
                            }
                        }
                        actual
                    };
                    after == actual
                })
            })
            .is_some()
    });
    assert!(fix.is_some(), "no diagnostic with desired fix");
}

/// Checks that there's a diagnostic *without* fix at `$0`.
pub(crate) fn check_no_fix(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
    let (db, file_position) = RootDatabase::with_position(ra_fixture);
    let diagnostic = super::full_diagnostics(
        &db,
        &DiagnosticsConfig::test_sample(),
        &AssistResolveStrategy::All,
        file_position.file_id.file_id(&db),
    )
    .pop()
    .unwrap();
    assert!(diagnostic.fixes.is_none(), "got a fix when none was expected: {diagnostic:?}");
}

#[track_caller]
pub(crate) fn check_diagnostics(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
    let mut config = DiagnosticsConfig::test_sample();
    config.disabled.insert("inactive-code".to_owned());
    check_diagnostics_with_config(config, ra_fixture)
}

#[track_caller]
pub(crate) fn check_diagnostics_with_disabled(
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    disabled: &[&str],
) {
    let mut config = DiagnosticsConfig::test_sample();
    config.disabled.extend(disabled.iter().map(|&s| s.to_owned()));
    check_diagnostics_with_config(config, ra_fixture)
}

#[track_caller]
pub(crate) fn check_diagnostics_with_config(
    config: DiagnosticsConfig,
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
) {
    let (db, files) = RootDatabase::with_many_files(ra_fixture);
    let mut annotations = files
        .iter()
        .copied()
        .flat_map(|file_id| {
            super::full_diagnostics(&db, &config, &AssistResolveStrategy::All, file_id.file_id(&db))
                .into_iter()
                .map(|d| {
                    let mut annotation = String::new();
                    if let Some(fixes) = &d.fixes {
                        assert!(!fixes.is_empty());
                        annotation.push_str("💡 ")
                    }
                    annotation.push_str(match d.severity {
                        Severity::Error => "error",
                        Severity::WeakWarning => "weak",
                        Severity::Warning => "warn",
                        Severity::Allow => "allow",
                    });
                    annotation.push_str(": ");
                    annotation.push_str(&d.message);
                    (d.range, annotation)
                })
        })
        .map(|(diagnostic, annotation)| (diagnostic.file_id, (diagnostic.range, annotation)))
        .into_group_map();
    for file_id in files {
        let file_id = file_id.file_id(&db);
        let line_index = db.line_index(file_id);

        let mut actual = annotations.remove(&file_id).unwrap_or_default();
        let mut expected = extract_annotations(db.file_text(file_id).text(&db));
        expected.sort_by_key(|(range, s)| (range.start(), s.clone()));
        actual.sort_by_key(|(range, s)| (range.start(), s.clone()));
        // FIXME: We should panic on duplicates instead, but includes currently cause us to report
        // diagnostics twice for the calling module when both files are queried.
        actual.dedup();
        // actual.iter().duplicates().for_each(|(range, msg)| {
        //     panic!("duplicate diagnostic at {:?}: {msg:?}", line_index.line_col(range.start()))
        // });
        if expected.is_empty() {
            // makes minicore smoke test debuggable
            for (e, _) in &actual {
                eprintln!(
                    "Code in range {e:?} = {}",
                    &db.file_text(file_id).text(&db)[usize::from(e.start())..usize::from(e.end())]
                )
            }
        }
        if expected != actual {
            let fneg = expected
                .iter()
                .filter(|x| !actual.contains(x))
                .map(|(range, s)| (line_index.line_col(range.start()), range, s))
                .collect::<Vec<_>>();
            let fpos = actual
                .iter()
                .filter(|x| !expected.contains(x))
                .map(|(range, s)| (line_index.line_col(range.start()), range, s))
                .collect::<Vec<_>>();

            panic!("Diagnostic test failed.\nFalse negatives: {fneg:?}\nFalse positives: {fpos:?}");
        }
    }
}

#[test]
fn test_disabled_diagnostics() {
    let mut config = DiagnosticsConfig::test_sample();
    config.disabled.insert("E0583".into());

    let (db, file_id) = RootDatabase::with_single_file(r#"mod foo;"#);
    let file_id = file_id.file_id(&db);

    let diagnostics = super::full_diagnostics(&db, &config, &AssistResolveStrategy::All, file_id);
    assert!(diagnostics.is_empty());

    let diagnostics = super::full_diagnostics(
        &db,
        &DiagnosticsConfig::test_sample(),
        &AssistResolveStrategy::All,
        file_id,
    );
    assert!(!diagnostics.is_empty());
}

#[test]
fn minicore_smoke_test() {
    if test_utils::skip_slow_tests() {
        return;
    }

    fn check(minicore: MiniCore) {
        let source = minicore.source_code();
        let mut config = DiagnosticsConfig::test_sample();
        // This should be ignored since we conditionally remove code which creates single item use with braces
        config.disabled.insert("unused_braces".to_owned());
        config.disabled.insert("unused_variables".to_owned());
        config.disabled.insert("remove-unnecessary-else".to_owned());
        check_diagnostics_with_config(config, &source);
    }

    // Checks that there is no diagnostic in minicore for each flag.
    for flag in MiniCore::available_flags() {
        if flag == "clone" {
            // Clone without copy has `moved-out-of-ref`, so ignoring.
            // FIXME: Maybe we should merge copy and clone in a single flag?
            continue;
        }
        eprintln!("Checking minicore flag {flag}");
        check(MiniCore::from_flags([flag]));
    }
    // And one time for all flags, to check codes which are behind multiple flags + prevent name collisions
    eprintln!("Checking all minicore flags");
    check(MiniCore::from_flags(MiniCore::available_flags()))
}
