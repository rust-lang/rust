#[cfg(not(feature = "in-rust-tree"))]
mod sourcegen;

use expect_test::Expect;
use ide_db::{
    assists::AssistResolveStrategy,
    base_db::{fixture::WithFixture, SourceDatabaseExt},
    RootDatabase,
};
use stdx::trim_indent;
use test_utils::{assert_eq_text, extract_annotations, MiniCore};

use crate::{DiagnosticsConfig, ExprFillDefaultMode, Severity};

/// Takes a multi-file input fixture with annotated cursor positions,
/// and checks that:
///  * a diagnostic is produced
///  * the first diagnostic fix trigger range touches the input cursor position
///  * that the contents of the file containing the cursor match `after` after the diagnostic fix is applied
#[track_caller]
pub(crate) fn check_fix(ra_fixture_before: &str, ra_fixture_after: &str) {
    check_nth_fix(0, ra_fixture_before, ra_fixture_after);
}
/// Takes a multi-file input fixture with annotated cursor positions,
/// and checks that:
///  * a diagnostic is produced
///  * every diagnostic fixes trigger range touches the input cursor position
///  * that the contents of the file containing the cursor match `after` after each diagnostic fix is applied
pub(crate) fn check_fixes(ra_fixture_before: &str, ra_fixtures_after: Vec<&str>) {
    for (i, ra_fixture_after) in ra_fixtures_after.iter().enumerate() {
        check_nth_fix(i, ra_fixture_before, ra_fixture_after)
    }
}

#[track_caller]
fn check_nth_fix(nth: usize, ra_fixture_before: &str, ra_fixture_after: &str) {
    let after = trim_indent(ra_fixture_after);

    let (db, file_position) = RootDatabase::with_position(ra_fixture_before);
    let mut conf = DiagnosticsConfig::test_sample();
    conf.expr_fill_default = ExprFillDefaultMode::Default;
    let diagnostic =
        super::diagnostics(&db, &conf, &AssistResolveStrategy::All, file_position.file_id)
            .pop()
            .expect("no diagnostics");
    let fix = &diagnostic.fixes.expect("diagnostic misses fixes")[nth];
    let actual = {
        let source_change = fix.source_change.as_ref().unwrap();
        let file_id = *source_change.source_file_edits.keys().next().unwrap();
        let mut actual = db.file_text(file_id).to_string();

        for edit in source_change.source_file_edits.values() {
            edit.apply(&mut actual);
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

/// Checks that there's a diagnostic *without* fix at `$0`.
pub(crate) fn check_no_fix(ra_fixture: &str) {
    let (db, file_position) = RootDatabase::with_position(ra_fixture);
    let diagnostic = super::diagnostics(
        &db,
        &DiagnosticsConfig::test_sample(),
        &AssistResolveStrategy::All,
        file_position.file_id,
    )
    .pop()
    .unwrap();
    assert!(diagnostic.fixes.is_none(), "got a fix when none was expected: {diagnostic:?}");
}

pub(crate) fn check_expect(ra_fixture: &str, expect: Expect) {
    let (db, file_id) = RootDatabase::with_single_file(ra_fixture);
    let diagnostics = super::diagnostics(
        &db,
        &DiagnosticsConfig::test_sample(),
        &AssistResolveStrategy::All,
        file_id,
    );
    expect.assert_debug_eq(&diagnostics)
}

#[track_caller]
pub(crate) fn check_diagnostics(ra_fixture: &str) {
    let mut config = DiagnosticsConfig::test_sample();
    config.disabled.insert("inactive-code".to_string());
    check_diagnostics_with_config(config, ra_fixture)
}

#[track_caller]
pub(crate) fn check_diagnostics_with_config(config: DiagnosticsConfig, ra_fixture: &str) {
    let (db, files) = RootDatabase::with_many_files(ra_fixture);
    for file_id in files {
        let diagnostics = super::diagnostics(&db, &config, &AssistResolveStrategy::All, file_id);

        let expected = extract_annotations(&db.file_text(file_id));
        let mut actual = diagnostics
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
                });
                annotation.push_str(": ");
                annotation.push_str(&d.message);
                (d.range, annotation)
            })
            .collect::<Vec<_>>();
        actual.sort_by_key(|(range, _)| range.start());
        if expected.is_empty() {
            // makes minicore smoke test debugable
            for (e, _) in &actual {
                eprintln!(
                    "Code in range {e:?} = {}",
                    &db.file_text(file_id)[usize::from(e.start())..usize::from(e.end())]
                )
            }
        }
        assert_eq!(expected, actual);
    }
}

#[test]
fn test_disabled_diagnostics() {
    let mut config = DiagnosticsConfig::test_sample();
    config.disabled.insert("unresolved-module".into());

    let (db, file_id) = RootDatabase::with_single_file(r#"mod foo;"#);

    let diagnostics = super::diagnostics(&db, &config, &AssistResolveStrategy::All, file_id);
    assert!(diagnostics.is_empty());

    let diagnostics = super::diagnostics(
        &db,
        &DiagnosticsConfig::test_sample(),
        &AssistResolveStrategy::All,
        file_id,
    );
    assert!(!diagnostics.is_empty());
}

#[test]
fn minicore_smoke_test() {
    fn check(minicore: MiniCore) {
        let source = minicore.source_code();
        let mut config = DiagnosticsConfig::test_sample();
        // This should be ignored since we conditionaly remove code which creates single item use with braces
        config.disabled.insert("unnecessary-braces".to_string());
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
