use super::*;

#[test]
fn closure_parent_strips_one_level() {
    assert_eq!(closure_parent("foo::{closure#0}"), Some("foo"));
}

#[test]
fn closure_parent_strips_nested_closures_one_level_at_a_time() {
    // merge_closures walks this repeatedly to reach the real root
    assert_eq!(closure_parent("foo::{closure#0}::{closure#1}"), Some("foo::{closure#0}"));
}

#[test]
fn closure_parent_returns_none_for_non_closures() {
    assert_eq!(closure_parent("foo::bar"), None);
}

#[test]
fn merge_monomorphizations_sums_counts_at_same_location() {
    let reports = vec![
        make_report("Vec<u8>::push", "lib.rs", 10, vec![Some(3), Some(0)]),
        make_report("Vec<String>::push", "lib.rs", 10, vec![Some(0), Some(5)]),
    ];
    let merged = merge_monomorphizations(reports);
    assert_eq!(merged.len(), 1, "same (filename, line_start) should collapse to one entry");
    assert_eq!(merged[0].line_counts, vec![Some(3), Some(5)]);
}

#[test]
fn merge_monomorphizations_keeps_different_locations_separate() {
    let reports = vec![
        make_report("foo", "lib.rs", 10, vec![Some(1)]),
        make_report("bar", "lib.rs", 20, vec![Some(1)]),
    ];
    let merged = merge_monomorphizations(reports);
    assert_eq!(merged.len(), 2, "different line_start must never merge");
}

#[test]
fn merge_closures_folds_a_closure_into_its_parent() {
    let reports = vec![
        make_report("outer", "lib.rs", 1, vec![Some(1), Some(1), Some(1)]),
        make_report("outer::{closure#0}", "lib.rs", 1, vec![Some(1), Some(1), Some(1)]),
    ];
    let merged = merge_closures(reports);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].demangled, "outer");
}

#[test]
fn merge_closures_aligns_by_real_line_number_not_vec_index() {
    // Regression test: this used to merge by summing/pushing at vec index i,
    // which only happened to work when both spans started at the same line.
    // A closure defined well inside its parent (different line_start, and a
    // much shorter span) must still land on the correct absolute lines.
    let parent =
        make_report("outer", "lib.rs", 100, vec![Some(5), Some(5), Some(5), Some(5), Some(5)]);
    let closure = make_report("outer::{closure#0}", "lib.rs", 102, vec![Some(9), Some(9)]);

    let merged = merge_closures(vec![parent, closure]);
    assert_eq!(merged.len(), 1);
    let r = &merged[0];

    // union of [100, 104] and [102, 103] is [100, 104], 5 lines
    assert_eq!(r.line_start, 100);
    assert_eq!(r.line_counts.len(), 5);
    // lines 100, 101, 103 only came from the parent
    assert_eq!(r.line_counts[0], Some(5)); // line 100
    assert_eq!(r.line_counts[1], Some(5)); // line 101
    // lines 102 and 103 were hit by both, so their counts add up
    assert_eq!(r.line_counts[2], Some(5 + 9)); // line 102
    assert_eq!(r.line_counts[3], Some(5 + 9)); // line 103
    assert_eq!(r.line_counts[4], Some(5)); // line 104, parent only
}

#[test]
fn resolve_source_path_falls_back_to_the_raw_filename() {
    // No /compiler/ part and the path is not real, so this should give
    // None rather than panic or land on some unrelated file.
    let result = resolve_source_path("/nonexistent/path/foo.rs", Path::new("/tmp"));
    assert!(result.is_none());
}
