use super::*;
use crate::transform::FunctionReport;

#[cfg(test)]
fn make_report(
    id: usize,
    demangled: &str,
    filename: &str,
    line_start: usize,
    line_counts: Vec<Option<u64>>,
    category: FunctionCategory,
) -> FunctionReport {
    FunctionReport {
        id,
        demangled: demangled.to_string(),
        filename: filename.to_string(),
        line_start,
        line_counts,
        category,
    }
}

#[test]
fn crate_name_extracts_the_crate_directory() {
    assert_eq!(crate_name("/home/user/rust/compiler/rustc_abi/src/lib.rs"), "rustc_abi");
}

#[test]
fn crate_name_is_unknown_outside_the_compiler_tree() {
    assert_eq!(crate_name("/home/user/somewhere/else.rs"), "unknown");
}

#[test]
fn file_path_in_crate_strips_the_crate_name_prefix() {
    assert_eq!(file_path_in_crate("/home/user/rust/compiler/rustc_abi/src/lib.rs"), "src/lib.rs");
}

#[test]
fn report_paths_produces_four_distinct_filenames() {
    let paths = report_paths("report");
    assert_eq!(paths.index, "report.html");
    assert_eq!(paths.fully_covered, "report_fully-covered.html");
    assert_eq!(paths.partially_covered, "report_partially-covered.html");
    assert_eq!(paths.uncovered, "report_uncovered.html");
    // all four must be distinct, or two categories would overwrite each other
    let all = [&paths.index, &paths.fully_covered, &paths.partially_covered, &paths.uncovered];
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            assert_ne!(a, b);
        }
    }
}

#[test]
fn report_paths_for_category_matches_the_named_fields() {
    let paths = report_paths("report");
    assert_eq!(paths.for_category(FunctionCategory::FullyCovered), paths.fully_covered);
    assert_eq!(paths.for_category(FunctionCategory::PartiallyCovered), paths.partially_covered);
    assert_eq!(paths.for_category(FunctionCategory::FullyUncovered), paths.uncovered);
}

#[test]
fn index_page_links_to_the_exact_filenames_report_paths_returns() {
    // Catches render_index hardcoding a name instead of asking ReportPaths.
    let paths = report_paths("report");
    let html = render_index(1, 2, 3, 100, 200, &paths).unwrap();
    assert!(html.contains(&paths.uncovered), "index must link to the uncovered page");
    assert!(html.contains(&paths.partially_covered), "index must link to the partial page");
    assert!(html.contains(&paths.fully_covered), "index must link to the fully-covered page");
}

#[test]
fn index_page_links_the_stylesheet_it_writes() {
    let paths = report_paths("report");
    let html = render_index(1, 2, 3, 100, 200, &paths).unwrap();
    assert!(html.contains(CSS_FILE), "index must link the css file write_static_assets emits");
}

#[test]
fn category_page_contains_the_function_name_and_source_view_placeholder() {
    let report = make_report(
        0,
        "rustc_abi::callconv::merge",
        "/rust/compiler/rustc_abi/src/callconv.rs",
        39,
        vec![Some(0)],
        FunctionCategory::FullyUncovered,
    );
    let functions = vec![&report];
    let paths = report_paths("report");

    let html = render_category_page(
        &functions,
        FunctionCategory::FullyUncovered,
        "report_sources",
        &paths,
    )
    .unwrap();

    assert!(
        html.contains("rustc_abi::callconv::merge"),
        "function name must appear in its own page"
    );
    assert!(
        html.contains("data-fn-id=\"0\""),
        "function needs a stable id for the source-shard lookup"
    );
    // The page fetches source itself, so only the placeholder is here.
    assert!(!html.contains("fn merge(self"), "source text must not be inlined in the page");
}

#[test]
fn category_page_uses_the_function_id_not_its_position_in_the_subset() {
    // Ids are handed out across all functions, but a page only gets one
    // category. A page whose first function is id 7 has to still say 7.
    let report = make_report(
        7,
        "a",
        "/rust/compiler/c/src/x.rs",
        1,
        vec![Some(0)],
        FunctionCategory::FullyUncovered,
    );
    let functions = vec![&report];
    let paths = report_paths("report");

    let html = render_category_page(
        &functions,
        FunctionCategory::FullyUncovered,
        "report_sources",
        &paths,
    )
    .unwrap();

    assert!(html.contains("data-fn-id=\"7\""));
    assert!(!html.contains("data-fn-id=\"0\""));
}

#[test]
fn category_page_groups_functions_under_crate_and_file() {
    let a = make_report(
        0,
        "a",
        "/rust/compiler/rustc_abi/src/x.rs",
        1,
        vec![Some(0)],
        FunctionCategory::FullyUncovered,
    );
    let b = make_report(
        1,
        "b",
        "/rust/compiler/rustc_middle/src/y.rs",
        1,
        vec![Some(0)],
        FunctionCategory::FullyUncovered,
    );
    let functions = vec![&a, &b];
    let paths = report_paths("report");

    let html = render_category_page(
        &functions,
        FunctionCategory::FullyUncovered,
        "report_sources",
        &paths,
    )
    .unwrap();

    assert!(html.contains("rustc_abi"));
    assert!(html.contains("rustc_middle"));
    assert!(html.contains("src/x.rs"));
    assert!(html.contains("src/y.rs"));
}
