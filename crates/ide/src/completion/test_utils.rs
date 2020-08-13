//! Runs completion for testing purposes.

use hir::Semantics;
use itertools::Itertools;
use stdx::{format_to, trim_indent};
use syntax::{AstNode, NodeOrToken, SyntaxElement};
use test_utils::assert_eq_text;

use crate::{
    completion::{completion_item::CompletionKind, CompletionConfig},
    mock_analysis::analysis_and_position,
    CompletionItem,
};

pub(crate) fn do_completion(code: &str, kind: CompletionKind) -> Vec<CompletionItem> {
    do_completion_with_config(CompletionConfig::default(), code, kind)
}

pub(crate) fn do_completion_with_config(
    config: CompletionConfig,
    code: &str,
    kind: CompletionKind,
) -> Vec<CompletionItem> {
    let mut kind_completions: Vec<CompletionItem> = get_all_completion_items(config, code)
        .into_iter()
        .filter(|c| c.completion_kind == kind)
        .collect();
    kind_completions.sort_by(|l, r| l.label().cmp(r.label()));
    kind_completions
}

pub(crate) fn completion_list(code: &str, kind: CompletionKind) -> String {
    completion_list_with_config(CompletionConfig::default(), code, kind)
}

pub(crate) fn completion_list_with_config(
    config: CompletionConfig,
    code: &str,
    kind: CompletionKind,
) -> String {
    let mut kind_completions: Vec<CompletionItem> = get_all_completion_items(config, code)
        .into_iter()
        .filter(|c| c.completion_kind == kind)
        .collect();
    kind_completions.sort_by_key(|c| c.label().to_owned());
    let label_width = kind_completions
        .iter()
        .map(|it| monospace_width(it.label()))
        .max()
        .unwrap_or_default()
        .min(16);
    kind_completions
        .into_iter()
        .map(|it| {
            let tag = it.kind().unwrap().tag();
            let var_name = format!("{} {}", tag, it.label());
            let mut buf = var_name;
            if let Some(detail) = it.detail() {
                let width = label_width.saturating_sub(monospace_width(it.label()));
                format_to!(buf, "{:width$} {}", "", detail, width = width);
            }
            format_to!(buf, "\n");
            buf
        })
        .collect()
}

fn monospace_width(s: &str) -> usize {
    s.chars().count()
}

pub(crate) fn check_edit(what: &str, ra_fixture_before: &str, ra_fixture_after: &str) {
    check_edit_with_config(CompletionConfig::default(), what, ra_fixture_before, ra_fixture_after)
}

pub(crate) fn check_edit_with_config(
    config: CompletionConfig,
    what: &str,
    ra_fixture_before: &str,
    ra_fixture_after: &str,
) {
    let ra_fixture_after = trim_indent(ra_fixture_after);
    let (analysis, position) = analysis_and_position(ra_fixture_before);
    let completions: Vec<CompletionItem> =
        analysis.completions(&config, position).unwrap().unwrap().into();
    let (completion,) = completions
        .iter()
        .filter(|it| it.lookup() == what)
        .collect_tuple()
        .unwrap_or_else(|| panic!("can't find {:?} completion in {:#?}", what, completions));
    let mut actual = analysis.file_text(position.file_id).unwrap().to_string();
    completion.text_edit().apply(&mut actual);
    assert_eq_text!(&ra_fixture_after, &actual)
}

pub(crate) fn check_pattern_is_applicable(code: &str, check: fn(SyntaxElement) -> bool) {
    let (analysis, pos) = analysis_and_position(code);
    analysis
        .with_db(|db| {
            let sema = Semantics::new(db);
            let original_file = sema.parse(pos.file_id);
            let token = original_file.syntax().token_at_offset(pos.offset).left_biased().unwrap();
            assert!(check(NodeOrToken::Token(token)));
        })
        .unwrap();
}

pub(crate) fn get_all_completion_items(
    config: CompletionConfig,
    code: &str,
) -> Vec<CompletionItem> {
    let (analysis, position) = analysis_and_position(code);
    analysis.completions(&config, position).unwrap().unwrap().into()
}
