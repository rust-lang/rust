//! Runs completion for testing purposes.

use hir::Semantics;
use itertools::Itertools;
use ra_syntax::{AstNode, NodeOrToken, SyntaxElement};
use stdx::format_to;
use test_utils::assert_eq_text;

use crate::{
    completion::{completion_item::CompletionKind, CompletionConfig},
    mock_analysis::analysis_and_position,
    CompletionItem,
};

pub(crate) fn do_completion(code: &str, kind: CompletionKind) -> Vec<CompletionItem> {
    do_completion_with_options(code, kind, &CompletionConfig::default())
}

pub(crate) fn do_completion_with_options(
    code: &str,
    kind: CompletionKind,
    options: &CompletionConfig,
) -> Vec<CompletionItem> {
    let mut kind_completions: Vec<CompletionItem> = get_all_completion_items(code, options)
        .into_iter()
        .filter(|c| c.completion_kind == kind)
        .collect();
    kind_completions.sort_by(|l, r| l.label().cmp(r.label()));
    kind_completions
}

pub(crate) fn completion_list(code: &str, kind: CompletionKind) -> String {
    completion_list_with_options(code, kind, &CompletionConfig::default())
}

pub(crate) fn completion_list_with_options(
    code: &str,
    kind: CompletionKind,
    options: &CompletionConfig,
) -> String {
    let mut kind_completions: Vec<CompletionItem> = get_all_completion_items(code, options)
        .into_iter()
        .filter(|c| c.completion_kind == kind)
        .collect();
    kind_completions.sort_by_key(|c| c.label().to_owned());
    kind_completions
        .into_iter()
        .map(|it| {
            let mut buf = format!("{} {}", it.kind().unwrap().tag(), it.label());
            if let Some(detail) = it.detail() {
                format_to!(buf, " {}", detail);
            }
            format_to!(buf, "\n");
            buf
        })
        .collect()
}

pub(crate) fn check_edit(what: &str, ra_fixture_before: &str, ra_fixture_after: &str) {
    let (analysis, position) = analysis_and_position(ra_fixture_before);
    let completions: Vec<CompletionItem> =
        analysis.completions(&CompletionConfig::default(), position).unwrap().unwrap().into();
    let (completion,) =
        completions.into_iter().filter(|it| it.label() == what).collect_tuple().unwrap();
    let mut actual = analysis.file_text(position.file_id).unwrap().to_string();
    completion.text_edit().apply(&mut actual);
    assert_eq_text!(ra_fixture_after, &actual)
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

fn get_all_completion_items(code: &str, options: &CompletionConfig) -> Vec<CompletionItem> {
    let (analysis, position) = analysis_and_position(code);
    analysis.completions(options, position).unwrap().unwrap().into()
}
