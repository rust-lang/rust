//! Runs completion for testing purposes.

use crate::{
    completion::{completion_item::CompletionKind, CompletionConfig},
    mock_analysis::{analysis_and_position, single_file_with_position},
    CompletionItem,
};
use hir::Semantics;
use ra_syntax::{AstNode, NodeOrToken, SyntaxElement};

pub(crate) fn do_completion(code: &str, kind: CompletionKind) -> Vec<CompletionItem> {
    do_completion_with_options(code, kind, &CompletionConfig::default())
}

pub(crate) fn get_completions(code: &str, kind: CompletionKind) -> Vec<String> {
    get_completions_with_options(code, kind, &CompletionConfig::default())
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

fn get_all_completion_items(code: &str, options: &CompletionConfig) -> Vec<CompletionItem> {
    let (analysis, position) = if code.contains("//-") {
        analysis_and_position(code)
    } else {
        single_file_with_position(code)
    };
    analysis.completions(options, position).unwrap().unwrap().into()
}

pub(crate) fn get_completions_with_options(
    code: &str,
    kind: CompletionKind,
    options: &CompletionConfig,
) -> Vec<String> {
    let mut kind_completions: Vec<CompletionItem> = get_all_completion_items(code, options)
        .into_iter()
        .filter(|c| c.completion_kind == kind)
        .collect();
    kind_completions.sort_by_key(|c| c.label().to_owned());
    kind_completions
        .into_iter()
        .map(|it| format!("{} {}", it.kind().unwrap().tag(), it.label()))
        .collect()
}

pub(crate) fn check_pattern_is_applicable(code: &str, check: fn(SyntaxElement) -> bool) {
    let (analysis, pos) = single_file_with_position(code);
    analysis
        .with_db(|db| {
            let sema = Semantics::new(db);
            let original_file = sema.parse(pos.file_id);
            let token = original_file.syntax().token_at_offset(pos.offset).left_biased().unwrap();
            assert!(check(NodeOrToken::Token(token)));
        })
        .unwrap();
}
