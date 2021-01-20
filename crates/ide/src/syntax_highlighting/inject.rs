//! "Recursive" Syntax highlighting for code in doctests and fixtures.

use hir::Semantics;
use ide_db::call_info::ActiveParameter;
use syntax::{ast, AstToken, SyntaxNode, SyntaxToken, TextRange, TextSize};

use crate::{Analysis, HlMod, HlRange, HlTag, RootDatabase};

use super::{highlights::Highlights, injector::Injector};

pub(super) fn ra_fixture(
    hl: &mut Highlights,
    sema: &Semantics<RootDatabase>,
    literal: ast::String,
    expanded: SyntaxToken,
) -> Option<()> {
    let active_parameter = ActiveParameter::at_token(&sema, expanded)?;
    if !active_parameter.name.starts_with("ra_fixture") {
        return None;
    }
    let value = literal.value()?;

    if let Some(range) = literal.open_quote_text_range() {
        hl.add(HlRange { range, highlight: HlTag::StringLiteral.into(), binding_hash: None })
    }

    let mut inj = Injector::default();

    let mut text = &*value;
    let mut offset: TextSize = 0.into();

    while !text.is_empty() {
        let marker = "$0";
        let idx = text.find(marker).unwrap_or(text.len());
        let (chunk, next) = text.split_at(idx);
        inj.add(chunk, TextRange::at(offset, TextSize::of(chunk)));

        text = next;
        offset += TextSize::of(chunk);

        if let Some(next) = text.strip_prefix(marker) {
            if let Some(range) = literal.map_range_up(TextRange::at(offset, TextSize::of(marker))) {
                hl.add(HlRange { range, highlight: HlTag::Keyword.into(), binding_hash: None });
            }

            text = next;

            let marker_len = TextSize::of(marker);
            offset += marker_len;
        }
    }

    let (analysis, tmp_file_id) = Analysis::from_single_file(inj.text().to_string());

    for mut hl_range in analysis.highlight(tmp_file_id).unwrap() {
        for range in inj.map_range_up(hl_range.range) {
            if let Some(range) = literal.map_range_up(range) {
                hl_range.range = range;
                hl.add(hl_range.clone());
            }
        }
    }

    if let Some(range) = literal.close_quote_text_range() {
        hl.add(HlRange { range, highlight: HlTag::StringLiteral.into(), binding_hash: None })
    }

    Some(())
}

const RUSTDOC_FENCE: &'static str = "```";
const RUSTDOC_FENCE_TOKENS: &[&'static str] = &[
    "",
    "rust",
    "should_panic",
    "ignore",
    "no_run",
    "compile_fail",
    "edition2015",
    "edition2018",
    "edition2021",
];

/// Injection of syntax highlighting of doctests.
pub(super) fn doc_comment(hl: &mut Highlights, node: &SyntaxNode) {
    let doc_comments = node
        .children_with_tokens()
        .filter_map(|it| it.into_token().and_then(ast::Comment::cast))
        .filter(|it| it.kind().doc.is_some());

    if !doc_comments.clone().any(|it| it.text().contains(RUSTDOC_FENCE)) {
        return;
    }

    let mut inj = Injector::default();
    inj.add_unmapped("fn doctest() {\n");

    let mut is_codeblock = false;
    let mut is_doctest = false;

    // Replace the original, line-spanning comment ranges by new, only comment-prefix
    // spanning comment ranges.
    let mut new_comments = Vec::new();
    for comment in doc_comments {
        match comment.text().find(RUSTDOC_FENCE) {
            Some(idx) => {
                is_codeblock = !is_codeblock;
                // Check whether code is rust by inspecting fence guards
                let guards = &comment.text()[idx + RUSTDOC_FENCE.len()..];
                let is_rust =
                    guards.split(',').all(|sub| RUSTDOC_FENCE_TOKENS.contains(&sub.trim()));
                is_doctest = is_codeblock && is_rust;
                continue;
            }
            None if !is_doctest => continue,
            None => (),
        }

        let line: &str = comment.text();
        let range = comment.syntax().text_range();

        let mut pos = TextSize::of(comment.prefix());
        // whitespace after comment is ignored
        if let Some(ws) = line[pos.into()..].chars().next().filter(|c| c.is_whitespace()) {
            pos += TextSize::of(ws);
        }
        // lines marked with `#` should be ignored in output, we skip the `#` char
        if let Some(ws) = line[pos.into()..].chars().next().filter(|&c| c == '#') {
            pos += TextSize::of(ws);
        }

        new_comments.push(TextRange::at(range.start(), pos));

        inj.add(&line[pos.into()..], TextRange::new(range.start() + pos, range.end()));
        inj.add_unmapped("\n");
    }
    inj.add_unmapped("\n}");

    let (analysis, tmp_file_id) = Analysis::from_single_file(inj.text().to_string());

    for h in analysis.with_db(|db| super::highlight(db, tmp_file_id, None, true)).unwrap() {
        for r in inj.map_range_up(h.range) {
            hl.add(HlRange {
                range: r,
                highlight: h.highlight | HlMod::Injected,
                binding_hash: h.binding_hash,
            });
        }
    }

    for range in new_comments {
        hl.add(HlRange {
            range,
            highlight: HlTag::Comment | HlMod::Documentation,
            binding_hash: None,
        });
    }
}
