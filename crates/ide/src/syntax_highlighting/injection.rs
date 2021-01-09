//! Syntax highlighting injections such as highlighting of documentation tests.

use std::convert::TryFrom;

use hir::Semantics;
use ide_db::call_info::ActiveParameter;
use itertools::Itertools;
use syntax::{ast, AstToken, SyntaxNode, SyntaxToken, TextRange, TextSize};

use crate::{Analysis, HlMod, HlRange, HlTag, RootDatabase};

use super::{highlights::Highlights, injector::Injector};

pub(super) fn highlight_injection(
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
    let marker_info = MarkerInfo::new(&*value);
    let (analysis, tmp_file_id) = Analysis::from_single_file(marker_info.cleaned_text.clone());

    if let Some(range) = literal.open_quote_text_range() {
        hl.add(HlRange { range, highlight: HlTag::StringLiteral.into(), binding_hash: None })
    }

    for mut hl_range in analysis.highlight(tmp_file_id).unwrap() {
        let range = marker_info.map_range_up(hl_range.range);
        if let Some(range) = literal.map_range_up(range) {
            hl_range.range = range;
            hl.add(hl_range);
        }
    }

    if let Some(range) = literal.close_quote_text_range() {
        hl.add(HlRange { range, highlight: HlTag::StringLiteral.into(), binding_hash: None })
    }

    Some(())
}

/// Data to remove `$0` from string and map ranges
#[derive(Default, Debug)]
struct MarkerInfo {
    cleaned_text: String,
    markers: Vec<TextRange>,
}

impl MarkerInfo {
    fn new(mut text: &str) -> Self {
        let marker = "$0";

        let mut res = MarkerInfo::default();
        let mut offset: TextSize = 0.into();
        while !text.is_empty() {
            let idx = text.find(marker).unwrap_or(text.len());
            let (chunk, next) = text.split_at(idx);
            text = next;
            res.cleaned_text.push_str(chunk);
            offset += TextSize::of(chunk);

            if let Some(next) = text.strip_prefix(marker) {
                text = next;

                let marker_len = TextSize::of(marker);
                res.markers.push(TextRange::at(offset, marker_len));
                offset += marker_len;
            }
        }
        res
    }
    fn map_range_up(&self, range: TextRange) -> TextRange {
        TextRange::new(
            self.map_offset_up(range.start(), true),
            self.map_offset_up(range.end(), false),
        )
    }
    fn map_offset_up(&self, mut offset: TextSize, start: bool) -> TextSize {
        for r in &self.markers {
            if r.start() < offset || (start && r.start() == offset) {
                offset += r.len()
            }
        }
        offset
    }
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

/// Extracts Rust code from documentation comments as well as a mapping from
/// the extracted source code back to the original source ranges.
/// Lastly, a vector of new comment highlight ranges (spanning only the
/// comment prefix) is returned which is used in the syntax highlighting
/// injection to replace the previous (line-spanning) comment ranges.
pub(super) fn extract_doc_comments(node: &SyntaxNode) -> Option<(Vec<HlRange>, Injector)> {
    let mut inj = Injector::default();
    // wrap the doctest into function body to get correct syntax highlighting
    let prefix = "fn doctest() {\n";
    let suffix = "}\n";

    let mut line_start = TextSize::of(prefix);
    let mut is_codeblock = false;
    let mut is_doctest = false;
    // Replace the original, line-spanning comment ranges by new, only comment-prefix
    // spanning comment ranges.
    let mut new_comments = Vec::new();

    inj.add_unmapped(prefix);
    let doctest = node
        .children_with_tokens()
        .filter_map(|el| el.into_token().and_then(ast::Comment::cast))
        .filter(|comment| comment.kind().doc.is_some())
        .filter(|comment| {
            if let Some(idx) = comment.text().find(RUSTDOC_FENCE) {
                is_codeblock = !is_codeblock;
                // Check whether code is rust by inspecting fence guards
                let guards = &comment.text()[idx + RUSTDOC_FENCE.len()..];
                let is_rust =
                    guards.split(',').all(|sub| RUSTDOC_FENCE_TOKENS.contains(&sub.trim()));
                is_doctest = is_codeblock && is_rust;
                false
            } else {
                is_doctest
            }
        })
        .map(|comment| {
            let prefix_len = comment.prefix().len();
            let line: &str = comment.text().as_str();
            let range = comment.syntax().text_range();

            // whitespace after comment is ignored
            let pos = if let Some(ws) = line.chars().nth(prefix_len).filter(|c| c.is_whitespace()) {
                prefix_len + ws.len_utf8()
            } else {
                prefix_len
            };

            // lines marked with `#` should be ignored in output, we skip the `#` char
            let pos = if let Some(ws) = line.chars().nth(pos).filter(|&c| c == '#') {
                pos + ws.len_utf8()
            } else {
                pos
            };

            new_comments.push(HlRange {
                range: TextRange::new(
                    range.start(),
                    range.start() + TextSize::try_from(pos).unwrap(),
                ),
                highlight: HlTag::Comment | HlMod::Documentation,
                binding_hash: None,
            });
            line_start += range.len() - TextSize::try_from(pos).unwrap();
            line_start += TextSize::of("\n");

            inj.add(
                &line[pos..],
                TextRange::new(range.start() + TextSize::try_from(pos).unwrap(), range.end()),
            );
            inj.add_unmapped("\n");
            line[pos..].to_owned()
        })
        .join("\n");
    inj.add_unmapped(suffix);

    if doctest.is_empty() {
        return None;
    }

    Some((new_comments, inj))
}

/// Injection of syntax highlighting of doctests.
pub(super) fn highlight_doc_comment(
    new_comments: Vec<HlRange>,
    inj: Injector,
    stack: &mut Highlights,
) {
    let (analysis, tmp_file_id) = Analysis::from_single_file(inj.text().to_string());
    for comment in new_comments {
        stack.add(comment);
    }

    for h in analysis.with_db(|db| super::highlight(db, tmp_file_id, None, true)).unwrap() {
        for r in inj.map_range_up(h.range) {
            stack.add(HlRange {
                range: r,
                highlight: h.highlight | HlMod::Injected,
                binding_hash: h.binding_hash,
            });
        }
    }
}
