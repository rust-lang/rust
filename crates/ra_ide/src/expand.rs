//! Utilities to work with files, produced by macros.
use std::iter::successors;

use hir::{InFile, Origin};
use ra_db::FileId;
use ra_syntax::{ast, AstNode, SyntaxNode, SyntaxToken, TextRange};

use crate::{db::RootDatabase, FileRange};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum OriginalRangeKind {
    /// Return range if any token is matched
    #[allow(dead_code)]
    Any,
    /// Return range if token is inside macro_call
    CallToken,
    /// Return whole macro call range if matched
    WholeCall,
}

pub(crate) fn original_range_by_kind(
    db: &RootDatabase,
    node: InFile<&SyntaxNode>,
    kind: OriginalRangeKind,
) -> Option<FileRange> {
    let expansion = node.file_id.expansion_info(db)?;

    // the input node has only one token ?
    let single = node.value.first_token()? == node.value.last_token()?;

    // FIXME: We should handle recurside macro expansions
    let range = match kind {
        OriginalRangeKind::WholeCall => expansion.call_node()?.map(|node| node.text_range()),
        _ => node.value.descendants().find_map(|it| {
            let first = it.first_token()?;
            let last = it.last_token()?;

            if !single && first == last {
                return None;
            }

            // Try to map first and last tokens of node, and, if success, return the union range of mapped tokens
            let (first, first_origin) = expansion.map_token_up(node.with_value(&first))?;
            let (last, last_origin) = expansion.map_token_up(node.with_value(&last))?;

            if first.file_id != last.file_id
                || first_origin != last_origin
                || (kind == OriginalRangeKind::CallToken && first_origin != Origin::Call)
            {
                return None;
            }

            // FIXME: Add union method in TextRange
            Some(first.with_value(union_range(first.value.text_range(), last.value.text_range())))
        })?,
    };

    return Some(FileRange { file_id: range.file_id.original_file(db), range: range.value });

    fn union_range(a: TextRange, b: TextRange) -> TextRange {
        let start = a.start().min(b.start());
        let end = a.end().max(b.end());
        TextRange::from_to(start, end)
    }
}

pub(crate) fn descend_into_macros(
    db: &RootDatabase,
    file_id: FileId,
    token: SyntaxToken,
) -> InFile<SyntaxToken> {
    let src = InFile::new(file_id.into(), token);

    successors(Some(src), |token| {
        let macro_call = token.value.ancestors().find_map(ast::MacroCall::cast)?;
        let tt = macro_call.token_tree()?;
        if !token.value.text_range().is_subrange(&tt.syntax().text_range()) {
            return None;
        }
        let source_analyzer =
            hir::SourceAnalyzer::new(db, token.with_value(token.value.parent()).as_ref(), None);
        let exp = source_analyzer.expand(db, token.with_value(&macro_call))?;
        exp.map_token_down(db, token.as_ref())
    })
    .last()
    .unwrap()
}
