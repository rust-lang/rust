//! Utilities to work with files, produced by macros.
use std::iter::successors;

use hir::{InFile, Origin};
use ra_db::FileId;
use ra_syntax::{ast, AstNode, SyntaxNode, SyntaxToken, TextRange};

use crate::{db::RootDatabase, FileRange};

pub(crate) fn original_range(db: &RootDatabase, node: InFile<&SyntaxNode>) -> FileRange {
    if let Some((range, Origin::Call)) = original_range_and_origin(db, node) {
        return range;
    }

    if let Some(expansion) = node.file_id.expansion_info(db) {
        if let Some(call_node) = expansion.call_node() {
            return FileRange {
                file_id: call_node.file_id.original_file(db),
                range: call_node.value.text_range(),
            };
        }
    }

    FileRange { file_id: node.file_id.original_file(db), range: node.value.text_range() }
}

fn original_range_and_origin(
    db: &RootDatabase,
    node: InFile<&SyntaxNode>,
) -> Option<(FileRange, Origin)> {
    let expansion = node.file_id.expansion_info(db)?;

    // the input node has only one token ?
    let single = node.value.first_token()? == node.value.last_token()?;

    // FIXME: We should handle recurside macro expansions
    let (range, origin) = node.value.descendants().find_map(|it| {
        let first = it.first_token()?;
        let last = it.last_token()?;

        if !single && first == last {
            return None;
        }

        // Try to map first and last tokens of node, and, if success, return the union range of mapped tokens
        let (first, first_origin) = expansion.map_token_up(node.with_value(&first))?;
        let (last, last_origin) = expansion.map_token_up(node.with_value(&last))?;

        if first.file_id != last.file_id || first_origin != last_origin {
            return None;
        }

        // FIXME: Add union method in TextRange
        Some((
            first.with_value(union_range(first.value.text_range(), last.value.text_range())),
            first_origin,
        ))
    })?;

    return Some((
        FileRange { file_id: range.file_id.original_file(db), range: range.value },
        origin,
    ));

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
