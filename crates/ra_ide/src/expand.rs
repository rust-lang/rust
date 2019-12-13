//! Utilities to work with files, produced by macros.
use std::iter::successors;

use hir::InFile;
use ra_db::FileId;
use ra_syntax::{ast, AstNode, SyntaxNode, SyntaxToken, TextRange};

use crate::{db::RootDatabase, FileRange};

pub(crate) fn original_range(db: &RootDatabase, node: InFile<&SyntaxNode>) -> FileRange {
    let expansion = match node.file_id.expansion_info(db) {
        None => {
            return FileRange {
                file_id: node.file_id.original_file(db),
                range: node.value.text_range(),
            }
        }
        Some(it) => it,
    };
    // FIXME: We should handle recurside macro expansions

    let range = node.value.descendants_with_tokens().find_map(|it| {
        match it.as_token() {
            // FIXME: Remove this branch after all `tt::TokenTree`s have a proper `TokenId`,
            // and return the range of the overall macro expansions if mapping first and last tokens fails.
            Some(token) => {
                let token = expansion.map_token_up(node.with_value(&token))?;
                Some(token.with_value(token.value.text_range()))
            }
            None => {
                // Try to map first and last tokens of node, and, if success, return the union range of mapped tokens
                let n = it.into_node()?;
                let first = expansion.map_token_up(node.with_value(&n.first_token()?))?;
                let last = expansion.map_token_up(node.with_value(&n.last_token()?))?;

                // FIXME: Is is possible ?
                if first.file_id != last.file_id {
                    return None;
                }

                // FIXME: Add union method in TextRange
                let range = union_range(first.value.text_range(), last.value.text_range());
                Some(first.with_value(range))
            }
        }
    });

    return match range {
        Some(it) => FileRange { file_id: it.file_id.original_file(db), range: it.value },
        None => {
            FileRange { file_id: node.file_id.original_file(db), range: node.value.text_range() }
        }
    };

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
