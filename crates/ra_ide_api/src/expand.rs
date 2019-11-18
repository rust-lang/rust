//! Utilities to work with files, produced by macros.
use std::iter::successors;

use hir::Source;
use ra_db::FileId;
use ra_syntax::{ast, AstNode, SyntaxNode, SyntaxToken};

use crate::{db::RootDatabase, FileRange};

pub(crate) fn original_range(db: &RootDatabase, node: Source<&SyntaxNode>) -> FileRange {
    let expansion = match node.file_id.expansion_info(db) {
        None => {
            return FileRange {
                file_id: node.file_id.original_file(db),
                range: node.ast.text_range(),
            }
        }
        Some(it) => it,
    };
    // FIXME: the following completely wrong.
    //
    // *First*, we should try to map first and last tokens of node, and, if that
    // fails, return the range of the overall macro expansions.
    //
    // *Second*, we should handle recurside macro expansions

    let token = node
        .ast
        .descendants_with_tokens()
        .filter_map(|it| it.into_token())
        .find_map(|it| expansion.map_token_up(node.with_ast(&it)));

    match token {
        Some(it) => FileRange { file_id: it.file_id.original_file(db), range: it.ast.text_range() },
        None => FileRange { file_id: node.file_id.original_file(db), range: node.ast.text_range() },
    }
}

pub(crate) fn descend_into_macros(
    db: &RootDatabase,
    file_id: FileId,
    token: SyntaxToken,
) -> Source<SyntaxToken> {
    let src = Source::new(file_id.into(), token);

    successors(Some(src), |token| {
        let macro_call = token.ast.ancestors().find_map(ast::MacroCall::cast)?;
        let tt = macro_call.token_tree()?;
        if !token.ast.text_range().is_subrange(&tt.syntax().text_range()) {
            return None;
        }
        let source_analyzer =
            hir::SourceAnalyzer::new(db, token.with_ast(token.ast.parent()).as_ref(), None);
        let exp = source_analyzer.expand(db, &macro_call)?;
        exp.map_token_down(db, token.as_ref())
    })
    .last()
    .unwrap()
}
