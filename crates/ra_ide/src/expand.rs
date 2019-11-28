//! Utilities to work with files, produced by macros.
use std::iter::successors;

use hir::InFile;
use ra_db::FileId;
use ra_syntax::{ast, AstNode, SyntaxNode, SyntaxToken};

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
    // FIXME: the following completely wrong.
    //
    // *First*, we should try to map first and last tokens of node, and, if that
    // fails, return the range of the overall macro expansions.
    //
    // *Second*, we should handle recurside macro expansions

    let token = node
        .value
        .descendants_with_tokens()
        .filter_map(|it| it.into_token())
        .find_map(|it| expansion.map_token_up(node.with_value(&it)));

    match token {
        Some(it) => {
            FileRange { file_id: it.file_id.original_file(db), range: it.value.text_range() }
        }
        None => {
            FileRange { file_id: node.file_id.original_file(db), range: node.value.text_range() }
        }
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
