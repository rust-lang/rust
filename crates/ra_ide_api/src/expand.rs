//! Utilities to work with files, produced by macros.
use std::iter::successors;

use hir::Source;
use ra_db::FileId;
use ra_syntax::{ast, AstNode, SyntaxNode, SyntaxToken};

use crate::{db::RootDatabase, FileRange};

pub(crate) fn original_range(db: &RootDatabase, node: Source<&SyntaxNode>) -> FileRange {
    let text_range = node.ast.text_range();
    let (file_id, range) = node
        .file_id
        .expansion_info(db)
        .and_then(|expansion_info| expansion_info.find_range(text_range))
        .unwrap_or((node.file_id, text_range));

    // FIXME: handle recursive macro generated macro
    FileRange { file_id: file_id.original_file(db), range }
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
