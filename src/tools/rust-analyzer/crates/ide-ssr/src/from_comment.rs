//! This module allows building an SSR MatchFinder by parsing the SSR rule
//! from a comment.

use ide_db::{
    base_db::{FilePosition, FileRange, SourceDatabase},
    RootDatabase,
};
use syntax::{
    ast::{self, AstNode, AstToken},
    TextRange,
};

use crate::MatchFinder;

/// Attempts to build an SSR MatchFinder from a comment at the given file
/// range. If successful, returns the MatchFinder and a TextRange covering
/// comment.
pub fn ssr_from_comment(
    db: &RootDatabase,
    frange: FileRange,
) -> Option<(MatchFinder<'_>, TextRange)> {
    let comment = {
        let file = db.parse(frange.file_id);
        file.tree().syntax().token_at_offset(frange.range.start()).find_map(ast::Comment::cast)
    }?;
    let comment_text_without_prefix = comment.text().strip_prefix(comment.prefix()).unwrap();
    let ssr_rule = comment_text_without_prefix.parse().ok()?;

    let lookup_context = FilePosition { file_id: frange.file_id, offset: frange.range.start() };

    let mut match_finder = MatchFinder::in_context(db, lookup_context, vec![]).ok()?;
    match_finder.add_rule(ssr_rule).ok()?;

    Some((match_finder, comment.syntax().text_range()))
}
