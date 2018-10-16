use rustc_hash::FxHashSet;

use ra_syntax::{
    ast, AstNode, Direction, File,
    SyntaxKind::{self, *},
    SyntaxNodeRef, TextRange,
};

#[derive(Debug, PartialEq, Eq)]
pub enum FoldKind {
    Comment,
    Imports,
}

#[derive(Debug)]
pub struct Fold {
    pub range: TextRange,
    pub kind: FoldKind,
}

pub fn folding_ranges(file: &File) -> Vec<Fold> {
    let mut res = vec![];
    let mut visited_comments = FxHashSet::default();

    for node in file.syntax().descendants() {
        // Fold items that span multiple lines
        if let Some(kind) = fold_kind(node.kind()) {
            if has_newline(node) {
                res.push(Fold {
                    range: node.range(),
                    kind,
                });
            }
        }

        // Also fold groups of comments
        if visited_comments.contains(&node) {
            continue;
        }
        if node.kind() == COMMENT {
            contiguous_range_for_comment(node, &mut visited_comments).map(|range| {
                res.push(Fold {
                    range,
                    kind: FoldKind::Comment,
                })
            });
        }
    }

    res
}

fn fold_kind(kind: SyntaxKind) -> Option<FoldKind> {
    match kind {
        COMMENT => Some(FoldKind::Comment),
        USE_ITEM => Some(FoldKind::Imports),
        _ => None,
    }
}

fn has_newline(node: SyntaxNodeRef) -> bool {
    for descendant in node.descendants() {
        if let Some(ws) = ast::Whitespace::cast(descendant) {
            if ws.has_newlines() {
                return true;
            }
        } else if let Some(comment) = ast::Comment::cast(descendant) {
            if comment.has_newlines() {
                return true;
            }
        }
    }

    false
}

fn contiguous_range_for_comment<'a>(
    first: SyntaxNodeRef<'a>,
    visited: &mut FxHashSet<SyntaxNodeRef<'a>>,
) -> Option<TextRange> {
    visited.insert(first);

    // Only fold comments of the same flavor
    let group_flavor = ast::Comment::cast(first)?.flavor();

    let mut last = first;
    for node in first.siblings(Direction::Next) {
        if let Some(ws) = ast::Whitespace::cast(node) {
            // There is a blank line, which means the group ends here
            if ws.count_newlines_lazy().take(2).count() == 2 {
                break;
            }

            // Ignore whitespace without blank lines
            continue;
        }

        match ast::Comment::cast(node) {
            Some(next_comment) if next_comment.flavor() == group_flavor => {
                visited.insert(node);
                last = node;
            }
            // The comment group ends because either:
            // * An element of a different kind was reached
            // * A comment of a different flavor was reached
            _ => break,
        }
    }

    if first != last {
        Some(TextRange::from_to(
            first.range().start(),
            last.range().end(),
        ))
    } else {
        // The group consists of only one element, therefore it cannot be folded
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_utils::extract_ranges;

    fn do_check(text: &str, fold_kinds: &[FoldKind]) {
        let (ranges, text) = extract_ranges(text);
        let file = File::parse(&text);
        let folds = folding_ranges(&file);

        assert_eq!(folds.len(), ranges.len());
        for ((fold, range), fold_kind) in folds
            .into_iter()
            .zip(ranges.into_iter())
            .zip(fold_kinds.into_iter())
        {
            assert_eq!(fold.range.start(), range.start());
            assert_eq!(fold.range.end(), range.end());
            assert_eq!(&fold.kind, fold_kind);
        }
    }

    #[test]
    fn test_fold_comments() {
        let text = r#"
<|>// Hello
// this is a multiline
// comment
//<|>

// But this is not

fn main() {
    <|>// We should
    // also
    // fold
    // this one.<|>
    <|>//! But this one is different
    //! because it has another flavor<|>
    <|>/* As does this
    multiline comment */<|>
}"#;

        let fold_kinds = &[
            FoldKind::Comment,
            FoldKind::Comment,
            FoldKind::Comment,
            FoldKind::Comment,
        ];
        do_check(text, fold_kinds);
    }

    #[test]
    fn test_fold_imports() {
        let text = r#"
<|>use std::{
    str,
    vec,
    io as iop
};<|>

fn main() {
}"#;

        let folds = &[FoldKind::Imports];
        do_check(text, folds);
    }

}
