use rustc_hash::FxHashSet;

use ra_syntax::{
    ast,
    AstNode,
    File, TextRange, SyntaxNodeRef,
    SyntaxKind,
    Direction,
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
    let mut group_members = FxHashSet::default();

    for node in file.syntax().descendants() {
        // Fold items that span multiple lines
        if let Some(kind) = fold_kind(node.kind()) {
            if has_newline(node) {
                res.push(Fold { range: node.range(), kind });
            }
        }

        // Also fold item *groups* that span multiple lines

        // Note: we need to skip elements of the group that we have already visited,
        // otherwise there will be folds for the whole group and for its sub groups
        if group_members.contains(&node) {
            continue;
        }

        if let Some(kind) = fold_kind(node.kind()) {
            contiguous_range_for_group(node.kind(), node, &mut group_members)
                .map(|range| res.push(Fold { range, kind }));
        }
    }

    res
}

fn fold_kind(kind: SyntaxKind) -> Option<FoldKind> {
    match kind {
        SyntaxKind::COMMENT => Some(FoldKind::Comment),
        SyntaxKind::USE_ITEM => Some(FoldKind::Imports),
        _ => None
    }
}

fn has_newline(
    node: SyntaxNodeRef,
) -> bool {
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

fn contiguous_range_for_group<'a>(
    group_kind: SyntaxKind,
    first: SyntaxNodeRef<'a>,
    visited: &mut FxHashSet<SyntaxNodeRef<'a>>,
) -> Option<TextRange> {
    visited.insert(first);

    let mut last = first;

    for node in first.siblings(Direction::Next) {
        visited.insert(node);
        if let Some(ws) = ast::Whitespace::cast(node) {
            // There is a blank line, which means the group ends here
            if ws.count_newlines_lazy().take(2).count() == 2 {
                break;
            }

            // Ignore whitespace without blank lines
            continue;
        }

        // The group ends when an element of a different kind is reached
        if node.kind() != group_kind {
            break;
        }

        // Keep track of the last node in the group
        last = node;
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

    #[test]
    fn test_fold_comments() {
        let text = r#"
// Hello
// this is a multiline
// comment
//

// But this is not

fn main() {
    // We should
    // also
    // fold
    // this one.
}"#;

        let file = File::parse(&text);
        let folds = folding_ranges(&file);
        assert_eq!(folds.len(), 2);
        assert_eq!(folds[0].range.start(), 1.into());
        assert_eq!(folds[0].range.end(), 46.into());
        assert_eq!(folds[0].kind, FoldKind::Comment);

        assert_eq!(folds[1].range.start(), 84.into());
        assert_eq!(folds[1].range.end(), 137.into());
        assert_eq!(folds[1].kind, FoldKind::Comment);
    }

    #[test]
    fn test_fold_imports() {
        let text = r#"
use std::str;
use std::vec;
use std::io as iop;

fn main() {
}"#;

        let file = File::parse(&text);
        let folds = folding_ranges(&file);
        assert_eq!(folds.len(), 1);
        assert_eq!(folds[0].range.start(), 1.into());
        assert_eq!(folds[0].range.end(), 48.into());
        assert_eq!(folds[0].kind, FoldKind::Imports);
    }


}
