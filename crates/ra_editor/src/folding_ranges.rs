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
    let mut visited = FxHashSet::default();

    for node in file.syntax().descendants() {
        if visited.contains(&node) {
            continue;
        }

        if let Some(comment) = ast::Comment::cast(node) {
            // Multiline comments (`/* ... */`) can only be folded if they span multiple lines
            let range = if let ast::CommentFlavor::Multiline = comment.flavor() {
                if comment.text().contains('\n') {
                    Some(comment.syntax().range())
                } else {
                    None
                }
            } else {
                contiguous_range_for(SyntaxKind::COMMENT, node, &mut visited)
            };

            range.map(|range| res.push(Fold { range, kind: FoldKind::Comment }));
        }

        if let SyntaxKind::USE_ITEM = node.kind() {
            contiguous_range_for(SyntaxKind::USE_ITEM, node, &mut visited)
                .map(|range| res.push(Fold { range, kind: FoldKind::Imports}));
        };
    }

    res
}

fn contiguous_range_for<'a>(
    kind: SyntaxKind,
    node: SyntaxNodeRef<'a>,
    visited: &mut FxHashSet<SyntaxNodeRef<'a>>,
) -> Option<TextRange> {
    visited.insert(node);

    let left = node;
    let mut right = node;
    for node in node.siblings(Direction::Next) {
        visited.insert(node);
        match node.kind() {
            SyntaxKind::WHITESPACE if !node.leaf_text().unwrap().as_str().contains("\n\n") => (),
            k => {
                if k == kind {
                    right = node
                } else {
                    break;
                }
            }
        }
    }
    if left != right {
        Some(TextRange::from_to(
            left.range().start(),
            right.range().end(),
        ))
    } else {
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
