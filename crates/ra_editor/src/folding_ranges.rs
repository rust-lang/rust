use std::collections::HashSet;

use ra_syntax::{
    File, TextRange, SyntaxNodeRef,
    SyntaxKind,
    algo::{walk, Direction, siblings},
};

pub enum FoldKind {
    Comment,
    Imports,
}

pub struct Fold {
    pub range: TextRange,
    pub kind: FoldKind,
}

pub fn folding_ranges(file: &File) -> Vec<Fold> {
    let syntax = file.syntax();

    let mut res = vec![];
    let mut visited = HashSet::new();

    for node in walk::preorder(syntax) {
        if visited.contains(&node) {
            continue;
        }

        let range_and_kind = match node.kind() {
            SyntaxKind::COMMENT => (
                contiguous_range_for(SyntaxKind::COMMENT, node, &mut visited),
                Some(FoldKind::Comment),
            ),
            SyntaxKind::USE_ITEM => (
                contiguous_range_for(SyntaxKind::USE_ITEM, node, &mut visited),
                Some(FoldKind::Imports),
            ),
            _ => (None, None),
        };

        match range_and_kind {
            (Some(range), Some(kind)) => {
                res.push(Fold {
                    range: range,
                    kind: kind
                });
            }
            _ => {}
        }
    }

    res
}

fn contiguous_range_for<'a>(
    kind: SyntaxKind,
    node: SyntaxNodeRef<'a>,
    visited: &mut HashSet<SyntaxNodeRef<'a>>,
) -> Option<TextRange> {
    visited.insert(node);

    let left = node;
    let mut right = node;
    for node in siblings(node, Direction::Forward) {
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