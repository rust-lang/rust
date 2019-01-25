use rustc_hash::FxHashSet;

use ra_syntax::{
    ast, AstNode, Direction, SourceFile, SyntaxNode, TextRange,
    SyntaxKind::{self, *},
};

#[derive(Debug, PartialEq, Eq)]
pub enum FoldKind {
    Comment,
    Imports,
    Mods,
    Block,
}

#[derive(Debug)]
pub struct Fold {
    pub range: TextRange,
    pub kind: FoldKind,
}

pub fn folding_ranges(file: &SourceFile) -> Vec<Fold> {
    let mut res = vec![];
    let mut visited_comments = FxHashSet::default();
    let mut visited_imports = FxHashSet::default();
    let mut visited_mods = FxHashSet::default();

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

        // Fold groups of comments
        if node.kind() == COMMENT && !visited_comments.contains(&node) {
            if let Some(range) = contiguous_range_for_comment(node, &mut visited_comments) {
                res.push(Fold {
                    range,
                    kind: FoldKind::Comment,
                })
            }
        }

        // Fold groups of imports
        if node.kind() == USE_ITEM && !visited_imports.contains(&node) {
            if let Some(range) = contiguous_range_for_group(node, &mut visited_imports) {
                res.push(Fold {
                    range,
                    kind: FoldKind::Imports,
                })
            }
        }

        // Fold groups of mods
        if node.kind() == MODULE && !has_visibility(&node) && !visited_mods.contains(&node) {
            if let Some(range) =
                contiguous_range_for_group_unless(node, has_visibility, &mut visited_mods)
            {
                res.push(Fold {
                    range,
                    kind: FoldKind::Mods,
                })
            }
        }
    }

    res
}

fn fold_kind(kind: SyntaxKind) -> Option<FoldKind> {
    match kind {
        COMMENT => Some(FoldKind::Comment),
        USE_ITEM => Some(FoldKind::Imports),
        NAMED_FIELD_DEF_LIST | FIELD_PAT_LIST | ITEM_LIST | EXTERN_ITEM_LIST | USE_TREE_LIST
        | BLOCK | ENUM_VARIANT_LIST | TOKEN_TREE => Some(FoldKind::Block),
        _ => None,
    }
}

fn has_visibility(node: &SyntaxNode) -> bool {
    return node.descendants().any(|n| n.kind() == VISIBILITY);
}

fn has_newline(node: &SyntaxNode) -> bool {
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
    first: &'a SyntaxNode,
    visited: &mut FxHashSet<&'a SyntaxNode>,
) -> Option<TextRange> {
    contiguous_range_for_group_unless(first, |_| false, visited)
}

fn contiguous_range_for_group_unless<'a>(
    first: &'a SyntaxNode,
    unless: impl Fn(&'a SyntaxNode) -> bool,
    visited: &mut FxHashSet<&'a SyntaxNode>,
) -> Option<TextRange> {
    visited.insert(first);

    let mut last = first;
    for node in first.siblings(Direction::Next) {
        if let Some(ws) = ast::Whitespace::cast(node) {
            // There is a blank line, which means that the group ends here
            if ws.count_newlines_lazy().take(2).count() == 2 {
                break;
            }

            // Ignore whitespace without blank lines
            continue;
        }

        // Stop if we find a node that doesn't belong to the group
        if node.kind() != first.kind() || unless(node) {
            break;
        }

        visited.insert(node);
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

fn contiguous_range_for_comment<'a>(
    first: &'a SyntaxNode,
    visited: &mut FxHashSet<&'a SyntaxNode>,
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
        let (ranges, text) = extract_ranges(text, "fold");
        let file = SourceFile::parse(&text);
        let folds = folding_ranges(&file);

        assert_eq!(
            folds.len(),
            ranges.len(),
            "The amount of folds is different than the expected amount"
        );
        assert_eq!(
            folds.len(),
            fold_kinds.len(),
            "The amount of fold kinds is different than the expected amount"
        );
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
<fold>// Hello
// this is a multiline
// comment
//</fold>

// But this is not

fn main() <fold>{
    <fold>// We should
    // also
    // fold
    // this one.</fold>
    <fold>//! But this one is different
    //! because it has another flavor</fold>
    <fold>/* As does this
    multiline comment */</fold>
}</fold>"#;

        let fold_kinds = &[
            FoldKind::Comment,
            FoldKind::Block,
            FoldKind::Comment,
            FoldKind::Comment,
            FoldKind::Comment,
        ];
        do_check(text, fold_kinds);
    }

    #[test]
    fn test_fold_imports() {
        let text = r#"
<fold>use std::<fold>{
    str,
    vec,
    io as iop
}</fold>;</fold>

fn main() <fold>{
}</fold>"#;

        let folds = &[FoldKind::Imports, FoldKind::Block, FoldKind::Block];
        do_check(text, folds);
    }

    #[test]
    fn test_fold_mods() {
        let text = r#"

pub mod foo;
<fold>mod after_pub;
mod after_pub_next;</fold>

<fold>mod before_pub;
mod before_pub_next;</fold>
pub mod bar;

mod not_folding_single;
pub mod foobar;
pub not_folding_single_next;

<fold>#[cfg(test)]
mod with_attribute;
mod with_attribute_next;</fold>

fn main() <fold>{
}</fold>"#;

        let folds = &[
            FoldKind::Mods,
            FoldKind::Mods,
            FoldKind::Mods,
            FoldKind::Block,
        ];
        do_check(text, folds);
    }

    #[test]
    fn test_fold_import_groups() {
        let text = r#"
<fold>use std::str;
use std::vec;
use std::io as iop;</fold>

<fold>use std::mem;
use std::f64;</fold>

use std::collections::HashMap;
// Some random comment
use std::collections::VecDeque;

fn main() <fold>{
}</fold>"#;

        let folds = &[FoldKind::Imports, FoldKind::Imports, FoldKind::Block];
        do_check(text, folds);
    }

    #[test]
    fn test_fold_import_and_groups() {
        let text = r#"
<fold>use std::str;
use std::vec;
use std::io as iop;</fold>

<fold>use std::mem;
use std::f64;</fold>

<fold>use std::collections::<fold>{
    HashMap,
    VecDeque,
}</fold>;</fold>
// Some random comment

fn main() <fold>{
}</fold>"#;

        let folds = &[
            FoldKind::Imports,
            FoldKind::Imports,
            FoldKind::Imports,
            FoldKind::Block,
            FoldKind::Block,
        ];
        do_check(text, folds);
    }

    #[test]
    fn test_folds_macros() {
        let text = r#"
macro_rules! foo <fold>{
    ($($tt:tt)*) => { $($tt)* }
}</fold>
"#;

        let folds = &[FoldKind::Block];
        do_check(text, folds);
    }
}
