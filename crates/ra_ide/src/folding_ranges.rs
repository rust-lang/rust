//! FIXME: write short doc here

use rustc_hash::FxHashSet;

use ra_syntax::{
    ast::{self, AstNode, AstToken, VisibilityOwner},
    Direction, NodeOrToken, SourceFile,
    SyntaxKind::{self, *},
    SyntaxNode, TextRange,
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

pub(crate) fn folding_ranges(file: &SourceFile) -> Vec<Fold> {
    let mut res = vec![];
    let mut visited_comments = FxHashSet::default();
    let mut visited_imports = FxHashSet::default();
    let mut visited_mods = FxHashSet::default();

    for element in file.syntax().descendants_with_tokens() {
        // Fold items that span multiple lines
        if let Some(kind) = fold_kind(element.kind()) {
            let is_multiline = match &element {
                NodeOrToken::Node(node) => node.text().contains_char('\n'),
                NodeOrToken::Token(token) => token.text().contains('\n'),
            };
            if is_multiline {
                res.push(Fold { range: element.text_range(), kind });
                continue;
            }
        }

        match element {
            NodeOrToken::Token(token) => {
                // Fold groups of comments
                if let Some(comment) = ast::Comment::cast(token) {
                    if !visited_comments.contains(&comment) {
                        if let Some(range) =
                            contiguous_range_for_comment(comment, &mut visited_comments)
                        {
                            res.push(Fold { range, kind: FoldKind::Comment })
                        }
                    }
                }
            }
            NodeOrToken::Node(node) => {
                // Fold groups of imports
                if node.kind() == USE_ITEM && !visited_imports.contains(&node) {
                    if let Some(range) = contiguous_range_for_group(&node, &mut visited_imports) {
                        res.push(Fold { range, kind: FoldKind::Imports })
                    }
                }

                // Fold groups of mods
                if node.kind() == MODULE && !has_visibility(&node) && !visited_mods.contains(&node)
                {
                    if let Some(range) =
                        contiguous_range_for_group_unless(&node, has_visibility, &mut visited_mods)
                    {
                        res.push(Fold { range, kind: FoldKind::Mods })
                    }
                }
            }
        }
    }

    res
}

fn fold_kind(kind: SyntaxKind) -> Option<FoldKind> {
    match kind {
        COMMENT => Some(FoldKind::Comment),
        USE_ITEM => Some(FoldKind::Imports),
        RECORD_FIELD_DEF_LIST
        | RECORD_FIELD_PAT_LIST
        | ITEM_LIST
        | EXTERN_ITEM_LIST
        | USE_TREE_LIST
        | BLOCK
        | MATCH_ARM_LIST
        | ENUM_VARIANT_LIST
        | TOKEN_TREE => Some(FoldKind::Block),
        _ => None,
    }
}

fn has_visibility(node: &SyntaxNode) -> bool {
    ast::Module::cast(node.clone()).and_then(|m| m.visibility()).is_some()
}

fn contiguous_range_for_group(
    first: &SyntaxNode,
    visited: &mut FxHashSet<SyntaxNode>,
) -> Option<TextRange> {
    contiguous_range_for_group_unless(first, |_| false, visited)
}

fn contiguous_range_for_group_unless(
    first: &SyntaxNode,
    unless: impl Fn(&SyntaxNode) -> bool,
    visited: &mut FxHashSet<SyntaxNode>,
) -> Option<TextRange> {
    visited.insert(first.clone());

    let mut last = first.clone();
    for element in first.siblings_with_tokens(Direction::Next) {
        let node = match element {
            NodeOrToken::Token(token) => {
                if let Some(ws) = ast::Whitespace::cast(token) {
                    if !ws.spans_multiple_lines() {
                        // Ignore whitespace without blank lines
                        continue;
                    }
                }
                // There is a blank line or another token, which means that the
                // group ends here
                break;
            }
            NodeOrToken::Node(node) => node,
        };

        // Stop if we find a node that doesn't belong to the group
        if node.kind() != first.kind() || unless(&node) {
            break;
        }

        visited.insert(node.clone());
        last = node;
    }

    if first != &last {
        Some(TextRange::from_to(first.text_range().start(), last.text_range().end()))
    } else {
        // The group consists of only one element, therefore it cannot be folded
        None
    }
}

fn contiguous_range_for_comment(
    first: ast::Comment,
    visited: &mut FxHashSet<ast::Comment>,
) -> Option<TextRange> {
    visited.insert(first.clone());

    // Only fold comments of the same flavor
    let group_kind = first.kind();
    if !group_kind.shape.is_line() {
        return None;
    }

    let mut last = first.clone();
    for element in first.syntax().siblings_with_tokens(Direction::Next) {
        match element {
            NodeOrToken::Token(token) => {
                if let Some(ws) = ast::Whitespace::cast(token.clone()) {
                    if !ws.spans_multiple_lines() {
                        // Ignore whitespace without blank lines
                        continue;
                    }
                }
                if let Some(c) = ast::Comment::cast(token) {
                    if c.kind() == group_kind {
                        visited.insert(c.clone());
                        last = c;
                        continue;
                    }
                }
                // The comment group ends because either:
                // * An element of a different kind was reached
                // * A comment of a different flavor was reached
                break;
            }
            NodeOrToken::Node(_) => break,
        };
    }

    if first != last {
        Some(TextRange::from_to(
            first.syntax().text_range().start(),
            last.syntax().text_range().end(),
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
        let parse = SourceFile::parse(&text);
        let folds = folding_ranges(&parse.tree());

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
        for ((fold, range), fold_kind) in
            folds.iter().zip(ranges.into_iter()).zip(fold_kinds.iter())
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

        let folds = &[FoldKind::Mods, FoldKind::Mods, FoldKind::Mods, FoldKind::Block];
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

    #[test]
    fn test_fold_match_arms() {
        let text = r#"
fn main() <fold>{
    match 0 <fold>{
        0 => 0,
        _ => 1,
    }</fold>
}</fold>"#;

        let folds = &[FoldKind::Block, FoldKind::Block];
        do_check(text, folds);
    }
}
