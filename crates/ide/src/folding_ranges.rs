//! FIXME: write short doc here

use rustc_hash::FxHashSet;

use syntax::{
    ast::{self, AstNode, AstToken, VisibilityOwner},
    Direction, NodeOrToken, SourceFile,
    SyntaxKind::{self, *},
    SyntaxNode, TextRange, TextSize,
};

#[derive(Debug, PartialEq, Eq)]
pub enum FoldKind {
    Comment,
    Imports,
    Mods,
    Block,
    ArgList,
    Region,
    Consts,
    Statics,
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
    let mut visited_consts = FxHashSet::default();
    let mut visited_statics = FxHashSet::default();
    // regions can be nested, here is a LIFO buffer
    let mut regions_starts: Vec<TextSize> = vec![];

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
                        // regions are not real comments
                        if comment.text().trim().starts_with("// region:") {
                            regions_starts.push(comment.syntax().text_range().start());
                        } else if comment.text().trim().starts_with("// endregion") {
                            if let Some(region) = regions_starts.pop() {
                                res.push(Fold {
                                    range: TextRange::new(
                                        region,
                                        comment.syntax().text_range().end(),
                                    ),
                                    kind: FoldKind::Region,
                                })
                            }
                        } else {
                            if let Some(range) =
                                contiguous_range_for_comment(comment, &mut visited_comments)
                            {
                                res.push(Fold { range, kind: FoldKind::Comment })
                            }
                        }
                    }
                }
            }
            NodeOrToken::Node(node) => {
                // Fold groups of imports
                if node.kind() == USE && !visited_imports.contains(&node) {
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

                // Fold groups of consts
                if node.kind() == CONST && !visited_consts.contains(&node) {
                    if let Some(range) = contiguous_range_for_group(&node, &mut visited_consts) {
                        res.push(Fold { range, kind: FoldKind::Consts })
                    }
                }
                // Fold groups of consts
                if node.kind() == STATIC && !visited_statics.contains(&node) {
                    if let Some(range) = contiguous_range_for_group(&node, &mut visited_statics) {
                        res.push(Fold { range, kind: FoldKind::Statics })
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
        ARG_LIST | PARAM_LIST => Some(FoldKind::ArgList),
        ASSOC_ITEM_LIST
        | RECORD_FIELD_LIST
        | RECORD_PAT_FIELD_LIST
        | RECORD_EXPR_FIELD_LIST
        | ITEM_LIST
        | EXTERN_ITEM_LIST
        | USE_TREE_LIST
        | BLOCK_EXPR
        | MATCH_ARM_LIST
        | VARIANT_LIST
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
        Some(TextRange::new(first.text_range().start(), last.text_range().end()))
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
                        // regions are not real comments
                        if c.text().trim().starts_with("// region:")
                            || c.text().trim().starts_with("// endregion")
                        {
                            break;
                        } else {
                            visited.insert(c.clone());
                            last = c;
                            continue;
                        }
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
        Some(TextRange::new(first.syntax().text_range().start(), last.syntax().text_range().end()))
    } else {
        // The group consists of only one element, therefore it cannot be folded
        None
    }
}

#[cfg(test)]
mod tests {
    use test_utils::extract_tags;

    use super::*;

    fn check(ra_fixture: &str) {
        let (ranges, text) = extract_tags(ra_fixture, "fold");

        let parse = SourceFile::parse(&text);
        let folds = folding_ranges(&parse.tree());
        assert_eq!(
            folds.len(),
            ranges.len(),
            "The amount of folds is different than the expected amount"
        );

        for (fold, (range, attr)) in folds.iter().zip(ranges.into_iter()) {
            assert_eq!(fold.range.start(), range.start());
            assert_eq!(fold.range.end(), range.end());

            let kind = match fold.kind {
                FoldKind::Comment => "comment",
                FoldKind::Imports => "imports",
                FoldKind::Mods => "mods",
                FoldKind::Block => "block",
                FoldKind::ArgList => "arglist",
                FoldKind::Region => "region",
                FoldKind::Consts => "consts",
                FoldKind::Statics => "statics",
            };
            assert_eq!(kind, &attr.unwrap());
        }
    }

    #[test]
    fn test_fold_comments() {
        check(
            r#"
<fold comment>// Hello
// this is a multiline
// comment
//</fold>

// But this is not

fn main() <fold block>{
    <fold comment>// We should
    // also
    // fold
    // this one.</fold>
    <fold comment>//! But this one is different
    //! because it has another flavor</fold>
    <fold comment>/* As does this
    multiline comment */</fold>
}</fold>"#,
        );
    }

    #[test]
    fn test_fold_imports() {
        check(
            r#"
use std::<fold block>{
    str,
    vec,
    io as iop
}</fold>;

fn main() <fold block>{
}</fold>"#,
        );
    }

    #[test]
    fn test_fold_mods() {
        check(
            r#"

pub mod foo;
<fold mods>mod after_pub;
mod after_pub_next;</fold>

<fold mods>mod before_pub;
mod before_pub_next;</fold>
pub mod bar;

mod not_folding_single;
pub mod foobar;
pub not_folding_single_next;

<fold mods>#[cfg(test)]
mod with_attribute;
mod with_attribute_next;</fold>

fn main() <fold block>{
}</fold>"#,
        );
    }

    #[test]
    fn test_fold_import_groups() {
        check(
            r#"
<fold imports>use std::str;
use std::vec;
use std::io as iop;</fold>

<fold imports>use std::mem;
use std::f64;</fold>

<fold imports>use std::collections::HashMap;
// Some random comment
use std::collections::VecDeque;</fold>

fn main() <fold block>{
}</fold>"#,
        );
    }

    #[test]
    fn test_fold_import_and_groups() {
        check(
            r#"
<fold imports>use std::str;
use std::vec;
use std::io as iop;</fold>

<fold imports>use std::mem;
use std::f64;</fold>

use std::collections::<fold block>{
    HashMap,
    VecDeque,
}</fold>;
// Some random comment

fn main() <fold block>{
}</fold>"#,
        );
    }

    #[test]
    fn test_folds_structs() {
        check(
            r#"
struct Foo <fold block>{
}</fold>
"#,
        );
    }

    #[test]
    fn test_folds_traits() {
        check(
            r#"
trait Foo <fold block>{
}</fold>
"#,
        );
    }

    #[test]
    fn test_folds_macros() {
        check(
            r#"
macro_rules! foo <fold block>{
    ($($tt:tt)*) => { $($tt)* }
}</fold>
"#,
        );
    }

    #[test]
    fn test_fold_match_arms() {
        check(
            r#"
fn main() <fold block>{
    match 0 <fold block>{
        0 => 0,
        _ => 1,
    }</fold>
}</fold>
"#,
        );
    }

    #[test]
    fn fold_big_calls() {
        check(
            r#"
fn main() <fold block>{
    frobnicate<fold arglist>(
        1,
        2,
        3,
    )</fold>
}</fold>
"#,
        )
    }

    #[test]
    fn fold_record_literals() {
        check(
            r#"
const _: S = S <fold block>{

}</fold>;
"#,
        )
    }

    #[test]
    fn fold_multiline_params() {
        check(
            r#"
fn foo<fold arglist>(
    x: i32,
    y: String,
)</fold> {}
"#,
        )
    }

    #[test]
    fn fold_region() {
        check(
            r#"
// 1. some normal comment
<fold region>// region: test
// 2. some normal comment
calling_function(x,y);
// endregion: test</fold>
"#,
        )
    }

    #[test]
    fn fold_consecutive_const() {
        check(
            r#"
<fold consts>const FIRST_CONST: &str = "first";
const SECOND_CONST: &str = "second";</fold>
            "#,
        )
    }

    #[test]
    fn fold_consecutive_static() {
        check(
            r#"
<fold statics>static FIRST_STATIC: &str = "first";
static SECOND_STATIC: &str = "second";</fold>
            "#,
        )
    }
}
