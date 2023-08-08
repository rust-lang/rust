use std::iter::successors;

use hir::Semantics;
use ide_db::RootDatabase;
use syntax::{
    algo::{self, skip_trivia_token},
    ast::{self, AstNode, AstToken},
    Direction, NodeOrToken,
    SyntaxKind::{self, *},
    SyntaxNode, SyntaxToken, TextRange, TextSize, TokenAtOffset, T,
};

use crate::FileRange;

// Feature: Expand and Shrink Selection
//
// Extends or shrinks the current selection to the encompassing syntactic construct
// (expression, statement, item, module, etc). It works with multiple cursors.
//
// This is a standard LSP feature and not a protocol extension.
//
// |===
// | Editor  | Shortcut
//
// | VS Code | kbd:[Alt+Shift+→], kbd:[Alt+Shift+←]
// |===
//
// image::https://user-images.githubusercontent.com/48062697/113020651-b42fc800-917a-11eb-8a4f-cf1a07859fac.gif[]
pub(crate) fn extend_selection(db: &RootDatabase, frange: FileRange) -> TextRange {
    let sema = Semantics::new(db);
    let src = sema.parse(frange.file_id);
    try_extend_selection(&sema, src.syntax(), frange).unwrap_or(frange.range)
}

fn try_extend_selection(
    sema: &Semantics<'_, RootDatabase>,
    root: &SyntaxNode,
    frange: FileRange,
) -> Option<TextRange> {
    let range = frange.range;

    let string_kinds = [COMMENT, STRING, BYTE_STRING, C_STRING];
    let list_kinds = [
        RECORD_PAT_FIELD_LIST,
        MATCH_ARM_LIST,
        RECORD_FIELD_LIST,
        TUPLE_FIELD_LIST,
        RECORD_EXPR_FIELD_LIST,
        VARIANT_LIST,
        USE_TREE_LIST,
        GENERIC_PARAM_LIST,
        GENERIC_ARG_LIST,
        TYPE_BOUND_LIST,
        PARAM_LIST,
        ARG_LIST,
        ARRAY_EXPR,
        TUPLE_EXPR,
        TUPLE_TYPE,
        TUPLE_PAT,
        WHERE_CLAUSE,
    ];

    if range.is_empty() {
        let offset = range.start();
        let mut leaves = root.token_at_offset(offset);
        if leaves.clone().all(|it| it.kind() == WHITESPACE) {
            return Some(extend_ws(root, leaves.next()?, offset));
        }
        let leaf_range = match leaves {
            TokenAtOffset::None => return None,
            TokenAtOffset::Single(l) => {
                if string_kinds.contains(&l.kind()) {
                    extend_single_word_in_comment_or_string(&l, offset)
                        .unwrap_or_else(|| l.text_range())
                } else {
                    l.text_range()
                }
            }
            TokenAtOffset::Between(l, r) => pick_best(l, r).text_range(),
        };
        return Some(leaf_range);
    };
    let node = match root.covering_element(range) {
        NodeOrToken::Token(token) => {
            if token.text_range() != range {
                return Some(token.text_range());
            }
            if let Some(comment) = ast::Comment::cast(token.clone()) {
                if let Some(range) = extend_comments(comment) {
                    return Some(range);
                }
            }
            token.parent()?
        }
        NodeOrToken::Node(node) => node,
    };

    // if we are in single token_tree, we maybe live in macro or attr
    if node.kind() == TOKEN_TREE {
        if let Some(macro_call) = node.ancestors().find_map(ast::MacroCall::cast) {
            if let Some(range) = extend_tokens_from_range(sema, macro_call, range) {
                return Some(range);
            }
        }
    }

    if node.text_range() != range {
        return Some(node.text_range());
    }

    let node = shallowest_node(&node);

    if node.parent().map(|n| list_kinds.contains(&n.kind())) == Some(true) {
        if let Some(range) = extend_list_item(&node) {
            return Some(range);
        }
    }

    node.parent().map(|it| it.text_range())
}

fn extend_tokens_from_range(
    sema: &Semantics<'_, RootDatabase>,
    macro_call: ast::MacroCall,
    original_range: TextRange,
) -> Option<TextRange> {
    let src = macro_call.syntax().covering_element(original_range);
    let (first_token, last_token) = match src {
        NodeOrToken::Node(it) => (it.first_token()?, it.last_token()?),
        NodeOrToken::Token(it) => (it.clone(), it),
    };

    let mut first_token = skip_trivia_token(first_token, Direction::Next)?;
    let mut last_token = skip_trivia_token(last_token, Direction::Prev)?;

    while !original_range.contains_range(first_token.text_range()) {
        first_token = skip_trivia_token(first_token.next_token()?, Direction::Next)?;
    }
    while !original_range.contains_range(last_token.text_range()) {
        last_token = skip_trivia_token(last_token.prev_token()?, Direction::Prev)?;
    }

    // compute original mapped token range
    let extended = {
        let fst_expanded = sema.descend_into_macros_single(first_token.clone());
        let lst_expanded = sema.descend_into_macros_single(last_token.clone());
        let mut lca =
            algo::least_common_ancestor(&fst_expanded.parent()?, &lst_expanded.parent()?)?;
        lca = shallowest_node(&lca);
        if lca.first_token() == Some(fst_expanded) && lca.last_token() == Some(lst_expanded) {
            lca = lca.parent()?;
        }
        lca
    };

    // Compute parent node range
    let validate = |token: &SyntaxToken| -> bool {
        let expanded = sema.descend_into_macros_single(token.clone());
        let parent = match expanded.parent() {
            Some(it) => it,
            None => return false,
        };
        algo::least_common_ancestor(&extended, &parent).as_ref() == Some(&extended)
    };

    // Find the first and last text range under expanded parent
    let first = successors(Some(first_token), |token| {
        let token = token.prev_token()?;
        skip_trivia_token(token, Direction::Prev)
    })
    .take_while(validate)
    .last()?;

    let last = successors(Some(last_token), |token| {
        let token = token.next_token()?;
        skip_trivia_token(token, Direction::Next)
    })
    .take_while(validate)
    .last()?;

    let range = first.text_range().cover(last.text_range());
    if range.contains_range(original_range) && original_range != range {
        Some(range)
    } else {
        None
    }
}

/// Find the shallowest node with same range, which allows us to traverse siblings.
fn shallowest_node(node: &SyntaxNode) -> SyntaxNode {
    node.ancestors().take_while(|n| n.text_range() == node.text_range()).last().unwrap()
}

fn extend_single_word_in_comment_or_string(
    leaf: &SyntaxToken,
    offset: TextSize,
) -> Option<TextRange> {
    let text: &str = leaf.text();
    let cursor_position: u32 = (offset - leaf.text_range().start()).into();

    let (before, after) = text.split_at(cursor_position as usize);

    fn non_word_char(c: char) -> bool {
        !(c.is_alphanumeric() || c == '_')
    }

    let start_idx = before.rfind(non_word_char)? as u32;
    let end_idx = after.find(non_word_char).unwrap_or(after.len()) as u32;

    let from: TextSize = (start_idx + 1).into();
    let to: TextSize = (cursor_position + end_idx).into();

    let range = TextRange::new(from, to);
    if range.is_empty() {
        None
    } else {
        Some(range + leaf.text_range().start())
    }
}

fn extend_ws(root: &SyntaxNode, ws: SyntaxToken, offset: TextSize) -> TextRange {
    let ws_text = ws.text();
    let suffix = TextRange::new(offset, ws.text_range().end()) - ws.text_range().start();
    let prefix = TextRange::new(ws.text_range().start(), offset) - ws.text_range().start();
    let ws_suffix = &ws_text[suffix];
    let ws_prefix = &ws_text[prefix];
    if ws_text.contains('\n') && !ws_suffix.contains('\n') {
        if let Some(node) = ws.next_sibling_or_token() {
            let start = match ws_prefix.rfind('\n') {
                Some(idx) => ws.text_range().start() + TextSize::from((idx + 1) as u32),
                None => node.text_range().start(),
            };
            let end = if root.text().char_at(node.text_range().end()) == Some('\n') {
                node.text_range().end() + TextSize::of('\n')
            } else {
                node.text_range().end()
            };
            return TextRange::new(start, end);
        }
    }
    ws.text_range()
}

fn pick_best(l: SyntaxToken, r: SyntaxToken) -> SyntaxToken {
    return if priority(&r) > priority(&l) { r } else { l };
    fn priority(n: &SyntaxToken) -> usize {
        match n.kind() {
            WHITESPACE => 0,
            IDENT | T![self] | T![super] | T![crate] | T![Self] | LIFETIME_IDENT => 2,
            _ => 1,
        }
    }
}

/// Extend list item selection to include nearby delimiter and whitespace.
fn extend_list_item(node: &SyntaxNode) -> Option<TextRange> {
    fn is_single_line_ws(node: &SyntaxToken) -> bool {
        node.kind() == WHITESPACE && !node.text().contains('\n')
    }

    fn nearby_delimiter(
        delimiter_kind: SyntaxKind,
        node: &SyntaxNode,
        dir: Direction,
    ) -> Option<SyntaxToken> {
        node.siblings_with_tokens(dir)
            .skip(1)
            .find(|node| match node {
                NodeOrToken::Node(_) => true,
                NodeOrToken::Token(it) => !is_single_line_ws(it),
            })
            .and_then(|it| it.into_token())
            .filter(|node| node.kind() == delimiter_kind)
    }

    let delimiter = match node.kind() {
        TYPE_BOUND => T![+],
        _ => T![,],
    };

    if let Some(delimiter_node) = nearby_delimiter(delimiter, node, Direction::Next) {
        // Include any following whitespace when delimiter is after list item.
        let final_node = delimiter_node
            .next_sibling_or_token()
            .and_then(|it| it.into_token())
            .filter(is_single_line_ws)
            .unwrap_or(delimiter_node);

        return Some(TextRange::new(node.text_range().start(), final_node.text_range().end()));
    }
    if let Some(delimiter_node) = nearby_delimiter(delimiter, node, Direction::Prev) {
        return Some(TextRange::new(delimiter_node.text_range().start(), node.text_range().end()));
    }

    None
}

fn extend_comments(comment: ast::Comment) -> Option<TextRange> {
    let prev = adj_comments(&comment, Direction::Prev);
    let next = adj_comments(&comment, Direction::Next);
    if prev != next {
        Some(TextRange::new(prev.syntax().text_range().start(), next.syntax().text_range().end()))
    } else {
        None
    }
}

fn adj_comments(comment: &ast::Comment, dir: Direction) -> ast::Comment {
    let mut res = comment.clone();
    for element in comment.syntax().siblings_with_tokens(dir) {
        let token = match element.as_token() {
            None => break,
            Some(token) => token,
        };
        if let Some(c) = ast::Comment::cast(token.clone()) {
            res = c
        } else if token.kind() != WHITESPACE || token.text().contains("\n\n") {
            break;
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use crate::fixture;

    use super::*;

    fn do_check(before: &str, afters: &[&str]) {
        let (analysis, position) = fixture::position(before);
        let before = analysis.file_text(position.file_id).unwrap();
        let range = TextRange::empty(position.offset);
        let mut frange = FileRange { file_id: position.file_id, range };

        for &after in afters {
            frange.range = analysis.extend_selection(frange).unwrap();
            let actual = &before[frange.range];
            assert_eq!(after, actual);
        }
    }

    #[test]
    fn test_extend_selection_arith() {
        do_check(r#"fn foo() { $01 + 1 }"#, &["1", "1 + 1", "{ 1 + 1 }"]);
    }

    #[test]
    fn test_extend_selection_list() {
        do_check(r#"fn foo($0x: i32) {}"#, &["x", "x: i32"]);
        do_check(r#"fn foo($0x: i32, y: i32) {}"#, &["x", "x: i32", "x: i32, "]);
        do_check(r#"fn foo($0x: i32,y: i32) {}"#, &["x", "x: i32", "x: i32,", "(x: i32,y: i32)"]);
        do_check(r#"fn foo(x: i32, $0y: i32) {}"#, &["y", "y: i32", ", y: i32"]);
        do_check(r#"fn foo(x: i32, $0y: i32, ) {}"#, &["y", "y: i32", "y: i32, "]);
        do_check(r#"fn foo(x: i32,$0y: i32) {}"#, &["y", "y: i32", ",y: i32"]);

        do_check(r#"const FOO: [usize; 2] = [ 22$0 , 33];"#, &["22", "22 , "]);
        do_check(r#"const FOO: [usize; 2] = [ 22 , 33$0];"#, &["33", ", 33"]);
        do_check(r#"const FOO: [usize; 2] = [ 22 , 33$0 ,];"#, &["33", "33 ,", "[ 22 , 33 ,]"]);

        do_check(r#"fn main() { (1, 2$0) }"#, &["2", ", 2", "(1, 2)"]);

        do_check(
            r#"
const FOO: [usize; 2] = [
    22,
    $033,
]"#,
            &["33", "33,"],
        );

        do_check(
            r#"
const FOO: [usize; 2] = [
    22
    , 33$0,
]"#,
            &["33", "33,"],
        );
    }

    #[test]
    fn test_extend_selection_start_of_the_line() {
        do_check(
            r#"
impl S {
$0    fn foo() {

    }
}"#,
            &["    fn foo() {\n\n    }\n"],
        );
    }

    #[test]
    fn test_extend_selection_doc_comments() {
        do_check(
            r#"
struct A;

/// bla
/// bla
struct B {
    $0
}
            "#,
            &["\n    \n", "{\n    \n}", "/// bla\n/// bla\nstruct B {\n    \n}"],
        )
    }

    #[test]
    fn test_extend_selection_comments() {
        do_check(
            r#"
fn bar(){}

// fn foo() {
// 1 + $01
// }

// fn foo(){}
    "#,
            &["1", "// 1 + 1", "// fn foo() {\n// 1 + 1\n// }"],
        );

        do_check(
            r#"
// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum Direction {
//  $0   Next,
//     Prev
// }
"#,
            &[
                "//     Next,",
                "// #[derive(Debug, Clone, Copy, PartialEq, Eq)]\n// pub enum Direction {\n//     Next,\n//     Prev\n// }",
            ],
        );

        do_check(
            r#"
/*
foo
_bar1$0*/
"#,
            &["_bar1", "/*\nfoo\n_bar1*/"],
        );

        do_check(r#"//!$0foo_2 bar"#, &["foo_2", "//!foo_2 bar"]);

        do_check(r#"/$0/foo bar"#, &["//foo bar"]);
    }

    #[test]
    fn test_extend_selection_prefer_idents() {
        do_check(
            r#"
fn main() { foo$0+bar;}
"#,
            &["foo", "foo+bar"],
        );
        do_check(
            r#"
fn main() { foo+$0bar;}
"#,
            &["bar", "foo+bar"],
        );
    }

    #[test]
    fn test_extend_selection_prefer_lifetimes() {
        do_check(r#"fn foo<$0'a>() {}"#, &["'a", "<'a>"]);
        do_check(r#"fn foo<'a$0>() {}"#, &["'a", "<'a>"]);
    }

    #[test]
    fn test_extend_selection_select_first_word() {
        do_check(r#"// foo bar b$0az quxx"#, &["baz", "// foo bar baz quxx"]);
        do_check(
            r#"
impl S {
fn foo() {
// hel$0lo world
}
}
"#,
            &["hello", "// hello world"],
        );
    }

    #[test]
    fn test_extend_selection_string() {
        do_check(
            r#"
fn bar(){}

" fn f$0oo() {"
"#,
            &["foo", "\" fn foo() {\""],
        );
    }

    #[test]
    fn test_extend_trait_bounds_list_in_where_clause() {
        do_check(
            r#"
fn foo<R>()
    where
        R: req::Request + 'static,
        R::Params: DeserializeOwned$0 + panic::UnwindSafe + 'static,
        R::Result: Serialize + 'static,
"#,
            &[
                "DeserializeOwned",
                "DeserializeOwned + ",
                "DeserializeOwned + panic::UnwindSafe + 'static",
                "R::Params: DeserializeOwned + panic::UnwindSafe + 'static",
                "R::Params: DeserializeOwned + panic::UnwindSafe + 'static,",
            ],
        );
        do_check(r#"fn foo<T>() where T: $0Copy"#, &["Copy"]);
        do_check(r#"fn foo<T>() where T: $0Copy + Display"#, &["Copy", "Copy + "]);
        do_check(r#"fn foo<T>() where T: $0Copy +Display"#, &["Copy", "Copy +"]);
        do_check(r#"fn foo<T>() where T: $0Copy+Display"#, &["Copy", "Copy+"]);
        do_check(r#"fn foo<T>() where T: Copy + $0Display"#, &["Display", "+ Display"]);
        do_check(r#"fn foo<T>() where T: Copy + $0Display + Sync"#, &["Display", "Display + "]);
        do_check(r#"fn foo<T>() where T: Copy +$0Display"#, &["Display", "+Display"]);
    }

    #[test]
    fn test_extend_trait_bounds_list_inline() {
        do_check(r#"fn foo<T: $0Copy>() {}"#, &["Copy"]);
        do_check(r#"fn foo<T: $0Copy + Display>() {}"#, &["Copy", "Copy + "]);
        do_check(r#"fn foo<T: $0Copy +Display>() {}"#, &["Copy", "Copy +"]);
        do_check(r#"fn foo<T: $0Copy+Display>() {}"#, &["Copy", "Copy+"]);
        do_check(r#"fn foo<T: Copy + $0Display>() {}"#, &["Display", "+ Display"]);
        do_check(r#"fn foo<T: Copy + $0Display + Sync>() {}"#, &["Display", "Display + "]);
        do_check(r#"fn foo<T: Copy +$0Display>() {}"#, &["Display", "+Display"]);
        do_check(
            r#"fn foo<T: Copy$0 + Display, U: Copy>() {}"#,
            &[
                "Copy",
                "Copy + ",
                "Copy + Display",
                "T: Copy + Display",
                "T: Copy + Display, ",
                "<T: Copy + Display, U: Copy>",
            ],
        );
    }

    #[test]
    fn test_extend_selection_on_tuple_in_type() {
        do_check(
            r#"fn main() { let _: (krate, $0_crate_def_map, module_id) = (); }"#,
            &["_crate_def_map", "_crate_def_map, ", "(krate, _crate_def_map, module_id)"],
        );
        // white space variations
        do_check(
            r#"fn main() { let _: (krate,$0_crate_def_map,module_id) = (); }"#,
            &["_crate_def_map", "_crate_def_map,", "(krate,_crate_def_map,module_id)"],
        );
        do_check(
            r#"
fn main() { let _: (
    krate,
    _crate$0_def_map,
    module_id
) = (); }"#,
            &[
                "_crate_def_map",
                "_crate_def_map,",
                "(\n    krate,\n    _crate_def_map,\n    module_id\n)",
            ],
        );
    }

    #[test]
    fn test_extend_selection_on_tuple_in_rvalue() {
        do_check(
            r#"fn main() { let var = (krate, _crate_def_map$0, module_id); }"#,
            &["_crate_def_map", "_crate_def_map, ", "(krate, _crate_def_map, module_id)"],
        );
        // white space variations
        do_check(
            r#"fn main() { let var = (krate,_crate$0_def_map,module_id); }"#,
            &["_crate_def_map", "_crate_def_map,", "(krate,_crate_def_map,module_id)"],
        );
        do_check(
            r#"
fn main() { let var = (
    krate,
    _crate_def_map$0,
    module_id
); }"#,
            &[
                "_crate_def_map",
                "_crate_def_map,",
                "(\n    krate,\n    _crate_def_map,\n    module_id\n)",
            ],
        );
    }

    #[test]
    fn test_extend_selection_on_tuple_pat() {
        do_check(
            r#"fn main() { let (krate, _crate_def_map$0, module_id) = var; }"#,
            &["_crate_def_map", "_crate_def_map, ", "(krate, _crate_def_map, module_id)"],
        );
        // white space variations
        do_check(
            r#"fn main() { let (krate,_crate$0_def_map,module_id) = var; }"#,
            &["_crate_def_map", "_crate_def_map,", "(krate,_crate_def_map,module_id)"],
        );
        do_check(
            r#"
fn main() { let (
    krate,
    _crate_def_map$0,
    module_id
) = var; }"#,
            &[
                "_crate_def_map",
                "_crate_def_map,",
                "(\n    krate,\n    _crate_def_map,\n    module_id\n)",
            ],
        );
    }

    #[test]
    fn extend_selection_inside_macros() {
        do_check(
            r#"macro_rules! foo { ($item:item) => {$item} }
                foo!{fn hello(na$0me:usize){}}"#,
            &[
                "name",
                "name:usize",
                "(name:usize)",
                "fn hello(name:usize){}",
                "{fn hello(name:usize){}}",
                "foo!{fn hello(name:usize){}}",
            ],
        );
    }

    #[test]
    fn extend_selection_inside_recur_macros() {
        do_check(
            r#" macro_rules! foo2 { ($item:item) => {$item} }
                macro_rules! foo { ($item:item) => {foo2!($item);} }
                foo!{fn hello(na$0me:usize){}}"#,
            &[
                "name",
                "name:usize",
                "(name:usize)",
                "fn hello(name:usize){}",
                "{fn hello(name:usize){}}",
                "foo!{fn hello(name:usize){}}",
            ],
        );
    }
}
