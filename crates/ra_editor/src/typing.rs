use std::mem;

use itertools::Itertools;
use ra_syntax::{
    algo::{find_covering_node, find_leaf_at_offset, LeafAtOffset},
    ast,
    text_utils::intersect,
    AstNode, Direction, SourceFile, SyntaxKind,
    SyntaxKind::*,
    SyntaxNode, TextRange, TextUnit,
};
use ra_text_edit::text_utils::contains_offset_nonstrict;

use crate::{find_node_at_offset, LocalEdit, TextEditBuilder};

pub fn join_lines(file: &SourceFile, range: TextRange) -> LocalEdit {
    let range = if range.is_empty() {
        let syntax = file.syntax();
        let text = syntax.text().slice(range.start()..);
        let pos = match text.find('\n') {
            None => {
                return LocalEdit {
                    label: "join lines".to_string(),
                    edit: TextEditBuilder::default().finish(),
                    cursor_position: None,
                };
            }
            Some(pos) => pos,
        };
        TextRange::offset_len(range.start() + pos, TextUnit::of_char('\n'))
    } else {
        range
    };

    let node = find_covering_node(file.syntax(), range);
    let mut edit = TextEditBuilder::default();
    for node in node.descendants() {
        let text = match node.leaf_text() {
            Some(text) => text,
            None => continue,
        };
        let range = match intersect(range, node.range()) {
            Some(range) => range,
            None => continue,
        } - node.range().start();
        for (pos, _) in text[range].bytes().enumerate().filter(|&(_, b)| b == b'\n') {
            let pos: TextUnit = (pos as u32).into();
            let off = node.range().start() + range.start() + pos;
            if !edit.invalidates_offset(off) {
                remove_newline(&mut edit, node, text.as_str(), off);
            }
        }
    }

    LocalEdit {
        label: "join lines".to_string(),
        edit: edit.finish(),
        cursor_position: None,
    }
}

pub fn on_enter(file: &SourceFile, offset: TextUnit) -> Option<LocalEdit> {
    let comment = find_leaf_at_offset(file.syntax(), offset)
        .left_biased()
        .and_then(ast::Comment::cast)?;

    if let ast::CommentFlavor::Multiline = comment.flavor() {
        return None;
    }

    let prefix = comment.prefix();
    if offset < comment.syntax().range().start() + TextUnit::of_str(prefix) + TextUnit::from(1) {
        return None;
    }

    let indent = node_indent(file, comment.syntax())?;
    let inserted = format!("\n{}{} ", indent, prefix);
    let cursor_position = offset + TextUnit::of_str(&inserted);
    let mut edit = TextEditBuilder::default();
    edit.insert(offset, inserted);
    Some(LocalEdit {
        label: "on enter".to_string(),
        edit: edit.finish(),
        cursor_position: Some(cursor_position),
    })
}

fn node_indent<'a>(file: &'a SourceFile, node: &SyntaxNode) -> Option<&'a str> {
    let ws = match find_leaf_at_offset(file.syntax(), node.range().start()) {
        LeafAtOffset::Between(l, r) => {
            assert!(r == node);
            l
        }
        LeafAtOffset::Single(n) => {
            assert!(n == node);
            return Some("");
        }
        LeafAtOffset::None => unreachable!(),
    };
    if ws.kind() != WHITESPACE {
        return None;
    }
    let text = ws.leaf_text().unwrap();
    let pos = text.as_str().rfind('\n').map(|it| it + 1).unwrap_or(0);
    Some(&text[pos..])
}

pub fn on_eq_typed(file: &SourceFile, offset: TextUnit) -> Option<LocalEdit> {
    let let_stmt: &ast::LetStmt = find_node_at_offset(file.syntax(), offset)?;
    if let_stmt.has_semi() {
        return None;
    }
    if let Some(expr) = let_stmt.initializer() {
        let expr_range = expr.syntax().range();
        if contains_offset_nonstrict(expr_range, offset) && offset != expr_range.start() {
            return None;
        }
        if file
            .syntax()
            .text()
            .slice(offset..expr_range.start())
            .contains('\n')
        {
            return None;
        }
    } else {
        return None;
    }
    let offset = let_stmt.syntax().range().end();
    let mut edit = TextEditBuilder::default();
    edit.insert(offset, ";".to_string());
    Some(LocalEdit {
        label: "add semicolon".to_string(),
        edit: edit.finish(),
        cursor_position: None,
    })
}

pub fn on_dot_typed(file: &SourceFile, offset: TextUnit) -> Option<LocalEdit> {
    let before_dot_offset = offset - TextUnit::of_char('.');

    let whitespace = find_leaf_at_offset(file.syntax(), before_dot_offset).left_biased()?;

    // find whitespace just left of the dot
    ast::Whitespace::cast(whitespace)?;

    // make sure there is a method call
    let method_call = whitespace
        .siblings(Direction::Prev)
        // first is whitespace
        .skip(1)
        .next()?;

    ast::MethodCallExpr::cast(method_call)?;

    // find how much the _method call is indented
    let method_chain_indent = method_call
        .parent()?
        .siblings(Direction::Prev)
        .skip(1)
        .next()?
        .leaf_text()
        .map(|x| last_line_indent_in_whitespace(x))?;

    let current_indent = TextUnit::of_str(last_line_indent_in_whitespace(whitespace.leaf_text()?));
    // TODO: indent is always 4 spaces now. A better heuristic could look on the previous line(s)

    let target_indent = TextUnit::of_str(method_chain_indent) + TextUnit::from_usize(4);

    let diff = target_indent - current_indent;

    let indent = "".repeat(diff.to_usize());

    let cursor_position = offset + diff;
    let mut edit = TextEditBuilder::default();
    edit.insert(before_dot_offset, indent);
    Some(LocalEdit {
        label: "indent dot".to_string(),
        edit: edit.finish(),
        cursor_position: Some(cursor_position),
    })
}

/// Finds the last line in the whitespace
fn last_line_indent_in_whitespace(ws: &str) -> &str {
    ws.split('\n').last().unwrap_or("")
}

fn remove_newline(
    edit: &mut TextEditBuilder,
    node: &SyntaxNode,
    node_text: &str,
    offset: TextUnit,
) {
    if node.kind() != WHITESPACE || node_text.bytes().filter(|&b| b == b'\n').count() != 1 {
        // The node is either the first or the last in the file
        let suff = &node_text[TextRange::from_to(
            offset - node.range().start() + TextUnit::of_char('\n'),
            TextUnit::of_str(node_text),
        )];
        let spaces = suff.bytes().take_while(|&b| b == b' ').count();

        edit.replace(
            TextRange::offset_len(offset, ((spaces + 1) as u32).into()),
            " ".to_string(),
        );
        return;
    }

    // Special case that turns something like:
    //
    // ```
    // my_function({<|>
    //    <some-expr>
    // })
    // ```
    //
    // into `my_function(<some-expr>)`
    if join_single_expr_block(edit, node).is_some() {
        return;
    }
    // ditto for
    //
    // ```
    // use foo::{<|>
    //    bar
    // };
    // ```
    if join_single_use_tree(edit, node).is_some() {
        return;
    }

    // The node is between two other nodes
    let prev = node.prev_sibling().unwrap();
    let next = node.next_sibling().unwrap();
    if is_trailing_comma(prev.kind(), next.kind()) {
        // Removes: trailing comma, newline (incl. surrounding whitespace)
        edit.delete(TextRange::from_to(prev.range().start(), node.range().end()));
    } else if prev.kind() == COMMA && next.kind() == R_CURLY {
        // Removes: comma, newline (incl. surrounding whitespace)
        let space = if let Some(left) = prev.prev_sibling() {
            compute_ws(left, next)
        } else {
            " "
        };
        edit.replace(
            TextRange::from_to(prev.range().start(), node.range().end()),
            space.to_string(),
        );
    } else if let (Some(_), Some(next)) = (ast::Comment::cast(prev), ast::Comment::cast(next)) {
        // Removes: newline (incl. surrounding whitespace), start of the next comment
        edit.delete(TextRange::from_to(
            node.range().start(),
            next.syntax().range().start() + TextUnit::of_str(next.prefix()),
        ));
    } else {
        // Remove newline but add a computed amount of whitespace characters
        edit.replace(node.range(), compute_ws(prev, next).to_string());
    }
}

fn is_trailing_comma(left: SyntaxKind, right: SyntaxKind) -> bool {
    match (left, right) {
        (COMMA, R_PAREN) | (COMMA, R_BRACK) => true,
        _ => false,
    }
}

fn join_single_expr_block(edit: &mut TextEditBuilder, node: &SyntaxNode) -> Option<()> {
    let block = ast::Block::cast(node.parent()?)?;
    let block_expr = ast::BlockExpr::cast(block.syntax().parent()?)?;
    let expr = single_expr(block)?;
    edit.replace(
        block_expr.syntax().range(),
        expr.syntax().text().to_string(),
    );
    Some(())
}

fn single_expr(block: &ast::Block) -> Option<&ast::Expr> {
    let mut res = None;
    for child in block.syntax().children() {
        if let Some(expr) = ast::Expr::cast(child) {
            if expr.syntax().text().contains('\n') {
                return None;
            }
            if mem::replace(&mut res, Some(expr)).is_some() {
                return None;
            }
        } else {
            match child.kind() {
                WHITESPACE | L_CURLY | R_CURLY => (),
                _ => return None,
            }
        }
    }
    res
}

fn join_single_use_tree(edit: &mut TextEditBuilder, node: &SyntaxNode) -> Option<()> {
    let use_tree_list = ast::UseTreeList::cast(node.parent()?)?;
    let (tree,) = use_tree_list.use_trees().collect_tuple()?;
    edit.replace(
        use_tree_list.syntax().range(),
        tree.syntax().text().to_string(),
    );
    Some(())
}

fn compute_ws(left: &SyntaxNode, right: &SyntaxNode) -> &'static str {
    match left.kind() {
        L_PAREN | L_BRACK => return "",
        L_CURLY => {
            if let USE_TREE = right.kind() {
                return "";
            }
        }
        _ => (),
    }
    match right.kind() {
        R_PAREN | R_BRACK => return "",
        R_CURLY => {
            if let USE_TREE = left.kind() {
                return "";
            }
        }
        DOT => return "",
        _ => (),
    }
    " "
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        add_cursor, assert_eq_text, check_action, extract_offset, extract_range,
};

    fn check_join_lines(before: &str, after: &str) {
        check_action(before, after, |file, offset| {
            let range = TextRange::offset_len(offset, 0.into());
            let res = join_lines(file, range);
            Some(res)
        })
    }

    #[test]
    fn test_join_lines_comma() {
        check_join_lines(
            r"
fn foo() {
    <|>foo(1,
    )
}
",
            r"
fn foo() {
    <|>foo(1)
}
",
        );
    }

    #[test]
    fn test_join_lines_lambda_block() {
        check_join_lines(
            r"
pub fn reparse(&self, edit: &AtomTextEdit) -> File {
    <|>self.incremental_reparse(edit).unwrap_or_else(|| {
        self.full_reparse(edit)
    })
}
",
            r"
pub fn reparse(&self, edit: &AtomTextEdit) -> File {
    <|>self.incremental_reparse(edit).unwrap_or_else(|| self.full_reparse(edit))
}
",
        );
    }

    #[test]
    fn test_join_lines_block() {
        check_join_lines(
            r"
fn foo() {
    foo(<|>{
        92
    })
}",
            r"
fn foo() {
    foo(<|>92)
}",
        );
    }

    #[test]
    fn test_join_lines_use_items_left() {
        // No space after the '{'
        check_join_lines(
            r"
<|>use ra_syntax::{
    TextUnit, TextRange,
};",
            r"
<|>use ra_syntax::{TextUnit, TextRange,
};",
        );
    }

    #[test]
    fn test_join_lines_use_items_right() {
        // No space after the '}'
        check_join_lines(
            r"
use ra_syntax::{
<|>    TextUnit, TextRange
};",
            r"
use ra_syntax::{
<|>    TextUnit, TextRange};",
        );
    }

    #[test]
    fn test_join_lines_use_items_right_comma() {
        // No space after the '}'
        check_join_lines(
            r"
use ra_syntax::{
<|>    TextUnit, TextRange,
};",
            r"
use ra_syntax::{
<|>    TextUnit, TextRange};",
        );
    }

    #[test]
    fn test_join_lines_use_tree() {
        check_join_lines(
            r"
use ra_syntax::{
    algo::<|>{
        find_leaf_at_offset,
    },
    ast,
};",
            r"
use ra_syntax::{
    algo::<|>find_leaf_at_offset,
    ast,
};",
        );
    }

    #[test]
    fn test_join_lines_normal_comments() {
        check_join_lines(
            r"
fn foo() {
    // Hello<|>
    // world!
}
",
            r"
fn foo() {
    // Hello<|> world!
}
",
        );
    }

    #[test]
    fn test_join_lines_doc_comments() {
        check_join_lines(
            r"
fn foo() {
    /// Hello<|>
    /// world!
}
",
            r"
fn foo() {
    /// Hello<|> world!
}
",
        );
    }

    #[test]
    fn test_join_lines_mod_comments() {
        check_join_lines(
            r"
fn foo() {
    //! Hello<|>
    //! world!
}
",
            r"
fn foo() {
    //! Hello<|> world!
}
",
        );
    }

    #[test]
    fn test_join_lines_multiline_comments_1() {
        check_join_lines(
            r"
fn foo() {
    // Hello<|>
    /* world! */
}
",
            r"
fn foo() {
    // Hello<|> world! */
}
",
        );
    }

    #[test]
    fn test_join_lines_multiline_comments_2() {
        check_join_lines(
            r"
fn foo() {
    // The<|>
    /* quick
    brown
    fox! */
}
",
            r"
fn foo() {
    // The<|> quick
    brown
    fox! */
}
",
        );
    }

    fn check_join_lines_sel(before: &str, after: &str) {
        let (sel, before) = extract_range(before);
        let file = SourceFile::parse(&before);
        let result = join_lines(&file, sel);
        let actual = result.edit.apply(&before);
        assert_eq_text!(after, &actual);
    }

    #[test]
    fn test_join_lines_selection_fn_args() {
        check_join_lines_sel(
            r"
fn foo() {
    <|>foo(1,
        2,
        3,
    <|>)
}
    ",
            r"
fn foo() {
    foo(1, 2, 3)
}
    ",
        );
    }

    #[test]
    fn test_join_lines_selection_struct() {
        check_join_lines_sel(
            r"
struct Foo <|>{
    f: u32,
}<|>
    ",
            r"
struct Foo { f: u32 }
    ",
        );
    }

    #[test]
    fn test_join_lines_selection_dot_chain() {
        check_join_lines_sel(
            r"
fn foo() {
    join(<|>type_params.type_params()
            .filter_map(|it| it.name())
            .map(|it| it.text())<|>)
}",
            r"
fn foo() {
    join(type_params.type_params().filter_map(|it| it.name()).map(|it| it.text()))
}",
        );
    }

    #[test]
    fn test_join_lines_selection_lambda_block_body() {
        check_join_lines_sel(
            r"
pub fn handle_find_matching_brace() {
    params.offsets
        .map(|offset| <|>{
            world.analysis().matching_brace(&file, offset).unwrap_or(offset)
        }<|>)
        .collect();
}",
            r"
pub fn handle_find_matching_brace() {
    params.offsets
        .map(|offset| world.analysis().matching_brace(&file, offset).unwrap_or(offset))
        .collect();
}",
        );
    }

    #[test]
    fn test_on_eq_typed() {
        fn do_check(before: &str, after: &str) {
            let (offset, before) = extract_offset(before);
            let file = SourceFile::parse(&before);
            let result = on_eq_typed(&file, offset).unwrap();
            let actual = result.edit.apply(&before);
            assert_eq_text!(after, &actual);
        }

        //     do_check(r"
        // fn foo() {
        //     let foo =<|>
        // }
        // ", r"
        // fn foo() {
        //     let foo =;
        // }
        // ");
        do_check(
            r"
fn foo() {
    let foo =<|> 1 + 1
}
",
            r"
fn foo() {
    let foo = 1 + 1;
}
",
        );
        //     do_check(r"
        // fn foo() {
        //     let foo =<|>
        //     let bar = 1;
        // }
        // ", r"
        // fn foo() {
        //     let foo =;
        //     let bar = 1;
        // }
        // ");
    }

    #[test]
    fn test_on_dot_typed() {
        fn do_check(before: &str, after: &str) {
            let (offset, before) = extract_offset(before);
            let file = SourceFile::parse(&before);
            if let Some(result) = on_eq_typed(&file, offset) {
                let actual = result.edit.apply(&before);
                assert_eq_text!(after, &actual);
            };
        }
        // indent if continuing chain call
        do_check(
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
        .<|>
    }
",
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .
    }
",
        );

        // do not indent if already indented
        do_check(
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .<|>
    }
",
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .
    }
",
        );

        // indent if the previous line is already indented
        do_check(
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .first()
        .<|>
    }
",
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .first()
            .
    }
",
        );

        // don't indent if indent matches previous line
        do_check(
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .first()
            .<|>
    }
",
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .first()
            .
    }
",
        );

        // don't indent if there is no method call on previous line
        do_check(
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        .<|>
    }
",
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        .
    }
",
        );

        // indent to match previous expr
        do_check(
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
.<|>
    }
",
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .
    }
",
        );
    }

    #[test]
    fn test_on_enter() {
        fn apply_on_enter(before: &str) -> Option<String> {
            let (offset, before) = extract_offset(before);
            let file = SourceFile::parse(&before);
            let result = on_enter(&file, offset)?;
            let actual = result.edit.apply(&before);
            let actual = add_cursor(&actual, result.cursor_position.unwrap());
            Some(actual)
        }

        fn do_check(before: &str, after: &str) {
            let actual = apply_on_enter(before).unwrap();
            assert_eq_text!(after, &actual);
        }

        fn do_check_noop(text: &str) {
            assert!(apply_on_enter(text).is_none())
        }

        do_check(
            r"
/// Some docs<|>
fn foo() {
}
",
            r"
/// Some docs
/// <|>
fn foo() {
}
",
        );
        do_check(
            r"
impl S {
    /// Some<|> docs.
    fn foo() {}
}
",
            r"
impl S {
    /// Some
    /// <|> docs.
    fn foo() {}
}
",
        );
        do_check_noop(r"<|>//! docz");
    }
}
