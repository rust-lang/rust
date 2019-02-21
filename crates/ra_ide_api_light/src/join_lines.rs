use itertools::Itertools;
use ra_syntax::{
    SourceFile, TextRange, TextUnit, AstNode, SyntaxNode,
    SyntaxKind::{self, WHITESPACE, COMMA, R_CURLY, R_PAREN, R_BRACK},
    algo::find_covering_node,
    ast,
};
use ra_fmt::{
    compute_ws, extract_trivial_expression
};
use crate::{
    LocalEdit, TextEditBuilder,
};

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
        let range = match range.intersection(&node.range()) {
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

    LocalEdit { label: "join lines".to_string(), edit: edit.finish(), cursor_position: None }
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

        edit.replace(TextRange::offset_len(offset, ((spaces + 1) as u32).into()), " ".to_string());
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
        let space = if let Some(left) = prev.prev_sibling() { compute_ws(left, next) } else { " " };
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

fn has_comma_after(node: &SyntaxNode) -> bool {
    let next = node.next_sibling();
    let nnext = node.next_sibling().and_then(|n| n.next_sibling());

    match (next, nnext) {
        // Whitespace followed by a comma is fine
        (Some(ws), Some(comma)) if ws.kind() == WHITESPACE && comma.kind() == COMMA => true,
        (Some(n), _) => n.kind() == COMMA,
        _ => false,
    }
}

fn join_single_expr_block(edit: &mut TextEditBuilder, node: &SyntaxNode) -> Option<()> {
    let block = ast::Block::cast(node.parent()?)?;
    let block_expr = ast::BlockExpr::cast(block.syntax().parent()?)?;
    let expr = extract_trivial_expression(block)?;

    let block_range = block_expr.syntax().range();
    let mut buf = expr.syntax().text().to_string();

    // Match block needs to have a comma after the block
    if let Some(match_arm) = block_expr.syntax().parent().and_then(ast::MatchArm::cast) {
        if !has_comma_after(match_arm.syntax()) {
            buf.push(',');
        }
    }

    edit.replace(block_range, buf);

    Some(())
}

fn join_single_use_tree(edit: &mut TextEditBuilder, node: &SyntaxNode) -> Option<()> {
    let use_tree_list = ast::UseTreeList::cast(node.parent()?)?;
    let (tree,) = use_tree_list.use_trees().collect_tuple()?;
    edit.replace(use_tree_list.syntax().range(), tree.syntax().text().to_string());
    Some(())
}

fn is_trailing_comma(left: SyntaxKind, right: SyntaxKind) -> bool {
    match (left, right) {
        (COMMA, R_PAREN) | (COMMA, R_BRACK) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{assert_eq_text, check_action, extract_range};

    use super::*;

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
    fn join_lines_adds_comma_for_block_in_match_arm() {
        check_join_lines(
            r"
fn foo(e: Result<U, V>) {
    match e {
        Ok(u) => <|>{
            u.foo()
        }
        Err(v) => v,
    }
}",
            r"
fn foo(e: Result<U, V>) {
    match e {
        Ok(u) => <|>u.foo(),
        Err(v) => v,
    }
}",
        );
    }

    #[test]
    fn join_lines_keeps_comma_for_block_in_match_arm() {
        // We already have a comma
        check_join_lines(
            r"
fn foo(e: Result<U, V>) {
    match e {
        Ok(u) => <|>{
            u.foo()
        },
        Err(v) => v,
    }
}",
            r"
fn foo(e: Result<U, V>) {
    match e {
        Ok(u) => <|>u.foo(),
        Err(v) => v,
    }
}",
        );

        // comma with whitespace between brace and ,
        check_join_lines(
            r"
fn foo(e: Result<U, V>) {
    match e {
        Ok(u) => <|>{
            u.foo()
        }    ,
        Err(v) => v,
    }
}",
            r"
fn foo(e: Result<U, V>) {
    match e {
        Ok(u) => <|>u.foo()    ,
        Err(v) => v,
    }
}",
        );

        // comma with newline between brace and ,
        check_join_lines(
            r"
fn foo(e: Result<U, V>) {
    match e {
        Ok(u) => <|>{
            u.foo()
        }
        ,
        Err(v) => v,
    }
}",
            r"
fn foo(e: Result<U, V>) {
    match e {
        Ok(u) => <|>u.foo()
        ,
        Err(v) => v,
    }
}",
        );
    }

    #[test]
    fn join_lines_keeps_comma_with_single_arg_tuple() {
        // A single arg tuple
        check_join_lines(
            r"
fn foo() {
    let x = (<|>{
       4
    },);
}",
            r"
fn foo() {
    let x = (<|>4,);
}",
        );

        // single arg tuple with whitespace between brace and comma
        check_join_lines(
            r"
fn foo() {
    let x = (<|>{
       4
    }   ,);
}",
            r"
fn foo() {
    let x = (<|>4   ,);
}",
        );

        // single arg tuple with newline between brace and comma
        check_join_lines(
            r"
fn foo() {
    let x = (<|>{
       4
    }
    ,);
}",
            r"
fn foo() {
    let x = (<|>4
    ,);
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
}
