use std::mem;

use libsyntax2::{
    TextUnit, TextRange, SyntaxNodeRef, File, AstNode,
    ast,
    algo::{
        walk::preorder,
        find_covering_node,
    },
    text_utils::{intersect, contains_offset_nonstrict},
    SyntaxKind::*,
};

use {ActionResult, EditBuilder, find_node_at_offset};

pub fn join_lines(file: &File, range: TextRange) -> ActionResult {
    let range = if range.is_empty() {
        let syntax = file.syntax();
        let text = syntax.text().slice(range.start()..);
        let pos = match text.find('\n') {
            None => return ActionResult {
                edit: EditBuilder::new().finish(),
                cursor_position: None
            },
            Some(pos) => pos
        };
        TextRange::offset_len(
            range.start() + pos,
            TextUnit::of_char('\n'),
        )
    } else {
        range
    };
    let node = find_covering_node(file.syntax(), range);
    let mut edit = EditBuilder::new();
    for node in preorder(node) {
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
            remove_newline(&mut edit, node, text.as_str(), off);
        }
    }

    ActionResult {
        edit: edit.finish(),
        cursor_position: None,
    }
}

pub fn on_eq_typed(file: &File, offset: TextUnit) -> Option<ActionResult> {
    let let_stmt: ast::LetStmt = find_node_at_offset(file.syntax(), offset)?;
    if let_stmt.has_semi() {
        return None;
    }
    if let Some(expr) = let_stmt.initializer() {
        let expr_range = expr.syntax().range();
        if contains_offset_nonstrict(expr_range, offset) && offset != expr_range.start() {
            return None;
        }
        if file.syntax().text().slice(offset..expr_range.start()).contains('\n') {
            return None;
        }
    } else {
        return None;
    }
    let offset = let_stmt.syntax().range().end();
    let mut edit = EditBuilder::new();
    edit.insert(offset, ";".to_string());
    Some(ActionResult {
        edit: edit.finish(),
        cursor_position: None,
    })
}

fn remove_newline(
    edit: &mut EditBuilder,
    node: SyntaxNodeRef,
    node_text: &str,
    offset: TextUnit,
) {
    if node.kind() == WHITESPACE && node_text.bytes().filter(|&b| b == b'\n').count() == 1 {
        if join_single_expr_block(edit, node).is_some() {
            return
        }
        match (node.prev_sibling(), node.next_sibling()) {
            (Some(prev), Some(next)) => {
                let range = TextRange::from_to(prev.range().start(), node.range().end());
                if prev.kind() == COMMA && (next.kind() == R_PAREN || next.kind() == R_BRACK) {
                    edit.delete(range);
                } else if prev.kind() == COMMA && next.kind() == R_CURLY {
                    edit.replace(range, " ".to_string());
                } else {
                    edit.replace(
                        node.range(),
                        compute_ws(prev, next).to_string(),
                    );
                }
                return;
            }
            _ => (),
        }
    }

    let suff = &node_text[TextRange::from_to(
        offset - node.range().start() + TextUnit::of_char('\n'),
        TextUnit::of_str(node_text),
    )];
    let spaces = suff.bytes().take_while(|&b| b == b' ').count();

    edit.replace(
        TextRange::offset_len(offset, ((spaces + 1) as u32).into()),
        " ".to_string(),
    );
}

fn join_single_expr_block(
    edit: &mut EditBuilder,
    node: SyntaxNodeRef,
) -> Option<()> {
    let block = ast::Block::cast(node.parent()?)?;
    let block_expr = ast::BlockExpr::cast(block.syntax().parent()?)?;
    let expr = single_expr(block)?;
    edit.replace(
        block_expr.syntax().range(),
        expr.syntax().text().to_string(),
    );
    Some(())
}

fn single_expr(block: ast::Block) -> Option<ast::Expr> {
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

fn compute_ws(left: SyntaxNodeRef, right: SyntaxNodeRef) -> &'static str {
    match left.kind() {
        L_PAREN | L_BRACK => return "",
        _ => (),
    }
    match right.kind() {
        R_PAREN | R_BRACK => return "",
        _ => (),
    }
    " "
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_utils::{check_action, extract_range, extract_offset};

    fn check_join_lines(before: &str, after: &str) {
        check_action(before, after, |file, offset| {
            let range = TextRange::offset_len(offset, 0.into());
            let res = join_lines(file, range);
            Some(res)
        })
    }

    #[test]
    fn test_join_lines_comma() {
        check_join_lines(r"
fn foo() {
    <|>foo(1,
    )
}
", r"
fn foo() {
    <|>foo(1)
}
");
    }

    #[test]
    fn test_join_lines_lambda_block() {
        check_join_lines(r"
pub fn reparse(&self, edit: &AtomEdit) -> File {
    <|>self.incremental_reparse(edit).unwrap_or_else(|| {
        self.full_reparse(edit)
    })
}
", r"
pub fn reparse(&self, edit: &AtomEdit) -> File {
    <|>self.incremental_reparse(edit).unwrap_or_else(|| self.full_reparse(edit))
}
");
    }

    #[test]
    fn test_join_lines_block() {
        check_join_lines(r"
fn foo() {
    foo(<|>{
        92
    })
}", r"
fn foo() {
    foo(<|>92)
}");
    }

    #[test]
    fn test_join_lines_selection() {
        fn do_check(before: &str, after: &str) {
            let (sel, before) = extract_range(before);
            let file = File::parse(&before);
            let result = join_lines(&file, sel);
            let actual = result.edit.apply(&before);
            assert_eq_text!(after, &actual);
        }

        do_check(r"
fn foo() {
    <|>foo(1,
        2,
        3,
    <|>)
}
    ", r"
fn foo() {
    foo(1, 2, 3)
}
    ");

        do_check(r"
struct Foo <|>{
    f: u32,
}<|>
    ", r"
struct Foo { f: u32 }
    ");
    }

    #[test]
    fn test_on_eq_typed() {
        fn do_check(before: &str, after: &str) {
            let (offset, before) = extract_offset(before);
            let file = File::parse(&before);
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
        do_check(r"
fn foo() {
    let foo =<|> 1 + 1
}
", r"
fn foo() {
    let foo = 1 + 1;
}
");
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
}
