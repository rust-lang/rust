use join_to_string::join;

use ra_syntax::{
    algo::{find_covering_node, find_leaf_at_offset},
    ast::{self, AstNode, AttrsOwner, NameOwner, TypeParamsOwner},
    Direction, SourceFileNode,
    SyntaxKind::{COMMA, WHITESPACE, COMMENT},
    SyntaxNodeRef, TextRange, TextUnit,
};

use crate::{find_node_at_offset, Edit, EditBuilder};

#[derive(Debug)]
pub struct LocalEdit {
    pub edit: Edit,
    pub cursor_position: Option<TextUnit>,
}

pub fn flip_comma<'a>(
    file: &'a SourceFileNode,
    offset: TextUnit,
) -> Option<impl FnOnce() -> LocalEdit + 'a> {
    let syntax = file.syntax();

    let comma = find_leaf_at_offset(syntax, offset).find(|leaf| leaf.kind() == COMMA)?;
    let prev = non_trivia_sibling(comma, Direction::Prev)?;
    let next = non_trivia_sibling(comma, Direction::Next)?;
    Some(move || {
        let mut edit = EditBuilder::new();
        edit.replace(prev.range(), next.text().to_string());
        edit.replace(next.range(), prev.text().to_string());
        LocalEdit {
            edit: edit.finish(),
            cursor_position: None,
        }
    })
}

pub fn add_derive<'a>(
    file: &'a SourceFileNode,
    offset: TextUnit,
) -> Option<impl FnOnce() -> LocalEdit + 'a> {
    let nominal = find_node_at_offset::<ast::NominalDef>(file.syntax(), offset)?;
    let node_start = derive_insertion_offset(nominal)?;
    return Some(move || {
        let derive_attr = nominal
            .attrs()
            .filter_map(|x| x.as_call())
            .filter(|(name, _arg)| name == "derive")
            .map(|(_name, arg)| arg)
            .next();
        let mut edit = EditBuilder::new();
        let offset = match derive_attr {
            None => {
                edit.insert(node_start, "#[derive()]\n".to_string());
                node_start + TextUnit::of_str("#[derive(")
            }
            Some(tt) => tt.syntax().range().end() - TextUnit::of_char(')'),
        };
        LocalEdit {
            edit: edit.finish(),
            cursor_position: Some(offset),
        }
    });

    // Insert `derive` after doc comments.
    fn derive_insertion_offset(nominal: ast::NominalDef) -> Option<TextUnit> {
        let non_ws_child = nominal
            .syntax()
            .children()
            .find(|it| it.kind() != COMMENT && it.kind() != WHITESPACE)?;
        Some(non_ws_child.range().start())
    }
}

pub fn add_impl<'a>(
    file: &'a SourceFileNode,
    offset: TextUnit,
) -> Option<impl FnOnce() -> LocalEdit + 'a> {
    let nominal = find_node_at_offset::<ast::NominalDef>(file.syntax(), offset)?;
    let name = nominal.name()?;

    Some(move || {
        let type_params = nominal.type_param_list();
        let mut edit = EditBuilder::new();
        let start_offset = nominal.syntax().range().end();
        let mut buf = String::new();
        buf.push_str("\n\nimpl");
        if let Some(type_params) = type_params {
            type_params.syntax().text().push_to(&mut buf);
        }
        buf.push_str(" ");
        buf.push_str(name.text().as_str());
        if let Some(type_params) = type_params {
            let lifetime_params = type_params
                .lifetime_params()
                .filter_map(|it| it.lifetime())
                .map(|it| it.text());
            let type_params = type_params
                .type_params()
                .filter_map(|it| it.name())
                .map(|it| it.text());
            join(lifetime_params.chain(type_params))
                .surround_with("<", ">")
                .to_buf(&mut buf);
        }
        buf.push_str(" {\n");
        let offset = start_offset + TextUnit::of_str(&buf);
        buf.push_str("\n}");
        edit.insert(start_offset, buf);
        LocalEdit {
            edit: edit.finish(),
            cursor_position: Some(offset),
        }
    })
}

pub fn introduce_variable<'a>(
    file: &'a SourceFileNode,
    range: TextRange,
) -> Option<impl FnOnce() -> LocalEdit + 'a> {
    let node = find_covering_node(file.syntax(), range);
    let expr = node.ancestors().filter_map(ast::Expr::cast).next()?;

    let anchor_stmt = anchor_stmt(expr)?;
    let indent = anchor_stmt.prev_sibling()?;
    if indent.kind() != WHITESPACE {
        return None;
    }
    return Some(move || {
        let mut buf = String::new();
        let mut edit = EditBuilder::new();

        buf.push_str("let var_name = ");
        expr.syntax().text().push_to(&mut buf);
        let is_full_stmt = if let Some(expr_stmt) = ast::ExprStmt::cast(anchor_stmt) {
            Some(expr.syntax()) == expr_stmt.expr().map(|e| e.syntax())
        } else {
            false
        };
        if is_full_stmt {
            edit.replace(expr.syntax().range(), buf);
        } else {
            buf.push_str(";");
            indent.text().push_to(&mut buf);
            edit.replace(expr.syntax().range(), "var_name".to_string());
            edit.insert(anchor_stmt.range().start(), buf);
        }
        let cursor_position = anchor_stmt.range().start() + TextUnit::of_str("let ");
        LocalEdit {
            edit: edit.finish(),
            cursor_position: Some(cursor_position),
        }
    });

    /// Statement or last in the block expression, which will follow
    /// the freshly introduced var.
    fn anchor_stmt(expr: ast::Expr) -> Option<SyntaxNodeRef> {
        expr.syntax().ancestors().find(|&node| {
            if ast::Stmt::cast(node).is_some() {
                return true;
            }
            if let Some(expr) = node
                .parent()
                .and_then(ast::Block::cast)
                .and_then(|it| it.expr())
            {
                if expr.syntax() == node {
                    return true;
                }
            }
            false
        })
    }
}

fn non_trivia_sibling(node: SyntaxNodeRef, direction: Direction) -> Option<SyntaxNodeRef> {
    node.siblings(direction)
        .skip(1)
        .find(|node| !node.kind().is_trivia())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{check_action, check_action_range};

    #[test]
    fn test_swap_comma() {
        check_action(
            "fn foo(x: i32,<|> y: Result<(), ()>) {}",
            "fn foo(y: Result<(), ()>,<|> x: i32) {}",
            |file, off| flip_comma(file, off).map(|f| f()),
        )
    }

    #[test]
    fn add_derive_new() {
        check_action(
            "struct Foo { a: i32, <|>}",
            "#[derive(<|>)]\nstruct Foo { a: i32, }",
            |file, off| add_derive(file, off).map(|f| f()),
        );
        check_action(
            "struct Foo { <|> a: i32, }",
            "#[derive(<|>)]\nstruct Foo {  a: i32, }",
            |file, off| add_derive(file, off).map(|f| f()),
        );
    }

    #[test]
    fn add_derive_existing() {
        check_action(
            "#[derive(Clone)]\nstruct Foo { a: i32<|>, }",
            "#[derive(Clone<|>)]\nstruct Foo { a: i32, }",
            |file, off| add_derive(file, off).map(|f| f()),
        );
    }

    #[test]
    fn add_derive_new_with_doc_comment() {
        check_action(
            "
/// `Foo` is a pretty important struct.
/// It does stuff.
struct Foo { a: i32<|>, }
            ",
            "
/// `Foo` is a pretty important struct.
/// It does stuff.
#[derive(<|>)]
struct Foo { a: i32, }
            ",
            |file, off| add_derive(file, off).map(|f| f()),
        );
    }

    #[test]
    fn test_add_impl() {
        check_action(
            "struct Foo {<|>}\n",
            "struct Foo {}\n\nimpl Foo {\n<|>\n}\n",
            |file, off| add_impl(file, off).map(|f| f()),
        );
        check_action(
            "struct Foo<T: Clone> {<|>}",
            "struct Foo<T: Clone> {}\n\nimpl<T: Clone> Foo<T> {\n<|>\n}",
            |file, off| add_impl(file, off).map(|f| f()),
        );
        check_action(
            "struct Foo<'a, T: Foo<'a>> {<|>}",
            "struct Foo<'a, T: Foo<'a>> {}\n\nimpl<'a, T: Foo<'a>> Foo<'a, T> {\n<|>\n}",
            |file, off| add_impl(file, off).map(|f| f()),
        );
    }

    #[test]
    fn test_introduce_var_simple() {
        check_action_range(
            "
fn foo() {
    foo(<|>1 + 1<|>);
}",
            "
fn foo() {
    let <|>var_name = 1 + 1;
    foo(var_name);
}",
            |file, range| introduce_variable(file, range).map(|f| f()),
        );
    }

    #[test]
    fn test_introduce_var_expr_stmt() {
        check_action_range(
            "
fn foo() {
    <|>1 + 1<|>;
}",
            "
fn foo() {
    let <|>var_name = 1 + 1;
}",
            |file, range| introduce_variable(file, range).map(|f| f()),
        );
    }

    #[test]
    fn test_introduce_var_part_of_expr_stmt() {
        check_action_range(
            "
fn foo() {
    <|>1<|> + 1;
}",
            "
fn foo() {
    let <|>var_name = 1;
    var_name + 1;
}",
            |file, range| introduce_variable(file, range).map(|f| f()),
        );
    }

    #[test]
    fn test_introduce_var_last_expr() {
        check_action_range(
            "
fn foo() {
    bar(<|>1 + 1<|>)
}",
            "
fn foo() {
    let <|>var_name = 1 + 1;
    bar(var_name)
}",
            |file, range| introduce_variable(file, range).map(|f| f()),
        );
    }

    #[test]
    fn test_introduce_var_last_full_expr() {
        check_action_range(
            "
fn foo() {
    <|>bar(1 + 1)<|>
}",
            "
fn foo() {
    let <|>var_name = bar(1 + 1);
    var_name
}",
            |file, range| introduce_variable(file, range).map(|f| f()),
        );
    }

}
