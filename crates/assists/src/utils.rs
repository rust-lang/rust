//! Assorted functions shared by several assists.

use std::ops;

use hir::HasSource;
use ide_db::{helpers::SnippetCap, RootDatabase};
use itertools::Itertools;
use syntax::{
    ast::edit::AstNodeEdit,
    ast::AttrsOwner,
    ast::NameOwner,
    ast::{self, edit, make, ArgListOwner},
    AstNode, Direction,
    SyntaxKind::*,
    SyntaxNode, TextSize, T,
};

use crate::ast_transform::{self, AstTransform, QualifyPaths, SubstituteTypeParams};

pub(crate) fn unwrap_trivial_block(block: ast::BlockExpr) -> ast::Expr {
    extract_trivial_expression(&block)
        .filter(|expr| !expr.syntax().text().contains_char('\n'))
        .unwrap_or_else(|| block.into())
}

pub fn extract_trivial_expression(block: &ast::BlockExpr) -> Option<ast::Expr> {
    let has_anything_else = |thing: &SyntaxNode| -> bool {
        let mut non_trivial_children =
            block.syntax().children_with_tokens().filter(|it| match it.kind() {
                WHITESPACE | T!['{'] | T!['}'] => false,
                _ => it.as_node() != Some(thing),
            });
        non_trivial_children.next().is_some()
    };

    if let Some(expr) = block.tail_expr() {
        if has_anything_else(expr.syntax()) {
            return None;
        }
        return Some(expr);
    }
    // Unwrap `{ continue; }`
    let (stmt,) = block.statements().next_tuple()?;
    if let ast::Stmt::ExprStmt(expr_stmt) = stmt {
        if has_anything_else(expr_stmt.syntax()) {
            return None;
        }
        let expr = expr_stmt.expr()?;
        match expr.syntax().kind() {
            CONTINUE_EXPR | BREAK_EXPR | RETURN_EXPR => return Some(expr),
            _ => (),
        }
    }
    None
}

/// This is a method with a heuristics to support test methods annotated with custom test annotations, such as
/// `#[test_case(...)]`, `#[tokio::test]` and similar.
/// Also a regular `#[test]` annotation is supported.
///
/// It may produce false positives, for example, `#[wasm_bindgen_test]` requires a different command to run the test,
/// but it's better than not to have the runnables for the tests at all.
pub fn test_related_attribute(fn_def: &ast::Fn) -> Option<ast::Attr> {
    fn_def.attrs().find_map(|attr| {
        let path = attr.path()?;
        if path.syntax().text().to_string().contains("test") {
            Some(attr)
        } else {
            None
        }
    })
}

#[derive(Copy, Clone, PartialEq)]
pub enum DefaultMethods {
    Only,
    No,
}

pub fn filter_assoc_items(
    db: &RootDatabase,
    items: &[hir::AssocItem],
    default_methods: DefaultMethods,
) -> Vec<ast::AssocItem> {
    fn has_def_name(item: &ast::AssocItem) -> bool {
        match item {
            ast::AssocItem::Fn(def) => def.name(),
            ast::AssocItem::TypeAlias(def) => def.name(),
            ast::AssocItem::Const(def) => def.name(),
            ast::AssocItem::MacroCall(_) => None,
        }
        .is_some()
    }

    items
        .iter()
        // Note: This throws away items with no source.
        .filter_map(|i| {
            let item = match i {
                hir::AssocItem::Function(i) => ast::AssocItem::Fn(i.source(db)?.value),
                hir::AssocItem::TypeAlias(i) => ast::AssocItem::TypeAlias(i.source(db)?.value),
                hir::AssocItem::Const(i) => ast::AssocItem::Const(i.source(db)?.value),
            };
            Some(item)
        })
        .filter(has_def_name)
        .filter(|it| match it {
            ast::AssocItem::Fn(def) => matches!(
                (default_methods, def.body()),
                (DefaultMethods::Only, Some(_)) | (DefaultMethods::No, None)
            ),
            _ => default_methods == DefaultMethods::No,
        })
        .collect::<Vec<_>>()
}

pub fn add_trait_assoc_items_to_impl(
    sema: &hir::Semantics<ide_db::RootDatabase>,
    items: Vec<ast::AssocItem>,
    trait_: hir::Trait,
    impl_def: ast::Impl,
    target_scope: hir::SemanticsScope,
) -> (ast::Impl, ast::AssocItem) {
    let impl_item_list = impl_def.assoc_item_list().unwrap_or_else(make::assoc_item_list);

    let n_existing_items = impl_item_list.assoc_items().count();
    let source_scope = sema.scope_for_def(trait_);
    let ast_transform = QualifyPaths::new(&target_scope, &source_scope)
        .or(SubstituteTypeParams::for_trait_impl(&source_scope, trait_, impl_def.clone()));

    let items = items
        .into_iter()
        .map(|it| ast_transform::apply(&*ast_transform, it))
        .map(|it| match it {
            ast::AssocItem::Fn(def) => ast::AssocItem::Fn(add_body(def)),
            ast::AssocItem::TypeAlias(def) => ast::AssocItem::TypeAlias(def.remove_bounds()),
            _ => it,
        })
        .map(|it| edit::remove_attrs_and_docs(&it));

    let new_impl_item_list = impl_item_list.append_items(items);
    let new_impl_def = impl_def.with_assoc_item_list(new_impl_item_list);
    let first_new_item =
        new_impl_def.assoc_item_list().unwrap().assoc_items().nth(n_existing_items).unwrap();
    return (new_impl_def, first_new_item);

    fn add_body(fn_def: ast::Fn) -> ast::Fn {
        match fn_def.body() {
            Some(_) => fn_def,
            None => {
                let body =
                    make::block_expr(None, Some(make::expr_todo())).indent(edit::IndentLevel(1));
                fn_def.with_body(body)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Cursor<'a> {
    Replace(&'a SyntaxNode),
    Before(&'a SyntaxNode),
}

impl<'a> Cursor<'a> {
    fn node(self) -> &'a SyntaxNode {
        match self {
            Cursor::Replace(node) | Cursor::Before(node) => node,
        }
    }
}

pub(crate) fn render_snippet(_cap: SnippetCap, node: &SyntaxNode, cursor: Cursor) -> String {
    assert!(cursor.node().ancestors().any(|it| it == *node));
    let range = cursor.node().text_range() - node.text_range().start();
    let range: ops::Range<usize> = range.into();

    let mut placeholder = cursor.node().to_string();
    escape(&mut placeholder);
    let tab_stop = match cursor {
        Cursor::Replace(placeholder) => format!("${{0:{}}}", placeholder),
        Cursor::Before(placeholder) => format!("$0{}", placeholder),
    };

    let mut buf = node.to_string();
    buf.replace_range(range, &tab_stop);
    return buf;

    fn escape(buf: &mut String) {
        stdx::replace(buf, '{', r"\{");
        stdx::replace(buf, '}', r"\}");
        stdx::replace(buf, '$', r"\$");
    }
}

pub(crate) fn vis_offset(node: &SyntaxNode) -> TextSize {
    node.children_with_tokens()
        .find(|it| !matches!(it.kind(), WHITESPACE | COMMENT | ATTR))
        .map(|it| it.text_range().start())
        .unwrap_or_else(|| node.text_range().start())
}

pub(crate) fn invert_boolean_expression(expr: ast::Expr) -> ast::Expr {
    if let Some(expr) = invert_special_case(&expr) {
        return expr;
    }
    make::expr_prefix(T![!], expr)
}

fn invert_special_case(expr: &ast::Expr) -> Option<ast::Expr> {
    match expr {
        ast::Expr::BinExpr(bin) => match bin.op_kind()? {
            ast::BinOp::NegatedEqualityTest => bin.replace_op(T![==]).map(|it| it.into()),
            ast::BinOp::EqualityTest => bin.replace_op(T![!=]).map(|it| it.into()),
            // Parenthesize composite boolean expressions before prefixing `!`
            ast::BinOp::BooleanAnd | ast::BinOp::BooleanOr => {
                Some(make::expr_prefix(T![!], make::expr_paren(expr.clone())))
            }
            _ => None,
        },
        ast::Expr::MethodCallExpr(mce) => {
            let receiver = mce.receiver()?;
            let method = mce.name_ref()?;
            let arg_list = mce.arg_list()?;

            let method = match method.text() {
                "is_some" => "is_none",
                "is_none" => "is_some",
                "is_ok" => "is_err",
                "is_err" => "is_ok",
                _ => return None,
            };
            Some(make::expr_method_call(receiver, method, arg_list))
        }
        ast::Expr::PrefixExpr(pe) if pe.op_kind()? == ast::PrefixOp::Not => {
            if let ast::Expr::ParenExpr(parexpr) = pe.expr()? {
                parexpr.expr()
            } else {
                pe.expr()
            }
        }
        // FIXME:
        // ast::Expr::Literal(true | false )
        _ => None,
    }
}

pub(crate) fn next_prev() -> impl Iterator<Item = Direction> {
    [Direction::Next, Direction::Prev].iter().copied()
}

pub(crate) fn does_pat_match_variant(pat: &ast::Pat, var: &ast::Pat) -> bool {
    let first_node_text = |pat: &ast::Pat| pat.syntax().first_child().map(|node| node.text());

    let pat_head = match pat {
        ast::Pat::IdentPat(bind_pat) => {
            if let Some(p) = bind_pat.pat() {
                first_node_text(&p)
            } else {
                return pat.syntax().text() == var.syntax().text();
            }
        }
        pat => first_node_text(pat),
    };

    let var_head = first_node_text(var);

    pat_head == var_head
}
