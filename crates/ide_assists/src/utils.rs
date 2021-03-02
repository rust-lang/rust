//! Assorted functions shared by several assists.

pub(crate) mod suggest_name;

use std::ops;

use ast::TypeBoundsOwner;
use hir::{Adt, HasSource, Semantics};
use ide_db::{
    helpers::{FamousDefs, SnippetCap},
    RootDatabase,
};
use itertools::Itertools;
use stdx::format_to;
use syntax::{
    ast::edit::AstNodeEdit,
    ast::AttrsOwner,
    ast::NameOwner,
    ast::{self, edit, make, ArgListOwner, GenericParamsOwner},
    AstNode, Direction, SmolStr,
    SyntaxKind::*,
    SyntaxNode, TextSize, T,
};

use crate::{
    assist_context::{AssistBuilder, AssistContext},
    ast_transform::{self, AstTransform, QualifyPaths, SubstituteTypeParams},
};

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

pub(crate) fn invert_boolean_expression(
    sema: &Semantics<RootDatabase>,
    expr: ast::Expr,
) -> ast::Expr {
    if let Some(expr) = invert_special_case(sema, &expr) {
        return expr;
    }
    make::expr_prefix(T![!], expr)
}

fn invert_special_case(sema: &Semantics<RootDatabase>, expr: &ast::Expr) -> Option<ast::Expr> {
    match expr {
        ast::Expr::BinExpr(bin) => match bin.op_kind()? {
            ast::BinOp::NegatedEqualityTest => bin.replace_op(T![==]).map(|it| it.into()),
            ast::BinOp::EqualityTest => bin.replace_op(T![!=]).map(|it| it.into()),
            // Swap `<` with `>=`, `<=` with `>`, ... if operands `impl Ord`
            ast::BinOp::LesserTest if bin_impls_ord(sema, bin) => {
                bin.replace_op(T![>=]).map(|it| it.into())
            }
            ast::BinOp::LesserEqualTest if bin_impls_ord(sema, bin) => {
                bin.replace_op(T![>]).map(|it| it.into())
            }
            ast::BinOp::GreaterTest if bin_impls_ord(sema, bin) => {
                bin.replace_op(T![<=]).map(|it| it.into())
            }
            ast::BinOp::GreaterEqualTest if bin_impls_ord(sema, bin) => {
                bin.replace_op(T![<]).map(|it| it.into())
            }
            // Parenthesize other expressions before prefixing `!`
            _ => Some(make::expr_prefix(T![!], make::expr_paren(expr.clone()))),
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

fn bin_impls_ord(sema: &Semantics<RootDatabase>, bin: &ast::BinExpr) -> bool {
    match (
        bin.lhs().and_then(|lhs| sema.type_of_expr(&lhs)),
        bin.rhs().and_then(|rhs| sema.type_of_expr(&rhs)),
    ) {
        (Some(lhs_ty), Some(rhs_ty)) if lhs_ty == rhs_ty => {
            let krate = sema.scope(bin.syntax()).module().map(|it| it.krate());
            let ord_trait = FamousDefs(sema, krate).core_cmp_Ord();
            ord_trait.map_or(false, |ord_trait| {
                lhs_ty.autoderef(sema.db).any(|ty| ty.impls_trait(sema.db, ord_trait, &[]))
            })
        }
        _ => false,
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

// Uses a syntax-driven approach to find any impl blocks for the struct that
// exist within the module/file
//
// Returns `None` if we've found an existing fn
//
// FIXME: change the new fn checking to a more semantic approach when that's more
// viable (e.g. we process proc macros, etc)
// FIXME: this partially overlaps with `find_impl_block_*`
pub(crate) fn find_struct_impl(
    ctx: &AssistContext,
    strukt: &ast::Adt,
    name: &str,
) -> Option<Option<ast::Impl>> {
    let db = ctx.db();
    let module = strukt.syntax().ancestors().find(|node| {
        ast::Module::can_cast(node.kind()) || ast::SourceFile::can_cast(node.kind())
    })?;

    let struct_def = match strukt {
        ast::Adt::Enum(e) => Adt::Enum(ctx.sema.to_def(e)?),
        ast::Adt::Struct(s) => Adt::Struct(ctx.sema.to_def(s)?),
        ast::Adt::Union(u) => Adt::Union(ctx.sema.to_def(u)?),
    };

    let block = module.descendants().filter_map(ast::Impl::cast).find_map(|impl_blk| {
        let blk = ctx.sema.to_def(&impl_blk)?;

        // FIXME: handle e.g. `struct S<T>; impl<U> S<U> {}`
        // (we currently use the wrong type parameter)
        // also we wouldn't want to use e.g. `impl S<u32>`

        let same_ty = match blk.target_ty(db).as_adt() {
            Some(def) => def == struct_def,
            None => false,
        };
        let not_trait_impl = blk.target_trait(db).is_none();

        if !(same_ty && not_trait_impl) {
            None
        } else {
            Some(impl_blk)
        }
    });

    if let Some(ref impl_blk) = block {
        if has_fn(impl_blk, name) {
            return None;
        }
    }

    Some(block)
}

fn has_fn(imp: &ast::Impl, rhs_name: &str) -> bool {
    if let Some(il) = imp.assoc_item_list() {
        for item in il.assoc_items() {
            if let ast::AssocItem::Fn(f) = item {
                if let Some(name) = f.name() {
                    if name.text().eq_ignore_ascii_case(rhs_name) {
                        return true;
                    }
                }
            }
        }
    }

    false
}

/// Find the start of the `impl` block for the given `ast::Impl`.
//
// FIXME: this partially overlaps with `find_struct_impl`
pub(crate) fn find_impl_block_start(impl_def: ast::Impl, buf: &mut String) -> Option<TextSize> {
    buf.push('\n');
    let start = impl_def.assoc_item_list().and_then(|it| it.l_curly_token())?.text_range().end();
    Some(start)
}

/// Find the end of the `impl` block for the given `ast::Impl`.
//
// FIXME: this partially overlaps with `find_struct_impl`
pub(crate) fn find_impl_block_end(impl_def: ast::Impl, buf: &mut String) -> Option<TextSize> {
    buf.push('\n');
    let end = impl_def
        .assoc_item_list()
        .and_then(|it| it.r_curly_token())?
        .prev_sibling_or_token()?
        .text_range()
        .end();
    Some(end)
}

// Generates the surrounding `impl Type { <code> }` including type and lifetime
// parameters
pub(crate) fn generate_impl_text(adt: &ast::Adt, code: &str) -> String {
    generate_impl_text_inner(adt, None, code)
}

// Generates the surrounding `impl <trait> for Type { <code> }` including type
// and lifetime parameters
pub(crate) fn generate_trait_impl_text(adt: &ast::Adt, trait_text: &str, code: &str) -> String {
    generate_impl_text_inner(adt, Some(trait_text), code)
}

fn generate_impl_text_inner(adt: &ast::Adt, trait_text: Option<&str>, code: &str) -> String {
    let generic_params = adt.generic_param_list();
    let mut buf = String::with_capacity(code.len());
    buf.push_str("\n\n");
    adt.attrs()
        .filter(|attr| attr.as_simple_call().map(|(name, _arg)| name == "cfg").unwrap_or(false))
        .for_each(|attr| buf.push_str(format!("{}\n", attr.to_string()).as_str()));
    buf.push_str("impl");
    if let Some(generic_params) = &generic_params {
        let lifetimes = generic_params.lifetime_params().map(|lt| format!("{}", lt.syntax()));
        let type_params = generic_params.type_params().map(|type_param| {
            let mut buf = String::new();
            if let Some(it) = type_param.name() {
                format_to!(buf, "{}", it.syntax());
            }
            if let Some(it) = type_param.colon_token() {
                format_to!(buf, "{} ", it);
            }
            if let Some(it) = type_param.type_bound_list() {
                format_to!(buf, "{}", it.syntax());
            }
            buf
        });
        let generics = lifetimes.chain(type_params).format(", ");
        format_to!(buf, "<{}>", generics);
    }
    buf.push(' ');
    if let Some(trait_text) = trait_text {
        buf.push_str(trait_text);
        buf.push_str(" for ");
    }
    buf.push_str(adt.name().unwrap().text());
    if let Some(generic_params) = generic_params {
        let lifetime_params = generic_params
            .lifetime_params()
            .filter_map(|it| it.lifetime())
            .map(|it| SmolStr::from(it.text()));
        let type_params = generic_params
            .type_params()
            .filter_map(|it| it.name())
            .map(|it| SmolStr::from(it.text()));
        format_to!(buf, "<{}>", lifetime_params.chain(type_params).format(", "))
    }

    match adt.where_clause() {
        Some(where_clause) => {
            format_to!(buf, "\n{}\n{{\n{}\n}}", where_clause, code);
        }
        None => {
            format_to!(buf, " {{\n{}\n}}", code);
        }
    }

    buf
}

pub(crate) fn add_method_to_adt(
    builder: &mut AssistBuilder,
    adt: &ast::Adt,
    impl_def: Option<ast::Impl>,
    method: &str,
) {
    let mut buf = String::with_capacity(method.len() + 2);
    if impl_def.is_some() {
        buf.push('\n');
    }
    buf.push_str(method);

    let start_offset = impl_def
        .and_then(|impl_def| find_impl_block_end(impl_def, &mut buf))
        .unwrap_or_else(|| {
            buf = generate_impl_text(&adt, &buf);
            adt.syntax().text_range().end()
        });

    builder.insert(start_offset, buf);
}
