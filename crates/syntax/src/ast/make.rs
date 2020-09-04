//! This module contains free-standing functions for creating AST fragments out
//! of smaller pieces.
//!
//! Note that all functions here intended to be stupid constructors, which just
//! assemble a finish node from immediate children. If you want to do something
//! smarter than that, it probably doesn't belong in this module.
use itertools::Itertools;
use stdx::format_to;

use crate::{ast, AstNode, SourceFile, SyntaxKind, SyntaxNode, SyntaxToken};

pub fn name(text: &str) -> ast::Name {
    ast_from_text(&format!("mod {};", text))
}

pub fn name_ref(text: &str) -> ast::NameRef {
    ast_from_text(&format!("fn f() {{ {}; }}", text))
}

pub fn ty(text: &str) -> ast::Type {
    ast_from_text(&format!("impl {} for D {{}};", text))
}

pub fn path_segment(name_ref: ast::NameRef) -> ast::PathSegment {
    ast_from_text(&format!("use {};", name_ref))
}
pub fn path_segment_self() -> ast::PathSegment {
    ast_from_text("use self;")
}
pub fn path_unqualified(segment: ast::PathSegment) -> ast::Path {
    path_from_text(&format!("use {}", segment))
}
pub fn path_qualified(qual: ast::Path, segment: ast::PathSegment) -> ast::Path {
    path_from_text(&format!("{}::{}", qual, segment))
}
// FIXME: make this private
pub fn path_from_text(text: &str) -> ast::Path {
    ast_from_text(text)
}

pub fn use_tree(
    path: ast::Path,
    use_tree_list: Option<ast::UseTreeList>,
    alias: Option<ast::Rename>,
    add_star: bool,
) -> ast::UseTree {
    let mut buf = "use ".to_string();
    buf += &path.syntax().to_string();
    if let Some(use_tree_list) = use_tree_list {
        format_to!(buf, "::{}", use_tree_list);
    }
    if add_star {
        buf += "::*";
    }

    if let Some(alias) = alias {
        format_to!(buf, " {}", alias);
    }
    ast_from_text(&buf)
}

pub fn use_tree_list(use_trees: impl IntoIterator<Item = ast::UseTree>) -> ast::UseTreeList {
    let use_trees = use_trees.into_iter().map(|it| it.syntax().clone()).join(", ");
    ast_from_text(&format!("use {{{}}};", use_trees))
}

pub fn use_(use_tree: ast::UseTree) -> ast::Use {
    ast_from_text(&format!("use {};", use_tree))
}

pub fn record_expr_field(name: ast::NameRef, expr: Option<ast::Expr>) -> ast::RecordExprField {
    return match expr {
        Some(expr) => from_text(&format!("{}: {}", name, expr)),
        None => from_text(&name.to_string()),
    };

    fn from_text(text: &str) -> ast::RecordExprField {
        ast_from_text(&format!("fn f() {{ S {{ {}, }} }}", text))
    }
}

pub fn record_field(name: ast::NameRef, ty: ast::Type) -> ast::RecordField {
    ast_from_text(&format!("struct S {{ {}: {}, }}", name, ty))
}

pub fn block_expr(
    stmts: impl IntoIterator<Item = ast::Stmt>,
    tail_expr: Option<ast::Expr>,
) -> ast::BlockExpr {
    let mut buf = "{\n".to_string();
    for stmt in stmts.into_iter() {
        format_to!(buf, "    {}\n", stmt);
    }
    if let Some(tail_expr) = tail_expr {
        format_to!(buf, "    {}\n", tail_expr)
    }
    buf += "}";
    ast_from_text(&format!("fn f() {}", buf))
}

pub fn expr_unit() -> ast::Expr {
    expr_from_text("()")
}
pub fn expr_empty_block() -> ast::Expr {
    expr_from_text("{}")
}
pub fn expr_unimplemented() -> ast::Expr {
    expr_from_text("unimplemented!()")
}
pub fn expr_unreachable() -> ast::Expr {
    expr_from_text("unreachable!()")
}
pub fn expr_todo() -> ast::Expr {
    expr_from_text("todo!()")
}
pub fn expr_path(path: ast::Path) -> ast::Expr {
    expr_from_text(&path.to_string())
}
pub fn expr_continue() -> ast::Expr {
    expr_from_text("continue")
}
pub fn expr_break() -> ast::Expr {
    expr_from_text("break")
}
pub fn expr_return() -> ast::Expr {
    expr_from_text("return")
}
pub fn expr_match(expr: ast::Expr, match_arm_list: ast::MatchArmList) -> ast::Expr {
    expr_from_text(&format!("match {} {}", expr, match_arm_list))
}
pub fn expr_if(condition: ast::Condition, then_branch: ast::BlockExpr) -> ast::Expr {
    expr_from_text(&format!("if {} {}", condition, then_branch))
}
pub fn expr_prefix(op: SyntaxKind, expr: ast::Expr) -> ast::Expr {
    let token = token(op);
    expr_from_text(&format!("{}{}", token, expr))
}
pub fn expr_call(f: ast::Expr, arg_list: ast::ArgList) -> ast::Expr {
    expr_from_text(&format!("{}{}", f, arg_list))
}
pub fn expr_method_call(receiver: ast::Expr, method: &str, arg_list: ast::ArgList) -> ast::Expr {
    expr_from_text(&format!("{}.{}{}", receiver, method, arg_list))
}
fn expr_from_text(text: &str) -> ast::Expr {
    ast_from_text(&format!("const C: () = {};", text))
}

pub fn condition(expr: ast::Expr, pattern: Option<ast::Pat>) -> ast::Condition {
    match pattern {
        None => ast_from_text(&format!("const _: () = while {} {{}};", expr)),
        Some(pattern) => {
            ast_from_text(&format!("const _: () = while let {} = {} {{}};", pattern, expr))
        }
    }
}

pub fn arg_list(args: impl IntoIterator<Item = ast::Expr>) -> ast::ArgList {
    ast_from_text(&format!("fn main() {{ ()({}) }}", args.into_iter().format(", ")))
}

pub fn ident_pat(name: ast::Name) -> ast::IdentPat {
    return from_text(name.text());

    fn from_text(text: &str) -> ast::IdentPat {
        ast_from_text(&format!("fn f({}: ())", text))
    }
}

pub fn wildcard_pat() -> ast::WildcardPat {
    return from_text("_");

    fn from_text(text: &str) -> ast::WildcardPat {
        ast_from_text(&format!("fn f({}: ())", text))
    }
}

/// Creates a tuple of patterns from an interator of patterns.
///
/// Invariant: `pats` must be length > 1
///
/// FIXME handle `pats` length == 1
pub fn tuple_pat(pats: impl IntoIterator<Item = ast::Pat>) -> ast::TuplePat {
    let pats_str = pats.into_iter().map(|p| p.to_string()).join(", ");
    return from_text(&format!("({})", pats_str));

    fn from_text(text: &str) -> ast::TuplePat {
        ast_from_text(&format!("fn f({}: ())", text))
    }
}

pub fn tuple_struct_pat(
    path: ast::Path,
    pats: impl IntoIterator<Item = ast::Pat>,
) -> ast::TupleStructPat {
    let pats_str = pats.into_iter().join(", ");
    return from_text(&format!("{}({})", path, pats_str));

    fn from_text(text: &str) -> ast::TupleStructPat {
        ast_from_text(&format!("fn f({}: ())", text))
    }
}

pub fn record_pat(path: ast::Path, pats: impl IntoIterator<Item = ast::Pat>) -> ast::RecordPat {
    let pats_str = pats.into_iter().join(", ");
    return from_text(&format!("{} {{ {} }}", path, pats_str));

    fn from_text(text: &str) -> ast::RecordPat {
        ast_from_text(&format!("fn f({}: ())", text))
    }
}

/// Returns a `BindPat` if the path has just one segment, a `PathPat` otherwise.
pub fn path_pat(path: ast::Path) -> ast::Pat {
    return from_text(&path.to_string());
    fn from_text(text: &str) -> ast::Pat {
        ast_from_text(&format!("fn f({}: ())", text))
    }
}

pub fn match_arm(pats: impl IntoIterator<Item = ast::Pat>, expr: ast::Expr) -> ast::MatchArm {
    let pats_str = pats.into_iter().join(" | ");
    return from_text(&format!("{} => {}", pats_str, expr));

    fn from_text(text: &str) -> ast::MatchArm {
        ast_from_text(&format!("fn f() {{ match () {{{}}} }}", text))
    }
}

pub fn match_arm_list(arms: impl IntoIterator<Item = ast::MatchArm>) -> ast::MatchArmList {
    let arms_str = arms
        .into_iter()
        .map(|arm| {
            let needs_comma = arm.expr().map_or(true, |it| !it.is_block_like());
            let comma = if needs_comma { "," } else { "" };
            format!("    {}{}\n", arm.syntax(), comma)
        })
        .collect::<String>();
    return from_text(&arms_str);

    fn from_text(text: &str) -> ast::MatchArmList {
        ast_from_text(&format!("fn f() {{ match () {{\n{}}} }}", text))
    }
}

pub fn where_pred(
    path: ast::Path,
    bounds: impl IntoIterator<Item = ast::TypeBound>,
) -> ast::WherePred {
    let bounds = bounds.into_iter().join(" + ");
    return from_text(&format!("{}: {}", path, bounds));

    fn from_text(text: &str) -> ast::WherePred {
        ast_from_text(&format!("fn f() where {} {{ }}", text))
    }
}

pub fn where_clause(preds: impl IntoIterator<Item = ast::WherePred>) -> ast::WhereClause {
    let preds = preds.into_iter().join(", ");
    return from_text(preds.as_str());

    fn from_text(text: &str) -> ast::WhereClause {
        ast_from_text(&format!("fn f() where {} {{ }}", text))
    }
}

pub fn let_stmt(pattern: ast::Pat, initializer: Option<ast::Expr>) -> ast::LetStmt {
    let text = match initializer {
        Some(it) => format!("let {} = {};", pattern, it),
        None => format!("let {};", pattern),
    };
    ast_from_text(&format!("fn f() {{ {} }}", text))
}
pub fn expr_stmt(expr: ast::Expr) -> ast::ExprStmt {
    let semi = if expr.is_block_like() { "" } else { ";" };
    ast_from_text(&format!("fn f() {{ {}{} (); }}", expr, semi))
}

pub fn token(kind: SyntaxKind) -> SyntaxToken {
    tokens::SOURCE_FILE
        .tree()
        .syntax()
        .descendants_with_tokens()
        .filter_map(|it| it.into_token())
        .find(|it| it.kind() == kind)
        .unwrap_or_else(|| panic!("unhandled token: {:?}", kind))
}

pub fn param(name: String, ty: String) -> ast::Param {
    ast_from_text(&format!("fn f({}: {}) {{ }}", name, ty))
}

pub fn param_list(pats: impl IntoIterator<Item = ast::Param>) -> ast::ParamList {
    let args = pats.into_iter().join(", ");
    ast_from_text(&format!("fn f({}) {{ }}", args))
}

pub fn generic_param(name: String, ty: Option<ast::TypeBoundList>) -> ast::GenericParam {
    let bound = match ty {
        Some(it) => format!(": {}", it),
        None => String::new(),
    };
    ast_from_text(&format!("fn f<{}{}>() {{ }}", name, bound))
}

pub fn generic_param_list(
    pats: impl IntoIterator<Item = ast::GenericParam>,
) -> ast::GenericParamList {
    let args = pats.into_iter().join(", ");
    ast_from_text(&format!("fn f<{}>() {{ }}", args))
}

pub fn visibility_pub_crate() -> ast::Visibility {
    ast_from_text("pub(crate) struct S")
}

pub fn fn_(
    visibility: Option<ast::Visibility>,
    fn_name: ast::Name,
    type_params: Option<ast::GenericParamList>,
    params: ast::ParamList,
    body: ast::BlockExpr,
) -> ast::Fn {
    let type_params =
        if let Some(type_params) = type_params { format!("<{}>", type_params) } else { "".into() };
    let visibility = match visibility {
        None => String::new(),
        Some(it) => format!("{} ", it),
    };
    ast_from_text(&format!("{}fn {}{}{} {}", visibility, fn_name, type_params, params, body))
}

fn ast_from_text<N: AstNode>(text: &str) -> N {
    let parse = SourceFile::parse(text);
    let node = match parse.tree().syntax().descendants().find_map(N::cast) {
        Some(it) => it,
        None => {
            panic!("Failed to make ast node `{}` from text {}", std::any::type_name::<N>(), text)
        }
    };
    let node = node.syntax().clone();
    let node = unroot(node);
    let node = N::cast(node).unwrap();
    assert_eq!(node.syntax().text_range().start(), 0.into());
    node
}

fn unroot(n: SyntaxNode) -> SyntaxNode {
    SyntaxNode::new_root(n.green().clone())
}

pub mod tokens {
    use once_cell::sync::Lazy;

    use crate::{ast, AstNode, Parse, SourceFile, SyntaxKind::*, SyntaxToken};

    pub(super) static SOURCE_FILE: Lazy<Parse<SourceFile>> =
        Lazy::new(|| SourceFile::parse("const C: <()>::Item = (1 != 1, 2 == 2, !true)\n;\n\n"));

    pub fn single_space() -> SyntaxToken {
        SOURCE_FILE
            .tree()
            .syntax()
            .descendants_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == WHITESPACE && it.text().as_str() == " ")
            .unwrap()
    }

    pub fn whitespace(text: &str) -> SyntaxToken {
        assert!(text.trim().is_empty());
        let sf = SourceFile::parse(text).ok().unwrap();
        sf.syntax().first_child_or_token().unwrap().into_token().unwrap()
    }

    pub fn doc_comment(text: &str) -> SyntaxToken {
        assert!(!text.trim().is_empty());
        let sf = SourceFile::parse(text).ok().unwrap();
        sf.syntax().first_child_or_token().unwrap().into_token().unwrap()
    }

    pub fn literal(text: &str) -> SyntaxToken {
        assert_eq!(text.trim(), text);
        let lit: ast::Literal = super::ast_from_text(&format!("fn f() {{ let _ = {}; }}", text));
        lit.syntax().first_child_or_token().unwrap().into_token().unwrap()
    }

    pub fn single_newline() -> SyntaxToken {
        SOURCE_FILE
            .tree()
            .syntax()
            .descendants_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == WHITESPACE && it.text().as_str() == "\n")
            .unwrap()
    }

    pub fn blank_line() -> SyntaxToken {
        SOURCE_FILE
            .tree()
            .syntax()
            .descendants_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == WHITESPACE && it.text().as_str() == "\n\n")
            .unwrap()
    }

    pub struct WsBuilder(SourceFile);

    impl WsBuilder {
        pub fn new(text: &str) -> WsBuilder {
            WsBuilder(SourceFile::parse(text).ok().unwrap())
        }
        pub fn ws(&self) -> SyntaxToken {
            self.0.syntax().first_child_or_token().unwrap().into_token().unwrap()
        }
    }
}
