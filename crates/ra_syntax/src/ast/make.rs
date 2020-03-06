//! This module contains free-standing functions for creating AST fragments out
//! of smaller pieces.
use itertools::Itertools;

use crate::{ast, AstNode, SourceFile, SyntaxKind, SyntaxNode, SyntaxToken};

pub fn name(text: &str) -> ast::Name {
    ast_from_text(&format!("mod {};", text))
}

pub fn name_ref(text: &str) -> ast::NameRef {
    ast_from_text(&format!("fn f() {{ {}; }}", text))
}

pub fn path_segment(name_ref: ast::NameRef) -> ast::PathSegment {
    ast_from_text(&format!("use {};", name_ref.syntax()))
}
pub fn path_unqualified(segment: ast::PathSegment) -> ast::Path {
    path_from_text(&format!("use {}", segment.syntax()))
}
pub fn path_qualified(qual: ast::Path, segment: ast::PathSegment) -> ast::Path {
    path_from_text(&format!("{}::{}", qual.syntax(), segment.syntax()))
}
fn path_from_text(text: &str) -> ast::Path {
    ast_from_text(text)
}

pub fn use_tree(
    path: ast::Path,
    use_tree_list: Option<ast::UseTreeList>,
    alias: Option<ast::Alias>,
) -> ast::UseTree {
    let mut buf = "use ".to_string();
    buf += &path.syntax().to_string();
    if let Some(use_tree_list) = use_tree_list {
        buf += &format!("::{}", use_tree_list.syntax());
    }
    if let Some(alias) = alias {
        buf += &format!(" {}", alias.syntax());
    }
    ast_from_text(&buf)
}

pub fn use_tree_list(use_trees: impl IntoIterator<Item = ast::UseTree>) -> ast::UseTreeList {
    let use_trees = use_trees.into_iter().map(|it| it.syntax().clone()).join(", ");
    ast_from_text(&format!("use {{{}}};", use_trees))
}

pub fn record_field(name: ast::NameRef, expr: Option<ast::Expr>) -> ast::RecordField {
    return match expr {
        Some(expr) => from_text(&format!("{}: {}", name.syntax(), expr.syntax())),
        None => from_text(&name.syntax().to_string()),
    };

    fn from_text(text: &str) -> ast::RecordField {
        ast_from_text(&format!("fn f() {{ S {{ {}, }} }}", text))
    }
}

pub fn block_expr(
    stmts: impl IntoIterator<Item = ast::Stmt>,
    tail_expr: Option<ast::Expr>,
) -> ast::BlockExpr {
    let mut text = "{\n".to_string();
    for stmt in stmts.into_iter() {
        text += &format!("    {}\n", stmt.syntax());
    }
    if let Some(tail_expr) = tail_expr {
        text += &format!("    {}\n", tail_expr.syntax())
    }
    text += "}";
    ast_from_text(&format!("fn f() {}", text))
}

pub fn block_from_expr(e: ast::Expr) -> ast::Block {
    return from_text(&format!("{{ {} }}", e.syntax()));

    fn from_text(text: &str) -> ast::Block {
        ast_from_text(&format!("fn f() {}", text))
    }
}

pub fn expr_unit() -> ast::Expr {
    expr_from_text("()")
}
pub fn expr_unimplemented() -> ast::Expr {
    expr_from_text("unimplemented!()")
}
pub fn expr_path(path: ast::Path) -> ast::Expr {
    expr_from_text(&path.syntax().to_string())
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
    expr_from_text(&format!("match {} {}", expr.syntax(), match_arm_list.syntax()))
}
pub fn expr_if(condition: ast::Expr, then_branch: ast::BlockExpr) -> ast::Expr {
    expr_from_text(&format!("if {} {}", condition.syntax(), then_branch.syntax()))
}
pub fn expr_prefix(op: SyntaxKind, expr: ast::Expr) -> ast::Expr {
    let token = token(op);
    expr_from_text(&format!("{}{}", token, expr.syntax()))
}
pub fn expr_from_text(text: &str) -> ast::Expr {
    ast_from_text(&format!("const C: () = {};", text))
}

pub fn bind_pat(name: ast::Name) -> ast::BindPat {
    return from_text(name.text());

    fn from_text(text: &str) -> ast::BindPat {
        ast_from_text(&format!("fn f({}: ())", text))
    }
}

pub fn placeholder_pat() -> ast::PlaceholderPat {
    return from_text("_");

    fn from_text(text: &str) -> ast::PlaceholderPat {
        ast_from_text(&format!("fn f({}: ())", text))
    }
}

pub fn tuple_struct_pat(
    path: ast::Path,
    pats: impl IntoIterator<Item = ast::Pat>,
) -> ast::TupleStructPat {
    let pats_str = pats.into_iter().map(|p| p.syntax().to_string()).join(", ");
    return from_text(&format!("{}({})", path.syntax(), pats_str));

    fn from_text(text: &str) -> ast::TupleStructPat {
        ast_from_text(&format!("fn f({}: ())", text))
    }
}

pub fn record_pat(path: ast::Path, pats: impl IntoIterator<Item = ast::Pat>) -> ast::RecordPat {
    let pats_str = pats.into_iter().map(|p| p.syntax().to_string()).join(", ");
    return from_text(&format!("{} {{ {} }}", path.syntax(), pats_str));

    fn from_text(text: &str) -> ast::RecordPat {
        ast_from_text(&format!("fn f({}: ())", text))
    }
}

/// Returns a `BindPat` if the path has just one segment, a `PathPat` otherwise.
pub fn path_pat(path: ast::Path) -> ast::Pat {
    let path_str = path.syntax().text().to_string();
    return from_text(path_str.as_str());
    fn from_text(text: &str) -> ast::Pat {
        ast_from_text(&format!("fn f({}: ())", text))
    }
}

pub fn match_arm(pats: impl IntoIterator<Item = ast::Pat>, expr: ast::Expr) -> ast::MatchArm {
    let pats_str = pats.into_iter().map(|p| p.syntax().to_string()).join(" | ");
    return from_text(&format!("{} => {}", pats_str, expr.syntax()));

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
    let bounds = bounds.into_iter().map(|b| b.syntax().to_string()).join(" + ");
    return from_text(&format!("{}: {}", path.syntax(), bounds));

    fn from_text(text: &str) -> ast::WherePred {
        ast_from_text(&format!("fn f() where {} {{ }}", text))
    }
}

pub fn where_clause(preds: impl IntoIterator<Item = ast::WherePred>) -> ast::WhereClause {
    let preds = preds.into_iter().map(|p| p.syntax().to_string()).join(", ");
    return from_text(preds.as_str());

    fn from_text(text: &str) -> ast::WhereClause {
        ast_from_text(&format!("fn f() where {} {{ }}", text))
    }
}

pub fn let_stmt(pattern: ast::Pat, initializer: Option<ast::Expr>) -> ast::LetStmt {
    let text = match initializer {
        Some(it) => format!("let {} = {};", pattern.syntax(), it.syntax()),
        None => format!("let {};", pattern.syntax()),
    };
    ast_from_text(&format!("fn f() {{ {} }}", text))
}
pub fn expr_stmt(expr: ast::Expr) -> ast::ExprStmt {
    ast_from_text(&format!("fn f() {{ {}; }}", expr.syntax()))
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

fn ast_from_text<N: AstNode>(text: &str) -> N {
    let parse = SourceFile::parse(text);
    let node = parse.tree().syntax().descendants().find_map(N::cast).unwrap();
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
        Lazy::new(|| SourceFile::parse("const C: <()>::Item = (1 != 1, 2 == 2, !true)\n;"));

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
