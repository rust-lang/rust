//! This module contains free-standing functions for creating AST fragments out
//! of smaller pieces.
use itertools::Itertools;

use crate::{ast, AstNode, SourceFile};

pub fn name_ref(text: &str) -> ast::NameRef {
    ast_from_text(&format!("fn f() {{ {}; }}", text))
}

pub fn path_from_name_ref(name_ref: ast::NameRef) -> ast::Path {
    path_from_text(&name_ref.syntax().to_string())
}
pub fn path_qualified(qual: ast::Path, name_ref: ast::NameRef) -> ast::Path {
    path_from_text(&format!("{}::{}", qual.syntax(), name_ref.syntax()))
}
fn path_from_text(text: &str) -> ast::Path {
    ast_from_text(text)
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
fn expr_from_text(text: &str) -> ast::Expr {
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
    pats: impl Iterator<Item = ast::Pat>,
) -> ast::TupleStructPat {
    let pats_str = pats.map(|p| p.syntax().to_string()).join(", ");
    return from_text(&format!("{}({})", path.syntax(), pats_str));

    fn from_text(text: &str) -> ast::TupleStructPat {
        ast_from_text(&format!("fn f({}: ())", text))
    }
}

pub fn record_pat(path: ast::Path, pats: impl Iterator<Item = ast::Pat>) -> ast::RecordPat {
    let pats_str = pats.map(|p| p.syntax().to_string()).join(", ");
    return from_text(&format!("{} {{ {} }}", path.syntax(), pats_str));

    fn from_text(text: &str) -> ast::RecordPat {
        ast_from_text(&format!("fn f({}: ())", text))
    }
}

pub fn path_pat(path: ast::Path) -> ast::PathPat {
    let path_str = path.syntax().text().to_string();
    return from_text(path_str.as_str());
    fn from_text(text: &str) -> ast::PathPat {
        ast_from_text(&format!("fn f({}: ())", text))
    }
}

pub fn match_arm(pats: impl Iterator<Item = ast::Pat>, expr: ast::Expr) -> ast::MatchArm {
    let pats_str = pats.map(|p| p.syntax().to_string()).join(" | ");
    return from_text(&format!("{} => {}", pats_str, expr.syntax()));

    fn from_text(text: &str) -> ast::MatchArm {
        ast_from_text(&format!("fn f() {{ match () {{{}}} }}", text))
    }
}

pub fn match_arm_list(arms: impl Iterator<Item = ast::MatchArm>) -> ast::MatchArmList {
    let arms_str = arms.map(|arm| format!("\n    {}", arm.syntax())).join(",");
    return from_text(&format!("{},\n", arms_str));

    fn from_text(text: &str) -> ast::MatchArmList {
        ast_from_text(&format!("fn f() {{ match () {{{}}} }}", text))
    }
}

pub fn where_pred(path: ast::Path, bounds: impl Iterator<Item = ast::TypeBound>) -> ast::WherePred {
    let bounds = bounds.map(|b| b.syntax().to_string()).join(" + ");
    return from_text(&format!("{}: {}", path.syntax(), bounds));

    fn from_text(text: &str) -> ast::WherePred {
        ast_from_text(&format!("fn f() where {} {{ }}", text))
    }
}

pub fn where_clause(preds: impl Iterator<Item = ast::WherePred>) -> ast::WhereClause {
    let preds = preds.map(|p| p.syntax().to_string()).join(", ");
    return from_text(preds.as_str());

    fn from_text(text: &str) -> ast::WhereClause {
        ast_from_text(&format!("fn f() where {} {{ }}", text))
    }
}

pub fn if_expression(condition: &ast::Expr, statement: &str) -> ast::IfExpr {
    ast_from_text(&format!(
        "fn f() {{ if !{} {{\n    {}\n}}\n}}",
        condition.syntax().text(),
        statement
    ))
}

fn ast_from_text<N: AstNode>(text: &str) -> N {
    let parse = SourceFile::parse(text);
    let res = parse.tree().syntax().descendants().find_map(N::cast).unwrap();
    res
}

pub mod tokens {
    use crate::{AstNode, Parse, SourceFile, SyntaxKind::*, SyntaxToken, T};
    use once_cell::sync::Lazy;

    static SOURCE_FILE: Lazy<Parse<SourceFile>> = Lazy::new(|| SourceFile::parse(",\n; ;"));

    pub fn comma() -> SyntaxToken {
        SOURCE_FILE
            .tree()
            .syntax()
            .descendants_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == T![,])
            .unwrap()
    }

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
