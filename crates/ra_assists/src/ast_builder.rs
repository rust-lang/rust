use itertools::Itertools;

use ra_syntax::{ast, AstNode, SourceFile};

pub struct AstBuilder<N: AstNode> {
    _phantom: std::marker::PhantomData<N>,
}

impl AstBuilder<ast::RecordField> {
    pub fn from_pieces(name: ast::NameRef, expr: Option<ast::Expr>) -> ast::RecordField {
        match expr {
            Some(expr) => Self::from_text(&format!("{}: {}", name.syntax(), expr.syntax())),
            None => Self::from_text(&name.syntax().to_string()),
        }
    }

    fn from_text(text: &str) -> ast::RecordField {
        ast_node_from_file_text(&format!("fn f() {{ S {{ {}, }} }}", text))
    }
}

impl AstBuilder<ast::Block> {
    pub fn single_expr(e: ast::Expr) -> ast::Block {
        Self::from_text(&format!("{{ {} }}", e.syntax()))
    }

    fn from_text(text: &str) -> ast::Block {
        ast_node_from_file_text(&format!("fn f() {}", text))
    }
}

impl AstBuilder<ast::Expr> {
    pub fn unit() -> ast::Expr {
        Self::from_text("()")
    }

    pub fn unimplemented() -> ast::Expr {
        Self::from_text("unimplemented!()")
    }

    fn from_text(text: &str) -> ast::Expr {
        ast_node_from_file_text(&format!("const C: () = {};", text))
    }
}

impl AstBuilder<ast::NameRef> {
    pub fn new(text: &str) -> ast::NameRef {
        ast_node_from_file_text(&format!("fn f() {{ {}; }}", text))
    }
}

impl AstBuilder<ast::Path> {
    pub fn from_name(name: ast::Name) -> ast::Path {
        let name = name.syntax().to_string();
        Self::from_text(name.as_str())
    }

    pub fn from_pieces(enum_name: ast::Name, var_name: ast::Name) -> ast::Path {
        Self::from_text(&format!("{}::{}", enum_name.syntax(), var_name.syntax()))
    }

    fn from_text(text: &str) -> ast::Path {
        ast_node_from_file_text(text)
    }
}

impl AstBuilder<ast::BindPat> {
    pub fn from_name(name: ast::Name) -> ast::BindPat {
        Self::from_text(name.text())
    }

    fn from_text(text: &str) -> ast::BindPat {
        ast_node_from_file_text(&format!("fn f({}: ())", text))
    }
}

impl AstBuilder<ast::PlaceholderPat> {
    pub fn placeholder() -> ast::PlaceholderPat {
        Self::from_text("_")
    }

    fn from_text(text: &str) -> ast::PlaceholderPat {
        ast_node_from_file_text(&format!("fn f({}: ())", text))
    }
}

impl AstBuilder<ast::TupleStructPat> {
    pub fn from_pieces(
        path: ast::Path,
        pats: impl Iterator<Item = ast::Pat>,
    ) -> ast::TupleStructPat {
        let pats_str = pats.map(|p| p.syntax().to_string()).collect::<Vec<_>>().join(", ");
        Self::from_text(&format!("{}({})", path.syntax(), pats_str))
    }

    fn from_text(text: &str) -> ast::TupleStructPat {
        ast_node_from_file_text(&format!("fn f({}: ())", text))
    }
}

impl AstBuilder<ast::RecordPat> {
    pub fn from_pieces(path: ast::Path, pats: impl Iterator<Item = ast::Pat>) -> ast::RecordPat {
        let pats_str = pats.map(|p| p.syntax().to_string()).collect::<Vec<_>>().join(", ");
        Self::from_text(&format!("{}{{ {} }}", path.syntax(), pats_str))
    }

    fn from_text(text: &str) -> ast::RecordPat {
        ast_node_from_file_text(&format!("fn f({}: ())", text))
    }
}

impl AstBuilder<ast::PathPat> {
    pub fn from_path(path: ast::Path) -> ast::PathPat {
        let path_str = path.syntax().text().to_string();
        Self::from_text(path_str.as_str())
    }

    fn from_text(text: &str) -> ast::PathPat {
        ast_node_from_file_text(&format!("fn f({}: ())", text))
    }
}

impl AstBuilder<ast::MatchArm> {
    pub fn from_pieces(pats: impl Iterator<Item = ast::Pat>, expr: ast::Expr) -> ast::MatchArm {
        let pats_str = pats.map(|p| p.syntax().to_string()).join(" | ");
        Self::from_text(&format!("{} => {}", pats_str, expr.syntax()))
    }

    fn from_text(text: &str) -> ast::MatchArm {
        ast_node_from_file_text(&format!("fn f() {{ match () {{{}}} }}", text))
    }
}

impl AstBuilder<ast::MatchArmList> {
    pub fn from_arms(arms: impl Iterator<Item = ast::MatchArm>) -> ast::MatchArmList {
        let arms_str = arms.map(|arm| format!("\n    {}", arm.syntax())).join(",");
        Self::from_text(&format!("{},\n", arms_str))
    }

    fn from_text(text: &str) -> ast::MatchArmList {
        ast_node_from_file_text(&format!("fn f() {{ match () {{{}}} }}", text))
    }
}

impl AstBuilder<ast::WherePred> {
    pub fn from_pieces(
        path: ast::Path,
        bounds: impl Iterator<Item = ast::TypeBound>,
    ) -> ast::WherePred {
        let bounds = bounds.map(|b| b.syntax().to_string()).collect::<Vec<_>>().join(" + ");
        Self::from_text(&format!("{}: {}", path.syntax(), bounds))
    }

    fn from_text(text: &str) -> ast::WherePred {
        ast_node_from_file_text(&format!("fn f() where {} {{ }}", text))
    }
}

impl AstBuilder<ast::WhereClause> {
    pub fn from_predicates(preds: impl Iterator<Item = ast::WherePred>) -> ast::WhereClause {
        let preds = preds.map(|p| p.syntax().to_string()).collect::<Vec<_>>().join(", ");
        Self::from_text(preds.as_str())
    }

    fn from_text(text: &str) -> ast::WhereClause {
        ast_node_from_file_text(&format!("fn f() where {} {{ }}", text))
    }
}

fn ast_node_from_file_text<N: AstNode>(text: &str) -> N {
    let parse = SourceFile::parse(text);
    let res = parse.tree().syntax().descendants().find_map(N::cast).unwrap();
    res
}

pub(crate) mod tokens {
    use once_cell::sync::Lazy;
    use ra_syntax::{AstNode, Parse, SourceFile, SyntaxKind::*, SyntaxToken, T};

    static SOURCE_FILE: Lazy<Parse<SourceFile>> = Lazy::new(|| SourceFile::parse(",\n; ;"));

    pub(crate) fn comma() -> SyntaxToken {
        SOURCE_FILE
            .tree()
            .syntax()
            .descendants_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == T![,])
            .unwrap()
    }

    pub(crate) fn single_space() -> SyntaxToken {
        SOURCE_FILE
            .tree()
            .syntax()
            .descendants_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == WHITESPACE && it.text().as_str() == " ")
            .unwrap()
    }

    #[allow(unused)]
    pub(crate) fn single_newline() -> SyntaxToken {
        SOURCE_FILE
            .tree()
            .syntax()
            .descendants_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == WHITESPACE && it.text().as_str() == "\n")
            .unwrap()
    }

    pub(crate) struct WsBuilder(SourceFile);

    impl WsBuilder {
        pub(crate) fn new(text: &str) -> WsBuilder {
            WsBuilder(SourceFile::parse(text).ok().unwrap())
        }
        pub(crate) fn ws(&self) -> SyntaxToken {
            self.0.syntax().first_child_or_token().unwrap().into_token().unwrap()
        }
    }

}
