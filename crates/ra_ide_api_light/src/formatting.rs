use ra_syntax::{
    ast, AstNode,
    SyntaxNode, SyntaxKind::*,
};

pub(crate) fn extract_trivial_expression(block: &ast::Block) -> Option<&ast::Expr> {
    let expr = block.expr()?;
    if expr.syntax().text().contains('\n') {
        return None;
    }
    let non_trivial_children = block.syntax().children().filter(|it| match it.kind() {
        WHITESPACE | L_CURLY | R_CURLY => false,
        _ => it != &expr.syntax(),
    });
    if non_trivial_children.count() > 0 {
        return None;
    }
    Some(expr)
}

pub(crate) fn compute_ws(left: &SyntaxNode, right: &SyntaxNode) -> &'static str {
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
