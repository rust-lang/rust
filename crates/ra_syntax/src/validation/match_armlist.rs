use crate::{
    ast::{self, AttrsOwner, AstNode},
    syntax_node::{
        SyntaxError,
        SyntaxErrorKind::*,
        Direction,
    },
};

pub(crate) fn validate_match_armlist(node: &ast::MatchArmList, errors: &mut Vec<SyntaxError>) {
    // Report errors for any inner attribute
    // which has a preceding matcharm or an outer attribute
    for inner_attr in node.attrs().filter(|s| s.is_inner()) {
        let any_errors = inner_attr.syntax().siblings(Direction::Prev).any(|s| {
            match (ast::MatchArm::cast(s), ast::Attr::cast(s)) {
                (Some(_), _) => true,
                // Outer attributes which preceed an inner attribute are not allowed
                (_, Some(a)) if !a.is_inner() => true,
                (_, Some(_)) => false,
                (None, None) => false,
            }
        });

        if any_errors {
            errors.push(SyntaxError::new(InvalidMatchInnerAttr, inner_attr.syntax().range()));
        }
    }
}
