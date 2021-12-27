//! When specifying SSR rule, you generally want to map one *kind* of thing to
//! the same kind of thing: path to path, expression to expression, type to
//! type.
//!
//! The problem is, while this *kind* is generally obvious to the human, the ide
//! needs to determine it somehow. We do this in a stupid way -- by pasting SSR
//! rule into different contexts and checking what works.

use parser::SyntaxKind;
use syntax::{ast, AstNode, SyntaxNode};

pub(crate) fn ty(s: &str) -> Result<SyntaxNode, ()> {
    let template = "type T = {};";
    let input = template.replace("{}", s);
    let parse = syntax::SourceFile::parse(&input);
    if !parse.errors().is_empty() {
        return Err(());
    }
    let node = parse.tree().syntax().descendants().find_map(ast::Type::cast).ok_or(())?;
    Ok(node.syntax().clone())
}
