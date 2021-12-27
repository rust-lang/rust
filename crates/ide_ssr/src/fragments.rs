//! When specifying SSR rule, you generally want to map one *kind* of thing to
//! the same kind of thing: path to path, expression to expression, type to
//! type.
//!
//! The problem is, while this *kind* is generally obvious to the human, the ide
//! needs to determine it somehow. We do this in a stupid way -- by pasting SSR
//! rule into different contexts and checking what works.

use syntax::{ast, AstNode, SyntaxNode};

pub(crate) fn ty(s: &str) -> Result<SyntaxNode, ()> {
    let template = "type T = {};";
    let input = template.replace("{}", s);
    let parse = syntax::SourceFile::parse(&input);
    if !parse.errors().is_empty() {
        return Err(());
    }
    let node = parse.tree().syntax().descendants().find_map(ast::Type::cast).ok_or(())?;
    if node.to_string() != s {
        return Err(());
    }
    Ok(node.syntax().clone_subtree())
}

pub(crate) fn item(s: &str) -> Result<SyntaxNode, ()> {
    let template = "{}";
    let input = template.replace("{}", s);
    let parse = syntax::SourceFile::parse(&input);
    if !parse.errors().is_empty() {
        return Err(());
    }
    let node = parse.tree().syntax().descendants().find_map(ast::Item::cast).ok_or(())?;
    if node.to_string() != s {
        return Err(());
    }
    Ok(node.syntax().clone_subtree())
}

pub(crate) fn pat(s: &str) -> Result<SyntaxNode, ()> {
    let template = "const _: () = {let {} = ();};";
    let input = template.replace("{}", s);
    let parse = syntax::SourceFile::parse(&input);
    if !parse.errors().is_empty() {
        return Err(());
    }
    let node = parse.tree().syntax().descendants().find_map(ast::Pat::cast).ok_or(())?;
    if node.to_string() != s {
        return Err(());
    }
    Ok(node.syntax().clone_subtree())
}

pub(crate) fn expr(s: &str) -> Result<SyntaxNode, ()> {
    let template = "const _: () = {};";
    let input = template.replace("{}", s);
    let parse = syntax::SourceFile::parse(&input);
    if !parse.errors().is_empty() {
        return Err(());
    }
    let node = parse.tree().syntax().descendants().find_map(ast::Expr::cast).ok_or(())?;
    if node.to_string() != s {
        return Err(());
    }
    Ok(node.syntax().clone_subtree())
}

pub(crate) fn stmt(s: &str) -> Result<SyntaxNode, ()> {
    let template = "const _: () = { {}; };";
    let input = template.replace("{}", s);
    let parse = syntax::SourceFile::parse(&input);
    if !parse.errors().is_empty() {
        return Err(());
    }
    let mut node =
        parse.tree().syntax().descendants().skip(2).find_map(ast::Stmt::cast).ok_or(())?;
    if !s.ends_with(';') && node.to_string().ends_with(';') {
        node = node.clone_for_update();
        node.syntax().last_token().map(|it| it.detach());
    }
    if node.to_string() != s {
        return Err(());
    }
    Ok(node.syntax().clone_subtree())
}
