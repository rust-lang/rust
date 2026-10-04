//! When specifying SSR rule, you generally want to map one *kind* of thing to
//! the same kind of thing: path to path, expression to expression, type to
//! type.
//!
//! The problem is, while this *kind* is generally obvious to the human, the ide
//! needs to determine it somehow. We do this in a stupid way -- by pasting SSR
//! rule into different contexts and checking what works.

use syntax::{AstNode, SyntaxNode, ast, syntax_editor::SyntaxEditor};

pub(crate) fn ty(s: &str) -> Result<SyntaxNode, ()> {
    fragment::<ast::Type>("type T = {};", s)
}

pub(crate) fn item(s: &str) -> Result<SyntaxNode, ()> {
    fragment::<ast::Item>("{}", s)
}

pub(crate) fn pat(s: &str) -> Result<SyntaxNode, ()> {
    fragment::<ast::Pat>("const _: () = {let {} = ();};", s)
}

pub(crate) fn expr(s: &str) -> Result<SyntaxNode, ()> {
    fragment::<ast::Expr>("const _: () = {};", s)
}

pub(crate) fn stmt(s: &str) -> Result<SyntaxNode, ()> {
    let template = "const _: () = { {}; };";
    let input = template.replace("{}", s);
    let parse = syntax::SourceFile::parse(&input, syntax::Edition::CURRENT);
    if !parse.errors().is_empty() {
        return Err(());
    }
    let node = parse.tree().syntax().descendants().skip(2).find_map(ast::Stmt::cast).ok_or(())?;
    let (editor, node) = SyntaxEditor::new(node.syntax().clone());
    let node = ast::Stmt::cast(node).ok_or(())?;
    if !s.ends_with(';')
        && node.to_string().ends_with(';')
        && let Some(token) = node.syntax().last_token()
    {
        editor.delete(token);
    }
    let node = editor.finish().new_root().clone();
    if node.to_string() != s {
        return Err(());
    }
    Ok(node)
}

fn fragment<T: AstNode>(template: &str, s: &str) -> Result<SyntaxNode, ()> {
    let s = s.trim();
    let input = template.replace("{}", s);
    let parse = syntax::SourceFile::parse(&input, syntax::Edition::CURRENT);
    if !parse.errors().is_empty() {
        return Err(());
    }
    let node = parse.tree().syntax().descendants().find_map(T::cast).ok_or(())?;
    let (_, node) = SyntaxEditor::new(node.syntax().clone());
    if node.text() != s {
        return Err(());
    }
    Ok(node)
}
