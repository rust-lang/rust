//! Utilities for formatting macro expanded nodes until we get a proper formatter.
use syntax::{
    NodeOrToken,
    SyntaxKind::{self, *},
    SyntaxNode, SyntaxToken, T, WalkEvent,
    ast::make,
    ted::{self, Position},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrettifyWsKind {
    Space,
    Indent(usize),
    Newline,
}

/// Renders a [`SyntaxNode`] with whitespace inserted between tokens that require them.
///
/// This is an internal API that is only exported because `mbe` needs it for tests and cannot depend
/// on `hir-expand`. For any purpose other than tests, you are supposed to use the `prettify_macro_expansion`
/// from `hir-expand` that handles `$crate` for you.
#[deprecated = "use `hir_expand::prettify_macro_expansion()` instead"]
pub fn prettify_macro_expansion(
    syn: SyntaxNode,
    dollar_crate_replacement: &mut dyn FnMut(&SyntaxToken) -> Option<SyntaxToken>,
    inspect_mods: impl FnOnce(&[(Position, PrettifyWsKind)]),
) -> SyntaxNode {
    let mut indent = 0;
    let mut last: Option<SyntaxKind> = None;
    let mut mods = Vec::new();
    let mut dollar_crate_replacements = Vec::new();
    let syn = syn.clone_subtree().clone_for_update();

    let before = Position::before;
    let after = Position::after;

    let do_indent = |pos: fn(_) -> Position, token: &SyntaxToken, indent| {
        (pos(token.clone()), PrettifyWsKind::Indent(indent))
    };
    let do_ws =
        |pos: fn(_) -> Position, token: &SyntaxToken| (pos(token.clone()), PrettifyWsKind::Space);
    let do_nl =
        |pos: fn(_) -> Position, token: &SyntaxToken| (pos(token.clone()), PrettifyWsKind::Newline);

    for event in syn.preorder_with_tokens() {
        let token = match event {
            WalkEvent::Enter(NodeOrToken::Token(token)) => token,
            WalkEvent::Leave(NodeOrToken::Node(node))
                if matches!(
                    node.kind(),
                    ATTR | MATCH_ARM | STRUCT | ENUM | UNION | FN | IMPL | MACRO_RULES
                ) =>
            {
                if indent > 0 {
                    mods.push((Position::after(node.clone()), PrettifyWsKind::Indent(indent)));
                }
                if node.parent().is_some() {
                    mods.push((Position::after(node), PrettifyWsKind::Newline));
                }
                continue;
            }
            _ => continue,
        };
        if token.kind() == SyntaxKind::IDENT
            && token.text() == "$crate"
            && let Some(replacement) = dollar_crate_replacement(&token)
        {
            dollar_crate_replacements.push((token.clone(), replacement));
        }
        let tok = &token;

        let is_next = |f: fn(SyntaxKind) -> bool, default| -> bool {
            tok.next_token().map(|it| f(it.kind())).unwrap_or(default)
        };
        let is_last =
            |f: fn(SyntaxKind) -> bool, default| -> bool { last.map(f).unwrap_or(default) };

        match tok.kind() {
            k if is_text(k)
                && is_next(|it| !it.is_punct() || matches!(it, T![_] | T![#]), false) =>
            {
                mods.push(do_ws(after, tok));
            }
            L_CURLY if is_next(|it| it != R_CURLY, true) => {
                indent += 1;
                if is_last(is_text, false) {
                    mods.push(do_ws(before, tok));
                }

                mods.push(do_indent(after, tok, indent));
                mods.push(do_nl(after, tok));
            }
            R_CURLY if is_last(|it| it != L_CURLY, true) => {
                indent = indent.saturating_sub(1);

                if indent > 0 {
                    mods.push(do_indent(before, tok, indent));
                }
                mods.push(do_nl(before, tok));
            }
            R_CURLY => {
                if indent > 0 {
                    mods.push(do_indent(after, tok, indent));
                }
                mods.push(do_nl(after, tok));
            }
            LIFETIME_IDENT if is_next(is_text, true) => {
                mods.push(do_ws(after, tok));
            }
            AS_KW | DYN_KW | IMPL_KW | CONST_KW | MUT_KW => {
                mods.push(do_ws(after, tok));
            }
            T![;] if is_next(|it| it != R_CURLY, true) => {
                if indent > 0 {
                    mods.push(do_indent(after, tok, indent));
                }
                mods.push(do_nl(after, tok));
            }
            T![=] if is_next(|it| it == T![>], false) => {
                // FIXME: this branch is for `=>` in macro_rules!, which is currently parsed as
                // two separate symbols.
                mods.push(do_ws(before, tok));
                mods.push(do_ws(after, &tok.next_token().unwrap()));
            }
            T![->] | T![=] | T![=>] => {
                mods.push(do_ws(before, tok));
                mods.push(do_ws(after, tok));
            }
            T![!] if is_last(|it| it == MACRO_RULES_KW, false) && is_next(is_text, false) => {
                mods.push(do_ws(after, tok));
            }
            _ => (),
        }

        last = Some(tok.kind());
    }

    inspect_mods(&mods);
    for (pos, insert) in mods {
        ted::insert_raw(
            pos,
            match insert {
                PrettifyWsKind::Space => make::tokens::single_space(),
                PrettifyWsKind::Indent(indent) => make::tokens::whitespace(&" ".repeat(4 * indent)),
                PrettifyWsKind::Newline => make::tokens::single_newline(),
            },
        );
    }
    for (old, new) in dollar_crate_replacements {
        ted::replace(old, new);
    }

    if let Some(it) = syn.last_token().filter(|it| it.kind() == SyntaxKind::WHITESPACE) {
        ted::remove(it);
    }

    syn
}

fn is_text(k: SyntaxKind) -> bool {
    // Consider all keywords in all editions.
    k.is_any_identifier() || k.is_literal() || k == UNDERSCORE
}
