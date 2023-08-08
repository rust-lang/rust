//! Utilities for formatting macro expanded nodes until we get a proper formatter.
use syntax::{
    ast::make,
    ted::{self, Position},
    NodeOrToken,
    SyntaxKind::{self, *},
    SyntaxNode, SyntaxToken, WalkEvent, T,
};

// FIXME: It would also be cool to share logic here and in the mbe tests,
// which are pretty unreadable at the moment.
/// Renders a [`SyntaxNode`] with whitespace inserted between tokens that require them.
pub fn insert_ws_into(syn: SyntaxNode) -> SyntaxNode {
    let mut indent = 0;
    let mut last: Option<SyntaxKind> = None;
    let mut mods = Vec::new();
    let syn = syn.clone_subtree().clone_for_update();

    let before = Position::before;
    let after = Position::after;

    let do_indent = |pos: fn(_) -> Position, token: &SyntaxToken, indent| {
        (pos(token.clone()), make::tokens::whitespace(&" ".repeat(2 * indent)))
    };
    let do_ws = |pos: fn(_) -> Position, token: &SyntaxToken| {
        (pos(token.clone()), make::tokens::single_space())
    };
    let do_nl = |pos: fn(_) -> Position, token: &SyntaxToken| {
        (pos(token.clone()), make::tokens::single_newline())
    };

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
                    mods.push((
                        Position::after(node.clone()),
                        make::tokens::whitespace(&" ".repeat(2 * indent)),
                    ));
                }
                if node.parent().is_some() {
                    mods.push((Position::after(node), make::tokens::single_newline()));
                }
                continue;
            }
            _ => continue,
        };
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
            MUT_KW if is_next(|it| it == SELF_KW, false) => {
                mods.push(do_ws(after, tok));
            }
            AS_KW | DYN_KW | IMPL_KW | CONST_KW => {
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

    for (pos, insert) in mods {
        ted::insert(pos, insert);
    }

    if let Some(it) = syn.last_token().filter(|it| it.kind() == SyntaxKind::WHITESPACE) {
        ted::remove(it);
    }

    syn
}

fn is_text(k: SyntaxKind) -> bool {
    k.is_keyword() || k.is_literal() || k == IDENT || k == UNDERSCORE
}
