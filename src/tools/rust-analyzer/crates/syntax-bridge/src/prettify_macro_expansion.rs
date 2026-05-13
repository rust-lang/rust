//! Utilities for formatting macro expanded nodes until we get a proper formatter.
use syntax::{
    NodeOrToken,
    SyntaxKind::{self, *},
    SyntaxNode, SyntaxToken, T, WalkEvent,
    syntax_editor::{Position, SyntaxEditor},
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
    let (editor, syn) = SyntaxEditor::new(syn);

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
            WalkEvent::Leave(NodeOrToken::Node(node)) => {
                let is_last_child =
                    node.parent().is_some_and(|parent| parent.last_child().as_ref() == Some(&node));
                let is_always_newline = matches!(node.kind(), ATTR);
                let is_non_last_newline = match node.kind() {
                    MATCH_ARM | STRUCT | ENUM | UNION | FN | IMPL | MACRO_RULES | EXTERN_BLOCK
                    | EXTERN_CRATE | MODULE => true,
                    EXPR_STMT if Some(R_CURLY) == node.last_token().map(|it| it.kind()) => true,
                    _ => false,
                };
                if (!is_last_child && is_non_last_newline) || is_always_newline {
                    mods.push((Position::after(node.clone()), PrettifyWsKind::Indent(indent)));
                    if node.parent().is_some() {
                        mods.push((Position::after(node), PrettifyWsKind::Newline));
                    }
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
                && is_next(|it| !it.is_punct() || matches!(it, T![_] | T![#] | L_CURLY), false) =>
            {
                mods.push(do_ws(after, tok));
            }
            L_CURLY if is_next(|it| it != R_CURLY, true) => {
                indent += 1;
                mods.push(do_indent(after, tok, indent));
                mods.push(do_nl(after, tok));
            }
            R_CURLY if is_last(|it| it != L_CURLY, true) => {
                indent = indent.saturating_sub(1);

                mods.push(do_indent(before, tok, indent));
                mods.push(do_nl(before, tok));
            }
            R_CURLY if is_next(|it| it == T![else], false) => {
                mods.push(do_indent(before, tok, indent));
                mods.push(do_nl(before, tok));
            }
            LIFETIME_IDENT if is_next(is_text, true) => {
                mods.push(do_ws(after, tok));
            }
            AS_KW | DYN_KW | IMPL_KW | CONST_KW | MUT_KW | LET_KW | MATCH_KW => {
                mods.push(do_ws(after, tok));
            }
            T![;] if is_next(|it| it != R_CURLY, true) => {
                mods.push(do_indent(after, tok, indent));
                if tok.text_range().end() != syn.text_range().end() {
                    mods.push(do_nl(after, tok));
                }
            }
            T![=] if let Some((last, next)) = last.zip(tok.next_token()) => {
                // FIXME: this branch is for `=>` in macro_rules!, which is currently parsed as
                // two separate symbols.
                match (last, next.kind()) {
                    (T![=], _) | (_, T![=]) => (),
                    // catch ..= += etc
                    #[rustfmt::skip]
                    (
                        T![!] | T![%] | T![&] | T![*] | T![+] | T![-] |
                        T![/] | T![<] | T![>] | T![^] | T![|] | T![.],
                        _,
                    ) => (),
                    (_, T![>]) => {
                        mods.push(do_ws(before, tok));
                        mods.push(do_ws(after, &next));
                    }
                    _ => {
                        mods.push(do_ws(before, tok));
                        mods.push(do_ws(after, tok));
                    }
                }
            }
            T![->] | T![=>] => {
                mods.push(do_ws(before, tok));
                mods.push(do_ws(after, tok));
            }
            T![:] if is_next(|it| it != T![:], false) && is_last(|it| it != T![:], false) => {
                // XXX: Why input included WHITESPACE?
                if is_next(|it| it != SyntaxKind::WHITESPACE, false) {
                    mods.push(do_ws(after, tok));
                }
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
        editor.insert(
            pos,
            match insert {
                PrettifyWsKind::Space => editor.make().whitespace(" "),
                PrettifyWsKind::Indent(0) => continue,
                PrettifyWsKind::Indent(indent) => editor.make().whitespace(&" ".repeat(4 * indent)),
                PrettifyWsKind::Newline => editor.make().whitespace("\n"),
            },
        );
    }
    for (old, new) in dollar_crate_replacements {
        editor.replace(old, new);
    }

    if let Some(it) = syn.last_token().filter(|it| it.kind() == SyntaxKind::WHITESPACE) {
        editor.delete(it);
    }

    editor.finish().new_root().clone()
}

fn is_text(k: SyntaxKind) -> bool {
    // Consider all keywords in all editions.
    k.is_any_identifier() || k.is_literal() || k == UNDERSCORE
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::{Expect, expect};

    #[expect(deprecated)]
    fn check_pretty(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
        let ra_fixture = stdx::trim_indent(ra_fixture);
        let source_file = syntax::ast::SourceFile::parse(&ra_fixture, span::Edition::CURRENT);
        let syn = remove_whitespaces(&source_file.syntax_node());

        let pretty = prettify_macro_expansion(syn, &mut |_| None, |_| ());
        let mut pretty = pretty.to_string();
        if pretty.contains('\n') {
            pretty.push('\n');
        }
        expect.assert_eq(&pretty);

        fn remove_whitespaces(node: &SyntaxNode) -> SyntaxNode {
            let (editor, node) = SyntaxEditor::new(node.clone());
            node.preorder_with_tokens().for_each(|it| match it {
                WalkEvent::Enter(NodeOrToken::Token(tok)) if tok.kind().is_trivia() => {
                    editor.delete(tok);
                }
                _ => (),
            });
            editor.finish().new_root().clone()
        }
    }

    #[test]
    fn test_in_macro() {
        check_pretty(
            r#"
            const X: i32 = x::y::z;
            macro_rules! foo {
                () => {
                    $crate::foo::bar!();
                    (1..2, 1..=2);
                    (a==b, a!=b, a<=b, a>=b, x+=2, x<<=2);
                };
            }
            "#,
            expect![[r#"
                const X: i32 = x::y::z;
                macro_rules! foo {
                    () => {
                        $crate::foo::bar!();
                        (1..2,1..=2);
                        (a==b,a!=b,a<=b,a>=b,x+=2,x<<=2);
                    };
                }
            "#]],
        );
    }

    #[test]
    fn test_curly_indent() {
        check_pretty(
            r#"
            const _: () = {
                {
                    2;
                    3
                }
            };
            "#,
            expect![[r#"
                const _: () = {
                    {
                        2;
                        3
                    }
                };
            "#]],
        );
    }

    #[test]
    fn test_pats() {
        check_pretty(
            r#"
            const _: () = {
                let x = 2;
                let mut y = 3;
                let ref mut z @ 0..5 = 4;
                let ref mut t @ 0..=5 = 4;
                let (x, ref y) = (5, 6);
                let (Foo { x, y }, Bar(z, t));
                let (&mut x, (y | y));
                match () {}
            };
            "#,
            expect![[r#"
                const _: () = {
                    let x = 2;
                    let mut y = 3;
                    let ref mut z@0..5 = 4;
                    let ref mut t@0..=5 = 4;
                    let (x,ref y) = (5,6);
                    let (Foo {
                        x,y
                    },Bar(z,t));
                    let (&mut x,(y|y));
                    match (){}
                };
            "#]],
        );
    }

    #[test]
    fn test_attrs() {
        check_pretty(
            r#"
            #[attr1]
            #[attr2]
            const _: () = {};
            #[attr1]
            const _: () = {
                #[attr2]
                {}
            };
            "#,
            expect![[r#"
                #[attr1]
                #[attr2]
                const _: () = {};
                #[attr1]
                const _: () = {
                    #[attr2]
                    {}
                };
            "#]],
        );
    }

    #[test]
    fn test_items() {
        check_pretty(
            r#"
            fn foo() {}
            struct Foo {}
            struct Foo;
            enum Foo {}
            impl Foo {}
            const _: () = {};
            static S: () = {};
            extern {}
            mod x {}
            mod x;
            type X = 2;
            use a;
            use b::{c, d};
            macro_rules! foo { () => {}; }
            "#,
            expect![[r#"
                fn foo(){}
                struct Foo {}
                struct Foo;

                enum Foo {}
                impl Foo {}
                const _: () = {};
                static S: () = {};
                extern {}
                mod x {}
                mod x;

                type X = 2;
                use a;
                use b::{
                    c,d
                };
                macro_rules! foo {
                    () => {};
                }
            "#]],
        );
    }

    #[test]
    fn test_exprs() {
        check_pretty(
            r#"
            const _: () = {
                let _ = 1+2;
                let _ = !true && false;
                let _ = foo() + !bar() + dbg!(2) + *x;
                let _ = async move || {};
                let _ = async move {};
                let _ = x.await;
                let _ = (1..2, 1..=2);
                'lab: for _ in 0..5 {
                    loop { }
                    break 'lab expr;
                    if let pat = expr {
                        foo()
                    } else if true {
                        bar()
                    } else {}
                    if true {} else if true {} else {}
                    fun()
                }
            };
            "#,
            expect![[r#"
                const _: () = {
                    let _ = 1+2;
                    let _ = !true&&false;
                    let _ = foo()+!bar()+dbg!(2)+*x;
                    let _ = async move||{};
                    let _ = async move {};
                    let _ = x.await;
                    let _ = (1..2,1..=2);
                    'lab: for _ in 0..5 {
                        loop {}
                        break 'lab expr;
                        if let pat = expr {
                            foo()
                        }else if true {
                            bar()
                        }else {}
                        if true {
                        }else if true {
                        }else {}
                        fun()
                    }
                };
            "#]],
        );
    }

    #[test]
    fn test_match_arm() {
        check_pretty(
            r#"
            const _: () = {
                match 2 {
                    tmp => foo!(),
                };
            };
            "#,
            expect![[r#"
                const _: () = {
                    match 2 {
                        tmp => foo!(),
                    };
                };
            "#]],
        );

        check_pretty(
            r#"
            const _: () = {
                match 2 {
                    tmp => {}
                };
            };
            "#,
            expect![[r#"
                const _: () = {
                    match 2 {
                        tmp => {}
                    };
                };
            "#]],
        );

        check_pretty(
            r#"
            const _: () = {
                match 2 {
                    1 => {}
                    2 => foo(),
                    _ => {},
                };
            };
            "#,
            expect![[r#"
                const _: () = {
                    match 2 {
                        1 => {}
                        2 => foo(),
                        _ => {},
                    };
                };
            "#]],
        );
    }
}
