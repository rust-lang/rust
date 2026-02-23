use ide_db::assists::AssistId;
use syntax::{
    AstNode, SyntaxToken, T,
    ast::{self, syntax_factory::SyntaxFactory},
};

use crate::{AssistContext, Assists};

// Assist: toggle_macro_delimiter
//
// Change macro delimiters in the order of `( -> { -> [ -> (`.
//
// ```
// macro_rules! sth {
//     () => {};
// }
//
// sth!$0( );
// ```
// ->
// ```
// macro_rules! sth {
//     () => {};
// }
//
// sth!{ }
// ```
pub(crate) fn toggle_macro_delimiter(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    #[derive(Debug)]
    enum MacroDelims {
        LPar,
        RPar,
        LBra,
        RBra,
        LCur,
        RCur,
    }

    let makro = ctx.find_node_at_offset::<ast::MacroCall>()?;

    let cursor_offset = ctx.offset();
    let semicolon = macro_semicolon(&makro);
    let token_tree = makro.token_tree()?;

    let ltoken = token_tree.left_delimiter_token()?;
    let rtoken = token_tree.right_delimiter_token()?;

    if !ltoken.text_range().contains(cursor_offset) && !rtoken.text_range().contains(cursor_offset)
    {
        return None;
    }

    let token = match ltoken.kind() {
        T!['{'] => MacroDelims::LCur,
        T!['('] => MacroDelims::LPar,
        T!['['] => MacroDelims::LBra,
        T!['}'] => MacroDelims::RBra,
        T![')'] => MacroDelims::RPar,
        T!['}'] => MacroDelims::RCur,
        _ => return None,
    };

    acc.add(
        AssistId::refactor("toggle_macro_delimiter"),
        match token {
            MacroDelims::LPar | MacroDelims::RPar => "Replace delimiters with braces",
            MacroDelims::LBra | MacroDelims::RBra => "Replace delimiters with parentheses",
            MacroDelims::LCur | MacroDelims::RCur => "Replace delimiters with brackets",
        },
        token_tree.syntax().text_range(),
        |builder| {
            let make = SyntaxFactory::with_mappings();
            let mut editor = builder.make_editor(makro.syntax());

            match token {
                MacroDelims::LPar | MacroDelims::RPar => {
                    editor.replace(ltoken, make.token(T!['{']));
                    editor.replace(rtoken, make.token(T!['}']));
                    if let Some(sc) = semicolon {
                        editor.delete(sc);
                    }
                }
                MacroDelims::LBra | MacroDelims::RBra => {
                    editor.replace(ltoken, make.token(T!['(']));
                    editor.replace(rtoken, make.token(T![')']));
                }
                MacroDelims::LCur | MacroDelims::RCur => {
                    editor.replace(ltoken, make.token(T!['[']));
                    if semicolon.is_some() || !needs_semicolon(token_tree) {
                        editor.replace(rtoken, make.token(T![']']));
                    } else {
                        editor.replace_with_many(
                            rtoken,
                            vec![make.token(T![']']).into(), make.token(T![;]).into()],
                        );
                    }
                }
            }
            editor.add_mappings(make.finish_with_mappings());
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

fn macro_semicolon(makro: &ast::MacroCall) -> Option<SyntaxToken> {
    makro.semicolon_token().or_else(|| {
        let macro_expr = ast::MacroExpr::cast(makro.syntax().parent()?)?;
        let expr_stmt = ast::ExprStmt::cast(macro_expr.syntax().parent()?)?;
        expr_stmt.semicolon_token()
    })
}

fn needs_semicolon(tt: ast::TokenTree) -> bool {
    (|| {
        let call = ast::MacroCall::cast(tt.syntax().parent()?)?;
        let container = call.syntax().parent()?;
        let kind = container.kind();

        if call.semicolon_token().is_some() {
            return Some(false);
        }

        Some(
            ast::ItemList::can_cast(kind)
                || ast::SourceFile::can_cast(kind)
                || ast::AssocItemList::can_cast(kind)
                || ast::ExternItemList::can_cast(kind)
                || ast::MacroItems::can_cast(kind)
                || ast::MacroExpr::can_cast(kind)
                    && ast::ExprStmt::cast(container.parent()?)
                        .is_some_and(|it| it.semicolon_token().is_none()),
        )
    })()
    .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_par() {
        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth {
    () => {};
}

sth!$0( );
            "#,
            r#"
macro_rules! sth {
    () => {};
}

sth!{ }
            "#,
        );

        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth {
    () => {};
}

fn foo() {
    sth!$0( );
}
            "#,
            r#"
macro_rules! sth {
    () => {};
}

fn foo() {
    sth!{ }
}
            "#,
        );
    }

    #[test]
    fn test_braces() {
        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth {
    () => {};
}

sth!$0{ }
            "#,
            r#"
macro_rules! sth {
    () => {};
}

sth![ ];
            "#,
        );

        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth {
    () => {};
}

fn foo() -> i32 {
    sth!$0{ }
    2
}
            "#,
            r#"
macro_rules! sth {
    () => {};
}

fn foo() -> i32 {
    sth![ ];
    2
}
            "#,
        );

        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth {
    () => {2};
}

fn foo() {
    sth!$0{ };
}
            "#,
            r#"
macro_rules! sth {
    () => {2};
}

fn foo() {
    sth![ ];
}
            "#,
        );

        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth {
    () => {2};
}

fn foo() -> i32 {
    sth!$0{ }
}
            "#,
            r#"
macro_rules! sth {
    () => {2};
}

fn foo() -> i32 {
    sth![ ]
}
            "#,
        );

        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth {
    () => {};
}
impl () {
    sth!$0{}
}
            "#,
            r#"
macro_rules! sth {
    () => {};
}
impl () {
    sth![];
}
            "#,
        );

        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth {
    () => {2};
}

fn foo() -> i32 {
    bar(sth!$0{ })
}
            "#,
            r#"
macro_rules! sth {
    () => {2};
}

fn foo() -> i32 {
    bar(sth![ ])
}
            "#,
        );
    }

    #[test]
    fn test_brackets() {
        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth {
    () => {};
}

sth!$0[ ];
            "#,
            r#"
macro_rules! sth {
    () => {};
}

sth!( );
            "#,
        )
    }

    #[test]
    fn test_indent() {
        check_assist(
            toggle_macro_delimiter,
            r#"
mod abc {
    macro_rules! sth {
        () => {};
    }

    sth!$0{ }
}
            "#,
            r#"
mod abc {
    macro_rules! sth {
        () => {};
    }

    sth![ ];
}
            "#,
        )
    }

    #[test]
    fn test_unrelated_par() {
        check_assist_not_applicable(
            toggle_macro_delimiter,
            r#"
macro_rules! prt {
    ($e:expr) => {{
        println!("{}", stringify!{$e});
    }};
}

prt!(($03 + 5));

            "#,
        )
    }

    #[test]
    fn test_longer_macros() {
        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! prt {
    ($e:expr) => {{
        println!("{}", stringify!{$e});
    }};
}

prt!$0((3 + 5));
"#,
            r#"
macro_rules! prt {
    ($e:expr) => {{
        println!("{}", stringify!{$e});
    }};
}

prt!{(3 + 5)}
"#,
        )
    }

    // FIXME @alibektas : Inner macro_call is not seen as such. So this doesn't work.
    #[test]
    fn test_nested_macros() {
        check_assist_not_applicable(
            toggle_macro_delimiter,
            r#"
macro_rules! prt {
    ($e:expr) => {{
        println!("{}", stringify!{$e});
    }};
}

macro_rules! abc {
    ($e:expr) => {{
        println!("{}", stringify!{$e});
    }};
}

prt!{abc!($03 + 5)};
"#,
        )
    }
}
