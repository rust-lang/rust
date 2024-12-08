use ide_db::assists::{AssistId, AssistKind};
use syntax::{
    ast::{self, make},
    ted, AstNode, T,
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

    let makro = ctx.find_node_at_offset::<ast::MacroCall>()?.clone_for_update();
    let makro_text_range = makro.syntax().text_range();

    let cursor_offset = ctx.offset();
    let semicolon = makro.semicolon_token();
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
        AssistId("toggle_macro_delimiter", AssistKind::Refactor),
        match token {
            MacroDelims::LPar | MacroDelims::RPar => "Replace delimiters with braces",
            MacroDelims::LBra | MacroDelims::RBra => "Replace delimiters with parentheses",
            MacroDelims::LCur | MacroDelims::RCur => "Replace delimiters with brackets",
        },
        token_tree.syntax().text_range(),
        |builder| {
            match token {
                MacroDelims::LPar | MacroDelims::RPar => {
                    ted::replace(ltoken, make::token(T!['{']));
                    ted::replace(rtoken, make::token(T!['}']));
                    if let Some(sc) = semicolon {
                        ted::remove(sc);
                    }
                }
                MacroDelims::LBra | MacroDelims::RBra => {
                    ted::replace(ltoken, make::token(T!['(']));
                    ted::replace(rtoken, make::token(T![')']));
                }
                MacroDelims::LCur | MacroDelims::RCur => {
                    ted::replace(ltoken, make::token(T!['[']));
                    ted::replace(rtoken, make::token(T![']']));
                }
            }
            builder.replace(makro_text_range, makro.syntax().text());
        },
    )
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
        )
    }

    #[test]
    fn test_braces() {
        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth {
    () => {};
}

sth!$0{ };
            "#,
            r#"
macro_rules! sth {
    () => {};
}

sth![ ];
            "#,
        )
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

    sth!$0{ };
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
