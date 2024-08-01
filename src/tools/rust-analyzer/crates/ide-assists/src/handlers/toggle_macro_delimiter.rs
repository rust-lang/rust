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
// macro_rules! sth  ();
// sth! $0( );
// ```
// ->
// ```
// macro_rules! sth! ();
// sth! { }
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

    enum MakroTypes {
        MacroRules(ast::MacroRules),
        MacroCall(ast::MacroCall),
    }

    let makro = if let Some(mc) = ctx.find_node_at_offset_with_descend::<ast::MacroCall>() {
        MakroTypes::MacroCall(mc)
    } else if let Some(mr) = ctx.find_node_at_offset_with_descend::<ast::MacroRules>() {
        MakroTypes::MacroRules(mr)
    } else {
        return None;
    };

    let cursor_offset = ctx.offset();
    let token_tree = match makro {
        MakroTypes::MacroRules(mr) => mr.token_tree()?.clone_for_update(),
        MakroTypes::MacroCall(md) => md.token_tree()?.clone_for_update(),
    };

    let token_tree_text_range = token_tree.syntax().text_range();
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
        AssistId("add_braces", AssistKind::Refactor),
        match token {
            MacroDelims::LPar => "Replace delimiters with braces",
            MacroDelims::RPar => "Replace delimiters with braces",
            MacroDelims::LBra => "Replace delimiters with parentheses",
            MacroDelims::RBra => "Replace delimiters with parentheses",
            MacroDelims::LCur => "Replace delimiters with brackets",
            MacroDelims::RCur => "Replace delimiters with brackets",
        },
        token_tree.syntax().text_range(),
        |builder| {
            match token {
                MacroDelims::LPar | MacroDelims::RPar => {
                    ted::replace(ltoken, make::token(T!['{']));
                    ted::replace(rtoken, make::token(T!['}']));
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
            builder.replace(token_tree_text_range, token_tree.syntax().text());
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
macro_rules! sth  ();
sth! $0( );
            "#,
            r#"
macro_rules! sth  ();
sth! { };
            "#,
        )
    }

    #[test]
    fn test_braclets() {
        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth  ();
sth! $0{ };
            "#,
            r#"
macro_rules! sth  ();
sth! [ ];
            "#,
        )
    }

    #[test]
    fn test_brackets() {
        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth  ();
sth! $0[ ];
            "#,
            r#"
macro_rules! sth  ();
sth! ( );
            "#,
        )
    }

    #[test]
    fn test_indent() {
        check_assist(
            toggle_macro_delimiter,
            r#"
mod abc {
    macro_rules! sth  ();
    sth! $0{ };
}
            "#,
            r#"
mod abc {
    macro_rules! sth  ();
    sth! [ ];
}
            "#,
        )
    }

    #[test]
    fn test_rules_par() {
        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth  $0();
sth! ( );
            "#,
            r#"
macro_rules! sth  {};
sth! ( );
            "#,
        )
    }

    #[test]
    fn test_rules_braclets() {
        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth  $0{};
sth! ( );
            "#,
            r#"
macro_rules! sth  [];
sth! ( );
            "#,
        )
    }

    #[test]
    fn test_rules_brackets() {
        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! sth  $0[];
sth! ( );
            "#,
            r#"
macro_rules! sth  ();
sth! ( );
            "#,
        )
    }

    #[test]
    fn test_unrelated_par() {
        check_assist_not_applicable(
            toggle_macro_delimiter,
            r#"
macro_rules! sth  [def$0()];
sth! ( );
            "#,
        )
    }
}
