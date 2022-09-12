use crate::{AssistContext, Assists};
use ide_db::{
    assists::{AssistId, AssistKind},
    syntax_helpers::{
        format_string::is_format_string,
        format_string_exprs::{parse_format_exprs, Arg},
    },
};
use itertools::Itertools;
use syntax::{ast, AstNode, AstToken, NodeOrToken, SyntaxKind::COMMA, TextRange};

// Assist: move_format_string_arg
//
// Move an expression out of a format string.
//
// ```
// macro_rules! format_args {
//     ($lit:literal $(tt:tt)*) => { 0 },
// }
// macro_rules! print {
//     ($($arg:tt)*) => (std::io::_print(format_args!($($arg)*)));
// }
//
// fn main() {
//     print!("{x + 1}$0");
// }
// ```
// ->
// ```
// macro_rules! format_args {
//     ($lit:literal $(tt:tt)*) => { 0 },
// }
// macro_rules! print {
//     ($($arg:tt)*) => (std::io::_print(format_args!($($arg)*)));
// }
//
// fn main() {
//     print!("{}"$0, x + 1);
// }
// ```

pub(crate) fn move_format_string_arg(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let fmt_string = ctx.find_token_at_offset::<ast::String>()?;
    let tt = fmt_string.syntax().parent().and_then(ast::TokenTree::cast)?;

    let expanded_t = ast::String::cast(
        ctx.sema.descend_into_macros_with_kind_preference(fmt_string.syntax().clone()),
    )?;
    if !is_format_string(&expanded_t) {
        return None;
    }

    let (new_fmt, extracted_args) = parse_format_exprs(fmt_string.text()).ok()?;
    if extracted_args.is_empty() {
        return None;
    }

    acc.add(
        AssistId(
            "move_format_string_arg",
            // if there aren't any expressions, then make the assist a RefactorExtract
            if extracted_args.iter().filter(|f| matches!(f, Arg::Expr(_))).count() == 0 {
                AssistKind::RefactorExtract
            } else {
                AssistKind::QuickFix
            },
        ),
        "Extract format args",
        tt.syntax().text_range(),
        |edit| {
            let fmt_range = fmt_string.syntax().text_range();

            // Replace old format string with new format string whose arguments have been extracted
            edit.replace(fmt_range, new_fmt);

            // Insert cursor at end of format string
            edit.insert(fmt_range.end(), "$0");

            // Extract existing arguments in macro
            let tokens =
                tt.token_trees_and_tokens().filter_map(NodeOrToken::into_token).collect_vec();

            let mut existing_args: Vec<String> = vec![];

            let mut current_arg = String::new();
            if let [_opening_bracket, format_string, _args_start_comma, tokens @ .., end_bracket] =
                tokens.as_slice()
            {
                for t in tokens {
                    if t.kind() == COMMA {
                        existing_args.push(current_arg.trim().into());
                        current_arg.clear();
                    } else {
                        current_arg.push_str(t.text());
                    }
                }
                existing_args.push(current_arg.trim().into());

                // delete everything after the format string till end bracket
                // we're going to insert the new arguments later
                edit.delete(TextRange::new(
                    format_string.text_range().end(),
                    end_bracket.text_range().start(),
                ));
            }

            // Start building the new args
            let mut existing_args = existing_args.into_iter();
            let mut args = String::new();

            let mut placeholder_idx = 1;

            for extracted_args in extracted_args {
                // remove expr from format string
                args.push_str(", ");

                match extracted_args {
                    Arg::Ident(s) | Arg::Expr(s) => {
                        // insert arg
                        args.push_str(&s);
                    }
                    Arg::Placeholder => {
                        // try matching with existing argument
                        match existing_args.next() {
                            Some(ea) => {
                                args.push_str(&ea);
                            }
                            None => {
                                // insert placeholder
                                args.push_str(&format!("${placeholder_idx}"));
                                placeholder_idx += 1;
                            }
                        }
                    }
                }
            }

            // Insert new args
            edit.insert(fmt_range.end(), args);
        },
    );

    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::check_assist;

    const MACRO_DECL: &'static str = r#"
macro_rules! format_args {
    ($lit:literal $(tt:tt)*) => { 0 },
}
macro_rules! print {
    ($($arg:tt)*) => (std::io::_print(format_args!($($arg)*)));
}
"#;

    fn add_macro_decl(s: &'static str) -> String {
        MACRO_DECL.to_string() + s
    }

    #[test]
    fn multiple_middle_arg() {
        check_assist(
            move_format_string_arg,
            &add_macro_decl(
                r#"
fn main() {
    print!("{} {x + 1:b} {}$0", y + 2, 2);
}
"#,
            ),
            &add_macro_decl(
                r#"
fn main() {
    print!("{} {:b} {}"$0, y + 2, x + 1, 2);
}
"#,
            ),
        );
    }

    #[test]
    fn single_arg() {
        check_assist(
            move_format_string_arg,
            &add_macro_decl(
                r#"
fn main() {
    print!("{obj.value:b}$0",);
}
"#,
            ),
            &add_macro_decl(
                r#"
fn main() {
    print!("{:b}"$0, obj.value);
}
"#,
            ),
        );
    }

    #[test]
    fn multiple_middle_placeholders_arg() {
        check_assist(
            move_format_string_arg,
            &add_macro_decl(
                r#"
fn main() {
    print!("{} {x + 1:b} {} {}$0", y + 2, 2);
}
"#,
            ),
            &add_macro_decl(
                r#"
fn main() {
    print!("{} {:b} {} {}"$0, y + 2, x + 1, 2, $1);
}
"#,
            ),
        );
    }

    #[test]
    fn multiple_trailing_args() {
        check_assist(
            move_format_string_arg,
            &add_macro_decl(
                r#"
fn main() {
    print!("{} {x + 1:b} {Struct(1, 2)}$0", 1);
}
"#,
            ),
            &add_macro_decl(
                r#"
fn main() {
    print!("{} {:b} {}"$0, 1, x + 1, Struct(1, 2));
}
"#,
            ),
        );
    }

    #[test]
    fn improper_commas() {
        check_assist(
            move_format_string_arg,
            &add_macro_decl(
                r#"
fn main() {
    print!("{} {x + 1:b} {Struct(1, 2)}$0", 1,);
}
"#,
            ),
            &add_macro_decl(
                r#"
fn main() {
    print!("{} {:b} {}"$0, 1, x + 1, Struct(1, 2));
}
"#,
            ),
        );
    }
}
