use crate::{AssistContext, Assists};
use hir::DescendPreference;
use ide_db::{
    assists::{AssistId, AssistKind},
    syntax_helpers::{
        format_string::is_format_string,
        format_string_exprs::{parse_format_exprs, Arg},
    },
};
use itertools::Itertools;
use stdx::format_to;
use syntax::{ast, AstNode, AstToken, NodeOrToken, SyntaxKind::COMMA, TextRange};

// Assist: extract_expressions_from_format_string
//
// Move an expression out of a format string.
//
// ```
// # //- minicore: fmt
// fn main() {
//     print!("{var} {x + 1}$0");
// }
// ```
// ->
// ```
// fn main() {
//     print!("{var} {}"$0, x + 1);
// }
// ```

pub(crate) fn extract_expressions_from_format_string(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let fmt_string = ctx.find_token_at_offset::<ast::String>()?;
    let tt = fmt_string.syntax().parent().and_then(ast::TokenTree::cast)?;

    let expanded_t = ast::String::cast(
        ctx.sema
            .descend_into_macros_single(DescendPreference::SameKind, fmt_string.syntax().clone()),
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
            "extract_expressions_from_format_string",
            // if there aren't any expressions, then make the assist a RefactorExtract
            if extracted_args.iter().filter(|f| matches!(f, Arg::Expr(_))).count() == 0 {
                AssistKind::RefactorExtract
            } else {
                AssistKind::QuickFix
            },
        ),
        "Extract format expressions",
        tt.syntax().text_range(),
        |edit| {
            let fmt_range = fmt_string.syntax().text_range();

            // Replace old format string with new format string whose arguments have been extracted
            edit.replace(fmt_range, new_fmt);

            // Insert cursor at end of format string
            edit.insert(fmt_range.end(), "$0");

            // Extract existing arguments in macro
            let tokens =
                tt.token_trees_and_tokens().collect_vec();

            let mut existing_args: Vec<String> = vec![];

            let mut current_arg = String::new();
            if let [_opening_bracket, NodeOrToken::Token(format_string), _args_start_comma, tokens @ .., NodeOrToken::Token(end_bracket)] =
                tokens.as_slice()
            {
                for t in tokens {
                    match t {
                        NodeOrToken::Node(n) => {
                            format_to!(current_arg, "{n}");
                        },
                        NodeOrToken::Token(t) if t.kind() == COMMA => {
                            existing_args.push(current_arg.trim().into());
                            current_arg.clear();
                        },
                        NodeOrToken::Token(t) => {
                            current_arg.push_str(t.text());
                        },
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
                match extracted_args {
                    Arg::Expr(s)=> {
                        args.push_str(", ");
                        // insert arg
                        args.push_str(&s);
                    }
                    Arg::Placeholder => {
                        args.push_str(", ");
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
                    Arg::Ident(_s) => (),
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

    #[test]
    fn multiple_middle_arg() {
        check_assist(
            extract_expressions_from_format_string,
            r#"
//- minicore: fmt
fn main() {
    print!("{} {x + 1:b} {}$0", y + 2, 2);
}
"#,
            r#"
fn main() {
    print!("{} {:b} {}"$0, y + 2, x + 1, 2);
}
"#,
        );
    }

    #[test]
    fn single_arg() {
        check_assist(
            extract_expressions_from_format_string,
            r#"
//- minicore: fmt
fn main() {
    print!("{obj.value:b}$0",);
}
"#,
            r#"
fn main() {
    print!("{:b}"$0, obj.value);
}
"#,
        );
    }

    #[test]
    fn multiple_middle_placeholders_arg() {
        check_assist(
            extract_expressions_from_format_string,
            r#"
//- minicore: fmt
fn main() {
    print!("{} {x + 1:b} {} {}$0", y + 2, 2);
}
"#,
            r#"
fn main() {
    print!("{} {:b} {} {}"$0, y + 2, x + 1, 2, $1);
}
"#,
        );
    }

    #[test]
    fn multiple_trailing_args() {
        check_assist(
            extract_expressions_from_format_string,
            r#"
//- minicore: fmt
fn main() {
    print!("{:b} {x + 1:b} {Struct(1, 2)}$0", 1);
}
"#,
            r#"
fn main() {
    print!("{:b} {:b} {}"$0, 1, x + 1, Struct(1, 2));
}
"#,
        );
    }

    #[test]
    fn improper_commas() {
        check_assist(
            extract_expressions_from_format_string,
            r#"
//- minicore: fmt
fn main() {
    print!("{} {x + 1:b} {Struct(1, 2)}$0", 1,);
}
"#,
            r#"
fn main() {
    print!("{} {:b} {}"$0, 1, x + 1, Struct(1, 2));
}
"#,
        );
    }

    #[test]
    fn nested_tt() {
        check_assist(
            extract_expressions_from_format_string,
            r#"
//- minicore: fmt
fn main() {
    print!("My name is {} {x$0 + x}", stringify!(Paperino))
}
"#,
            r#"
fn main() {
    print!("My name is {} {}"$0, stringify!(Paperino), x + x)
}
"#,
        );
    }

    #[test]
    fn extract_only_expressions() {
        check_assist(
            extract_expressions_from_format_string,
            r#"
//- minicore: fmt
fn main() {
    let var = 1 + 1;
    print!("foobar {var} {var:?} {x$0 + x}")
}
"#,
            r#"
fn main() {
    let var = 1 + 1;
    print!("foobar {var} {var:?} {}"$0, x + x)
}
"#,
        );
    }

    #[test]
    fn escaped_literals() {
        check_assist(
            extract_expressions_from_format_string,
            r#"
//- minicore: fmt
fn main() {
    print!("\n$ {x + 1}$0");
}
            "#,
            r#"
fn main() {
    print!("\n$ {}"$0, x + 1);
}
            "#,
        );
    }
}
