use crate::{AssistContext, Assists, utils};
use ide_db::{
    assists::{AssistId, AssistKind},
    syntax_helpers::format_string_exprs::{Arg, parse_format_exprs},
};
use itertools::Itertools;
use syntax::{
    AstNode, AstToken, NodeOrToken,
    SyntaxKind::WHITESPACE,
    T,
    ast::{self, make, syntax_factory::SyntaxFactory},
};

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
    let tt_delimiter = tt.left_delimiter_token()?.kind();

    let _ = ctx.sema.as_format_args_parts(&fmt_string)?;

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
            None,
        ),
        "Extract format expressions",
        tt.syntax().text_range(),
        |edit| {
            // Extract existing arguments in macro
            let tokens = tt.token_trees_and_tokens().collect_vec();

            let existing_args = if let [
                _opening_bracket,
                NodeOrToken::Token(_format_string),
                _args_start_comma,
                tokens @ ..,
                NodeOrToken::Token(_end_bracket),
            ] = tokens.as_slice()
            {
                let args = tokens
                    .split(|it| matches!(it, NodeOrToken::Token(t) if t.kind() == T![,]))
                    .map(|arg| {
                        // Strip off leading and trailing whitespace tokens
                        let arg = match arg.split_first() {
                            Some((NodeOrToken::Token(t), rest)) if t.kind() == WHITESPACE => rest,
                            _ => arg,
                        };

                        match arg.split_last() {
                            Some((NodeOrToken::Token(t), rest)) if t.kind() == WHITESPACE => rest,
                            _ => arg,
                        }
                    });

                args.collect()
            } else {
                vec![]
            };

            // Start building the new args
            let mut existing_args = existing_args.into_iter();
            let mut new_tt_bits = vec![NodeOrToken::Token(make::tokens::literal(&new_fmt))];
            let mut placeholder_indexes = vec![];

            for arg in extracted_args {
                if matches!(arg, Arg::Expr(_) | Arg::Placeholder) {
                    // insert ", " before each arg
                    new_tt_bits.extend_from_slice(&[
                        NodeOrToken::Token(make::token(T![,])),
                        NodeOrToken::Token(make::tokens::single_space()),
                    ]);
                }

                match arg {
                    Arg::Expr(s) => {
                        // insert arg
                        // FIXME: use the crate's edition for parsing
                        let expr =
                            ast::Expr::parse(&s, syntax::Edition::CURRENT_FIXME).syntax_node();
                        let mut expr_tt = utils::tt_from_syntax(expr);
                        new_tt_bits.append(&mut expr_tt);
                    }
                    Arg::Placeholder => {
                        // try matching with existing argument
                        match existing_args.next() {
                            Some(arg) => {
                                new_tt_bits.extend_from_slice(arg);
                            }
                            None => {
                                placeholder_indexes.push(new_tt_bits.len());
                                new_tt_bits.push(NodeOrToken::Token(make::token(T![_])));
                            }
                        }
                    }
                    Arg::Ident(_s) => (),
                }
            }

            // Insert new args
            let make = SyntaxFactory::with_mappings();
            let new_tt = make.token_tree(tt_delimiter, new_tt_bits);
            let mut editor = edit.make_editor(tt.syntax());
            editor.replace(tt.syntax(), new_tt.syntax());

            if let Some(cap) = ctx.config.snippet_cap {
                // Add placeholder snippets over placeholder args
                for pos in placeholder_indexes {
                    // Skip the opening delimiter
                    let Some(NodeOrToken::Token(placeholder)) =
                        new_tt.token_trees_and_tokens().skip(1).nth(pos)
                    else {
                        continue;
                    };

                    if stdx::always!(placeholder.kind() == T![_]) {
                        let annotation = edit.make_placeholder_snippet(cap);
                        editor.add_annotation(placeholder, annotation);
                    }
                }

                // Add the final tabstop after the format literal
                if let Some(NodeOrToken::Token(literal)) = new_tt.token_trees_and_tokens().nth(1) {
                    let annotation = edit.make_tabstop_after(cap);
                    editor.add_annotation(literal, annotation);
                }
            }
            editor.add_mappings(make.finish_with_mappings());
            edit.add_file_edits(ctx.vfs_file_id(), editor);
        },
    );

    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_no_snippet_cap};

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
    print!("{} {:b} {} {}"$0, y + 2, x + 1, 2, ${1:_});
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

    #[test]
    fn without_snippets() {
        check_assist_no_snippet_cap(
            extract_expressions_from_format_string,
            r#"
//- minicore: fmt
fn main() {
    print!("{} {x + 1:b} {} {}$0", y + 2, 2);
}
"#,
            r#"
fn main() {
    print!("{} {:b} {} {}", y + 2, x + 1, 2, _);
}
"#,
        );
    }
}
