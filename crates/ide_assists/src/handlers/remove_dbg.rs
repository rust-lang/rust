use syntax::{
    ast::{self, AstNode},
    match_ast, SyntaxElement, TextRange, TextSize, T,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: remove_dbg
//
// Removes `dbg!()` macro call.
//
// ```
// fn main() {
//     $0dbg!(92);
// }
// ```
// ->
// ```
// fn main() {
//     92;
// }
// ```
pub(crate) fn remove_dbg(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let macro_call = ctx.find_node_at_offset::<ast::MacroCall>()?;
    let new_contents = adjusted_macro_contents(&macro_call)?;

    let macro_text_range = macro_call.syntax().text_range();
    let macro_end = if macro_call.semicolon_token().is_some() {
        macro_text_range.end() - TextSize::of(';')
    } else {
        macro_text_range.end()
    };

    acc.add(
        AssistId("remove_dbg", AssistKind::Refactor),
        "Remove dbg!()",
        macro_text_range,
        |builder| {
            builder.replace(TextRange::new(macro_text_range.start(), macro_end), new_contents);
        },
    )
}

fn adjusted_macro_contents(macro_call: &ast::MacroCall) -> Option<String> {
    let contents = get_valid_macrocall_contents(&macro_call, "dbg")?;
    let macro_text_with_brackets = macro_call.token_tree()?.syntax().text();
    let macro_text_in_brackets = macro_text_with_brackets.slice(TextRange::new(
        TextSize::of('('),
        macro_text_with_brackets.len() - TextSize::of(')'),
    ));

    Some(
        if !is_leaf_or_control_flow_expr(macro_call)
            && needs_parentheses_around_macro_contents(contents)
        {
            format!("({})", macro_text_in_brackets)
        } else {
            macro_text_in_brackets.to_string()
        },
    )
}

fn is_leaf_or_control_flow_expr(macro_call: &ast::MacroCall) -> bool {
    macro_call.syntax().next_sibling().is_none()
        || match macro_call.syntax().parent() {
            Some(parent) => match_ast! {
                match parent {
                    ast::Condition(_it) => true,
                    ast::MatchExpr(_it) => true,
                    _ => false,
                }
            },
            None => false,
        }
}

/// Verifies that the given macro_call actually matches the given name
/// and contains proper ending tokens, then returns the contents between the ending tokens
fn get_valid_macrocall_contents(
    macro_call: &ast::MacroCall,
    macro_name: &str,
) -> Option<Vec<SyntaxElement>> {
    let path = macro_call.path()?;
    let name_ref = path.segment()?.name_ref()?;

    // Make sure it is actually a dbg-macro call, dbg followed by !
    let excl = path.syntax().next_sibling_or_token()?;
    if name_ref.text() != macro_name || excl.kind() != T![!] {
        return None;
    }

    let mut children_with_tokens = macro_call.token_tree()?.syntax().children_with_tokens();
    let first_child = children_with_tokens.next()?;
    let mut contents_between_brackets = children_with_tokens.collect::<Vec<_>>();
    let last_child = contents_between_brackets.pop()?;

    if contents_between_brackets.is_empty() {
        None
    } else {
        match (first_child.kind(), last_child.kind()) {
            (T!['('], T![')']) | (T!['['], T![']']) | (T!['{'], T!['}']) => {
                Some(contents_between_brackets)
            }
            _ => None,
        }
    }
}

fn needs_parentheses_around_macro_contents(macro_contents: Vec<SyntaxElement>) -> bool {
    if macro_contents.len() < 2 {
        return false;
    }
    let mut macro_contents = macro_contents.into_iter().peekable();
    let mut unpaired_brackets_in_contents = Vec::new();
    while let Some(element) = macro_contents.next() {
        match element.kind() {
            T!['('] | T!['['] | T!['{'] => unpaired_brackets_in_contents.push(element),
            T![')'] => {
                if !matches!(unpaired_brackets_in_contents.pop(), Some(correct_bracket) if correct_bracket.kind() == T!['('])
                {
                    return true;
                }
            }
            T![']'] => {
                if !matches!(unpaired_brackets_in_contents.pop(), Some(correct_bracket) if correct_bracket.kind() == T!['['])
                {
                    return true;
                }
            }
            T!['}'] => {
                if !matches!(unpaired_brackets_in_contents.pop(), Some(correct_bracket) if correct_bracket.kind() == T!['{'])
                {
                    return true;
                }
            }
            symbol_kind => {
                let symbol_not_in_bracket = unpaired_brackets_in_contents.is_empty();
                if symbol_not_in_bracket
                    && symbol_kind != T![:] // paths
                    && (symbol_kind != T![.] // field/method access
                        || macro_contents // range expressions consist of two SyntaxKind::Dot in macro invocations
                            .peek()
                            .map(|element| element.kind() == T![.])
                            .unwrap_or(false))
                    && symbol_kind != T![?] // try operator
                    && (symbol_kind.is_punct() || symbol_kind == T![as])
                {
                    return true;
                }
            }
        }
    }
    !unpaired_brackets_in_contents.is_empty()
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

    #[test]
    fn test_remove_dbg() {
        check_assist(remove_dbg, "$0dbg!(1 + 1)", "1 + 1");

        check_assist(remove_dbg, "dbg!$0((1 + 1))", "(1 + 1)");

        check_assist(remove_dbg, "dbg!(1 $0+ 1)", "1 + 1");

        check_assist(remove_dbg, "let _ = $0dbg!(1 + 1)", "let _ = 1 + 1");

        check_assist(
            remove_dbg,
            "
fn foo(n: usize) {
    if let Some(_) = dbg!(n.$0checked_sub(4)) {
        // ...
    }
}
",
            "
fn foo(n: usize) {
    if let Some(_) = n.checked_sub(4) {
        // ...
    }
}
",
        );

        check_assist(remove_dbg, "$0dbg!(Foo::foo_test()).bar()", "Foo::foo_test().bar()");
    }

    #[test]
    fn test_remove_dbg_with_brackets_and_braces() {
        check_assist(remove_dbg, "dbg![$01 + 1]", "1 + 1");
        check_assist(remove_dbg, "dbg!{$01 + 1}", "1 + 1");
    }

    #[test]
    fn test_remove_dbg_not_applicable() {
        check_assist_not_applicable(remove_dbg, "$0vec![1, 2, 3]");
        check_assist_not_applicable(remove_dbg, "$0dbg(5, 6, 7)");
        check_assist_not_applicable(remove_dbg, "$0dbg!(5, 6, 7");
    }

    #[test]
    fn test_remove_dbg_target() {
        check_assist_target(
            remove_dbg,
            "
fn foo(n: usize) {
    if let Some(_) = dbg!(n.$0checked_sub(4)) {
        // ...
    }
}
",
            "dbg!(n.checked_sub(4))",
        );
    }

    #[test]
    fn test_remove_dbg_keep_semicolon() {
        // https://github.com/rust-analyzer/rust-analyzer/issues/5129#issuecomment-651399779
        // not quite though
        // adding a comment at the end of the line makes
        // the ast::MacroCall to include the semicolon at the end
        check_assist(
            remove_dbg,
            r#"let res = $0dbg!(1 * 20); // needless comment"#,
            r#"let res = 1 * 20; // needless comment"#,
        );
    }

    #[test]
    fn remove_dbg_from_non_leaf_simple_expression() {
        check_assist(
            remove_dbg,
            "
fn main() {
    let mut a = 1;
    while dbg!$0(a) < 10000 {
        a += 1;
    }
}
",
            "
fn main() {
    let mut a = 1;
    while a < 10000 {
        a += 1;
    }
}
",
        );
    }

    #[test]
    fn test_remove_dbg_keep_expression() {
        check_assist(
            remove_dbg,
            r#"let res = $0dbg!(a + b).foo();"#,
            r#"let res = (a + b).foo();"#,
        );

        check_assist(remove_dbg, r#"let res = $0dbg!(2 + 2) * 5"#, r#"let res = (2 + 2) * 5"#);
        check_assist(remove_dbg, r#"let res = $0dbg![2 + 2] * 5"#, r#"let res = (2 + 2) * 5"#);
    }

    #[test]
    fn test_remove_dbg_method_chaining() {
        check_assist(
            remove_dbg,
            r#"let res = $0dbg!(foo().bar()).baz();"#,
            r#"let res = foo().bar().baz();"#,
        );
        check_assist(
            remove_dbg,
            r#"let res = $0dbg!(foo.bar()).baz();"#,
            r#"let res = foo.bar().baz();"#,
        );
    }

    #[test]
    fn test_remove_dbg_field_chaining() {
        check_assist(remove_dbg, r#"let res = $0dbg!(foo.bar).baz;"#, r#"let res = foo.bar.baz;"#);
    }

    #[test]
    fn test_remove_dbg_from_inside_fn() {
        check_assist_target(
            remove_dbg,
            r#"
fn square(x: u32) -> u32 {
    x * x
}

fn main() {
    let x = square(dbg$0!(5 + 10));
    println!("{}", x);
}"#,
            "dbg!(5 + 10)",
        );

        check_assist(
            remove_dbg,
            r#"
fn square(x: u32) -> u32 {
    x * x
}

fn main() {
    let x = square(dbg$0!(5 + 10));
    println!("{}", x);
}"#,
            r#"
fn square(x: u32) -> u32 {
    x * x
}

fn main() {
    let x = square(5 + 10);
    println!("{}", x);
}"#,
        );
    }

    #[test]
    fn test_remove_dbg_try_expr() {
        check_assist(
            remove_dbg,
            r#"let res = $0dbg!(result?).foo();"#,
            r#"let res = result?.foo();"#,
        );
    }

    #[test]
    fn test_remove_dbg_await_expr() {
        check_assist(
            remove_dbg,
            r#"let res = $0dbg!(fut.await).foo();"#,
            r#"let res = fut.await.foo();"#,
        );
    }

    #[test]
    fn test_remove_dbg_as_cast() {
        check_assist(
            remove_dbg,
            r#"let res = $0dbg!(3 as usize).foo();"#,
            r#"let res = (3 as usize).foo();"#,
        );
    }

    #[test]
    fn test_remove_dbg_index_expr() {
        check_assist(
            remove_dbg,
            r#"let res = $0dbg!(array[3]).foo();"#,
            r#"let res = array[3].foo();"#,
        );
        check_assist(
            remove_dbg,
            r#"let res = $0dbg!(tuple.3).foo();"#,
            r#"let res = tuple.3.foo();"#,
        );
    }

    #[test]
    fn test_remove_dbg_range_expr() {
        check_assist(
            remove_dbg,
            r#"let res = $0dbg!(foo..bar).foo();"#,
            r#"let res = (foo..bar).foo();"#,
        );
        check_assist(
            remove_dbg,
            r#"let res = $0dbg!(foo..=bar).foo();"#,
            r#"let res = (foo..=bar).foo();"#,
        );
    }

    #[test]
    fn test_remove_dbg_followed_by_block() {
        check_assist(
            remove_dbg,
            r#"fn foo() {
    if $0dbg!(x || y) {}
}"#,
            r#"fn foo() {
    if x || y {}
}"#,
        );
        check_assist(
            remove_dbg,
            r#"fn foo() {
    while let foo = $0dbg!(&x) {}
}"#,
            r#"fn foo() {
    while let foo = &x {}
}"#,
        );
        check_assist(
            remove_dbg,
            r#"fn foo() {
    if let foo = $0dbg!(&x) {}
}"#,
            r#"fn foo() {
    if let foo = &x {}
}"#,
        );
        check_assist(
            remove_dbg,
            r#"fn foo() {
    match $0dbg!(&x) {}
}"#,
            r#"fn foo() {
    match &x {}
}"#,
        );
    }
}
