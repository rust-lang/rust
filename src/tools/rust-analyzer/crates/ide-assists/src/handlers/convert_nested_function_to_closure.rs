use ide_db::assists::AssistId;
use syntax::ast::{self, HasGenericParams, HasName};
use syntax::{AstNode, SyntaxKind};

use crate::assist_context::{AssistContext, Assists};

// Assist: convert_nested_function_to_closure
//
// Converts a function that is defined within the body of another function into a closure.
//
// ```
// fn main() {
//     fn fo$0o(label: &str, number: u64) {
//         println!("{}: {}", label, number);
//     }
//
//     foo("Bar", 100);
// }
// ```
// ->
// ```
// fn main() {
//     let foo = |label: &str, number: u64| {
//         println!("{}: {}", label, number);
//     };
//
//     foo("Bar", 100);
// }
// ```
pub(crate) fn convert_nested_function_to_closure(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let name = ctx.find_node_at_offset::<ast::Name>()?;
    let function = name.syntax().parent().and_then(ast::Fn::cast)?;

    if !is_nested_function(&function) || is_generic(&function) || has_modifiers(&function) {
        return None;
    }

    let target = function.syntax().text_range();
    let body = function.body()?;
    let name = function.name()?;
    let param_list = function.param_list()?;

    acc.add(
        AssistId::refactor_rewrite("convert_nested_function_to_closure"),
        "Convert nested function to closure",
        target,
        |edit| {
            let params = &param_list.syntax().text().to_string();
            let params = params.strip_prefix('(').unwrap_or(params);
            let params = params.strip_suffix(')').unwrap_or(params);

            let mut body = body.to_string();
            if !has_semicolon(&function) {
                body.push(';');
            }
            edit.replace(target, format!("let {name} = |{params}| {body}"));
        },
    )
}

/// Returns whether the given function is nested within the body of another function.
fn is_nested_function(function: &ast::Fn) -> bool {
    function.syntax().ancestors().skip(1).find_map(ast::Item::cast).is_some_and(|it| {
        matches!(it, ast::Item::Fn(_) | ast::Item::Static(_) | ast::Item::Const(_))
    })
}

/// Returns whether the given nested function has generic parameters.
fn is_generic(function: &ast::Fn) -> bool {
    function.generic_param_list().is_some()
}

/// Returns whether the given nested function has any modifiers:
///
/// - `async`,
/// - `const` or
/// - `unsafe`
fn has_modifiers(function: &ast::Fn) -> bool {
    function.async_token().is_some()
        || function.const_token().is_some()
        || function.unsafe_token().is_some()
}

/// Returns whether the given nested function has a trailing semicolon.
fn has_semicolon(function: &ast::Fn) -> bool {
    function
        .syntax()
        .next_sibling_or_token()
        .map(|t| t.kind() == SyntaxKind::SEMICOLON)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::convert_nested_function_to_closure;

    #[test]
    fn convert_nested_function_to_closure_works() {
        check_assist(
            convert_nested_function_to_closure,
            r#"
fn main() {
    fn $0foo(a: u64, b: u64) -> u64 {
        2 * (a + b)
    }

    _ = foo(3, 4);
}
            "#,
            r#"
fn main() {
    let foo = |a: u64, b: u64| {
        2 * (a + b)
    };

    _ = foo(3, 4);
}
            "#,
        );
    }

    #[test]
    fn convert_nested_function_to_closure_works_with_existing_semicolon() {
        check_assist(
            convert_nested_function_to_closure,
            r#"
fn main() {
    fn foo$0(a: u64, b: u64) -> u64 {
        2 * (a + b)
    };

    _ = foo(3, 4);
}
            "#,
            r#"
fn main() {
    let foo = |a: u64, b: u64| {
        2 * (a + b)
    };

    _ = foo(3, 4);
}
            "#,
        );
    }

    #[test]
    fn convert_nested_function_to_closure_is_not_suggested_on_top_level_function() {
        check_assist_not_applicable(
            convert_nested_function_to_closure,
            r#"
fn ma$0in() {}
            "#,
        );
    }

    #[test]
    fn convert_nested_function_to_closure_is_not_suggested_when_cursor_off_name() {
        check_assist_not_applicable(
            convert_nested_function_to_closure,
            r#"
fn main() {
    fn foo(a: u64, $0b: u64) -> u64 {
        2 * (a + b)
    }

    _ = foo(3, 4);
}
            "#,
        );
    }

    #[test]
    fn convert_nested_function_to_closure_is_not_suggested_if_function_has_generic_params() {
        check_assist_not_applicable(
            convert_nested_function_to_closure,
            r#"
fn main() {
    fn fo$0o<S: Into<String>>(s: S) -> String {
        s.into()
    }

    _ = foo("hello");
}
            "#,
        );
    }

    #[test]
    fn convert_nested_function_to_closure_is_not_suggested_if_function_has_modifier() {
        check_assist_not_applicable(
            convert_nested_function_to_closure,
            r#"
fn main() {
    const fn fo$0o(s: String) -> String {
        s
    }

    _ = foo("hello");
}
            "#,
        );
    }
}
