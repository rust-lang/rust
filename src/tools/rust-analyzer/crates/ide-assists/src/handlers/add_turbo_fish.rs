use ide_db::defs::{Definition, NameRefClass};
use itertools::Itertools;
use syntax::{ast, AstNode, SyntaxKind, T};

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: add_turbo_fish
//
// Adds `::<_>` to a call of a generic method or function.
//
// ```
// fn make<T>() -> T { todo!() }
// fn main() {
//     let x = make$0();
// }
// ```
// ->
// ```
// fn make<T>() -> T { todo!() }
// fn main() {
//     let x = make::<${0:_}>();
// }
// ```
pub(crate) fn add_turbo_fish(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let ident = ctx.find_token_syntax_at_offset(SyntaxKind::IDENT).or_else(|| {
        let arg_list = ctx.find_node_at_offset::<ast::ArgList>()?;
        if arg_list.args().next().is_some() {
            return None;
        }
        cov_mark::hit!(add_turbo_fish_after_call);
        cov_mark::hit!(add_type_ascription_after_call);
        arg_list.l_paren_token()?.prev_token().filter(|it| it.kind() == SyntaxKind::IDENT)
    })?;
    let next_token = ident.next_token()?;
    if next_token.kind() == T![::] {
        cov_mark::hit!(add_turbo_fish_one_fish_is_enough);
        return None;
    }
    let name_ref = ast::NameRef::cast(ident.parent()?)?;
    let def = match NameRefClass::classify(&ctx.sema, &name_ref)? {
        NameRefClass::Definition(def) => def,
        NameRefClass::FieldShorthand { .. } | NameRefClass::ExternCrateShorthand { .. } => {
            return None
        }
    };
    let fun = match def {
        Definition::Function(it) => it,
        _ => return None,
    };
    let generics = hir::GenericDef::Function(fun).params(ctx.sema.db);
    if generics.is_empty() {
        cov_mark::hit!(add_turbo_fish_non_generic);
        return None;
    }

    if let Some(let_stmt) = ctx.find_node_at_offset::<ast::LetStmt>() {
        if let_stmt.colon_token().is_none() {
            let type_pos = let_stmt.pat()?.syntax().last_token()?.text_range().end();
            let semi_pos = let_stmt.syntax().last_token()?.text_range().end();

            acc.add(
                AssistId("add_type_ascription", AssistKind::RefactorRewrite),
                "Add `: _` before assignment operator",
                ident.text_range(),
                |builder| {
                    if let_stmt.semicolon_token().is_none() {
                        builder.insert(semi_pos, ";");
                    }
                    match ctx.config.snippet_cap {
                        Some(cap) => builder.insert_snippet(cap, type_pos, ": ${0:_}"),
                        None => builder.insert(type_pos, ": _"),
                    }
                },
            )?
        } else {
            cov_mark::hit!(add_type_ascription_already_typed);
        }
    }

    let number_of_arguments = generics
        .iter()
        .filter(|param| {
            matches!(param, hir::GenericParam::TypeParam(_) | hir::GenericParam::ConstParam(_))
        })
        .count();

    acc.add(
        AssistId("add_turbo_fish", AssistKind::RefactorRewrite),
        "Add `::<>`",
        ident.text_range(),
        |builder| {
            builder.trigger_signature_help();
            match ctx.config.snippet_cap {
                Some(cap) => {
                    let fish_head = get_snippet_fish_head(number_of_arguments);
                    let snip = format!("::<{fish_head}>");
                    builder.insert_snippet(cap, ident.text_range().end(), snip)
                }
                None => {
                    let fish_head = std::iter::repeat("_").take(number_of_arguments).format(", ");
                    let snip = format!("::<{fish_head}>");
                    builder.insert(ident.text_range().end(), snip);
                }
            }
        },
    )
}

/// This will create a snippet string with tabstops marked
fn get_snippet_fish_head(number_of_arguments: usize) -> String {
    let mut fish_head = (1..number_of_arguments)
        .format_with("", |i, f| f(&format_args!("${{{i}:_}}, ")))
        .to_string();

    // tabstop 0 is a special case and always the last one
    fish_head.push_str("${0:_}");
    fish_head
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_by_label, check_assist_not_applicable};

    use super::*;

    #[test]
    fn add_turbo_fish_function() {
        check_assist(
            add_turbo_fish,
            r#"
fn make<T>() -> T {}
fn main() {
    make$0();
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    make::<${0:_}>();
}
"#,
        );
    }

    #[test]
    fn add_turbo_fish_function_multiple_generic_types() {
        check_assist(
            add_turbo_fish,
            r#"
fn make<T, A>() -> T {}
fn main() {
    make$0();
}
"#,
            r#"
fn make<T, A>() -> T {}
fn main() {
    make::<${1:_}, ${0:_}>();
}
"#,
        );
    }

    #[test]
    fn add_turbo_fish_function_many_generic_types() {
        check_assist(
            add_turbo_fish,
            r#"
fn make<T, A, B, C, D, E, F>() -> T {}
fn main() {
    make$0();
}
"#,
            r#"
fn make<T, A, B, C, D, E, F>() -> T {}
fn main() {
    make::<${1:_}, ${2:_}, ${3:_}, ${4:_}, ${5:_}, ${6:_}, ${0:_}>();
}
"#,
        );
    }

    #[test]
    fn add_turbo_fish_after_call() {
        cov_mark::check!(add_turbo_fish_after_call);
        check_assist(
            add_turbo_fish,
            r#"
fn make<T>() -> T {}
fn main() {
    make()$0;
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    make::<${0:_}>();
}
"#,
        );
    }

    #[test]
    fn add_turbo_fish_method() {
        check_assist(
            add_turbo_fish,
            r#"
struct S;
impl S {
    fn make<T>(&self) -> T {}
}
fn main() {
    S.make$0();
}
"#,
            r#"
struct S;
impl S {
    fn make<T>(&self) -> T {}
}
fn main() {
    S.make::<${0:_}>();
}
"#,
        );
    }

    #[test]
    fn add_turbo_fish_one_fish_is_enough() {
        cov_mark::check!(add_turbo_fish_one_fish_is_enough);
        check_assist_not_applicable(
            add_turbo_fish,
            r#"
fn make<T>() -> T {}
fn main() {
    make$0::<()>();
}
"#,
        );
    }

    #[test]
    fn add_turbo_fish_non_generic() {
        cov_mark::check!(add_turbo_fish_non_generic);
        check_assist_not_applicable(
            add_turbo_fish,
            r#"
fn make() -> () {}
fn main() {
    make$0();
}
"#,
        );
    }

    #[test]
    fn add_type_ascription_function() {
        check_assist_by_label(
            add_turbo_fish,
            r#"
fn make<T>() -> T {}
fn main() {
    let x = make$0();
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    let x: ${0:_} = make();
}
"#,
            "Add `: _` before assignment operator",
        );
    }

    #[test]
    fn add_type_ascription_after_call() {
        cov_mark::check!(add_type_ascription_after_call);
        check_assist_by_label(
            add_turbo_fish,
            r#"
fn make<T>() -> T {}
fn main() {
    let x = make()$0;
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    let x: ${0:_} = make();
}
"#,
            "Add `: _` before assignment operator",
        );
    }

    #[test]
    fn add_type_ascription_method() {
        check_assist_by_label(
            add_turbo_fish,
            r#"
struct S;
impl S {
    fn make<T>(&self) -> T {}
}
fn main() {
    let x = S.make$0();
}
"#,
            r#"
struct S;
impl S {
    fn make<T>(&self) -> T {}
}
fn main() {
    let x: ${0:_} = S.make();
}
"#,
            "Add `: _` before assignment operator",
        );
    }

    #[test]
    fn add_type_ascription_already_typed() {
        cov_mark::check!(add_type_ascription_already_typed);
        check_assist(
            add_turbo_fish,
            r#"
fn make<T>() -> T {}
fn main() {
    let x: () = make$0();
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    let x: () = make::<${0:_}>();
}
"#,
        );
    }

    #[test]
    fn add_type_ascription_append_semicolon() {
        check_assist_by_label(
            add_turbo_fish,
            r#"
fn make<T>() -> T {}
fn main() {
    let x = make$0()
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    let x: ${0:_} = make();
}
"#,
            "Add `: _` before assignment operator",
        );
    }

    #[test]
    fn add_turbo_fish_function_lifetime_parameter() {
        check_assist(
            add_turbo_fish,
            r#"
fn make<'a, T, A>(t: T, a: A) {}
fn main() {
    make$0(5, 2);
}
"#,
            r#"
fn make<'a, T, A>(t: T, a: A) {}
fn main() {
    make::<${1:_}, ${0:_}>(5, 2);
}
"#,
        );
    }

    #[test]
    fn add_turbo_fish_function_const_parameter() {
        check_assist(
            add_turbo_fish,
            r#"
fn make<T, const N: usize>(t: T) {}
fn main() {
    make$0(3);
}
"#,
            r#"
fn make<T, const N: usize>(t: T) {}
fn main() {
    make::<${1:_}, ${0:_}>(3);
}
"#,
        );
    }
}
