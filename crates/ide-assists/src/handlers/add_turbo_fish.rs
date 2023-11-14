use either::Either;
use ide_db::defs::{Definition, NameRefClass};
use syntax::{
    ast::{self, make, HasArgList},
    ted, AstNode,
};

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
    let turbofish_target =
        ctx.find_node_at_offset::<ast::PathSegment>().map(Either::Left).or_else(|| {
            let callable_expr = ctx.find_node_at_offset::<ast::CallableExpr>()?;

            if callable_expr.arg_list()?.args().next().is_some() {
                return None;
            }

            cov_mark::hit!(add_turbo_fish_after_call);
            cov_mark::hit!(add_type_ascription_after_call);

            match callable_expr {
                ast::CallableExpr::Call(it) => {
                    let ast::Expr::PathExpr(path) = it.expr()? else {
                        return None;
                    };

                    Some(Either::Left(path.path()?.segment()?))
                }
                ast::CallableExpr::MethodCall(it) => Some(Either::Right(it)),
            }
        })?;

    let already_has_turbofish = match &turbofish_target {
        Either::Left(path_segment) => path_segment.generic_arg_list().is_some(),
        Either::Right(method_call) => method_call.generic_arg_list().is_some(),
    };

    if already_has_turbofish {
        cov_mark::hit!(add_turbo_fish_one_fish_is_enough);
        return None;
    }

    let name_ref = match &turbofish_target {
        Either::Left(path_segment) => path_segment.name_ref()?,
        Either::Right(method_call) => method_call.name_ref()?,
    };
    let ident = name_ref.ident_token()?;

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
            if let_stmt.pat().is_none() {
                return None;
            }

            acc.add(
                AssistId("add_type_ascription", AssistKind::RefactorRewrite),
                "Add `: _` before assignment operator",
                ident.text_range(),
                |edit| {
                    let let_stmt = edit.make_mut(let_stmt);

                    if let_stmt.semicolon_token().is_none() {
                        ted::append_child(let_stmt.syntax(), make::tokens::semicolon());
                    }

                    let placeholder_ty = make::ty_placeholder().clone_for_update();

                    let_stmt.set_ty(Some(placeholder_ty.clone()));

                    if let Some(cap) = ctx.config.snippet_cap {
                        edit.add_placeholder_snippet(cap, placeholder_ty);
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
        |edit| {
            edit.trigger_signature_help();

            let new_arg_list = match turbofish_target {
                Either::Left(path_segment) => {
                    edit.make_mut(path_segment).get_or_create_generic_arg_list()
                }
                Either::Right(method_call) => {
                    edit.make_mut(method_call).get_or_create_generic_arg_list()
                }
            };

            let fish_head = get_fish_head(number_of_arguments).clone_for_update();

            // Note: we need to replace the `new_arg_list` instead of being able to use something like
            // `GenericArgList::add_generic_arg` as `PathSegment::get_or_create_generic_arg_list`
            // always creates a non-turbofish form generic arg list.
            ted::replace(new_arg_list.syntax(), fish_head.syntax());

            if let Some(cap) = ctx.config.snippet_cap {
                for arg in fish_head.generic_args() {
                    edit.add_placeholder_snippet(cap, arg)
                }
            }
        },
    )
}

/// This will create a turbofish generic arg list corresponding to the number of arguments
fn get_fish_head(number_of_arguments: usize) -> ast::GenericArgList {
    let args = (0..number_of_arguments).map(|_| make::type_arg(make::ty_placeholder()).into());
    make::turbofish_generic_arg_list(args)
}

#[cfg(test)]
mod tests {
    use crate::tests::{
        check_assist, check_assist_by_label, check_assist_not_applicable,
        check_assist_not_applicable_by_label,
    };

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
    fn add_type_ascription_missing_pattern() {
        check_assist_not_applicable_by_label(
            add_turbo_fish,
            r#"
fn make<T>() -> T {}
fn main() {
    let = make$0()
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
