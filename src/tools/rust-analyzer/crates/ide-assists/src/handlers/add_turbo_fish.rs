use either::Either;
use ide_db::defs::{Definition, NameRefClass};
use syntax::{
    AstNode,
    ast::{self, HasArgList, HasGenericArgs, make, syntax_factory::SyntaxFactory},
    syntax_editor::Position,
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists},
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
        NameRefClass::Definition(def, _) => def,
        NameRefClass::FieldShorthand { .. } | NameRefClass::ExternCrateShorthand { .. } => {
            return None;
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
            let_stmt.pat()?;

            acc.add(
                AssistId::refactor_rewrite("add_type_ascription"),
                "Add `: _` before assignment operator",
                ident.text_range(),
                |builder| {
                    let mut editor = builder.make_editor(let_stmt.syntax());

                    if let_stmt.semicolon_token().is_none() {
                        editor.insert(
                            Position::last_child_of(let_stmt.syntax()),
                            make::tokens::semicolon(),
                        );
                    }

                    let placeholder_ty = make::ty_placeholder().clone_for_update();

                    if let Some(pat) = let_stmt.pat() {
                        let elements = vec![
                            make::token(syntax::SyntaxKind::COLON).into(),
                            make::token(syntax::SyntaxKind::WHITESPACE).into(),
                            placeholder_ty.syntax().clone().into(),
                        ];
                        editor.insert_all(Position::after(pat.syntax()), elements);
                        if let Some(cap) = ctx.config.snippet_cap {
                            editor.add_annotation(
                                placeholder_ty.syntax(),
                                builder.make_placeholder_snippet(cap),
                            );
                        }
                    }

                    builder.add_file_edits(ctx.vfs_file_id(), editor);
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
        AssistId::refactor_rewrite("add_turbo_fish"),
        "Add `::<>`",
        ident.text_range(),
        |builder| {
            builder.trigger_parameter_hints();

            let make = SyntaxFactory::with_mappings();
            let mut editor = match &turbofish_target {
                Either::Left(it) => builder.make_editor(it.syntax()),
                Either::Right(it) => builder.make_editor(it.syntax()),
            };

            let fish_head = get_fish_head(&make, number_of_arguments);

            match turbofish_target {
                Either::Left(path_segment) => {
                    if let Some(generic_arg_list) = path_segment.generic_arg_list() {
                        editor.replace(generic_arg_list.syntax(), fish_head.syntax());
                    } else {
                        editor.insert(
                            Position::last_child_of(path_segment.syntax()),
                            fish_head.syntax(),
                        );
                    }
                }
                Either::Right(method_call) => {
                    if let Some(generic_arg_list) = method_call.generic_arg_list() {
                        editor.replace(generic_arg_list.syntax(), fish_head.syntax());
                    } else {
                        let position = if let Some(arg_list) = method_call.arg_list() {
                            Position::before(arg_list.syntax())
                        } else {
                            Position::last_child_of(method_call.syntax())
                        };
                        editor.insert(position, fish_head.syntax());
                    }
                }
            };

            if let Some(cap) = ctx.config.snippet_cap {
                for arg in fish_head.generic_args() {
                    editor.add_annotation(arg.syntax(), builder.make_placeholder_snippet(cap));
                }
            }

            editor.add_mappings(make.finish_with_mappings());
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

/// This will create a turbofish generic arg list corresponding to the number of arguments
fn get_fish_head(make: &SyntaxFactory, number_of_arguments: usize) -> ast::GenericArgList {
    let args = (0..number_of_arguments).map(|_| make::type_arg(make::ty_placeholder()).into());
    make.generic_arg_list(args, true)
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
