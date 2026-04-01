use ide_db::{FileId, FxHashSet};
use syntax::{
    AstNode, SmolStr, T, TextRange, ToSmolStr,
    ast::{self, HasGenericParams, HasName, syntax_factory::SyntaxFactory},
    format_smolstr,
    syntax_editor::{Element, Position, SyntaxEditor},
};

use crate::{AssistContext, AssistId, Assists};

static ASSIST_NAME: &str = "introduce_named_lifetime";
static ASSIST_LABEL: &str = "Introduce named lifetime";

// Assist: introduce_named_lifetime
//
// Change an anonymous lifetime to a named lifetime.
//
// ```
// impl Cursor<'_$0> {
//     fn node(self) -> &SyntaxNode {
//         match self {
//             Cursor::Replace(node) | Cursor::Before(node) => node,
//         }
//     }
// }
// ```
// ->
// ```
// impl<'a> Cursor<'a> {
//     fn node(self) -> &SyntaxNode {
//         match self {
//             Cursor::Replace(node) | Cursor::Before(node) => node,
//         }
//     }
// }
// ```
pub(crate) fn introduce_named_lifetime(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    // FIXME: How can we handle renaming any one of multiple anonymous lifetimes?
    // FIXME: should also add support for the case fun(f: &Foo) -> &$0Foo
    let lifetime =
        ctx.find_node_at_offset::<ast::Lifetime>().filter(|lifetime| lifetime.text() == "'_")?;
    let file_id = ctx.vfs_file_id();
    let lifetime_loc = lifetime.lifetime_ident_token()?.text_range();

    if let Some(fn_def) = lifetime.syntax().ancestors().find_map(ast::Fn::cast) {
        generate_fn_def_assist(acc, fn_def, lifetime_loc, lifetime, file_id)
    } else if let Some(impl_def) = lifetime.syntax().ancestors().find_map(ast::Impl::cast) {
        generate_impl_def_assist(acc, impl_def, lifetime_loc, lifetime, file_id)
    } else {
        None
    }
}

/// Given a type parameter list, generate a unique lifetime parameter name
/// which is not in the list
fn generate_unique_lifetime_param_name(
    existing_params: Option<ast::GenericParamList>,
) -> Option<SmolStr> {
    let used_lifetime_param: FxHashSet<SmolStr> = existing_params
        .iter()
        .flat_map(|params| params.lifetime_params())
        .map(|p| p.syntax().text().to_smolstr())
        .collect();
    ('a'..='z').map(|c| format_smolstr!("'{c}")).find(|lt| !used_lifetime_param.contains(lt))
}

fn generate_fn_def_assist(
    acc: &mut Assists,
    fn_def: ast::Fn,
    lifetime_loc: TextRange,
    lifetime: ast::Lifetime,
    file_id: FileId,
) -> Option<()> {
    let param_list = fn_def.param_list()?;
    let new_lifetime_name = generate_unique_lifetime_param_name(fn_def.generic_param_list())?;
    let self_param =
        param_list.self_param().filter(|p| p.lifetime().is_none() && p.amp_token().is_some());

    let loc_needing_lifetime = if let Some(self_param) = self_param {
        Some(NeedsLifetime::SelfParam(self_param))
    } else {
        let unnamed_refs: Vec<_> = param_list
            .params()
            .filter_map(|param| match param.ty() {
                Some(ast::Type::RefType(ref_type)) if ref_type.lifetime().is_none() => {
                    Some(NeedsLifetime::RefType(ref_type))
                }
                _ => None,
            })
            .collect();

        match unnamed_refs.len() {
            1 => Some(unnamed_refs.into_iter().next()?),
            0 => None,
            _ => return None,
        }
    };

    acc.add(AssistId::refactor(ASSIST_NAME), ASSIST_LABEL, lifetime_loc, |edit| {
        let root = fn_def.syntax().ancestors().last().unwrap().clone();
        let mut editor = SyntaxEditor::new(root);
        let factory = SyntaxFactory::with_mappings();

        if let Some(generic_list) = fn_def.generic_param_list() {
            insert_lifetime_param(&mut editor, &factory, &generic_list, &new_lifetime_name);
        } else {
            insert_new_generic_param_list_fn(&mut editor, &factory, &fn_def, &new_lifetime_name);
        }

        editor.replace(lifetime.syntax(), factory.lifetime(&new_lifetime_name).syntax());

        if let Some(pos) = loc_needing_lifetime.and_then(|l| l.to_position()) {
            editor.insert_all(
                pos,
                vec![
                    factory.lifetime(&new_lifetime_name).syntax().clone().into(),
                    factory.whitespace(" ").into(),
                ],
            );
        }

        edit.add_file_edits(file_id, editor);
    })
}

fn insert_new_generic_param_list_fn(
    editor: &mut SyntaxEditor,
    factory: &SyntaxFactory,
    fn_def: &ast::Fn,
    lifetime_name: &str,
) -> Option<()> {
    let name = fn_def.name()?;

    editor.insert_all(
        Position::after(name.syntax()),
        vec![
            factory.token(T![<]).syntax_element(),
            factory.lifetime(lifetime_name).syntax().syntax_element(),
            factory.token(T![>]).syntax_element(),
        ],
    );

    Some(())
}

enum NeedsLifetime {
    SelfParam(ast::SelfParam),
    RefType(ast::RefType),
}

impl NeedsLifetime {
    fn to_position(self) -> Option<Position> {
        match self {
            Self::SelfParam(it) => Some(Position::after(it.amp_token()?)),
            Self::RefType(it) => Some(Position::after(it.amp_token()?)),
        }
    }
}

fn generate_impl_def_assist(
    acc: &mut Assists,
    impl_def: ast::Impl,
    lifetime_loc: TextRange,
    lifetime: ast::Lifetime,
    file_id: FileId,
) -> Option<()> {
    let new_lifetime_name = generate_unique_lifetime_param_name(impl_def.generic_param_list())?;

    acc.add(AssistId::refactor(ASSIST_NAME), ASSIST_LABEL, lifetime_loc, |edit| {
        let root = impl_def.syntax().ancestors().last().unwrap().clone();
        let mut editor = SyntaxEditor::new(root);
        let factory = SyntaxFactory::without_mappings();

        if let Some(generic_list) = impl_def.generic_param_list() {
            insert_lifetime_param(&mut editor, &factory, &generic_list, &new_lifetime_name);
        } else {
            insert_new_generic_param_list_imp(&mut editor, &factory, &impl_def, &new_lifetime_name);
        }

        editor.replace(lifetime.syntax(), factory.lifetime(&new_lifetime_name).syntax());

        edit.add_file_edits(file_id, editor);
    })
}

fn insert_new_generic_param_list_imp(
    editor: &mut SyntaxEditor,
    factory: &SyntaxFactory,
    impl_: &ast::Impl,
    lifetime_name: &str,
) -> Option<()> {
    let impl_kw = impl_.impl_token()?;

    editor.insert_all(
        Position::after(impl_kw),
        vec![
            factory.token(T![<]).syntax_element(),
            factory.lifetime(lifetime_name).syntax().syntax_element(),
            factory.token(T![>]).syntax_element(),
        ],
    );

    Some(())
}

fn insert_lifetime_param(
    editor: &mut SyntaxEditor,
    factory: &SyntaxFactory,
    generic_list: &ast::GenericParamList,
    lifetime_name: &str,
) -> Option<()> {
    let r_angle = generic_list.r_angle_token()?;
    let needs_comma = generic_list.generic_params().next().is_some();

    let mut elements = Vec::new();

    if needs_comma {
        elements.push(factory.token(T![,]).syntax_element());
        elements.push(factory.whitespace(" ").syntax_element());
    }

    let lifetime = factory.lifetime(lifetime_name);
    elements.push(lifetime.syntax().clone().into());

    editor.insert_all(Position::before(r_angle), elements);
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn test_example_case() {
        check_assist(
            introduce_named_lifetime,
            r#"impl Cursor<'_$0> {
                fn node(self) -> &SyntaxNode {
                    match self {
                        Cursor::Replace(node) | Cursor::Before(node) => node,
                    }
                }
            }"#,
            r#"impl<'a> Cursor<'a> {
                fn node(self) -> &SyntaxNode {
                    match self {
                        Cursor::Replace(node) | Cursor::Before(node) => node,
                    }
                }
            }"#,
        );
    }

    #[test]
    fn test_example_case_simplified() {
        check_assist(
            introduce_named_lifetime,
            r#"impl Cursor<'_$0> {"#,
            r#"impl<'a> Cursor<'a> {"#,
        );
    }

    #[test]
    fn test_example_case_cursor_after_tick() {
        check_assist(
            introduce_named_lifetime,
            r#"impl Cursor<'$0_> {"#,
            r#"impl<'a> Cursor<'a> {"#,
        );
    }

    #[test]
    fn test_impl_with_other_type_param() {
        check_assist(
            introduce_named_lifetime,
            "impl<I> fmt::Display for SepByBuilder<'_$0, I>
        where
            I: Iterator,
            I::Item: fmt::Display,
        {",
            "impl<I, 'a> fmt::Display for SepByBuilder<'a, I>
        where
            I: Iterator,
            I::Item: fmt::Display,
        {",
        )
    }

    #[test]
    fn test_example_case_cursor_before_tick() {
        check_assist(
            introduce_named_lifetime,
            r#"impl Cursor<$0'_> {"#,
            r#"impl<'a> Cursor<'a> {"#,
        );
    }

    #[test]
    fn test_not_applicable_cursor_position() {
        check_assist_not_applicable(introduce_named_lifetime, r#"impl Cursor<'_>$0 {"#);
        check_assist_not_applicable(introduce_named_lifetime, r#"impl Cursor$0<'_> {"#);
    }

    #[test]
    fn test_not_applicable_lifetime_already_name() {
        check_assist_not_applicable(introduce_named_lifetime, r#"impl Cursor<'a$0> {"#);
        check_assist_not_applicable(introduce_named_lifetime, r#"fn my_fun<'a>() -> X<'a$0>"#);
    }

    #[test]
    fn test_with_type_parameter() {
        check_assist(
            introduce_named_lifetime,
            r#"impl<T> Cursor<T, '_$0>"#,
            r#"impl<T, 'a> Cursor<T, 'a>"#,
        );
    }

    #[test]
    fn test_with_existing_lifetime_name_conflict() {
        check_assist(
            introduce_named_lifetime,
            r#"impl<'a, 'b> Cursor<'a, 'b, '_$0>"#,
            r#"impl<'a, 'b, 'c> Cursor<'a, 'b, 'c>"#,
        );
    }

    #[test]
    fn test_function_return_value_anon_lifetime_param() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun() -> X<'_$0>"#,
            r#"fn my_fun<'a>() -> X<'a>"#,
        );
    }

    #[test]
    fn test_function_return_value_anon_reference_lifetime() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun() -> &'_$0 X"#,
            r#"fn my_fun<'a>() -> &'a X"#,
        );
    }

    #[test]
    fn test_function_param_anon_lifetime() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun(x: X<'_$0>)"#,
            r#"fn my_fun<'a>(x: X<'a>)"#,
        );
    }

    #[test]
    fn test_function_add_lifetime_to_params() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun(f: &Foo) -> X<'_$0>"#,
            r#"fn my_fun<'a>(f: &'a Foo) -> X<'a>"#,
        );
    }

    #[test]
    fn test_function_add_lifetime_to_params_in_presence_of_other_lifetime() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun<'other>(f: &Foo, b: &'other Bar) -> X<'_$0>"#,
            r#"fn my_fun<'other, 'a>(f: &'a Foo, b: &'other Bar) -> X<'a>"#,
        );
    }

    #[test]
    fn test_function_not_applicable_without_self_and_multiple_unnamed_param_lifetimes() {
        // this is not permitted under lifetime elision rules
        check_assist_not_applicable(
            introduce_named_lifetime,
            r#"fn my_fun(f: &Foo, b: &Bar) -> X<'_$0>"#,
        );
    }

    #[test]
    fn test_function_add_lifetime_to_self_ref_param() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun<'other>(&self, f: &Foo, b: &'other Bar) -> X<'_$0>"#,
            r#"fn my_fun<'other, 'a>(&'a self, f: &Foo, b: &'other Bar) -> X<'a>"#,
        );
    }

    #[test]
    fn test_function_add_lifetime_to_param_with_non_ref_self() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun<'other>(self, f: &Foo, b: &'other Bar) -> X<'_$0>"#,
            r#"fn my_fun<'other, 'a>(self, f: &'a Foo, b: &'other Bar) -> X<'a>"#,
        );
    }

    #[test]
    fn test_function_add_lifetime_to_self_ref_mut() {
        check_assist(
            introduce_named_lifetime,
            r#"fn foo(&mut self) -> &'_$0 ()"#,
            r#"fn foo<'a>(&'a mut self) -> &'a ()"#,
        );
    }
}
