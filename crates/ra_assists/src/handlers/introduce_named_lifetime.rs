use ra_syntax::{
    ast::{self, NameOwner, TypeAscriptionOwner, TypeParamsOwner},
    AstNode, SyntaxKind, TextRange, TextSize,
};
use rustc_hash::FxHashSet;

use crate::{assist_context::AssistBuilder, AssistContext, AssistId, AssistKind, Assists};

static ASSIST_NAME: &str = "introduce_named_lifetime";
static ASSIST_LABEL: &str = "Introduce named lifetime";

// Assist: introduce_named_lifetime
//
// Change an anonymous lifetime to a named lifetime.
//
// ```
// impl Cursor<'_<|>> {
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
// FIXME: How can we handle renaming any one of multiple anonymous lifetimes?
// FIXME: should also add support for the case fun(f: &Foo) -> &<|>Foo
pub(crate) fn introduce_named_lifetime(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let lifetime_token = ctx
        .find_token_at_offset(SyntaxKind::LIFETIME)
        .filter(|lifetime| lifetime.text() == "'_")?;
    if let Some(fn_def) = lifetime_token.ancestors().find_map(ast::FnDef::cast) {
        generate_fn_def_assist(acc, &fn_def, lifetime_token.text_range())
    } else if let Some(impl_def) = lifetime_token.ancestors().find_map(ast::ImplDef::cast) {
        generate_impl_def_assist(acc, &impl_def, lifetime_token.text_range())
    } else {
        None
    }
}

/// Generate the assist for the fn def case
fn generate_fn_def_assist(
    acc: &mut Assists,
    fn_def: &ast::FnDef,
    lifetime_loc: TextRange,
) -> Option<()> {
    let param_list: ast::ParamList = fn_def.param_list()?;
    let new_lifetime_param = generate_unique_lifetime_param_name(&fn_def.type_param_list())?;
    let end_of_fn_ident = fn_def.name()?.ident_token()?.text_range().end();
    let self_param =
        // use the self if it's a reference and has no explicit lifetime
        param_list.self_param().filter(|p| p.lifetime_token().is_none() && p.amp_token().is_some());
    // compute the location which implicitly has the same lifetime as the anonymous lifetime
    let loc_needing_lifetime = if let Some(self_param) = self_param {
        // if we have a self reference, use that
        Some(self_param.self_token()?.text_range().start())
    } else {
        // otherwise, if there's a single reference parameter without a named liftime, use that
        let fn_params_without_lifetime: Vec<_> = param_list
            .params()
            .filter_map(|param| match param.ascribed_type() {
                Some(ast::TypeRef::ReferenceType(ascribed_type))
                    if ascribed_type.lifetime_token() == None =>
                {
                    Some(ascribed_type.amp_token()?.text_range().end())
                }
                _ => None,
            })
            .collect();
        match fn_params_without_lifetime.len() {
            1 => Some(fn_params_without_lifetime.into_iter().nth(0)?),
            0 => None,
            // multiple unnnamed is invalid. assist is not applicable
            _ => return None,
        }
    };
    acc.add(AssistId(ASSIST_NAME, AssistKind::Refactor), ASSIST_LABEL, lifetime_loc, |builder| {
        add_lifetime_param(fn_def, builder, end_of_fn_ident, new_lifetime_param);
        builder.replace(lifetime_loc, format!("'{}", new_lifetime_param));
        loc_needing_lifetime.map(|loc| builder.insert(loc, format!("'{} ", new_lifetime_param)));
    })
}

/// Generate the assist for the impl def case
fn generate_impl_def_assist(
    acc: &mut Assists,
    impl_def: &ast::ImplDef,
    lifetime_loc: TextRange,
) -> Option<()> {
    let new_lifetime_param = generate_unique_lifetime_param_name(&impl_def.type_param_list())?;
    let end_of_impl_kw = impl_def.impl_token()?.text_range().end();
    acc.add(AssistId(ASSIST_NAME, AssistKind::Refactor), ASSIST_LABEL, lifetime_loc, |builder| {
        add_lifetime_param(impl_def, builder, end_of_impl_kw, new_lifetime_param);
        builder.replace(lifetime_loc, format!("'{}", new_lifetime_param));
    })
}

/// Given a type parameter list, generate a unique lifetime parameter name
/// which is not in the list
fn generate_unique_lifetime_param_name(
    existing_type_param_list: &Option<ast::TypeParamList>,
) -> Option<char> {
    match existing_type_param_list {
        Some(type_params) => {
            let used_lifetime_params: FxHashSet<_> = type_params
                .lifetime_params()
                .map(|p| p.syntax().text().to_string()[1..].to_owned())
                .collect();
            (b'a'..=b'z').map(char::from).find(|c| !used_lifetime_params.contains(&c.to_string()))
        }
        None => Some('a'),
    }
}

/// Add the lifetime param to `builder`. If there are type parameters in `type_params_owner`, add it to the end. Otherwise
/// add new type params brackets with the lifetime parameter at `new_type_params_loc`.
fn add_lifetime_param<TypeParamsOwner: ast::TypeParamsOwner>(
    type_params_owner: &TypeParamsOwner,
    builder: &mut AssistBuilder,
    new_type_params_loc: TextSize,
    new_lifetime_param: char,
) {
    match type_params_owner.type_param_list() {
        // add the new lifetime parameter to an existing type param list
        Some(type_params) => {
            builder.insert(
                (u32::from(type_params.syntax().text_range().end()) - 1).into(),
                format!(", '{}", new_lifetime_param),
            );
        }
        // create a new type param list containing only the new lifetime parameter
        None => {
            builder.insert(new_type_params_loc, format!("<'{}>", new_lifetime_param));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn test_example_case() {
        check_assist(
            introduce_named_lifetime,
            r#"impl Cursor<'_<|>> {
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
            r#"impl Cursor<'_<|>> {"#,
            r#"impl<'a> Cursor<'a> {"#,
        );
    }

    #[test]
    fn test_example_case_cursor_after_tick() {
        check_assist(
            introduce_named_lifetime,
            r#"impl Cursor<'<|>_> {"#,
            r#"impl<'a> Cursor<'a> {"#,
        );
    }

    #[test]
    fn test_impl_with_other_type_param() {
        check_assist(
            introduce_named_lifetime,
            "impl<I> fmt::Display for SepByBuilder<'_<|>, I>
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
            r#"impl Cursor<<|>'_> {"#,
            r#"impl<'a> Cursor<'a> {"#,
        );
    }

    #[test]
    fn test_not_applicable_cursor_position() {
        check_assist_not_applicable(introduce_named_lifetime, r#"impl Cursor<'_><|> {"#);
        check_assist_not_applicable(introduce_named_lifetime, r#"impl Cursor<|><'_> {"#);
    }

    #[test]
    fn test_not_applicable_lifetime_already_name() {
        check_assist_not_applicable(introduce_named_lifetime, r#"impl Cursor<'a<|>> {"#);
        check_assist_not_applicable(introduce_named_lifetime, r#"fn my_fun<'a>() -> X<'a<|>>"#);
    }

    #[test]
    fn test_with_type_parameter() {
        check_assist(
            introduce_named_lifetime,
            r#"impl<T> Cursor<T, '_<|>>"#,
            r#"impl<T, 'a> Cursor<T, 'a>"#,
        );
    }

    #[test]
    fn test_with_existing_lifetime_name_conflict() {
        check_assist(
            introduce_named_lifetime,
            r#"impl<'a, 'b> Cursor<'a, 'b, '_<|>>"#,
            r#"impl<'a, 'b, 'c> Cursor<'a, 'b, 'c>"#,
        );
    }

    #[test]
    fn test_function_return_value_anon_lifetime_param() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun() -> X<'_<|>>"#,
            r#"fn my_fun<'a>() -> X<'a>"#,
        );
    }

    #[test]
    fn test_function_return_value_anon_reference_lifetime() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun() -> &'_<|> X"#,
            r#"fn my_fun<'a>() -> &'a X"#,
        );
    }

    #[test]
    fn test_function_param_anon_lifetime() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun(x: X<'_<|>>)"#,
            r#"fn my_fun<'a>(x: X<'a>)"#,
        );
    }

    #[test]
    fn test_function_add_lifetime_to_params() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun(f: &Foo) -> X<'_<|>>"#,
            r#"fn my_fun<'a>(f: &'a Foo) -> X<'a>"#,
        );
    }

    #[test]
    fn test_function_add_lifetime_to_params_in_presence_of_other_lifetime() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun<'other>(f: &Foo, b: &'other Bar) -> X<'_<|>>"#,
            r#"fn my_fun<'other, 'a>(f: &'a Foo, b: &'other Bar) -> X<'a>"#,
        );
    }

    #[test]
    fn test_function_not_applicable_without_self_and_multiple_unnamed_param_lifetimes() {
        // this is not permitted under lifetime elision rules
        check_assist_not_applicable(
            introduce_named_lifetime,
            r#"fn my_fun(f: &Foo, b: &Bar) -> X<'_<|>>"#,
        );
    }

    #[test]
    fn test_function_add_lifetime_to_self_ref_param() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun<'other>(&self, f: &Foo, b: &'other Bar) -> X<'_<|>>"#,
            r#"fn my_fun<'other, 'a>(&'a self, f: &Foo, b: &'other Bar) -> X<'a>"#,
        );
    }

    #[test]
    fn test_function_add_lifetime_to_param_with_non_ref_self() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun<'other>(self, f: &Foo, b: &'other Bar) -> X<'_<|>>"#,
            r#"fn my_fun<'other, 'a>(self, f: &'a Foo, b: &'other Bar) -> X<'a>"#,
        );
    }
}
