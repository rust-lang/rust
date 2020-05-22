use crate::{AssistContext, AssistId, Assists};
use ra_syntax::{ast, ast::TypeParamsOwner, AstNode, SyntaxKind};
use std::collections::HashSet;

/// Assist: change_lifetime_anon_to_named
///
/// Change an anonymous lifetime to a named lifetime.
///
/// ```
/// impl Cursor<'_<|>> {
///     fn node(self) -> &SyntaxNode {
///         match self {
///             Cursor::Replace(node) | Cursor::Before(node) => node,
///         }
///     }
/// }
/// ```
/// ->
/// ```
/// impl<'a> Cursor<'a> {
///     fn node(self) -> &SyntaxNode {
///         match self {
///             Cursor::Replace(node) | Cursor::Before(node) => node,
///         }
///     }
/// }
/// ```
// FIXME: How can we handle renaming any one of multiple anonymous lifetimes?
pub(crate) fn change_lifetime_anon_to_named(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let lifetime_token = ctx.find_token_at_offset(SyntaxKind::LIFETIME)?;
    let lifetime_arg = ast::LifetimeArg::cast(lifetime_token.parent())?;
    if lifetime_arg.syntax().text() != "'_" {
        return None;
    }
    let next_token = lifetime_token.next_token()?;
    if next_token.kind() != SyntaxKind::R_ANGLE {
        // only allow naming the last anonymous lifetime
        return None;
    }
    match lifetime_arg.syntax().ancestors().find_map(ast::ImplDef::cast) {
        Some(impl_def) => {
            // get the `impl` keyword so we know where to add the lifetime argument
            let impl_kw = impl_def.syntax().first_child_or_token()?.into_token()?;
            if impl_kw.kind() != SyntaxKind::IMPL_KW {
                return None;
            }
            let new_lifetime_param = match impl_def.type_param_list() {
                Some(type_params) => {
                    let used_lifetime_params: HashSet<_> = type_params
                        .lifetime_params()
                        .map(|p| {
                            let mut param_name = p.syntax().text().to_string();
                            param_name.remove(0);
                            param_name
                        })
                        .collect();
                    "abcdefghijklmnopqrstuvwxyz"
                        .chars()
                        .find(|c| !used_lifetime_params.contains(&c.to_string()))?
                }
                None => 'a',
            };
            acc.add(
                AssistId("change_lifetime_anon_to_named"),
                "Give anonymous lifetime a name",
                lifetime_arg.syntax().text_range(),
                |builder| {
                    match impl_def.type_param_list() {
                        Some(type_params) => {
                            builder.insert(
                                (u32::from(type_params.syntax().text_range().end()) - 1).into(),
                                format!(", '{}", new_lifetime_param),
                            );
                        }
                        None => {
                            builder.insert(
                                impl_kw.text_range().end(),
                                format!("<'{}>", new_lifetime_param),
                            );
                        }
                    }
                    builder.replace(
                        lifetime_arg.syntax().text_range(),
                        format!("'{}", new_lifetime_param),
                    );
                },
            )
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn test_example_case() {
        check_assist(
            change_lifetime_anon_to_named,
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
            change_lifetime_anon_to_named,
            r#"impl Cursor<'_<|>> {"#,
            r#"impl<'a> Cursor<'a> {"#,
        );
    }

    #[test]
    fn test_not_applicable() {
        check_assist_not_applicable(change_lifetime_anon_to_named, r#"impl Cursor<'_><|> {"#);
        check_assist_not_applicable(change_lifetime_anon_to_named, r#"impl Cursor<|><'_> {"#);
        check_assist_not_applicable(change_lifetime_anon_to_named, r#"impl Cursor<'a<|>> {"#);
    }

    #[test]
    fn test_with_type_parameter() {
        check_assist(
            change_lifetime_anon_to_named,
            r#"impl<T> Cursor<T, '_<|>>"#,
            r#"impl<T, 'a> Cursor<T, 'a>"#,
        );
    }

    #[test]
    fn test_with_existing_lifetime_name_conflict() {
        check_assist(
            change_lifetime_anon_to_named,
            r#"impl<'a, 'b> Cursor<'a, 'b, '_<|>>"#,
            r#"impl<'a, 'b, 'c> Cursor<'a, 'b, 'c>"#,
        );
    }
}
