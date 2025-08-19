use ide_db::source_change::SourceChangeBuilder;
use itertools::Itertools;
use syntax::{
    NodeOrToken, SyntaxToken, T, TextRange, algo,
    ast::{self, AstNode, make, syntax_factory::SyntaxFactory},
};

use crate::{AssistContext, AssistId, Assists};

// Assist: wrap_unwrap_cfg_attr
//
// Wraps an attribute to a cfg_attr attribute or unwraps a cfg_attr attribute to the inner attributes.
//
// ```
// #[derive$0(Debug)]
// struct S {
//    field: i32
// }
// ```
// ->
// ```
// #[cfg_attr($0, derive(Debug))]
// struct S {
//    field: i32
// }
// ```

enum WrapUnwrapOption {
    WrapDerive { derive: TextRange, attr: ast::Attr },
    WrapAttr(ast::Attr),
}

/// Attempts to get the derive attribute from a derive attribute list
///
/// This will collect all the tokens in the "path" within the derive attribute list
/// But a derive attribute list doesn't have paths. So we need to collect all the tokens before and after the ident
///
/// If this functions return None just map to WrapAttr
fn attempt_get_derive(attr: ast::Attr, ident: SyntaxToken) -> WrapUnwrapOption {
    let attempt_attr = || {
        {
            let mut derive = ident.text_range();
            // TokenTree is all the tokens between the `(` and `)`. They do not have paths. So a path `serde::Serialize` would be [Ident Colon Colon Ident]
            // So lets say we have derive(Debug, serde::Serialize, Copy) ident would be on Serialize
            // We need to grab all previous tokens until we find a `,` or `(` and all following tokens until we find a `,` or `)`
            // We also want to consume the following comma if it exists

            let mut prev = algo::skip_trivia_token(
                ident.prev_sibling_or_token()?.into_token()?,
                syntax::Direction::Prev,
            )?;
            let mut following = algo::skip_trivia_token(
                ident.next_sibling_or_token()?.into_token()?,
                syntax::Direction::Next,
            )?;
            if (prev.kind() == T![,] || prev.kind() == T!['('])
                && (following.kind() == T![,] || following.kind() == T![')'])
            {
                // This would be a single ident such as Debug. As no path is present
                if following.kind() == T![,] {
                    derive = derive.cover(following.text_range());
                } else if following.kind() == T![')'] && prev.kind() == T![,] {
                    derive = derive.cover(prev.text_range());
                }

                Some(WrapUnwrapOption::WrapDerive { derive, attr: attr.clone() })
            } else {
                let mut consumed_comma = false;
                // Collect the path
                while let Some(prev_token) = algo::skip_trivia_token(prev, syntax::Direction::Prev)
                {
                    let kind = prev_token.kind();
                    if kind == T![,] {
                        consumed_comma = true;
                        derive = derive.cover(prev_token.text_range());
                        break;
                    } else if kind == T!['('] {
                        break;
                    } else {
                        derive = derive.cover(prev_token.text_range());
                    }
                    prev = prev_token.prev_sibling_or_token()?.into_token()?;
                }
                while let Some(next_token) =
                    algo::skip_trivia_token(following.clone(), syntax::Direction::Next)
                {
                    let kind = next_token.kind();
                    match kind {
                        T![,] if !consumed_comma => {
                            derive = derive.cover(next_token.text_range());
                            break;
                        }
                        T![')'] | T![,] => break,
                        _ => derive = derive.cover(next_token.text_range()),
                    }
                    following = next_token.next_sibling_or_token()?.into_token()?;
                }
                Some(WrapUnwrapOption::WrapDerive { derive, attr: attr.clone() })
            }
        }
    };
    if ident.parent().and_then(ast::TokenTree::cast).is_none()
        || !attr.simple_name().map(|v| v.eq("derive")).unwrap_or_default()
    {
        WrapUnwrapOption::WrapAttr(attr)
    } else {
        attempt_attr().unwrap_or(WrapUnwrapOption::WrapAttr(attr))
    }
}
pub(crate) fn wrap_unwrap_cfg_attr(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let option = if ctx.has_empty_selection() {
        let ident = ctx.find_token_syntax_at_offset(T![ident]);
        let attr = ctx.find_node_at_offset::<ast::Attr>();
        match (attr, ident) {
            (Some(attr), Some(ident))
                if attr.simple_name().map(|v| v.eq("derive")).unwrap_or_default() =>
            {
                Some(attempt_get_derive(attr, ident))
            }

            (Some(attr), _) => Some(WrapUnwrapOption::WrapAttr(attr)),
            _ => None,
        }
    } else {
        let covering_element = ctx.covering_element();
        match covering_element {
            NodeOrToken::Node(node) => ast::Attr::cast(node).map(WrapUnwrapOption::WrapAttr),
            NodeOrToken::Token(ident) if ident.kind() == syntax::T![ident] => {
                let attr = ident.parent_ancestors().find_map(ast::Attr::cast)?;
                Some(attempt_get_derive(attr, ident))
            }
            _ => None,
        }
    }?;
    match option {
        WrapUnwrapOption::WrapAttr(attr) if attr.simple_name().as_deref() == Some("cfg_attr") => {
            unwrap_cfg_attr(acc, attr)
        }
        WrapUnwrapOption::WrapAttr(attr) => wrap_cfg_attr(acc, ctx, attr),
        WrapUnwrapOption::WrapDerive { derive, attr } => wrap_derive(acc, ctx, attr, derive),
    }
}

fn wrap_derive(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
    attr: ast::Attr,
    derive_element: TextRange,
) -> Option<()> {
    let range = attr.syntax().text_range();
    let token_tree = attr.token_tree()?;
    let mut path_text = String::new();

    let mut cfg_derive_tokens = Vec::new();
    let mut new_derive = Vec::new();

    for tt in token_tree.token_trees_and_tokens() {
        let NodeOrToken::Token(token) = tt else {
            continue;
        };
        if token.kind() == T!['('] || token.kind() == T![')'] {
            continue;
        }

        if derive_element.contains_range(token.text_range()) {
            if token.kind() != T![,] && token.kind() != syntax::SyntaxKind::WHITESPACE {
                path_text.push_str(token.text());
                cfg_derive_tokens.push(NodeOrToken::Token(token));
            }
        } else {
            new_derive.push(NodeOrToken::Token(token));
        }
    }
    let handle_source_change = |edit: &mut SourceChangeBuilder| {
        let make = SyntaxFactory::with_mappings();
        let mut editor = edit.make_editor(attr.syntax());
        let new_derive = make.attr_outer(
            make.meta_token_tree(make.ident_path("derive"), make.token_tree(T!['('], new_derive)),
        );
        let meta = make.meta_token_tree(
            make.ident_path("cfg_attr"),
            make.token_tree(
                T!['('],
                vec![
                    NodeOrToken::Token(make.token(T![,])),
                    NodeOrToken::Token(make.whitespace(" ")),
                    NodeOrToken::Token(make.ident("derive")),
                    NodeOrToken::Node(make.token_tree(T!['('], cfg_derive_tokens)),
                ],
            ),
        );

        let cfg_attr = make.attr_outer(meta);
        editor.replace_with_many(
            attr.syntax(),
            vec![
                new_derive.syntax().clone().into(),
                make.whitespace("\n").into(),
                cfg_attr.syntax().clone().into(),
            ],
        );

        if let Some(snippet_cap) = ctx.config.snippet_cap
            && let Some(first_meta) =
                cfg_attr.meta().and_then(|meta| meta.token_tree()).and_then(|tt| tt.l_paren_token())
        {
            let tabstop = edit.make_tabstop_after(snippet_cap);
            editor.add_annotation(first_meta, tabstop);
        }

        editor.add_mappings(make.finish_with_mappings());
        edit.add_file_edits(ctx.vfs_file_id(), editor);
    };

    acc.add(
        AssistId::refactor("wrap_unwrap_cfg_attr"),
        format!("Wrap #[derive({path_text})] in `cfg_attr`",),
        range,
        handle_source_change,
    );
    Some(())
}
fn wrap_cfg_attr(acc: &mut Assists, ctx: &AssistContext<'_>, attr: ast::Attr) -> Option<()> {
    let range = attr.syntax().text_range();
    let path = attr.path()?;
    let handle_source_change = |edit: &mut SourceChangeBuilder| {
        let make = SyntaxFactory::with_mappings();
        let mut editor = edit.make_editor(attr.syntax());
        let mut raw_tokens =
            vec![NodeOrToken::Token(make.token(T![,])), NodeOrToken::Token(make.whitespace(" "))];
        path.syntax().descendants_with_tokens().for_each(|it| {
            if let NodeOrToken::Token(token) = it {
                raw_tokens.push(NodeOrToken::Token(token));
            }
        });
        if let Some(meta) = attr.meta() {
            if let (Some(eq), Some(expr)) = (meta.eq_token(), meta.expr()) {
                raw_tokens.push(NodeOrToken::Token(make.whitespace(" ")));
                raw_tokens.push(NodeOrToken::Token(eq));
                raw_tokens.push(NodeOrToken::Token(make.whitespace(" ")));

                expr.syntax().descendants_with_tokens().for_each(|it| {
                    if let NodeOrToken::Token(token) = it {
                        raw_tokens.push(NodeOrToken::Token(token));
                    }
                });
            } else if let Some(tt) = meta.token_tree() {
                raw_tokens.extend(tt.token_trees_and_tokens());
            }
        }
        let meta =
            make.meta_token_tree(make.ident_path("cfg_attr"), make.token_tree(T!['('], raw_tokens));
        let cfg_attr =
            if attr.excl_token().is_some() { make.attr_inner(meta) } else { make.attr_outer(meta) };

        editor.replace(attr.syntax(), cfg_attr.syntax());

        if let Some(snippet_cap) = ctx.config.snippet_cap
            && let Some(first_meta) =
                cfg_attr.meta().and_then(|meta| meta.token_tree()).and_then(|tt| tt.l_paren_token())
        {
            let tabstop = edit.make_tabstop_after(snippet_cap);
            editor.add_annotation(first_meta, tabstop);
        }

        editor.add_mappings(make.finish_with_mappings());
        edit.add_file_edits(ctx.vfs_file_id(), editor);
    };
    acc.add(
        AssistId::refactor("wrap_unwrap_cfg_attr"),
        "Convert to `cfg_attr`",
        range,
        handle_source_change,
    );
    Some(())
}
fn unwrap_cfg_attr(acc: &mut Assists, attr: ast::Attr) -> Option<()> {
    let range = attr.syntax().text_range();
    let meta = attr.meta()?;
    let meta_tt = meta.token_tree()?;
    let mut inner_attrs = Vec::with_capacity(1);
    let mut found_comma = false;
    let mut iter = meta_tt.token_trees_and_tokens().skip(1).peekable();
    while let Some(tt) = iter.next() {
        if let NodeOrToken::Token(token) = &tt {
            if token.kind() == T![')'] {
                break;
            }
            if token.kind() == T![,] {
                found_comma = true;
                continue;
            }
        }
        if !found_comma {
            continue;
        }
        let Some(attr_name) = tt.into_token().and_then(|token| {
            if token.kind() == T![ident] { Some(make::ext::ident_path(token.text())) } else { None }
        }) else {
            continue;
        };
        let next_tt = iter.next()?;
        let meta = match next_tt {
            NodeOrToken::Node(tt) => make::meta_token_tree(attr_name, tt),
            NodeOrToken::Token(token) if token.kind() == T![,] || token.kind() == T![')'] => {
                make::meta_path(attr_name)
            }
            NodeOrToken::Token(token) => {
                let equals = algo::skip_trivia_token(token, syntax::Direction::Next)?;
                if equals.kind() != T![=] {
                    return None;
                }
                let expr_token =
                    algo::skip_trivia_token(equals.next_token()?, syntax::Direction::Next)
                        .and_then(|it| {
                            if it.kind().is_literal() {
                                Some(make::expr_literal(it.text()))
                            } else {
                                None
                            }
                        })?;
                make::meta_expr(attr_name, ast::Expr::Literal(expr_token))
            }
        };
        if attr.excl_token().is_some() {
            inner_attrs.push(make::attr_inner(meta));
        } else {
            inner_attrs.push(make::attr_outer(meta));
        }
    }
    if inner_attrs.is_empty() {
        return None;
    }
    let handle_source_change = |f: &mut SourceChangeBuilder| {
        let inner_attrs = inner_attrs.iter().map(|it| it.to_string()).join("\n");
        f.replace(range, inner_attrs);
    };
    acc.add(
        AssistId::refactor("wrap_unwrap_cfg_attr"),
        "Extract Inner Attributes from `cfg_attr`",
        range,
        handle_source_change,
    );
    Some(())
}
#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::*;

    #[test]
    fn test_basic_to_from_cfg_attr() {
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            #[derive$0(Debug)]
            pub struct Test {
                test: u32,
            }
            "#,
            r#"
            #[cfg_attr($0, derive(Debug))]
            pub struct Test {
                test: u32,
            }
            "#,
        );
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            #[cfg_attr(debug_assertions, $0 derive(Debug))]
            pub struct Test {
                test: u32,
            }
            "#,
            r#"
            #[derive(Debug)]
            pub struct Test {
                test: u32,
            }
            "#,
        );
    }
    #[test]
    fn to_from_path_attr() {
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            pub struct Test {
                #[foo$0]
                test: u32,
            }
            "#,
            r#"
            pub struct Test {
                #[cfg_attr($0, foo)]
                test: u32,
            }
            "#,
        );
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            pub struct Test {
                #[cfg_attr(debug_assertions$0, foo)]
                test: u32,
            }
            "#,
            r#"
            pub struct Test {
                #[foo]
                test: u32,
            }
            "#,
        );
    }
    #[test]
    fn to_from_eq_attr() {
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            pub struct Test {
                #[foo = "bar"$0]
                test: u32,
            }
            "#,
            r#"
            pub struct Test {
                #[cfg_attr($0, foo = "bar")]
                test: u32,
            }
            "#,
        );
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            pub struct Test {
                #[cfg_attr(debug_assertions$0, foo = "bar")]
                test: u32,
            }
            "#,
            r#"
            pub struct Test {
                #[foo = "bar"]
                test: u32,
            }
            "#,
        );
    }
    #[test]
    fn inner_attrs() {
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            #![no_std$0]
            "#,
            r#"
            #![cfg_attr($0, no_std)]
            "#,
        );
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            #![cfg_attr(not(feature = "std")$0, no_std)]
            "#,
            r#"
            #![no_std]
            "#,
        );
    }
    #[test]
    fn test_derive_wrap() {
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            #[derive(Debug$0, Clone, Copy)]
            pub struct Test {
                test: u32,
            }
            "#,
            r#"
            #[derive( Clone, Copy)]
            #[cfg_attr($0, derive(Debug))]
            pub struct Test {
                test: u32,
            }
            "#,
        );
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            #[derive(Clone, Debug$0, Copy)]
            pub struct Test {
                test: u32,
            }
            "#,
            r#"
            #[derive(Clone,  Copy)]
            #[cfg_attr($0, derive(Debug))]
            pub struct Test {
                test: u32,
            }
            "#,
        );
    }
    #[test]
    fn test_derive_wrap_with_path() {
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            #[derive(std::fmt::Debug$0, Clone, Copy)]
            pub struct Test {
                test: u32,
            }
            "#,
            r#"
            #[derive( Clone, Copy)]
            #[cfg_attr($0, derive(std::fmt::Debug))]
            pub struct Test {
                test: u32,
            }
            "#,
        );
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            #[derive(Clone, std::fmt::Debug$0, Copy)]
            pub struct Test {
                test: u32,
            }
            "#,
            r#"
            #[derive(Clone, Copy)]
            #[cfg_attr($0, derive(std::fmt::Debug))]
            pub struct Test {
                test: u32,
            }
            "#,
        );
    }
    #[test]
    fn test_derive_wrap_at_end() {
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            #[derive(std::fmt::Debug, Clone, Cop$0y)]
            pub struct Test {
                test: u32,
            }
            "#,
            r#"
            #[derive(std::fmt::Debug, Clone)]
            #[cfg_attr($0, derive(Copy))]
            pub struct Test {
                test: u32,
            }
            "#,
        );
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            #[derive(Clone, Copy, std::fmt::D$0ebug)]
            pub struct Test {
                test: u32,
            }
            "#,
            r#"
            #[derive(Clone, Copy)]
            #[cfg_attr($0, derive(std::fmt::Debug))]
            pub struct Test {
                test: u32,
            }
            "#,
        );
    }
}
