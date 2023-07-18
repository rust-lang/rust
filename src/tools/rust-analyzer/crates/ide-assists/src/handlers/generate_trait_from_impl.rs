use crate::assist_context::{AssistContext, Assists};
use ide_db::assists::AssistId;
use syntax::{
    ast::{self, edit::IndentLevel, make, HasGenericParams, HasVisibility},
    ted, AstNode, SyntaxKind,
};

// NOTES :
// We generate erroneous code if a function is declared const (E0379)
// This is left to the user to correct as our only option is to remove the
// function completely which we should not be doing.

// Assist: generate_trait_from_impl
//
// Generate trait for an already defined inherent impl and convert impl to a trait impl.
//
// ```
// struct Foo<const N: usize>([i32; N]);
//
// macro_rules! const_maker {
//     ($t:ty, $v:tt) => {
//         const CONST: $t = $v;
//     };
// }
//
// impl<const N: usize> Fo$0o<N> {
//     // Used as an associated constant.
//     const CONST_ASSOC: usize = N * 4;
//
//     fn create() -> Option<()> {
//         Some(())
//     }
//
//     const_maker! {i32, 7}
// }
// ```
// ->
// ```
// struct Foo<const N: usize>([i32; N]);
//
// macro_rules! const_maker {
//     ($t:ty, $v:tt) => {
//         const CONST: $t = $v;
//     };
// }
//
// trait ${0:TraitName}<const N: usize> {
//     // Used as an associated constant.
//     const CONST_ASSOC: usize = N * 4;
//
//     fn create() -> Option<()>;
//
//     const_maker! {i32, 7}
// }
//
// impl<const N: usize> ${0:TraitName}<N> for Foo<N> {
//     // Used as an associated constant.
//     const CONST_ASSOC: usize = N * 4;
//
//     fn create() -> Option<()> {
//         Some(())
//     }
//
//     const_maker! {i32, 7}
// }
// ```
pub(crate) fn generate_trait_from_impl(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    // Get AST Node
    let impl_ast = ctx.find_node_at_offset::<ast::Impl>()?;

    // Check if cursor is to the left of assoc item list's L_CURLY.
    // if no L_CURLY then return.
    let l_curly = impl_ast.assoc_item_list()?.l_curly_token()?;

    let cursor_offset = ctx.offset();
    let l_curly_offset = l_curly.text_range();
    if cursor_offset >= l_curly_offset.start() {
        return None;
    }

    // If impl is not inherent then we don't really need to go any further.
    if impl_ast.for_token().is_some() {
        return None;
    }

    let assoc_items = impl_ast.assoc_item_list()?;
    let first_element = assoc_items.assoc_items().next();
    if first_element.is_none() {
        // No reason for an assist.
        return None;
    }

    let impl_name = impl_ast.self_ty()?;

    acc.add(
        AssistId("generate_trait_from_impl", ide_db::assists::AssistKind::Generate),
        "Generate trait from impl",
        impl_ast.syntax().text_range(),
        |builder| {
            let trait_items = assoc_items.clone_for_update();
            let impl_items = assoc_items.clone_for_update();

            trait_items.assoc_items().for_each(|item| {
                strip_body(&item);
                remove_items_visibility(&item);
            });

            impl_items.assoc_items().for_each(|item| {
                remove_items_visibility(&item);
            });

            let trait_ast = make::trait_(
                false,
                "NewTrait",
                impl_ast.generic_param_list(),
                impl_ast.where_clause(),
                trait_items,
            );

            // Change `impl Foo` to `impl NewTrait for Foo`
            let arg_list = if let Some(genpars) = impl_ast.generic_param_list() {
                genpars.to_generic_args().to_string()
            } else {
                "".to_string()
            };

            if let Some(snippet_cap) = ctx.config.snippet_cap {
                builder.replace_snippet(
                    snippet_cap,
                    impl_name.syntax().text_range(),
                    format!("${{0:TraitName}}{} for {}", arg_list, impl_name.to_string()),
                );

                // Insert trait before TraitImpl
                builder.insert_snippet(
                    snippet_cap,
                    impl_ast.syntax().text_range().start(),
                    format!(
                        "{}\n\n{}",
                        trait_ast.to_string().replace("NewTrait", "${0:TraitName}"),
                        IndentLevel::from_node(impl_ast.syntax())
                    ),
                );
            } else {
                builder.replace(
                    impl_name.syntax().text_range(),
                    format!("NewTrait{} for {}", arg_list, impl_name.to_string()),
                );

                // Insert trait before TraitImpl
                builder.insert(
                    impl_ast.syntax().text_range().start(),
                    format!(
                        "{}\n\n{}",
                        trait_ast.to_string(),
                        IndentLevel::from_node(impl_ast.syntax())
                    ),
                );
            }

            builder.replace(assoc_items.syntax().text_range(), impl_items.to_string());
        },
    );

    Some(())
}

/// `E0449` Trait items always share the visibility of their trait
fn remove_items_visibility(item: &ast::AssocItem) {
    match item {
        ast::AssocItem::Const(c) => {
            if let Some(vis) = c.visibility() {
                ted::remove(vis.syntax());
            }
        }
        ast::AssocItem::Fn(f) => {
            if let Some(vis) = f.visibility() {
                ted::remove(vis.syntax());
            }
        }
        ast::AssocItem::TypeAlias(t) => {
            if let Some(vis) = t.visibility() {
                ted::remove(vis.syntax());
            }
        }
        _ => (),
    }
}

fn strip_body(item: &ast::AssocItem) {
    match item {
        ast::AssocItem::Fn(f) => {
            if let Some(body) = f.body() {
                // In constrast to function bodies, we want to see no ws before a semicolon.
                // So let's remove them if we see any.
                if let Some(prev) = body.syntax().prev_sibling_or_token() {
                    if prev.kind() == SyntaxKind::WHITESPACE {
                        ted::remove(prev);
                    }
                }

                ted::replace(body.syntax(), make::tokens::semicolon());
            }
        }
        _ => (),
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_no_snippet_cap, check_assist_not_applicable};

    #[test]
    fn test_trigger_when_cursor_on_header() {
        check_assist_not_applicable(
            generate_trait_from_impl,
            r#"
struct Foo(f64);

impl Foo { $0
    fn add(&mut self, x: f64) {
        self.0 += x;
    }
}"#,
        );
    }

    #[test]
    fn test_assoc_item_fn() {
        check_assist_no_snippet_cap(
            generate_trait_from_impl,
            r#"
struct Foo(f64);

impl F$0oo {
    fn add(&mut self, x: f64) {
        self.0 += x;
    }
}"#,
            r#"
struct Foo(f64);

trait NewTrait {
    fn add(&mut self, x: f64);
}

impl NewTrait for Foo {
    fn add(&mut self, x: f64) {
        self.0 += x;
    }
}"#,
        )
    }

    #[test]
    fn test_assoc_item_macro() {
        check_assist_no_snippet_cap(
            generate_trait_from_impl,
            r#"
struct Foo;

macro_rules! const_maker {
    ($t:ty, $v:tt) => {
        const CONST: $t = $v;
    };
}

impl F$0oo {
    const_maker! {i32, 7}
}"#,
            r#"
struct Foo;

macro_rules! const_maker {
    ($t:ty, $v:tt) => {
        const CONST: $t = $v;
    };
}

trait NewTrait {
    const_maker! {i32, 7}
}

impl NewTrait for Foo {
    const_maker! {i32, 7}
}"#,
        )
    }

    #[test]
    fn test_assoc_item_const() {
        check_assist_no_snippet_cap(
            generate_trait_from_impl,
            r#"
struct Foo;

impl F$0oo {
    const ABC: i32 = 3;
}"#,
            r#"
struct Foo;

trait NewTrait {
    const ABC: i32 = 3;
}

impl NewTrait for Foo {
    const ABC: i32 = 3;
}"#,
        )
    }

    #[test]
    fn test_impl_with_generics() {
        check_assist_no_snippet_cap(
            generate_trait_from_impl,
            r#"
struct Foo<const N: usize>([i32; N]);

impl<const N: usize> F$0oo<N> {
    // Used as an associated constant.
    const CONST: usize = N * 4;
}
            "#,
            r#"
struct Foo<const N: usize>([i32; N]);

trait NewTrait<const N: usize> {
    // Used as an associated constant.
    const CONST: usize = N * 4;
}

impl<const N: usize> NewTrait<N> for Foo<N> {
    // Used as an associated constant.
    const CONST: usize = N * 4;
}
            "#,
        )
    }

    #[test]
    fn test_trait_items_should_not_have_vis() {
        check_assist_no_snippet_cap(
            generate_trait_from_impl,
            r#"
struct Foo;

impl F$0oo {
    pub fn a_func() -> Option<()> {
        Some(())
    }
}"#,
            r#"
struct Foo;

trait NewTrait {
     fn a_func() -> Option<()>;
}

impl NewTrait for Foo {
     fn a_func() -> Option<()> {
        Some(())
    }
}"#,
        )
    }

    #[test]
    fn test_empty_inherent_impl() {
        check_assist_not_applicable(
            generate_trait_from_impl,
            r#"
impl Emp$0tyImpl{}
"#,
        )
    }

    #[test]
    fn test_not_top_level_impl() {
        check_assist_no_snippet_cap(
            generate_trait_from_impl,
            r#"
mod a {
    impl S$0 {
        fn foo() {}
    }
}"#,
            r#"
mod a {
    trait NewTrait {
        fn foo();
    }

    impl NewTrait for S {
        fn foo() {}
    }
}"#,
        )
    }

    #[test]
    fn test_snippet_cap_is_some() {
        check_assist(
            generate_trait_from_impl,
            r#"
struct Foo<const N: usize>([i32; N]);

impl<const N: usize> F$0oo<N> {
    // Used as an associated constant.
    const CONST: usize = N * 4;
}
            "#,
            r#"
struct Foo<const N: usize>([i32; N]);

trait ${0:TraitName}<const N: usize> {
    // Used as an associated constant.
    const CONST: usize = N * 4;
}

impl<const N: usize> ${0:TraitName}<N> for Foo<N> {
    // Used as an associated constant.
    const CONST: usize = N * 4;
}
            "#,
        )
    }
}
