use crate::assist_context::{AssistContext, Assists};
use ide_db::assists::AssistId;
use syntax::{
    AstNode, SyntaxKind, T,
    ast::{self, HasGenericParams, HasName, HasVisibility, edit_in_place::Indent, make},
    syntax_editor::{Position, SyntaxEditor},
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
// trait ${0:NewTrait}<const N: usize> {
//     // Used as an associated constant.
//     const CONST_ASSOC: usize = N * 4;
//
//     fn create() -> Option<()>;
//
//     const_maker! {i32, 7}
// }
//
// impl<const N: usize> ${0:NewTrait}<N> for Foo<N> {
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

    let impl_assoc_items = impl_ast.assoc_item_list()?;
    let first_element = impl_assoc_items.assoc_items().next();
    first_element.as_ref()?;

    let impl_name = impl_ast.self_ty()?;

    acc.add(
        AssistId::generate("generate_trait_from_impl"),
        "Generate trait from impl",
        impl_ast.syntax().text_range(),
        |builder| {
            let trait_items: ast::AssocItemList = {
                let trait_items = impl_assoc_items.clone_subtree();
                let mut trait_items_editor = SyntaxEditor::new(trait_items.syntax().clone());

                trait_items.assoc_items().for_each(|item| {
                    strip_body(&mut trait_items_editor, &item);
                    remove_items_visibility(&mut trait_items_editor, &item);
                });
                ast::AssocItemList::cast(trait_items_editor.finish().new_root().clone()).unwrap()
            };
            let trait_ast = make::trait_(
                false,
                "NewTrait",
                impl_ast.generic_param_list(),
                impl_ast.where_clause(),
                trait_items,
            )
            .clone_for_update();

            let trait_name = trait_ast.name().expect("new trait should have a name");
            let trait_name_ref = make::name_ref(&trait_name.to_string()).clone_for_update();

            // Change `impl Foo` to `impl NewTrait for Foo`
            let mut elements = vec![
                trait_name_ref.syntax().clone().into(),
                make::tokens::single_space().into(),
                make::token(T![for]).into(),
                make::tokens::single_space().into(),
            ];

            if let Some(params) = impl_ast.generic_param_list() {
                let gen_args = &params.to_generic_args().clone_for_update();
                elements.insert(1, gen_args.syntax().clone().into());
            }

            let mut editor = builder.make_editor(impl_ast.syntax());
            impl_assoc_items.assoc_items().for_each(|item| {
                remove_items_visibility(&mut editor, &item);
            });

            editor.insert_all(Position::before(impl_name.syntax()), elements);

            // Insert trait before TraitImpl
            editor.insert_all(
                Position::before(impl_ast.syntax()),
                vec![
                    trait_ast.syntax().clone().into(),
                    make::tokens::whitespace(&format!("\n\n{}", impl_ast.indent_level())).into(),
                ],
            );

            // Link the trait name & trait ref names together as a placeholder snippet group
            if let Some(cap) = ctx.config.snippet_cap {
                let placeholder = builder.make_placeholder_snippet(cap);
                editor.add_annotation(trait_name.syntax(), placeholder);
                editor.add_annotation(trait_name_ref.syntax(), placeholder);
            }

            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    );

    Some(())
}

/// `E0449` Trait items always share the visibility of their trait
fn remove_items_visibility(editor: &mut SyntaxEditor, item: &ast::AssocItem) {
    if let Some(has_vis) = ast::AnyHasVisibility::cast(item.syntax().clone()) {
        if let Some(vis) = has_vis.visibility()
            && let Some(token) = vis.syntax().next_sibling_or_token()
            && token.kind() == SyntaxKind::WHITESPACE
        {
            editor.delete(token);
        }
        if let Some(vis) = has_vis.visibility() {
            editor.delete(vis.syntax());
        }
    }
}

fn strip_body(editor: &mut SyntaxEditor, item: &ast::AssocItem) {
    if let ast::AssocItem::Fn(f) = item
        && let Some(body) = f.body()
    {
        // In contrast to function bodies, we want to see no ws before a semicolon.
        // So let's remove them if we see any.
        if let Some(prev) = body.syntax().prev_sibling_or_token()
            && prev.kind() == SyntaxKind::WHITESPACE
        {
            editor.delete(prev);
        }

        editor.replace(body.syntax(), make::tokens::semicolon());
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

trait ${0:NewTrait}<const N: usize> {
    // Used as an associated constant.
    const CONST: usize = N * 4;
}

impl<const N: usize> ${0:NewTrait}<N> for Foo<N> {
    // Used as an associated constant.
    const CONST: usize = N * 4;
}
            "#,
        )
    }
}
