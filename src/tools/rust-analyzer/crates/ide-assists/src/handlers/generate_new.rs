use ide_db::{
    imports::import_assets::item_for_path_search, use_trivial_constructor::use_trivial_constructor,
};
use syntax::{
    ast::{self, AstNode, HasName, HasVisibility, StructKind, edit_in_place::Indent, make},
    ted,
};

use crate::{
    AssistContext, AssistId, Assists,
    utils::{find_struct_impl, generate_impl},
};

// Assist: generate_new
//
// Adds a `fn new` for a type.
//
// ```
// struct Ctx<T: Clone> {
//      data: T,$0
// }
// ```
// ->
// ```
// struct Ctx<T: Clone> {
//      data: T,
// }
//
// impl<T: Clone> Ctx<T> {
//     fn $0new(data: T) -> Self {
//         Self { data }
//     }
// }
// ```
pub(crate) fn generate_new(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;

    // We want to only apply this to non-union structs with named fields
    let field_list = match strukt.kind() {
        StructKind::Record(named) => named,
        _ => return None,
    };

    // Return early if we've found an existing new fn
    let impl_def =
        find_struct_impl(ctx, &ast::Adt::Struct(strukt.clone()), &[String::from("new")])?;

    let current_module = ctx.sema.scope(strukt.syntax())?.module();

    let target = strukt.syntax().text_range();
    acc.add(AssistId::generate("generate_new"), "Generate `new`", target, |builder| {
        let trivial_constructors = field_list
            .fields()
            .map(|f| {
                let name = f.name()?;

                let ty = ctx.sema.resolve_type(&f.ty()?)?;

                let item_in_ns = hir::ItemInNs::from(hir::ModuleDef::from(ty.as_adt()?));

                let type_path = current_module.find_path(
                    ctx.sema.db,
                    item_for_path_search(ctx.sema.db, item_in_ns)?,
                    ctx.config.import_path_config(),
                )?;

                let edition = current_module.krate().edition(ctx.db());

                let expr = use_trivial_constructor(
                    ctx.sema.db,
                    ide_db::helpers::mod_path_to_ast(&type_path, edition),
                    &ty,
                    edition,
                )?;

                Some(make::record_expr_field(make::name_ref(&name.text()), Some(expr)))
            })
            .collect::<Vec<_>>();

        let params = field_list.fields().enumerate().filter_map(|(i, f)| {
            if trivial_constructors[i].is_none() {
                let name = f.name()?;
                let ty = f.ty()?;

                Some(make::param(make::ident_pat(false, false, name).into(), ty))
            } else {
                None
            }
        });
        let params = make::param_list(None, params);

        let fields = field_list.fields().enumerate().filter_map(|(i, f)| {
            let constructor = trivial_constructors[i].clone();
            if constructor.is_some() {
                constructor
            } else {
                Some(make::record_expr_field(make::name_ref(&f.name()?.text()), None))
            }
        });
        let fields = make::record_expr_field_list(fields);

        let record_expr = make::record_expr(make::ext::ident_path("Self"), fields);
        let body = make::block_expr(None, Some(record_expr.into()));

        let ret_type = make::ret_type(make::ty_path(make::ext::ident_path("Self")));

        let fn_ = make::fn_(
            strukt.visibility(),
            make::name("new"),
            None,
            None,
            params,
            body,
            Some(ret_type),
            false,
            false,
            false,
            false,
        )
        .clone_for_update();
        fn_.indent(1.into());

        // Add a tabstop before the name
        if let Some(cap) = ctx.config.snippet_cap {
            if let Some(name) = fn_.name() {
                builder.add_tabstop_before(cap, name);
            }
        }

        // Get the mutable version of the impl to modify
        let impl_def = if let Some(impl_def) = impl_def {
            fn_.indent(impl_def.indent_level());
            builder.make_mut(impl_def)
        } else {
            // Generate a new impl to add the method to
            let impl_def = generate_impl(&ast::Adt::Struct(strukt.clone()));
            let indent_level = strukt.indent_level();
            fn_.indent(indent_level);

            // Insert it after the adt
            let strukt = builder.make_mut(strukt.clone());

            ted::insert_all_raw(
                ted::Position::after(strukt.syntax()),
                vec![
                    make::tokens::whitespace(&format!("\n\n{indent_level}")).into(),
                    impl_def.syntax().clone().into(),
                ],
            );

            impl_def
        };

        // Add the `new` method at the start of the impl
        impl_def.get_or_create_assoc_item_list().add_item_at_start(fn_.into());
    })
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

    #[test]
    fn test_generate_new_with_zst_fields() {
        check_assist(
            generate_new,
            r#"
struct Empty;

struct Foo { empty: Empty $0}
"#,
            r#"
struct Empty;

struct Foo { empty: Empty }

impl Foo {
    fn $0new() -> Self {
        Self { empty: Empty }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Empty;

struct Foo { baz: String, empty: Empty $0}
"#,
            r#"
struct Empty;

struct Foo { baz: String, empty: Empty }

impl Foo {
    fn $0new(baz: String) -> Self {
        Self { baz, empty: Empty }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
enum Empty { Bar }

struct Foo { empty: Empty $0}
"#,
            r#"
enum Empty { Bar }

struct Foo { empty: Empty }

impl Foo {
    fn $0new() -> Self {
        Self { empty: Empty::Bar }
    }
}
"#,
        );

        // make sure the assist only works on unit variants
        check_assist(
            generate_new,
            r#"
struct Empty {}

struct Foo { empty: Empty $0}
"#,
            r#"
struct Empty {}

struct Foo { empty: Empty }

impl Foo {
    fn $0new(empty: Empty) -> Self {
        Self { empty }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
enum Empty { Bar {} }

struct Foo { empty: Empty $0}
"#,
            r#"
enum Empty { Bar {} }

struct Foo { empty: Empty }

impl Foo {
    fn $0new(empty: Empty) -> Self {
        Self { empty }
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_new() {
        check_assist(
            generate_new,
            r#"
struct Foo {$0}
"#,
            r#"
struct Foo {}

impl Foo {
    fn $0new() -> Self {
        Self {  }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Foo<T: Clone> {$0}
"#,
            r#"
struct Foo<T: Clone> {}

impl<T: Clone> Foo<T> {
    fn $0new() -> Self {
        Self {  }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Foo<'a, T: Foo<'a>> {$0}
"#,
            r#"
struct Foo<'a, T: Foo<'a>> {}

impl<'a, T: Foo<'a>> Foo<'a, T> {
    fn $0new() -> Self {
        Self {  }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Foo { baz: String $0}
"#,
            r#"
struct Foo { baz: String }

impl Foo {
    fn $0new(baz: String) -> Self {
        Self { baz }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Foo { baz: String, qux: Vec<i32> $0}
"#,
            r#"
struct Foo { baz: String, qux: Vec<i32> }

impl Foo {
    fn $0new(baz: String, qux: Vec<i32>) -> Self {
        Self { baz, qux }
    }
}
"#,
        );
    }

    #[test]
    fn check_that_visibility_modifiers_dont_get_brought_in() {
        check_assist(
            generate_new,
            r#"
struct Foo { pub baz: String, pub qux: Vec<i32> $0}
"#,
            r#"
struct Foo { pub baz: String, pub qux: Vec<i32> }

impl Foo {
    fn $0new(baz: String, qux: Vec<i32>) -> Self {
        Self { baz, qux }
    }
}
"#,
        );
    }

    #[test]
    fn check_it_reuses_existing_impls() {
        check_assist(
            generate_new,
            r#"
struct Foo {$0}

impl Foo {}
"#,
            r#"
struct Foo {}

impl Foo {
    fn $0new() -> Self {
        Self {  }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Foo {$0}

impl Foo {
    fn qux(&self) {}
}
"#,
            r#"
struct Foo {}

impl Foo {
    fn $0new() -> Self {
        Self {  }
    }

    fn qux(&self) {}
}
"#,
        );

        check_assist(
            generate_new,
            r#"
struct Foo {$0}

impl Foo {
    fn qux(&self) {}
    fn baz() -> i32 {
        5
    }
}
"#,
            r#"
struct Foo {}

impl Foo {
    fn $0new() -> Self {
        Self {  }
    }

    fn qux(&self) {}
    fn baz() -> i32 {
        5
    }
}
"#,
        );
    }

    #[test]
    fn non_zero_indent() {
        check_assist(
            generate_new,
            r#"
mod foo {
    struct $0Foo {}
}
"#,
            r#"
mod foo {
    struct Foo {}

    impl Foo {
        fn $0new() -> Self {
            Self {  }
        }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
mod foo {
    mod bar {
        struct $0Foo {}
    }
}
"#,
            r#"
mod foo {
    mod bar {
        struct Foo {}

        impl Foo {
            fn $0new() -> Self {
                Self {  }
            }
        }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
mod foo {
    struct $0Foo {}

    impl Foo {
        fn some() {}
    }
}
"#,
            r#"
mod foo {
    struct Foo {}

    impl Foo {
        fn $0new() -> Self {
            Self {  }
        }

        fn some() {}
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
mod foo {
    mod bar {
        struct $0Foo {}

        impl Foo {
            fn some() {}
        }
    }
}
"#,
            r#"
mod foo {
    mod bar {
        struct Foo {}

        impl Foo {
            fn $0new() -> Self {
                Self {  }
            }

            fn some() {}
        }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
mod foo {
    mod bar {
struct $0Foo {}

        impl Foo {
            fn some() {}
        }
    }
}
"#,
            r#"
mod foo {
    mod bar {
struct Foo {}

        impl Foo {
            fn $0new() -> Self {
                Self {  }
            }

            fn some() {}
        }
    }
}
"#,
        );
    }

    #[test]
    fn check_visibility_of_new_fn_based_on_struct() {
        check_assist(
            generate_new,
            r#"
pub struct Foo {$0}
"#,
            r#"
pub struct Foo {}

impl Foo {
    pub fn $0new() -> Self {
        Self {  }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
pub(crate) struct Foo {$0}
"#,
            r#"
pub(crate) struct Foo {}

impl Foo {
    pub(crate) fn $0new() -> Self {
        Self {  }
    }
}
"#,
        );
    }

    #[test]
    fn generate_new_not_applicable_if_fn_exists() {
        check_assist_not_applicable(
            generate_new,
            r#"
struct Foo {$0}

impl Foo {
    fn new() -> Self {
        Self
    }
}
"#,
        );

        check_assist_not_applicable(
            generate_new,
            r#"
struct Foo {$0}

impl Foo {
    fn New() -> Self {
        Self
    }
}
"#,
        );
    }

    #[test]
    fn generate_new_target() {
        check_assist_target(
            generate_new,
            r#"
struct SomeThingIrrelevant;
/// Has a lifetime parameter
struct Foo<'a, T: Foo<'a>> {$0}
struct EvenMoreIrrelevant;
"#,
            "/// Has a lifetime parameter
struct Foo<'a, T: Foo<'a>> {}",
        );
    }

    #[test]
    fn test_unrelated_new() {
        check_assist(
            generate_new,
            r#"
pub struct AstId<N: AstNode> {
    file_id: HirFileId,
    file_ast_id: FileAstId<N>,
}

impl<N: AstNode> AstId<N> {
    pub fn new(file_id: HirFileId, file_ast_id: FileAstId<N>) -> AstId<N> {
        AstId { file_id, file_ast_id }
    }
}

pub struct Source<T> {
    pub file_id: HirFileId,$0
    pub ast: T,
}

impl<T> Source<T> {
    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> Source<U> {
        Source { file_id: self.file_id, ast: f(self.ast) }
    }
}
"#,
            r#"
pub struct AstId<N: AstNode> {
    file_id: HirFileId,
    file_ast_id: FileAstId<N>,
}

impl<N: AstNode> AstId<N> {
    pub fn new(file_id: HirFileId, file_ast_id: FileAstId<N>) -> AstId<N> {
        AstId { file_id, file_ast_id }
    }
}

pub struct Source<T> {
    pub file_id: HirFileId,
    pub ast: T,
}

impl<T> Source<T> {
    pub fn $0new(file_id: HirFileId, ast: T) -> Self {
        Self { file_id, ast }
    }

    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> Source<U> {
        Source { file_id: self.file_id, ast: f(self.ast) }
    }
}
"#,
        );
    }
}
