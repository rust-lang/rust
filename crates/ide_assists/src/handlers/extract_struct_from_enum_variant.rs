use std::iter;

use either::Either;
use hir::{AsName, Module, ModuleDef, Name, Variant};
use ide_db::{
    defs::Definition,
    helpers::{
        insert_use::{insert_use, ImportScope},
        mod_path_to_ast,
    },
    search::FileReference,
    RootDatabase,
};
use rustc_hash::FxHashSet;
use syntax::{
    algo::{find_node_at_offset, SyntaxRewriter},
    ast::{self, edit::IndentLevel, make, AstNode, NameOwner, VisibilityOwner},
    SourceFile, SyntaxElement, SyntaxNode, T,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: extract_struct_from_enum_variant
//
// Extracts a struct from enum variant.
//
// ```
// enum A { $0One(u32, u32) }
// ```
// ->
// ```
// struct One(pub u32, pub u32);
//
// enum A { One(One) }
// ```
pub(crate) fn extract_struct_from_enum_variant(
    acc: &mut Assists,
    ctx: &AssistContext,
) -> Option<()> {
    let variant = ctx.find_node_at_offset::<ast::Variant>()?;
    let field_list = extract_field_list_if_applicable(&variant)?;

    let variant_name = variant.name()?;
    let variant_hir = ctx.sema.to_def(&variant)?;
    if existing_definition(ctx.db(), &variant_name, &variant_hir) {
        return None;
    }

    let enum_ast = variant.parent_enum();
    let enum_hir = ctx.sema.to_def(&enum_ast)?;
    let target = variant.syntax().text_range();
    acc.add(
        AssistId("extract_struct_from_enum_variant", AssistKind::RefactorRewrite),
        "Extract struct from enum variant",
        target,
        |builder| {
            let variant_hir_name = variant_hir.name(ctx.db());
            let enum_module_def = ModuleDef::from(enum_hir);
            let usages =
                Definition::ModuleDef(ModuleDef::Variant(variant_hir)).usages(&ctx.sema).all();

            let mut visited_modules_set = FxHashSet::default();
            let current_module = enum_hir.module(ctx.db());
            visited_modules_set.insert(current_module);
            let mut def_rewriter = None;
            for (file_id, references) in usages {
                let mut rewriter = SyntaxRewriter::default();
                let source_file = ctx.sema.parse(file_id);
                for reference in references {
                    update_reference(
                        ctx,
                        &mut rewriter,
                        reference,
                        &source_file,
                        &enum_module_def,
                        &variant_hir_name,
                        &mut visited_modules_set,
                    );
                }
                if file_id == ctx.frange.file_id {
                    def_rewriter = Some(rewriter);
                    continue;
                }
                builder.edit_file(file_id);
                builder.rewrite(rewriter);
            }
            let mut rewriter = def_rewriter.unwrap_or_default();
            update_variant(&mut rewriter, &variant);
            extract_struct_def(
                &mut rewriter,
                &enum_ast,
                variant_name.clone(),
                &field_list,
                &variant.parent_enum().syntax().clone().into(),
                enum_ast.visibility(),
            );
            builder.edit_file(ctx.frange.file_id);
            builder.rewrite(rewriter);
        },
    )
}

fn extract_field_list_if_applicable(
    variant: &ast::Variant,
) -> Option<Either<ast::RecordFieldList, ast::TupleFieldList>> {
    match variant.kind() {
        ast::StructKind::Record(field_list) if field_list.fields().next().is_some() => {
            Some(Either::Left(field_list))
        }
        ast::StructKind::Tuple(field_list) if field_list.fields().count() > 1 => {
            Some(Either::Right(field_list))
        }
        _ => None,
    }
}

fn existing_definition(db: &RootDatabase, variant_name: &ast::Name, variant: &Variant) -> bool {
    variant
        .parent_enum(db)
        .module(db)
        .scope(db, None)
        .into_iter()
        .filter(|(_, def)| match def {
            // only check type-namespace
            hir::ScopeDef::ModuleDef(def) => matches!(
                def,
                ModuleDef::Module(_)
                    | ModuleDef::Adt(_)
                    | ModuleDef::Variant(_)
                    | ModuleDef::Trait(_)
                    | ModuleDef::TypeAlias(_)
                    | ModuleDef::BuiltinType(_)
            ),
            _ => false,
        })
        .any(|(name, _)| name == variant_name.as_name())
}

fn insert_import(
    ctx: &AssistContext,
    rewriter: &mut SyntaxRewriter,
    scope_node: &SyntaxNode,
    module: &Module,
    enum_module_def: &ModuleDef,
    variant_hir_name: &Name,
) -> Option<()> {
    let db = ctx.db();
    let mod_path = module.find_use_path_prefixed(
        db,
        enum_module_def.clone(),
        ctx.config.insert_use.prefix_kind,
    );
    if let Some(mut mod_path) = mod_path {
        mod_path.pop_segment();
        mod_path.push_segment(variant_hir_name.clone());
        let scope = ImportScope::find_insert_use_container(scope_node, &ctx.sema)?;
        *rewriter += insert_use(&scope, mod_path_to_ast(&mod_path), ctx.config.insert_use);
    }
    Some(())
}

fn extract_struct_def(
    rewriter: &mut SyntaxRewriter,
    enum_: &ast::Enum,
    variant_name: ast::Name,
    field_list: &Either<ast::RecordFieldList, ast::TupleFieldList>,
    start_offset: &SyntaxElement,
    visibility: Option<ast::Visibility>,
) -> Option<()> {
    let pub_vis = Some(make::visibility_pub());
    let field_list = match field_list {
        Either::Left(field_list) => {
            make::record_field_list(field_list.fields().flat_map(|field| {
                Some(make::record_field(pub_vis.clone(), field.name()?, field.ty()?))
            }))
            .into()
        }
        Either::Right(field_list) => make::tuple_field_list(
            field_list
                .fields()
                .flat_map(|field| Some(make::tuple_field(pub_vis.clone(), field.ty()?))),
        )
        .into(),
    };

    rewriter.insert_before(
        start_offset,
        make::struct_(visibility, variant_name, None, field_list).syntax(),
    );
    rewriter.insert_before(start_offset, &make::tokens::blank_line());

    if let indent_level @ 1..=usize::MAX = IndentLevel::from_node(enum_.syntax()).0 as usize {
        rewriter
            .insert_before(start_offset, &make::tokens::whitespace(&" ".repeat(4 * indent_level)));
    }
    Some(())
}

fn update_variant(rewriter: &mut SyntaxRewriter, variant: &ast::Variant) -> Option<()> {
    let name = variant.name()?;
    let tuple_field = make::tuple_field(None, make::ty(name.text()));
    let replacement = make::variant(
        name,
        Some(ast::FieldList::TupleFieldList(make::tuple_field_list(iter::once(tuple_field)))),
    );
    rewriter.replace(variant.syntax(), replacement.syntax());
    Some(())
}

fn update_reference(
    ctx: &AssistContext,
    rewriter: &mut SyntaxRewriter,
    reference: FileReference,
    source_file: &SourceFile,
    enum_module_def: &ModuleDef,
    variant_hir_name: &Name,
    visited_modules_set: &mut FxHashSet<Module>,
) -> Option<()> {
    let offset = reference.range.start();
    let (segment, expr) = if let Some(path_expr) =
        find_node_at_offset::<ast::PathExpr>(source_file.syntax(), offset)
    {
        // tuple variant
        (path_expr.path()?.segment()?, path_expr.syntax().parent()?)
    } else if let Some(record_expr) =
        find_node_at_offset::<ast::RecordExpr>(source_file.syntax(), offset)
    {
        // record variant
        (record_expr.path()?.segment()?, record_expr.syntax().clone())
    } else {
        return None;
    };

    let module = ctx.sema.scope(&expr).module()?;
    if !visited_modules_set.contains(&module) {
        if insert_import(ctx, rewriter, &expr, &module, enum_module_def, variant_hir_name).is_some()
        {
            visited_modules_set.insert(module);
        }
    }
    rewriter.insert_after(segment.syntax(), &make::token(T!['(']));
    rewriter.insert_after(segment.syntax(), segment.syntax());
    rewriter.insert_after(&expr, &make::token(T![')']));
    Some(())
}

#[cfg(test)]
mod tests {
    use ide_db::helpers::FamousDefs;

    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_extract_struct_several_fields_tuple() {
        check_assist(
            extract_struct_from_enum_variant,
            "enum A { $0One(u32, u32) }",
            r#"struct One(pub u32, pub u32);

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_several_fields_named() {
        check_assist(
            extract_struct_from_enum_variant,
            "enum A { $0One { foo: u32, bar: u32 } }",
            r#"struct One{ pub foo: u32, pub bar: u32 }

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_one_field_named() {
        check_assist(
            extract_struct_from_enum_variant,
            "enum A { $0One { foo: u32 } }",
            r#"struct One{ pub foo: u32 }

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_enum_variant_name_value_namespace() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"const One: () = ();
enum A { $0One(u32, u32) }"#,
            r#"const One: () = ();
struct One(pub u32, pub u32);

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_pub_visibility() {
        check_assist(
            extract_struct_from_enum_variant,
            "pub enum A { $0One(u32, u32) }",
            r#"pub struct One(pub u32, pub u32);

pub enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_with_complex_imports() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"mod my_mod {
    fn another_fn() {
        let m = my_other_mod::MyEnum::MyField(1, 1);
    }

    pub mod my_other_mod {
        fn another_fn() {
            let m = MyEnum::MyField(1, 1);
        }

        pub enum MyEnum {
            $0MyField(u8, u8),
        }
    }
}

fn another_fn() {
    let m = my_mod::my_other_mod::MyEnum::MyField(1, 1);
}"#,
            r#"use my_mod::my_other_mod::MyField;

mod my_mod {
    use self::my_other_mod::MyField;

    fn another_fn() {
        let m = my_other_mod::MyEnum::MyField(MyField(1, 1));
    }

    pub mod my_other_mod {
        fn another_fn() {
            let m = MyEnum::MyField(MyField(1, 1));
        }

        pub struct MyField(pub u8, pub u8);

        pub enum MyEnum {
            MyField(MyField),
        }
    }
}

fn another_fn() {
    let m = my_mod::my_other_mod::MyEnum::MyField(MyField(1, 1));
}"#,
        );
    }

    #[test]
    fn extract_record_fix_references() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum E {
    $0V { i: i32, j: i32 }
}

fn f() {
    let e = E::V { i: 9, j: 2 };
}
"#,
            r#"
struct V{ pub i: i32, pub j: i32 }

enum E {
    V(V)
}

fn f() {
    let e = E::V(V { i: 9, j: 2 });
}
"#,
        )
    }

    #[test]
    fn test_several_files() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
//- /main.rs
enum E {
    $0V(i32, i32)
}
mod foo;

//- /foo.rs
use crate::E;
fn f() {
    let e = E::V(9, 2);
}
"#,
            r#"
//- /main.rs
struct V(pub i32, pub i32);

enum E {
    V(V)
}
mod foo;

//- /foo.rs
use crate::{E, V};
fn f() {
    let e = E::V(V(9, 2));
}
"#,
        )
    }

    #[test]
    fn test_several_files_record() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
//- /main.rs
enum E {
    $0V { i: i32, j: i32 }
}
mod foo;

//- /foo.rs
use crate::E;
fn f() {
    let e = E::V { i: 9, j: 2 };
}
"#,
            r#"
//- /main.rs
struct V{ pub i: i32, pub j: i32 }

enum E {
    V(V)
}
mod foo;

//- /foo.rs
use crate::{E, V};
fn f() {
    let e = E::V(V { i: 9, j: 2 });
}
"#,
        )
    }

    #[test]
    fn test_extract_struct_record_nested_call_exp() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum A { $0One { a: u32, b: u32 } }

struct B(A);

fn foo() {
    let _ = B(A::One { a: 1, b: 2 });
}
"#,
            r#"
struct One{ pub a: u32, pub b: u32 }

enum A { One(One) }

struct B(A);

fn foo() {
    let _ = B(A::One(One { a: 1, b: 2 }));
}
"#,
        );
    }

    fn check_not_applicable(ra_fixture: &str) {
        let fixture =
            format!("//- /main.rs crate:main deps:core\n{}\n{}", ra_fixture, FamousDefs::FIXTURE);
        check_assist_not_applicable(extract_struct_from_enum_variant, &fixture)
    }

    #[test]
    fn test_extract_enum_not_applicable_for_element_with_no_fields() {
        check_not_applicable("enum A { $0One }");
    }

    #[test]
    fn test_extract_enum_not_applicable_if_struct_exists() {
        check_not_applicable(
            r#"struct One;
        enum A { $0One(u8, u32) }"#,
        );
    }

    #[test]
    fn test_extract_not_applicable_one_field() {
        check_not_applicable(r"enum A { $0One(u32) }");
    }

    #[test]
    fn test_extract_not_applicable_no_field_tuple() {
        check_not_applicable(r"enum A { $0None() }");
    }

    #[test]
    fn test_extract_not_applicable_no_field_named() {
        check_not_applicable(r"enum A { $0None {} }");
    }
}
