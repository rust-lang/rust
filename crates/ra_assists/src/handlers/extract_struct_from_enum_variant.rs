use hir_expand::name::AsName;
use ra_ide_db::{
    defs::Definition, imports_locator::ImportsLocator, search::Reference, RootDatabase,
};
use ra_syntax::{
    algo::find_node_at_offset,
    ast::{self, AstNode, NameOwner},
    SourceFile, SyntaxNode, TextRange, TextSize,
};
use stdx::format_to;

use crate::{
    assist_context::{AssistBuilder, AssistDirector},
    utils::insert_use_statement,
    AssistContext, AssistId, Assists,
};
use ast::{ArgListOwner, VisibilityOwner};
use hir::{EnumVariant, Module, ModuleDef};
use ra_db::FileId;
use ra_fmt::leading_indent;
use rustc_hash::FxHashSet;

// Assist extract_struct_from_enum
//
// Extracts a from struct from enum variant
//
// ```
// enum A { <|>One(u32, u32) }
// ```
// ->
// ```
// struct One(pub u32, pub u32);
//
// enum A { One(One) }"
// ```
pub(crate) fn extract_struct_from_enum(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let variant = ctx.find_node_at_offset::<ast::EnumVariant>()?;
    let field_list = match variant.kind() {
        ast::StructKind::Tuple(field_list) => field_list,
        _ => return None,
    };
    let variant_name = variant.name()?.to_string();
    let enum_ast = variant.parent_enum();
    let enum_name = enum_ast.name().unwrap().to_string();
    let visibility = enum_ast.visibility();
    let variant_hir = ctx.sema.to_def(&variant)?;

    if existing_struct_def(ctx.db, &variant_name, &variant_hir) {
        return None;
    }

    let target = variant.syntax().text_range();
    return acc.add_in_multiple_files(
        AssistId("extract_struct_from_enum_variant"),
        "Extract struct from enum variant",
        target,
        |edit| {
            let definition = Definition::ModuleDef(ModuleDef::EnumVariant(variant_hir));
            let res = definition.find_usages(&ctx.db, None);
            let module_def = mod_def_for_target_module(ctx, &enum_name);
            let start_offset = variant.parent_enum().syntax().text_range().start();
            let mut visited_modules_set: FxHashSet<Module> = FxHashSet::default();
            visited_modules_set.insert(module_def.module(ctx.db).unwrap());
            for reference in res {
                let source_file = ctx.sema.parse(reference.file_range.file_id);
                update_reference(
                    ctx,
                    edit,
                    reference,
                    &source_file,
                    &module_def,
                    &mut visited_modules_set,
                );
            }
            extract_struct_def(
                edit,
                enum_ast.syntax(),
                &variant_name,
                &field_list.to_string(),
                start_offset,
                ctx.frange.file_id,
                &visibility,
            );
            let list_range = field_list.syntax().text_range();
            update_variant(edit, &variant_name, ctx.frange.file_id, list_range);
        },
    );
}

fn existing_struct_def(db: &RootDatabase, variant_name: &str, variant: &EnumVariant) -> bool {
    let module_defs = variant.parent_enum(db).module(db).scope(db, None);
    for (name, _) in module_defs {
        if name.to_string() == variant_name.to_string() {
            return true;
        }
    }
    false
}

fn mod_def_for_target_module(ctx: &AssistContext, enum_name: &str) -> ModuleDef {
    ImportsLocator::new(ctx.db).find_imports(enum_name).first().unwrap().left().unwrap()
}

fn insert_import(
    ctx: &AssistContext,
    builder: &mut AssistBuilder,
    path: &ast::PathExpr,
    module: &Module,
    module_def: &ModuleDef,
    path_segment: ast::NameRef,
) -> Option<()> {
    let db = ctx.db;
    let mod_path = module.find_use_path(db, module_def.clone());
    if let Some(mut mod_path) = mod_path {
        mod_path.segments.pop();
        mod_path.segments.push(path_segment.as_name());
        insert_use_statement(path.syntax(), &mod_path, ctx, builder.text_edit_builder());
    }
    Some(())
}

fn extract_struct_def(
    edit: &mut AssistDirector,
    enum_ast: &SyntaxNode,
    variant_name: &str,
    variant_list: &str,
    start_offset: TextSize,
    file_id: FileId,
    visibility: &Option<ast::Visibility>,
) -> Option<()> {
    let visibility_string = if let Some(visibility) = visibility {
        format!("{} ", visibility.to_string())
    } else {
        "".to_string()
    };
    let mut buf = String::new();
    let indent = if let Some(indent) = leading_indent(enum_ast) {
        indent.to_string()
    } else {
        "".to_string()
    };

    format_to!(
        buf,
        r#"{}struct {}{};

{}"#,
        visibility_string,
        variant_name,
        list_with_visibility(variant_list),
        indent
    );
    edit.perform(file_id, |builder| {
        builder.insert(start_offset, buf);
    });
    Some(())
}

fn update_variant(
    edit: &mut AssistDirector,
    variant_name: &str,
    file_id: FileId,
    list_range: TextRange,
) -> Option<()> {
    let inside_variant_range = TextRange::new(
        list_range.start().checked_add(TextSize::from(1))?,
        list_range.end().checked_sub(TextSize::from(1))?,
    );
    edit.perform(file_id, |builder| {
        builder.set_file(file_id);
        builder.replace(inside_variant_range, variant_name);
    });
    Some(())
}

fn update_reference(
    ctx: &AssistContext,
    edit: &mut AssistDirector,
    reference: Reference,
    source_file: &SourceFile,
    module_def: &ModuleDef,
    visited_modules_set: &mut FxHashSet<Module>,
) -> Option<()> {
    let path_expr: ast::PathExpr = find_node_at_offset::<ast::PathExpr>(
        source_file.syntax(),
        reference.file_range.range.start(),
    )?;
    let call = path_expr.syntax().parent().and_then(ast::CallExpr::cast)?;
    let list = call.arg_list()?;
    let segment = path_expr.path()?.segment()?;
    let list_range = list.syntax().text_range();
    let inside_list_range = TextRange::new(
        list_range.start().checked_add(TextSize::from(1))?,
        list_range.end().checked_sub(TextSize::from(1))?,
    );
    edit.perform(reference.file_range.file_id, |builder| {
        let module = ctx.sema.scope(&path_expr.syntax()).module().unwrap();
        if !visited_modules_set.contains(&module) {
            if insert_import(
                ctx,
                builder,
                &path_expr,
                &module,
                module_def,
                segment.name_ref().unwrap(),
            )
            .is_some()
            {
                visited_modules_set.insert(module);
            }
        }
        builder.replace(inside_list_range, format!("{}{}", segment, list));
    });
    Some(())
}

fn list_with_visibility(list: &str) -> String {
    list.split(',')
        .map(|part| {
            let index = if part.chars().next().unwrap() == '(' { 1usize } else { 0 };
            let mut mod_part = part.trim().to_string();
            mod_part.insert_str(index, "pub ");
            mod_part
        })
        .collect::<Vec<String>>()
        .join(", ")
}

#[cfg(test)]
mod tests {

    use crate::{
        tests::{check_assist, check_assist_not_applicable},
        utils::FamousDefs,
    };

    use super::*;

    #[test]
    fn test_extract_struct_several_fields() {
        check_assist(
            extract_struct_from_enum,
            "enum A { <|>One(u32, u32) }",
            r#"struct One(pub u32, pub u32);

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_one_field() {
        check_assist(
            extract_struct_from_enum,
            "enum A { <|>One(u32) }",
            r#"struct One(pub u32);

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_pub_visibility() {
        check_assist(
            extract_struct_from_enum,
            "pub enum A { <|>One(u32, u32) }",
            r#"pub struct One(pub u32, pub u32);

pub enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_with_complex_imports() {
        check_assist(
            extract_struct_from_enum,
            r#"mod my_mod {
    fn another_fn() {
        let m = my_other_mod::MyEnum::MyField(1, 1);
    }

    pub mod my_other_mod {
        fn another_fn() {
            let m = MyEnum::MyField(1, 1);
        }

        pub enum MyEnum {
            <|>MyField(u8, u8),
        }
    }
}

fn another_fn() {
    let m = my_mod::my_other_mod::MyEnum::MyField(1, 1);
}"#,
            r#"use my_mod::my_other_mod::MyField;

mod my_mod {
    use my_other_mod::MyField;

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

    fn check_not_applicable(ra_fixture: &str) {
        let fixture =
            format!("//- main.rs crate:main deps:core\n{}\n{}", ra_fixture, FamousDefs::FIXTURE);
        check_assist_not_applicable(extract_struct_from_enum, &fixture)
    }

    #[test]
    fn test_extract_enum_not_applicable_for_element_with_no_fields() {
        check_not_applicable("enum A { <|>One }");
    }

    #[test]
    fn test_extract_enum_not_applicable_if_struct_exists() {
        check_not_applicable(
            r#"struct One;
        enum A { <|>One(u8) }"#,
        );
    }
}
