use hir::{AsName, EnumVariant, Module, ModuleDef, Name};
use ide_db::{defs::Definition, search::Reference, RootDatabase};
use rustc_hash::{FxHashMap, FxHashSet};
use syntax::{
    algo::find_node_at_offset,
    algo::SyntaxRewriter,
    ast::{self, edit::IndentLevel, make, ArgListOwner, AstNode, NameOwner, VisibilityOwner},
    SourceFile, SyntaxElement,
};

use crate::{
    utils::{insert_use, mod_path_to_ast, ImportScope},
    AssistContext, AssistId, AssistKind, Assists,
};

// Assist: extract_struct_from_enum_variant
//
// Extracts a struct from enum variant.
//
// ```
// enum A { <|>One(u32, u32) }
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

    fn is_applicable_variant(variant: &ast::Variant) -> bool {
        1 < match variant.kind() {
            ast::StructKind::Record(field_list) => field_list.fields().count(),
            ast::StructKind::Tuple(field_list) => field_list.fields().count(),
            ast::StructKind::Unit => 0,
        }
    }

    if !is_applicable_variant(&variant) {
        return None;
    }

    let field_list = match variant.kind() {
        ast::StructKind::Tuple(field_list) => field_list,
        _ => return None,
    };

    let variant_name = variant.name()?;
    let variant_hir = ctx.sema.to_def(&variant)?;
    if existing_definition(ctx.db(), &variant_name, &variant_hir) {
        return None;
    }
    let enum_ast = variant.parent_enum();
    let visibility = enum_ast.visibility();
    let enum_hir = ctx.sema.to_def(&enum_ast)?;
    let variant_hir_name = variant_hir.name(ctx.db());
    let enum_module_def = ModuleDef::from(enum_hir);
    let current_module = enum_hir.module(ctx.db());
    let target = variant.syntax().text_range();
    acc.add(
        AssistId("extract_struct_from_enum_variant", AssistKind::RefactorRewrite),
        "Extract struct from enum variant",
        target,
        |builder| {
            let definition = Definition::ModuleDef(ModuleDef::EnumVariant(variant_hir));
            let res = definition.usages(&ctx.sema).all();

            let mut visited_modules_set = FxHashSet::default();
            visited_modules_set.insert(current_module);
            let mut rewriters = FxHashMap::default();
            for reference in res {
                let rewriter = rewriters
                    .entry(reference.file_range.file_id)
                    .or_insert_with(SyntaxRewriter::default);
                let source_file = ctx.sema.parse(reference.file_range.file_id);
                update_reference(
                    ctx,
                    rewriter,
                    reference,
                    &source_file,
                    &enum_module_def,
                    &variant_hir_name,
                    &mut visited_modules_set,
                );
            }
            let mut rewriter =
                rewriters.remove(&ctx.frange.file_id).unwrap_or_else(SyntaxRewriter::default);
            for (file_id, rewriter) in rewriters {
                builder.edit_file(file_id);
                builder.rewrite(rewriter);
            }
            builder.edit_file(ctx.frange.file_id);
            update_variant(&mut rewriter, &variant_name, &field_list);
            extract_struct_def(
                &mut rewriter,
                &enum_ast,
                variant_name.clone(),
                &field_list,
                &variant.parent_enum().syntax().clone().into(),
                visibility,
            );
            builder.rewrite(rewriter);
        },
    )
}

fn existing_definition(db: &RootDatabase, variant_name: &ast::Name, variant: &EnumVariant) -> bool {
    variant
        .parent_enum(db)
        .module(db)
        .scope(db, None)
        .into_iter()
        .filter(|(_, def)| match def {
            // only check type-namespace
            hir::ScopeDef::ModuleDef(def) => matches!(def,
                ModuleDef::Module(_) | ModuleDef::Adt(_) |
                ModuleDef::EnumVariant(_) | ModuleDef::Trait(_) |
                ModuleDef::TypeAlias(_) | ModuleDef::BuiltinType(_)
            ),
            _ => false,
        })
        .any(|(name, _)| name == variant_name.as_name())
}

fn insert_import(
    ctx: &AssistContext,
    rewriter: &mut SyntaxRewriter,
    path: &ast::PathExpr,
    module: &Module,
    enum_module_def: &ModuleDef,
    variant_hir_name: &Name,
) -> Option<()> {
    let db = ctx.db();
    let mod_path = module.find_use_path(db, enum_module_def.clone());
    if let Some(mut mod_path) = mod_path {
        mod_path.segments.pop();
        mod_path.segments.push(variant_hir_name.clone());
        let scope = ImportScope::find_insert_use_container(path.syntax(), ctx)?;

        *rewriter += insert_use(&scope, mod_path_to_ast(&mod_path), ctx.config.insert_use.merge);
    }
    Some(())
}

fn extract_struct_def(
    rewriter: &mut SyntaxRewriter,
    enum_: &ast::Enum,
    variant_name: ast::Name,
    variant_list: &ast::TupleFieldList,
    start_offset: &SyntaxElement,
    visibility: Option<ast::Visibility>,
) -> Option<()> {
    let variant_list = make::tuple_field_list(
        variant_list
            .fields()
            .flat_map(|field| Some(make::tuple_field(Some(make::visibility_pub()), field.ty()?))),
    );

    rewriter.insert_before(
        start_offset,
        make::struct_(visibility, variant_name, None, variant_list.into()).syntax(),
    );
    rewriter.insert_before(start_offset, &make::tokens::blank_line());

    if let indent_level @ 1..=usize::MAX = IndentLevel::from_node(enum_.syntax()).0 as usize {
        rewriter
            .insert_before(start_offset, &make::tokens::whitespace(&" ".repeat(4 * indent_level)));
    }
    Some(())
}

fn update_variant(
    rewriter: &mut SyntaxRewriter,
    variant_name: &ast::Name,
    field_list: &ast::TupleFieldList,
) -> Option<()> {
    let (l, r): (SyntaxElement, SyntaxElement) =
        (field_list.l_paren_token()?.into(), field_list.r_paren_token()?.into());
    let replacement = vec![l, variant_name.syntax().clone().into(), r];
    rewriter.replace_with_many(field_list.syntax(), replacement);
    Some(())
}

fn update_reference(
    ctx: &AssistContext,
    rewriter: &mut SyntaxRewriter,
    reference: Reference,
    source_file: &SourceFile,
    enum_module_def: &ModuleDef,
    variant_hir_name: &Name,
    visited_modules_set: &mut FxHashSet<Module>,
) -> Option<()> {
    let path_expr: ast::PathExpr = find_node_at_offset::<ast::PathExpr>(
        source_file.syntax(),
        reference.file_range.range.start(),
    )?;
    let call = path_expr.syntax().parent().and_then(ast::CallExpr::cast)?;
    let list = call.arg_list()?;
    let segment = path_expr.path()?.segment()?;
    let module = ctx.sema.scope(&path_expr.syntax()).module()?;
    if !visited_modules_set.contains(&module) {
        if insert_import(ctx, rewriter, &path_expr, &module, enum_module_def, variant_hir_name)
            .is_some()
        {
            visited_modules_set.insert(module);
        }
    }

    let lparen = syntax::SyntaxElement::from(list.l_paren_token()?);
    let rparen = syntax::SyntaxElement::from(list.r_paren_token()?);
    rewriter.insert_after(&lparen, segment.syntax());
    rewriter.insert_after(&lparen, &lparen);
    rewriter.insert_before(&rparen, &rparen);
    Some(())
}

#[cfg(test)]
mod tests {
    use crate::{
        tests::{check_assist, check_assist_not_applicable},
        utils::FamousDefs,
    };

    use super::*;

    #[test]
    fn test_extract_struct_several_fields_tuple() {
        check_assist(
            extract_struct_from_enum_variant,
            "enum A { <|>One(u32, u32) }",
            r#"struct One(pub u32, pub u32);

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_several_fields_named() {
        check_assist(
            extract_struct_from_enum_variant,
            "enum A { <|>One { foo: u32, bar: u32 } }",
            r#"struct One {
    pub foo: u32,
    pub bar: u32
}

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_enum_variant_name_value_namespace() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"const One: () = ();
enum A { <|>One(u32, u32) }"#,
            r#"const One: () = ();
struct One(pub u32, pub u32);

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_pub_visibility() {
        check_assist(
            extract_struct_from_enum_variant,
            "pub enum A { <|>One(u32, u32) }",
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
            format!("//- /main.rs crate:main deps:core\n{}\n{}", ra_fixture, FamousDefs::FIXTURE);
        check_assist_not_applicable(extract_struct_from_enum_variant, &fixture)
    }

    #[test]
    fn test_extract_enum_not_applicable_for_element_with_no_fields() {
        check_not_applicable("enum A { <|>One }");
    }

    #[test]
    fn test_extract_enum_not_applicable_if_struct_exists() {
        check_not_applicable(
            r#"struct One;
        enum A { <|>One(u8, u32) }"#,
        );
    }

    #[test]
    fn test_extract_not_applicable_one_field() {
        check_not_applicable(r"enum A { <|>One(u32) }");
    }
}
