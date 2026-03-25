use hir::{ModPath, ModuleDef};
use ide_db::{FileId, RootDatabase, famous_defs::FamousDefs};
use syntax::{
    Edition,
    ast::{self, AstNode, HasName, edit::AstNodeEdit, syntax_factory::SyntaxFactory},
    syntax_editor::Position,
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists, SourceChangeBuilder},
    utils::generate_trait_impl_intransitive_with_item,
};

// Assist: generate_deref
//
// Generate `Deref` impl using the given struct field.
//
// ```
// # //- minicore: deref, deref_mut
// struct A;
// struct B {
//    $0a: A
// }
// ```
// ->
// ```
// struct A;
// struct B {
//    a: A
// }
//
// impl core::ops::Deref for B {
//     type Target = A;
//
//     fn deref(&self) -> &Self::Target {
//         &self.a
//     }
// }
// ```
pub(crate) fn generate_deref(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    generate_record_deref(acc, ctx).or_else(|| generate_tuple_deref(acc, ctx))
}

fn generate_record_deref(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;
    let field = ctx.find_node_at_offset::<ast::RecordField>()?;

    let deref_type_to_generate = match existing_deref_impl(&ctx.sema, &strukt) {
        None => DerefType::Deref,
        Some(DerefType::Deref) => DerefType::DerefMut,
        Some(DerefType::DerefMut) => {
            cov_mark::hit!(test_add_record_deref_impl_already_exists);
            return None;
        }
    };

    let module = ctx.sema.to_def(&strukt)?.module(ctx.db());
    let cfg = ctx.config.find_path_config(ctx.sema.is_nightly(module.krate(ctx.db())));
    let trait_ = deref_type_to_generate.to_trait(&ctx.sema, module.krate(ctx.db()))?;
    let trait_path = module.find_path(ctx.db(), ModuleDef::Trait(trait_), cfg)?;

    let field_type = field.ty()?;
    let field_name = field.name()?;
    let target = field.syntax().text_range();
    let file_id = ctx.vfs_file_id();
    acc.add(
        AssistId::generate("generate_deref"),
        format!("Generate `{deref_type_to_generate:?}` impl using `{field_name}`"),
        target,
        |edit| {
            generate_edit(
                ctx.db(),
                edit,
                file_id,
                strukt,
                field_type,
                &field_name.to_string(),
                deref_type_to_generate,
                trait_path,
                module.krate(ctx.db()).edition(ctx.db()),
            )
        },
    )
}

fn generate_tuple_deref(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;
    let field = ctx.find_node_at_offset::<ast::TupleField>()?;
    let field_list = ctx.find_node_at_offset::<ast::TupleFieldList>()?;
    let field_list_index = field_list.syntax().children().position(|s| &s == field.syntax())?;

    let deref_type_to_generate = match existing_deref_impl(&ctx.sema, &strukt) {
        None => DerefType::Deref,
        Some(DerefType::Deref) => DerefType::DerefMut,
        Some(DerefType::DerefMut) => {
            cov_mark::hit!(test_add_field_deref_impl_already_exists);
            return None;
        }
    };

    let module = ctx.sema.to_def(&strukt)?.module(ctx.db());
    let cfg = ctx.config.find_path_config(ctx.sema.is_nightly(module.krate(ctx.sema.db)));
    let trait_ = deref_type_to_generate.to_trait(&ctx.sema, module.krate(ctx.db()))?;
    let trait_path = module.find_path(ctx.db(), ModuleDef::Trait(trait_), cfg)?;

    let field_type = field.ty()?;
    let target = field.syntax().text_range();
    let file_id = ctx.vfs_file_id();
    acc.add(
        AssistId::generate("generate_deref"),
        format!("Generate `{deref_type_to_generate:?}` impl using `{field}`"),
        target,
        |edit| {
            generate_edit(
                ctx.db(),
                edit,
                file_id,
                strukt,
                field_type,
                &field_list_index.to_string(),
                deref_type_to_generate,
                trait_path,
                module.krate(ctx.db()).edition(ctx.db()),
            )
        },
    )
}

fn generate_edit(
    db: &RootDatabase,
    edit: &mut SourceChangeBuilder,
    file_id: FileId,
    strukt: ast::Struct,
    field_type: ast::Type,
    field_name: &str,
    deref_type: DerefType,
    trait_path: ModPath,
    edition: Edition,
) {
    let make = SyntaxFactory::with_mappings();
    let strukt_adt = ast::Adt::Struct(strukt.clone());
    let trait_ty = make.ty(&trait_path.display(db, edition).to_string());

    let assoc_items: Vec<ast::AssocItem> = match deref_type {
        DerefType::Deref => {
            let target_alias =
                make.ty_alias([], "Target", None, None, None, Some((field_type, None)));
            let ret_ty =
                make.ty_ref(make.ty_path(make.path_from_text("Self::Target")).into(), false);
            let field_expr = make.expr_field(make.expr_path(make.ident_path("self")), field_name);
            let body = make.block_expr([], Some(make.expr_ref(field_expr.into(), false)));
            let fn_ = make
                .fn_(
                    [],
                    None,
                    make.name("deref"),
                    None,
                    None,
                    make.param_list(Some(make.self_param()), []),
                    body,
                    Some(make.ret_type(ret_ty)),
                    false,
                    false,
                    false,
                    false,
                )
                .indent(1.into());
            vec![ast::AssocItem::TypeAlias(target_alias), ast::AssocItem::Fn(fn_)]
        }
        DerefType::DerefMut => {
            let ret_ty =
                make.ty_ref(make.ty_path(make.path_from_text("Self::Target")).into(), true);
            let field_expr = make.expr_field(make.expr_path(make.ident_path("self")), field_name);
            let body = make.block_expr([], Some(make.expr_ref(field_expr.into(), true)));
            let fn_ = make
                .fn_(
                    [],
                    None,
                    make.name("deref_mut"),
                    None,
                    None,
                    make.param_list(Some(make.mut_self_param()), []),
                    body,
                    Some(make.ret_type(ret_ty)),
                    false,
                    false,
                    false,
                    false,
                )
                .indent(1.into());
            vec![ast::AssocItem::Fn(fn_)]
        }
    };

    let body = make.assoc_item_list(assoc_items);
    let indent = strukt.indent_level();
    let impl_ = generate_trait_impl_intransitive_with_item(&make, &strukt_adt, trait_ty, body)
        .indent(indent);

    let mut editor = edit.make_editor(strukt.syntax());
    editor.insert_all(
        Position::after(strukt.syntax()),
        vec![make.whitespace(&format!("\n\n{indent}")).into(), impl_.syntax().clone().into()],
    );
    editor.add_mappings(make.finish_with_mappings());
    edit.add_file_edits(file_id, editor);
}

fn existing_deref_impl(
    sema: &hir::Semantics<'_, RootDatabase>,
    strukt: &ast::Struct,
) -> Option<DerefType> {
    let strukt = sema.to_def(strukt)?;
    let krate = strukt.module(sema.db).krate(sema.db);

    let deref_trait = FamousDefs(sema, krate).core_ops_Deref()?;
    let deref_mut_trait = FamousDefs(sema, krate).core_ops_DerefMut()?;
    let strukt_type = strukt.ty(sema.db);

    if strukt_type.impls_trait(sema.db, deref_trait, &[]) {
        if strukt_type.impls_trait(sema.db, deref_mut_trait, &[]) {
            Some(DerefType::DerefMut)
        } else {
            Some(DerefType::Deref)
        }
    } else {
        None
    }
}

#[derive(Debug)]
enum DerefType {
    Deref,
    DerefMut,
}

impl DerefType {
    fn to_trait(
        &self,
        sema: &hir::Semantics<'_, RootDatabase>,
        krate: hir::Crate,
    ) -> Option<hir::Trait> {
        match self {
            DerefType::Deref => FamousDefs(sema, krate).core_ops_Deref(),
            DerefType::DerefMut => FamousDefs(sema, krate).core_ops_DerefMut(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_generate_record_deref() {
        check_assist(
            generate_deref,
            r#"
//- minicore: deref
struct A { }
struct B { $0a: A }"#,
            r#"
struct A { }
struct B { a: A }

impl core::ops::Deref for B {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.a
    }
}"#,
        );
    }

    #[test]
    fn test_generate_record_deref_with_generic() {
        check_assist(
            generate_deref,
            r#"
//- minicore: deref
struct A<T>($0T);
"#,
            r#"
struct A<T>(T);

impl<T> core::ops::Deref for A<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_record_deref_short_path() {
        check_assist(
            generate_deref,
            r#"
//- minicore: deref
use core::ops::Deref;
struct A { }
struct B { $0a: A }"#,
            r#"
use core::ops::Deref;
struct A { }
struct B { a: A }

impl Deref for B {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.a
    }
}"#,
        );
    }

    #[test]
    fn test_generate_field_deref_idx_0() {
        check_assist(
            generate_deref,
            r#"
//- minicore: deref
struct A { }
struct B($0A);"#,
            r#"
struct A { }
struct B(A);

impl core::ops::Deref for B {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}"#,
        );
    }
    #[test]
    fn test_generate_field_deref_idx_1() {
        check_assist(
            generate_deref,
            r#"
//- minicore: deref
struct A { }
struct B(u8, $0A);"#,
            r#"
struct A { }
struct B(u8, A);

impl core::ops::Deref for B {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.1
    }
}"#,
        );
    }

    #[test]
    fn test_generates_derefmut_when_deref_present() {
        check_assist(
            generate_deref,
            r#"
//- minicore: deref, deref_mut
struct B { $0a: u8 }

impl core::ops::Deref for B {}
"#,
            r#"
struct B { a: u8 }

impl core::ops::DerefMut for B {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.a
    }
}

impl core::ops::Deref for B {}
"#,
        );
    }

    #[test]
    fn test_generate_record_deref_not_applicable_if_already_impl() {
        cov_mark::check!(test_add_record_deref_impl_already_exists);
        check_assist_not_applicable(
            generate_deref,
            r#"
//- minicore: deref, deref_mut
struct A { }
struct B { $0a: A }

impl core::ops::Deref for B {}
impl core::ops::DerefMut for B {}
"#,
        )
    }

    #[test]
    fn test_generate_field_deref_not_applicable_if_already_impl() {
        cov_mark::check!(test_add_field_deref_impl_already_exists);
        check_assist_not_applicable(
            generate_deref,
            r#"
//- minicore: deref, deref_mut
struct A { }
struct B($0A)

impl core::ops::Deref for B {}
impl core::ops::DerefMut for B {}
"#,
        )
    }
}
