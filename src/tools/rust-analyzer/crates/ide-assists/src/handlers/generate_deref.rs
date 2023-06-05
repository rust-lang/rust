use std::fmt::Display;

use hir::{ModPath, ModuleDef};
use ide_db::{famous_defs::FamousDefs, RootDatabase};
use syntax::{
    ast::{self, HasName},
    AstNode, SyntaxNode,
};

use crate::{
    assist_context::{AssistContext, Assists, SourceChangeBuilder},
    utils::generate_trait_impl_text,
    AssistId, AssistKind,
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
    let trait_ = deref_type_to_generate.to_trait(&ctx.sema, module.krate())?;
    let trait_path =
        module.find_use_path(ctx.db(), ModuleDef::Trait(trait_), ctx.config.prefer_no_std)?;

    let field_type = field.ty()?;
    let field_name = field.name()?;
    let target = field.syntax().text_range();
    acc.add(
        AssistId("generate_deref", AssistKind::Generate),
        format!("Generate `{deref_type_to_generate:?}` impl using `{field_name}`"),
        target,
        |edit| {
            generate_edit(
                ctx.db(),
                edit,
                strukt,
                field_type.syntax(),
                field_name.syntax(),
                deref_type_to_generate,
                trait_path,
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
    let trait_ = deref_type_to_generate.to_trait(&ctx.sema, module.krate())?;
    let trait_path =
        module.find_use_path(ctx.db(), ModuleDef::Trait(trait_), ctx.config.prefer_no_std)?;

    let field_type = field.ty()?;
    let target = field.syntax().text_range();
    acc.add(
        AssistId("generate_deref", AssistKind::Generate),
        format!("Generate `{deref_type_to_generate:?}` impl using `{field}`"),
        target,
        |edit| {
            generate_edit(
                ctx.db(),
                edit,
                strukt,
                field_type.syntax(),
                field_list_index,
                deref_type_to_generate,
                trait_path,
            )
        },
    )
}

fn generate_edit(
    db: &RootDatabase,
    edit: &mut SourceChangeBuilder,
    strukt: ast::Struct,
    field_type_syntax: &SyntaxNode,
    field_name: impl Display,
    deref_type: DerefType,
    trait_path: ModPath,
) {
    let start_offset = strukt.syntax().text_range().end();
    let impl_code = match deref_type {
        DerefType::Deref => format!(
            r#"    type Target = {field_type_syntax};

    fn deref(&self) -> &Self::Target {{
        &self.{field_name}
    }}"#,
        ),
        DerefType::DerefMut => format!(
            r#"    fn deref_mut(&mut self) -> &mut Self::Target {{
        &mut self.{field_name}
    }}"#,
        ),
    };
    let strukt_adt = ast::Adt::Struct(strukt);
    let deref_impl =
        generate_trait_impl_text(&strukt_adt, &trait_path.display(db).to_string(), &impl_code);
    edit.insert(start_offset, deref_impl);
}

fn existing_deref_impl(
    sema: &hir::Semantics<'_, RootDatabase>,
    strukt: &ast::Struct,
) -> Option<DerefType> {
    let strukt = sema.to_def(strukt)?;
    let krate = strukt.module(sema.db).krate();

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
