use std::fmt::Display;

use ide_db::{famous_defs::FamousDefs, RootDatabase};
use syntax::{
    ast::{self, HasName},
    AstNode, SyntaxNode,
};

use crate::{
    assist_context::{AssistBuilder, AssistContext, Assists},
    utils::generate_trait_impl_text,
    AssistId, AssistKind,
};

// Assist: generate_deref
//
// Generate `Deref` impl using the given struct field.
//
// ```
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
// impl std::ops::Deref for B {
//     type Target = A;
//
//     fn deref(&self) -> &Self::Target {
//         &self.a
//     }
// }
// ```
pub(crate) fn generate_deref(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    generate_record_deref(acc, ctx).or_else(|| generate_tuple_deref(acc, ctx))
}

fn generate_record_deref(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;
    let field = ctx.find_node_at_offset::<ast::RecordField>()?;

    if existing_deref_impl(&ctx.sema, &strukt).is_some() {
        cov_mark::hit!(test_add_record_deref_impl_already_exists);
        return None;
    }

    let field_type = field.ty()?;
    let field_name = field.name()?;
    let target = field.syntax().text_range();
    acc.add(
        AssistId("generate_deref", AssistKind::Generate),
        format!("Generate `Deref` impl using `{}`", field_name),
        target,
        |edit| generate_edit(edit, strukt, field_type.syntax(), field_name.syntax()),
    )
}

fn generate_tuple_deref(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;
    let field = ctx.find_node_at_offset::<ast::TupleField>()?;
    let field_list = ctx.find_node_at_offset::<ast::TupleFieldList>()?;
    let field_list_index =
        field_list.syntax().children().into_iter().position(|s| &s == field.syntax())?;

    if existing_deref_impl(&ctx.sema, &strukt).is_some() {
        cov_mark::hit!(test_add_field_deref_impl_already_exists);
        return None;
    }

    let field_type = field.ty()?;
    let target = field.syntax().text_range();
    acc.add(
        AssistId("generate_deref", AssistKind::Generate),
        format!("Generate `Deref` impl using `{}`", field.syntax()),
        target,
        |edit| generate_edit(edit, strukt, field_type.syntax(), field_list_index),
    )
}

fn generate_edit(
    edit: &mut AssistBuilder,
    strukt: ast::Struct,
    field_type_syntax: &SyntaxNode,
    field_name: impl Display,
) {
    let start_offset = strukt.syntax().text_range().end();
    let impl_code = format!(
        r#"    type Target = {0};

    fn deref(&self) -> &Self::Target {{
        &self.{1}
    }}"#,
        field_type_syntax, field_name
    );
    let strukt_adt = ast::Adt::Struct(strukt);
    let deref_impl = generate_trait_impl_text(&strukt_adt, "std::ops::Deref", &impl_code);
    edit.insert(start_offset, deref_impl);
}

fn existing_deref_impl(
    sema: &'_ hir::Semantics<'_, RootDatabase>,
    strukt: &ast::Struct,
) -> Option<()> {
    let strukt = sema.to_def(strukt)?;
    let krate = strukt.module(sema.db).krate();

    let deref_trait = FamousDefs(sema, Some(krate)).core_ops_Deref()?;
    let strukt_type = strukt.ty(sema.db);

    if strukt_type.impls_trait(sema.db, deref_trait, &[]) {
        Some(())
    } else {
        None
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
            r#"struct A { }
struct B { $0a: A }"#,
            r#"struct A { }
struct B { a: A }

impl std::ops::Deref for B {
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
            r#"struct A { }
struct B($0A);"#,
            r#"struct A { }
struct B(A);

impl std::ops::Deref for B {
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
            r#"struct A { }
struct B(u8, $0A);"#,
            r#"struct A { }
struct B(u8, A);

impl std::ops::Deref for B {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.1
    }
}"#,
        );
    }

    #[test]
    fn test_generate_record_deref_not_applicable_if_already_impl() {
        cov_mark::check!(test_add_record_deref_impl_already_exists);
        check_assist_not_applicable(
            generate_deref,
            r#"
//- minicore: deref
struct A { }
struct B { $0a: A }

impl core::ops::Deref for B {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.a
    }
}"#,
        )
    }

    #[test]
    fn test_generate_field_deref_not_applicable_if_already_impl() {
        cov_mark::check!(test_add_field_deref_impl_already_exists);
        check_assist_not_applicable(
            generate_deref,
            r#"
//- minicore: deref
struct A { }
struct B($0A)

impl core::ops::Deref for B {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}"#,
        )
    }
}
