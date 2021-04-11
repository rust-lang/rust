use ide_db::{helpers::FamousDefs, RootDatabase};
use syntax::{
    ast::{self, NameOwner},
    AstNode,
};

use crate::{
    assist_context::{AssistContext, Assists},
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
    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;
    let field = ctx.find_node_at_offset::<ast::RecordField>()?;

    if existing_deref_impl(&ctx.sema, &strukt).is_some() {
        cov_mark::hit!(test_add_deref_impl_already_exists);
        return None;
    }

    let field_type = field.ty()?;
    let field_name = field.name()?;
    let target = field.syntax().text_range();
    acc.add(
        AssistId("generate_deref", AssistKind::Generate),
        format!("Generate `Deref` impl using `{}`", field_name),
        target,
        |edit| {
            let start_offset = strukt.syntax().text_range().end();
            let impl_code = format!(
                r#"    type Target = {0};

    fn deref(&self) -> &Self::Target {{
        &self.{1}
    }}"#,
                field_type.syntax(),
                field_name.syntax()
            );
            let strukt_adt = ast::Adt::Struct(strukt);
            // Q for reviewer: Is there a better way to specify the trait_text, e.g.
            // - can I have it auto `use std::ops::Deref`, and then just use `Deref` as the trait text?
            //   Or is there a helper that might detect if `std::ops::Deref` has been used, and pick `Deref`,
            //   otherwise, pick `std::ops::Deref` for the trait_text.
            let deref_impl = generate_trait_impl_text(&strukt_adt, "std::ops::Deref", &impl_code);
            edit.insert(start_offset, deref_impl);
        },
    )
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
    fn test_generate_deref() {
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

    fn check_not_applicable(ra_fixture: &str) {
        let fixture = format!(
            "//- /main.rs crate:main deps:core,std\n{}\n{}",
            ra_fixture,
            FamousDefs::FIXTURE
        );
        check_assist_not_applicable(generate_deref, &fixture)
    }

    #[test]
    fn test_generate_deref_not_applicable_if_already_impl() {
        cov_mark::check!(test_add_deref_impl_already_exists);
        check_not_applicable(
            r#"struct A { }
struct B { $0a: A }

impl std::ops::Deref for B {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.a
    }
}"#,
        )
    }
}
