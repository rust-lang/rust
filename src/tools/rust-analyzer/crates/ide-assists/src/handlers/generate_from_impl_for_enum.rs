use hir::next_solver::{DbInterner, TypingMode};
use ide_db::{RootDatabase, famous_defs::FamousDefs};
use syntax::ast::{self, AstNode, HasName};

use crate::{
    AssistContext, AssistId, Assists,
    utils::{generate_trait_impl_text_intransitive, is_selected},
};

// Assist: generate_from_impl_for_enum
//
// Adds a From impl for this enum variant with one tuple field.
//
// ```
// enum A { $0One(u32) }
// ```
// ->
// ```
// enum A { One(u32) }
//
// impl From<u32> for A {
//     fn from(v: u32) -> Self {
//         Self::One(v)
//     }
// }
// ```
pub(crate) fn generate_from_impl_for_enum(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let variant = ctx.find_node_at_offset::<ast::Variant>()?;
    let adt = ast::Adt::Enum(variant.parent_enum());
    let variants = selected_variants(ctx, &variant)?;

    let target = variant.syntax().text_range();
    acc.add(
        AssistId::generate("generate_from_impl_for_enum"),
        "Generate `From` impl for this enum variant(s)",
        target,
        |edit| {
            let start_offset = variant.parent_enum().syntax().text_range().end();
            let from_impl = variants
                .into_iter()
                .map(|variant_info| {
                    let from_trait = format!("From<{}>", variant_info.ty);
                    let impl_code = generate_impl_code(variant_info);
                    generate_trait_impl_text_intransitive(&adt, &from_trait, &impl_code)
                })
                .collect::<String>();
            edit.insert(start_offset, from_impl);
        },
    )
}

fn generate_impl_code(VariantInfo { name, field_name, ty }: VariantInfo) -> String {
    if let Some(field) = field_name {
        format!(
            r#"    fn from({field}: {ty}) -> Self {{
        Self::{name} {{ {field} }}
    }}"#
        )
    } else {
        format!(
            r#"    fn from(v: {ty}) -> Self {{
        Self::{name}(v)
    }}"#
        )
    }
}

struct VariantInfo {
    name: ast::Name,
    field_name: Option<ast::Name>,
    ty: ast::Type,
}

fn selected_variants(ctx: &AssistContext<'_>, variant: &ast::Variant) -> Option<Vec<VariantInfo>> {
    variant
        .parent_enum()
        .variant_list()?
        .variants()
        .filter(|it| is_selected(it, ctx.selection_trimmed(), true))
        .map(|variant| {
            let (name, ty) = extract_variant_info(&ctx.sema, &variant)?;
            Some(VariantInfo { name: variant.name()?, field_name: name, ty })
        })
        .collect()
}

fn extract_variant_info(
    sema: &'_ hir::Semantics<'_, RootDatabase>,
    variant: &ast::Variant,
) -> Option<(Option<ast::Name>, ast::Type)> {
    let (field_name, field_type) = match variant.kind() {
        ast::StructKind::Tuple(field_list) => {
            if field_list.fields().count() != 1 {
                return None;
            }
            (None, field_list.fields().next()?.ty()?)
        }
        ast::StructKind::Record(field_list) => {
            if field_list.fields().count() != 1 {
                return None;
            }
            let field = field_list.fields().next()?;
            (Some(field.name()?), field.ty()?)
        }
        ast::StructKind::Unit => return None,
    };

    if existing_from_impl(sema, variant).is_some() {
        cov_mark::hit!(test_add_from_impl_already_exists);
        return None;
    }
    Some((field_name, field_type))
}

fn existing_from_impl(
    sema: &'_ hir::Semantics<'_, RootDatabase>,
    variant: &ast::Variant,
) -> Option<()> {
    let db = sema.db;
    let variant = sema.to_def(variant)?;
    let krate = variant.module(db).krate(db);
    let from_trait = FamousDefs(sema, krate).core_convert_From()?;
    let interner = DbInterner::new_with(db, krate.base());
    use hir::next_solver::infer::DbInternerInferExt;
    let infcx = interner.infer_ctxt().build(TypingMode::non_body_analysis());

    let variant = variant.instantiate_infer(&infcx);
    let enum_ = variant.parent_enum(sema.db);
    let field_ty = variant.fields(sema.db).first()?.ty(sema.db);
    let enum_ty = enum_.ty(sema.db);
    tracing::debug!(?enum_, ?field_ty, ?enum_ty);
    enum_ty.impls_trait(infcx, from_trait, &[field_ty]).then_some(())
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_generate_from_impl_for_enum() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One(u32) }
"#,
            r#"
enum A { One(u32) }

impl From<u32> for A {
    fn from(v: u32) -> Self {
        Self::One(v)
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_from_impl_for_multiple_enum_variants() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0Foo(u32), Bar$0(i32) }
"#,
            r#"
enum A { Foo(u32), Bar(i32) }

impl From<u32> for A {
    fn from(v: u32) -> Self {
        Self::Foo(v)
    }
}

impl From<i32> for A {
    fn from(v: i32) -> Self {
        Self::Bar(v)
    }
}
"#,
        );
    }

    // FIXME(next-solver): it would be nice to not be *required* to resolve the
    // path in order to properly generate assists
    #[test]
    fn test_generate_from_impl_for_enum_complicated_path() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
mod foo { pub mod bar { pub mod baz { pub struct Boo; } } }
enum A { $0One(foo::bar::baz::Boo) }
"#,
            r#"
mod foo { pub mod bar { pub mod baz { pub struct Boo; } } }
enum A { One(foo::bar::baz::Boo) }

impl From<foo::bar::baz::Boo> for A {
    fn from(v: foo::bar::baz::Boo) -> Self {
        Self::One(v)
    }
}
"#,
        );
    }

    #[test]
    fn test_add_from_impl_no_element() {
        check_assist_not_applicable(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One }
"#,
        );
    }

    #[test]
    fn test_add_from_impl_more_than_one_element_in_tuple() {
        check_assist_not_applicable(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One(u32, String) }
"#,
        );
    }

    #[test]
    fn test_add_from_impl_struct_variant() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One { x: u32 } }
"#,
            r#"
enum A { One { x: u32 } }

impl From<u32> for A {
    fn from(x: u32) -> Self {
        Self::One { x }
    }
}
"#,
        );
    }

    #[test]
    fn test_add_from_impl_already_exists() {
        cov_mark::check!(test_add_from_impl_already_exists);
        check_assist_not_applicable(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One(u32), }

impl From<u32> for A {
    fn from(v: u32) -> Self {
        Self::One(v)
    }
}
"#,
        );
    }

    #[test]
    fn test_add_from_impl_different_variant_impl_exists() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One(u32), Two(String), }

impl From<String> for A {
    fn from(v: String) -> Self {
        A::Two(v)
    }
}

pub trait From<T> {
    fn from(T) -> Self;
}
"#,
            r#"
enum A { One(u32), Two(String), }

impl From<u32> for A {
    fn from(v: u32) -> Self {
        Self::One(v)
    }
}

impl From<String> for A {
    fn from(v: String) -> Self {
        A::Two(v)
    }
}

pub trait From<T> {
    fn from(T) -> Self;
}
"#,
        );
    }

    #[test]
    fn test_add_from_impl_static_str() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One(&'static str) }
"#,
            r#"
enum A { One(&'static str) }

impl From<&'static str> for A {
    fn from(v: &'static str) -> Self {
        Self::One(v)
    }
}
"#,
        );
    }

    #[test]
    fn test_add_from_impl_generic_enum() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum Generic<T, U: Clone> { $0One(T), Two(U) }
"#,
            r#"
enum Generic<T, U: Clone> { One(T), Two(U) }

impl<T, U: Clone> From<T> for Generic<T, U> {
    fn from(v: T) -> Self {
        Self::One(v)
    }
}
"#,
        );
    }

    #[test]
    fn test_add_from_impl_with_lifetime() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum Generic<'a> { $0One(&'a i32) }
"#,
            r#"
enum Generic<'a> { One(&'a i32) }

impl<'a> From<&'a i32> for Generic<'a> {
    fn from(v: &'a i32) -> Self {
        Self::One(v)
    }
}
"#,
        );
    }
}
