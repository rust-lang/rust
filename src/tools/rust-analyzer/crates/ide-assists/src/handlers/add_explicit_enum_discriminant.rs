use hir::Semantics;
use ide_db::{RootDatabase, assists::AssistId, source_change::SourceChangeBuilder};
use syntax::{
    AstNode,
    ast::{self, Radix},
};

use crate::{AssistContext, Assists, utils::add_group_separators};

// Assist: add_explicit_enum_discriminant
//
// Adds explicit discriminant to all enum variants.
//
// ```
// enum TheEnum$0 {
//     Foo,
//     Bar,
//     Baz = 42,
//     Quux,
// }
// ```
// ->
// ```
// enum TheEnum {
//     Foo = 0,
//     Bar = 1,
//     Baz = 42,
//     Quux = 43,
// }
// ```
pub(crate) fn add_explicit_enum_discriminant(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let enum_node = ctx.find_node_at_offset::<ast::Enum>()?;
    let enum_def = ctx.sema.to_def(&enum_node)?;

    let is_data_carrying = enum_def.is_data_carrying(ctx.db());
    let has_primitive_repr = enum_def.repr(ctx.db()).and_then(|repr| repr.int).is_some();

    // Data carrying enums without a primitive repr have no stable discriminants.
    if is_data_carrying && !has_primitive_repr {
        return None;
    }

    let variant_list = enum_node.variant_list()?;

    // Don't offer the assist if the enum has no variants or if all variants already have an
    // explicit discriminant.
    if variant_list.variants().all(|variant_node| variant_node.expr().is_some()) {
        return None;
    }

    acc.add(
        AssistId::refactor_rewrite("add_explicit_enum_discriminant"),
        "Add explicit enum discriminants",
        enum_node.syntax().text_range(),
        |builder| {
            let mut radix = Radix::Decimal;
            for variant_node in variant_list.variants() {
                add_variant_discriminant(&ctx.sema, builder, &variant_node, &mut radix);
            }
        },
    );

    Some(())
}

fn add_variant_discriminant(
    sema: &Semantics<'_, RootDatabase>,
    builder: &mut SourceChangeBuilder,
    variant_node: &ast::Variant,
    radix: &mut Radix,
) {
    if let Some(expr) = variant_node.expr() {
        *radix = expr_radix(&expr).unwrap_or(*radix);
        return;
    }

    let Some(variant_def) = sema.to_def(variant_node) else {
        return;
    };
    let Ok(discriminant) = variant_def.eval(sema.db) else {
        return;
    };

    let variant_range = variant_node.syntax().text_range();

    let (group_size, prefix, text) = match radix {
        Radix::Binary => (4, "0b", format!("{discriminant:b}")),
        Radix::Octal => (3, "0o", format!("{discriminant:o}")),
        Radix::Decimal => (6, "", discriminant.to_string()),
        Radix::Hexadecimal => (4, "0x", format!("{discriminant:x}")),
    };
    let pretty_num = add_group_separators(&text, group_size);
    builder.insert(variant_range.end(), format!(" = {prefix}{pretty_num}"));
}

fn expr_radix(expr: &ast::Expr) -> Option<Radix> {
    if let ast::Expr::Literal(lit) = expr
        && let ast::LiteralKind::IntNumber(num) = lit.kind()
    {
        Some(num.radix())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::add_explicit_enum_discriminant;

    #[test]
    fn non_primitive_repr_non_data_bearing_add_discriminant() {
        check_assist(
            add_explicit_enum_discriminant,
            r#"
enum TheEnum$0 {
    Foo,
    Bar,
    Baz = 42,
    Quux,
    FooBar = -5,
    FooBaz,
}
"#,
            r#"
enum TheEnum {
    Foo = 0,
    Bar = 1,
    Baz = 42,
    Quux = 43,
    FooBar = -5,
    FooBaz = -4,
}
"#,
        );
    }

    #[test]
    fn primitive_repr_data_bearing_add_discriminant() {
        check_assist(
            add_explicit_enum_discriminant,
            r#"
#[repr(u8)]
$0enum TheEnum {
    Foo { x: u32 },
    Bar,
    Baz(String),
    Quux,
}
"#,
            r#"
#[repr(u8)]
enum TheEnum {
    Foo { x: u32 } = 0,
    Bar = 1,
    Baz(String) = 2,
    Quux = 3,
}
"#,
        );
    }

    #[test]
    fn non_primitive_repr_data_bearing_not_applicable() {
        check_assist_not_applicable(
            add_explicit_enum_discriminant,
            r#"
enum TheEnum$0 {
    Foo,
    Bar(u16),
    Baz,
}
"#,
        );
    }

    #[test]
    fn primitive_repr_non_data_bearing_add_discriminant() {
        check_assist(
            add_explicit_enum_discriminant,
            r#"
#[repr(i64)]
enum TheEnum {
    Foo = 1 << 63,
    Bar,
    Baz$0 = 0x7fff_ffff_ffff_fffe,
    Quux,
}
"#,
            r#"
#[repr(i64)]
enum TheEnum {
    Foo = 1 << 63,
    Bar = -9_223372_036854_775807,
    Baz = 0x7fff_ffff_ffff_fffe,
    Quux = 0x7fff_ffff_ffff_ffff,
}
"#,
        );
    }

    #[test]
    fn discriminants_already_explicit_not_applicable() {
        check_assist_not_applicable(
            add_explicit_enum_discriminant,
            r#"
enum TheEnum$0 {
    Foo = 0,
    Bar = 4,
}
"#,
        );
    }

    #[test]
    fn empty_enum_not_applicable() {
        check_assist_not_applicable(
            add_explicit_enum_discriminant,
            r#"
enum TheEnum$0 {}
"#,
        );
    }
}
