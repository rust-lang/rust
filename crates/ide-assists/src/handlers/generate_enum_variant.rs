use hir::{HasSource, InFile};
use ide_db::assists::{AssistId, AssistKind};
use syntax::{
    ast::{self, edit::IndentLevel},
    AstNode, TextSize,
};

use crate::assist_context::{AssistContext, Assists};

// Assist: generate_enum_variant
//
// Adds a variant to an enum.
//
// ```
// enum Countries {
//     Ghana,
// }
//
// fn main() {
//     let country = Countries::Lesotho$0;
// }
// ```
// ->
// ```
// enum Countries {
//     Ghana,
//     Lesotho,
// }
//
// fn main() {
//     let country = Countries::Lesotho;
// }
// ```
pub(crate) fn generate_enum_variant(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let path_expr: ast::PathExpr = ctx.find_node_at_offset()?;
    let path = path_expr.path()?;

    if ctx.sema.resolve_path(&path).is_some() {
        // No need to generate anything if the path resolves
        return None;
    }

    let name_ref = path.segment()?.name_ref()?;
    if name_ref.text().starts_with(char::is_lowercase) {
        // Don't suggest generating variant if the name starts with a lowercase letter
        return None;
    }

    if let Some(hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Enum(e)))) =
        ctx.sema.resolve_path(&path.qualifier()?)
    {
        let target = path.syntax().text_range();
        return add_variant_to_accumulator(acc, ctx, target, e, &name_ref);
    }

    None
}

fn add_variant_to_accumulator(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
    target: syntax::TextRange,
    adt: hir::Enum,
    name_ref: &ast::NameRef,
) -> Option<()> {
    let db = ctx.db();
    let InFile { file_id, value: enum_node } = adt.source(db)?.original_ast_node(db)?;
    let enum_indent = IndentLevel::from_node(&enum_node.syntax());

    let variant_list = enum_node.variant_list()?;
    let offset = variant_list.syntax().text_range().end() - TextSize::of('}');
    let empty_enum = variant_list.variants().next().is_none();

    acc.add(
        AssistId("generate_enum_variant", AssistKind::Generate),
        "Generate variant",
        target,
        |builder| {
            builder.edit_file(file_id.original_file(db));
            let text = format!(
                "{maybe_newline}{indent_1}{name},\n{enum_indent}",
                maybe_newline = if empty_enum { "\n" } else { "" },
                indent_1 = IndentLevel(1),
                name = name_ref,
                enum_indent = enum_indent
            );
            builder.insert(offset, text)
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn generate_basic_enum_variant_in_empty_enum() {
        check_assist(
            generate_enum_variant,
            r"
enum Foo {}
fn main() {
    Foo::Bar$0
}
",
            r"
enum Foo {
    Bar,
}
fn main() {
    Foo::Bar
}
",
        )
    }

    #[test]
    fn generate_basic_enum_variant_in_non_empty_enum() {
        check_assist(
            generate_enum_variant,
            r"
enum Foo {
    Bar,
}
fn main() {
    Foo::Baz$0
}
",
            r"
enum Foo {
    Bar,
    Baz,
}
fn main() {
    Foo::Baz
}
",
        )
    }

    #[test]
    fn generate_basic_enum_variant_in_different_file() {
        check_assist(
            generate_enum_variant,
            r"
//- /main.rs
mod foo;
use foo::Foo;

fn main() {
    Foo::Baz$0
}

//- /foo.rs
enum Foo {
    Bar,
}
",
            r"
enum Foo {
    Bar,
    Baz,
}
",
        )
    }

    #[test]
    fn not_applicable_for_existing_variant() {
        check_assist_not_applicable(
            generate_enum_variant,
            r"
enum Foo {
    Bar,
}
fn main() {
    Foo::Bar$0
}
",
        )
    }

    #[test]
    fn not_applicable_for_lowercase() {
        check_assist_not_applicable(
            generate_enum_variant,
            r"
enum Foo {
    Bar,
}
fn main() {
    Foo::new$0
}
",
        )
    }

    #[test]
    fn indentation_level_is_correct() {
        check_assist(
            generate_enum_variant,
            r"
mod m {
    enum Foo {
        Bar,
    }
}
fn main() {
    m::Foo::Baz$0
}
",
            r"
mod m {
    enum Foo {
        Bar,
        Baz,
    }
}
fn main() {
    m::Foo::Baz
}
",
        )
    }
}
