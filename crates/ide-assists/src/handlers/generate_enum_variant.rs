use hir::{HasSource, HirDisplay, InFile};
use ide_db::assists::{AssistId, AssistKind};
use syntax::{
    ast::{self, make, HasArgList},
    AstNode,
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
        return add_variant_to_accumulator(acc, ctx, target, e, &name_ref, &path);
    }

    None
}

fn add_variant_to_accumulator(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
    target: syntax::TextRange,
    adt: hir::Enum,
    name_ref: &ast::NameRef,
    path: &ast::Path,
) -> Option<()> {
    let db = ctx.db();
    let InFile { file_id, value: enum_node } = adt.source(db)?.original_ast_node(db)?;

    acc.add(
        AssistId("generate_enum_variant", AssistKind::Generate),
        "Generate variant",
        target,
        |builder| {
            builder.edit_file(file_id.original_file(db));
            let node = builder.make_mut(enum_node);
            let variant = make_variant(ctx, name_ref, &path);
            node.variant_list().map(|it| it.add_variant(variant.clone_for_update()));
        },
    )
}

fn make_variant(
    ctx: &AssistContext<'_>,
    name_ref: &ast::NameRef,
    path: &ast::Path,
) -> ast::Variant {
    let field_list = make_field_list(ctx, path);
    make::variant(make::name(&name_ref.text()), field_list)
}

fn make_field_list(ctx: &AssistContext<'_>, path: &ast::Path) -> Option<ast::FieldList> {
    let scope = ctx.sema.scope(&path.syntax())?;
    if let Some(call_expr) =
        path.syntax().parent().and_then(|it| it.parent()).and_then(ast::CallExpr::cast)
    {
        make_tuple_field_list(call_expr, ctx, &scope)
    } else {
        None
    }
}

fn make_tuple_field_list(
    call_expr: ast::CallExpr,
    ctx: &AssistContext<'_>,
    scope: &hir::SemanticsScope<'_>,
) -> Option<ast::FieldList> {
    let args = call_expr.arg_list()?.args();
    let tuple_fields = args.map(|arg| {
        let ty = expr_ty(ctx, arg, &scope);
        make::tuple_field(None, ty)
    });
    Some(make::tuple_field_list(tuple_fields).into())
}

fn expr_ty(ctx: &AssistContext<'_>, arg: ast::Expr, scope: &hir::SemanticsScope<'_>) -> ast::Type {
    let ty = ctx.sema.type_of_expr(&arg).map(|it| it.adjusted());
    let text = ty
        .and_then(|it| it.display_source_code(ctx.db(), scope.module().into()).ok())
        .unwrap_or_else(|| "_".to_string());
    make::ty(&text)
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

    #[test]
    fn associated_single_element_tuple() {
        check_assist(
            generate_enum_variant,
            r"
enum Foo {}
fn main() {
    Foo::Bar$0(true)
}
",
            r"
enum Foo {
    Bar(bool),
}
fn main() {
    Foo::Bar(true)
}
",
        )
    }

    #[test]
    fn associated_single_element_tuple_unknown_type() {
        check_assist(
            generate_enum_variant,
            r"
enum Foo {}
fn main() {
    Foo::Bar$0(x)
}
",
            r"
enum Foo {
    Bar(_),
}
fn main() {
    Foo::Bar(x)
}
",
        )
    }

    #[test]
    fn associated_multi_element_tuple() {
        check_assist(
            generate_enum_variant,
            r"
struct Struct {}
enum Foo {}
fn main() {
    Foo::Bar$0(true, x, Struct {})
}
",
            r"
struct Struct {}
enum Foo {
    Bar(bool, _, Struct),
}
fn main() {
    Foo::Bar(true, x, Struct {})
}
",
        )
    }
}
