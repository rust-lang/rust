use hir::{HasSource, HirDisplay, InFile};
use ide_db::assists::{AssistId, AssistKind};
use syntax::{
    ast::{self, make, HasArgList},
    match_ast, AstNode, SyntaxNode,
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
    let path: ast::Path = ctx.find_node_at_offset()?;
    let parent = path_parent(&path)?;

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
        return add_variant_to_accumulator(acc, ctx, target, e, &name_ref, parent);
    }

    None
}

#[derive(Debug)]
enum PathParent {
    PathExpr(ast::PathExpr),
    RecordExpr(ast::RecordExpr),
    PathPat(ast::PathPat),
    UseTree(ast::UseTree),
}

impl PathParent {
    fn syntax(&self) -> &SyntaxNode {
        match self {
            PathParent::PathExpr(it) => it.syntax(),
            PathParent::RecordExpr(it) => it.syntax(),
            PathParent::PathPat(it) => it.syntax(),
            PathParent::UseTree(it) => it.syntax(),
        }
    }

    fn make_field_list(&self, ctx: &AssistContext<'_>) -> Option<ast::FieldList> {
        let scope = ctx.sema.scope(self.syntax())?;

        match self {
            PathParent::PathExpr(it) => {
                if let Some(call_expr) = it.syntax().parent().and_then(ast::CallExpr::cast) {
                    make_tuple_field_list(call_expr, ctx, &scope)
                } else {
                    None
                }
            }
            PathParent::RecordExpr(it) => make_record_field_list(it, ctx, &scope),
            PathParent::UseTree(_) | PathParent::PathPat(_) => None,
        }
    }
}

fn path_parent(path: &ast::Path) -> Option<PathParent> {
    let parent = path.syntax().parent()?;

    match_ast! {
        match parent {
            ast::PathExpr(it) => Some(PathParent::PathExpr(it)),
            ast::RecordExpr(it) => Some(PathParent::RecordExpr(it)),
            ast::PathPat(it) => Some(PathParent::PathPat(it)),
            ast::UseTree(it) => Some(PathParent::UseTree(it)),
            _ => None
        }
    }
}

fn add_variant_to_accumulator(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
    target: syntax::TextRange,
    adt: hir::Enum,
    name_ref: &ast::NameRef,
    parent: PathParent,
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
            let variant = make_variant(ctx, name_ref, parent);
            node.variant_list().map(|it| it.add_variant(variant.clone_for_update()));
        },
    )
}

fn make_variant(
    ctx: &AssistContext<'_>,
    name_ref: &ast::NameRef,
    parent: PathParent,
) -> ast::Variant {
    let field_list = parent.make_field_list(ctx);
    make::variant(make::name(&name_ref.text()), field_list)
}

fn make_record_field_list(
    record: &ast::RecordExpr,
    ctx: &AssistContext<'_>,
    scope: &hir::SemanticsScope<'_>,
) -> Option<ast::FieldList> {
    let fields = record.record_expr_field_list()?.fields();
    let record_fields = fields.map(|field| {
        let name = name_from_field(&field);

        let ty = field
            .expr()
            .and_then(|it| expr_ty(ctx, it, scope))
            .unwrap_or_else(make::ty_placeholder);

        make::record_field(None, name, ty)
    });
    Some(make::record_field_list(record_fields).into())
}

fn name_from_field(field: &ast::RecordExprField) -> ast::Name {
    let text = match field.name_ref() {
        Some(it) => it.to_string(),
        None => name_from_field_shorthand(field).unwrap_or("unknown".to_string()),
    };
    make::name(&text)
}

fn name_from_field_shorthand(field: &ast::RecordExprField) -> Option<String> {
    let path = match field.expr()? {
        ast::Expr::PathExpr(path_expr) => path_expr.path(),
        _ => None,
    }?;
    Some(path.as_single_name_ref()?.to_string())
}

fn make_tuple_field_list(
    call_expr: ast::CallExpr,
    ctx: &AssistContext<'_>,
    scope: &hir::SemanticsScope<'_>,
) -> Option<ast::FieldList> {
    let args = call_expr.arg_list()?.args();
    let tuple_fields = args.map(|arg| {
        let ty = expr_ty(ctx, arg, scope).unwrap_or_else(make::ty_placeholder);
        make::tuple_field(None, ty)
    });
    Some(make::tuple_field_list(tuple_fields).into())
}

fn expr_ty(
    ctx: &AssistContext<'_>,
    arg: ast::Expr,
    scope: &hir::SemanticsScope<'_>,
) -> Option<ast::Type> {
    let ty = ctx.sema.type_of_expr(&arg).map(|it| it.adjusted())?;
    let text = ty.display_source_code(ctx.db(), scope.module().into(), false).ok()?;
    Some(make::ty(&text))
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
pub enum Foo {
    Bar,
}
",
            r"
pub enum Foo {
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
    pub enum Foo {
        Bar,
    }
}
fn main() {
    m::Foo::Baz$0
}
",
            r"
mod m {
    pub enum Foo {
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

    #[test]
    fn associated_record() {
        check_assist(
            generate_enum_variant,
            r"
enum Foo {}
fn main() {
    Foo::$0Bar { x: true }
}
",
            r"
enum Foo {
    Bar { x: bool },
}
fn main() {
    Foo::Bar { x: true }
}
",
        )
    }

    #[test]
    fn associated_record_unknown_type() {
        check_assist(
            generate_enum_variant,
            r"
enum Foo {}
fn main() {
    Foo::$0Bar { x: y }
}
",
            r"
enum Foo {
    Bar { x: _ },
}
fn main() {
    Foo::Bar { x: y }
}
",
        )
    }

    #[test]
    fn associated_record_field_shorthand() {
        check_assist(
            generate_enum_variant,
            r"
enum Foo {}
fn main() {
    let x = true;
    Foo::$0Bar { x }
}
",
            r"
enum Foo {
    Bar { x: bool },
}
fn main() {
    let x = true;
    Foo::Bar { x }
}
",
        )
    }

    #[test]
    fn associated_record_field_shorthand_unknown_type() {
        check_assist(
            generate_enum_variant,
            r"
enum Foo {}
fn main() {
    Foo::$0Bar { x }
}
",
            r"
enum Foo {
    Bar { x: _ },
}
fn main() {
    Foo::Bar { x }
}
",
        )
    }

    #[test]
    fn associated_record_field_multiple_fields() {
        check_assist(
            generate_enum_variant,
            r"
struct Struct {}
enum Foo {}
fn main() {
    Foo::$0Bar { x, y: x, s: Struct {} }
}
",
            r"
struct Struct {}
enum Foo {
    Bar { x: _, y: _, s: Struct },
}
fn main() {
    Foo::Bar { x, y: x, s: Struct {} }
}
",
        )
    }

    #[test]
    fn use_tree() {
        check_assist(
            generate_enum_variant,
            r"
//- /main.rs
mod foo;
use foo::Foo::Bar$0;

//- /foo.rs
pub enum Foo {}
",
            r"
pub enum Foo {
    Bar,
}
",
        )
    }

    #[test]
    fn not_applicable_for_path_type() {
        check_assist_not_applicable(
            generate_enum_variant,
            r"
enum Foo {}
impl Foo::Bar$0 {}
",
        )
    }

    #[test]
    fn path_pat() {
        check_assist(
            generate_enum_variant,
            r"
enum Foo {}
fn foo(x: Foo) {
    match x {
        Foo::Bar$0 =>
    }
}
",
            r"
enum Foo {
    Bar,
}
fn foo(x: Foo) {
    match x {
        Foo::Bar =>
    }
}
",
        )
    }
}
