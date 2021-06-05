use ide_db::defs::{Definition, NameRefClass};
use syntax::{
    ast::{self, AstNode, VisibilityOwner},
    match_ast, SyntaxNode,
};

use crate::{assist_context::AssistBuilder, AssistContext, AssistId, AssistKind, Assists};

// Assist: convert_tuple_variant_to_named_variant
//
// Converts tuple variant to variant with named fields.
//
// ```
// enum A {
//     Variant(usize),
// }
//
// impl A {
//     fn new(value: usize) -> A {
//         A::Variant(value)
//     }
//
//     fn new_with_default() -> A {
//         A::new(Default::default())
//     }
//
//     fn value(self) -> usize {
//         match self {
//             A::Variant(value) => value,
//         }
//     }
// }
// ```
// ->
// ```
// enum A {
//     Variant {
//         field1: usize
//     },
// }
//
// impl A {
//     fn new(value: usize) -> A {
//         A::Variant {
//             field1: value,
//         }
//     }
//
//     fn new_with_default() -> A {
//         A::new(Default::default())
//     }
//
//     fn value(self) -> usize {
//         match self {
//             A::Variant { field1: value } => value,
//         }
//     }
// }
// ```
pub(crate) fn convert_tuple_variant_to_named_variant(
    acc: &mut Assists,
    ctx: &AssistContext,
) -> Option<()> {
    let variant = ctx.find_node_at_offset::<ast::Variant>()?;
    let tuple_fields = match variant.field_list()? {
        ast::FieldList::TupleFieldList(it) => it,
        ast::FieldList::RecordFieldList(_) => return None,
    };
    let variant_def = ctx.sema.to_def(&variant)?;

    let target = variant.syntax().text_range();
    acc.add(
        AssistId("convert_tuple_variant_to_named_variant", AssistKind::RefactorRewrite),
        "Convert to named struct",
        target,
        |edit| {
            let names = generate_names(tuple_fields.fields());
            edit_field_references(ctx, edit, tuple_fields.fields(), &names); // TODO: is this needed?
            edit_variant_references(ctx, edit, variant_def, &names);
            edit_variant_def(ctx, edit, tuple_fields, names);
        },
    )
}

fn edit_variant_def(
    ctx: &AssistContext,
    edit: &mut AssistBuilder,
    tuple_fields: ast::TupleFieldList,
    names: Vec<ast::Name>,
) {
    let record_fields = tuple_fields
        .fields()
        .zip(names)
        .filter_map(|(f, name)| Some(ast::make::record_field(f.visibility(), name, f.ty()?)));
    let record_fields = ast::make::record_field_list(record_fields);
    let tuple_fields_text_range = tuple_fields.syntax().text_range();

    edit.edit_file(ctx.frange.file_id);

    edit.insert(tuple_fields_text_range.start(), ast::make::tokens::single_space().text());

    edit.replace(tuple_fields_text_range, record_fields.to_string());
}

fn edit_variant_references(
    ctx: &AssistContext,
    edit: &mut AssistBuilder,
    variant: hir::Variant,
    names: &[ast::Name],
) {
    let variant_def = Definition::ModuleDef(hir::ModuleDef::Variant(variant));
    let usages = variant_def.usages(&ctx.sema).include_self_refs().all();

    let edit_node = |edit: &mut AssistBuilder, node: SyntaxNode| -> Option<()> {
        match_ast! {
            match node {
                ast::TupleStructPat(tuple_struct_pat) => {
                    edit.replace(
                        tuple_struct_pat.syntax().text_range(),
                        ast::make::record_pat_with_fields(
                            tuple_struct_pat.path()?,
                            ast::make::record_pat_field_list(tuple_struct_pat.fields().zip(names).map(
                                |(pat, name)| {
                                    ast::make::record_pat_field(
                                        ast::make::name_ref(&name.to_string()),
                                        pat,
                                    )
                                },
                            )),
                        )
                        .to_string(),
                    );
                },
                // for tuple struct creations like Foo(42)
                ast::CallExpr(call_expr) => {
                    let path = call_expr.syntax().descendants().find_map(ast::PathExpr::cast).and_then(|expr| expr.path())?;

                    // this also includes method calls like Foo::new(42), we should skip them
                    if let Some(name_ref) = path.segment().and_then(|s| s.name_ref()) {
                        match NameRefClass::classify(&ctx.sema, &name_ref) {
                            Some(NameRefClass::Definition(Definition::SelfType(_))) => {},
                            Some(NameRefClass::Definition(def)) if def == variant_def => {},
                            _ => return None,
                        };
                    }

                    let arg_list = call_expr.syntax().descendants().find_map(ast::ArgList::cast)?;

                    edit.replace(
                        call_expr.syntax().text_range(),
                        ast::make::record_expr(
                            path,
                            ast::make::record_expr_field_list(arg_list.args().zip(names).map(
                                |(expr, name)| {
                                    ast::make::record_expr_field(
                                        ast::make::name_ref(&name.to_string()),
                                        Some(expr),
                                    )
                                },
                            )),
                        )
                        .to_string(),
                    );
                },
                _ => return None,
            }
        }
        Some(())
    };

    for (file_id, refs) in usages {
        edit.edit_file(file_id);
        for r in refs {
            for node in r.name.syntax().ancestors() {
                if edit_node(edit, node).is_some() {
                    break;
                }
            }
        }
    }
}

fn edit_field_references(
    ctx: &AssistContext,
    edit: &mut AssistBuilder,
    fields: impl Iterator<Item = ast::TupleField>,
    names: &[ast::Name],
) {
    for (field, name) in fields.zip(names) {
        let field = match ctx.sema.to_def(&field) {
            Some(it) => it,
            None => continue,
        };
        let def = Definition::Field(field);
        let usages = def.usages(&ctx.sema).all();
        for (file_id, refs) in usages {
            edit.edit_file(file_id);
            for r in refs {
                if let Some(name_ref) = r.name.as_name_ref() {
                    edit.replace(name_ref.syntax().text_range(), name.text());
                }
            }
        }
    }
}

fn generate_names(fields: impl Iterator<Item = ast::TupleField>) -> Vec<ast::Name> {
    fields.enumerate().map(|(i, _)| ast::make::name(&format!("field{}", i + 1))).collect()
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn not_applicable_other_than_tuple_variant() {
        check_assist_not_applicable(
            convert_tuple_variant_to_named_variant,
            r#"enum Enum { Variant$0 { value: usize } };"#,
        );
        check_assist_not_applicable(convert_tuple_variant_to_named_variant, r#"enum Enum { Variant$0 }"#);
    }

    #[test]
    fn convert_simple_variant() {
        check_assist(
            convert_tuple_variant_to_named_variant,
            r#"
enum A {
    $0Variant(usize),
}

impl A {
    fn new(value: usize) -> A {
        A::Variant(value)
    }

    fn new_with_default() -> A {
        A::new(Default::default())
    }

    fn value(self) -> usize {
        match self {
            A::Variant(value) => value,
        }
    }
}"#,
            r#"
enum A {
    Variant { field1: usize },
}

impl A {
    fn new(value: usize) -> A {
        A::Variant { field1: value }
    }

    fn new_with_default() -> A {
        A::new(Default::default())
    }

    fn value(self) -> usize {
        match self {
            A::Variant { field1: value } => value,
        }
    }
}"#,
        );
    }

    #[test]
    fn convert_variant_referenced_via_self_kw() {
        check_assist(
            convert_tuple_variant_to_named_variant,
            r#"
enum A {
    $0Variant(usize),
}

impl A {
    fn new(value: usize) -> A {
        Self::Variant(value)
    }

    fn new_with_default() -> A {
        Self::new(Default::default())
    }

    fn value(self) -> usize {
        match self {
            Self::Variant(value) => value,
        }
    }
}"#,
            r#"
enum A {
    Variant { field1: usize },
}

impl A {
    fn new(value: usize) -> A {
        Self::Variant { field1: value }
    }

    fn new_with_default() -> A {
        Self::new(Default::default())
    }

    fn value(self) -> usize {
        match self {
            Self::Variant { field1: value } => value,
        }
    }
}"#,
        );
    }

    #[test]
    fn convert_destructured_variant() {
        check_assist(
            convert_tuple_variant_to_named_variant,
            r#"
enum A {
    $0Variant(usize),
}

impl A {
    fn into_inner(self) -> usize {
        let A::Variant(first) = self;
        first
    }

    fn into_inner_via_self(self) -> usize {
        let Self::Variant(first) = self;
        first
    }
}"#,
            r#"
enum A {
    Variant { field1: usize },
}

impl A {
    fn into_inner(self) -> usize {
        let A::Variant { field1: first } = self;
        first
    }

    fn into_inner_via_self(self) -> usize {
        let Self::Variant { field1: first } = self;
        first
    }
}"#,
        );
    }

    #[test]
    fn convert_variant_with_wrapped_references() {
        check_assist(
            convert_tuple_variant_to_named_variant,
            r#"
enum Inner {
    $0Variant(usize),
}
enum Outer {
    Variant(Inner),
}

impl Outer {
    fn new() -> Self {
        Self::Variant(Inner::Variant(42))
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer::Variant(Inner::Variant(x)) = self;
        x
    }
}"#,
            r#"
enum Inner {
    Variant { field1: usize },
}
enum Outer {
    Variant(Inner),
}

impl Outer {
    fn new() -> Self {
        Self::Variant(Inner::Variant { field1: 42 })
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer::Variant(Inner::Variant { field1: x }) = self;
        x
    }
}"#,
        );

        check_assist(
            convert_tuple_variant_to_named_variant,
            r#"
enum Inner {
    Variant(usize),
}
enum Outer {
    $0Variant(Inner),
}

impl Outer {
    fn new() -> Self {
        Self::Variant(Inner::Variant(42))
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer::Variant(Inner::Variant(x)) = self;
        x
    }
}"#,
            r#"
enum Inner {
    Variant(usize),
}
enum Outer {
    Variant { field1: Inner },
}

impl Outer {
    fn new() -> Self {
        Self::Variant { field1: Inner::Variant(42) }
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer::Variant { field1: Inner::Variant(x) } = self;
        x
    }
}"#,
        );
    }

    #[test]
    fn convert_variant_with_multi_file_references() {
        check_assist(
            convert_tuple_variant_to_named_variant,
            r#"
//- /main.rs
struct Inner;
enum A {
    $0Variant(Inner),
}

mod foo;

//- /foo.rs
use crate::{A, Inner};
fn f() {
    let a = A::Variant(Inner);
}
"#,
            r#"
//- /main.rs
struct Inner;
enum A {
    Variant { field1: Inner },
}

mod foo;

//- /foo.rs
use crate::{A, Inner};
fn f() {
    let a = A::Variant { field1: Inner };
}
"#,
        );
    }

    #[test]
    fn convert_directly_used_variant() {
        check_assist(
            convert_tuple_variant_to_named_variant,
            r#"
//- /main.rs
struct Inner;
enum A {
    $0Variant(Inner),
}

mod foo;

//- /foo.rs
use crate::{A::Variant, Inner};
fn f() {
    let a = Variant(Inner);
}
"#,
            r#"
//- /main.rs
struct Inner;
enum A {
    Variant { field1: Inner },
}

mod foo;

//- /foo.rs
use crate::{A::Variant, Inner};
fn f() {
    let a = Variant { field1: Inner };
}
"#,
        );
    }
}
