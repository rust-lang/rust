use std::iter;

use either::Either;
use hir::{Module, ModuleDef, Name, Variant};
use ide_db::{
    defs::Definition,
    helpers::{
        insert_use::{insert_use, ImportScope, InsertUseConfig},
        mod_path_to_ast,
    },
    search::FileReference,
    RootDatabase,
};
use itertools::Itertools;
use rustc_hash::FxHashSet;
use syntax::{
    ast::{
        self, edit::IndentLevel, edit_in_place::Indent, make, AstNode, HasAttrs, HasGenericParams,
        HasName, HasTypeBounds, HasVisibility,
    },
    match_ast,
    ted::{self, Position},
    SyntaxKind::*,
    SyntaxNode, T,
};

use crate::{assist_context::AssistBuilder, AssistContext, AssistId, AssistKind, Assists};

// Assist: extract_struct_from_enum_variant
//
// Extracts a struct from enum variant.
//
// ```
// enum A { $0One(u32, u32) }
// ```
// ->
// ```
// struct One(u32, u32);
//
// enum A { One(One) }
// ```
pub(crate) fn extract_struct_from_enum_variant(
    acc: &mut Assists,
    ctx: &AssistContext,
) -> Option<()> {
    let variant = ctx.find_node_at_offset::<ast::Variant>()?;
    let field_list = extract_field_list_if_applicable(&variant)?;

    let variant_name = variant.name()?;
    let variant_hir = ctx.sema.to_def(&variant)?;
    if existing_definition(ctx.db(), &variant_name, &variant_hir) {
        cov_mark::hit!(test_extract_enum_not_applicable_if_struct_exists);
        return None;
    }

    let enum_ast = variant.parent_enum();
    let enum_hir = ctx.sema.to_def(&enum_ast)?;
    let target = variant.syntax().text_range();
    acc.add(
        AssistId("extract_struct_from_enum_variant", AssistKind::RefactorRewrite),
        "Extract struct from enum variant",
        target,
        |builder| {
            let variant_hir_name = variant_hir.name(ctx.db());
            let enum_module_def = ModuleDef::from(enum_hir);
            let usages = Definition::Variant(variant_hir).usages(&ctx.sema).all();

            let mut visited_modules_set = FxHashSet::default();
            let current_module = enum_hir.module(ctx.db());
            visited_modules_set.insert(current_module);
            // record file references of the file the def resides in, we only want to swap to the edited file in the builder once
            let mut def_file_references = None;
            for (file_id, references) in usages {
                if file_id == ctx.file_id() {
                    def_file_references = Some(references);
                    continue;
                }
                builder.edit_file(file_id);
                let processed = process_references(
                    ctx,
                    builder,
                    &mut visited_modules_set,
                    &enum_module_def,
                    &variant_hir_name,
                    references,
                );
                processed.into_iter().for_each(|(path, node, import)| {
                    apply_references(ctx.config.insert_use, path, node, import)
                });
            }
            builder.edit_file(ctx.file_id());

            let variant = builder.make_mut(variant.clone());
            if let Some(references) = def_file_references {
                let processed = process_references(
                    ctx,
                    builder,
                    &mut visited_modules_set,
                    &enum_module_def,
                    &variant_hir_name,
                    references,
                );
                processed.into_iter().for_each(|(path, node, import)| {
                    apply_references(ctx.config.insert_use, path, node, import)
                });
            }

            let indent = enum_ast.indent_level();
            let def = create_struct_def(variant_name.clone(), &variant, &field_list, &enum_ast);
            def.reindent_to(indent);

            let start_offset = &variant.parent_enum().syntax().clone();
            ted::insert_all_raw(
                ted::Position::before(start_offset),
                vec![
                    def.syntax().clone().into(),
                    make::tokens::whitespace(&format!("\n\n{}", indent)).into(),
                ],
            );

            update_variant(&variant, enum_ast.generic_param_list());
        },
    )
}

fn extract_field_list_if_applicable(
    variant: &ast::Variant,
) -> Option<Either<ast::RecordFieldList, ast::TupleFieldList>> {
    match variant.kind() {
        ast::StructKind::Record(field_list) if field_list.fields().next().is_some() => {
            Some(Either::Left(field_list))
        }
        ast::StructKind::Tuple(field_list) if field_list.fields().count() > 1 => {
            Some(Either::Right(field_list))
        }
        _ => None,
    }
}

fn existing_definition(db: &RootDatabase, variant_name: &ast::Name, variant: &Variant) -> bool {
    variant
        .parent_enum(db)
        .module(db)
        .scope(db, None)
        .into_iter()
        .filter(|(_, def)| match def {
            // only check type-namespace
            hir::ScopeDef::ModuleDef(def) => matches!(
                def,
                ModuleDef::Module(_)
                    | ModuleDef::Adt(_)
                    | ModuleDef::Variant(_)
                    | ModuleDef::Trait(_)
                    | ModuleDef::TypeAlias(_)
                    | ModuleDef::BuiltinType(_)
            ),
            _ => false,
        })
        .any(|(name, _)| name.to_string() == variant_name.to_string())
}

fn create_struct_def(
    variant_name: ast::Name,
    variant: &ast::Variant,
    field_list: &Either<ast::RecordFieldList, ast::TupleFieldList>,
    enum_: &ast::Enum,
) -> ast::Struct {
    let enum_vis = enum_.visibility();

    let insert_vis = |node: &'_ SyntaxNode, vis: &'_ SyntaxNode| {
        let vis = vis.clone_for_update();
        ted::insert(ted::Position::before(node), vis);
    };

    // for fields without any existing visibility, use visibility of enum
    let field_list: ast::FieldList = match field_list {
        Either::Left(field_list) => {
            let field_list = field_list.clone_for_update();

            if let Some(vis) = &enum_vis {
                field_list
                    .fields()
                    .filter(|field| field.visibility().is_none())
                    .filter_map(|field| field.name())
                    .for_each(|it| insert_vis(it.syntax(), vis.syntax()));
            }

            field_list.into()
        }
        Either::Right(field_list) => {
            let field_list = field_list.clone_for_update();

            if let Some(vis) = &enum_vis {
                field_list
                    .fields()
                    .filter(|field| field.visibility().is_none())
                    .filter_map(|field| field.ty())
                    .for_each(|it| insert_vis(it.syntax(), vis.syntax()));
            }

            field_list.into()
        }
    };

    field_list.reindent_to(IndentLevel::single());

    // FIXME: This uses all the generic params of the enum, but the variant might not use all of them.
    let strukt = make::struct_(enum_vis, variant_name, enum_.generic_param_list(), field_list)
        .clone_for_update();

    // FIXME: Consider making this an actual function somewhere (like in `AttrsOwnerEdit`) after some deliberation
    let attrs_and_docs = |node: &SyntaxNode| {
        let mut select_next_ws = false;
        node.children_with_tokens().filter(move |child| {
            let accept = match child.kind() {
                ATTR | COMMENT => {
                    select_next_ws = true;
                    return true;
                }
                WHITESPACE if select_next_ws => true,
                _ => false,
            };
            select_next_ws = false;

            accept
        })
    };

    // copy attributes & comments from variant
    let variant_attrs = attrs_and_docs(variant.syntax())
        .map(|tok| match tok.kind() {
            WHITESPACE => make::tokens::single_newline().into(),
            _ => tok,
        })
        .collect();
    ted::insert_all(Position::first_child_of(strukt.syntax()), variant_attrs);

    // copy attributes from enum
    ted::insert_all(
        Position::first_child_of(strukt.syntax()),
        enum_.attrs().map(|it| it.syntax().clone_for_update().into()).collect(),
    );
    strukt
}

fn update_variant(variant: &ast::Variant, generic: Option<ast::GenericParamList>) -> Option<()> {
    let name = variant.name()?;
    let ty = match generic {
        // FIXME: This uses all the generic params of the enum, but the variant might not use all of them.
        Some(gpl) => {
            let gpl = gpl.clone_for_update();
            gpl.generic_params().for_each(|gp| {
                let tbl = match gp {
                    ast::GenericParam::LifetimeParam(it) => it.type_bound_list(),
                    ast::GenericParam::TypeParam(it) => it.type_bound_list(),
                    ast::GenericParam::ConstParam(_) => return,
                };
                if let Some(tbl) = tbl {
                    tbl.remove();
                }
            });
            make::ty(&format!("{}<{}>", name.text(), gpl.generic_params().join(", ")))
        }
        None => make::ty(&name.text()),
    };
    let tuple_field = make::tuple_field(None, ty);
    let replacement = make::variant(
        name,
        Some(ast::FieldList::TupleFieldList(make::tuple_field_list(iter::once(tuple_field)))),
    )
    .clone_for_update();
    ted::replace(variant.syntax(), replacement.syntax());
    Some(())
}

fn apply_references(
    insert_use_cfg: InsertUseConfig,
    segment: ast::PathSegment,
    node: SyntaxNode,
    import: Option<(ImportScope, hir::ModPath)>,
) {
    if let Some((scope, path)) = import {
        insert_use(&scope, mod_path_to_ast(&path), &insert_use_cfg);
    }
    // deep clone to prevent cycle
    let path = make::path_from_segments(iter::once(segment.clone_subtree()), false);
    ted::insert_raw(ted::Position::before(segment.syntax()), path.clone_for_update().syntax());
    ted::insert_raw(ted::Position::before(segment.syntax()), make::token(T!['(']));
    ted::insert_raw(ted::Position::after(&node), make::token(T![')']));
}

fn process_references(
    ctx: &AssistContext,
    builder: &mut AssistBuilder,
    visited_modules: &mut FxHashSet<Module>,
    enum_module_def: &ModuleDef,
    variant_hir_name: &Name,
    refs: Vec<FileReference>,
) -> Vec<(ast::PathSegment, SyntaxNode, Option<(ImportScope, hir::ModPath)>)> {
    // we have to recollect here eagerly as we are about to edit the tree we need to calculate the changes
    // and corresponding nodes up front
    refs.into_iter()
        .flat_map(|reference| {
            let (segment, scope_node, module) = reference_to_node(&ctx.sema, reference)?;
            let segment = builder.make_mut(segment);
            let scope_node = builder.make_syntax_mut(scope_node);
            if !visited_modules.contains(&module) {
                let mod_path = module.find_use_path_prefixed(
                    ctx.sema.db,
                    *enum_module_def,
                    ctx.config.insert_use.prefix_kind,
                );
                if let Some(mut mod_path) = mod_path {
                    mod_path.pop_segment();
                    mod_path.push_segment(variant_hir_name.clone());
                    let scope = ImportScope::find_insert_use_container(&scope_node, &ctx.sema)?;
                    visited_modules.insert(module);
                    return Some((segment, scope_node, Some((scope, mod_path))));
                }
            }
            Some((segment, scope_node, None))
        })
        .collect()
}

fn reference_to_node(
    sema: &hir::Semantics<RootDatabase>,
    reference: FileReference,
) -> Option<(ast::PathSegment, SyntaxNode, hir::Module)> {
    let segment =
        reference.name.as_name_ref()?.syntax().parent().and_then(ast::PathSegment::cast)?;
    let parent = segment.parent_path().syntax().parent()?;
    let expr_or_pat = match_ast! {
        match parent {
            ast::PathExpr(_it) => parent.parent()?,
            ast::RecordExpr(_it) => parent,
            ast::TupleStructPat(_it) => parent,
            ast::RecordPat(_it) => parent,
            _ => return None,
        }
    };
    let module = sema.scope(&expr_or_pat).module()?;
    Some((segment, expr_or_pat, module))
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_extract_struct_several_fields_tuple() {
        check_assist(
            extract_struct_from_enum_variant,
            "enum A { $0One(u32, u32) }",
            r#"struct One(u32, u32);

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_several_fields_named() {
        check_assist(
            extract_struct_from_enum_variant,
            "enum A { $0One { foo: u32, bar: u32 } }",
            r#"struct One{ foo: u32, bar: u32 }

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_one_field_named() {
        check_assist(
            extract_struct_from_enum_variant,
            "enum A { $0One { foo: u32 } }",
            r#"struct One{ foo: u32 }

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_carries_over_generics() {
        check_assist(
            extract_struct_from_enum_variant,
            r"enum En<T> { Var { a: T$0 } }",
            r#"struct Var<T>{ a: T }

enum En<T> { Var(Var<T>) }"#,
        );
    }

    #[test]
    fn test_extract_struct_carries_over_attributes() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"#[derive(Debug)]
#[derive(Clone)]
enum Enum { Variant{ field: u32$0 } }"#,
            r#"#[derive(Debug)]#[derive(Clone)] struct Variant{ field: u32 }

#[derive(Debug)]
#[derive(Clone)]
enum Enum { Variant(Variant) }"#,
        );
    }

    #[test]
    fn test_extract_struct_indent_to_parent_enum() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum Enum {
    Variant {
        field: u32$0
    }
}"#,
            r#"
struct Variant{
    field: u32
}

enum Enum {
    Variant(Variant)
}"#,
        );
    }

    #[test]
    fn test_extract_struct_indent_to_parent_enum_in_mod() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
mod indenting {
    enum Enum {
        Variant {
            field: u32$0
        }
    }
}"#,
            r#"
mod indenting {
    struct Variant{
        field: u32
    }

    enum Enum {
        Variant(Variant)
    }
}"#,
        );
    }

    #[test]
    fn test_extract_struct_keep_comments_and_attrs_one_field_named() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum A {
    $0One {
        // leading comment
        /// doc comment
        #[an_attr]
        foo: u32
        // trailing comment
    }
}"#,
            r#"
struct One{
    // leading comment
    /// doc comment
    #[an_attr]
    foo: u32
    // trailing comment
}

enum A {
    One(One)
}"#,
        );
    }

    #[test]
    fn test_extract_struct_keep_comments_and_attrs_several_fields_named() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum A {
    $0One {
        // comment
        /// doc
        #[attr]
        foo: u32,
        // comment
        #[attr]
        /// doc
        bar: u32
    }
}"#,
            r#"
struct One{
    // comment
    /// doc
    #[attr]
    foo: u32,
    // comment
    #[attr]
    /// doc
    bar: u32
}

enum A {
    One(One)
}"#,
        );
    }

    #[test]
    fn test_extract_struct_keep_comments_and_attrs_several_fields_tuple() {
        check_assist(
            extract_struct_from_enum_variant,
            "enum A { $0One(/* comment */ #[attr] u32, /* another */ u32 /* tail */) }",
            r#"
struct One(/* comment */ #[attr] u32, /* another */ u32 /* tail */);

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_keep_comments_and_attrs_on_variant_struct() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum A {
    /* comment */
    // other
    /// comment
    #[attr]
    $0One {
        a: u32
    }
}"#,
            r#"
/* comment */
// other
/// comment
#[attr]
struct One{
    a: u32
}

enum A {
    One(One)
}"#,
        );
    }

    #[test]
    fn test_extract_struct_keep_comments_and_attrs_on_variant_tuple() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum A {
    /* comment */
    // other
    /// comment
    #[attr]
    $0One(u32, u32)
}"#,
            r#"
/* comment */
// other
/// comment
#[attr]
struct One(u32, u32);

enum A {
    One(One)
}"#,
        );
    }

    #[test]
    fn test_extract_struct_keep_existing_visibility_named() {
        check_assist(
            extract_struct_from_enum_variant,
            "enum A { $0One{ a: u32, pub(crate) b: u32, pub(super) c: u32, d: u32 } }",
            r#"
struct One{ a: u32, pub(crate) b: u32, pub(super) c: u32, d: u32 }

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_keep_existing_visibility_tuple() {
        check_assist(
            extract_struct_from_enum_variant,
            "enum A { $0One(u32, pub(crate) u32, pub(super) u32, u32) }",
            r#"
struct One(u32, pub(crate) u32, pub(super) u32, u32);

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_enum_variant_name_value_namespace() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"const One: () = ();
enum A { $0One(u32, u32) }"#,
            r#"const One: () = ();
struct One(u32, u32);

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_no_visibility() {
        check_assist(
            extract_struct_from_enum_variant,
            "enum A { $0One(u32, u32) }",
            r#"
struct One(u32, u32);

enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_pub_visibility() {
        check_assist(
            extract_struct_from_enum_variant,
            "pub enum A { $0One(u32, u32) }",
            r#"
pub struct One(pub u32, pub u32);

pub enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_pub_in_mod_visibility() {
        check_assist(
            extract_struct_from_enum_variant,
            "pub(in something) enum A { $0One{ a: u32, b: u32 } }",
            r#"
pub(in something) struct One{ pub(in something) a: u32, pub(in something) b: u32 }

pub(in something) enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_pub_crate_visibility() {
        check_assist(
            extract_struct_from_enum_variant,
            "pub(crate) enum A { $0One{ a: u32, b: u32, c: u32 } }",
            r#"
pub(crate) struct One{ pub(crate) a: u32, pub(crate) b: u32, pub(crate) c: u32 }

pub(crate) enum A { One(One) }"#,
        );
    }

    #[test]
    fn test_extract_struct_with_complex_imports() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"mod my_mod {
    fn another_fn() {
        let m = my_other_mod::MyEnum::MyField(1, 1);
    }

    pub mod my_other_mod {
        fn another_fn() {
            let m = MyEnum::MyField(1, 1);
        }

        pub enum MyEnum {
            $0MyField(u8, u8),
        }
    }
}

fn another_fn() {
    let m = my_mod::my_other_mod::MyEnum::MyField(1, 1);
}"#,
            r#"use my_mod::my_other_mod::MyField;

mod my_mod {
    use self::my_other_mod::MyField;

    fn another_fn() {
        let m = my_other_mod::MyEnum::MyField(MyField(1, 1));
    }

    pub mod my_other_mod {
        fn another_fn() {
            let m = MyEnum::MyField(MyField(1, 1));
        }

        pub struct MyField(pub u8, pub u8);

        pub enum MyEnum {
            MyField(MyField),
        }
    }
}

fn another_fn() {
    let m = my_mod::my_other_mod::MyEnum::MyField(MyField(1, 1));
}"#,
        );
    }

    #[test]
    fn extract_record_fix_references() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum E {
    $0V { i: i32, j: i32 }
}

fn f() {
    let E::V { i, j } = E::V { i: 9, j: 2 };
}
"#,
            r#"
struct V{ i: i32, j: i32 }

enum E {
    V(V)
}

fn f() {
    let E::V(V { i, j }) = E::V(V { i: 9, j: 2 });
}
"#,
        )
    }

    #[test]
    fn extract_record_fix_references2() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum E {
    $0V(i32, i32)
}

fn f() {
    let E::V(i, j) = E::V(9, 2);
}
"#,
            r#"
struct V(i32, i32);

enum E {
    V(V)
}

fn f() {
    let E::V(V(i, j)) = E::V(V(9, 2));
}
"#,
        )
    }

    #[test]
    fn test_several_files() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
//- /main.rs
enum E {
    $0V(i32, i32)
}
mod foo;

//- /foo.rs
use crate::E;
fn f() {
    let e = E::V(9, 2);
}
"#,
            r#"
//- /main.rs
struct V(i32, i32);

enum E {
    V(V)
}
mod foo;

//- /foo.rs
use crate::{E, V};
fn f() {
    let e = E::V(V(9, 2));
}
"#,
        )
    }

    #[test]
    fn test_several_files_record() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
//- /main.rs
enum E {
    $0V { i: i32, j: i32 }
}
mod foo;

//- /foo.rs
use crate::E;
fn f() {
    let e = E::V { i: 9, j: 2 };
}
"#,
            r#"
//- /main.rs
struct V{ i: i32, j: i32 }

enum E {
    V(V)
}
mod foo;

//- /foo.rs
use crate::{E, V};
fn f() {
    let e = E::V(V { i: 9, j: 2 });
}
"#,
        )
    }

    #[test]
    fn test_extract_struct_record_nested_call_exp() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum A { $0One { a: u32, b: u32 } }

struct B(A);

fn foo() {
    let _ = B(A::One { a: 1, b: 2 });
}
"#,
            r#"
struct One{ a: u32, b: u32 }

enum A { One(One) }

struct B(A);

fn foo() {
    let _ = B(A::One(One { a: 1, b: 2 }));
}
"#,
        );
    }

    #[test]
    fn test_extract_enum_not_applicable_for_element_with_no_fields() {
        check_assist_not_applicable(extract_struct_from_enum_variant, r#"enum A { $0One }"#);
    }

    #[test]
    fn test_extract_enum_not_applicable_if_struct_exists() {
        cov_mark::check!(test_extract_enum_not_applicable_if_struct_exists);
        check_assist_not_applicable(
            extract_struct_from_enum_variant,
            r#"
struct One;
enum A { $0One(u8, u32) }
"#,
        );
    }

    #[test]
    fn test_extract_not_applicable_one_field() {
        check_assist_not_applicable(extract_struct_from_enum_variant, r"enum A { $0One(u32) }");
    }

    #[test]
    fn test_extract_not_applicable_no_field_tuple() {
        check_assist_not_applicable(extract_struct_from_enum_variant, r"enum A { $0None() }");
    }

    #[test]
    fn test_extract_not_applicable_no_field_named() {
        check_assist_not_applicable(extract_struct_from_enum_variant, r"enum A { $0None {} }");
    }
}
