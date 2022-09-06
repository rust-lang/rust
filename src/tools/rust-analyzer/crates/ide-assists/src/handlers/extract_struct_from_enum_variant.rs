use std::iter;

use either::Either;
use hir::{Module, ModuleDef, Name, Variant};
use ide_db::{
    defs::Definition,
    helpers::mod_path_to_ast,
    imports::insert_use::{insert_use, ImportScope, InsertUseConfig},
    search::FileReference,
    FxHashSet, RootDatabase,
};
use itertools::{Itertools, Position};
use syntax::{
    ast::{
        self, edit::IndentLevel, edit_in_place::Indent, make, AstNode, HasAttrs, HasGenericParams,
        HasName, HasVisibility,
    },
    match_ast, ted, SyntaxElement,
    SyntaxKind::*,
    SyntaxNode, T,
};

use crate::{assist_context::SourceChangeBuilder, AssistContext, AssistId, AssistKind, Assists};

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
    ctx: &AssistContext<'_>,
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

            let generic_params = enum_ast
                .generic_param_list()
                .and_then(|known_generics| extract_generic_params(&known_generics, &field_list));
            let generics = generic_params.as_ref().map(|generics| generics.clone_for_update());
            let def =
                create_struct_def(variant_name.clone(), &variant, &field_list, generics, &enum_ast);

            let enum_ast = variant.parent_enum();
            let indent = enum_ast.indent_level();
            def.reindent_to(indent);

            ted::insert_all(
                ted::Position::before(enum_ast.syntax()),
                vec![
                    def.syntax().clone().into(),
                    make::tokens::whitespace(&format!("\n\n{indent}")).into(),
                ],
            );

            update_variant(&variant, generic_params.map(|g| g.clone_for_update()));
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

fn extract_generic_params(
    known_generics: &ast::GenericParamList,
    field_list: &Either<ast::RecordFieldList, ast::TupleFieldList>,
) -> Option<ast::GenericParamList> {
    let mut generics = known_generics.generic_params().map(|param| (param, false)).collect_vec();

    let tagged_one = match field_list {
        Either::Left(field_list) => field_list
            .fields()
            .filter_map(|f| f.ty())
            .fold(false, |tagged, ty| tag_generics_in_variant(&ty, &mut generics) || tagged),
        Either::Right(field_list) => field_list
            .fields()
            .filter_map(|f| f.ty())
            .fold(false, |tagged, ty| tag_generics_in_variant(&ty, &mut generics) || tagged),
    };

    let generics = generics.into_iter().filter_map(|(param, tag)| tag.then(|| param));
    tagged_one.then(|| make::generic_param_list(generics))
}

fn tag_generics_in_variant(ty: &ast::Type, generics: &mut [(ast::GenericParam, bool)]) -> bool {
    let mut tagged_one = false;

    for token in ty.syntax().descendants_with_tokens().filter_map(SyntaxElement::into_token) {
        for (param, tag) in generics.iter_mut().filter(|(_, tag)| !tag) {
            match param {
                ast::GenericParam::LifetimeParam(lt)
                    if matches!(token.kind(), T![lifetime_ident]) =>
                {
                    if let Some(lt) = lt.lifetime() {
                        if lt.text().as_str() == token.text() {
                            *tag = true;
                            tagged_one = true;
                            break;
                        }
                    }
                }
                param if matches!(token.kind(), T![ident]) => {
                    if match param {
                        ast::GenericParam::ConstParam(konst) => konst
                            .name()
                            .map(|name| name.text().as_str() == token.text())
                            .unwrap_or_default(),
                        ast::GenericParam::TypeParam(ty) => ty
                            .name()
                            .map(|name| name.text().as_str() == token.text())
                            .unwrap_or_default(),
                        ast::GenericParam::LifetimeParam(lt) => lt
                            .lifetime()
                            .map(|lt| lt.text().as_str() == token.text())
                            .unwrap_or_default(),
                    } {
                        *tag = true;
                        tagged_one = true;
                        break;
                    }
                }
                _ => (),
            }
        }
    }

    tagged_one
}

fn create_struct_def(
    name: ast::Name,
    variant: &ast::Variant,
    field_list: &Either<ast::RecordFieldList, ast::TupleFieldList>,
    generics: Option<ast::GenericParamList>,
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

    let strukt = make::struct_(enum_vis, name, generics, field_list).clone_for_update();

    // take comments from variant
    ted::insert_all(
        ted::Position::first_child_of(strukt.syntax()),
        take_all_comments(variant.syntax()),
    );

    // copy attributes from enum
    ted::insert_all(
        ted::Position::first_child_of(strukt.syntax()),
        enum_
            .attrs()
            .flat_map(|it| {
                vec![it.syntax().clone_for_update().into(), make::tokens::single_newline().into()]
            })
            .collect(),
    );

    strukt
}

fn update_variant(variant: &ast::Variant, generics: Option<ast::GenericParamList>) -> Option<()> {
    let name = variant.name()?;
    let ty = generics
        .filter(|generics| generics.generic_params().count() > 0)
        .map(|generics| {
            let mut generic_str = String::with_capacity(8);

            for (p, more) in generics.generic_params().with_position().map(|p| match p {
                Position::First(p) | Position::Middle(p) => (p, true),
                Position::Last(p) | Position::Only(p) => (p, false),
            }) {
                match p {
                    ast::GenericParam::ConstParam(konst) => {
                        if let Some(name) = konst.name() {
                            generic_str.push_str(name.text().as_str());
                        }
                    }
                    ast::GenericParam::LifetimeParam(lt) => {
                        if let Some(lt) = lt.lifetime() {
                            generic_str.push_str(lt.text().as_str());
                        }
                    }
                    ast::GenericParam::TypeParam(ty) => {
                        if let Some(name) = ty.name() {
                            generic_str.push_str(name.text().as_str());
                        }
                    }
                }
                if more {
                    generic_str.push_str(", ");
                }
            }

            make::ty(&format!("{}<{}>", &name.text(), &generic_str))
        })
        .unwrap_or_else(|| make::ty(&name.text()));

    // change from a record to a tuple field list
    let tuple_field = make::tuple_field(None, ty);
    let field_list = make::tuple_field_list(iter::once(tuple_field)).clone_for_update();
    ted::replace(variant.field_list()?.syntax(), field_list.syntax());

    // remove any ws after the name
    if let Some(ws) = name
        .syntax()
        .siblings_with_tokens(syntax::Direction::Next)
        .find_map(|tok| tok.into_token().filter(|tok| tok.kind() == WHITESPACE))
    {
        ted::remove(SyntaxElement::Token(ws));
    }

    Some(())
}

// Note: this also detaches whitespace after comments,
// since `SyntaxNode::splice_children` (and by extension `ted::insert_all_raw`)
// detaches nodes. If we only took the comments, we'd leave behind the old whitespace.
fn take_all_comments(node: &SyntaxNode) -> Vec<SyntaxElement> {
    let mut remove_next_ws = false;
    node.children_with_tokens()
        .filter_map(move |child| match child.kind() {
            COMMENT => {
                remove_next_ws = true;
                child.detach();
                Some(child)
            }
            WHITESPACE if remove_next_ws => {
                remove_next_ws = false;
                child.detach();
                Some(make::tokens::single_newline().into())
            }
            _ => {
                remove_next_ws = false;
                None
            }
        })
        .collect()
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
    ctx: &AssistContext<'_>,
    builder: &mut SourceChangeBuilder,
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
    sema: &hir::Semantics<'_, RootDatabase>,
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
    let module = sema.scope(&expr_or_pat)?.module();
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
            r#"
#[derive(Debug)]
#[derive(Clone)]
enum Enum { Variant{ field: u32$0 } }"#,
            r#"
#[derive(Debug)]
#[derive(Clone)]
struct Variant{ field: u32 }

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
    fn test_extract_struct_move_struct_variant_comments() {
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
struct One{
    a: u32
}

enum A {
    #[attr]
    One(One)
}"#,
        );
    }

    #[test]
    fn test_extract_struct_move_tuple_variant_comments() {
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
struct One(u32, u32);

enum A {
    #[attr]
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

    #[test]
    fn test_extract_struct_only_copies_needed_generics() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum X<'a, 'b, 'x> {
    $0A { a: &'a &'x mut () },
    B { b: &'b () },
    C { c: () },
}
"#,
            r#"
struct A<'a, 'x>{ a: &'a &'x mut () }

enum X<'a, 'b, 'x> {
    A(A<'a, 'x>),
    B { b: &'b () },
    C { c: () },
}
"#,
        );
    }

    #[test]
    fn test_extract_struct_with_liftime_type_const() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum X<'b, T, V, const C: usize> {
    $0A { a: T, b: X<'b>, c: [u8; C] },
    D { d: V },
}
"#,
            r#"
struct A<'b, T, const C: usize>{ a: T, b: X<'b>, c: [u8; C] }

enum X<'b, T, V, const C: usize> {
    A(A<'b, T, C>),
    D { d: V },
}
"#,
        );
    }

    #[test]
    fn test_extract_struct_without_generics() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum X<'a, 'b> {
    A { a: &'a () },
    B { b: &'b () },
    $0C { c: () },
}
"#,
            r#"
struct C{ c: () }

enum X<'a, 'b> {
    A { a: &'a () },
    B { b: &'b () },
    C(C),
}
"#,
        );
    }

    #[test]
    fn test_extract_struct_keeps_trait_bounds() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum En<T: TraitT, V: TraitV> {
    $0A { a: T },
    B { b: V },
}
"#,
            r#"
struct A<T: TraitT>{ a: T }

enum En<T: TraitT, V: TraitV> {
    A(A<T>),
    B { b: V },
}
"#,
        );
    }
}
