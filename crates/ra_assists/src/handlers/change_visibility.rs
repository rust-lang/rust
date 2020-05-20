use ra_syntax::{
    ast::{self, NameOwner, VisibilityOwner},
    AstNode,
    SyntaxKind::{
        ATTR, COMMENT, CONST_DEF, ENUM_DEF, FN_DEF, MODULE, STRUCT_DEF, TRAIT_DEF, VISIBILITY,
        WHITESPACE,
    },
    SyntaxNode, TextRange, TextSize, T,
};

use hir::{db::HirDatabase, HasSource, HasVisibility, PathResolution};
use test_utils::mark;

use crate::{AssistContext, AssistId, Assists};
use ra_db::FileId;

// Assist: change_visibility
//
// Adds or changes existing visibility specifier.
//
// ```
// <|>fn frobnicate() {}
// ```
// ->
// ```
// pub(crate) fn frobnicate() {}
// ```
pub(crate) fn change_visibility(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    if let Some(vis) = ctx.find_node_at_offset::<ast::Visibility>() {
        return change_vis(acc, vis);
    }
    add_vis(acc, ctx)
        .or_else(|| add_vis_to_referenced_module_def(acc, ctx))
        .or_else(|| add_vis_to_referenced_record_field(acc, ctx))
}

fn add_vis(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let item_keyword = ctx.token_at_offset().find(|leaf| match leaf.kind() {
        T![const] | T![fn] | T![mod] | T![struct] | T![enum] | T![trait] => true,
        _ => false,
    });

    let (offset, target) = if let Some(keyword) = item_keyword {
        let parent = keyword.parent();
        let def_kws = vec![CONST_DEF, FN_DEF, MODULE, STRUCT_DEF, ENUM_DEF, TRAIT_DEF];
        // Parent is not a definition, can't add visibility
        if !def_kws.iter().any(|&def_kw| def_kw == parent.kind()) {
            return None;
        }
        // Already have visibility, do nothing
        if parent.children().any(|child| child.kind() == VISIBILITY) {
            return None;
        }
        (vis_offset(&parent), keyword.text_range())
    } else if let Some(field_name) = ctx.find_node_at_offset::<ast::Name>() {
        let field = field_name.syntax().ancestors().find_map(ast::RecordFieldDef::cast)?;
        if field.name()? != field_name {
            mark::hit!(change_visibility_field_false_positive);
            return None;
        }
        if field.visibility().is_some() {
            return None;
        }
        (vis_offset(field.syntax()), field_name.syntax().text_range())
    } else if let Some(field) = ctx.find_node_at_offset::<ast::TupleFieldDef>() {
        if field.visibility().is_some() {
            return None;
        }
        (vis_offset(field.syntax()), field.syntax().text_range())
    } else {
        return None;
    };

    acc.add(AssistId("change_visibility"), "Change visibility to pub(crate)", target, |edit| {
        edit.insert(offset, "pub(crate) ");
        edit.set_cursor(offset);
    })
}

fn add_vis_to_referenced_module_def(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let path: ast::Path = ctx.find_node_at_offset()?;
    let path_res = ctx.sema.resolve_path(&path)?;
    let def = match path_res {
        PathResolution::Def(def) => def,
        _ => return None,
    };

    let current_module = ctx.sema.scope(&path.syntax()).module()?;
    let target_module = def.module(ctx.db)?;

    let vis = target_module.visibility_of(ctx.db, &def)?;
    if vis.is_visible_from(ctx.db, current_module.into()) {
        return None;
    };

    let (offset, target, target_file, target_name) = target_data_for_def(ctx.db, def)?;

    let missing_visibility =
        if current_module.krate() == target_module.krate() { "pub(crate)" } else { "pub" };

    let assist_label = match target_name {
        None => format!("Change visibility to {}", missing_visibility),
        Some(name) => format!("Change visibility of {} to {}", name, missing_visibility),
    };

    acc.add(AssistId("change_visibility"), assist_label, target, |edit| {
        edit.set_file(target_file);
        edit.insert(offset, format!("{} ", missing_visibility));
        edit.set_cursor(offset);
    })
}

fn add_vis_to_referenced_record_field(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let record_field: ast::RecordField = ctx.find_node_at_offset()?;
    let (record_field_def, _) = ctx.sema.resolve_record_field(&record_field)?;

    let current_module = ctx.sema.scope(record_field.syntax()).module()?;
    let visibility = record_field_def.visibility(ctx.db);
    if visibility.is_visible_from(ctx.db, current_module.into()) {
        return None;
    }

    let parent = record_field_def.parent_def(ctx.db);
    let parent_name = parent.name(ctx.db);
    let target_module = parent.module(ctx.db);

    let in_file_source = record_field_def.source(ctx.db);
    let (offset, target) = match in_file_source.value {
        hir::FieldSource::Named(it) => {
            let s = it.syntax();
            (vis_offset(s), s.text_range())
        }
        hir::FieldSource::Pos(it) => {
            let s = it.syntax();
            (vis_offset(s), s.text_range())
        }
    };

    let missing_visibility =
        if current_module.krate() == target_module.krate() { "pub(crate)" } else { "pub" };
    let target_file = in_file_source.file_id.original_file(ctx.db);

    let target_name = record_field_def.name(ctx.db);
    let assist_label =
        format!("Change visibility of {}.{} to {}", parent_name, target_name, missing_visibility);

    acc.add(AssistId("change_visibility"), assist_label, target, |edit| {
        edit.set_file(target_file);
        edit.insert(offset, format!("{} ", missing_visibility));
        edit.set_cursor(offset)
    })
}

fn target_data_for_def(
    db: &dyn HirDatabase,
    def: hir::ModuleDef,
) -> Option<(TextSize, TextRange, FileId, Option<hir::Name>)> {
    fn offset_target_and_file_id<S, Ast>(
        db: &dyn HirDatabase,
        x: S,
    ) -> (TextSize, TextRange, FileId)
    where
        S: HasSource<Ast = Ast>,
        Ast: AstNode,
    {
        let source = x.source(db);
        let in_file_syntax = source.syntax();
        let file_id = in_file_syntax.file_id;
        let syntax = in_file_syntax.value;
        (vis_offset(syntax), syntax.text_range(), file_id.original_file(db.upcast()))
    }

    let target_name;
    let (offset, target, target_file) = match def {
        hir::ModuleDef::Function(f) => {
            target_name = Some(f.name(db));
            offset_target_and_file_id(db, f)
        }
        hir::ModuleDef::Adt(adt) => {
            target_name = Some(adt.name(db));
            match adt {
                hir::Adt::Struct(s) => offset_target_and_file_id(db, s),
                hir::Adt::Union(u) => offset_target_and_file_id(db, u),
                hir::Adt::Enum(e) => offset_target_and_file_id(db, e),
            }
        }
        hir::ModuleDef::Const(c) => {
            target_name = c.name(db);
            offset_target_and_file_id(db, c)
        }
        hir::ModuleDef::Static(s) => {
            target_name = s.name(db);
            offset_target_and_file_id(db, s)
        }
        hir::ModuleDef::Trait(t) => {
            target_name = Some(t.name(db));
            offset_target_and_file_id(db, t)
        }
        hir::ModuleDef::TypeAlias(t) => {
            target_name = Some(t.name(db));
            offset_target_and_file_id(db, t)
        }
        hir::ModuleDef::Module(m) => {
            target_name = m.name(db);
            let in_file_source = m.declaration_source(db)?;
            let file_id = in_file_source.file_id.original_file(db.upcast());
            let syntax = in_file_source.value.syntax();
            (vis_offset(syntax), syntax.text_range(), file_id)
        }
        // Enum variants can't be private, we can't modify builtin types
        hir::ModuleDef::EnumVariant(_) | hir::ModuleDef::BuiltinType(_) => return None,
    };

    Some((offset, target, target_file, target_name))
}

fn vis_offset(node: &SyntaxNode) -> TextSize {
    node.children_with_tokens()
        .skip_while(|it| match it.kind() {
            WHITESPACE | COMMENT | ATTR => true,
            _ => false,
        })
        .next()
        .map(|it| it.text_range().start())
        .unwrap_or_else(|| node.text_range().start())
}

fn change_vis(acc: &mut Assists, vis: ast::Visibility) -> Option<()> {
    if vis.syntax().text() == "pub" {
        let target = vis.syntax().text_range();
        return acc.add(
            AssistId("change_visibility"),
            "Change Visibility to pub(crate)",
            target,
            |edit| {
                edit.replace(vis.syntax().text_range(), "pub(crate)");
                edit.set_cursor(vis.syntax().text_range().start())
            },
        );
    }
    if vis.syntax().text() == "pub(crate)" {
        let target = vis.syntax().text_range();
        return acc.add(
            AssistId("change_visibility"),
            "Change visibility to pub",
            target,
            |edit| {
                edit.replace(vis.syntax().text_range(), "pub");
                edit.set_cursor(vis.syntax().text_range().start());
            },
        );
    }
    None
}

#[cfg(test)]
mod tests {
    use test_utils::mark;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

    #[test]
    fn change_visibility_adds_pub_crate_to_items() {
        check_assist(change_visibility, "<|>fn foo() {}", "<|>pub(crate) fn foo() {}");
        check_assist(change_visibility, "f<|>n foo() {}", "<|>pub(crate) fn foo() {}");
        check_assist(change_visibility, "<|>struct Foo {}", "<|>pub(crate) struct Foo {}");
        check_assist(change_visibility, "<|>mod foo {}", "<|>pub(crate) mod foo {}");
        check_assist(change_visibility, "<|>trait Foo {}", "<|>pub(crate) trait Foo {}");
        check_assist(change_visibility, "m<|>od {}", "<|>pub(crate) mod {}");
        check_assist(
            change_visibility,
            "unsafe f<|>n foo() {}",
            "<|>pub(crate) unsafe fn foo() {}",
        );
    }

    #[test]
    fn change_visibility_works_with_struct_fields() {
        check_assist(
            change_visibility,
            r"struct S { <|>field: u32 }",
            r"struct S { <|>pub(crate) field: u32 }",
        );
        check_assist(change_visibility, r"struct S ( <|>u32 )", r"struct S ( <|>pub(crate) u32 )");
    }

    #[test]
    fn change_visibility_field_false_positive() {
        mark::check!(change_visibility_field_false_positive);
        check_assist_not_applicable(
            change_visibility,
            r"struct S { field: [(); { let <|>x = ();}] }",
        )
    }

    #[test]
    fn change_visibility_pub_to_pub_crate() {
        check_assist(change_visibility, "<|>pub fn foo() {}", "<|>pub(crate) fn foo() {}")
    }

    #[test]
    fn change_visibility_pub_crate_to_pub() {
        check_assist(change_visibility, "<|>pub(crate) fn foo() {}", "<|>pub fn foo() {}")
    }

    #[test]
    fn change_visibility_const() {
        check_assist(change_visibility, "<|>const FOO = 3u8;", "<|>pub(crate) const FOO = 3u8;");
    }

    #[test]
    fn change_visibility_handles_comment_attrs() {
        check_assist(
            change_visibility,
            r"
            /// docs

            // comments

            #[derive(Debug)]
            <|>struct Foo;
            ",
            r"
            /// docs

            // comments

            #[derive(Debug)]
            <|>pub(crate) struct Foo;
            ",
        )
    }

    #[test]
    fn change_visibility_of_fn_via_path() {
        check_assist(
            change_visibility,
            r"mod foo { fn foo() {} }
              fn main() { foo::foo<|>() } ",
            r"mod foo { <|>pub(crate) fn foo() {} }
              fn main() { foo::foo() } ",
        );
        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub fn foo() {} }
              fn main() { foo::foo<|>() } ",
        )
    }

    #[test]
    fn change_visibility_of_adt_in_submodule_via_path() {
        check_assist(
            change_visibility,
            r"mod foo { struct Foo; }
              fn main() { foo::Foo<|> } ",
            r"mod foo { <|>pub(crate) struct Foo; }
              fn main() { foo::Foo } ",
        );
        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub struct Foo; }
              fn main() { foo::Foo<|> } ",
        );
        check_assist(
            change_visibility,
            r"mod foo { enum Foo; }
              fn main() { foo::Foo<|> } ",
            r"mod foo { <|>pub(crate) enum Foo; }
              fn main() { foo::Foo } ",
        );
        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub enum Foo; }
              fn main() { foo::Foo<|> } ",
        );
        check_assist(
            change_visibility,
            r"mod foo { union Foo; }
              fn main() { foo::Foo<|> } ",
            r"mod foo { <|>pub(crate) union Foo; }
              fn main() { foo::Foo } ",
        );
        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub union Foo; }
              fn main() { foo::Foo<|> } ",
        );
    }

    #[test]
    fn change_visibility_of_adt_in_other_file_via_path() {
        check_assist(
            change_visibility,
            r"
              //- /main.rs
              mod foo;
              fn main() { foo::Foo<|> }

              //- /foo.rs
              struct Foo;
              ",
            r"<|>pub(crate) struct Foo;

",
        );
    }

    #[test]
    fn change_visibility_of_struct_field_via_path() {
        check_assist(
            change_visibility,
            r"mod foo { pub struct Foo { bar: (), } }
              fn main() { foo::Foo { <|>bar: () }; } ",
            r"mod foo { pub struct Foo { <|>pub(crate) bar: (), } }
              fn main() { foo::Foo { bar: () }; } ",
        );
        check_assist(
            change_visibility,
            r"//- /lib.rs
              mod foo;
              fn main() { foo::Foo { <|>bar: () }; }
              //- /foo.rs
              pub struct Foo { bar: () }
              ",
            r"pub struct Foo { <|>pub(crate) bar: () }

",
        );
        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub struct Foo { pub bar: (), } }
              fn main() { foo::Foo { <|>bar: () }; } ",
        );
        check_assist_not_applicable(
            change_visibility,
            r"//- /lib.rs
              mod foo;
              fn main() { foo::Foo { <|>bar: () }; }
              //- /foo.rs
              pub struct Foo { pub bar: () }
              ",
        );
    }

    #[test]
    fn change_visibility_of_enum_variant_field_via_path() {
        check_assist(
            change_visibility,
            r"mod foo { pub enum Foo { Bar { bar: () } } }
              fn main() { foo::Foo::Bar { <|>bar: () }; } ",
            r"mod foo { pub enum Foo { Bar { <|>pub(crate) bar: () } } }
              fn main() { foo::Foo::Bar { bar: () }; } ",
        );
        check_assist(
            change_visibility,
            r"//- /lib.rs
              mod foo;
              fn main() { foo::Foo::Bar { <|>bar: () }; }
              //- /foo.rs
              pub enum Foo { Bar { bar: () } }
              ",
            r"pub enum Foo { Bar { <|>pub(crate) bar: () } }

",
        );
        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub struct Foo { pub bar: (), } }
              fn main() { foo::Foo { <|>bar: () }; } ",
        );
        check_assist_not_applicable(
            change_visibility,
            r"//- /lib.rs
              mod foo;
              fn main() { foo::Foo { <|>bar: () }; }
              //- /foo.rs
              pub struct Foo { pub bar: () }
              ",
        );
    }

    #[test]
    #[ignore]
    // FIXME reenable this test when `Semantics::resolve_record_field` works with union fields
    fn change_visibility_of_union_field_via_path() {
        check_assist(
            change_visibility,
            r"mod foo { pub union Foo { bar: (), } }
              fn main() { foo::Foo { <|>bar: () }; } ",
            r"mod foo { pub union Foo { <|>pub(crate) bar: (), } }
              fn main() { foo::Foo { bar: () }; } ",
        );
        check_assist(
            change_visibility,
            r"//- /lib.rs
              mod foo;
              fn main() { foo::Foo { <|>bar: () }; }
              //- /foo.rs
              pub union Foo { bar: () }
              ",
            r"pub union Foo { <|>pub(crate) bar: () }

",
        );
        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub union Foo { pub bar: (), } }
              fn main() { foo::Foo { <|>bar: () }; } ",
        );
        check_assist_not_applicable(
            change_visibility,
            r"//- /lib.rs
              mod foo;
              fn main() { foo::Foo { <|>bar: () }; }
              //- /foo.rs
              pub union Foo { pub bar: () }
              ",
        );
    }

    #[test]
    fn not_applicable_for_enum_variants() {
        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub enum Foo {Foo1} }
              fn main() { foo::Foo::Foo1<|> } ",
        );
    }

    #[test]
    fn change_visibility_of_const_via_path() {
        check_assist(
            change_visibility,
            r"mod foo { const FOO: () = (); }
              fn main() { foo::FOO<|> } ",
            r"mod foo { <|>pub(crate) const FOO: () = (); }
              fn main() { foo::FOO } ",
        );
        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub const FOO: () = (); }
              fn main() { foo::FOO<|> } ",
        );
    }

    #[test]
    fn change_visibility_of_static_via_path() {
        check_assist(
            change_visibility,
            r"mod foo { static FOO: () = (); }
              fn main() { foo::FOO<|> } ",
            r"mod foo { <|>pub(crate) static FOO: () = (); }
              fn main() { foo::FOO } ",
        );
        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub static FOO: () = (); }
              fn main() { foo::FOO<|> } ",
        );
    }

    #[test]
    fn change_visibility_of_trait_via_path() {
        check_assist(
            change_visibility,
            r"mod foo { trait Foo { fn foo(&self) {} } }
              fn main() { let x: &dyn foo::<|>Foo; } ",
            r"mod foo { <|>pub(crate) trait Foo { fn foo(&self) {} } }
              fn main() { let x: &dyn foo::Foo; } ",
        );
        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub trait Foo { fn foo(&self) {} } }
              fn main() { let x: &dyn foo::Foo<|>; } ",
        );
    }

    #[test]
    fn change_visibility_of_type_alias_via_path() {
        check_assist(
            change_visibility,
            r"mod foo { type Foo = (); }
              fn main() { let x: foo::Foo<|>; } ",
            r"mod foo { <|>pub(crate) type Foo = (); }
              fn main() { let x: foo::Foo; } ",
        );
        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub type Foo = (); }
              fn main() { let x: foo::Foo<|>; } ",
        );
    }

    #[test]
    fn change_visibility_of_module_via_path() {
        check_assist(
            change_visibility,
            r"mod foo { mod bar { fn bar() {} } }
              fn main() { foo::bar<|>::bar(); } ",
            r"mod foo { <|>pub(crate) mod bar { fn bar() {} } }
              fn main() { foo::bar::bar(); } ",
        );

        check_assist(
            change_visibility,
            r"
            //- /main.rs
            mod foo;
            fn main() { foo::bar<|>::baz(); }

            //- /foo.rs
            mod bar {
                pub fn baz() {}
            }
            ",
            r"<|>pub(crate) mod bar {
    pub fn baz() {}
}

",
        );

        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub mod bar { pub fn bar() {} } }
              fn main() { foo::bar<|>::bar(); } ",
        );
    }

    #[test]
    fn change_visibility_of_inline_module_in_other_file_via_path() {
        check_assist(
            change_visibility,
            r"
            //- /main.rs
            mod foo;
            fn main() { foo::bar<|>::baz(); }

            //- /foo.rs
            mod bar;

            //- /foo/bar.rs
            pub fn baz() {}
            }
            ",
            r"<|>pub(crate) mod bar;
",
        );
    }

    #[test]
    fn change_visibility_of_module_declaration_in_other_file_via_path() {
        check_assist(
            change_visibility,
            r"//- /main.rs
              mod foo;
              fn main() { foo::bar<|>>::baz(); }

              //- /foo.rs
              mod bar {
                  pub fn baz() {}
              }",
            r"<|>pub(crate) mod bar {
    pub fn baz() {}
}
",
        );
    }

    #[test]
    #[ignore]
    // FIXME handle reexports properly
    fn change_visibility_of_reexport() {
        check_assist(
            change_visibility,
            r"
            mod foo {
                use bar::Baz;
                mod bar { pub(super) struct Baz; }
            }
            foo::Baz<|>
            ",
            r"
            mod foo {
                <|>pub(crate) use bar::Baz;
                mod bar { pub(super) struct Baz; }
            }
            foo::Baz
            ",
        )
    }

    #[test]
    fn adds_pub_when_target_is_in_another_crate() {
        check_assist(
            change_visibility,
            r"//- /main.rs crate:a deps:foo
              foo::Bar<|>
              //- /lib.rs crate:foo
              struct Bar;",
            r"<|>pub struct Bar;
",
        )
    }

    #[test]
    fn change_visibility_target() {
        check_assist_target(change_visibility, "<|>fn foo() {}", "fn");
        check_assist_target(change_visibility, "pub(crate)<|> fn foo() {}", "pub(crate)");
        check_assist_target(change_visibility, "struct S { <|>field: u32 }", "field");
    }
}
