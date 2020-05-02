use ra_syntax::{
    ast::{self, NameOwner, VisibilityOwner},
    AstNode,
    SyntaxKind::{
        ATTR, COMMENT, CONST_DEF, ENUM_DEF, FN_DEF, MODULE, STRUCT_DEF, TRAIT_DEF, VISIBILITY,
        WHITESPACE,
    },
    SyntaxNode, TextRange, TextSize, T,
};

use hir::{db::HirDatabase, HasSource, PathResolution};
use test_utils::tested_by;

use crate::{AssistContext, AssistId, Assists};

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
            tested_by!(change_visibility_field_false_positive);
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

fn add_missing_vis(ctx: AssistCtx) -> Option<Assist> {
    let path: ast::Path = ctx.find_node_at_offset()?;
    let path_res = dbg!(ctx.sema.resolve_path(&path))?;
    let def = match path_res {
        PathResolution::Def(def) => def,
        _ => return None,
    };
    dbg!(&def);

    let current_module = dbg!(ctx.sema.scope(&path.syntax()).module())?;
    let target_module = dbg!(def.module(ctx.db))?;

    let vis = dbg!(target_module.visibility_of(ctx.db, &def))?;
    if vis.is_visible_from(ctx.db, current_module.into()) {
        return None;
    };
    let target_name;

    let (offset, target) = match def {
        hir::ModuleDef::Function(f) => {
            target_name = Some(f.name(ctx.db));
            offset_and_target(ctx.db, f)
        }
        hir::ModuleDef::Adt(adt) => {
            target_name = Some(adt.name(ctx.db));
            match adt {
                hir::Adt::Struct(s) => offset_and_target(ctx.db, s),
                hir::Adt::Union(u) => offset_and_target(ctx.db, u),
                hir::Adt::Enum(e) => offset_and_target(ctx.db, e),
            }
        }
        hir::ModuleDef::Const(c) => {
            target_name = c.name(ctx.db);
            offset_and_target(ctx.db, c)
        }
        hir::ModuleDef::Static(s) => {
            target_name = s.name(ctx.db);
            offset_and_target(ctx.db, s)
        }
        hir::ModuleDef::Trait(t) => {
            target_name = Some(t.name(ctx.db));
            offset_and_target(ctx.db, t)
        }
        hir::ModuleDef::TypeAlias(t) => {
            target_name = Some(t.name(ctx.db));
            offset_and_target(ctx.db, t)
        }
        hir::ModuleDef::Module(m) => {
            target_name = m.name(ctx.db);
            let source = dbg!(m.declaration_source(ctx.db))?.value;
            let syntax = source.syntax();
            (vis_offset(syntax), syntax.text_range())
        }
        // Enum variants can't be private, we can't modify builtin types
        hir::ModuleDef::EnumVariant(_) | hir::ModuleDef::BuiltinType(_) => return None,
    };

    // FIXME if target is in another crate, add `pub` instead of `pub(crate)`

    let assist_label = match target_name {
        None => "Change visibility to pub(crate)".to_string(),
        Some(name) => format!("Change visibility of {} to pub(crate)", name),
    };
    let target_file = target_module.definition_source(ctx.db).file_id.original_file(ctx.db);

    ctx.add_assist(AssistId("change_visibility"), assist_label, target, |edit| {
        edit.set_file(target_file);
        edit.insert(offset, "pub(crate) ");
        edit.set_cursor(offset);
    })
}

fn offset_and_target<S, Ast>(db: &dyn HirDatabase, x: S) -> (TextSize, TextRange)
where
    S: HasSource<Ast = Ast>,
    Ast: AstNode,
{
    let source = x.source(db);
    let syntax = source.syntax().value;
    (vis_offset(syntax), syntax.text_range())
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
    use test_utils::covers;

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
        covers!(change_visibility_field_false_positive);
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
    // FIXME this requires a separate implementation, struct fields are not a ast::Path
    fn change_visibility_of_struct_field_via_path() {
        check_assist(
            change_visibility,
            r"mod foo { pub struct Foo { bar: (), } }
              fn main() { foo::Foo { <|>bar: () }; } ",
            r"mod foo { pub struct Foo { <|>pub(crate) bar: (), } }
              fn main() { foo::Foo { bar: () }; } ",
        );
        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub struct Foo { pub bar: (), } }
              fn main() { foo::Foo { <|>bar: () }; } ",
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
            r"
            //- /main.rs
            mod foo;
            fn main() { foo::bar<|>>::baz(); }

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
    }

    #[test]
    fn change_visibility_target() {
        check_assist_target(change_visibility, "<|>fn foo() {}", "fn");
        check_assist_target(change_visibility, "pub(crate)<|> fn foo() {}", "pub(crate)");
        check_assist_target(change_visibility, "struct S { <|>field: u32 }", "field");
    }
}
