use hir::{HasSource, HasVisibility, ModuleDef, PathResolution, ScopeDef, db::HirDatabase};
use ide_db::FileId;
use syntax::{
    AstNode, TextRange,
    ast::{self, HasVisibility as _, edit_in_place::HasVisibilityEdit, make},
};

use crate::{AssistContext, AssistId, Assists};

// Assist: fix_visibility
//
// Note that there is some duplication between this and the no_such_field diagnostic.
//
// Makes inaccessible item public.
//
// ```
// mod m {
//     fn frobnicate() {}
// }
// fn main() {
//     m::frobnicate$0();
// }
// ```
// ->
// ```
// mod m {
//     $0pub(crate) fn frobnicate() {}
// }
// fn main() {
//     m::frobnicate();
// }
// ```
pub(crate) fn fix_visibility(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    add_vis_to_referenced_module_def(acc, ctx)
}

fn add_vis_to_referenced_module_def(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let path: ast::Path = ctx.find_node_at_offset()?;
    let qualifier = path.qualifier()?;
    let name_ref = path.segment()?.name_ref()?;
    let qualifier_res = ctx.sema.resolve_path(&qualifier)?;
    let PathResolution::Def(ModuleDef::Module(module)) = qualifier_res else {
        return None;
    };
    let (_, def) = module
        .scope(ctx.db(), None)
        .into_iter()
        .find(|(name, _)| name.as_str() == name_ref.text().trim_start_matches("r#"))?;
    let ScopeDef::ModuleDef(def) = def else {
        return None;
    };

    let current_module = ctx.sema.scope(path.syntax())?.module();
    let target_module = def.module(ctx.db())?;

    if def.visibility(ctx.db()).is_visible_from(ctx.db(), current_module.into()) {
        return None;
    };

    let (vis_owner, target, target_file, target_name) = target_data_for_def(ctx.db(), def)?;

    let missing_visibility = if current_module.krate() == target_module.krate() {
        make::visibility_pub_crate()
    } else {
        make::visibility_pub()
    };

    let assist_label = match target_name {
        None => format!("Change visibility to {missing_visibility}"),
        Some(name) => {
            format!(
                "Change visibility of {} to {missing_visibility}",
                name.display(ctx.db(), current_module.krate().edition(ctx.db()))
            )
        }
    };

    acc.add(AssistId::quick_fix("fix_visibility"), assist_label, target, |edit| {
        edit.edit_file(target_file);

        let vis_owner = edit.make_mut(vis_owner);
        vis_owner.set_visibility(Some(missing_visibility.clone_for_update()));

        if let Some((cap, vis)) = ctx.config.snippet_cap.zip(vis_owner.visibility()) {
            edit.add_tabstop_before(cap, vis);
        }
    })
}

fn target_data_for_def(
    db: &dyn HirDatabase,
    def: hir::ModuleDef,
) -> Option<(ast::AnyHasVisibility, TextRange, FileId, Option<hir::Name>)> {
    fn offset_target_and_file_id<S, Ast>(
        db: &dyn HirDatabase,
        x: S,
    ) -> Option<(ast::AnyHasVisibility, TextRange, FileId)>
    where
        S: HasSource<Ast = Ast>,
        Ast: AstNode + ast::HasVisibility,
    {
        let source = x.source(db)?;
        let in_file_syntax = source.syntax();
        let file_id = in_file_syntax.file_id;
        let range = in_file_syntax.value.text_range();
        Some((
            ast::AnyHasVisibility::new(source.value),
            range,
            file_id.original_file(db).file_id(db),
        ))
    }

    let target_name;
    let (offset, target, target_file) = match def {
        hir::ModuleDef::Function(f) => {
            target_name = Some(f.name(db));
            offset_target_and_file_id(db, f)?
        }
        hir::ModuleDef::Adt(adt) => {
            target_name = Some(adt.name(db));
            match adt {
                hir::Adt::Struct(s) => offset_target_and_file_id(db, s)?,
                hir::Adt::Union(u) => offset_target_and_file_id(db, u)?,
                hir::Adt::Enum(e) => offset_target_and_file_id(db, e)?,
            }
        }
        hir::ModuleDef::Const(c) => {
            target_name = c.name(db);
            offset_target_and_file_id(db, c)?
        }
        hir::ModuleDef::Static(s) => {
            target_name = Some(s.name(db));
            offset_target_and_file_id(db, s)?
        }
        hir::ModuleDef::Trait(t) => {
            target_name = Some(t.name(db));
            offset_target_and_file_id(db, t)?
        }
        hir::ModuleDef::TraitAlias(t) => {
            target_name = Some(t.name(db));
            offset_target_and_file_id(db, t)?
        }
        hir::ModuleDef::TypeAlias(t) => {
            target_name = Some(t.name(db));
            offset_target_and_file_id(db, t)?
        }
        hir::ModuleDef::Module(m) => {
            target_name = m.name(db);
            let in_file_source = m.declaration_source(db)?;
            let file_id = in_file_source.file_id.original_file(db);
            let range = in_file_source.value.syntax().text_range();
            (ast::AnyHasVisibility::new(in_file_source.value), range, file_id.file_id(db))
        }
        // FIXME
        hir::ModuleDef::Macro(_) => return None,
        // Enum variants can't be private, we can't modify builtin types
        hir::ModuleDef::Variant(_) | hir::ModuleDef::BuiltinType(_) => return None,
    };

    Some((offset, target, target_file, target_name))
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn fix_visibility_of_fn() {
        check_assist(
            fix_visibility,
            r"mod foo { fn foo() {} }
              fn main() { foo::foo$0() } ",
            r"mod foo { $0pub(crate) fn foo() {} }
              fn main() { foo::foo() } ",
        );
        check_assist_not_applicable(
            fix_visibility,
            r"mod foo { pub fn foo() {} }
              fn main() { foo::foo$0() } ",
        )
    }

    #[test]
    fn fix_visibility_of_adt_in_submodule() {
        check_assist(
            fix_visibility,
            r"mod foo { struct Foo; }
              fn main() { foo::Foo$0 } ",
            r"mod foo { $0pub(crate) struct Foo; }
              fn main() { foo::Foo } ",
        );
        check_assist_not_applicable(
            fix_visibility,
            r"mod foo { pub struct Foo; }
              fn main() { foo::Foo$0 } ",
        );
        check_assist(
            fix_visibility,
            r"mod foo { enum Foo; }
              fn main() { foo::Foo$0 } ",
            r"mod foo { $0pub(crate) enum Foo; }
              fn main() { foo::Foo } ",
        );
        check_assist_not_applicable(
            fix_visibility,
            r"mod foo { pub enum Foo; }
              fn main() { foo::Foo$0 } ",
        );
        check_assist(
            fix_visibility,
            r"mod foo { union Foo; }
              fn main() { foo::Foo$0 } ",
            r"mod foo { $0pub(crate) union Foo; }
              fn main() { foo::Foo } ",
        );
        check_assist_not_applicable(
            fix_visibility,
            r"mod foo { pub union Foo; }
              fn main() { foo::Foo$0 } ",
        );
    }

    #[test]
    fn fix_visibility_of_adt_in_other_file() {
        check_assist(
            fix_visibility,
            r"
//- /main.rs
mod foo;
fn main() { foo::Foo$0 }

//- /foo.rs
struct Foo;
",
            r"$0pub(crate) struct Foo;
",
        );
    }

    #[test]
    fn fix_visibility_of_enum_variant_field() {
        // Enum variants, as well as their fields, always get the enum's visibility. In fact, rustc
        // rejects any visibility specifiers on them, so this assist should never fire on them.
        check_assist_not_applicable(
            fix_visibility,
            r"mod foo { pub enum Foo { Bar { bar: () } } }
              fn main() { foo::Foo::Bar { $0bar: () }; } ",
        );
        check_assist_not_applicable(
            fix_visibility,
            r"
//- /lib.rs
mod foo;
fn main() { foo::Foo::Bar { $0bar: () }; }
//- /foo.rs
pub enum Foo { Bar { bar: () } }
",
        );
        check_assist_not_applicable(
            fix_visibility,
            r"mod foo { pub struct Foo { pub bar: (), } }
              fn main() { foo::Foo { $0bar: () }; } ",
        );
        check_assist_not_applicable(
            fix_visibility,
            r"
//- /lib.rs
mod foo;
fn main() { foo::Foo { $0bar: () }; }
//- /foo.rs
pub struct Foo { pub bar: () }
",
        );
    }

    #[test]
    fn fix_visibility_of_const() {
        check_assist(
            fix_visibility,
            r"mod foo { const FOO: () = (); }
              fn main() { foo::FOO$0 } ",
            r"mod foo { $0pub(crate) const FOO: () = (); }
              fn main() { foo::FOO } ",
        );
        check_assist_not_applicable(
            fix_visibility,
            r"mod foo { pub const FOO: () = (); }
              fn main() { foo::FOO$0 } ",
        );
    }

    #[test]
    fn fix_visibility_of_static() {
        check_assist(
            fix_visibility,
            r"mod foo { static FOO: () = (); }
              fn main() { foo::FOO$0 } ",
            r"mod foo { $0pub(crate) static FOO: () = (); }
              fn main() { foo::FOO } ",
        );
        check_assist_not_applicable(
            fix_visibility,
            r"mod foo { pub static FOO: () = (); }
              fn main() { foo::FOO$0 } ",
        );
    }

    #[test]
    fn fix_visibility_of_trait() {
        check_assist(
            fix_visibility,
            r"mod foo { trait Foo { fn foo(&self) {} } }
              fn main() { let x: &dyn foo::$0Foo; } ",
            r"mod foo { $0pub(crate) trait Foo { fn foo(&self) {} } }
              fn main() { let x: &dyn foo::Foo; } ",
        );
        check_assist_not_applicable(
            fix_visibility,
            r"mod foo { pub trait Foo { fn foo(&self) {} } }
              fn main() { let x: &dyn foo::Foo$0; } ",
        );
    }

    #[test]
    fn fix_visibility_of_type_alias() {
        check_assist(
            fix_visibility,
            r"mod foo { type Foo = (); }
              fn main() { let x: foo::Foo$0; } ",
            r"mod foo { $0pub(crate) type Foo = (); }
              fn main() { let x: foo::Foo; } ",
        );
        check_assist_not_applicable(
            fix_visibility,
            r"mod foo { pub type Foo = (); }
              fn main() { let x: foo::Foo$0; } ",
        );
    }

    #[test]
    fn fix_visibility_of_module() {
        check_assist(
            fix_visibility,
            r"mod foo { mod bar { fn bar() {} } }
              fn main() { foo::bar$0::bar(); } ",
            r"mod foo { $0pub(crate) mod bar { fn bar() {} } }
              fn main() { foo::bar::bar(); } ",
        );

        check_assist(
            fix_visibility,
            r"
//- /main.rs
mod foo;
fn main() { foo::bar$0::baz(); }

//- /foo.rs
mod bar {
    pub fn baz() {}
}
",
            r"$0pub(crate) mod bar {
    pub fn baz() {}
}
",
        );

        check_assist_not_applicable(
            fix_visibility,
            r"mod foo { pub mod bar { pub fn bar() {} } }
              fn main() { foo::bar$0::bar(); } ",
        );
    }

    #[test]
    fn fix_visibility_of_inline_module_in_other_file() {
        check_assist(
            fix_visibility,
            r"
//- /main.rs
mod foo;
fn main() { foo::bar$0::baz(); }

//- /foo.rs
mod bar;
//- /foo/bar.rs
pub fn baz() {}
",
            r"$0pub(crate) mod bar;
",
        );
    }

    #[test]
    fn fix_visibility_of_module_declaration_in_other_file() {
        check_assist(
            fix_visibility,
            r"
//- /main.rs
mod foo;
fn main() { foo::bar$0>::baz(); }

//- /foo.rs
mod bar {
    pub fn baz() {}
}
",
            r"$0pub(crate) mod bar {
    pub fn baz() {}
}
",
        );
    }

    #[test]
    fn adds_pub_when_target_is_in_another_crate() {
        check_assist(
            fix_visibility,
            r"
//- /main.rs crate:a deps:foo
foo::Bar$0
//- /lib.rs crate:foo
struct Bar;
",
            r"$0pub struct Bar;
",
        )
    }

    #[test]
    fn replaces_pub_crate_with_pub() {
        check_assist(
            fix_visibility,
            r"
//- /main.rs crate:a deps:foo
foo::Bar$0
//- /lib.rs crate:foo
pub(crate) struct Bar;
",
            r"$0pub struct Bar;
",
        );
    }

    #[test]
    fn fix_visibility_of_reexport() {
        // FIXME: broken test, this should fix visibility of the re-export
        // rather than the struct.
        check_assist(
            fix_visibility,
            r#"
mod foo {
    use bar::Baz;
    mod bar { pub(super) struct Baz; }
}
foo::Baz$0
"#,
            r#"
mod foo {
    use bar::Baz;
    mod bar { $0pub(crate) struct Baz; }
}
foo::Baz
"#,
        )
    }
}
