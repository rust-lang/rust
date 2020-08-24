use either::Either;
use hir::{AssocItem, MacroDef, ModuleDef, Name, PathResolution, ScopeDef, SemanticsScope};
use ide_db::{
    defs::{classify_name_ref, Definition, NameRefClass},
    RootDatabase,
};
use syntax::{algo, ast, match_ast, AstNode, SyntaxNode, SyntaxToken, T};

use crate::{
    assist_context::{AssistBuilder, AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: expand_glob_import
//
// Expands glob imports.
//
// ```
// mod foo {
//     pub struct Bar;
//     pub struct Baz;
// }
//
// use foo::*<|>;
//
// fn qux(bar: Bar, baz: Baz) {}
// ```
// ->
// ```
// mod foo {
//     pub struct Bar;
//     pub struct Baz;
// }
//
// use foo::{Baz, Bar};
//
// fn qux(bar: Bar, baz: Baz) {}
// ```
pub(crate) fn expand_glob_import(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let star = ctx.find_token_at_offset(T![*])?;
    let mod_path = find_mod_path(&star)?;
    let module = match ctx.sema.resolve_path(&mod_path)? {
        PathResolution::Def(ModuleDef::Module(it)) => it,
        _ => return None,
    };

    let source_file = ctx.source_file();
    let scope = ctx.sema.scope_at_offset(source_file.syntax(), ctx.offset());

    let defs_in_mod = find_defs_in_mod(ctx, scope, module)?;
    let name_refs_in_source_file =
        source_file.syntax().descendants().filter_map(ast::NameRef::cast).collect();
    let used_names = find_used_names(ctx, defs_in_mod, name_refs_in_source_file);

    let parent = star.parent().parent()?;
    acc.add(
        AssistId("expand_glob_import", AssistKind::RefactorRewrite),
        "Expand glob import",
        parent.text_range(),
        |builder| {
            replace_ast(builder, &parent, mod_path, used_names);
        },
    )
}

fn find_mod_path(star: &SyntaxToken) -> Option<ast::Path> {
    star.ancestors().find_map(|n| ast::UseTree::cast(n).and_then(|u| u.path()))
}

#[derive(PartialEq)]
enum Def {
    ModuleDef(ModuleDef),
    MacroDef(MacroDef),
}

impl Def {
    fn name(&self, db: &RootDatabase) -> Option<Name> {
        match self {
            Def::ModuleDef(def) => def.name(db),
            Def::MacroDef(def) => def.name(db),
        }
    }
}

fn find_defs_in_mod(
    ctx: &AssistContext,
    from: SemanticsScope<'_>,
    module: hir::Module,
) -> Option<Vec<Def>> {
    let module_scope = module.scope(ctx.db(), from.module());

    let mut defs = vec![];
    for (_, def) in module_scope {
        match def {
            ScopeDef::ModuleDef(def) => defs.push(Def::ModuleDef(def)),
            ScopeDef::MacroDef(def) => defs.push(Def::MacroDef(def)),
            _ => continue,
        }
    }

    Some(defs)
}

fn find_used_names(
    ctx: &AssistContext,
    defs_in_mod: Vec<Def>,
    name_refs_in_source_file: Vec<ast::NameRef>,
) -> Vec<Name> {
    let defs_in_source_file = name_refs_in_source_file
        .iter()
        .filter_map(|r| classify_name_ref(&ctx.sema, r))
        .filter_map(|rc| match rc {
            NameRefClass::Definition(Definition::ModuleDef(def)) => Some(Def::ModuleDef(def)),
            NameRefClass::Definition(Definition::Macro(def)) => Some(Def::MacroDef(def)),
            _ => None,
        })
        .collect::<Vec<Def>>();

    defs_in_mod
        .iter()
        .filter(|def| {
            if let Def::ModuleDef(ModuleDef::Trait(tr)) = def {
                for item in tr.items(ctx.db()) {
                    if let AssocItem::Function(f) = item {
                        if defs_in_source_file.contains(&Def::ModuleDef(ModuleDef::Function(f))) {
                            return true;
                        }
                    }
                }
            }

            defs_in_source_file.contains(def)
        })
        .filter_map(|d| d.name(ctx.db()))
        .collect()
}

fn replace_ast(
    builder: &mut AssistBuilder,
    node: &SyntaxNode,
    path: ast::Path,
    used_names: Vec<Name>,
) {
    let replacement: Either<ast::UseTree, ast::UseTreeList> = match used_names.as_slice() {
        [name] => Either::Left(ast::make::use_tree(
            ast::make::path_from_text(&format!("{}::{}", path, name)),
            None,
            None,
            false,
        )),
        names => Either::Right(ast::make::use_tree_list(names.iter().map(|n| {
            ast::make::use_tree(ast::make::path_from_text(&n.to_string()), None, None, false)
        }))),
    };

    let mut replace_node = |replacement: Either<ast::UseTree, ast::UseTreeList>| {
        algo::diff(node, &replacement.either(|u| u.syntax().clone(), |ut| ut.syntax().clone()))
            .into_text_edit(builder.text_edit_builder());
    };

    match_ast! {
        match node {
            ast::UseTree(use_tree) => {
                replace_node(replacement);
            },
            ast::UseTreeList(use_tree_list) => {
                replace_node(replacement);
            },
            ast::Use(use_item) => {
                builder.replace_ast(use_item, ast::make::use_(replacement.left_or_else(|ut| ast::make::use_tree(path, Some(ut), None, false))));
            },
            _ => {},
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn expanding_glob_import() {
        check_assist(
            expand_glob_import,
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
    pub struct Qux;

    pub fn f() {}
}

use foo::*<|>;

fn qux(bar: Bar, baz: Baz) {
    f();
}
",
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
    pub struct Qux;

    pub fn f() {}
}

use foo::{Baz, Bar, f};

fn qux(bar: Bar, baz: Baz) {
    f();
}
",
        )
    }

    #[test]
    fn expanding_glob_import_with_existing_explicit_names() {
        check_assist(
            expand_glob_import,
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
    pub struct Qux;

    pub fn f() {}
}

use foo::{*<|>, f};

fn qux(bar: Bar, baz: Baz) {
    f();
}
",
            r"
mod foo {
    pub struct Bar;
    pub struct Baz;
    pub struct Qux;

    pub fn f() {}
}

use foo::{Baz, Bar, f};

fn qux(bar: Bar, baz: Baz) {
    f();
}
",
        )
    }

    #[test]
    fn expanding_nested_glob_import() {
        check_assist(
            expand_glob_import,
            r"
mod foo {
    mod bar {
        pub struct Bar;
        pub struct Baz;
        pub struct Qux;

        pub fn f() {}
    }

    mod baz {
        pub fn g() {}
    }
}

use foo::{bar::{*<|>, f}, baz::*};

fn qux(bar: Bar, baz: Baz) {
    f();
    g();
}
",
            r"
mod foo {
    mod bar {
        pub struct Bar;
        pub struct Baz;
        pub struct Qux;

        pub fn f() {}
    }

    mod baz {
        pub fn g() {}
    }
}

use foo::{bar::{Baz, Bar, f}, baz::*};

fn qux(bar: Bar, baz: Baz) {
    f();
    g();
}
",
        )
    }

    #[test]
    fn expanding_glob_import_with_macro_defs() {
        check_assist(
            expand_glob_import,
            r"
//- /lib.rs crate:foo
#[macro_export]
macro_rules! bar {
    () => ()
}

pub fn baz() {}

//- /main.rs crate:main deps:foo
use foo::*<|>;

fn main() {
    bar!();
    baz();
}
",
            r"
use foo::{bar, baz};

fn main() {
    bar!();
    baz();
}
",
        )
    }

    #[test]
    fn expanding_glob_import_with_trait_method_uses() {
        check_assist(
            expand_glob_import,
            r"
//- /lib.rs crate:foo
pub trait Tr {
    fn method(&self) {}
}
impl Tr for () {}

//- /main.rs crate:main deps:foo
use foo::*<|>;

fn main() {
    ().method();
}
",
            r"
use foo::Tr;

fn main() {
    ().method();
}
",
        )
    }

    #[test]
    fn expanding_is_not_applicable_if_cursor_is_not_in_star_token() {
        check_assist_not_applicable(
            expand_glob_import,
            r"
    mod foo {
        pub struct Bar;
        pub struct Baz;
        pub struct Qux;
    }

    use foo::Bar<|>;

    fn qux(bar: Bar, baz: Baz) {}
    ",
        )
    }
}
